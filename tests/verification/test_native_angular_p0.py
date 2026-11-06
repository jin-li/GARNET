"""Tiny algebraic P0 Galerkin checks only; no actual pin solve."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.linalg import eig
from scipy.special import eval_legendre

import garnet.native_angular_p0 as module
from garnet.mesh import assemble_mesh
from garnet.moment_materials import MomentMaterial
from garnet.native_angular_source import NativeAngularSource, ProductQuadrature
from garnet.native_mesh_streaming import NativeMeshStreaming
from garnet.native_mesh_transport import NativeMeshTransport
from garnet.native_angular_p0 import P0GalerkinTransport


def material(total, sp0, *, orientation='to-major', lmax=1):
    g = len(total)
    matrices = {0: np.asarray(sp0, dtype=float)}
    for order in range(1, lmax+1):
        matrices[order] = np.array([[.015, -.004], [.003, .02]])/order
    if orientation == 'from-major':
        matrices = {order: matrix.T for order, matrix in matrices.items()}
    return MomentMaterial(np.asarray(total, dtype=float), np.array([.7, .3]), np.array([.12, .3]),
                          np.array([2., 5.]), {}, matrices, {}, {}, {},
                          {'orientation_contract': orientation, 'energy_bounds_mev': [20., 1., 0.]}, {})


def bound(orientation='to-major', lmax=1, shear=0., bc_mode='mixed'):
    points = np.array([[0., 0.], [1., 0.], [3., 0.], [0., 1.], [1., 1.], [3., 1.]])
    points[:, 0] += shear*points[:, 1]
    mesh = assemble_mesh(points, [(0, 1, 4, 3), (1, 2, 5, 4)], [0, 1])
    materials = (material([1.3, 1.7], [[.25, .08], [.12, .35]], orientation=orientation, lmax=lmax),
                 material([1.1, 1.5], [[.22, .07], [.1, .31]], orientation=orientation, lmax=lmax))
    source = NativeAngularSource(ProductQuadrature(lmax), materials, np.array([0, 1]))
    stream = NativeMeshStreaming(mesh, source)
    bc = tuple('reflective' if bc_mode == 'reflective' or (bc_mode == 'mixed' and abs(normal[0]) < 1e-13)
               else 'vacuum' for normal in stream.boundary_normals)
    return stream, bc


def dense_oracle(stream, bc):
    """Independent angular Legendre kernel/face equations, plus explicit tiny R/P."""
    nc, na, ng = stream.shape
    size, coarse = nc*na*ng, nc*ng
    a, f, r, p = np.zeros((size, size)), np.zeros((size, size)), np.zeros((coarse, size)), np.zeros((size, coarse))
    functional = np.zeros(size)
    directions, weights = stream.quadrature.directions, stream.quadrature.weights

    def index(c, n, g):
        return (c*na+n)*ng+g

    for cell, mid in enumerate(stream.source.cell_material_ids):
        m = stream.source.materials[mid]
        for angle in range(na):
            for group in range(ng):
                row = index(cell, angle, group)
                a[row, row] += stream.area[cell]*m.total[group]
                r[cell*ng+group, row] = weights[angle]
                p[row, cell*ng+group] = 1/(4*np.pi)
                functional[row] = stream.area[cell]*m.nu_sigma_f[group]*weights[angle]
                for incoming in range(na):
                    dot = sum(directions[angle, axis]*directions[incoming, axis] for axis in range(3))
                    for from_group in range(ng):
                        column = index(cell, incoming, from_group)
                        kernel = 0.
                        for order, matrix in m.production.items():
                            value = matrix[from_group, group] if m.provenance['orientation_contract'] == 'from-major' else matrix[group, from_group]
                            kernel += (2*order+1)*eval_legendre(order, dot)*value
                        a[row, column] -= stream.area[cell]*weights[incoming]*kernel/(4*np.pi)
                        f[row, column] += stream.area[cell]*m.chi[group]*m.nu_sigma_f[from_group]*weights[incoming]/(4*np.pi)
    for (left, right), area, normal in zip(stream.owners, stream.measures, stream.normals):
        for angle in range(na):
            transfer = area*sum(directions[angle, axis]*normal[axis] for axis in (0, 1))
            upstream = left if transfer >= 0 else right
            for group in range(ng):
                a[index(left, angle, group), index(upstream, angle, group)] += transfer
                a[index(right, angle, group), index(upstream, angle, group)] -= transfer
    for spec, owner, area, normal in zip(bc, stream.boundary_owners, stream.boundary_measures, stream.boundary_normals):
        for angle in range(na):
            transfer = area*sum(directions[angle, axis]*normal[axis] for axis in (0, 1))
            for group in range(ng):
                if transfer > 0:
                    a[index(owner, angle, group), index(owner, angle, group)] += transfer
                elif transfer < 0 and spec == 'reflective':
                    reflected = directions[angle].copy()
                    reflected[:2] -= 2*np.dot(reflected[:2], normal)*normal
                    target = int(np.argmin(np.linalg.norm(directions-reflected, axis=1)))
                    assert np.linalg.norm(directions[target]-reflected) < 1e-12
                    a[index(owner, angle, group), index(owner, target, group)] += transfer
    return a, f, r, p, functional


@pytest.mark.parametrize('shear,bc_mode', [(0., 'reflective'), (0., 'mixed'), (.4, 'mixed'), (.4, 'vacuum')])
def test_independent_dense_rap_and_actual_matrixfree_rap(shear, bc_mode):
    stream, bc = bound(lmax=3, shear=shear, bc_mode=bc_mode)
    op = P0GalerkinTransport(stream, bc)
    a, _, r, p, _ = dense_oracle(stream, bc)
    np.testing.assert_allclose(r @ p, np.eye(4), atol=2e-15, rtol=0)
    np.testing.assert_allclose(op.coarse_matrix.toarray(), r @ a @ p, atol=5e-15, rtol=0)
    scalar = np.random.default_rng(83).normal(size=(2, 2))
    np.testing.assert_allclose(op.restrict(op.prolong(scalar)), scalar, atol=2e-15, rtol=0)
    actual_rap = op.restrict(op.action(op.prolong(scalar)))
    np.testing.assert_allclose(actual_rap.ravel(), op.coarse_matrix @ scalar.ravel(), atol=3e-15, rtol=0)


def test_actual_discrete_half_ranges_not_continuum_quarter():
    stream, bc = bound(bc_mode='reflective')
    op = P0GalerkinTransport(stream, bc)
    area, normal = stream.measures[0], stream.normals[0]
    speed = stream.quadrature.directions[:, :2] @ normal
    coefficient = area*np.dot(stream.quadrature.weights, np.maximum(speed, 0))/(4*np.pi)
    assert not np.isclose(coefficient, area/4)
    assert op.coarse_matrix[0, 2] == pytest.approx(-coefficient, abs=2e-16)
    np.testing.assert_allclose(op.coarse_matrix.toarray()[0:2, 0:2],
                               stream.area[0]*(np.diag(stream.total[0])-stream.source.production[0][0])+np.eye(2)*coefficient)
    oblique, _ = bound(shear=.4)
    q, normal = oblique.quadrature, oblique.normals[0]
    positive = np.dot(q.weights, np.maximum(q.directions[:, :2] @ normal, 0))
    negative = np.dot(q.weights, np.maximum(-q.directions[:, :2] @ normal, 0))
    assert positive == pytest.approx(negative, abs=2e-15)


def test_source_orientation_and_signed_higher_orders_unchanged():
    stream, bc = bound(orientation='from-major', lmax=7)
    original = [matrix.tobytes() for matrix in stream.source.production]
    op = P0GalerkinTransport(stream, bc)
    other, other_bc = bound(orientation='to-major', lmax=7)
    equivalent = P0GalerkinTransport(other, other_bc)
    np.testing.assert_array_equal(op.coarse_matrix.toarray(), equivalent.coarse_matrix.toarray())
    assert [matrix.tobytes() for matrix in stream.source.production] == original
    assert np.any(stream.source.production[0][1:] < 0)
    field = np.arange(np.prod(stream.shape), dtype=float).reshape(stream.shape)/29 - 2
    np.testing.assert_array_equal(op.action(field), equivalent.action(field))


def test_fixed_factor_once_linearity_repeatability_and_operator_byte_identity():
    stream, bc = bound()
    original_factor = module.spilu
    with patch.object(module, 'spilu', wraps=original_factor) as factor:
        op = P0GalerkinTransport(stream, bc)
        default = NativeMeshTransport(stream, bc)
        rng = np.random.default_rng(91)
        x, y = rng.normal(size=stream.shape), rng.normal(size=stream.shape)
        first = op.precondition(x)
        np.testing.assert_array_equal(first, op.precondition(x))
        np.testing.assert_allclose(op.precondition(2*x-.3*y), 2*first-.3*op.precondition(y), atol=3e-15, rtol=2e-14)
        assert factor.call_count == 1
    assert op.action(x).tobytes() == default.action(x).tobytes()
    assert op.fission(x).tobytes() == default.fission(x).tobytes()
    assert default.precondition(x).tobytes() == (x.ravel()/default.diagonal.ravel()).reshape(x.shape).tobytes()
    assert default.preconditioner_provenance() == {'preconditioner': 'diagonal_streaming_collision'}
    config = op.preconditioner_provenance()['p0_galerkin']
    assert config['factorization_count'] == 1 and config['fill_factor_is_not_memory_bound']
    assert config['factor_lower_nnz'] == op._factor.L.nnz and config['factor_upper_nnz'] == op._factor.U.nnz
    actual_bytes = sum(value.nbytes for matrix in (op._factor.L, op._factor.U)
                       for value in (matrix.data, matrix.indices, matrix.indptr))
    actual_bytes += op._factor.perm_r.nbytes+op._factor.perm_c.nbytes
    assert config['retained_factor_array_bytes'] == actual_bytes
    assert len(config['implementation_sha256']) == 5
    assert not config['adaptive'] and not config['fallback'] and not config['regularization']
    with pytest.raises(TypeError):
        config['ilu_fill_factor'] = 1
    exposed = op.coarse_matrix
    with pytest.raises(ValueError):
        exposed.data.setflags(write=True)
    exposed.data = np.ones_like(exposed.data)
    assert not np.array_equal(exposed.toarray(), op.coarse_matrix.toarray())


@pytest.mark.parametrize('drop,fill', [(1e-6, 20.), (.8, 1.)])
def test_heterogeneous_dense_physical_root_invariance_even_weak_ilu(drop, fill):
    stream, bc = bound()
    settings = dict(inner_tolerance=1e-11, k_tolerance=1e-10, source_tolerance=1e-9,
                    full_tolerance=1e-9, max_outer=200, max_inner=300, restart=32)
    op = P0GalerkinTransport(stream, bc, ilu_drop_tolerance=drop, ilu_fill_factor=fill, **settings)
    a, f, _, _, functional = dense_oracle(stream, bc)
    values, vectors = eig(f, a)
    valid = np.flatnonzero(np.isfinite(values) & (np.abs(values.imag) < 1e-10) & (values.real > 0))
    chosen = valid[np.argmax(values[valid].real)]
    k = values[chosen].real
    other = np.isfinite(values) & (np.arange(values.size) != chosen)
    assert np.all(np.abs(values[other]) < k*(1-1e-6))
    angular = vectors[:, chosen].real
    angular /= functional @ angular
    assert angular.min() >= -64*np.finfo(float).eps*np.max(np.abs(angular))
    result = op.solve()
    diagonal = NativeMeshTransport(stream, bc, **settings).solve()
    assert result.converged, result.reason
    assert diagonal.converged, diagonal.reason
    assert result.k_eff == pytest.approx(k, rel=2e-9)
    assert result.k_eff == pytest.approx(diagonal.k_eff, rel=2e-9)
    np.testing.assert_allclose(result.angular_flux.ravel(), angular, rtol=2e-8, atol=2e-9)
    rhs = f @ result.angular_flux.ravel()/result.k_eff
    assert np.linalg.norm(a @ result.angular_flux.ravel()-rhs)/np.linalg.norm(rhs) <= 1e-9
    assert result.provenance['preconditioner'] == 'fixed_p0_galerkin_ilu_then_streaming_diagonal'
    assert result.provenance['p0_galerkin']['ilu_drop_tolerance'] == drop
    assert result.history[-1]['true_inner_residual'] <= 1e-11


def test_budget_failures_before_factor_and_after_retained_factor_measurement():
    stream, bc = bound()
    base = NativeMeshTransport(stream, bc)
    with patch.object(module, 'spilu') as factor:
        with pytest.raises(ValueError, match='allocation estimate'):
            P0GalerkinTransport(stream, bc, max_bytes=base.memory_estimate_bytes+1)
        factor.assert_not_called()
        with pytest.raises(ValueError, match='allocation estimate'):
            P0GalerkinTransport(stream, bc, ilu_fill_factor=1e308)
        factor.assert_not_called()
    with patch.object(module, '_sparse_bytes', return_value=2**40):
        with pytest.raises(ValueError, match='allocation estimate'):
            P0GalerkinTransport(stream, bc)


@pytest.mark.parametrize('storage', ['superlu', 'packed'])
def test_post_setup_reservation_preserves_constructor_peak_and_nonfactor_allowances(storage):
    if storage == 'packed':
        _require_packed_backend()
    from dataclasses import FrozenInstanceError

    stream, bc = bound()
    op = P0GalerkinTransport(stream, bc, factor_storage=storage)
    config = op._p0_provenance
    assert config['memory_accounting_version'] == 'p0_phase_liveness_v1'
    assert op.memory_estimate_bytes == config['constructor_peak_estimate_bytes']
    assert config['post_setup_reservation_bytes'] == op._post_setup_reservation_bytes
    assert config['post_setup_nonfactor_allowance_bytes'] == config['extra_nonfactor_estimate_bytes']
    assert config['post_setup_base_transport_estimate_bytes'] == config['base_memory_estimate_bytes']
    assert config['post_setup_coarse_csr_array_bytes'] == module._sparse_bytes(op._coarse)
    expected_peak = (config['base_memory_estimate_bytes']+config['extra_nonfactor_estimate_bytes']
                     +max(config['factor_preflight_estimate_bytes'],
                          config.get('packed_payload_array_bytes', config['retained_factor_array_bytes'])))
    if storage == 'packed':
        from garnet.packed_superlu import PackedSuperLU

        assert isinstance(op._factor, PackedSuperLU)
        assert len(op._factor.owned_arrays) == 9
        assert all(array.base is None and array.flags.owndata
                   for array in op._factor.owned_arrays)
        assert not config['original_superlu_retained']
        assert config['post_setup_factor_array_bytes'] == op._factor.payload_nbytes
        assert op._post_setup_reservation_bytes == (
            config['base_memory_estimate_bytes']+config['extra_nonfactor_estimate_bytes']
            +op._factor.payload_nbytes+module._sparse_bytes(op._coarse))
        # Packing retains the original measured-factor allowance in its peak.
        assert op.memory_estimate_bytes == config['packing_peak_estimate_bytes']
        assert op.memory_estimate_bytes >= expected_peak
    else:
        assert op.memory_estimate_bytes == expected_peak
        assert op._post_setup_reservation_bytes == op.memory_estimate_bytes
    with pytest.raises(FrozenInstanceError):
        op._post_setup_reservation_bytes = 0


def test_packed_P0_preflight_still_rejects_before_factorization():
    _require_packed_backend()
    stream, bc = bound()
    base = NativeMeshTransport(stream, bc)
    with patch.object(module, 'spilu') as factor:
        with pytest.raises(ValueError, match='allocation estimate'):
            P0GalerkinTransport(stream, bc, factor_storage='packed',
                                max_bytes=base.memory_estimate_bytes+1)
        factor.assert_not_called()


def test_factor_failure_no_retry_and_nonfinite_application_fails_closed():
    stream, bc = bound()
    with patch.object(module, 'spilu', side_effect=RuntimeError('synthetic singular')) as factor:
        with pytest.raises(ValueError, match='without fallback'):
            P0GalerkinTransport(stream, bc)
        assert factor.call_count == 1
    op = P0GalerkinTransport(stream, bc)
    object.__setattr__(op, '_factor', SimpleNamespace(solve=lambda rhs: np.full_like(rhs, np.nan)))
    with pytest.raises(ValueError, match='nonfinite'):
        op.precondition(np.ones(stream.shape))
    result = op.fixed_source(np.ones(stream.shape))
    assert not result.converged and result.reason.startswith('inner_arithmetic_failure:')


def test_singular_coarse_zero_diagonal_and_corrupted_half_ranges_fail():
    stream, bc = bound(bc_mode='reflective')
    singular = material([1., 2.], [[0., 1.], [2., 0.]])
    source = NativeAngularSource(stream.quadrature, (singular,), np.array([0]))
    # One reflective unit cell leaves the exactly representable singular block
    # [[1,-1],[-2,2]], without face-addition roundoff perturbing its determinant.
    points = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    mesh = assemble_mesh(points, [(0, 1, 2, 3)], [0])
    bad = NativeMeshStreaming(mesh, source)
    with pytest.raises(ValueError, match='factorization failed'):
        P0GalerkinTransport(bad, ('reflective',)*len(mesh.bfaces))
    original = NativeMeshTransport.__post_init__

    def zero_diagonal(self):
        original(self)
        object.__setattr__(self, 'diagonal', np.zeros(self.streaming.shape))

    with patch.object(NativeMeshTransport, '__post_init__', zero_diagonal):
        with pytest.raises(ValueError, match='positive streaming diagonal'):
            P0GalerkinTransport(stream, bc)
    q = ProductQuadrature(1)
    weights = q.weights.copy()
    weights[q.directions[:, 0] > .5] *= 2
    object.__setattr__(q, 'weights', weights)
    with pytest.raises(ValueError, match='half ranges differ'):
        module._half_range(q, 1., np.array([1., 0.]))


@pytest.mark.parametrize('config', [{'ilu_drop_tolerance': 0}, {'ilu_drop_tolerance': 2},
                                    {'ilu_fill_factor': .5}, {'ilu_fill_factor': np.inf}])
def test_invalid_ilu_settings(config):
    stream, bc = bound()
    with pytest.raises(ValueError):
        P0GalerkinTransport(stream, bc, **config)


def test_invalid_projection_fields():
    stream, bc = bound()
    op = P0GalerkinTransport(stream, bc)
    for field in (np.ones((1, 2)), np.ones((2, 2), dtype=complex), np.full((2, 2), np.nan)):
        with pytest.raises(ValueError):
            op.prolong(field)
    for field in (np.zeros((2, 2)), np.full(stream.shape, np.inf)):
        with pytest.raises(ValueError):
            op.restrict(field)


def _require_packed_backend():
    from garnet.packed_superlu import _check_backend
    try:
        _check_backend()
    except RuntimeError as exc:
        pytest.skip(str(exc))
