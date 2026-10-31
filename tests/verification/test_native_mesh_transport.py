"""Tiny independent angular eigenproblem oracles; no pin calculation."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import eig

from garnet.mesh import assemble_mesh
from garnet.moment_materials import MomentMaterial
from garnet.native_angular_source import NativeAngularSource, ProductQuadrature
from garnet.native_mesh_streaming import NativeMeshStreaming
from garnet.native_mesh_transport import NativeMeshTransport


def material(total, sp0, nsf, chi, sp1=None, kappa=None):
    total = np.asarray(total, dtype=float)
    kappa = np.full_like(total, .6) if kappa is None else np.asarray(kappa, dtype=float)
    return MomentMaterial(total, np.asarray(chi, dtype=float), np.asarray(nsf, dtype=float), kappa,
                          {}, {0: np.asarray(sp0, dtype=float),
                               1: np.zeros((len(total), len(total))) if sp1 is None else np.asarray(sp1)},
                          {}, {}, {}, {'orientation_contract': 'to-major',
                                       'energy_bounds_mev': np.arange(len(total)+1, dtype=float)[::-1]}, {})


def streaming(heterogeneous=False, shear=0., total=1., sp0=.3, nsf=.2, chi=1.):
    points = np.array([[0., 0.], [1., 0.], [3., 0.], [0., 1.], [1., 1.], [3., 1.]])
    points[:, 0] += shear*points[:, 1]
    cells = [(0, 1, 4, 3), (1, 2, 5, 4)] if heterogeneous else [(0, 1, 4, 3)]
    mesh = assemble_mesh(points, cells, np.arange(len(cells)))
    if heterogeneous:
        materials = (material([1.3, 1.7], [[.25, .08], [.12, .35]], [.12, .3], [.7, .3],
                              [[.015, -.004], [.003, .02]]),
                     material([1.1, 1.5], [[.22, .07], [.1, .31]], [.08, .22], [1.2, .2],
                              [[.01, .002], [-.003, .018]]))
    else:
        materials = (material([total], [[sp0]], [nsf], [chi]),)
    source = NativeAngularSource(ProductQuadrature(1), materials, np.arange(mesh.n_cells))
    return NativeMeshStreaming(mesh, source)


def solver(s=None, **kwargs):
    s = streaming() if s is None else s
    bc = kwargs.pop('boundary_conditions', ('reflective',)*len(s.boundary_owners))
    return NativeMeshTransport(s, bc, **kwargs)


def independent_dense(s, bc):
    """Direct P0/P1 kernel and face-row assembly; no action/assembled columns."""
    nc, na, ng = s.shape
    size = nc*na*ng
    a, f = np.zeros((size, size)), np.zeros((size, size))
    functional = np.zeros(size)
    directions, weights = s.quadrature.directions, s.quadrature.weights

    def index(cell, angle, group):
        return (cell*na+angle)*ng+group

    for cell, mid in enumerate(s.source.cell_material_ids):
        m = s.source.materials[mid]
        p0, p1 = m.production[0], m.production[1]
        if m.provenance['orientation_contract'] == 'from-major':
            p0, p1 = p0.T, p1.T
        for n in range(na):
            for g in range(ng):
                row = index(cell, n, g)
                a[row, row] += s.area[cell]*m.total[g]
                functional[row] = s.area[cell]*weights[n]*m.nu_sigma_f[g]
                for nprime in range(na):
                    dot = sum(directions[n, axis]*directions[nprime, axis] for axis in (0, 1, 2))
                    for gp in range(ng):
                        column = index(cell, nprime, gp)
                        a[row, column] -= s.area[cell]*weights[nprime]*(p0[g, gp]+3*dot*p1[g, gp])/(4*np.pi)
                        f[row, column] += s.area[cell]*m.chi[g]*m.nu_sigma_f[gp]*weights[nprime]/(4*np.pi)
    for face in range(len(s.owners)):
        left, right = s.owners[face]
        for n in range(na):
            coefficient = s.measures[face]*sum(directions[n, axis]*s.normals[face, axis] for axis in (0, 1))
            upstream = left if coefficient >= 0 else right
            for g in range(ng):
                a[index(left, n, g), index(upstream, n, g)] += coefficient
                a[index(right, n, g), index(upstream, n, g)] -= coefficient
    for face in range(len(s.boundary_owners)):
        owner, normal = s.boundary_owners[face], s.boundary_normals[face]
        for n in range(na):
            coefficient = s.boundary_measures[face]*sum(directions[n, axis]*normal[axis] for axis in (0, 1))
            for g in range(ng):
                if coefficient > 0:
                    a[index(owner, n, g), index(owner, n, g)] += coefficient
                elif coefficient < 0 and bc[face] == 'reflective':
                    target = directions[n].copy()
                    target[:2] -= 2*np.dot(target[:2], normal)*normal
                    reflected = int(np.argmin(np.linalg.norm(directions-target, axis=1)))
                    assert np.linalg.norm(directions[reflected]-target) < 1e-12
                    a[index(owner, n, g), index(owner, reflected, g)] += coefficient
    return a, f, functional


def test_homogeneous_reflective_analytic_k_and_immutable_cold_result():
    op = solver()
    result = op.solve()
    assert result.converged, result.reason
    assert result.k_eff == pytest.approx(.2/(1.-.3), rel=2e-7)
    np.testing.assert_allclose(result.angular_flux, result.angular_flux.flat[0], rtol=2e-7)
    assert op.production(result.angular_flux) == pytest.approx(1., abs=2e-15)
    assert result.history[-1]['true_inner_residual'] <= 1e-7
    assert result.history[-1]['full_residual'] <= 1e-5
    assert result.provenance['cold_start'] and result.provenance['linear_initial_guess'] == 'zero_every_outer_step'
    assert not result.provenance['physical_admission']
    for value in (result.angular_flux, result.scalar_flux, op.diagonal, op.chi, op.nu_sigma_f, op.kappa_fission):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(TypeError):
        result.history[0]['k_eff'] = 0
    with pytest.raises(TypeError):
        result.provenance['cold_start'] = False


@pytest.mark.parametrize('chi', [.8, 1.7])
def test_native_chi_unchanged_no_renormalization(chi):
    op = solver(streaming(chi=chi))
    np.testing.assert_array_equal(op.chi, [[chi]])
    result = op.solve()
    assert result.converged, result.reason
    assert result.k_eff == pytest.approx(chi*.2/(1.-.3), rel=2e-7)


def test_reflection_arbitrary_field_both_axes_heterogeneous_weighted_net_zero():
    op = solver(streaming(heterogeneous=True))
    s = op.streaming
    flux = np.arange(np.prod(s.shape), dtype=float).reshape(s.shape)/11 - .3
    incoming = op.reflective_incoming(flux)
    _, currents = s.face_currents(flux, incoming)
    np.testing.assert_allclose(currents, 0, atol=4e-15)
    axes = set()
    for face, permutation in enumerate(op.reflection_permutations):
        normal = s.boundary_normals[face]
        axis = int(np.argmax(np.abs(normal)))
        axes.add(axis)
        np.testing.assert_array_equal(permutation[permutation], np.arange(s.shape[1]))
        np.testing.assert_array_equal(s.quadrature.weights[permutation], s.quadrature.weights)
        np.testing.assert_allclose(s.quadrature.directions[permutation, axis], -s.quadrature.directions[:, axis], atol=2e-15)
        for tangent in ({0, 1, 2}-{axis}):
            np.testing.assert_allclose(s.quadrature.directions[permutation, tangent], s.quadrature.directions[:, tangent], atol=2e-15)
    assert axes == {0, 1}


@pytest.mark.parametrize('vacuum_left', [False, True])
def test_heterogeneous_dense_generalized_p0_p1_oracle(vacuum_left):
    s = streaming(heterogeneous=True)
    bc = tuple('vacuum' if vacuum_left and normal[0] < -.5 else 'reflective' for normal in s.boundary_normals)
    op = solver(s, boundary_conditions=bc, inner_tolerance=1e-11, k_tolerance=1e-10,
                source_tolerance=1e-9, full_tolerance=1e-9, max_outer=200, max_inner=200, restart=32)
    a, f, p = independent_dense(s, bc)
    field = np.arange(np.prod(s.shape), dtype=float).reshape(s.shape)/19 - .5
    np.testing.assert_allclose(op.action(field).ravel(), a @ field.ravel(), atol=3e-15)
    np.testing.assert_allclose(op.fission(field).ravel(), f @ field.ravel(), atol=3e-16)
    assert op.production(field) == pytest.approx(p @ field.ravel(), abs=4e-15)
    values, vectors = eig(f, a)
    admitted = np.flatnonzero((np.abs(values.imag) < 1e-10) & (values.real > 0) & np.isfinite(values))
    chosen = admitted[np.argmax(values[admitted].real)]
    eigenvalue = values[chosen].real
    other_finite = np.isfinite(values) & (np.arange(values.size) != chosen)
    assert np.all(np.abs(values[other_finite]) < eigenvalue*(1-1e-6))
    eigenvector = vectors[:, chosen].real
    eigenvector /= p @ eigenvector
    roundoff = 64*np.finfo(float).eps*np.max(np.abs(eigenvector))
    assert np.min(eigenvector) >= -roundoff
    result = op.solve()
    assert result.converged, (result.reason, result.history[-1])
    assert result.k_eff == pytest.approx(eigenvalue, rel=2e-9)
    np.testing.assert_allclose(result.angular_flux.ravel(), eigenvector, rtol=2e-8, atol=2e-9)
    rhs = f @ result.angular_flux.ravel()/result.k_eff
    true_full = np.linalg.norm(a @ result.angular_flux.ravel()-rhs)/np.linalg.norm(rhs)
    assert true_full <= 1e-9
    assert result.history[-1]['full_residual'] == pytest.approx(true_full, abs=2e-14)
    assert result.diagnostics['scalar_nonnegative'] and result.diagnostics['heat_power_nonnegative']


def test_fixed_source_true_residual_zero_start_and_step_limit():
    op = solver(streaming(heterogeneous=True), max_inner=200, restart=32)
    rhs = np.arange(np.prod(op.streaming.shape), dtype=float).reshape(op.streaming.shape)/31 + .1
    result = op.fixed_source(rhs)
    assert result.converged, result.reason
    assert result.gmres_info == 0
    norm = np.linalg.norm((op.action(result.angular_flux)-rhs).ravel())/np.linalg.norm(rhs.ravel())
    assert norm <= 1e-7
    assert result.true_residual == pytest.approx(norm, abs=1e-16)
    limited = solver(op.streaming, max_inner=1, restart=1)
    failure = limited.fixed_source(rhs)
    assert not failure.converged and failure.inner_steps <= 1
    assert failure.gmres_info != 0 or failure.true_residual > limited.inner_tolerance
    outer_failure = limited.solve()
    assert not outer_failure.converged and outer_failure.reason.startswith('inner_failure:')


def test_outer_exhaustion_and_no_k_only_acceptance():
    op = solver(streaming(heterogeneous=True), max_outer=1, max_inner=200, restart=32)
    result = op.solve()
    assert not result.converged and result.reason == 'outer_iteration_limit'
    assert len(result.history) == 1 and result.history[0]['inner_info'] == 0


@pytest.mark.parametrize('total, sp0, nsf', [(0., 0., .2), (1., 1., .2), (0., 0., 0.)])
def test_singular_or_void_fails_closed(total, sp0, nsf):
    result = solver(streaming(total=total, sp0=sp0, nsf=nsf), max_inner=10, restart=4, max_outer=3).solve()
    assert not result.converged
    assert result.reason.startswith(('inner_failure:', 'nonpositive_initial_production', 'nonpositive_k'))
    assert np.all(np.isfinite(result.angular_flux))


def test_exact_zero_rhs_and_zero_diagonal_fail_without_floor():
    op = solver()
    result = op.fixed_source(np.zeros(op.streaming.shape))
    assert not result.converged and result.reason == 'zero_or_nonfinite_rhs_norm'
    # Inject a degenerate diagonal to test the numerical guard directly. Valid
    # bounded closed cells normally have nonzero in-plane streaming outflow.
    object.__setattr__(op, 'diagonal', np.zeros(op.streaming.shape))
    result = op.fixed_source(np.ones(op.streaming.shape))
    assert not result.converged and result.reason == 'zero_streaming_diagonal'


def test_positivity_scales_angular_diagnostics_and_nonpositive_production():
    op = solver()
    # Large anisotropy gives signed angular flux but positive scalar and power.
    directions = op.streaming.quadrature.directions
    flux = np.broadcast_to((1+3*directions[:, 0])[None, :, None], op.streaming.shape)
    diagnostics = op._diagnostics(flux)
    assert diagnostics['scalar_nonnegative'] and diagnostics['heat_power_nonnegative']
    assert diagnostics['angular_flux_min'] < 0 and diagnostics['weighted_negative_angular_mass'] > 0
    assert not op._diagnostics(-np.ones(op.streaming.shape))['scalar_nonnegative']
    assert not op._diagnostics(-np.ones(op.streaming.shape))['heat_power_nonnegative']
    tiny = np.full(op.streaming.shape, -1e-14)
    assert op._diagnostics(tiny)['scalar_nonnegative']
    strict = solver(scalar_flux_scale=1e-3, power_scale=1e-3)
    assert not strict._diagnostics(tiny)['scalar_nonnegative']
    assert not strict._diagnostics(tiny)['heat_power_nonnegative']
    with pytest.raises(ValueError, match='nu_sigma_f'):
        solver(streaming(nsf=-.2))


def test_heat_diagnostics_use_native_kappa_not_neutron_production():
    s = streaming(heterogeneous=True)
    op = solver(s)
    flux = np.arange(np.prod(s.shape), dtype=float).reshape(s.shape)/17 + 1
    scalar = np.sum(s.quadrature.weights[None, :, None]*flux, axis=1)
    heat = np.array([s.area[cell]*np.dot(s.source.materials[mid].kappa_fission, scalar[cell])
                     for cell, mid in enumerate(s.source.cell_material_ids)])
    production = np.array([s.area[cell]*np.dot(s.source.materials[mid].nu_sigma_f, scalar[cell])
                           for cell, mid in enumerate(s.source.cell_material_ids)])
    diagnostics = op._diagnostics(flux)
    assert diagnostics['cell_heat_power_min'] == pytest.approx(heat.min())
    assert diagnostics['total_heat_power'] == pytest.approx(heat.sum())
    assert diagnostics['total_neutron_production'] == pytest.approx(production.sum())
    assert not np.isclose(heat.sum(), production.sum())


@pytest.mark.parametrize('field', ['chi', 'nu_sigma_f', 'kappa_fission'])
@pytest.mark.parametrize('value', [-1., np.nan, np.inf])
def test_physical_fission_vectors_reject_negative_nonfinite(field, value):
    s = streaming()
    bad_material = replace(s.source.materials[0], **{field: np.array([value])})
    bad_source = replace(s.source, materials=(bad_material,))
    points = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    mesh = assemble_mesh(points, [(0, 1, 2, 3)], [0])
    with pytest.raises(ValueError, match=field):
        solver(NativeMeshStreaming(mesh, bad_source))


def test_boundary_validation_oblique_vacuum_and_invalid_controls():
    s = streaming()
    for bc in ('reflective', (), ('periodic',)*len(s.boundary_owners)):
        with pytest.raises(ValueError):
            NativeMeshTransport(s, bc)
    oblique = streaming(shear=.4)
    with pytest.raises(ValueError, match='Cartesian'):
        solver(oblique)
    vacuum = solver(oblique, boundary_conditions=('vacuum',)*len(oblique.boundary_owners))
    assert all(value is None for value in vacuum.reflection_permutations)
    for control in ({'max_inner': 0}, {'max_outer': -1}, {'restart': True},
                    {'inner_tolerance': 1e-4}, {'k_tolerance': np.nan},
                    {'scalar_flux_scale': 0}, {'power_scale': -1}):
        with pytest.raises(ValueError):
            solver(s, **control)
    for operation in (lambda: vacuum.action(np.zeros((2, 2))),
                      lambda: vacuum.fixed_source(np.full(s.shape, np.nan))):
        with pytest.raises(ValueError):
            operation()
    with pytest.raises(ValueError, match='allocation estimate'):
        solver(s, max_bytes=1)
    with pytest.raises(ValueError, match='allocation estimate'):
        solver(s, max_bytes=100000, max_outer=100)
