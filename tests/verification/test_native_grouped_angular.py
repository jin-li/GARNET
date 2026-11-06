"""Manufactured operator equivalence only; no physical data or solver runs."""
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from garnet.mesh import assemble_mesh
from garnet.moment_materials import MomentMaterial
from garnet.native_angular_source import NativeAngularSource, ProductQuadrature
from garnet.native_grouped_angular import GroupAngularOperators
from garnet.native_mesh_streaming import NativeMeshStreaming
from garnet.native_mesh_transport import NativeMeshTransport


def manufactured(lmax=1, orientation='to-major', cells=3, shear=0.):
    points = np.array([(x+shear*y, y) for y in (0., 1.) for x in range(cells+1)])
    mesh = assemble_mesh(points, [(i, i+1, cells+2+i, cells+1+i) for i in range(cells)],
                         np.arange(cells) % 3)
    materials = []
    for mid in range(4):  # Fourth material deliberately has no cells.
        matrices = [np.array([[.21, .07, .02], [.13, .31, .06], [.09, .04, .25]])]
        matrices += [np.array([[-.011, .017, -.003], [.023, -.007, .005],
                               [.009, -.013, -.019]])*order for order in range(1, lmax+1)]
        matrices = [matrix*(mid+1) if mid != 2 else np.zeros_like(matrix) for matrix in matrices]
        stored = {l: p.T.copy() if orientation == 'from-major' else p.copy()
                  for l, p in enumerate(matrices)}
        materials.append(MomentMaterial(
            np.array([1.1, 1.4, 1.7]) if mid != 2 else np.zeros(3),
            np.array([1.2, .3, .1]), np.zeros(3), np.zeros(3), {}, stored, {}, {}, {},
            {'orientation_contract': orientation, 'energy_bounds_mev': [20., 2., .1, 0.]}, {}))
    source = NativeAngularSource(ProductQuadrature(lmax), tuple(materials), np.arange(cells) % 3)
    return NativeMeshStreaming(mesh, source)


def boundary(s, kind):
    return tuple('vacuum' if kind == 'vacuum' or (kind == 'mixed' and n[0] < -.5)
                 else 'reflective' for n in s.boundary_normals)


@pytest.mark.parametrize('lmax', [0, 1, 7])
@pytest.mark.parametrize('orientation', ['to-major', 'from-major'])
@pytest.mark.parametrize('kind', ['vacuum', 'reflective', 'mixed'])
def test_group_reconstruction_matches_existing_full_operators(lmax, orientation, kind):
    s = manufactured(lmax, orientation)
    bc = boundary(s, kind)
    op = GroupAngularOperators(s, bc)
    reference = NativeMeshTransport(s, bc)
    rng = np.random.default_rng(731)
    storage = rng.normal(size=(s.shape[0], s.shape[1], 6))
    flux = storage[:, ::-1, ::2]  # Signed and noncontiguous in angles/groups.
    flux.setflags(write=False)
    before = storage.copy()
    matrices = tuple(p.copy() for p in s.source.production)
    chi = tuple(m.chi.copy() for m in s.source.materials)
    moments = np.stack([op.project_group(flux[:, :, g]) for g in range(3)], axis=2)
    cached = moments.copy()
    np.testing.assert_allclose(moments, s.source.moments(flux), atol=3e-13, rtol=2e-13)
    full_scatter = s.area[:, None, None]*s.source.scatter(flux)
    rebuilt_action = np.empty(s.shape)
    for g in range(3):
        scatter = op.scatter_to_group(moments, g)
        self_scatter = op.same_group_scatter(flux[:, :, g], g)
        np.testing.assert_allclose(scatter, full_scatter[:, :, g], atol=3e-13, rtol=2e-13)
        np.testing.assert_allclose(scatter-self_scatter,
                                   op.scatter_to_group(moments, g, exclude_self=True), atol=3e-13)
        # Independently remove the source group in the coupled reference.
        off_flux = flux.copy()
        off_flux[:, :, g] = 0
        np.testing.assert_allclose(op.scatter_to_group(moments, g, exclude_self=True),
                                   s.area[:, None]*s.source.scatter(off_flux)[:, :, g], atol=3e-13)
        only = np.zeros(s.shape)
        only[:, :, g] = flux[:, :, g]
        np.testing.assert_allclose(op.group_action(flux[:, :, g], g),
                                   reference.action(only)[:, :, g], atol=4e-13, rtol=2e-13)
        np.testing.assert_allclose(op.diagonal_group(g), reference.diagonal[:, :, g], atol=2e-15)
        rebuilt_action[:, :, g] = op.streaming_group_action(flux[:, :, g], g)-scatter
    np.testing.assert_allclose(rebuilt_action, reference.action(flux), atol=4e-13, rtol=2e-13)
    np.testing.assert_array_equal(moments, cached)
    np.testing.assert_array_equal(storage, before)
    for actual, original in zip(s.source.production, matrices):
        np.testing.assert_array_equal(actual, original)
    for actual, original in zip(s.source.materials, chi):
        np.testing.assert_array_equal(actual.chi, original)
    np.testing.assert_array_equal(s.source.energy_bounds_mev, [20., 2., .1, 0.])
    assert op.streaming is s


@pytest.mark.parametrize('lmax', [0, 1])
def test_independent_p0_p1_closed_form(lmax):
    s = manufactured(lmax)
    op = GroupAngularOperators(s, boundary(s, 'vacuum'))
    q = s.quadrature
    a = np.array([1.7, -.4, .9])
    b = np.array([[.2, -.1, .7], [.6, .3, -.2], [-.4, .5, .1]]) if lmax else np.zeros((3, 3))
    field = np.broadcast_to(a+q.directions @ b, s.shape).copy()
    moments = np.stack([op.project_group(field[:, :, g]) for g in range(3)], axis=2)
    for cell, mid in enumerate(s.source.cell_material_ids):
        p = s.source.production[mid]
        expected = np.broadcast_to(a @ p[0].T, (q.n_angles, 3)).copy()
        if lmax:
            expected += (q.directions @ b) @ p[1].T
        for g in range(3):
            np.testing.assert_allclose(op.scatter_to_group(moments, g)[cell],
                                       s.area[cell]*expected[:, g], atol=3e-15)
    # Quadrature-representable local moment cache -> angular field -> moments.
    rng = np.random.default_rng(82)
    coefficients = rng.normal(size=(s.shape[0], q.harmonics.shape[1]))
    represented = coefficients @ q.harmonics.T
    np.testing.assert_allclose(op.project_group(represented), 4*np.pi*coefficients, atol=2e-14)


@pytest.mark.parametrize('kind', ['vacuum', 'reflective', 'mixed'])
def test_boundary_incoming_and_conservative_current_balance(kind):
    s = manufactured(1)
    bc = boundary(s, kind)
    op, reference = GroupAngularOperators(s, bc), NativeMeshTransport(s, bc)
    field = np.random.default_rng(7).normal(size=s.shape)
    incoming = reference.reflective_incoming(field)
    interior, outward = s.face_currents(field, incoming)
    for face, permutation in enumerate(op.reflection_permutations):
        if permutation is not None:
            np.testing.assert_array_equal(permutation[permutation], np.arange(s.shape[1]))
            np.testing.assert_array_equal(s.quadrature.weights[permutation], s.quadrature.weights)
            assert not permutation.flags.writeable
            np.testing.assert_allclose(outward[face], 0, atol=3e-15)
    for g in range(3):
        action = op.streaming_group_action(field[:, :, g], g)
        collision = np.array([s.area[c]*s.total[mid, g] for c, mid in enumerate(s.source.cell_material_ids)])
        balance = (action-collision[:, None]*field[:, :, g]) @ s.quadrature.weights
        expected = np.zeros(s.shape[0])
        for f, (left, right) in enumerate(s.owners):
            expected[left] += interior[f, g]
            expected[right] -= interior[f, g]
        np.add.at(expected, s.boundary_owners, outward[:, g])
        np.testing.assert_allclose(balance, expected, atol=5e-15)
        assert balance.sum() == pytest.approx(outward[:, g].sum(), abs=6e-15)
        np.testing.assert_allclose(action, (s.action(field)-s.boundary_rhs(incoming))[:, :, g], atol=3e-15)


def test_batches_cross_512_without_global_material_or_angle_face_work(monkeypatch):
    s = manufactured(1, cells=513)
    op = GroupAngularOperators(s, boundary(s, 'mixed'))
    field = np.random.default_rng(3).normal(size=s.shape)
    expected = s.area[:, None, None]*s.source.scatter(field)
    moments = np.stack([op.project_group(field[:, :, g]) for g in range(3)], axis=2)
    # Operator calls must never invoke the existing multigroup guards/actions.
    def forbidden(*args, **kwargs):
        raise AssertionError('multigroup path invoked')
    monkeypatch.setattr(NativeMeshStreaming, '_guard', forbidden)
    monkeypatch.setattr(NativeAngularSource, '_batch_size', forbidden)
    for g in range(3):
        np.testing.assert_allclose(op.scatter_to_group(moments, g), expected[:, :, g], atol=2e-14)
        op.group_action(field[:, :, g], g)


def test_invalid_shapes_dtypes_indices_and_nonfinite():
    s = manufactured()
    op = GroupAngularOperators(s, boundary(s, 'vacuum'))
    good = np.ones(op.shape)
    moments = np.ones((op.shape[0], 4, 3))
    for invalid in (good.tolist(), np.ones((1, 1)), good.astype(complex), good.astype(bool),
                    np.full(op.shape, np.nan), np.full(op.shape, np.inf)):
        for operation in (op.project_group, lambda x: op.group_action(x, 0)):
            with pytest.raises(ValueError):
                operation(invalid)
    for invalid in (moments[:, :, 0], moments.astype(complex), moments.astype(bool),
                    np.full(moments.shape, np.inf)):
        with pytest.raises(ValueError):
            op.scatter_to_group(invalid, 0)
    for g in (-1, True, 1., 3, '0'):
        for operation in (lambda: op.group_action(good, g), lambda: op.diagonal_group(g),
                          lambda: op.scatter_to_group(moments, g)):
            with pytest.raises(ValueError):
                operation()
    with pytest.raises(ValueError, match='boolean'):
        op.scatter_to_group(moments, 0, exclude_self=1)
    # Integer and single precision inputs are consumed without mutation.
    np.testing.assert_allclose(op.group_action(good.astype(np.int32), 0), op.group_action(good, 0))
    np.testing.assert_allclose(op.project_group(good.astype(np.float32)), op.project_group(good))
    with pytest.raises(FrozenInstanceError):
        op.max_bytes = 1


def test_explicit_boundary_validation_and_oblique_geometry():
    s = manufactured()
    for bc in ('vacuum', [], (), ('periodic',)*len(s.boundary_owners)):
        with pytest.raises(ValueError):
            GroupAngularOperators(s, bc)
    oblique = manufactured(shear=.3)
    with pytest.raises(ValueError, match='Cartesian'):
        GroupAngularOperators(oblique, boundary(oblique, 'reflective'))
    GroupAngularOperators(oblique, boundary(oblique, 'vacuum'))
    with pytest.raises(ValueError, match='NativeMeshStreaming'):
        GroupAngularOperators(None, ())


def test_budget_rejected_before_own_allocations(monkeypatch):
    s = manufactured(7)
    bc = boundary(s, 'reflective')
    normal = GroupAngularOperators(s, bc)
    limited = replace(normal, max_bytes=8*(100*s.shape[1]+1))
    field = np.ones(normal.shape)
    moments = np.ones((normal.shape[0], 64, 3))
    def forbidden(*args, **kwargs):
        raise AssertionError('allocated before budget rejection')
    monkeypatch.setattr(np, 'empty', forbidden)
    monkeypatch.setattr(np, 'zeros', forbidden)
    monkeypatch.setattr(np, 'arange', forbidden)
    monkeypatch.setattr(np, 'isfinite', forbidden)
    for operation in (lambda: limited.project_group(field), lambda: limited.scatter_to_group(moments, 0),
                      lambda: limited.same_group_scatter(field, 0), lambda: limited.group_action(field, 0),
                      lambda: limited.streaming_group_action(field, 0), lambda: limited.diagonal_group(0),
                      lambda: GroupAngularOperators(s, bc, max_bytes=1)):
        with pytest.raises(ValueError, match='allocation estimate'):
            operation()
    assert normal.allocation_estimate_bytes > limited.max_bytes


def test_invalid_reflection_permutation_is_rejected(monkeypatch):
    s = manufactured(1)
    original = ProductQuadrature.reflection
    def broken(q, axis):
        value = original(q, axis).copy()
        value[0] = value[1]
        return value
    monkeypatch.setattr(ProductQuadrature, 'reflection', broken)
    with pytest.raises(ValueError, match='exact specular'):
        GroupAngularOperators(s, boundary(s, 'reflective'))


def test_nonfinite_arithmetic_fails_closed():
    s = manufactured(1)
    op = GroupAngularOperators(s, boundary(s, 'vacuum'))
    # Finite signed inputs may overflow projection/reconstruction; no clipping.
    with np.errstate(over='ignore', invalid='ignore'):
        with pytest.raises(ValueError, match='finite'):
            op.project_group(np.full(op.shape, np.finfo(float).max))
        with pytest.raises(ValueError, match='finite'):
            op.streaming_group_action(np.full(op.shape, np.finfo(float).max), 2)
