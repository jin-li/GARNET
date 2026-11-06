"""Tiny algebraic streaming checks; no transport solve or physical qualification."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import coo_matrix

import garnet.native_mesh_streaming as streaming_module
from garnet.mesh import assemble_mesh
from garnet.moment_materials import MomentMaterial
from garnet.native_angular_source import NativeAngularSource, ProductQuadrature
from garnet.native_mesh_streaming import NativeMeshStreaming


def mesh(two=True, shear=0., scale=1.):
    points = np.array([[0., 0.], [1., 0.], [3., 0.], [0., 1.], [1., 1.], [3., 1.]])
    points[:, 0] += shear*points[:, 1]
    points *= scale
    cells = [(0, 1, 4, 3), (1, 2, 5, 4)] if two else [(0, 1, 4, 3)]
    return assemble_mesh(points, cells, np.arange(len(cells)))


def material(total, lmax=1):
    total = np.asarray(total, dtype=float)
    groups = total.size
    return MomentMaterial(total, np.zeros(groups), np.zeros(groups), np.zeros(groups), {},
                          {l: np.eye(groups)*(.2 if l == 0 else -.1) for l in range(lmax+1)},
                          {}, {}, {}, {'orientation_contract': 'to-major',
                                       'energy_bounds_mev': np.arange(groups+1, dtype=float)[::-1]}, {})


def operator(two=True, shear=0., scale=1., lmax=1, max_bytes=256*1024**2):
    grid = mesh(two, shear, scale)
    materials = (material([.7, 1.2], lmax), material([.4, 0.], lmax))
    source = NativeAngularSource(ProductQuadrature(lmax, n_azimuth=4), materials,
                                 np.arange(grid.n_cells))
    return grid, NativeMeshStreaming(grid, source, max_bytes=max_bytes)


def test_one_cell_analytic_and_unrenormalized_xy():
    _, op = operator(two=False)
    flux = np.full(op.shape, 2.)
    speed = np.sqrt(2/3)  # polar cosine +/-sqrt(1/3), not planar unit length
    expected = 2*(np.array([.7, 1.2]) + speed)
    np.testing.assert_allclose(op.action(flux), np.broadcast_to(expected, op.shape), atol=1e-15)
    np.testing.assert_array_equal(op.boundary_rhs(), np.zeros(op.shape))
    incoming = np.full((4, op.shape[1], 2), 3.)
    np.testing.assert_allclose(op.boundary_rhs(incoming), 3*speed, atol=1e-15)


def test_two_cell_unequal_volume_both_streaming_directions_and_tangent():
    _, op = operator()
    flux = np.ones(op.shape)
    flux[0] = [2., 3.]
    flux[1] = [5., 7.]
    speed = np.sqrt(2/3)
    for angle, left, right in ((0, np.array([.7, 1.2])*flux[0, 0] + speed*flux[0, 0],
                                  2*np.array([.4, 0.])*flux[1, 0] + speed*(flux[1, 0]-flux[0, 0])),
                               (2, np.array([.7, 1.2])*flux[0, 0] + speed*(flux[0, 0]-flux[1, 0]),
                                  2*np.array([.4, 0.])*flux[1, 0] + speed*flux[1, 0])):
        np.testing.assert_allclose(op.action(flux)[:, angle], [left, right], atol=2e-15)
    # y-directed ordinate is tangent to the shared vertical edge, up to trig roundoff.
    tangential = op.assembled(1, 0).toarray()
    np.testing.assert_allclose(tangential[0, 1], 0, atol=1e-15)
    np.testing.assert_allclose(tangential[1, 0], 0, atol=1e-15)


@pytest.mark.parametrize('shear', [0., .4])
def test_independent_sparse_action_rhs_oracle_and_m_matrix(shear):
    _, op = operator(shear=shear)
    flux = np.arange(np.prod(op.shape), dtype=float).reshape(op.shape)/13 - .4
    incoming = np.arange(len(op.boundary_owners)*op.shape[1]*2, dtype=float).reshape(-1, op.shape[1], 2)/17
    rhs = np.zeros(op.shape)
    # Scalar direct boundary oracle independent of boundary_rhs's indexed accumulation.
    for face in range(len(op.boundary_owners)):
        for angle in range(op.shape[1]):
            dot = sum(op.quadrature.directions[angle, axis]*op.boundary_normals[face, axis]
                      for axis in (0, 1))
            for group in range(2):
                if dot < 0:
                    rhs[op.boundary_owners[face], angle, group] += (
                        -op.boundary_measures[face]*dot*incoming[face, angle, group])
    np.testing.assert_allclose(op.boundary_rhs(incoming), rhs, atol=2e-15)
    result = op.action(flux)
    for angle in range(op.shape[1]):
        for group in range(2):
            matrix = op.assembled(angle, group)
            np.testing.assert_allclose(matrix @ flux[:, angle, group], result[:, angle, group], atol=2e-15)
            dense = matrix.toarray()
            assert np.all(np.diag(dense) >= 0)
            assert np.all((dense-np.diag(np.diag(dense))) <= 0)
    np.testing.assert_allclose(op.residual(flux, incoming), result-rhs, atol=2e-15)


@pytest.mark.parametrize('scale', [1e-6, 1., 1e6])
def test_constant_field_matched_incoming_and_geometric_scaling(scale):
    _, op = operator(shear=.4, scale=scale)
    flux = np.broadcast_to(np.array([2., 3.]), op.shape).copy()
    incoming = np.broadcast_to(flux[0], (len(op.boundary_owners), op.shape[1], 2)).copy()
    expected = np.stack([op.area[c]*op.total[m]*flux[c] for c, m in enumerate(op.source.cell_material_ids)])
    np.testing.assert_allclose(op.residual(flux, incoming), expected,
                               atol=5e-15*max(scale, scale**2), rtol=2e-14)


def test_global_interior_cancellation_and_physical_face_currents():
    _, op = operator(shear=.4)
    flux = np.arange(np.prod(op.shape), dtype=float).reshape(op.shape)/7 + 1
    incoming = np.full((len(op.boundary_owners), op.shape[1], 2), .3)
    collision = np.stack([op.area[c]*op.total[m]*flux[c]
                          for c, m in enumerate(op.source.cell_material_ids)])
    interior, boundary = op.face_currents(flux, incoming)
    net = np.einsum('n,cng->g', op.quadrature.weights, op.residual(flux, incoming)-collision)
    np.testing.assert_allclose(net, boundary.sum(axis=0), atol=2e-14)
    # Direct physical angular-current oracle on the oblique shared face.
    normal = np.array([1., -.4])/np.sqrt(1.16)
    speed = op.quadrature.directions[:, :2] @ normal
    trace = np.where(speed[:, None] >= 0, flux[0], flux[1])
    expected = np.sqrt(1.16)*np.sum(op.quadrature.weights[:, None]*speed[:, None]*trace, axis=0)
    np.testing.assert_allclose(interior[0], expected, atol=2e-15)


def test_source_coupling_volume_exactly_once_and_signed_source():
    _, op = operator(shear=.4)
    flux = np.arange(np.prod(op.shape), dtype=float).reshape(op.shape)/11 - 2
    native = op.source.scatter(flux)
    difference = op.residual(flux)-op.residual(flux, include_scattering=True)
    np.testing.assert_allclose(difference, op.area[:, None, None]*native, atol=1e-15)


def test_outgoing_incoming_values_ignored_and_vacuum():
    _, op = operator(shear=.4)
    incoming = np.ones((len(op.boundary_owners), op.shape[1], 2))
    masked = incoming.copy()
    for face, normal in enumerate(op.boundary_normals):
        masked[face, op.quadrature.directions[:, :2] @ normal >= 0] = np.nan
    np.testing.assert_array_equal(op.boundary_rhs(masked), op.boundary_rhs(incoming))
    flux = np.ones(op.shape)
    for actual, expected in zip(op.face_currents(flux, masked), op.face_currents(flux, incoming)):
        np.testing.assert_array_equal(actual, expected)
    zero = np.zeros_like(incoming)
    np.testing.assert_array_equal(op.boundary_rhs(), op.boundary_rhs(zero))
    with pytest.raises(ValueError, match='nonfinite'):
        op.boundary_rhs(np.full_like(incoming, np.nan))


def test_consumed_arrays_frozen_and_mesh_mutation_isolated():
    grid, op = operator()
    before = op.action(np.ones(op.shape))
    grid.area[:] = 99
    grid.face_normals[:] *= -1
    grid.bface_centers[:] = 999
    np.testing.assert_array_equal(op.action(np.ones(op.shape)), before)
    for name in ('area', 'centroid', 'owners', 'measures', 'normals', 'centers',
                 'boundary_owners', 'boundary_measures', 'boundary_normals', 'boundary_centers', 'total'):
        with pytest.raises(ValueError):
            getattr(op, name).setflags(write=True)
    with pytest.raises(AttributeError):
        op.total = np.ones((2, 2))


def test_interior_orientation_contract_owner_swap_and_normal_reversal():
    grid, op = operator(shear=.4)
    left, right, measure, distance = grid.faces[0]
    swapped = replace(grid, faces=[(right, left, measure, distance)],
                      face_normals=-grid.face_normals)
    reversed_op = NativeMeshStreaming(swapped, op.source)
    flux = np.arange(np.prod(op.shape), dtype=float).reshape(op.shape)/7 - .5
    np.testing.assert_allclose(reversed_op.action(flux), op.action(flux), atol=2e-15)
    with pytest.raises(ValueError, match='left-to-right'):
        NativeMeshStreaming(replace(grid, faces=[(right, left, measure, distance)]), op.source)


@pytest.mark.parametrize('invalid_total', [np.array([1+2j, 1+0j]), np.array(['bad', 'total'])])
def test_reject_nonreal_total_dtype(invalid_total):
    grid, op = operator()
    malformed = replace(op.source.materials[0])
    # MomentMaterial normally casts ndarray fields to float64. Inject malformed
    # legacy data to verify streaming's own validation boundary, before arithmetic.
    object.__setattr__(malformed, 'total', invalid_total)
    bad_source = NativeAngularSource(op.quadrature, (malformed, op.source.materials[1]), np.array([0, 1]))
    with pytest.raises(ValueError, match='native total'):
        NativeMeshStreaming(grid, bad_source)


def test_bad_geometry_and_source_binding():
    grid, op = operator()
    bad_cases = [replace(grid, face_normals=None), replace(grid, bface_centers=None),
                 replace(grid, area=np.array([0., 2.])),
                 replace(grid, centroid=np.ones((2, 3))),
                 replace(grid, face_normals=-grid.face_normals),
                 replace(grid, face_normals=2*grid.face_normals),
                 replace(grid, bface_normals=-grid.bface_normals),
                 replace(grid, faces=[(0, 0, 1., 1.)]),
                 replace(grid, faces=[(0, 2, 1., 1.)]),
                 replace(grid, faces=[(0., 1, 1., 1.)]),
                 replace(grid, faces=[(0, 1, -1., 1.)]),
                 replace(grid, faces=[(0, 1, np.nan, 1.)])]
    # Positive but inconsistent measure preserves orientation and fails closure.
    bad_cases.append(replace(grid, faces=[(0, 1, 1.1, 1.)]))
    for bad in bad_cases:
        with pytest.raises(ValueError):
            NativeMeshStreaming(bad, op.source)
    with pytest.raises(ValueError, match='cell count mismatch'):
        NativeMeshStreaming(mesh(two=False), op.source)
    for total in ([-1., 1.], [np.inf, 1.], [np.nan, 1.]):
        bad_material = replace(op.source.materials[0], total=np.asarray(total))
        bad_source = NativeAngularSource(op.quadrature, (bad_material, op.source.materials[1]), np.array([0, 1]))
        with pytest.raises(ValueError, match='native total'):
            NativeMeshStreaming(grid, bad_source)


def test_zero_total_no_invertibility_promise():
    grid, op = operator()
    materials = tuple(replace(m, total=np.zeros(2)) for m in op.source.materials)
    zero = NativeMeshStreaming(grid, NativeAngularSource(op.quadrature, materials, np.array([0, 1])))
    np.testing.assert_array_equal(zero.total, 0)
    assert np.all(np.isfinite(zero.action(np.ones(op.shape))))


def test_input_and_budget_failures():
    grid, op = operator()
    for bad in (np.ones((2, 2)), np.ones(op.shape, dtype=complex), np.full(op.shape, np.inf)):
        with pytest.raises(ValueError):
            op.action(bad)
    with pytest.raises(ValueError):
        op.boundary_rhs(np.zeros((1, 1, 1)))
    for indices in ((-1, 0), (op.shape[1], 0), (0, 2), (True, 0)):
        with pytest.raises(ValueError):
            op.assembled(*indices)
    with pytest.raises(ValueError, match='allocation estimate'):
        NativeMeshStreaming(grid, op.source, max_bytes=1)
    # Construction fits; runtime flux/source/current workspace does not.
    limited = NativeMeshStreaming(grid, op.source, max_bytes=3000)
    for operation in (lambda: limited.action(np.ones(op.shape)),
                      lambda: limited.boundary_rhs(),
                      lambda: limited.face_currents(np.ones(op.shape))):
        with pytest.raises(ValueError, match='allocation estimate'):
            operation()


def old_scalar_coo(op, angle, group):
    """Literal pre-vectorization builder; retain its operation/insertion order."""
    nc = len(op.area)
    omega = op.quadrature.directions[angle]
    rows, columns, values = [], [], []
    for cell in range(nc):
        rows.append(cell)
        columns.append(cell)
        values.append(op.area[cell]*op.total[op.source.cell_material_ids[cell], group])
    for face in range(len(op.owners)):
        left, right = op.owners[face]
        nx, ny = op.normals[face]
        coefficient = op.measures[face]*(omega[0]*nx + omega[1]*ny)
        upstream = left if coefficient >= 0 else right
        rows.extend((left, right))
        columns.extend((upstream, upstream))
        values.extend((coefficient, -coefficient))
    for face in range(len(op.boundary_owners)):
        nx, ny = op.boundary_normals[face]
        coefficient = op.boundary_measures[face]*(omega[0]*nx + omega[1]*ny)
        if coefficient > 0:
            owner = op.boundary_owners[face]
            rows.append(owner)
            columns.append(owner)
            values.append(coefficient)
    return (np.array(rows, dtype=np.int64), np.array(columns, dtype=np.int64),
            np.array(values, dtype=np.float64))


def manufactured_assembly_storage():
    """Algebraic storage only, without claiming validated physical geometry.

    Repeated owners, permuted material IDs, zero totals and exact tangencies
    make COO ordering observable. Real constructed operators are checked too.
    """
    directions = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.],
                           [.6, .8, 0.], [-.6, -.8, 0.], [0., 0., 1.]])
    return SimpleNamespace(
        area=np.array([2., 3., 5.]),
        total=np.array([[0., .25, 4.], [2., .5, 8.]]),
        source=SimpleNamespace(cell_material_ids=np.array([1, 0, 1], dtype=np.int64)),
        owners=np.array([[0, 1], [2, 1], [0, 2], [1, 0]], dtype=np.int64),
        measures=np.array([2., .5, 4., 3.]),
        normals=np.array([[1., 0.], [0., -1.], [-.6, .8], [-1., 0.]]),
        boundary_owners=np.array([0, 0, 2, 1, 2], dtype=np.int64),
        boundary_measures=np.array([2., 3., .5, 4., 5.]),
        boundary_normals=np.array([[1., 0.], [-1., 0.], [0., 1.],
                                   [0., -1.], [.6, .8]]),
        quadrature=SimpleNamespace(directions=directions), shape=(3, 6, 3),
        max_bytes=256*1024**2, _finite=NativeMeshStreaming._finite)


def assert_old_coo_equivalence(op, angle, group, monkeypatch):
    rows, columns, values = old_scalar_coo(op, angle, group)
    expected = coo_matrix((values, (rows, columns)), shape=(len(op.area),)*2).tocsr()
    captured = []

    def checked_coo(entries, shape):
        actual_values, (actual_rows, actual_columns) = entries
        for actual, want in ((actual_rows, rows), (actual_columns, columns)):
            assert actual.dtype == np.dtype(np.int64)
            np.testing.assert_array_equal(actual, want)
        assert actual_values.dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(actual_values.view(np.uint64), values.view(np.uint64))
        captured.append(True)
        return coo_matrix(entries, shape=shape)

    with monkeypatch.context() as patch:
        patch.setattr(streaming_module, 'coo_matrix', checked_coo)
        actual = NativeMeshStreaming.assembled(op, angle, group)
    assert captured == [True]
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.data.view(np.uint64), expected.data.view(np.uint64))
    for flux in (np.ones(len(op.area)), np.arange(len(op.area), dtype=float)-.75):
        np.testing.assert_array_equal((actual @ flux).view(np.uint64),
                                      (expected @ flux).view(np.uint64))
    return rows, columns, values


@pytest.mark.parametrize('shear', [0., .4])
def test_assembled_matches_old_scalar_coo_on_validated_mesh(shear, monkeypatch):
    _, op = operator(shear=shear)
    for angle in range(op.shape[1]):
        for group in range(op.shape[2]):
            assert_old_coo_equivalence(op, angle, group, monkeypatch)


def test_assembled_manufactured_directions_groups_and_readonly_storage(monkeypatch):
    op = manufactured_assembly_storage()
    arrays = [value for value in vars(op).values() if isinstance(value, np.ndarray)]
    arrays += [op.source.cell_material_ids, op.quadrature.directions]
    originals = [value.tobytes() for value in arrays]
    for value in arrays:
        value.flags.writeable = False
    for angle in range(op.shape[1]):
        for group in range(op.shape[2]):
            assert_old_coo_equivalence(op, np.int64(angle), np.int32(group), monkeypatch)
    assert [value.tobytes() for value in arrays] == originals
    assert all(not value.flags.writeable for value in arrays)
    rows, columns, values = assert_old_coo_equivalence(op, 5, 0, monkeypatch)
    nc = len(op.area)
    assert len(rows) == nc + 2*len(op.owners)  # tangent boundaries omitted
    np.testing.assert_array_equal(columns[nc::2], op.owners[:, 0])
    assert np.all(values[nc:] == 0)
    np.testing.assert_array_equal(np.signbit(values[nc+1::2]), ~np.signbit(values[nc::2]))


def test_assembled_cancellation_preserves_multiply_add_order(monkeypatch):
    op = manufactured_assembly_storage()
    op.normals[:3] = [[.8, -.6], [np.nextafter(.8, np.inf), -.6],
                      [np.nextafter(.8, 0.), -.6]]
    for angle in (3, 4):
        for group in range(op.shape[2]):
            _, columns, values = assert_old_coo_equivalence(op, angle, group, monkeypatch)
            assert values[len(op.area)] == 0.
            assert columns[len(op.area)] == op.owners[0, 0]


@pytest.mark.parametrize('empty_interior,empty_boundary', [(True, False), (False, True), (True, True)])
def test_assembled_empty_face_storage(empty_interior, empty_boundary, monkeypatch):
    op = manufactured_assembly_storage()
    if empty_interior:
        op.owners = np.empty((0, 2), dtype=np.int64)
        op.measures = np.empty(0)
        op.normals = np.empty((0, 2))
    if empty_boundary:
        op.boundary_owners = np.empty(0, dtype=np.int64)
        op.boundary_measures = np.empty(0)
        op.boundary_normals = np.empty((0, 2))
    for angle in (0, 1, 5):
        assert_old_coo_equivalence(op, angle, 0, monkeypatch)


def test_assembled_budget_unchanged_and_guard_precedes_workspace(monkeypatch):
    op = manufactured_assembly_storage()
    required = 128*(len(op.area)+4*len(op.owners)+len(op.boundary_owners)+1)
    op.max_bytes = required-1

    def unexpected(*args, **kwargs):
        raise AssertionError('allocated or computed workspace before budget guard')

    with monkeypatch.context() as patch:
        for name in ('empty', 'arange', 'flatnonzero', 'isfinite'):
            patch.setattr(streaming_module.np, name, unexpected)
        with pytest.raises(ValueError, match=f'allocation estimate {required} bytes'):
            NativeMeshStreaming.assembled(op, 0, 0)
    op.max_bytes = required
    assert_old_coo_equivalence(op, 0, 0, monkeypatch)


@pytest.mark.parametrize('angle,group', [(-1, 0), (6, 0), (True, 0), (0., 0),
                                         (0, -1), (0, 3), (0, False), (0, 1.)])
def test_assembled_indices_rejected_before_allocation(angle, group, monkeypatch):
    op = manufactured_assembly_storage()
    with monkeypatch.context() as patch:
        patch.setattr(streaming_module.np, 'empty',
                      lambda *args, **kwargs: pytest.fail('allocated for invalid index'))
        with pytest.raises(ValueError):
            NativeMeshStreaming.assembled(op, angle, group)


@pytest.mark.parametrize('field,index', [('area', 0), ('measures', 0),
                                        ('boundary_measures', 0), ('boundary_measures', 1)])
def test_assembled_nonfinite_intermediates_rejected(field, index):
    op = manufactured_assembly_storage()
    getattr(op, field)[index] = np.finfo(np.float64).max
    # Manufactured nonunit direction induces positive/negative boundary overflow.
    # Negative overflow used to be omitted; the new builder rejects it as well.
    if field != 'area':
        op.quadrature.directions[0, 0] = 2.
    with pytest.raises(ValueError, match='nonfinite'):
        NativeMeshStreaming.assembled(op, 0, 0)
