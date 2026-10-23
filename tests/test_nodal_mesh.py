"""Independent assembler and conservation checks for rectilinear FV geometry."""
import numpy as np
import pytest

from garnet.mesh import assemble_mesh_3d, MeshGrid, MeshGroupOperator
from garnet.nodal_mesh import build_rectilinear_mesh_3d


def assembled_reference(x, y, z, mask, tags):
    """Build explicit Gmsh-order hexahedra independently of face construction."""
    coordinates = np.array([(a, b, c) for c in z for b in y for a in x], dtype=float)
    node = np.arange(len(coordinates)).reshape(len(z), len(y), len(x))
    cells = []
    for k, j, i in np.argwhere(mask):
        cells.append(tuple(node[kk, jj, ii] for kk, jj, ii in (
            (k, j, i), (k, j, i+1), (k, j+1, i+1), (k, j+1, i),
            (k+1, j, i), (k+1, j, i+1), (k+1, j+1, i+1), (k+1, j+1, i))))
    return assemble_mesh_3d(coordinates, cells, tags[mask], normal_distances=True)


def interior_rows(mesh):
    rows = []
    for index, (i, j, area, distance) in enumerate(mesh.faces):
        di, dj = mesh.face_distances[index]
        if i > j:
            i, j, di, dj = j, i, dj, di
        rows.append((i, j, area, distance, di, dj))
    return np.array(sorted(rows)).reshape(-1, 6)


def boundary_rows(mesh):
    # Owner, area and distance can coincide on opposite faces. Canonical rows
    # preserve multiplicity and compare geometry without relying on face order.
    return np.array(sorted(mesh.bfaces)).reshape(-1, 3)


@pytest.mark.parametrize('mask', [np.ones((2, 2, 3), dtype=bool),
    np.array([[[1, 1, 0], [0, 1, 1]], [[1, 0, 1], [1, 1, 0]]], dtype=bool)])
def test_unequal_masked_geometry_matches_independent_assembler(mask):
    x, y, z = [-3., -.5, .25, 4.], [-2., 1., 5.], [-1., .5, 3.]
    tags = np.arange(mask.size).reshape(mask.shape)
    mesh, topology = build_rectilinear_mesh_3d(x, y, z, active_mask=mask, tags=tags)
    reference = assembled_reference(x, y, z, mask, tags)
    np.testing.assert_allclose(mesh.centroid, reference.centroid, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(mesh.area, reference.area, rtol=2e-14)
    np.testing.assert_allclose(interior_rows(mesh), interior_rows(reference), rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(boundary_rows(mesh), boundary_rows(reference), rtol=2e-14, atol=2e-14)
    np.testing.assert_array_equal(mesh.cell_tag, reference.cell_tag)
    np.testing.assert_array_equal(topology['grid_indices'], np.argwhere(mask))
    np.testing.assert_array_equal(topology['grid_to_cell'][mask], np.arange(mask.sum()))
    assert np.all(topology['grid_to_cell'][~mask] == -1)
    assert mesh.coords.shape == (0, 3) and mesh.cells == [()]*mask.sum()


def test_face_partition_normals_closure_and_volume():
    x, y, z = [0., 1., 3., 7.], [-2., 1., 6.], [0., 2., 5.]
    mask = np.array([[[1, 1, 0], [1, 0, 1]], [[1, 1, 1], [0, 1, 1]]], dtype=bool)
    mesh, topology = build_rectilinear_mesh_3d(x, y, z, active_mask=mask, tags=np.zeros(mask.shape, dtype=int))
    faces = np.asarray(mesh.faces)
    boundaries = np.asarray(mesh.bfaces)
    owners, sides = topology['face_owners'], topology['face_sides']
    normals = topology['side_normals']
    assert len({tuple(pair) for pair in owners}) == len(owners)
    assert np.all(owners[:, 0] < owners[:, 1])
    np.testing.assert_array_equal(normals[sides[:, 0]], -normals[sides[:, 1]])
    areas = np.zeros((mesh.n_cells, 6))
    closure = np.zeros((mesh.n_cells, 3))
    for endpoint in range(2):
        np.add.at(areas, (owners[:, endpoint], sides[:, endpoint]), faces[:, 2])
        np.add.at(closure, owners[:, endpoint], faces[:, 2, None]*normals[sides[:, endpoint]])
    bowners, bsides = topology['boundary_owners'], topology['boundary_sides']
    np.add.at(areas, (bowners, bsides), boundaries[:, 1])
    np.add.at(closure, bowners, boundaries[:, 1, None]*normals[bsides])
    wx, wy, wz = topology['widths'].T
    np.testing.assert_allclose(areas, np.column_stack((wy*wz, wx*wz, wy*wz, wx*wz, wx*wy, wx*wy)))
    np.testing.assert_allclose(closure, 0, atol=1e-13)
    full_volumes = np.diff(z)[:, None, None]*np.diff(y)[None, :, None]*np.diff(x)[None, None, :]
    np.testing.assert_allclose(mesh.area.sum(), full_volumes[mask].sum())
    assert np.all(mesh.area > 0) and np.all(mesh.face_distances > 0) and np.all(boundaries[:, 2] > 0)
    assert topology['boundary_outer_box'].any() and topology['boundary_masked'].any()
    np.testing.assert_array_equal(topology['boundary_masked'], ~topology['boundary_outer_box'])
    # Independently look along every boundary normal to classify each exposure.
    for owner, side, outer in zip(bowners, bsides, topology['boundary_outer_box']):
        grid = topology['grid_indices'][owner]+normals[side][::-1]
        inside = all(0 <= i < n for i, n in zip(grid, mask.shape))
        assert outer == (not inside)
        if inside:
            assert not mask[tuple(grid)]


def test_two_material_unequal_width_current_and_conservation():
    mesh, _ = build_rectilinear_mesh_3d([0., 2., 8.], [0., 3.], [0., 5.],
                                      active_mask=np.ones((1, 1, 2), dtype=bool),
                                      tags=np.array([[[7, 9]]]))
    assert mesh.faces == [(0, 1, 15., 4.)]
    np.testing.assert_array_equal(mesh.face_distances, [[1., 3.]])
    diffusion = np.array([2., 5.])
    phi = np.array([11., 3.])
    # Two exact half-cell resistances in series, independent of the operator.
    current = 15.*(phi[0]-phi[1])/(1./diffusion[0]+3./diffusion[1])
    operator = MeshGroupOperator(np, MeshGrid(mesh, 'reflective'), diffusion,
                                 np.zeros(2), bc='reflective')
    residual = operator.apply(phi.reshape(2, 1, 1)).reshape(-1)
    np.testing.assert_allclose(residual, [current, -current], rtol=2e-14)
    np.testing.assert_allclose(residual.sum(), 0, atol=1e-13)


def test_single_cell_disconnected_cells_empty_faces_and_input_ownership():
    x = np.array([0., 1., 2., 4.])
    mask = np.array([[[1, 0, 1]]], dtype=bool)
    tags = np.array([[[3, -999, -5]]])
    mesh, topology = build_rectilinear_mesh_3d(x, [0., 2.], [0., 3.], active_mask=mask, tags=tags)
    assert mesh.faces == [] and mesh.face_distances.shape == (0, 2)
    assert len(mesh.bfaces) == 12
    np.testing.assert_array_equal(mesh.cell_tag, [3, -5])
    x[:] = 99
    tags[:] = 99
    mask[:] = False
    np.testing.assert_array_equal(topology['x_edges'], [0., 1., 2., 4.])
    np.testing.assert_array_equal(mesh.cell_tag, [3, -5])
    assert mesh.n_cells == 2
    one, topology = build_rectilinear_mesh_3d([0, 2], [0, 3], [0, 5],
                    active_mask=np.ones((1, 1, 1), dtype=bool), tags=np.ones((1, 1, 1), dtype=int))
    assert one.area.tolist() == [30.] and len(one.bfaces) == 6
    assert topology['boundary_outer_box'].all() and not topology['boundary_masked'].any()


@pytest.mark.parametrize('edges', [[0], [[0, 1]], [0, 0], [1, 0], [0, np.nan], [0, np.inf],
                                  [0j, 1+0j], ['0', '1'], [False, True], [-1e308, 1e308]])
def test_invalid_edges(edges):
    with pytest.raises(ValueError, match='x_edges'):
        build_rectilinear_mesh_3d(edges, [0, 1], [0, 1],
            active_mask=np.ones((1, 1, 1), dtype=bool), tags=np.ones((1, 1, 1), dtype=int))


@pytest.mark.parametrize('mask,tags', [
    (np.ones((1, 1, 1), dtype=int), np.ones((1, 1, 1), dtype=int)),
    (np.ones((1, 1), dtype=bool), np.ones((1, 1, 1), dtype=int)),
    (np.zeros((1, 1, 1), dtype=bool), np.ones((1, 1, 1), dtype=int)),
    (np.ones((1, 1, 1), dtype=bool), np.ones((1, 1, 1), dtype=float)),
    (np.ones((1, 1, 1), dtype=bool), np.ones((1, 1, 1), dtype=bool)),
    (np.ones((1, 1, 1), dtype=bool), np.ones((1, 1), dtype=int)),
])
def test_invalid_mask_or_tags(mask, tags):
    with pytest.raises(ValueError):
        build_rectilinear_mesh_3d([0, 1], [0, 1], [0, 1], active_mask=mask, tags=tags)


@pytest.mark.parametrize('scale', [1e200, 1e-200])
def test_nonfinite_or_underflowed_geometry_rejected(scale):
    with pytest.raises(ValueError, match='volumes'):
        build_rectilinear_mesh_3d([0, scale], [0, scale], [0, scale],
            active_mask=np.ones((1, 1, 1), dtype=bool), tags=np.ones((1, 1, 1), dtype=int))
