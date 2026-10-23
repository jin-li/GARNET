"""Physical boundary labels and oriented geometry for fitted FV meshes."""

import numpy as np
import pytest

from garnet.mesh import Mesh, assemble_mesh, assemble_mesh_3d, read_gmsh


def _assert_oriented(mesh):
    dim = mesh.coords.shape[1]
    assert mesh.face_centers.shape == (len(mesh.faces), dim)
    assert mesh.bface_centers.shape == (len(mesh.bfaces), dim)
    assert np.allclose(np.linalg.norm(mesh.face_normals, axis=1), 1)
    assert np.allclose(np.linalg.norm(mesh.bface_normals, axis=1), 1)
    for (i, j, measure, distance), center, normal in zip(
            mesh.faces, mesh.face_centers, mesh.face_normals):
        assert measure > 0 and distance > 0
        assert (center - mesh.centroid[i]) @ normal > 0
        assert (mesh.centroid[j] - center) @ normal > 0
    for (i, measure, distance), center, normal in zip(
            mesh.bfaces, mesh.bface_centers, mesh.bface_normals):
        assert measure > 0 and distance > 0
        assert (center - mesh.centroid[i]) @ normal > 0
    # Constant vector fields have zero integrated divergence cell by cell.
    closure = np.zeros_like(mesh.centroid)
    for (i, j, measure, _), normal in zip(mesh.faces, mesh.face_normals):
        closure[i] += measure * normal
        closure[j] -= measure * normal
    for (i, measure, _), normal in zip(mesh.bfaces, mesh.bface_normals):
        closure[i] += measure * normal
    assert np.allclose(closure, 0, atol=1e-14)


def test_legacy_positional_mesh_metadata_is_optional():
    mesh = Mesh(np.zeros((3, 2)), [(0, 1, 2)], np.array([1]),
                np.zeros((1, 2)), np.ones(1), [], [])
    assert mesh.face_distances is None
    assert mesh.face_centers is None
    assert mesh.bface_tags is None


@pytest.mark.parametrize('cell,expected', [
    ((0, 1, 2), 1.e-10),
    ((2, 1, 0), 1.e-10),
    ((0, 1, 3, 2), 2.e-10),
])
def test_thin_cell_area_is_preserved_away_from_origin(cell, expected):
    # The global shoelace formula subtracts O(100) products to obtain O(1e-10).
    # Analytic triangle/rectangle areas provide an independent small-cell check.
    coords = np.array([(0., 0.), (1.e-5, 0.), (0., 2.e-5), (1.e-5, 2.e-5)])
    for offset in ((0., 0.), (10., -10.), (-10., 10.)):
        mesh = assemble_mesh(coords + offset, [cell], [7])
        assert mesh.area[0] == pytest.approx(expected, rel=1.e-9, abs=0.)


def test_hanging_node_faces_use_half_edge_geometry():
    coords = [(0, 0), (1, 0), (1, 1), (0, 1), (1, .5),
              (2, 0), (2, .5), (2, 1)]
    mesh = assemble_mesh(coords, [(0, 1, 2, 3), (1, 5, 6, 4), (4, 6, 7, 2)],
                         [10, 20, 20], boundary_tags={(0, 3): 42})
    _assert_oriented(mesh)
    split = [k for k, f in enumerate(mesh.faces) if f[0] == 0]
    assert len(split) == 2
    assert sorted(mesh.face_centers[split, 1]) == [.25, .75]
    assert np.allclose(mesh.face_centers[split, 0], 1)
    assert np.allclose(mesh.face_normals[split], [[1, 0], [1, 0]])
    assert np.allclose([mesh.faces[k][2] for k in split], .5)
    assert np.allclose(mesh.face_distances[split], .5)
    assert sorted(mesh.bface_tags) == [0] * 6 + [42]


def _write_gmsh(path, nodes, elements):
    path.write_text('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n'
                    + str(len(nodes)) + '\n' + '\n'.join(nodes)
                    + '\n$EndNodes\n$Elements\n' + str(len(elements)) + '\n'
                    + '\n'.join(elements) + '\n$EndElements\n')


def test_gmsh_2d_sparse_node_ids_variable_tags_and_boundary_labels(tmp_path):
    path = tmp_path / 'quad.msh'
    _write_gmsh(path, ['40 0 0 0', '8 1 0 0', '99 1 1 0', '3 0 1 0'],
                ['1 3 4 17 99 200 300 40 8 99 3',
                 '2 1 1 31 40 8', '3 1 3 32 8 9 8 99',
                 '4 1 0 99 3', '5 1 2 34 100 3 40'])
    mesh = read_gmsh(path)
    _assert_oriented(mesh)
    assert mesh.coords.shape == (4, 2)
    assert mesh.cells == [(0, 1, 2, 3)]
    assert mesh.cell_tag.tolist() == [17]
    labels = {tuple(center): tag for center, tag in zip(mesh.bface_centers, mesh.bface_tags)}
    assert labels == {(.5, 0): 31, (1, .5): 32, (.5, 1): 0, (0, .5): 34}
    assert mesh.face_distances.shape == (0, 2)


def test_gmsh_untagged_material_defaults_to_zero(tmp_path):
    path = tmp_path / 'triangle.msh'
    _write_gmsh(path, ['10 0 0 0', '20 1 0 0', '30 0 1 0'], ['1 2 0 10 20 30'])
    assert read_gmsh(path).cell_tag.tolist() == [0]


def test_gmsh_3d_surface_tags_and_normal_distance_opt_in(tmp_path):
    path = tmp_path / 'tet.msh'
    _write_gmsh(path, ['20 0 0 0', '4 2 0 0', '80 0 1 0', '7 0 0 3'],
                ['1 4 1 51 20 4 80 7', '2 2 3 61 71 81 20 4 80',
                 '3 2 0 20 7 4', '4 2 2 62 72 20 80 7',
                 '5 2 1 63 4 7 80'])
    legacy = read_gmsh(path)
    normal = read_gmsh(path, normal_distances=True)
    _assert_oriented(normal)
    assert legacy.face_distances is None
    assert normal.face_distances.shape == (0, 2)
    assert normal.cell_tag.tolist() == [51]
    assert sorted(normal.bface_tags) == [0, 61, 62, 63]
    for (i, _, distance), center, direction in zip(normal.bfaces, normal.bface_centers,
                                                   normal.bface_normals):
        assert distance == pytest.approx((center - normal.centroid[i]) @ direction)
    assert not np.allclose([f[2] for f in legacy.bfaces], [f[2] for f in normal.bfaces])


def test_two_tets_have_opposing_shared_face_geometry():
    mesh = assemble_mesh_3d([(0, 0, 0), (1, 0, 0), (0, 1, 0),
                             (0, 0, 1), (1, 1, 1)],
                            [(0, 1, 2, 3), (1, 2, 3, 4)], [7, 8], normal_distances=True)
    _assert_oriented(mesh)
    (i, j, _, _), = mesh.faces
    center, = mesh.face_centers
    normal, = mesh.face_normals
    assert mesh.face_distances[0] == pytest.approx([
        (center - mesh.centroid[i]) @ normal, (mesh.centroid[j] - center) @ normal])


@pytest.mark.parametrize('coords,cells', [
    ([(0, 0), (1, 0), (2, 0)], [(0, 1, 2)]),
    ([(0, 0), (1, 0), (1, 1), (0, 1)], [(0, 1, 2, 3)] * 3),
])
def test_degenerate_and_nonmanifold_2d_geometry_is_rejected(coords, cells):
    with pytest.raises(ValueError):
        assemble_mesh(coords, cells, [1] * len(cells))


def test_degenerate_3d_geometry_is_rejected():
    with pytest.raises(ValueError):
        assemble_mesh_3d([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
                         [(0, 1, 2, 3)], [1])
