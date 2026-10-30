"""Strict opt-in polygon geometry; no solver runs or geometry generation."""

import numpy as np
import pytest

from garnet.mesh import MeshGrid, MeshGroupOperator
from garnet.polygon_mesh import (
    assemble_polygon_mesh, clip_star_polygon, polygon_mesh_audit,
)


def _boundary(cells, tag=7):
    owners = {}
    for cell in cells:
        for a, b in zip(cell, cell[1:] + cell[:1]):
            key = tuple(sorted((a, b)))
            owners[key] = owners.get(key, 0) + 1
    return {key: tag for key, count in owners.items() if count == 1}


def _mesh(coords, cells, **kwargs):
    return assemble_polygon_mesh(coords, cells, [1] * len(cells),
                                 boundary_tags=kwargs.pop("boundary_tags", _boundary(cells)), **kwargs)


def test_skew_quad_true_centroid_and_translation():
    points = np.array([[0., 0.], [4., 0.], [2., 2.], [0., 2.]])
    for shift in (np.zeros(2), np.array([1.e8, -2.e8])):
        mesh = _mesh(points + shift, [(0, 1, 2, 3)])
        np.testing.assert_allclose(mesh.centroid[0] - shift, [14 / 9, 8 / 9], atol=2.e-8)
        assert mesh.area[0] == 6
        assert not np.allclose(mesh.centroid[0] - shift, points.mean(axis=0))
        audit = polygon_mesh_audit(mesh)
        assert audit.total_area == 6 and audit.n_boundary_faces == 4
        np.testing.assert_allclose(audit.first_moment, mesh.centroid[0] * 6)
        assert audit.minimum_normal_distance > 0


def test_two_cell_geometry_and_conservative_flux():
    points = [[0, 0], [2, 0], [1.5, 2], [0, 2], [4, 0], [4, 2]]
    mesh = _mesh(points, [(0, 1, 2, 3), (1, 4, 5, 2)])
    assert len(mesh.faces) == 1
    i, j, length, _ = mesh.faces[0]
    normal, center = mesh.face_normals[0], mesh.face_centers[0]
    np.testing.assert_allclose(mesh.face_distances[0],
        [(center - mesh.centroid[i]) @ normal, (mesh.centroid[j] - center) @ normal])
    assert np.all(mesh.face_distances > 0)
    grid = MeshGrid(mesh, "reflective")
    diffusion = np.array([2., 5.]).reshape(grid.shape)
    op = MeshGroupOperator(np, grid, diffusion, np.zeros(grid.shape))
    phi = np.array([1., 3.]).reshape(grid.shape)
    actual = op.apply(phi).ravel()
    weight = length / np.sum(mesh.face_distances[0] / diffusion.ravel()[[i, j]])
    np.testing.assert_allclose(actual[[i, j]], [-2 * weight, 2 * weight])
    assert abs(actual.sum()) < 1.e-14
    assert set(mesh.bface_tags) == {7}


def test_boundary_tags_are_exact_and_exhaustive():
    points = [[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1]]
    cells = [(0, 1, 2, 3), (1, 4, 5, 2)]
    boundary = _boundary(cells)
    for labels in ({}, {**boundary, (1, 2): 7}, {**boundary, (0, 4): 7}):
        with pytest.raises(ValueError, match="boundary tags"):
            _mesh(points, cells, boundary_tags=labels)
    boundary[next(iter(boundary))] = 0
    assert 0 in _mesh(points, cells, boundary_tags=boundary).bface_tags


def test_hanging_interface_is_not_repaired_or_inferred_as_boundary():
    points = [[0, 0], [1, 0], [1, 2], [0, 2], [2, 0], [2, 1], [1, 1], [2, 2]]
    cells = [(0, 1, 2, 3), (1, 4, 5, 6), (6, 5, 7, 2)]
    labels = {(0, 1): 1, (0, 3): 1, (2, 3): 1, (1, 4): 1,
              (4, 5): 1, (5, 7): 1, (2, 7): 1}
    with pytest.raises(ValueError, match="exactly all boundary"):
        _mesh(points, cells, boundary_tags=labels)


@pytest.mark.parametrize("points,cells,match", [
    ([[0, 0], [1, 0], [0, 1]], [(0, 2, 1)], "CCW"),
    ([[0, 0], [1, 0], [0, 1]], [(0, 1, 3)], "index"),
    ([[0, 0], [1, 0], [0, 1]], [(0, 1, -1)], "index"),
    ([[0, 0], [1, 0], [0, 1], [0, 0]], [(0, 1, 2)], "duplicate coordinates"),
    ([[0, 0], [1, 0], [0, 1]], [(0, 1, 1, 2)], "fan|repeated|edge"),
    ([[0, 0], [1, 0], [2, 0]], [(0, 1, 2)], "area"),
    ([[0, 0], [1, 0], [0, 1]], [(0, 1, 2)] * 3, "nonmanifold"),
    ([[0, 0], [1, 0], [0, 1]], [(0, 1, 2)] * 2, "opposite orientations"),
])
def test_invalid_meshes(points, cells, match):
    with pytest.raises(ValueError, match=match):
        _mesh(points, cells)


def test_nonstar_and_self_intersecting_polygons_rejected():
    c_shape = [[0, 0], [3, 0], [3, 1], [1, 1], [1, 2], [3, 2], [3, 3], [0, 3]]
    with pytest.raises(ValueError, match="centroid fan"):
        clip_star_polygon(c_shape)
    angles = np.arange(5) * 2 * np.pi / 5
    pentagram = np.column_stack((np.cos(angles), np.sin(angles)))[[0, 2, 4, 1, 3]]
    with pytest.raises(ValueError, match="winding"):
        clip_star_polygon(pentagram)


def test_clip_keeps_whole_polygon_and_exact_moments():
    vertices = np.array([[1., 0.], [3., 0.], [2., 1.], [1., 1.]])
    pieces = clip_star_polygon(vertices)
    assert len(pieces) == 1
    np.testing.assert_array_equal(pieces[0].vertices, vertices)
    assert pieces[0].area == 1.5
    np.testing.assert_allclose(pieces[0].first_moment, [8 / 3, 2 / 3])


def test_concave_star_disconnected_intersection_keeps_all_fan_parts():
    s = np.sqrt(8.)
    vertices = np.array([[.5, 0], [s, s], [0, 1], [-s, s],
                         [-.5, 0], [-s, -s], [0, -1], [s, -s]])
    pieces = clip_star_polygon(vertices, halfplanes=[(1, 0, 1)])
    assert len(pieces) == 4
    assert all(piece.area > 0 for piece in pieces)
    assert all(np.all(piece.vertices[:, 0] >= 1) for piece in pieces)
    # Each disconnected component is the triangle cut from one lobe.
    a = np.array([1., s * .5 / (s - .5)])
    b = np.array([1., 1 + (s - 1) / s])
    expected = abs(np.linalg.det(np.array([np.array([s, s]) - a, b - a])))
    assert float(sum(piece.area for piece in pieces)) == pytest.approx(expected)
    assert float(sum(piece.first_moment[1] for piece in pieces)) == pytest.approx(0, abs=1.e-14)


def test_default_wedge_clip_and_tiny_valid_pieces():
    vertices = np.array([[-1., -1.], [2., -1.], [2., 2.], [-1., 2.]])
    pieces = clip_star_polygon(vertices)
    assert float(sum(p.area for p in pieces)) == pytest.approx(2.)
    np.testing.assert_allclose(sum((p.first_moment for p in pieces)), [8 / 3, 4 / 3])
    tiny = 1.e-12 * np.array([[0., 0.], [1., 0.], [1., 1.]])
    mesh = _mesh(tiny, [(0, 1, 2)])
    assert mesh.area[0] == pytest.approx(5.e-25, rel=1.e-14, abs=0)
    clipped = clip_star_polygon(tiny, halfplanes=[(1, 0, 5.e-13)])
    assert all(p.area > 0 for p in clipped)
    assert float(sum(p.area for p in clipped)) == pytest.approx(3.75e-25, rel=1.e-14, abs=0)


def test_clip_discards_only_zero_area_contacts():
    square = [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert clip_star_polygon(square, halfplanes=[(1, 0, 2)]) == []
    assert clip_star_polygon(square, halfplanes=[(1, 0, 1)]) == []


def test_near_plane_classification_requires_explicit_caller_resolution():
    vertices = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    offset = np.longdouble(2) + 4 * np.finfo(np.longdouble).eps
    with pytest.raises(ValueError, match="ambiguous"):
        clip_star_polygon(vertices, halfplanes=np.array([[1, 1, offset]], dtype=np.longdouble))
    # Example caller-owned normalization; the tolerance is deliberately absent
    # from the generic API. Audit the displacement before calling the helper.
    near = np.array([[0., -2.e-12], [1., 3.e-12], [1., 1.]])
    projected = near.copy()
    projected[np.abs(projected[:, 1]) <= 1.e-10, 1] = 0
    assert np.max(np.linalg.norm(projected - near, axis=1)) <= 3.e-12
    pieces = clip_star_polygon(projected)
    assert len(pieces) == 1 and pieces[0].area == .5
    np.testing.assert_array_equal(near[:, 1], [-2.e-12, 3.e-12, 1.])


def test_real_template_near_diagonal_fan_slivers_are_retained():
    # Cell 10861 from native-assembly-construction-02/mesh.npz; self-contained
    # fixture so this test does not depend on a scratch archive being present.
    vertices = np.array([
        [9.990310967659424, 10.05079667671087],
        [10.05128722528263, 9.986147443407138],
        [10.13800332323707, 9.990310967601928],
        [10.20265255638457, 10.05128722505274],
        [10.19848903234603, 10.13800332317959],
        [10.13751277473898, 10.20265255615466],
        [10.05079667676835, 10.19848903228853],
        [9.986147443637035, 10.13751277450907],
    ])
    pieces = clip_star_polygon(vertices)
    assert len(pieces) == 8
    assert sum(0 < piece.area < 1.e-18 for piece in pieces) == 3
    # Reference moments from direct whole-convex-polygon intersection with
    # exact Fraction arithmetic on the binary input values (no centroid fan).
    assert float(sum(piece.area for piece in pieces)) == pytest.approx(0.018595781823080435, rel=2.e-14)
    np.testing.assert_allclose(sum((piece.first_moment for piece in pieces)),
                               [0.18831658776278085, 0.18709851505099276], rtol=2.e-14)
    center = _mesh(vertices, [tuple(range(len(vertices)))]).centroid[0]
    projected_center = np.repeat(center.mean(), 2)
    displacement = np.linalg.norm(projected_center - center)
    assert 1.e-10 < displacement < 1.1e-10
    projected_pieces = clip_star_polygon(vertices, fan_center=projected_center)
    assert len(projected_pieces) == 5
    assert min(p.area for p in projected_pieces) > .002
    np.testing.assert_allclose(sum(p.area for p in projected_pieces), sum(p.area for p in pieces), rtol=2.e-14)
    np.testing.assert_allclose(sum((p.first_moment for p in projected_pieces)),
                               sum((p.first_moment for p in pieces)), rtol=2.e-14)


@pytest.mark.parametrize("center", [[0, 0], [2, 2], [np.nan, 0], [0]])
def test_fan_center_requires_strict_interior_kernel_point(center):
    with pytest.raises(ValueError, match="fan_center|centroid fan"):
        clip_star_polygon([[0, 0], [1, 0], [1, 1], [0, 1]], fan_center=center)


def test_invalid_values_and_noninteger_ids():
    for coords in ([[0, 0], [1, 0], [0, np.nan]], [[0, 0, 0]] * 3):
        with pytest.raises(ValueError, match="coords"):
            _mesh(coords, [(0, 1, 2)])
    with pytest.raises(ValueError, match="integers"):
        _mesh([[0, 0], [1, 0], [0, 1]], [(0., 1., 2.)])
    with pytest.raises(ValueError, match="halfplanes"):
        clip_star_polygon([[0, 0], [1, 0], [0, 1]], halfplanes=[(0, 0, 0)])
