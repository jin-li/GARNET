"""Closed tiny CPU geometry fixtures; no external data, solve or admission."""
from copy import deepcopy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from ndgpu.mesh import Mesh
from ndgpu.coupled_mesh_adapter import to_orthogonal_mesh
from ndgpu.coupled_moments import (
    BoundaryFace, InteriorFace, OrthogonalMesh, CoupledMomentOperator,
)
from test_coupled_moments import material, oracle


def fixture(dim=2):
    # Two boxes of widths 2 and 4; transverse lengths 3 (and 5).
    transverse = np.array([3., 5.][:dim - 1])
    cross = float(np.prod(transverse))
    centroids = np.column_stack(([1., 4.], np.tile(transverse / 2, (2, 1))))
    mesh = Mesh(np.empty((0, dim)), [(), ()], np.array([99, -8]),
                centroids, np.array([2., 4.]) * cross,
                [(np.int64(0), np.int64(1), cross, 3.)], [], np.array([[1., 2.]]))
    ni = np.eye(dim)[:1]
    normals = []
    expected_boundary = []
    for cell, width in enumerate((2., 4.)):
        directions = [(0, -1 if cell == 0 else 1)]
        directions += [(axis, sign) for axis in range(1, dim) for sign in (-1, 1)]
        lengths = np.r_[width, transverse]
        for axis, sign in directions:
            normal = np.eye(dim)[axis] * sign
            distance = lengths[axis] / 2
            area = np.prod(np.delete(lengths, axis))
            kind = 'vacuum' if len(normals) % 2 else 'reflective'
            mesh.bfaces.append((cell, float(area), float(distance)))
            normals.append(normal)
            center = centroids[cell].copy()
            # Explicit endpoints for manual OrthogonalMesh, independent of adapter.
            center[axis] = (0. if cell == 0 else 6.) if axis == 0 else (0. if sign < 0 else transverse[axis - 1])
            expected_boundary.append(BoundaryFace(cell, float(area), tuple(center),
                                                  tuple(normal), float(distance), kind))
    kinds = [f.kind for f in expected_boundary]
    manual = OrthogonalMesh(mesh.area, centroids,
                            (InteriorFace(0, 1, cross, tuple(np.r_[2., transverse / 2]),
                                          tuple(ni[0]), 1., 2.),), expected_boundary)
    return mesh, ni, np.array(normals), kinds, manual


def convert(mesh, ni, nb, kinds):
    return to_orthogonal_mesh(mesh, interior_normals=ni,
                              boundary_normals=nb, boundary_kinds=kinds)


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('family', ['sp3', 'sdp1'])
def test_explicit_geometry_operator_oracle_and_immutability(dim, family):
    mesh, ni, nb, kinds, manual = fixture(dim)
    before = deepcopy(mesh)
    ni_before, nb_before, kinds_before = ni.copy(), nb.copy(), list(kinds)
    converted = convert(mesh, ni, nb, kinds)
    np.testing.assert_array_equal(converted.volumes, manual.volumes)
    np.testing.assert_array_equal(converted.centroids, manual.centroids)
    assert converted.interior == manual.interior
    assert converted.boundary == manual.boundary
    mats = (material(), material(1))
    actual = CoupledMomentOperator(converted, mats, family=family)
    explicit = CoupledMomentOperator(manual, mats, family=family)
    u = np.arange(8.) / 3 - 1
    np.testing.assert_array_equal(actual.action(u), explicit.action(u))
    np.testing.assert_allclose(actual.action(u), oracle(manual, mats, family) @ u,
                               rtol=3e-14, atol=3e-14)
    np.testing.assert_array_equal(actual.fission_action(u), explicit.fission_action(u))
    for name in ('coords', 'cell_tag', 'centroid', 'area', 'face_distances'):
        np.testing.assert_array_equal(getattr(mesh, name), getattr(before, name))
    assert mesh.cells == before.cells and mesh.faces == before.faces and mesh.bfaces == before.bfaces
    np.testing.assert_array_equal(ni, ni_before)
    np.testing.assert_array_equal(nb, nb_before)
    assert kinds == kinds_before
    with pytest.raises(ValueError):
        converted.volumes[0] = 10
    with pytest.raises(ValueError):
        converted.centroids[0, 0] = 10
    with pytest.raises(FrozenInstanceError):
        converted.interior[0].area = 10
    mesh.area[0] = 10
    nb[0] *= -1
    assert converted.volumes[0] == manual.volumes[0]
    assert converted.boundary == manual.boundary


@pytest.mark.parametrize('dim', [2, 3])
def test_permutation_translation_and_conservation(dim):
    mesh, ni, nb, kinds, _ = fixture(dim)
    original = convert(mesh, ni, nb, kinds)
    # Reverse cell order, boundary order, and interior owner orientation.
    shifted = deepcopy(mesh)
    shifted.centroid = mesh.centroid[::-1] + np.arange(dim) * 7.25 + 11
    shifted.area = mesh.area[::-1].copy()
    shifted.faces = [(0, 1, mesh.faces[0][2], 3.)]
    shifted.face_distances = mesh.face_distances[:, ::-1].copy()
    shifted.bfaces = [(1 - i, a, d) for i, a, d in mesh.bfaces[::-1]]
    other = convert(shifted, -ni, nb[::-1], kinds[::-1])
    op = CoupledMomentOperator(original, (material(), material(1)))
    swapped = CoupledMomentOperator(other, (material(1), material()))
    u = np.arange(8.).reshape(2, 2, 2)
    np.testing.assert_allclose(swapped.action(u[::-1]), op.action(u)[::-1], rtol=2e-14, atol=2e-14)
    reflective = convert(mesh, ni, nb, ['reflective'] * len(nb))
    conservative = CoupledMomentOperator(reflective, (material(), material(1)))
    x = u.reshape(2, 4)
    leakage = conservative.action(u).reshape(2, 4) - np.stack([a @ b for a, b in zip(conservative.reaction, x)])
    np.testing.assert_allclose(leakage.sum(axis=0), 0., atol=3e-14)


@pytest.mark.parametrize('bad', ['missing', 'shape', 'negative', 'nan', 'legacy', 'skew',
                                'normal_shape', 'normal_nan', 'nonunit', 'reversed',
                                'missing_boundary', 'closure', 'volume', 'duplicate',
                                'boundary_duplicate', 'unsupported', 'scalar_kind'])
def test_bad_geometry_refused_without_fallback(bad):
    mesh, ni, nb, kinds, _ = fixture(3)
    if bad == 'missing': mesh.face_distances = None
    elif bad == 'shape': mesh.face_distances = np.array([1., 2.])
    elif bad == 'negative': mesh.face_distances[0, 0] = -1
    elif bad == 'nan': mesh.area[0] = np.nan
    elif bad == 'legacy': mesh.faces[0] = (0, 1, 15., 3.1)
    elif bad == 'skew': mesh.centroid[1, 1] += .1
    elif bad == 'normal_shape': ni = np.ones((1, 2))
    elif bad == 'normal_nan': nb[0, 0] = np.nan
    elif bad == 'nonunit': ni *= 2
    elif bad == 'reversed': ni *= -1
    elif bad == 'missing_boundary':
        mesh.bfaces.pop(); nb = nb[:-1]; kinds.pop()
    elif bad == 'closure': mesh.bfaces[0] = (0, 14., 1.)
    elif bad == 'volume': mesh.area[0] *= 1.01
    elif bad == 'duplicate':
        mesh.faces *= 2; mesh.face_distances = np.tile(mesh.face_distances, (2, 1)); ni = np.tile(ni, (2, 1))
    elif bad == 'boundary_duplicate':
        mesh.bfaces.append(mesh.bfaces[0]); nb = np.vstack((nb, nb[0])); kinds.append(kinds[0])
    elif bad == 'unsupported': kinds[0] = 'zero'
    elif bad == 'scalar_kind': kinds = 'vacuum'
    with pytest.raises(ValueError): convert(mesh, ni, nb, kinds)


@pytest.mark.parametrize('owner', [True, np.bool_(False), .5, 1., -1, 2, np.nan, '0'])
@pytest.mark.parametrize('boundary', [False, True])
def test_owners_never_truncated_or_boolean(owner, boundary):
    mesh, ni, nb, kinds, _ = fixture()
    if boundary:
        mesh.bfaces[0] = (owner, *mesh.bfaces[0][1:])
    else:
        mesh.faces[0] = (owner, *mesh.faces[0][1:])
    with pytest.raises(ValueError, match='exact integer'):
        convert(mesh, ni, nb, kinds)


@pytest.mark.parametrize('value', [0., -1., np.nan, np.inf])
@pytest.mark.parametrize('field', ['volume', 'face_area', 'boundary_area', 'boundary_distance'])
def test_finite_positive_measures(value, field):
    mesh, ni, nb, kinds, _ = fixture()
    if field == 'volume': mesh.area[0] = value
    elif field == 'face_area': mesh.faces[0] = (0, 1, value, 3.)
    elif field == 'boundary_area': mesh.bfaces[0] = (0, value, 1.)
    else: mesh.bfaces[0] = (0, 3., value)
    with pytest.raises(ValueError): convert(mesh, ni, nb, kinds)


def test_single_cell_empty_interiors_and_required_metadata():
    mesh, _, nb, kinds, _ = fixture()
    mesh.cells = [()]
    mesh.centroid = mesh.centroid[:1]
    mesh.area = mesh.area[:1]
    mesh.faces = []
    mesh.face_distances = np.empty((0, 2))
    mesh.bfaces = mesh.bfaces[:3] + [(0, 3., 1.)]
    nb = np.vstack((nb[:3], [1., 0.]))
    kinds = kinds[:3] + ['reflective']
    converted = convert(mesh, np.empty((0, 2)), nb, kinds)
    assert not converted.interior and len(converted.boundary) == 4
    with pytest.raises(TypeError): to_orthogonal_mesh(mesh)
    with pytest.raises(ValueError): convert(mesh, np.empty((0, 2)), nb, kinds[:-1])


def test_geometry_validator_is_called(monkeypatch):
    import ndgpu.coupled_mesh_adapter as adapter
    calls = []
    monkeypatch.setattr(adapter, '_geometry', lambda mesh: calls.append(mesh))
    mesh, ni, nb, kinds, _ = fixture()
    result = convert(mesh, ni, nb, kinds)
    assert calls == [result]
