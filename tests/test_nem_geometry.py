"""Geometry invariants and independent integral checks of leakage closure."""
import numpy as np
import pytest

from ndgpu.nem_geometry import CuboidNodalGrid


def chain(edges, axis=0):
    b = np.tile([[0., 2.], [-1., 2.], [0., 4.]], (len(edges)-1, 1, 1))
    b[:, axis, :] = np.column_stack((edges[:-1], edges[1:]))
    return b


def polynomial_means(bounds, axis, center, width, coefficients):
    # Gauss integration of the actual polynomial, independent of fit matrix.
    q, w = np.polynomial.legendre.leggauss(5)
    lo, hi = bounds[:, axis, :].T
    x = (lo[:, None]+hi[:, None])/2 + (hi-lo)[:, None]*q/2
    t = (x-center)/width
    return np.polynomial.polynomial.polyval(t, coefficients) @ (w/2)


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('node', [0, 1, 2, 3])
def test_unequal_width_quadratic_moments_both_and_one_sided(axis, node):
    bounds = chain([-3., -1.5, .75, 4.25, 5.], axis)
    grid = CuboidNodalGrid(bounds)
    samples, weights = grid.leakage_weights(node, axis)
    width = grid.widths[node, axis]
    center = bounds[node, axis, 0]+width/2
    coeff = [1.7, -.38, .91]
    means = polynomial_means(bounds[samples], axis, center, width, coeff)
    np.testing.assert_allclose(weights@means, [coeff[1]/12, coeff[2]/60],
                               rtol=2e-13, atol=2e-15)
    assert samples[0] == node
    assert len(samples) == 3
    assert grid.leakage_degrees[node, axis] == 2
    assert grid.leakage_condition[node, axis] <= 1e12
    if node == 0:
        np.testing.assert_array_equal(samples, [0, 1, 2])
    elif node == 3:
        np.testing.assert_array_equal(samples, [3, 2, 1])
    else:
        np.testing.assert_array_equal(samples, [node, node-1, node+1])


@pytest.mark.parametrize('count', [1, 2])
def test_short_chains_linear_and_constant(count):
    bounds = chain([0., 1.2, 4.][:count+1])
    grid = CuboidNodalGrid(bounds)
    for node in range(count):
        samples, weights = grid.leakage_weights(node, 0)
        means = polynomial_means(bounds[samples], 0,
                                 bounds[node, 0].mean(), grid.widths[node, 0], [2., .6])
        np.testing.assert_allclose(weights@means, [.6/12 if count == 2 else 0., 0.], atol=1e-15)
        assert grid.leakage_degrees[node, 0] == count-1
        for axis in (1, 2):
            sample, weight = grid.leakage_weights(node, axis)
            np.testing.assert_array_equal(sample, [node])
            np.testing.assert_array_equal(weight, [[0.], [0.]])
            assert grid.leakage_degrees[node, axis] == 0
            assert grid.leakage_condition[node, axis] == 1.


def test_whole_face_orientation_counts_volumes_and_conservation():
    bounds = np.array([[[x, x+2], [y, y+3], [z, z+4]]
                       for x in (0, 2) for y in (0, 3) for z in (0, 4)], float)
    grid = CuboidNodalGrid(bounds)
    assert grid.n_nodes == 8
    assert len(grid.face_nodes) == 36
    assert len(grid.boundary_faces) == 24
    np.testing.assert_array_equal(grid.volumes, np.full(8, 24.))
    np.testing.assert_array_equal(grid.node_face_areas, np.tile([12, 8, 12, 8, 6, 6], (8, 1)))
    pairs = ((0, 2), (1, 3), (4, 5))
    incidence = np.zeros((8, 36))
    for face, (low, high) in enumerate(grid.face_nodes):
        axis = grid.face_axes[face]
        for node, end, side in ((low, 1, pairs[axis][1]), (high, 0, pairs[axis][0])):
            if node >= 0:
                assert grid.node_faces[node, side] == face
                assert bounds[node, axis, end] == grid.face_planes[face]
                assert grid.neighbors[node, side] == (high if node == low else low)
                incidence[node, face] = grid.normal_signs[side]*grid.face_areas[face]
    internal = np.all(grid.face_nodes >= 0, axis=1)
    np.testing.assert_array_equal(incidence[:, internal].sum(axis=0), 0.)
    assert np.all(np.count_nonzero(incidence[:, ~internal], axis=0) == 1)


def test_permutation_preserves_face_ids_geometry_and_leakage():
    bounds = chain([-2., 0., 1., 4., 6.])
    original = CuboidNodalGrid(bounds)
    permutation = np.array([2, 0, 3, 1])
    permuted = CuboidNodalGrid(bounds[permutation])
    for key in ('face_axes', 'face_planes', 'face_areas', 'boundary_faces'):
        np.testing.assert_array_equal(getattr(original, key), getattr(permuted, key))
    mapped = permuted.face_nodes.copy()
    mapped[mapped >= 0] = permutation[mapped[mapped >= 0]]
    np.testing.assert_array_equal(mapped, original.face_nodes)
    for new, old in enumerate(permutation):
        nodes, weights = original.leakage_weights(int(old), 0)
        changed_nodes, changed_weights = permuted.leakage_weights(new, 0)
        np.testing.assert_array_equal(permutation[changed_nodes], nodes)
        np.testing.assert_array_equal(changed_weights, weights)


def test_void_gap_does_not_supply_a_sample_or_a_shared_face():
    bounds = chain([0., 1., 2., 3.])
    bounds[2, 0] += 1.
    grid = CuboidNodalGrid(bounds)
    assert len(grid.boundary_faces) == 16
    np.testing.assert_array_equal(grid.leakage_degrees[:, 0], [1, 1, 0])
    assert grid.neighbors[1, 2] == -1
    np.testing.assert_array_equal(grid.leakage_weights(2, 0)[0], [2])


def test_mixed_boundary_conditions_and_input_copy():
    bounds = chain([0., 1., 3.])
    default = CuboidNodalGrid(bounds)
    bc = ['reflective' if i % 2 else 'vacuum' for i in range(len(default.boundary_faces))]
    grid = CuboidNodalGrid(bounds.reshape(-1, 6), bc)
    np.testing.assert_array_equal(grid.boundary_alpha, [.5 if i % 2 == 0 else 0. for i in range(10)])
    for face, node, side in zip(grid.boundary_faces, grid.boundary_nodes, grid.boundary_sides):
        assert grid.node_faces[node, side] == face
        assert grid.neighbors[node, side] == -1
    saved = grid.bounds.copy()
    bounds[:] = 99
    np.testing.assert_array_equal(grid.bounds, saved)
    for value in vars(grid).values():
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable
    for value in grid.leakage_weights(0, 0):
        assert not value.flags.writeable
    np.testing.assert_array_equal(CuboidNodalGrid(saved, 'reflective').boundary_alpha, 0.)


@pytest.mark.parametrize('bad', [[], np.zeros((1, 5)), np.zeros((1, 3, 2)),
                               [[[0, 1], [0, np.inf], [0, 1]]],
                               [[[0, 1], [0, np.nan], [0, 1]]],
                               [[[1, 0], [0, 1], [0, 1]]],
                               np.ones((1, 6), dtype=complex),
                               [[[0., 1e200], [0., 1e200], [0., 1e200]]]])
def test_invalid_bounds(bad):
    with pytest.raises(ValueError):
        CuboidNodalGrid(bad)


@pytest.mark.parametrize('bc', ['periodic', 'dirichlet', ['vacuum'], 0.,
                              ['vacuum']*5+[None]])
def test_invalid_boundary_conditions(bc):
    with pytest.raises(ValueError):
        CuboidNodalGrid(chain([0., 1.]), bc)


@pytest.mark.parametrize('bounds', [
    # Duplicate, positive volume intersection, partial face, and tiled hanging face.
    [[[0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]]],
    [[[0, 2], [0, 1], [0, 1]], [[1, 3], [0, 1], [0, 1]]],
    [[[0, 1], [0, 2], [0, 1]], [[1, 2], [1, 3], [0, 1]]],
    [[[0, 1], [0, 2], [0, 1]], [[1, 2], [0, 1], [0, 1]], [[1, 2], [1, 2], [0, 1]]],
])
def test_overlapping_or_nonconforming_cuboids_rejected(bounds):
    with pytest.raises(ValueError, match='overlap|hanging'):
        CuboidNodalGrid(bounds)


def test_edge_and_corner_contacts_are_boundaries():
    bounds = np.array([[[0, 1], [0, 1], [0, 1]],
                       [[1, 2], [1, 2], [0, 1]],
                       [[2, 3], [2, 3], [1, 2]]])
    grid = CuboidNodalGrid(bounds)
    assert len(grid.boundary_faces) == 18
    np.testing.assert_array_equal(grid.leakage_degrees, 0)


def test_ill_conditioned_reconstruction_is_rejected():
    with pytest.raises(ValueError, match='ill-conditioned'):
        CuboidNodalGrid(chain([0., 1e-8, 1., 2.]))


@pytest.mark.parametrize('node,axis', [(-1, 0), (1, 0), (0, 3), (0, -1), (0., 0), (0, True)])
def test_invalid_leakage_index(node, axis):
    grid = CuboidNodalGrid(chain([0., 1.]))
    with pytest.raises(ValueError):
        grid.leakage_weights(node, axis)
