"""Primary boundary interpolation, checked independently of production weights."""
import itertools

import numpy as np
import pytest

from ndgpu.nem import NEMDiffusionSolver
from ndgpu.nem_geometry import CuboidNodalGrid
import test_nem as augmented

PAIRS = ((0, 2), (1, 3), (4, 5))


def chain(edges, axis=0):
    bounds = np.tile([[0., 1.], [0., 1.], [0., 1.]], (len(edges)-1, 1, 1))
    bounds[:, axis] = np.column_stack((edges[:-1], edges[1:]))
    return bounds


def grid_with_sides(bounds, axis=0, ends=('vacuum', 'vacuum')):
    geometry = CuboidNodalGrid(bounds)
    conditions = ['reflective']*len(geometry.boundary_faces)
    for k, side in enumerate(geometry.boundary_sides):
        if side in PAIRS[axis]:
            conditions[k] = ends[PAIRS[axis].index(side)]
    return CuboidNodalGrid(bounds, conditions, leakage_boundary='primary_rule')


def independent_weights(grid, node, axis):
    """Gauss integrals in a Legendre basis, with independent constraint mapping."""
    q, w = np.polynomial.legendre.leggauss(8)
    center = grid.bounds[node, axis].mean()
    width = grid.widths[node, axis]
    basis = lambda t: np.column_stack((np.ones_like(t), t, (3*t*t-1)/2))
    rows, sources = [], []
    for side, sample in [(None, node)] + [(s, int(grid.neighbors[node, s])) for s in PAIRS[axis]]:
        if sample >= 0:
            lo, hi = (grid.bounds[sample, axis]-center)/width
        else:
            k = np.flatnonzero((grid.boundary_nodes == node) & (grid.boundary_sides == side))[0]
            sign = -1 if side == PAIRS[axis][0] else 1
            if grid.boundary_kinds[k] == 'vacuum':
                rows.append(basis(np.array([sign/2]))[0])
                sources.append(None)
                continue
            lo, hi = sign-.5, sign+.5
            sample = node
        rows.append((w/2) @ basis((lo+hi)/2+(hi-lo)*q/2))
        sources.append(sample)
    t = q/2
    targets = np.array([(w*t/2) @ basis(t), (w*(3*t*t-.25)/2) @ basis(t)])
    targets[:, 0] = 0.
    raw = targets @ np.linalg.inv(rows)
    samples = list(dict.fromkeys(s for s in sources if s is not None))
    weights = np.column_stack([raw[:, [k for k, s in enumerate(sources) if s == sample]].sum(axis=1)
                               for sample in samples])
    return np.array(samples), weights


@pytest.mark.parametrize('ends, expected', [
    (('vacuum', 'vacuum'), [0., -1/10]),
    (('reflective', 'reflective'), [0., 0.]),
    (('vacuum', 'reflective'), [1/8, -1/40]),
    (('reflective', 'vacuum'), [-1/8, -1/40]),
])
@pytest.mark.parametrize('axis', range(3))
def test_single_node_exact_boundary_constraints(ends, expected, axis):
    grid = grid_with_sides(chain([-2., 3.], axis), axis, ends)
    ids, weights = grid.leakage_weights(0, axis)
    np.testing.assert_array_equal(ids, [0])
    np.testing.assert_allclose(weights[:, 0], expected, atol=2e-17)
    assert grid.leakage_degrees[0, axis] == 2
    assert grid.leakage_boundary == 'primary_rule'
    np.testing.assert_array_equal(grid.boundary_alpha, np.where(grid.boundary_kinds == 'vacuum', .5, 0))
    assert not grid.boundary_kinds.flags.writeable


@pytest.mark.parametrize('boundary, expected', [
    ('vacuum', [[5/48, 1/48], [-3/80, 1/80]]),
    ('reflective', [[-1/24, 1/24], [-1/120, 1/120]]),
])
def test_equal_width_boundary_rational_weights(boundary, expected):
    grid = CuboidNodalGrid(chain([0., 1., 2., 3.]), boundary, leakage_boundary='primary_rule')
    ids, weights = grid.leakage_weights(0, 0)
    np.testing.assert_array_equal(ids, [0, 1])
    np.testing.assert_allclose(weights, expected, atol=2e-17)


@pytest.mark.parametrize('ends', list(itertools.product(('vacuum', 'reflective'), repeat=2)))
@pytest.mark.parametrize('axis', range(3))
def test_unequal_width_integrals_and_axis_reversal(ends, axis):
    bounds = chain([-3., -.8, .3, 4.7], axis)
    grid = grid_with_sides(bounds, axis, ends)
    reversed_bounds = bounds[::-1].copy()
    reversed_bounds[:, axis] = -reversed_bounds[:, axis, ::-1]
    reverse = grid_with_sides(reversed_bounds, axis, ends[::-1])
    for node in range(3):
        ids, weights = grid.leakage_weights(node, axis)
        expected_ids, expected = independent_weights(grid, node, axis)
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_allclose(weights, expected, rtol=2e-13, atol=1e-16)
        rid, rw = reverse.leakage_weights(2-node, axis)
        lookup = {2-int(i): k for k, i in enumerate(rid)}
        np.testing.assert_allclose(rw[:, [lookup[int(i)] for i in ids]],
                                   np.array([[-1.], [1.]])*weights, atol=1e-16)
        assert np.all(ids >= 0)
        assert not ids.flags.writeable and not weights.flags.writeable
    old = CuboidNodalGrid(bounds)
    np.testing.assert_array_equal(grid.leakage_weights(1, axis)[0], old.leakage_weights(1, axis)[0])
    np.testing.assert_allclose(grid.leakage_weights(1, axis)[1], old.leakage_weights(1, axis)[1], atol=1e-16)


def test_default_unchanged_and_no_bridges_across_void():
    bounds = chain([0., 1., 2., 3.])
    default = CuboidNodalGrid(bounds)
    explicit = CuboidNodalGrid(bounds, leakage_boundary='one_sided')
    for node in range(3):
        for axis in range(3):
            for a, b in zip(default.leakage_weights(node, axis), explicit.leakage_weights(node, axis)):
                np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(default.leakage_weights(0, 0)[0], [0, 1, 2])
    bounds[2, 0] += 1
    primary = CuboidNodalGrid(bounds, leakage_boundary='primary_rule')
    np.testing.assert_array_equal(primary.leakage_weights(1, 0)[0], [1, 0])
    np.testing.assert_array_equal(primary.leakage_weights(2, 0)[0], [2])
    np.testing.assert_allclose(primary.leakage_weights(2, 0)[1][:, 0], [0, -.1], atol=1e-16)
    for name in ('bounds', 'widths', 'volumes', 'boundary_alpha', 'face_nodes'):
        np.testing.assert_array_equal(getattr(primary, name), getattr(CuboidNodalGrid(bounds), name))


@pytest.mark.parametrize('policy', ['primary', 'vacuum', 'PRIMARY_RULE', '', None, 1, [], np.array(['primary_rule'])])
def test_invalid_policy_rejected(policy):
    with pytest.raises(ValueError, match='leakage_boundary'):
        CuboidNodalGrid(chain([0., 1.]), leakage_boundary=policy)


@pytest.mark.parametrize('attribute', ['leakage_boundary', 'boundary_kinds'])
def test_cached_solver_rejects_boundary_policy_metadata_replacement(attribute):
    grid = CuboidNodalGrid(chain([0., 1.]), leakage_boundary='primary_rule')
    solver = NEMDiffusionSolver(grid, *augmented.materials(grid, 2))
    if attribute == 'leakage_boundary':
        grid.leakage_boundary = 'one_sided'
    else:
        grid.boundary_kinds = np.array(['reflective']*len(grid.boundary_faces))
    with pytest.raises(ValueError, match='changed after cached'):
        solver.solve_fixed_source([1., 1.])


def independent_pencil(monkeypatch, grid, material):
    # Reuse only the old uncondensed balance/response assembly. Its leakage
    # implementation is replaced with this file's independent Gauss closure.
    monkeypatch.setattr(augmented, 'independent_weights', independent_weights)
    return augmented.full_augmented(grid, *material)


def test_fixed_source_full_system_and_conservation(monkeypatch):
    bounds = np.array([[[x, x+1], [y, y+1], [0., 1.]] for x in (0., 1.) for y in (0., 1.)])
    grid = CuboidNodalGrid(bounds, leakage_boundary='primary_rule')
    material = augmented.materials(grid, 2)
    solver = NEMDiffusionSolver(grid, *material)
    A, _ = independent_pencil(monkeypatch, grid, material)
    rng = np.random.default_rng(814)
    q0 = .4+rng.random((grid.n_nodes, 2))
    qm = rng.normal(size=(grid.n_nodes, 3, 2, 2))*.003
    drive = rng.normal(size=(len(grid.boundary_faces), 2))*.002
    rhs = augmented.physical_rhs(grid, q0, qm, drive)
    expected = np.linalg.solve(A, rhs)
    result = solver.solve_fixed_source(q0, qm, drive)
    state = augmented.full_state(result)
    np.testing.assert_allclose(state, expected, rtol=2e-10, atol=2e-12)
    residual = A@state-rhs
    scale = abs(A)@abs(state)+abs(rhs)
    assert np.max(abs(residual)/np.maximum(scale, np.finfo(float).tiny)) < 1e-11
    loss = np.einsum('ngh,nh->ng', solver.R, result.means)
    boundary_out = (grid.normal_signs[grid.boundary_sides, None]*result.currents[grid.boundary_faces]
                    *grid.face_areas[grid.boundary_faces, None]).sum(axis=0)
    np.testing.assert_allclose((grid.volumes[:, None]*(q0-loss)).sum(axis=0), boundary_out, rtol=1e-11)
    for key in solver._residual_keys:
        assert result.diagnostics[key] < 1e-10


@pytest.mark.parametrize('boundary', ['vacuum', 'reflective'])
def test_dense_fission_pencil_and_positive_eigenpair(monkeypatch, boundary):
    grid = CuboidNodalGrid(chain([0., .7, 1.8]), boundary, leakage_boundary='primary_rule')
    material = augmented.materials(grid, 2)
    material[3][1] *= 1.4
    solver = NEMDiffusionSolver(grid, *material)
    A, F = independent_pencil(monkeypatch, grid, material)
    values, vectors = np.linalg.eig(np.linalg.solve(A, F))
    dominant = np.argmax(abs(values))
    assert abs(values[dominant].imag) < 1e-12
    result = solver.solve_eigenvalue(max_iterations=2000)
    assert result.k_eff == pytest.approx(values[dominant].real, rel=2e-9)
    expected = vectors[:, dominant].real
    expected /= np.sum(grid.volumes[:, None]*material[3]*expected[:grid.n_nodes*2].reshape(-1, 2))
    state = augmented.full_state(result)
    np.testing.assert_allclose(state, expected, rtol=2e-7, atol=1e-10)
    residual = A@state-F@state/result.k_eff
    scale = abs(A)@abs(state)+abs(F)@abs(state)/result.k_eff
    assert np.max(abs(residual)/np.maximum(scale, np.finfo(float).tiny)) < 1e-9
    assert result.diagnostics['full_eigen_residual'] < 1e-9
    for name in ('minimum_mean', 'minimum_side_face_mean', 'minimum_profile_flux'):
        assert result.diagnostics[name] > 0


def test_analytic_manufactured_state_with_primary_moment_forcing(monkeypatch):
    bounds = np.array([[[x, x+h], [y, y+k], [0., 1.3]]
                       for x, h in ((0., .7), (.7, 1.1))
                       for y, k in ((0., .4), (.4, .8))])
    grid = CuboidNodalGrid(bounds, leakage_boundary='primary_rule')
    N, G = grid.n_nodes, 2
    material = list(augmented.materials(grid, G))
    material[0][:] = [1., .4]  # continuous analytic current across interfaces
    solver = NEMDiffusionSolver(grid, *material)
    A, _ = independent_pencil(monkeypatch, grid, material)
    c = grid.bounds.mean(axis=2)
    linear = np.array([[.12, -.07], [.08, .11], [-.02, .03]])
    quadratic = np.array([[.02, .03], [-.01, .01], [.015, -.01]])
    means = np.broadcast_to([2., 1.5], (N, G)).copy()
    means += c @ linear + (c*c+grid.widths**2/12) @ quadratic
    moments = np.empty((N, 3, 2, G))
    sides = np.empty((N, 6, G))
    for a, (sm, sp) in enumerate(PAIRS):
        moments[:, a, 0] = grid.widths[:, a, None]*(linear[a]+2*c[:, a, None]*quadratic[a])/12
        moments[:, a, 1] = grid.widths[:, a, None]**2*quadratic[a]/60
        for end, side in enumerate((sm, sp)):
            x = grid.bounds[:, a, end, None]
            sides[:, side] = (means + (x-c[:, a, None])*linear[a]
                              + (x*x-c[:, a, None]**2-grid.widths[:, a, None]**2/12)*quadratic[a])
    currents = np.array([-material[0][0]*(linear[a]+2*plane*quadratic[a])
                         for a, plane in zip(grid.face_axes, grid.face_planes)])
    expected = np.r_[means.ravel(), moments.ravel(), currents.ravel(), sides.ravel()]
    rhs = A @ expected
    ng = N*G
    # Analytic polynomial satisfies response and continuity rows. The moment
    # forcing includes the independently computed primary boundary closure.
    np.testing.assert_allclose(rhs[7*ng:13*ng], 0, atol=3e-14)
    face_rhs = rhs[13*ng:].reshape(-1, G)
    internal = np.all(grid.face_nodes >= 0, axis=1)
    np.testing.assert_allclose(face_rhs[internal], 0, atol=1e-14)
    q0 = rhs[:ng].reshape(N, G)
    qm = rhs[ng:7*ng].reshape(N, 3, 2, G)
    result = solver.solve_fixed_source(q0, qm, face_rhs[grid.boundary_faces])
    np.testing.assert_allclose(augmented.full_state(result), expected, rtol=2e-10, atol=1e-12)
    for key in solver._residual_keys:
        assert result.diagnostics[key] < 1e-10
    for name in ('minimum_mean', 'minimum_side_face_mean', 'minimum_profile_flux'):
        assert result.diagnostics[name] > 0
