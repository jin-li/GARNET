"""Analytical homogeneous eigenpairs and independently corrupted returned fields."""
from dataclasses import replace

import numpy as np
import pytest

from garnet.branch_nem_validation import assess_nem_solution
from garnet.nem import NEMDiffusionSolver
from garnet.nem_geometry import CuboidNodalGrid
from garnet.nem_higher_order import NEMHigherOrderDiffusionSolver


def homogeneous(*, degree=4, groups=1, boundary='reflective', base=False):
    bounds = [[0, 10, 0, 10, 0, 10], [10, 20, 0, 10, 0, 10]]
    if boundary == 'vacuum-x':
        preliminary = CuboidNodalGrid(bounds, 'reflective')
        boundary = ['vacuum' if side in (0, 2) else 'reflective'
                    for side in preliminary.boundary_sides]
    grid = CuboidNodalGrid(bounds, boundary)
    if groups == 1:
        D, sa, ss, fission, chi = [1.], [.03], [[.1]], np.array([[.08]]), [1.]
    else:
        D, sa = [1., .5], [.03, .05]
        ss = [[.1, .02], [.01, .2]]
        fission, chi = np.array([[.05, .02], [.01, .06]]), [.7, .3]
    cls = NEMDiffusionSolver if base else NEMHigherOrderDiffusionSolver
    kwargs = {} if base else {'degree': degree}
    solver = cls(grid, D, sa, ss, fission.sum(axis=0), chi, fission_matrix=fission, **kwargs)
    result = solver.solve_eigenvalue(rtol=1e-9)
    return solver, result


@pytest.mark.parametrize('degree,base', [(4, True), (4, False), (8, False), (16, False)])
def test_homogeneous_reflective_analytical_eigenpair(degree, base):
    solver, result = homogeneous(degree=degree, base=base)
    assert result.k_eff == pytest.approx(.08 / .03, rel=1e-10)
    report = assess_nem_solution(solver, result)
    assert report['status'] == 'pass'
    assert report['maximum_balance_error'] < 1e-5
    assert report['fission_production'] == pytest.approx(1.)
    assert report['minimum_profile_flux'] > 0
    # Diagnostic tampering cannot suppress independent recomputation.
    result.diagnostics.clear()
    assert assess_nem_solution(solver, result)['status'] == 'pass'


def test_full_nonrankone_fission_matches_independent_dense_problem():
    solver, result = homogeneous(groups=2, degree=8)
    R = np.array([[.05, -.01], [-.02, .06]])
    F = np.array([[.05, .02], [.01, .06]])
    assert np.linalg.det(F) != 0
    k = max(np.linalg.eigvals(np.linalg.solve(R, F)))
    assert result.k_eff == pytest.approx(k, rel=1e-8)
    assert assess_nem_solution(solver, result)['volume_balance_error'] < 1e-8
    # This eigenpair does not satisfy the rank-one chi*nuSigma_f replacement.
    wrong_source = solver.chi * np.sum(solver.nu_sigma_f * result.means, axis=1)[:, None] / result.k_eff
    assert np.max(abs(np.einsum('ngh,nh->ng', solver.R, result.means) - wrong_source)) > 1e-7


@pytest.mark.parametrize('field', ['means', 'moments', 'currents', 'side_face_means', 'corrected_face_means', 'k_eff'])
def test_corrupt_returned_fields_are_rejected(field):
    solver, result = homogeneous(groups=2)
    if field == 'k_eff':
        changed = result.k_eff * 1.05
    else:
        changed = getattr(result, field).copy()
        changed.flat[0] += np.max(abs(result.means)) * .2
    with pytest.raises(ValueError, match='balance|profile'):
        assess_nem_solution(solver, replace(result, **{field: changed}))


@pytest.mark.parametrize('field', ['means', 'moments', 'currents', 'side_face_means', 'corrected_face_means'])
def test_nonfinite_arrays_fail_closed(field):
    solver, result = homogeneous()
    changed = getattr(result, field).copy(); changed.flat[0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        assess_nem_solution(solver, replace(result, **{field: changed}))


@pytest.mark.parametrize('value', [0, -1, np.nan, np.inf, True, [1e-5], '1e-5', 1j])
def test_tolerance_must_be_positive_finite_scalar(value):
    solver, result = homogeneous()
    with pytest.raises(ValueError, match='balance_tolerance'):
        assess_nem_solution(solver, result, balance_tolerance=value)


@pytest.mark.parametrize('value', [None, 0, -1, np.nan, np.inf, True])
def test_k_must_be_positive_finite(value):
    solver, result = homogeneous()
    with pytest.raises(ValueError, match='k_eff'):
        assess_nem_solution(solver, replace(result, k_eff=value))


def test_positive_normalization_and_physical_flux_required():
    solver, result = homogeneous()
    zero = replace(result, means=np.zeros_like(result.means))
    with pytest.raises(ValueError, match='production normalization'):
        assess_nem_solution(solver, zero)
    with pytest.raises(ValueError, match='negative physical'):
        assess_nem_solution(solver, replace(result, means=-result.means))
    changed = result.moments.copy()
    changed[:, :, 1] = result.means[:, None, :] * 10
    with pytest.raises(ValueError, match='negative reconstructed profile'):
        assess_nem_solution(solver, replace(result, moments=changed))


def test_vacuum_robin_balance_and_corruption():
    solver, result = homogeneous(boundary='vacuum-x', degree=8)
    report = assess_nem_solution(solver, result)
    assert report['boundary_balance_error'] < 1e-5
    assert report['maximum_absolute_boundary_residual'] < 1e-8
    # The global-current orientation must be converted to outward normals.
    grid = solver.grid
    outward = grid.normal_signs[grid.boundary_sides, None] * result.currents[grid.boundary_faces]
    np.testing.assert_allclose(outward, grid.boundary_alpha[:, None]
                               * result.side_face_means[grid.boundary_nodes, grid.boundary_sides], atol=1e-10)
    changed = result.currents.copy(); changed[grid.boundary_faces[0]] *= 1.1
    with pytest.raises(ValueError, match='boundary_balance_error'):
        assess_nem_solution(solver, replace(result, currents=changed))


def test_arbitrary_positive_flux_normalization_is_accepted():
    solver, result = homogeneous()
    scaled = replace(result, **{field: getattr(result, field) * 7
                                for field in ('means', 'moments', 'currents', 'side_face_means', 'corrected_face_means')})
    assert assess_nem_solution(solver, scaled)['fission_production'] == pytest.approx(7.)


@pytest.mark.parametrize('mutation', ['fission', 'geometry', 'degree'])
def test_stale_solver_configuration_cannot_be_certified(mutation):
    solver, result = homogeneous()
    if mutation == 'fission':
        # Replacing F bypasses array write protection and invalidates the
        # eigenpair even though cached loss-factorization arrays still exist.
        solver.fission = solver.fission * 1.1
    elif mutation == 'geometry':
        solver.grid.widths = solver.grid.widths * 1.1
    else:
        solver.degree = 8
    with pytest.raises(ValueError, match='changed after cached loss assembly'):
        assess_nem_solution(solver, result)
