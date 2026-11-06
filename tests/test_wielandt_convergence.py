"""Wielandt integration with the fork's settled-source convergence contract."""

import numpy as np
import pytest

from garnet import DiffusionEigenSolver, Grid, Material
from garnet import solver as solver_module


def _solver(shape=(2, 2, 2), bc="reflective"):
    material = Material(
        diffusion=[1., .3], sigma_a=[.2, .04], nu_sigma_f=[.2, .1],
        sigma_s=[[0., .1], [.1, 0.]], chi=[.5, .5])
    grid = Grid(shape, (4., 4., 4.))
    return DiffusionEigenSolver(grid, [material], np.zeros(grid.shape, dtype=int),
                                bc=bc, device='cpu')


def test_shifted_solve_matches_independent_homogeneous_group_pencil():
    solver = _solver()
    loss = np.array([[.3, -.1], [-.1, .14]])
    fission = np.outer([.5, .5], [.2, .1])
    truth = np.linalg.eigvals(np.linalg.solve(loss, fission)).real.max()
    result = solver.solve(wielandt_shift=.2, tol_k=1e-11, tol_source=1e-10,
                          inner_rtol_floor=1e-12)
    assert result.converged and result.failure_reason is None
    assert result.multigroup_iterations > 0
    assert result.k_eff == pytest.approx(truth, abs=1e-10)
    assert result.fixed_source_error <= result.fixed_source_tolerance
    assert result.balance_error < 1e-10
    assert np.all(result.flux_numpy >= -result.scalar_flux_negative_tolerance)


def test_failed_shifted_solve_does_not_advance_k_or_anderson(monkeypatch):
    calls = []
    def failed_solve(self, state, *args):
        calls.append(True)
        # Simulate FGMRES exhausting its budget and returning an inaccurate iterate.
        return [1.5 * value for value in state], self.maxiter
    monkeypatch.setattr(solver_module._ShiftedMultigroupSystem, 'solve', failed_solve)
    monkeypatch.setattr(solver_module, '_anderson_source', lambda hist, raw, *a: raw)
    solver = _solver()
    first = solver.solve(max_outer=1)
    result = solver.solve(wielandt_shift=.2, max_outer=10)
    assert calls == [True]
    assert not result.converged and result.outer_iterations == 2
    assert result.k_history == first.k_history
    assert result.source_error_history == first.source_error_history
    assert result.k_eff == first.k_eff
    assert result.fixed_source_error > result.fixed_source_tolerance
    assert result.failure_reason == 'shifted multigroup budget exhausted before requested balance'
    assert np.isfinite(result.balance_error)
    assert result.multigroup_iterations == 200


def test_shift_preserves_first_outer_scattering_budget_failure(monkeypatch):
    def unexpected_solve(*args):
        pytest.fail('unsettled first outer entered shifted solve')
    monkeypatch.setattr(solver_module._ShiftedMultigroupSystem, 'solve', unexpected_solve)
    result = _solver().solve(wielandt_shift=.2, max_group_sweeps=1, k_guess=1.234)
    assert not result.converged and result.outer_iterations == 1
    assert result.k_eff == 1.234 and result.k_history == []
    assert result.multigroup_iterations == 0
    assert result.failure_reason == 'fixed-source group budget exhausted before requested balance'


@pytest.mark.parametrize('value', [0., -1., np.inf, np.nan, 1.])
def test_shifted_residual_cap_is_validated(value):
    with pytest.raises(ValueError, match='rtol_cap'):
        _solver().solve(wielandt_shift=.2, multigroup_kwargs={'rtol_cap': value})


def test_actual_shifted_fgmres_budget_exhaustion():
    result = _solver(shape=(3, 2, 2), bc='vacuum').solve(
        wielandt_shift=.2, inner_rtol_floor=1e-12,
        multigroup_kwargs={'maxiter': 1})
    assert not result.converged and result.outer_iterations == 2
    assert result.multigroup_iterations == 1 and len(result.k_history) == 1
    assert result.fixed_source_error > result.fixed_source_tolerance
    assert result.failure_reason == 'shifted multigroup budget exhausted before requested balance'


@pytest.mark.parametrize('negative_roundoff', [False, True])
def test_masked_support_and_negative_roundoff_shift_anchor(monkeypatch, negative_roundoff):
    active = np.ones((3, 1, 1), dtype=bool)
    active[-1] = False
    material = Material(diffusion=[1.], sigma_a=[.2], nu_sigma_f=[.2])
    solver = DiffusionEigenSolver(Grid(active.shape, (3., 1., 1.)), material,
                                 active=active, device='cpu')
    if negative_roundoff:
        # Expose the inactive trace to production so this fixture tests an
        # actually negative source, rather than masked production times phi=0.
        solver.nu_sigma_f[0][-1] = .2
        original_phi = solver._phi
        def rounded_trace(state, adjoint=False):
            phi = original_phi(state, adjoint).copy()
            phi[-1] = -1e-16
            return phi
        monkeypatch.setattr(solver, '_phi', rounded_trace)
        assert solver._fission_source(solver._initial_state(), solver.nu_sigma_f)[-1] < 0
        monkeypatch.setattr(solver_module._ShiftedMultigroupSystem, 'solve',
                            lambda *a: pytest.fail('negative source entered ratio-based shift'))
    # Masked zero source cells must not enter a component-ratio division.
    with np.errstate(divide='raise', invalid='raise'):
        result = solver.solve(wielandt_shift=.2,
                              tol_k=1e-10, tol_source=1e-9)
    assert result.converged and result.balance_error < 1e-9
    if negative_roundoff:
        assert result.multigroup_iterations == 0
        assert np.all(result.flux[:, -1] == -1e-16)
    else:
        assert np.all(np.abs(result.flux[:, -1]) < result.scalar_flux_negative_tolerance)
