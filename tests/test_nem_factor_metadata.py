"""Factor diagnostics must not export SuperLU's L/U sparse matrices."""
import numpy as np
import pytest

import garnet.nem as nem
from garnet.nem_geometry import CuboidNodalGrid


class ScalarFactorProxy:
    """Keep real SuperLU solves and counters, prohibit factor materialization."""

    def __init__(self, factor):
        self.factor = factor
        self.solve_calls = 0

    @property
    def nnz(self):
        return self.factor.nnz

    @property
    def L(self):
        raise AssertionError('diagnostics must not materialize SuperLU L')

    @property
    def U(self):
        raise AssertionError('diagnostics must not materialize SuperLU U')

    def solve(self, *args, **kwargs):
        self.solve_calls += 1
        return self.factor.solve(*args, **kwargs)


def guarded_factors(monkeypatch):
    real_splu = nem.splu
    factors = []

    def factor(matrix, **kwargs):
        assert kwargs == {'permc_spec': 'MMD_AT_PLUS_A'}
        result = ScalarFactorProxy(real_splu(matrix, **kwargs))
        factors.append(result)
        return result

    monkeypatch.setattr(nem, 'splu', factor)
    return factors


def tiny_solver(mode, **overrides):
    grid = CuboidNodalGrid([[[0., 1.], [0., 1.], [0., 1.]]], 'reflective')
    args = dict(D=[.7, .4], sigma_a=[.7, 1.4],
                sigma_s=[[0., .1], [.1, 0.]],
                nu_sigma_f=[.1, .2], chi=[1., 0.])
    if mode == 'native_p0_unqualified':
        # Include negative signed reduced absorption, as supported by native P0.
        args.update(sigma_a=None, sigma_s=None, native_total=[[1., 1.8]],
                    native_scattering_production=[[[.2, 1.1], [.1, .3]]])
    args.update(overrides)
    return nem.NEMDiffusionSolver(grid, **args)


def assert_metadata(info, factor):
    assert info['factor_nnz'] == int(factor.nnz)
    assert type(info['factor_nnz']) is int
    assert info['factor_nnz_kind'] == 'superlu_reported_L_plus_U'
    assert info['factor_entries_materialized_for_count'] is False
    assert info['condition_estimate_is_certified_upper_bound'] is False


def assert_same_state(actual, expected):
    for name in ('means', 'moments', 'currents', 'side_face_means', 'corrected_face_means'):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name),
                                   rtol=2e-11, atol=2e-13)


@pytest.mark.parametrize('mode', ['absorption_scattering', 'native_p0_unqualified'])
def test_constructor_and_solves_use_scalar_factor_metadata(monkeypatch, mode):
    reference = tiny_solver(mode)
    expected_fixed = reference.solve_fixed_source([.4, .7])
    expected_eigen = reference.solve_eigenvalue()
    factors = guarded_factors(monkeypatch)

    actual = tiny_solver(mode)
    assert len(factors) == 1
    factor = factors[0]
    assert factor.solve_calls > 0  # The real condition estimator used this LU.
    assert_metadata(actual._backend_evidence, factor)
    assert actual._backend_evidence['reaction_mode'] == mode
    np.testing.assert_array_equal(actual._loss_matrix.toarray(), reference._loss_matrix.toarray())
    np.testing.assert_array_equal(actual._scaled_matrix.toarray(), reference._scaled_matrix.toarray())
    np.testing.assert_array_equal(actual._row_scale, reference._row_scale)
    np.testing.assert_array_equal(actual._column_scale, reference._column_scale)
    # Independently apply the loss inverse, including transpose used for norm estimation.
    rhs = np.linspace(.1, 1., actual._scaled_matrix.shape[0])
    for trans in ('N', 'T'):
        np.testing.assert_allclose(factor.solve(rhs, trans=trans),
                                   reference._lu.solve(rhs, trans=trans), rtol=1e-13, atol=1e-14)

    fixed = actual.solve_fixed_source([.4, .7])
    eigen = actual.solve_eigenvalue()
    assert_same_state(fixed, expected_fixed)
    assert_same_state(eigen, expected_eigen)
    assert eigen.k_eff == pytest.approx(expected_eigen.k_eff, rel=1e-12)
    exact_k = actual.nu_sigma_f[0] @ np.linalg.solve(actual.R[0], actual.chi[0])
    assert eigen.k_eff == pytest.approx(exact_k, rel=2e-10)
    for result in (fixed, eigen):
        assert_metadata(result.diagnostics, factor)
    assert fixed.diagnostics['scaled_backward_error'] < actual.rtol
    assert eigen.diagnostics['full_eigen_residual'] < 1e-10
    assert len(factors) == 1  # Diagnostics and solves reuse the original factor.


@pytest.mark.parametrize('mode', ['absorption_scattering', 'native_p0_unqualified'])
def test_global_condition_rejection_keeps_scalar_metadata(monkeypatch, mode):
    guarded_factors(monkeypatch)
    # Scalar local responses have condition one, so limit=1 reaches the global
    # guard. Reflective face constraints coupled to the mean are not orthogonal.
    args = dict(D=[1.], nu_sigma_f=[.1], chi=[1.], condition_limit=1.)
    if mode == 'absorption_scattering':
        args.update(sigma_a=[.2], sigma_s=[[0.]])
    else:
        args.update(native_total=[[1.]], native_scattering_production=[[[.8]]])
    with pytest.raises(nem.NEMSolverError, match='condensed loss condition') as error:
        tiny_solver(mode, **args)
    info = error.value.diagnostics
    assert info['maximum_local_loss_condition_2_scaled'] == 1.
    assert info['condition_number_1_scaled_estimate'] > 1.
    assert info['factor_nnz'] > 0
    assert info['factor_nnz_kind'] == 'superlu_reported_L_plus_U'
    assert info['factor_entries_materialized_for_count'] is False


@pytest.mark.parametrize('mode', ['absorption_scattering', 'native_p0_unqualified'])
def test_local_condition_and_positive_initial_production_guards_unchanged(monkeypatch, mode):
    guarded_factors(monkeypatch)
    with pytest.raises(nem.NEMSolverError, match='local loss response condition'):
        tiny_solver(mode, condition_limit=1.)
    solver = tiny_solver(mode)
    with pytest.raises(ValueError, match='positive finite initial fission production'):
        solver.solve_eigenvalue(initial_means=[0., 0.])
    negative = solver.solve_fixed_source([-.4, -.7])
    assert negative.diagnostics['negative_mean_count'] == 2
    assert np.all(negative.means < 0)


def test_native_nonpositive_eigen_production_still_rejected(monkeypatch):
    guarded_factors(monkeypatch)
    solver = tiny_solver('native_p0_unqualified', native_total=[[1., 1.]],
                         native_scattering_production=[[[.2, 1.1], [1.1, .2]]])
    with pytest.raises(nem.NEMSolverError, match='nonpositive or nonfinite eigen production') as error:
        solver.solve_eigenvalue()
    assert error.value.result is not None
    assert error.value.diagnostics['factor_entries_materialized_for_count'] is False
