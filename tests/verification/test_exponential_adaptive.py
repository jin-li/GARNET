"""Physical-state adaptive FTM, with independent references and rollback checks."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from garnet import Kinetics, Material, Model, ThermalMaterial
from garnet.timescheme import physical_state_errors
from tools.assess_exponential_damping import BETA, DECAY, GENERATION, CASES, problem
from tests.verification.test_exponential_damping import core


FUEL = Material(name='fuel', diffusion=[1.3], sigma_a=[0.03], nu_sigma_f=[0.035])


def point_core(beta=(BETA,), decay=(DECAY,)):
    return (Model((1, 1, 1), (1, 1, 1)).fill(FUEL).set_boundary('reflective')
            .set_kinetics(Kinetics([2.2e5], beta, decay)))


def run_case(name, scheme, rtol, *, estimator='step-doubling', **options):
    rho, duration, event = CASES[name]
    changed = FUEL.replace(sigma_a=[0.03 * (1 - rho)])
    return point_core().transient(duration, 0.01, device='cpu', time_scheme=scheme,
        adaptive_ftm={'rtol': rtol, 'estimator': estimator, 'min_dt': 1e-11, 'max_dt': 0.02},
        event_times=[event] if event else None,
        materials_at=lambda t: {'fuel': changed} if t > event else {},
        tol_step=1e-11, multigroup_kwargs={'rtol': 1e-12}, **options)


def test_physical_norm_separates_fields_and_is_invariant_to_relative_block_units():
    fine = [np.array([1., 2.]), np.array([1e6, 2e6])]
    coarse = [np.array([0.99, 2.]), fine[1].copy()]
    errors = physical_state_errors(fine, coarse, fine, rtol=1e-3)
    assert errors[0] == pytest.approx(10 / np.sqrt(2))
    assert errors[1] == 0
    scaled = [fine[0], fine[1] * 1e-12]
    scaled_coarse = [coarse[0], coarse[1] * 1e-12]
    assert physical_state_errors(scaled, scaled_coarse, scaled, rtol=1e-3) == errors


def test_physical_norm_handles_masks_volumes_and_initially_dark_fields():
    fine = [np.array([0., 2., 0.])]
    coarse = [np.array([0., 1., 100.])]
    errors = physical_state_errors(fine, coarse, fine, rtol=0.1,
                                   weights=np.array([1., 3., 0.]))
    assert errors == pytest.approx([5 * np.sqrt(3 / 4)])
    assert physical_state_errors([np.zeros(2)], [np.zeros(2)], [np.zeros(2)], rtol=0.1) == [0.]
    assert physical_state_errors([np.zeros(2)], [np.ones(2)], [np.zeros(2)], rtol=0.1) == [np.inf]
    assert physical_state_errors([np.zeros(2)], [np.ones(2)], [np.zeros(2)],
                                 rtol=0.1, atol=0.5) == [2.]
    assert physical_state_errors([np.array([np.nan])], [np.ones(1)], [np.ones(1)],
                                 rtol=0.1) == [np.inf]


@pytest.mark.parametrize('scheme', ['ftm2', 'ftm-bdf2'])
@pytest.mark.parametrize('name', ['prompt-rise', 'superprompt-rise', 'shutdown', 'interior-shutdown'])
@pytest.mark.parametrize('estimator', ['step-doubling', 'history'])
def test_tolerance_refinement_against_independent_physical_kinetics(name, scheme, estimator):
    _, _, reference = problem(name)
    errors = []
    for tolerance in (1e-3, 1e-5):
        result = run_case(name, scheme, tolerance, estimator=estimator)
        expected = reference(result.times)
        errors.append(max(np.max(np.abs(result.power / expected[:, 0] - 1)),
            np.max(np.abs(result.precursors_numpy.ravel() / GENERATION / expected[-1, 1:] - 1))))
        assert max(result.local_errors) <= 1.0
        assert result.power.min() > 0
        assert result.times[-1] == CASES[name][1]
        assert len(result.local_error_blocks) == result.n_steps
        assert all(len(blocks) == 2 for blocks in result.local_error_blocks)
    assert errors[1] < errors[0] / 3, errors
    assert errors[1] < 0.003, errors


@pytest.mark.parametrize('geometry', ['cartesian', 'cylindrical', 'triangular'])
@pytest.mark.parametrize('scheme', ['ftm2', 'ftm-bdf2'])
def test_rejections_restore_all_histories_and_callbacks_see_only_commits(geometry, scheme):
    model = core(geometry)
    changed = FUEL.replace(sigma_a=[0.03 * (1 - 0.002)])
    observed = []
    options = dict(device='cpu', time_scheme=scheme, tol_step=1e-11,
        step_solver='monolithic', multigroup_kwargs={'rtol': 1e-12},
        materials_at=lambda t: {'fuel': changed} if t > 0.02 else {})
    adaptive = model.transient(0.06, 0.01, adaptive_ftm={
        'rtol': 1e-5, 'min_dt': 1e-11, 'max_dt': 0.02}, event_times=[0.02],
        on_step=lambda t, phi, power: observed.append((t, power)), **options)
    assert adaptive.rejected_steps > 0
    event_index = np.flatnonzero(adaptive.times == 0.02)[0]
    np.testing.assert_allclose(adaptive.power[:event_index + 1], 1.0, atol=1e-9)
    assert adaptive.ftm_trial_diagnostics[event_index][0]['frequency'] == 0.0
    assert adaptive.ftm_trial_diagnostics[event_index][0]['order'] == (1 if scheme == 'ftm-bdf2' else 2)
    np.testing.assert_allclose(observed, np.column_stack((adaptive.times[1:], adaptive.power[1:])))
    # Replay ONLY the accepted half steps: leaked coarse/rejected histories
    # would change the recurrence, frequency estimates or precursor state.
    half_widths = np.repeat(np.diff(adaptive.times) / 2, 2)
    replay = model.transient(0.06, half_widths, event_times=[0.02], **options)
    np.testing.assert_allclose(adaptive.power, replay.power[::2], rtol=2e-9)
    np.testing.assert_allclose(adaptive.flux_numpy, replay.flux_numpy, rtol=2e-9)
    np.testing.assert_allclose(adaptive.precursors_numpy, replay.precursors_numpy, rtol=2e-9)
    np.testing.assert_allclose(adaptive.step_source_weights, np.diff(adaptive.times))
    assert sum(adaptive.feedback_iterations) + sum(adaptive.rejected_feedback_iterations) == 3 * (
        adaptive.n_steps + adaptive.rejected_steps)
    assert all(len(trials) == 3 for trials in adaptive.ftm_trial_diagnostics)
    assert len(adaptive.rejected_ftm_trial_diagnostics) == adaptive.rejected_steps


@pytest.mark.parametrize('scheme', ['ftm2', 'ftm-bdf2'])
@pytest.mark.parametrize('output_rtol', [None, 1e-4])
@pytest.mark.parametrize('estimator', ['step-doubling', 'history'])
def test_nonequilibrium_six_family_precursors_against_radau(scheme, output_rtol, estimator):
    beta = np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273])
    decay = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
    initial = np.r_[1.0, 0.7 * beta / (GENERATION * decay)]

    def rhs(t, state):
        return np.r_[(0.001 * t / 0.05 - beta.sum()) / GENERATION * state[0]
                     + decay @ state[1:], beta / GENERATION * state[0] - decay * state[1:]]

    reference = solve_ivp(rhs, (0, 0.05), initial, method='Radau',
                           rtol=2e-12, atol=1e-13, dense_output=True)
    assert reference.success
    result = point_core(beta, decay).transient(0.05, 0.01, device='cpu', time_scheme=scheme,
        adaptive_ftm={'rtol': 1e-6, 'estimator': estimator, 'output_rtol': output_rtol,
                      'min_dt': 1e-11}, tol_step=1e-12,
        multigroup_kwargs={'rtol': 1e-13},
        initial_precursors=(initial[1:] * GENERATION).reshape(6, 1, 1, 1),
        materials_at=lambda t: {'fuel': FUEL.replace(sigma_a=[0.03 * (1 - 0.001 * t / 0.05)])})
    expected = reference.sol(result.times)
    assert np.max(np.abs(result.power / expected[0] - 1)) < 1e-4
    np.testing.assert_allclose(result.precursors_numpy.ravel() / GENERATION,
                               expected[1:, -1], rtol=1e-4)
    assert all(len(blocks) == 7 for blocks in result.local_error_blocks)
    if output_rtol is not None:
        assert np.asarray(result.output_error_blocks).shape == (result.n_steps, 8)
        assert np.max(result.output_error_blocks) <= 1


@pytest.mark.parametrize('scheme', ['ftm2', 'ftm-bdf2'])
@pytest.mark.parametrize('output_rtol', [None, 1e-4])
@pytest.mark.parametrize('estimator', ['step-doubling', 'history'])
def test_localized_two_group_shape_change_against_independent_matrix_exponential(scheme, output_rtol, estimator):
    diffusion, absorption = np.array([1.4, 0.4]), np.array([0.02, 0.1])
    fission, spectrum = np.array([0.005, 0.15]), np.array([0.9, 0.1])
    scatter, speeds = np.array([[0., 0.03], [0.002, 0.]]), np.array([200., 60.])
    beta, decay = np.array([0.005, 0.002]), np.array([0.08, 0.4])
    delayed = np.array([[0.95, 0.05], [0.8, 0.2]])
    fuel = Material(name='fuel', diffusion=diffusion, sigma_a=absorption,
                    nu_sigma_f=fission, sigma_s=scatter, chi=spectrum)
    left = fuel.replace(name='left')
    model = (Model((4, 1, 1), (4, 1, 1)).fill(fuel).add_box(left, x=(0, 2))
             .set_boundary('reflective').set_kinetics(Kinetics(speeds, beta, decay,
                                                                            chi_delayed=delayed)))
    steady = model.steady(device='cpu', tol_k=1e-13, tol_source=1e-12)
    initial_flux = steady.flux_numpy.reshape(2, 4).copy()
    initial_flux /= np.sum(fission[:, None] * initial_flux) / steady.k_eff
    source = fission @ initial_flux / steady.k_eff
    initial = np.r_[initial_flux.ravel(), (beta[:, None] / decay[:, None] * source).ravel()]
    # Independent 1-D reflecting FV diffusion matrix: unit cell widths and
    # unchanged face diffusion, with absorption changed only in the left half.
    laplace = np.diag([-1., -2., -2., -1.]) + np.diag(np.ones(3), 1) + np.diag(np.ones(3), -1)
    matrix = np.zeros((16, 16))
    changed_absorption = np.broadcast_to(absorption[:, None], (2, 4)).copy()
    changed_absorption[:, :2] *= np.array([0.8, 0.95])[:, None]
    for group in range(2):
        row = slice(4 * group, 4 * (group + 1))
        matrix[row, row] += speeds[group] * (diffusion[group] * laplace - np.diag(
            changed_absorption[group] + scatter[group].sum()))
        for other in range(2):
            col = slice(4 * other, 4 * (other + 1))
            matrix[row, col] += speeds[group] * np.eye(4) * (
                scatter[other, group] + (spectrum[group] - beta @ delayed[:, group])
                * fission[other] / steady.k_eff)
        for family in range(2):
            col = slice(8 + 4 * family, 12 + 4 * family)
            matrix[row, col] = speeds[group] * delayed[family, group] * decay[family] * np.eye(4)
    for family in range(2):
        row = slice(8 + 4 * family, 12 + 4 * family)
        matrix[row, row] = -decay[family] * np.eye(4)
        for group in range(2):
            matrix[row, 4 * group:4 * (group + 1)] = beta[family] * fission[group] / steady.k_eff * np.eye(4)
    changed = left.replace(sigma_a=absorption * [0.8, 0.95])
    snapshots = [initial_flux.ravel()]
    result = model.transient(0.1, 0.02, device='cpu', time_scheme=scheme,
        initial_steady=steady, adaptive_ftm={'rtol': 1e-6, 'output_rtol': output_rtol,
                                            'estimator': estimator, 'min_dt': 1e-10},
        materials_at=lambda t: {'left': changed} if t > 0 else {},
        on_step=lambda t, phi, power: snapshots.append(np.asarray(phi).ravel().copy()),
        tol_step=1e-12, multigroup_kwargs={'rtol': 1e-13})
    expected = expm(0.1 * matrix) @ initial
    actual = np.r_[result.flux_numpy.ravel(), result.precursors_numpy.ravel()]
    np.testing.assert_allclose(actual, expected, rtol=2e-5)
    assert np.ptp(result.flux_numpy[0]) / np.mean(result.flux_numpy[0]) > 1e-3
    assert all(len(blocks) == 4 for blocks in result.local_error_blocks)
    if output_rtol is not None:
        sample = np.linspace(0, 0.1, 201)
        trajectory = expm_multiply(matrix, initial, start=0, stop=0.1, num=201)
        sampled = np.column_stack([np.interp(sample, result.times, np.asarray(snapshots)[:, g])
                                   for g in range(8)])
        np.testing.assert_allclose(sampled, trajectory[:, :8], rtol=1e-4)
        assert np.asarray(result.output_error_blocks).shape == (result.n_steps, 5)
        assert np.max(result.output_error_blocks) <= 1


@pytest.mark.parametrize('scheme', ['ftm2', 'ftm-bdf2'])
def test_frequency_correctors_charge_every_trial_solve(scheme):
    result = run_case('prompt-rise', scheme, 1e-3,
        ftm_kwargs={'frequency_iterations': 2, 'frequency_tolerance': 1e-10})
    trials = result.ftm_trial_diagnostics + result.rejected_ftm_trial_diagnostics
    extra = sum(trial['frequency_refinements'] for attempt in trials for trial in attempt)
    assert extra > 0
    assert sum(result.feedback_iterations) + sum(result.rejected_feedback_iterations) == (
        3 * (result.n_steps + result.rejected_steps) + extra)


def test_alignment_times_do_not_restart_and_zero_duration_has_no_trials():
    result = point_core().transient(0.04, 0.01, device='cpu', adaptive_ftm={},
                                   step_alignment_times=[0.013])
    assert 0.013 in result.times
    np.testing.assert_allclose(result.power, 1.0, atol=1e-10)
    result = point_core().transient(0.0, 0.01, device='cpu', adaptive_ftm={})
    assert result.n_steps == 0
    assert not result.ftm_trial_diagnostics


@pytest.mark.parametrize('options,match', [
    ({'adaptive_bdf': {}}, 'not both'),
    ({'time_scheme': 'bdf2'}, 'requires ftm2'),
    ({'time_scheme': 'ftm'}, 'requires ftm2'),
    ({'step_solver': 'fixed-point'}, 'monolithic'),
    ({'dt': [0.01]}, 'scalar'),
    ({'adaptive_ftm': {'rtol': 0}}, 'rtol'),
    ({'adaptive_ftm': {'automatic_order': True}}, 'unknown'),
    ({'adaptive_ftm': {'max_rejections': 0}}, 'max_rejections'),
])
def test_invalid_adaptive_combinations_fail_clearly(options, match):
    controls = dict(t_end=0.01, dt=0.01, device='cpu', adaptive_ftm={})
    controls.update(options)
    with pytest.raises(ValueError, match=match):
        point_core().transient(**controls)


def test_error_at_minimum_width_fails_clearly():
    with pytest.raises(RuntimeError, match='min_dt'):
        point_core().transient(0.01, 0.01, device='cpu', adaptive_ftm={
            'rtol': 1e-12, 'min_dt': 0.01}, materials_at=lambda t: {
                'fuel': FUEL.replace(sigma_a=[0.029])} if t > 0 else {})


def test_coupled_driver_refuses_neutron_only_controller():
    model = point_core().configure_thermal({'fuel': ThermalMaterial(0.25)}, total_power=100)
    with pytest.raises(ValueError, match='neutron/precursor'):
        model.coupled_transient(0.01, 0.01, device='cpu', transient_kwargs={'adaptive_ftm': {}})
