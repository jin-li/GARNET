"""Physical-history estimator, same-trial audits and rollback verification."""

import numpy as np
import pytest

from garnet.timescheme import ExponentialTheta, physical_history_prediction
from tests.verification.test_exponential_adaptive import FUEL, point_core
from tests.verification.test_exponential_damping import core
from tools.assess_exponential_damping import CASES, problem


@pytest.mark.parametrize('ratio', [0.25, 0.6, 1., 1.7, 2.])
def test_physical_prediction_is_exact_for_quadratics_on_unequal_widths(ratio):
    previous_width, width = 0.03, 0.03 * ratio
    def state(t):
        return [unit * np.array([1 + 2*t + 4*t*t, 3 - t + 2*t*t, 0.])
                for unit in (1e-9, 1., 1e9)]
    result = physical_history_prediction(state(-previous_width), state(0), state(width),
        half_width=width, previous_half_width=previous_width)
    np.testing.assert_allclose(result, state(2 * width), rtol=5e-15)


def test_prediction_defect_has_cubic_scaling_on_a_smooth_cubic():
    defects = []
    for width in (0.02, 0.01, 0.005):
        value = lambda t: [np.array([1 + t**3])]
        predicted = physical_history_prediction(value(-1.3 * width), value(0), value(width),
            half_width=width, previous_half_width=1.3 * width)
        defects.append(float(abs(predicted[0][0] - value(2 * width)[0][0])))
    np.testing.assert_allclose(np.array(defects[:-1]) / defects[1:], 8, rtol=1e-8)


@pytest.mark.parametrize('scheme', ['ftm2', 'ftm-bdf2'])
@pytest.mark.parametrize('name', ['superprompt-rise', 'interior-shutdown'])
def test_audit_keeps_reference_trajectory_and_counts_while_recording_paired_estimates(scheme, name):
    rho, duration, event = CASES[name]
    changed = FUEL.replace(sigma_a=[0.03 * (1 - rho)])
    options = dict(device='cpu', time_scheme=scheme,
        event_times=[event] if event else None,
        materials_at=lambda t: {'fuel': changed} if t > event else {},
        tol_step=1e-11, multigroup_kwargs={'rtol': 1e-12})
    controls = dict(rtol=1e-4, output_rtol=1e-4, min_dt=1e-11, max_dt=0.02)
    reference = point_core().transient(duration, 0.01, adaptive_ftm=controls, **options)
    audit = point_core().transient(duration, 0.01, adaptive_ftm=dict(
        controls, estimator='history-audit'), **options)
    np.testing.assert_array_equal(reference.times, audit.times)
    np.testing.assert_array_equal(reference.power, audit.power)
    np.testing.assert_array_equal(reference.flux_numpy, audit.flux_numpy)
    np.testing.assert_array_equal(reference.precursors_numpy, audit.precursors_numpy)
    assert reference.feedback_iterations == audit.feedback_iterations
    assert reference.rejected_feedback_iterations == audit.rejected_feedback_iterations
    assert reference.local_error_blocks == audit.local_error_blocks
    assert reference.output_error_blocks == audit.output_error_blocks
    assert not reference.ftm_estimator_diagnostics
    assert any(d['history_error_blocks'] is not None for d in audit.ftm_estimator_diagnostics)
    assert all(d['method'] == 'step-doubling' for d in audit.ftm_estimator_diagnostics)
    if event:
        index = np.flatnonzero(audit.times == event)[0]
        assert audit.ftm_estimator_diagnostics[index]['reason'] == 'bootstrap'


@pytest.mark.parametrize('scheme', ['ftm2', 'ftm-bdf2'])
@pytest.mark.parametrize('geometry', ['cartesian', 'cylindrical', 'triangular'])
def test_history_rejections_events_and_accepted_half_step_replay(scheme, geometry):
    model = core(geometry)
    changed = FUEL.replace(sigma_a=[0.03 * (1 - 0.002)])
    observed = []
    options = dict(device='cpu', time_scheme=scheme, tol_step=1e-11,
        step_solver='monolithic', multigroup_kwargs={'rtol': 1e-12},
        materials_at=lambda t: {'fuel': changed} if t > 0.02 else {})
    result = model.transient(0.06, 0.01, adaptive_ftm={
        'rtol': 1e-5, 'output_rtol': 1e-4, 'min_dt': 1e-11, 'max_dt': 0.02,
        'estimator': 'history'}, event_times=[0.02], step_alignment_times=[0.037],
        on_step=lambda t, phi, power: observed.append((t, power)), **options)
    assert result.rejected_steps > 0
    assert any(d['method'] == 'history' for d in result.ftm_estimator_diagnostics)
    assert len(result.ftm_estimator_diagnostics) == result.n_steps
    assert len(result.rejected_ftm_estimator_diagnostics) == result.rejected_steps
    event_index = np.flatnonzero(result.times == 0.02)[0]
    assert result.ftm_estimator_diagnostics[event_index]['reason'] == 'bootstrap'
    assert 0.037 in result.times
    replay = model.transient(0.06, np.repeat(result.step_widths / 2, 2),
                            event_times=[0.02], **options)
    np.testing.assert_allclose(result.power, replay.power[::2], rtol=2e-9)
    np.testing.assert_allclose(result.flux_numpy, replay.flux_numpy, rtol=2e-9)
    np.testing.assert_allclose(result.precursors_numpy, replay.precursors_numpy, rtol=2e-9)
    np.testing.assert_allclose(observed, np.column_stack((result.times[1:], result.power[1:])))
    attempts = result.ftm_trial_diagnostics + result.rejected_ftm_trial_diagnostics
    assert sum(result.feedback_iterations) + sum(result.rejected_feedback_iterations) == sum(
        1 + trial['frequency_refinements'] for trials in attempts for trial in trials)
    assert any(len(trials) == 2 for trials in result.ftm_trial_diagnostics)
    assert max(result.local_errors) <= 1
    assert np.max(result.output_error_blocks) <= 1


@pytest.mark.parametrize('scheme', ['ftm2', 'ftm-bdf2'])
def test_history_reduces_solves_on_the_same_output_limited_rapid_growth_grid(scheme):
    rho, duration, _ = CASES['superprompt-rise']
    changed = FUEL.replace(sigma_a=[0.03 * (1 - rho)])
    _, _, reference = problem('superprompt-rise')
    runs = []
    for mode in ('step-doubling', 'history'):
        runs.append(point_core().transient(duration, 0.01, device='cpu', time_scheme=scheme,
            adaptive_ftm=dict(rtol=1e-4, output_rtol=1e-4, min_dt=1e-11,
                              max_dt=0.02, estimator=mode),
            materials_at=lambda t: {'fuel': changed} if t > 0 else {},
            tol_step=1e-11, multigroup_kwargs={'rtol': 1e-12}))
    reference_run, cheap = runs
    np.testing.assert_array_equal(reference_run.times, cheap.times)
    np.testing.assert_array_equal(reference_run.power, cheap.power)
    work = lambda run: sum(run.feedback_iterations) + sum(run.rejected_feedback_iterations)
    assert work(cheap) < 0.7 * work(reference_run)
    sample = np.linspace(0, duration, 8001)
    assert np.max(np.abs(cheap.power_at(sample) / reference(sample)[:, 0] - 1)) < 1e-3


def test_unexpected_half_step_be_fallback_replays_the_oracle_and_charges_discarded_work(monkeypatch):
    original = ExponentialTheta.prepare_step
    half_calls = 0
    def injected(self, step, dt):
        nonlocal half_calls
        original(self, step, dt)
        if dt == 0.005:
            half_calls += 1
            if half_calls == 3:  # First cheap half after the bootstrap macro.
                self.fallback_reason = 'injected-backward-euler'
    monkeypatch.setattr(ExponentialTheta, 'prepare_step', injected)
    result = point_core().transient(0.03, 0.01, device='cpu', time_scheme='ftm2',
        adaptive_ftm=dict(estimator='history', rtol=1e-4, min_dt=1e-10, max_dt=0.01))
    assert result.ftm_estimator_diagnostics[1]['reason'] == 'trial-fallback'
    assert len(result.ftm_trial_diagnostics[1]) == 4
    assert result.feedback_iterations[1] == 4
    np.testing.assert_allclose(result.power, 1, atol=1e-10)


def test_tiny_alignment_width_uses_oracle_and_zero_duration_has_no_estimates():
    result = point_core().transient(0.04, 0.01, device='cpu', adaptive_ftm=dict(
        estimator='history', rtol=1e-4, min_dt=1e-12, max_dt=0.01),
        step_alignment_times=[0.020000001])
    assert 0.020000001 in result.times
    assert any(d['reason'] == 'history-width-ratio' for d in result.ftm_estimator_diagnostics)
    result = point_core().transient(0., 0.01, device='cpu', adaptive_ftm={'estimator': 'history'})
    assert not result.ftm_estimator_diagnostics


@pytest.mark.parametrize('mode', ['unknown', '', None])
def test_invalid_estimator_controls_fail_clearly(mode):
    with pytest.raises(ValueError, match='estimator'):
        point_core().transient(0.01, 0.01, device='cpu', adaptive_ftm={'estimator': mode})


@pytest.mark.parametrize('width,previous', [(0, 1), (1, np.nan), (3, 1), (0.1, 1)])
def test_invalid_prediction_widths_fail_clearly(width, previous):
    with pytest.raises(ValueError, match='width'):
        physical_history_prediction([np.ones(2)], [np.ones(2)], [np.ones(2)],
                                    half_width=width, previous_half_width=previous)
