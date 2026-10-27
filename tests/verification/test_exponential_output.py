"""Output-aware adaptive FTM against physical trajectories and replay."""

import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.linalg import expm

from garnet import Kinetics, Material, Model
from tests.verification.test_exponential_adaptive import FUEL, point_core
from tests.verification.test_exponential_damping import core
from tools.assess_exponential_damping import CASES, problem


@pytest.mark.parametrize('scheme', ['ftm2', 'ftm-bdf2'])
def test_rapid_growth_output_and_energy_refine_at_fixed_state_tolerance(scheme):
    rho, duration, _ = CASES['superprompt-rise']
    _, _, reference = problem('superprompt-rise')
    changed = FUEL.replace(sigma_a=[0.03 * (1 - rho)])
    sample = np.linspace(0, duration, 8001)
    expected = reference(sample)[:, 0]
    errors, energies = [], []
    for output_rtol in (None, 1e-3, 1e-4):
        result = point_core().transient(duration, 0.01, device='cpu', time_scheme=scheme,
            adaptive_ftm={'rtol': 1e-7, 'min_dt': 1e-11, 'max_dt': 0.02,
                          'output_rtol': output_rtol},
            materials_at=lambda t: {'fuel': changed} if t > 0 else {},
            tol_step=1e-11, multigroup_kwargs={'rtol': 1e-12})
        errors.append(np.max(np.abs(result.power_at(sample) / expected - 1)))
        energies.append(abs(trapezoid(result.power, result.times) / trapezoid(expected, sample) - 1))
        assert np.max(np.abs(result.power / reference(result.times)[:, 0] - 1)) < 1e-5
        if output_rtol is None:
            assert not result.output_error_blocks
            assert not result.rejected_output_error_blocks
        else:
            assert len(result.output_error_blocks) == result.n_steps
            assert len(result.rejected_output_error_blocks) == result.rejected_steps
            assert np.max(result.output_error_blocks) <= 1
            assert all(len(blocks) == 3 for blocks in result.output_error_blocks)
            assert max(result.local_errors) <= 1
    if scheme == 'ftm2':
        assert errors[0] > 0.02  # Reproduce checkpoint four's output discrepancy.
    assert errors[2] < 1e-4
    assert errors[2] < errors[1] / 5
    assert energies[2] < 1e-4
    assert energies[2] < energies[1] / 5


@pytest.mark.parametrize('geometry', ['cartesian', 'cylindrical', 'triangular'])
@pytest.mark.parametrize('scheme', ['ftm2', 'ftm-bdf2'])
def test_output_rejections_restore_histories_events_and_callback_state(geometry, scheme):
    model = core(geometry)
    changed = FUEL.replace(sigma_a=[0.03 * (1 - 0.002)])
    observed = []
    options = dict(device='cpu', time_scheme=scheme, tol_step=1e-11,
        step_solver='monolithic', multigroup_kwargs={'rtol': 1e-12},
        materials_at=lambda t: {'fuel': changed} if t > 0.02 else {})
    result = model.transient(0.06, 0.01, adaptive_ftm={
        'rtol': 1e-2, 'output_rtol': 1e-5, 'min_dt': 1e-11, 'max_dt': 0.02},
        event_times=[0.02], step_alignment_times=[0.037],
        on_step=lambda t, phi, power: observed.append((t, power)), **options)
    assert 0.02 in result.times and 0.037 in result.times
    assert any(max(state) <= 1 < max(output) for state, output in zip(
        result.rejected_error_blocks, result.rejected_output_error_blocks))
    np.testing.assert_allclose(observed, np.column_stack((result.times[1:], result.power[1:])))
    replay = model.transient(0.06, np.repeat(result.step_widths / 2, 2),
                             event_times=[0.02], **options)
    np.testing.assert_allclose(result.power, replay.power[::2], rtol=2e-9)
    np.testing.assert_allclose(result.flux_numpy, replay.flux_numpy, rtol=2e-9)
    np.testing.assert_allclose(result.precursors_numpy, replay.precursors_numpy, rtol=2e-9)
    assert sum(result.feedback_iterations) + sum(result.rejected_feedback_iterations) == 3 * (
        result.n_steps + result.rejected_steps)


@pytest.mark.parametrize('scheme', ['ftm2', 'ftm-bdf2'])
def test_output_guard_resolves_spectral_shape_when_total_power_is_constant(scheme):
    fuel = Material(name='fuel', diffusion=[1., 1.], sigma_a=[0.03, 0.03],
        nu_sigma_f=[0.035, 0.035], chi=[0.5, 0.5], sigma_s=[[0., 0.01], [0.01, 0.]])
    changed = fuel.replace(sigma_s=[[0., 0.1], [0.01, 0.]])
    model = (Model((1, 1, 1), (1, 1, 1)).fill(fuel).set_boundary('reflective')
             .set_kinetics(Kinetics([100., 100.], [0.], [0.08])))
    # Equal absorption/fission and velocities conserve total flux and power;
    # asymmetric scattering redistributes the spectrum on a 1/14 s scale.
    matrix = 100 * np.array([[-0.115, 0.025], [0.115, -0.025]])
    sample = np.linspace(0, 0.5, 501)
    initial = np.full(2, 1 / 0.06)
    expected = np.array([expm(t * matrix) @ initial for t in sample])
    errors = []
    for output_rtol in (None, 1e-4):
        snapshots = [initial]
        result = model.transient(0.5, 0.1, device='cpu', time_scheme=scheme,
            adaptive_ftm={'rtol': 1e-2, 'output_rtol': output_rtol, 'min_dt': 1e-10},
            materials_at=lambda t: {'fuel': changed} if t > 0 else {},
            on_step=lambda t, phi, power: snapshots.append(np.asarray(phi).reshape(2).copy()),
            tol_step=1e-12, multigroup_kwargs={'rtol': 1e-13})
        sampled = np.column_stack([np.interp(sample, result.times, np.asarray(snapshots)[:, g])
                                   for g in range(2)])
        errors.append(np.max(np.abs(sampled / expected - 1)))
        np.testing.assert_allclose(result.power, 1., atol=1e-9)
        if output_rtol is not None:
            assert np.max(result.output_error_blocks) <= 1
            assert np.max(np.asarray(result.output_error_blocks)[:, -1]) < 1e-4
            assert any(max(blocks[:2]) > 1 for blocks in result.rejected_output_error_blocks)
    assert errors[1] < 1e-3
    assert errors[1] < errors[0] / 5


@pytest.mark.parametrize('scheme', ['ftm2', 'ftm-bdf2'])
def test_power_chord_detects_changing_fission_with_constant_physical_flux(scheme):
    model = point_core(beta=(0.,))
    result = model.transient(0.2, 0.2, device='cpu', time_scheme=scheme,
        adaptive_ftm={'rtol': 1e-5, 'output_rtol': 1e-4, 'min_dt': 1e-10},
        materials_at=lambda t: {'fuel': FUEL.replace(
            sigma_a=[0.03 * (1 + t*t)], nu_sigma_f=[0.035 * (1 + t*t)])},
        tol_step=1e-12, multigroup_kwargs={'rtol': 1e-13})
    sample = np.linspace(0, 0.2, 1001)
    assert np.max(np.abs(result.power_at(sample) / (1 + sample**2) - 1)) < 1e-4
    np.testing.assert_allclose(result.power, 1 + result.times**2, atol=1e-10)
    assert result.rejected_steps > 0
    assert any(blocks[-1] > 1 and max(blocks[:-1]) < 1e-4
               for blocks in result.rejected_output_error_blocks)
    assert np.max(result.output_error_blocks) <= 1


@pytest.mark.parametrize('controls,match', [
    ({'output_rtol': 0}, 'output_rtol'),
    ({'output_rtol': -1}, 'output_rtol'),
    ({'output_rtol': np.nan}, 'output_rtol'),
    ({'output_rtol': np.inf}, 'output_rtol'),
    ({'output_atol': 0}, 'requires output_rtol'),
    ({'output_rtol': 1e-3, 'output_atol': -1}, 'output_atol'),
    ({'output_rtol': 1e-3, 'output_atol': np.nan}, 'output_atol'),
    ({'output_rtol': 1e-3, 'output_atol': np.inf}, 'output_atol'),
])
def test_invalid_output_controls_fail_clearly(controls, match):
    with pytest.raises(ValueError, match=match):
        point_core().transient(0.01, 0.01, device='cpu', adaptive_ftm=controls)


def test_output_budget_at_minimum_width_fails_and_zero_duration_needs_no_trials():
    with pytest.raises(RuntimeError, match='min_dt'):
        point_core(beta=(0.,)).transient(0.1, 0.1, device='cpu', adaptive_ftm={
            'rtol': 1e-2, 'output_rtol': 1e-10, 'min_dt': 0.1},
            materials_at=lambda t: {'fuel': FUEL.replace(
                sigma_a=[0.03 * (1 + t*t)], nu_sigma_f=[0.035 * (1 + t*t)])})
    result = point_core().transient(0., 0.01, device='cpu', adaptive_ftm={'output_rtol': 1e-4})
    assert not result.output_error_blocks and not result.rejected_output_error_blocks
