"""Independent damping, full-state transformation and diffusion API checks."""

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.integrate import solve_ivp

from garnet import CylindricalModel, HexLattice, Kinetics, Material, Model, ThermalMaterial
from garnet._exponential import ExponentialBDF2Step, ExponentialSDIRK2Step
from garnet.timescheme import ExponentialBDF2, make_time_scheme
from tools.assess_exponential_damping import (BETA, DECAY, GENERATION,
    integrate_damped, point_matrix)


FUEL = Material(name="fuel", diffusion=[1.3], sigma_a=[0.03], nu_sigma_f=[0.035])
KINETICS = Kinetics([2.2e5], [BETA], [DECAY])


def core(geometry="cartesian", kinetics=KINETICS):
    if geometry == "cartesian":
        return Model((4, 3, 1), (4, 3, 1)).fill(FUEL).set_boundary("reflective").set_kinetics(kinetics)
    if geometry == "cylindrical":
        return (CylindricalModel(4, 3, (4, 3)).fill(FUEL)
                .set_boundary("reflective").set_kinetics(kinetics))
    return (HexLattice(pitch=4, refine=1).set_disk(0, FUEL)
            .set_boundary("reflective").set_kinetics(kinetics).build())


def run_uniform(*, geometry="cartesian", rho=0.002, widths=0.002, duration=0.02,
                frequency=None, scheme="ftm-bdf2", **kwargs):
    model = core(geometry)
    changed = FUEL.replace(sigma_a=[0.03 * (1 - rho)])
    options = {'ftm_kwargs': {'frequency': frequency}} if scheme == 'ftm-bdf2' else {}
    return model.transient(duration, widths, device="cpu", time_scheme=scheme,
        materials_at=lambda t: {'fuel': changed} if t > 0 else {},
        tol_step=1e-11, step_solver="monolithic", multigroup_kwargs={'rtol': 1e-12},
        **options, **kwargs)


@pytest.mark.parametrize("frequency", [-0.7, 0.0, 0.7, None])
def test_dense_sdirk2_has_second_order_for_smooth_time_dependence(frequency):
    errors = []
    for width in (0.04, 0.02, 0.01):
        result = integrate_damped(lambda t: np.array([[2 * t]]), [1.0],
                                  [width] * round(0.2 / width), frequency=frequency)
        errors.append(np.max(np.abs(result.states[:, 0] / np.exp(result.times**2) - 1)))
        assert result.implicit_solves == 2 * (len(result.times) - 1)
    assert errors[0] / errors[1] > 3.7, errors
    assert errors[1] / errors[2] > 3.7, errors


@pytest.mark.parametrize("method", ['sdirk2', 'bdf2'])
@pytest.mark.parametrize("rho", [-0.002, 0.002])
def test_prescribed_complete_kinetic_eigenmodes_are_exact(method, rho):
    matrix = point_matrix(rho)
    values, vectors = np.linalg.eig(matrix)
    index = int(np.argmax(values))
    omega = float(values[index])
    initial = vectors[:, index] / vectors[0, index]
    result = integrate_damped(lambda t: matrix, initial, [0.03, 0.01, 0.02, 0.025],
                              frequency=omega, method=method)
    exact = np.exp(result.times[:, None] * omega) * initial
    np.testing.assert_allclose(result.states, exact, rtol=3e-13)


def test_sdirk2_damps_stiff_decay_but_is_not_an_unconditional_positivity_method():
    # The trapezoidal stiff-mode gain approaches -1; SDIRK2 approaches zero.
    result = integrate_damped(lambda t: np.diag([-1.0, -1e8]), [1.0, 1.0],
                              [0.1], frequency=0.0)
    assert abs(result.states[-1, 1]) < 1e-6
    trapezoidal_gain = (1 - 0.5e7) / (1 + 0.5e7)
    assert abs(trapezoidal_gain) > 0.999
    counterexample = integrate_damped(lambda t: np.array([[-10.0]]), [1.0],
                                      [1.0], frequency=0.0)
    assert counterexample.states[-1, 0] < 0


@pytest.mark.parametrize("geometry", ['cartesian', 'cylindrical', 'triangular'])
@pytest.mark.parametrize("frequency", [None, 0.0, 0.7])
def test_fv_transformed_bdf2_matches_independent_full_matrix_recurrence(geometry, frequency):
    widths = [0.001, 0.0015, 0.002, 0.0015, 0.002]
    duration = sum(widths)
    result = run_uniform(geometry=geometry, widths=widths, duration=duration, frequency=frequency)
    matrix = point_matrix(0.002)
    initial = [1.0, BETA / (GENERATION * DECAY)]
    dense = integrate_damped(lambda t: matrix, initial, widths, frequency=frequency, method='bdf2')
    np.testing.assert_allclose(result.power, dense.states[:, 0], rtol=5e-10)
    metrics = result.model.grid.cylindrical_metrics() if geometry != 'triangular' else None
    weights = 1.0 if metrics is None else metrics[0]
    # The solver normalizes the metric-weighted fission-source sum to one.
    # Integrate the precursor field in that same normalization.
    np.testing.assert_allclose(np.sum(result.precursors_numpy[0] * weights) / GENERATION,
                               dense.states[-1, 1], rtol=5e-10)
    np.testing.assert_allclose(result.step_frequencies, dense.frequencies, atol=2e-7)
    assert sum(result.feedback_iterations) == len(widths)
    assert result.time_orders == [1, 2, 2, 2, 2]


@pytest.mark.parametrize("frequency", [None, 0.0, 0.7, -0.7])
def test_fv_smooth_prompt_problem_retains_second_order(frequency):
    model = (Model((1, 1, 1), (1, 1, 1)).fill(FUEL).set_boundary('reflective')
             .set_kinetics(Kinetics([2.2e5], [0.0], [DECAY])))
    errors = []
    for width in (0.02, 0.01, 0.005):
        result = model.transient(0.2, width, device='cpu', time_scheme='ftm-bdf2',
            ftm_kwargs={'frequency': frequency}, step_solver='monolithic', tol_step=1e-12,
            multigroup_kwargs={'rtol': 1e-13},
            materials_at=lambda t: {'fuel': FUEL.replace(sigma_a=[0.03 - 2 * t / 2.2e5])})
        errors.append(np.max(np.abs(result.power / np.exp(result.times**2) - 1)))
    assert errors[0] / errors[1] > 3.7, errors
    assert errors[1] / errors[2] > 3.7, errors


def test_zero_frequency_reproduces_existing_variable_step_bdf2():
    widths = [0.001, 0.0015, 0.002, 0.0015, 0.002]
    transformed = run_uniform(widths=widths, duration=sum(widths), frequency=0.0)
    baseline = run_uniform(widths=widths, duration=sum(widths), scheme='bdf2')
    np.testing.assert_allclose(transformed.power, baseline.power, rtol=2e-11)
    np.testing.assert_allclose(transformed.precursors_numpy, baseline.precursors_numpy, rtol=2e-11)


def test_bounds_use_actual_be_and_report_the_cause():
    transformed = run_uniform(widths=0.005, duration=0.02, frequency=1000.0)
    baseline = run_uniform(widths=0.005, duration=0.02, scheme='backward-euler')
    np.testing.assert_allclose(transformed.power, baseline.power, rtol=2e-11)
    np.testing.assert_allclose(transformed.precursors_numpy, baseline.precursors_numpy, rtol=2e-11)
    assert transformed.ftm_fallback_reasons == ['frequency-bound'] * 4
    assert transformed.time_orders == [1] * 4
    large_ratio = run_uniform(widths=[0.001, 0.004, 0.004], duration=0.009, frequency=0.0)
    assert large_ratio.ftm_fallback_reasons[1] == 'step-ratio-bound'
    assert large_ratio.time_orders == [1, 1, 2]
    # A drastic shrink can make the older state's exponential unsafe even
    # though the current interval is within the ordinary frequency bound.
    scheme = ExponentialBDF2(frequency=10.0)
    scheme.start([np.array([1.0])])
    scheme.prepare_step(1, 0.1)
    scheme.push([np.array([1.0])])
    scheme.prepare_step(2, 0.001)
    assert scheme.fallback_reason == 'history-frequency-bound'
    assert scheme.order_at(2) == 1


def test_interior_event_restarts_history_after_the_continuous_event_state():
    model = core()
    changed = FUEL.replace(sigma_a=[0.03 * (1 - 0.002)])
    observed = []
    result = model.transient(0.04, 0.002, device='cpu', time_scheme='ftm-bdf2',
        event_times=[0.02], materials_at=lambda t: {'fuel': changed} if t > 0.02 else {},
        on_step=lambda t, phi, p: observed.append(t), step_solver='monolithic',
        tol_step=1e-11, multigroup_kwargs={'rtol': 1e-12})
    event = int(np.flatnonzero(np.isclose(result.times, 0.02))[0])
    np.testing.assert_allclose(result.power[:event + 1], 1.0, atol=1e-9)
    assert result.time_orders[event] == 1
    assert result.step_frequencies[event] == 0.0
    np.testing.assert_allclose(observed, result.times[1:])
    np.testing.assert_allclose(result.step_source_weights, result.step_widths)


def test_refinement_counts_repeated_solves_without_advancing_history_twice():
    result = core().transient(0.02, 0.002, device='cpu', time_scheme='ftm-bdf2',
        materials_at=lambda t: {'fuel': FUEL.replace(sigma_a=[0.03 - 2 * t / 2.2e5])},
        ftm_kwargs={'frequency_iterations': 4, 'frequency_tolerance': 1e-10},
        step_solver='monolithic', tol_step=1e-12, multigroup_kwargs={'rtol': 1e-13})
    assert sum(result.step_frequency_refinements) > 0
    assert sum(result.feedback_iterations) == result.n_steps + sum(result.step_frequency_refinements)
    assert len(result.times) == 11
    defects = result.step_widths * np.abs(np.diff(np.log(result.power)) / result.step_widths
                                         - np.asarray(result.step_frequencies))
    np.testing.assert_allclose(result.step_frequency_log_defects, defects, atol=2e-15)


def test_adaptive_bdf_cannot_silently_control_a_frequency_transformed_recurrence():
    with pytest.raises(ValueError, match='requires bdf'):
        core().transient(0.02, 0.002, device='cpu', time_scheme='ftm-bdf2',
                         adaptive_bdf={'rtol': 1e-3})
    assert isinstance(make_time_scheme('ftm-bdf2'), ExponentialBDF2)


@pytest.mark.parametrize('frequency', [None, 0.0, 0.5])
def test_six_family_nonequilibrium_precursors_against_independent_radau(frequency):
    beta = np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273])
    decay = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
    model = (Model((1, 1, 1), (1, 1, 1)).fill(FUEL).set_boundary('reflective')
             .set_kinetics(Kinetics([2.2e5], beta, decay)))
    initial = np.r_[1.0, 0.7 * beta / (GENERATION * decay)]

    def rhs(t, state):
        rho = 0.001 * t / 0.05
        return np.r_[(rho - beta.sum()) / GENERATION * state[0]
                     + np.dot(decay, state[1:]),
                     beta / GENERATION * state[0] - decay * state[1:]]

    reference = solve_ivp(rhs, (0, 0.05), initial, method='Radau',
                          rtol=2e-12, atol=1e-13, dense_output=True)
    assert reference.success
    errors = []
    for width in (0.00125, 0.000625, 0.0003125):
        result = model.transient(0.05, width, device='cpu', time_scheme='ftm-bdf2',
            ftm_kwargs={'frequency': frequency}, step_solver='monolithic', tol_step=1e-12,
            multigroup_kwargs={'rtol': 1e-13},
            initial_precursors=(initial[1:] * GENERATION).reshape(6, 1, 1, 1),
            materials_at=lambda t: {'fuel': FUEL.replace(sigma_a=[0.03 * (1 - 0.001 * t / 0.05)])})
        expected = reference.sol(result.times)
        errors.append(max(np.max(np.abs(result.power / expected[0] - 1)),
            np.max(np.abs(result.precursors_numpy.ravel() / GENERATION / expected[1:, -1] - 1))))
    assert errors[0] / errors[1] > 3.6, errors
    assert errors[1] / errors[2] > 3.7, errors


@pytest.mark.parametrize('frequency', [None, 0.5])
def test_two_group_distinct_delayed_spectra_against_matrix_exponential(frequency):
    fission, absorption = np.array([0.005, 0.15]), np.array([0.02, 0.1])
    scatter = np.array([[0.0, 0.03], [0.002, 0.0]])
    spectrum, velocities = np.array([0.9, 0.1]), np.array([200.0, 60.0])
    beta, decay = np.array([0.005, 0.002]), np.array([0.08, 0.4])
    delayed = np.array([[0.95, 0.05], [0.8, 0.2]])
    fuel = Material(name='fuel', diffusion=[1.4, 0.4], sigma_a=absorption,
                    nu_sigma_f=fission, sigma_s=scatter, chi=spectrum)
    model = (Model((1, 1, 1), (1, 1, 1)).fill(fuel).set_boundary('reflective')
             .set_kinetics(Kinetics(velocities, beta, decay, chi_delayed=delayed)))
    # Resolve the fixed-source equations more tightly than the final balance.
    steady = model.steady(device='cpu', tol_k=1e-13, tol_source=1e-12,
                          tol_fixed_source=1e-13)
    assert steady.converged
    phi = steady.flux.ravel()
    phi = phi / (fission @ phi / steady.k_eff)
    initial = np.r_[phi, beta / decay]
    changed = fuel.replace(sigma_a=absorption * [0.98, 0.99])
    matrix = np.zeros((4, 4))
    matrix[:2, :2] = velocities[:, None] * (
        scatter.T - np.diag(changed.sigma_a + scatter.sum(axis=1))
        + np.outer(spectrum - beta @ delayed, fission / steady.k_eff))
    matrix[:2, 2:] = velocities[:, None] * delayed.T * decay
    matrix[2:, :2] = beta[:, None] * fission / steady.k_eff
    matrix[2:, 2:] = -np.diag(decay)
    expected = expm(0.2 * matrix) @ initial
    errors = []
    for width in (0.01, 0.005, 0.0025):
        result = model.transient(0.2, width, device='cpu', time_scheme='ftm-bdf2',
            initial_steady=steady, ftm_kwargs={'frequency': frequency},
            materials_at=lambda t: {'fuel': changed} if t > 0 else {},
            step_solver='monolithic', tol_step=1e-12, multigroup_kwargs={'rtol': 1e-13})
        actual = np.r_[result.flux.ravel(), result.precursors_numpy.ravel()]
        errors.append(np.max(np.abs(actual / expected - 1)))
    assert errors[0] / errors[1] > 3.7, errors
    assert errors[1] / errors[2] > 3.7, errors


def test_mapped_material_kinetics_agree_between_spatial_solvers():
    fuel = Material(name='fuel', diffusion=[1.4, 0.4], sigma_a=[0.01, 0.15],
                    nu_sigma_f=[0.007, 0.2], sigma_s=[[0, 0.01], [0, 0]])
    other = fuel.replace(name='other', sigma_a=[0.012, 0.14])
    model = (Model((4, 2, 1), (4, 2, 1)).fill(fuel).add_box(other, x=(2, 4))
             .set_boundary('reflective').set_kinetics(Kinetics(
                 [[1e7, 2e5], [0.9e7, 2.1e5]], [[0.005, 0.0025], [0.006, 0.002]],
                 [0.08, 0.4], chi_delayed=[[0.95, 0.05], [0.85, 0.15]])))
    options = dict(t_end=0.004, dt=0.001, device='cpu', time_scheme='ftm-bdf2',
        materials_at=lambda t: {'other': other.replace(sigma_a=[0.0119, 0.1399])} if t > 0 else {},
        ftm_kwargs={'frequency': 0.5}, tol_step=1e-11, max_sweeps=2000)
    groupwise = model.transient(**options)
    block = model.transient(**options, step_solver='monolithic', multigroup_kwargs={'rtol': 1e-12})
    np.testing.assert_allclose(groupwise.power, block.power, rtol=2e-8)
    np.testing.assert_allclose(groupwise.precursors_numpy, block.precursors_numpy, rtol=2e-8)


def test_stiff_shutdown_fast_mode_decays_in_the_fv_engine():
    result = run_uniform(rho=-0.5, widths=0.01, duration=0.2, frequency=0.0)
    matrix = point_matrix(-0.5)
    values, vectors = np.linalg.eig(matrix)
    fast = int(np.argmin(values))
    initial = np.array([1.0, BETA / (GENERATION * DECAY)])
    final = np.array([result.power[-1], np.sum(result.precursors_numpy) / GENERATION])
    initial_modes = np.linalg.solve(vectors, initial)
    final_modes = np.linalg.solve(vectors, final)
    assert abs(final_modes[fast] / initial_modes[fast]) < 1e-7
    assert np.min(result.power) > 0.0


@pytest.mark.parametrize('geometry', ['cartesian', 'cylindrical', 'triangular'])
def test_coupled_equilibrium_keeps_existing_thermal_exchange_and_progress(geometry):
    model = core(geometry).configure_thermal(
        {'fuel': ThermalMaterial(0.25, sink_coeff=0.02,
                                 sink_temperature=650, heat_capacity=2.5)},
        total_power=100)
    hot = model.coupled_steady(device='cpu')
    observed = []
    result = model.coupled_transient(0.04, 0.01, dt_thermal=0.02, device='cpu',
        initial_coupled_steady=hot, transient_kwargs={'time_scheme': 'ftm-bdf2'},
        on_step=lambda t, phi, power, temperature: observed.append(t))
    np.testing.assert_allclose(result.power, 1.0, atol=2e-7)
    np.testing.assert_allclose(result.temperature, hot.temperature, rtol=1e-10)
    np.testing.assert_allclose(result.step_source_weights, 0.01)
    np.testing.assert_allclose(observed, result.times[1:])


@pytest.mark.parametrize('width,frequency,previous', [(0, 0, None), (1, np.inf, None),
    (1, 1, None), (1, 0, 0), (1, 0, -1), (1, 0, np.inf)])
def test_unsafe_scalar_bdf_coefficients_are_rejected(width, frequency, previous):
    with pytest.raises(ValueError):
        ExponentialBDF2Step(width, frequency, previous)
