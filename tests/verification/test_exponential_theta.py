"""Second-order neutron FTM against independent transient solutions."""

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm

from garnet import DiffusionEigenSolver, Grid, Kinetics, Material, TransientSolver
from garnet.timescheme import ExponentialTheta


def _solve(*, dt, frequency=None, beta=0.0, precursor_fraction=None,
           scheme="ftm2", restarts=None, ftm_kwargs=None, step_solver="monolithic",
           rate_degree=1):
    velocity, absorption, decay = 2.2e5, 0.03, 0.08
    grid = Grid((1, 1, 1), (1.0, 1.0, 1.0))
    base = Material(diffusion=[1.3], sigma_a=[absorption], nu_sigma_f=[0.035])
    materials = [base]

    def problem(t):
        if precursor_fraction is not None:
            return materials, None
        # Smooth prompt rate with exact P(t)=exp(t**(rate_degree+1)).
        rate = (rate_degree+1)*t**rate_degree
        return [base.replace(sigma_a=[absorption-rate/velocity])], None

    kinetics = Kinetics(velocities=[velocity], beta=[beta], decay=[decay])
    options = ({} if scheme not in ("ftm", "ftm2") else
               {"ftm_kwargs": {"frequency": frequency, **(ftm_kwargs or {})}})
    if precursor_fraction is not None:
        options["initial_precursors"] = np.full((1, 1, 1, 1),
                                                precursor_fraction*beta/decay)
    return TransientSolver(grid, problem, kinetics, bc="reflective", device="cpu").solve(
        t_end=0.2, dt=dt, time_scheme=scheme, bdf_restart_times=restarts,
        tol_step=1e-12, step_solver=step_solver, multigroup_kwargs={"rtol": 1e-13},
        steady_kwargs={"tol_k": 1e-12, "tol_source": 1e-12}, **options)


@pytest.mark.parametrize("frequency", [None, -0.7, 0.7])
def test_smooth_prompt_problem_is_second_order(frequency):
    errors = []
    for width in (0.04, 0.02, 0.01):
        result = _solve(dt=width, frequency=frequency)
        exact = np.exp(result.times**2)
        errors.append(np.max(np.abs(result.power/exact-1)))
        assert result.time_orders == [2] * (len(result.times)-1)
    assert errors[0]/errors[1] > 3.7, errors
    assert errors[1]/errors[2] > 3.7, errors


@pytest.mark.parametrize("frequency,refinements", [(None, 0), (0.0, 0), (0.5, 0), (None, 4)])
def test_nonequilibrium_precursors_seed_the_physical_derivative(frequency, refinements):
    beta, decay, generation = 0.0065, 0.08, 1/6600
    matrix = np.array([[-beta/generation, decay], [beta/generation, -decay]])
    initial = np.array([1.0, 0.7*beta/(generation*decay)])
    errors = []
    for width in (0.005, 0.0025, 0.00125):
        result = _solve(dt=width, frequency=frequency, beta=beta, precursor_fraction=0.7,
                        ftm_kwargs={"frequency_iterations": refinements})
        exact = np.array([expm(matrix*t) @ initial for t in result.times])
        errors.append(np.max(np.abs(result.power/exact[:, 0]-1)))
        np.testing.assert_allclose(result.precursors.ravel(), exact[-1, 1]*generation,
                                   rtol=2e-5)
    assert errors[0]/errors[1] > 3.8, errors
    assert errors[1]/errors[2] > 3.8, errors


def test_zero_frequency_prompt_limit_matches_cn():
    transformed = _solve(dt=0.02, frequency=0.0)
    cn = _solve(dt=0.02, scheme="crank-nicolson")
    np.testing.assert_allclose(transformed.power, cn.power, rtol=2e-11)


def test_unsafe_frequency_uses_actual_be_and_reports_first_order():
    transformed = _solve(dt=0.02, frequency=100.0, beta=0.0065, precursor_fraction=0.7)
    be = _solve(dt=0.02, scheme="backward-euler", beta=0.0065, precursor_fraction=0.7)
    np.testing.assert_allclose(transformed.power, be.power, rtol=2e-11)
    np.testing.assert_allclose(transformed.precursors, be.precursors, rtol=2e-11)
    assert transformed.time_orders == [1] * 10
    assert transformed.ftm_fallback_reasons == ["frequency-bound"] * 10


def test_nonuniform_steps_use_accepted_frequency_and_endpoint_heat_weight():
    widths = [0.02, 0.04, 0.01, 0.03, 0.02, 0.04, 0.01, 0.03]
    result = _solve(dt=widths)
    np.testing.assert_allclose(result.step_frequencies[1:],
                               np.diff(np.log(result.power[:-1]))/widths[:-1],
                               rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(result.step_source_weights, widths)
    # Smoothly nonuniform refinement must retain second order too.
    errors = []
    for subdivisions in (1, 2, 4):
        schedule = [h/subdivisions for h in widths for _ in range(subdivisions)]
        run = _solve(dt=schedule, frequency=0.7)
        errors.append(np.max(np.abs(run.power/np.exp(run.times**2)-1)))
    assert errors[0]/errors[1] > 3.7, errors
    assert errors[1]/errors[2] > 3.7, errors


def test_restart_reseeds_the_accepted_physical_rhs():
    original = _solve(dt=0.02, frequency=0.7, beta=0.0065)
    restarted = _solve(dt=0.02, frequency=0.7, beta=0.0065, restarts=[0.1])
    np.testing.assert_allclose(restarted.power, original.power, rtol=3e-11)
    np.testing.assert_allclose(restarted.precursors, original.precursors, rtol=3e-11)


def test_precursor_interpolation_uses_both_endpoint_production_products():
    beta_old = np.array([0.0065])
    scheme = ExponentialTheta(frequency=0.7)
    scheme.start([np.array([1.0])])
    scheme.seed_derivative([np.array([0.0])], [beta_old])
    beta_old[:] = 0.9  # Accepted history must be an independent snapshot.
    scheme.prepare_step(1, 0.2)
    old_C, old_S, new_S, new_beta, decay = 0.08, 1.2, 1.5, 0.008, 0.4
    actual = scheme.precursor_update(1, [[np.array([old_C])]], np.array([new_S]),
                                      np.array([old_S]), [new_beta], [decay], 0.2)[0]

    def integrand(t):
        production = (0.0065*old_S*(1-t/0.2)
                      + np.exp(-0.7*0.2)*new_beta*new_S*t/0.2)
        return np.exp(-decay*(0.2-t)+0.7*t)*production

    expected = np.exp(-decay*0.2)*old_C + quad(integrand, 0, 0.2, epsabs=1e-14)[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-14)


def test_multigroup_changing_fission_source_against_independent_ode():
    grid = Grid((1, 1, 1), (1.0, 1.0, 1.0))
    absorption, fission = np.array([0.02, 0.1]), np.array([0.005, 0.15])
    scatter = np.array([[0.0, 0.03], [0.002, 0.0]])
    spectrum = np.array([0.9, 0.1])
    velocities = np.array([200.0, 60.0])
    beta, decay = np.array([0.005, 0.002]), np.array([0.08, 0.4])
    delayed = np.array([[0.95, 0.05], [0.8, 0.2]])
    base = Material(diffusion=[1.4, 0.4], sigma_a=absorption, nu_sigma_f=fission,
                    sigma_s=scatter, chi=spectrum)

    def coefficients(t):
        return absorption*(1-np.array([0.04, 0.02])*t), fission*(1+np.array([0.02, -0.01])*t)

    def problem(t):
        sigma_a, nu_f = coefficients(t)
        return [base.replace(sigma_a=sigma_a, nu_sigma_f=nu_f)], None

    # Resolve the fixed-source equations more tightly than the final balance.
    steady = DiffusionEigenSolver(grid, [base], bc="reflective", device="cpu").solve(
        tol_k=1e-13, tol_source=1e-12, tol_fixed_source=1e-13)
    assert steady.converged
    phi0 = steady.flux[:, 0, 0, 0]
    phi0 = phi0/(np.dot(fission, phi0)/steady.k_eff)
    prompt = spectrum-beta @ delayed

    def rhs(t, state):
        sigma_a, nu_f = coefficients(t)
        phi, precursors = state[:2], state[2:]
        source = np.dot(nu_f, phi)/steady.k_eff
        neutron = (-(sigma_a+scatter.sum(axis=1))*phi+scatter.T @ phi
                   + prompt*source + (decay*precursors) @ delayed)
        return np.r_[velocities*neutron, beta*source-decay*precursors]

    initial = np.r_[phi0, beta/decay]
    reference = solve_ivp(rhs, (0, 0.2), initial, method="Radau", dense_output=True,
                          rtol=2e-12, atol=1e-13)
    assert reference.success
    kinetics = Kinetics(velocities=velocities, beta=beta, decay=decay, chi_delayed=delayed)
    errors = []
    for width in (0.02, 0.01, 0.005):
        result = TransientSolver(grid, problem, kinetics, bc="reflective", device="cpu").solve(
            t_end=0.2, dt=width, time_scheme="ftm2", initial_steady=steady,
            step_solver="monolithic", multigroup_kwargs={"rtol": 1e-13}, tol_step=1e-12)
        exact = reference.sol(0.2)
        actual = np.r_[result.flux.ravel(), result.precursors.ravel()]
        errors.append(np.max(np.abs(actual/exact-1)))
    assert errors[0]/errors[1] > 3.8, errors
    assert errors[1]/errors[2] > 3.8, errors


def test_startup_uses_two_actual_be_steps_and_restarts_at_declared_events():
    result = _solve(dt=0.02, beta=0.0065, restarts=[0.1],
                    ftm_kwargs={"startup_be_steps": 2})
    be = _solve(dt=0.02, beta=0.0065, scheme="backward-euler")
    np.testing.assert_allclose(result.power[:3], be.power[:3], rtol=2e-11)
    np.testing.assert_allclose(result.step_source_weights, 0.02)
    assert result.time_orders == [1, 1, 2, 2, 2, 1, 1, 2, 2, 2]
    assert [i for i, reason in enumerate(result.ftm_fallback_reasons)
            if reason == "startup-backward-euler"] == [0, 1, 5, 6]


@pytest.mark.parametrize("scheme", ["ftm", "ftm2"])
def test_frequency_refinement_counts_all_solves_and_reports_the_final_defect(scheme):
    result = _solve(dt=0.02, scheme=scheme,
                    ftm_kwargs={"frequency_iterations": 4, "frequency_tolerance": 1e-9})
    assert sum(result.step_frequency_refinements) > 0
    assert sum(result.feedback_iterations) == 10+sum(result.step_frequency_refinements)
    assert len(result.times) == 11
    observed = np.diff(np.log(result.power))/0.02
    defects = 0.02*np.abs(observed-np.asarray(result.step_frequencies))
    np.testing.assert_allclose(result.step_frequency_log_defects, defects, atol=2e-15)
    for count, defect in zip(result.step_frequency_refinements, defects):
        assert count <= 4
        assert defect <= 1e-9+2e-15 or count == 4


def test_capped_frequency_refinement_does_not_claim_convergence():
    result = _solve(dt=0.02, ftm_kwargs={"frequency_iterations": 1,
                    "frequency_relaxation": 0.5, "frequency_tolerance": 1e-14})
    assert max(result.step_frequency_log_defects) > 1e-8
    assert result.step_frequency_refinements == [1]*10


def test_frequency_bound_during_refinement_resolves_with_actual_be():
    options = {"frequency_iterations": 4, "max_exponent": 0.01}
    result = _solve(dt=0.05, beta=0.0065, precursor_fraction=0.7, ftm_kwargs=options)
    be = _solve(dt=0.05, beta=0.0065, precursor_fraction=0.7, scheme="backward-euler")
    assert result.step_frequency_refinements[0] == 1
    assert result.ftm_fallback_reasons[0] == "frequency-bound"
    assert result.time_orders[0] == 1
    assert result.step_frequency_log_defects[0] is None
    assert result.power[1] == pytest.approx(be.power[1], rel=2e-11)


def test_corrected_ftm2_keeps_second_order_on_smooth_problems():
    errors = []
    for width in (0.04, 0.02, 0.01):
        result = _solve(dt=width, rate_degree=2,
                        ftm_kwargs={"frequency_iterations": 4, "frequency_tolerance": 1e-10})
        errors.append(np.max(np.abs(result.power/np.exp(result.times**3)-1)))
    assert errors[0]/errors[1] > 3.7, errors
    assert errors[1]/errors[2] > 3.7, errors


def test_converged_frequency_is_exact_for_a_linear_scalar_growth_rate():
    # Here the fixed point enforces log(P_new/P_old)/h=(a_new+a_old)/2.
    # The trapezoidal growth-rate integral is exact for a(t)=2*t.
    result = _solve(dt=0.04, ftm_kwargs={"frequency_iterations": 8,
                                        "frequency_tolerance": 1e-12})
    np.testing.assert_allclose(result.power, np.exp(result.times**2), rtol=2e-11)


@pytest.mark.parametrize("options", [
    {"startup_be_steps": -1}, {"startup_be_steps": 0.5}, {"startup_be_steps": np.inf},
    {"startup_be_steps": True}, {"frequency_iterations": -1}, {"frequency_iterations": 0.5},
    {"frequency_iterations": np.nan}, {"frequency_iterations": True},
    {"frequency_iterations": 1, "frequency": 0.5}, {"frequency_tolerance": 0},
    {"frequency_tolerance": np.nan}, {"frequency_relaxation": 0}, {"frequency_relaxation": 1.1}])
def test_invalid_startup_and_refinement_controls_are_refused(options):
    with pytest.raises(ValueError):
        ExponentialTheta(**options)


def test_frequency_refinement_requires_monolithic_steps():
    with pytest.raises(ValueError, match="requires monolithic"):
        _solve(dt=0.02, step_solver="fixed-point", ftm_kwargs={"frequency_iterations": 1})


def test_implicit_feedback_with_frequency_refinement_is_explicitly_refused():
    grid = Grid((1, 1, 1), (1.0, 1.0, 1.0))
    fuel = Material(diffusion=[1.3], sigma_a=[0.03], nu_sigma_f=[0.035])
    kin = Kinetics(velocities=[2.2e5], beta=[0.0065], decay=[0.08])
    solver = TransientSolver(grid, lambda t: ([fuel], None), kin, device="cpu",
                             xs_update_at=lambda t: None,
                             feedback_iteration=lambda *args: True)
    with pytest.raises(ValueError, match="implicit feedback"):
        solver.solve(t_end=0.1, dt=0.01, time_scheme="ftm2", step_solver="monolithic",
                     ftm_kwargs={"frequency_iterations": 1})


def test_aligned_interior_prompt_event_preserves_continuity_and_converges():
    grid = Grid((1, 1, 1), (1.0, 1.0, 1.0))
    fuel = Material(diffusion=[1.3], sigma_a=[0.03], nu_sigma_f=[0.035])
    initial, changed = [fuel], [fuel.replace(sigma_a=[0.03*(1-0.002)])]
    # The event endpoint is the old-material limit; the next interval uses
    # the inserted material. The flux and inventories stay continuous.
    event = 0.1
    problem = lambda t: (initial if t <= event else changed, None)
    beta, decay, generation = 0.0065, 0.08, 1/6600
    kin = Kinetics(velocities=[2.2e5], beta=[beta], decay=[decay])
    matrix = np.array([[(0.002-beta)/generation, decay], [beta/generation, -decay]])
    reference_initial = np.array([1.0, beta/(generation*decay)])
    errors = []
    for width in (0.001, 0.0005, 0.00025):
        result = TransientSolver(grid, problem, kin, bc="reflective", device="cpu").solve(
            t_end=0.2, dt=width, time_scheme="ftm2", ftm_kwargs={"startup_be_steps": 2},
            bdf_restart_times=[event], step_solver="monolithic", tol_step=1e-12,
            multigroup_kwargs={"rtol": 1e-13})
        exact = np.array([(expm(matrix*max(0.0, t-event)) @ reference_initial)[0]
                          for t in result.times])
        errors.append(np.max(np.abs(result.power/exact-1)))
        index = np.argmin(np.abs(result.times-event))
        assert result.power[index] == pytest.approx(1.0, rel=2e-11)
        assert result.time_orders[index:index+2] == [1, 1]
        assert result.ftm_fallback_reasons.count("startup-backward-euler") == 4
    assert errors[0]/errors[1] > 3.7, errors
    assert errors[1]/errors[2] > 3.7, errors
