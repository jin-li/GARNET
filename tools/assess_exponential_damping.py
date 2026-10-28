"""Independent kinetics gate for a damped second-order frequency transform.

    python tools/assess_exponential_damping.py

The SDIRK2 research driver solves small dense neutron/precursor systems. It
does not run spatial diffusion or thermal integration. Its implicit stage
counts are prospective spatial work, not measured FV/GPU costs. BE, FTM2 and
BDF controls below run the actual one-cell diffusion API on the same physics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy

from garnet import Kinetics, Material, Model
from garnet._exponential import (ExponentialBDF2Step, ExponentialSDIRK2Step,
                                 population_frequency)


ROOT = Path(__file__).resolve().parents[1]
CASES = {"prompt-rise": (0.002, 1.0, 0.0),
         "superprompt-rise": (0.01, 0.5, 0.0),
         "shutdown": (-0.05, 0.2, 0.0),
         "interior-shutdown": (-0.05, 0.25, 0.05)}
BETA, DECAY, GENERATION = 0.0065, 0.08, 1.0 / 6600.0


@dataclass
class DampingResult:
    times: np.ndarray
    states: np.ndarray
    frequencies: list
    fallback_reasons: list
    implicit_solves: int
    minimum_stage_component: float


def integrate_damped(matrix_at, initial, widths, *, frequency=None,
                     restarts=(), max_exponent=0.5, population=None,
                     method="sdirk2"):
    """Transformed SDIRK2/BDF2 for small linear kinetics systems only.

    Both neutron and precursor unknowns are transformed. Matrix coefficients
    are sampled at the two RK stage times. Only completed macro steps enter
    population history; the predictor uses the previous accepted width.
    Unsafe or nonpositive population histories use actual one-step BE.
    There is no time adaptation, positivity correction or dense output.
    """
    if method not in ("sdirk2", "bdf2"):
        raise ValueError("method must be sdirk2 or bdf2")
    widths = np.asarray(widths, dtype=float)
    state = np.asarray(initial, dtype=float).copy()
    if (widths.ndim != 1 or not widths.size or not np.all(np.isfinite(widths))
            or np.any(widths <= 0)):
        raise ValueError("widths must be a nonempty vector of finite positive values")
    if state.ndim != 1 or not state.size or not np.all(np.isfinite(state)):
        raise ValueError("initial state must be a finite nonempty vector")
    if frequency is not None and not np.isfinite(frequency):
        raise ValueError("frequency must be finite")
    if not np.isfinite(max_exponent) or not 0 < max_exponent < 1:
        raise ValueError("max_exponent must lie strictly between zero and one")
    times = np.r_[0.0, np.cumsum(widths)]
    events = np.asarray(restarts, dtype=float)
    if (events.ndim != 1 or not np.all(np.isfinite(events))
            or np.any(events <= 0) or np.any(events >= times[-1])):
        raise ValueError("restarts must be finite interior macro endpoints")
    for event in events:
        if not np.any(np.isclose(times[1:-1], event, rtol=1e-12, atol=1e-14)):
            raise ValueError("restarts must align with macro endpoints")
    population = (lambda y: float(y[0])) if population is None else population
    identity = np.eye(state.size)

    def matrix(t):
        value = np.asarray(matrix_at(float(t)), dtype=float)
        if value.shape != identity.shape or not np.all(np.isfinite(value)):
            raise ValueError("matrix_at must return a finite square matrix matching the state")
        return value

    states, frequencies, reasons = [state.copy()], [], []
    previous_population = previous_width = older = None
    current_population = float(population(state))
    solves, minimum = 0, float(np.min(state))
    for index, width in enumerate(widths):
        if np.any(np.isclose(times[index], events, rtol=1e-12, atol=1e-14)):
            previous_population = previous_width = older = None
        reason, omega = "", frequency
        if omega is None:
            omega = 0.0
            if previous_population is not None:
                try:
                    omega = population_frequency(current_population, previous_population,
                                                 previous_width)
                except ValueError:
                    reason = "invalid-population-history"
        if not np.isfinite(omega * width) or abs(omega * width) > max_exponent:
            reason = "frequency-bound"
        if method == 'bdf2' and previous_width is not None and not reason:
            if width / previous_width > 1 + np.sqrt(2):
                reason = "step-ratio-bound"
            elif abs(omega * (width + previous_width)) > 2 * max_exponent:
                reason = "history-frequency-bound"
        accepted = state.copy()
        if reason:
            omega = 0.0
            state = np.linalg.solve(identity - width * matrix(times[index + 1]), state)
            solves += 1
            minimum = min(minimum, float(np.min(state)))
        elif method == 'sdirk2':
            step = ExponentialSDIRK2Step(float(width), float(omega))
            first = np.linalg.solve(
                step.diagonal_rate * identity
                - matrix(times[index] + step.stage_fraction(1) * width),
                step.history(1, state) / step.stage_width)
            state = np.linalg.solve(
                step.diagonal_rate * identity - matrix(times[index + 1]),
                step.history(2, state, first) / step.stage_width)
            solves += 2
            minimum = min(minimum, float(np.min(first)), float(np.min(state)))
        else:
            step = ExponentialBDF2Step(float(width), float(omega), previous_width)
            state = np.linalg.solve(step.diagonal_factor * identity
                                    - width * matrix(times[index + 1]),
                                    step.history(state, older))
            solves += 1
            minimum = min(minimum, float(np.min(state)))
        if not np.all(np.isfinite(state)):
            raise RuntimeError("nonfinite kinetic state")
        previous_population, current_population = current_population, float(population(state))
        previous_width = float(width)
        older = accepted
        states.append(state.copy())
        frequencies.append(float(omega))
        reasons.append(reason)
    return DampingResult(times, np.asarray(states), frequencies, reasons, solves, minimum)


def widths_for(duration, maximum, *, graded=False, event=0.0):
    """Prescribed mesh independent of the reference; split and regrade at events."""
    endpoints, time, width = [], 0.0, 1e-5 if graded else maximum
    while time < duration - 1e-14:
        end = min(time + min(width, maximum), duration)
        if time < event < end or np.isclose(end, event, rtol=1e-12, atol=1e-14):
            end = event
        endpoints.append(end)
        time = end
        if graded:
            width = 1e-5 if time == event else min(width * 1.1, maximum)
    return np.diff(np.r_[0.0, endpoints])


def point_matrix(rho):
    return np.array([[(rho - BETA) / GENERATION, DECAY],
                     [BETA / GENERATION, -DECAY]])


def problem(name):
    rho, duration, event = CASES[name]
    initial = np.array([1.0, BETA / (GENERATION * DECAY)])
    after = point_matrix(rho)
    values, vectors = np.linalg.eig(after)
    amplitudes = np.linalg.solve(vectors, initial)

    def reference(times):
        elapsed = np.maximum(np.asarray(times) - event, 0.0)
        return (np.exp(elapsed[..., None] * values) * amplitudes) @ vectors.T

    before = point_matrix(0.0)
    return initial, lambda t: before if t <= event else after, reference


def configurations():
    rows = []
    for graded in (False, True):
        for width in ((0.01, 0.005, 0.002, 0.001) if not graded else (0.01, 0.002, 0.001)):
            for method in ("sdirk2-zero", "sdirk2-lagged", "transformed-bdf2-zero",
                           "transformed-bdf2-lagged", "ftm2-zero", "ftm2-lagged"):
                rows.append({"method": method, "dt": width, "graded": graded})
    rows.extend({"method": "backward-euler", "dt": h, "graded": graded}
                for h, graded in ((0.001, False), (0.01, True), (0.002, True)))
    rows.extend({"method": "adaptive-bdf", "order": order, "rtol": rtol, "dt": 1e-5}
                for order, rtol in ((1, 1e-6), (5, 1e-4), (5, 1e-5), (5, 1e-7)))
    return rows


def case(name):
    rho, duration, event = CASES[name]
    initial, matrix_at, reference = problem(name)
    fuel = Material(name="fuel", diffusion=[1.3], sigma_a=[0.03], nu_sigma_f=[0.035])
    changed = fuel.replace(sigma_a=[0.03 * (1 - rho)])
    core = (Model(size=(1, 1, 1), cells=(1, 1, 1)).fill(fuel)
            .set_boundary("reflective").set_kinetics(
                Kinetics(velocities=[2.2e5], beta=[BETA], decay=[DECAY])))
    steady = core.steady(device="cpu", tol_k=1e-12, tol_source=1e-11)
    sample = np.linspace(0, duration, 8001)
    expected = reference(sample)[:, 0]
    records = []
    for config in configurations():
        method = config["method"]
        widths = widths_for(duration, config["dt"], graded=config.get("graded", False), event=event)
        if method.startswith("sdirk2"):
            result = integrate_damped(matrix_at, initial, widths,
                frequency=0.0 if method.endswith("zero") else None,
                restarts=[event] if event else [])
            times, power = result.times, result.states[:, 0]
            precursor = result.states[-1, 1:]
            solves = result.implicit_solves
            row = {"implementation": "dense research kinetics; prospective FV work only",
                   "rejected_steps": 0, "fallback_steps": sum(bool(r) for r in result.fallback_reasons),
                   "minimum_stage_component": result.minimum_stage_component,
                   "krylov_iterations": None, "fgmres_iterations": None,
                   "operator_rebuilds": None}
        else:
            options = {}
            if method.startswith("ftm2"):
                options['ftm_kwargs'] = {"startup_be_steps": 2}
                if method.endswith("zero"):
                    options['ftm_kwargs']['frequency'] = 0.0
                scheme, dt = "ftm2", widths
            elif method.startswith('transformed-bdf2'):
                scheme, dt = 'ftm-bdf2', widths
                options['ftm_kwargs'] = {'frequency': 0.0} if method.endswith('zero') else {}
            elif method == "adaptive-bdf":
                scheme, dt = f"bdf{config['order']}", config['dt']
                options['adaptive_bdf'] = {"rtol": config['rtol'], "max_dt": 0.02,
                    "min_dt": 1e-11, "max_rejections": 40,
                    "automatic_order": config['order'] > 1}
            else:
                scheme, dt = method, widths
            if event:
                options['bdf_restart_times'] = [event]
            result = core.transient(duration, dt, device="cpu", initial_steady=steady,
                materials_at=lambda t: {"fuel": changed} if t > event else {},
                time_scheme=scheme, step_solver="monolithic", tol_step=1e-10,
                multigroup_kwargs={"rtol": 1e-11}, **options)
            times, power = result.times, result.power
            precursor = result.precursors_numpy.ravel() / GENERATION
            solves = sum(result.feedback_iterations) + sum(result.rejected_feedback_iterations)
            row = {"implementation": "one-cell finite-volume diffusion through public API",
                   "rejected_steps": result.rejected_steps,
                   "fallback_steps": sum(bool(r) for r in result.ftm_fallback_reasons),
                   "minimum_stage_component": None,
                   "krylov_iterations": result.total_inner_iterations,
                   "fgmres_iterations": sum(result.step_iterations) + sum(result.rejected_step_iterations),
                   "operator_rebuilds": result.operator_rebuilds}
        endpoint_error = float(np.max(np.abs(power / reference(times)[:, 0] - 1)))
        output_error = float(np.max(np.abs(np.interp(sample, times, power) / expected - 1)))
        differences = np.diff(power) * np.sign(rho)
        wrong_direction = int(np.count_nonzero(
            differences < -1e-12 * np.maximum(np.abs(power[:-1]), np.abs(power[1:]))))
        row.update(configuration=config, implicit_solves=int(solves), accepted_steps=len(times) - 1,
            max_endpoint_relative_power_error=endpoint_error,
            max_common_grid_relative_power_error=output_error,
            max_relative_power_error=max(endpoint_error, output_error),
            final_relative_precursor_error=float(np.max(np.abs(precursor / reference(times[-1])[1:] - 1))),
            minimum_power=float(np.min(power)), wrong_direction_increments=wrong_direction)
        records.append(row)
        print(name, method, config.get('rtol', config['dt']), 'graded' if config.get('graded') else '',
              f"error={100 * row['max_relative_power_error']:.5f}% solves={solves}", flush=True)
    budgets = {}
    for budget in (0.01, 0.001):
        choices = {}
        for family in ("sdirk2-zero", "sdirk2-lagged", "transformed-bdf2-zero",
                       "transformed-bdf2-lagged", "ftm2-zero", "ftm2-lagged",
                       "backward-euler", "adaptive-bdf"):
            qualifying = [r for r in records if r['configuration']['method'] == family
                and r['max_relative_power_error'] <= budget
                and r['final_relative_precursor_error'] <= budget
                and r['minimum_power'] >= 0 and r['wrong_direction_increments'] == 0]
            best = min(qualifying, key=lambda r: r['implicit_solves']) if qualifying else None
            choices[family] = best
        budgets[str(budget)] = choices
    return {"reactivity": rho, "duration_s": duration, "event_s": event,
            "reference": "Independent eigen-decomposition of the physical two-mode kinetics matrix",
            "records": records, "fewest_tested_solves_at_budget": budgets}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', choices=list(CASES), default=list(CASES))
    parser.add_argument('--output', type=Path, default=ROOT / (
        'benchmark-results/exponential-second-order/chunk3-damping.json'))
    args = parser.parse_args()
    sources = ['garnet/_exponential.py', 'garnet/timescheme.py', 'garnet/transient.py',
               'garnet/model.py', 'tools/assess_exponential_damping.py']
    report = {"date": "2026-10-26", "scope": "Independent neutron/precursor damping gate; no thermal transform",
        "implementation_status": "SDIRK2 is dense kinetics only; transformed BDF2, FTM2 and baseline controls use actual FV through the model API",
        "environment": {"python": platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        "beta": BETA, 'decay_per_s': DECAY, 'generation_time_s': GENERATION,
        "graded_mesh": {"initial_width_s": 1e-5, "growth": 1.1, "adaptive": False},
        "adaptive_controls": {"initial_dt_s": 1e-5, "max_dt_s": 0.02,
            "min_dt_s": 1e-11, "max_rejections": 40,
            "reason_for_rejection_cap": "Tight BDF control at an interior shutdown needs more than 20 consecutive halvings after the restart"},
        "error_sampling": "Maximum of own endpoints and common 8001-point linear-interpolation grid; final precursor error also gated",
        "work_comparison": "Two dense SDIRK2 stages are prospective spatial solves; transformed BDF2 and controls count actual FV endpoint solves, including rejected attempts. No runtime comparison.",
        "budget_filter": "Power and final precursor errors within budget; nonnegative power; no wrong-direction macro increments",
        "timing_claim": None, "reference_doi": "10.1137/0714068",
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        'cases': {}}
    for name in args.cases:
        report['cases'][name] = case(name)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
