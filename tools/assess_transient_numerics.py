"""Reproduce analytic probes for the transient-method development plans.

Run from the repository root::

    python3 tools/assess_transient_numerics.py \
        --output benchmark-results/transient-numerics-assessment/analytic-probes.json

Standard-library only. These are independent point-kinetics and norm-formula
calculations, not executions or performance measurements of GARNET. The FTM
probe uses lagged neutron-population frequency and a constant transformed
end-of-step fission source for precursor integration. It is one possible
first-order prototype, not a reproduction of all SIMULATE algorithms.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import platform
import subprocess


BETA = 0.0065
DECAY = 0.08
GENERATION_TIME = 1e-4
REACTIVITY = 0.002


def coefficients():
    """Matrix of dn/dt = a*n + b*C, dC/dt = c*n + d*C."""
    return ((REACTIVITY - BETA) / GENERATION_TIME, DECAY,
            BETA / GENERATION_TIME, -DECAY)


def eigenvalues():
    a, b, c, d = coefficients()
    gap = math.sqrt((a - d)**2 + 4 * b * c)
    return (a + d + gap) / 2, (a + d - gap) / 2


def exact_state(t):
    """Closed-form response to a reactivity step from equilibrium at t=0."""
    a, b, _, _ = coefficients()
    slow, fast = eigenvalues()
    C0 = BETA / (GENERATION_TIME * DECAY)
    derivative0 = a + b * C0
    slow_weight = (derivative0 - fast) / (slow - fast)
    n_slow = slow_weight * math.exp(slow * t)
    n_fast = (1 - slow_weight) * math.exp(fast * t)
    population = n_slow + n_fast
    derivative = slow * n_slow + fast * n_fast
    return population, (derivative - a * population) / b


def precursor_integral(width, frequency):
    """Integral H = h*phi_1(-(lambda+omega)*h), including its finite limit."""
    z = (DECAY + frequency) * width
    if abs(z) < 1e-7:
        return width * (1 - z / 2 + z*z / 6 - z**3 / 24 + z**4 / 120)
    return width * (-math.expm1(-z) / z)


def step(method, state, width, frequency=0.0):
    n, precursor = state
    a, b, c, d = coefficients()
    if method == "backward-euler":
        determinant = (1 - width*a) * (1 - width*d) - width*width*b*c
        return (((1 - width*d)*n + width*b*precursor) / determinant,
                (width*c*n + (1 - width*a)*precursor) / determinant)
    if method != "lagged-ftm":
        raise ValueError(method)
    integral = precursor_integral(width, frequency)
    surviving = math.exp(-DECAY * width) * precursor
    new_n = ((math.exp(frequency * width) * n / width + DECAY * surviving)
             / (1 / width + frequency - a - DECAY * c * integral))
    return new_n, surviving + c * integral * new_n


def probe_window(method, width, start, end):
    """Sample relative population error at this method's step endpoints.

    The tail window is deliberately initialized with exact state and one
    exact PREVIOUS population sample. This isolates tail behavior and excludes
    any error inherited from resolving the initial prompt jump.
    """
    count = round((end - start) / width)
    assert math.isclose(count * width, end - start, abs_tol=1e-12)
    state = exact_state(start)
    previous = exact_state(start - width)[0] if start > 0 else None
    maximum = 0.0
    for k in range(count):
        frequency = (0.0 if previous is None else
                     math.log(state[0] / previous) / width)
        previous = state[0]
        state = step(method, state, width, frequency)
        assert all(math.isfinite(value) and value > 0 for value in state)
        reference = exact_state(start + (k + 1) * width)[0]
        maximum = max(maximum, abs(state[0] / reference - 1))
    return {"steps": count, "max_endpoint_relative_population_error": maximum}


def ratio_of_norms(corrected, predicted, rtol):
    """Scalar-component specialization of timescheme._error_norm_sums."""
    return math.sqrt(
        sum((c - p)**2 for c, p in zip(corrected, predicted))
        / sum((rtol * c)**2 for c in corrected))


def norm_probes():
    out = []
    for multiplier in (1.0, 1000.0):
        corrected = (1.0, multiplier)
        predicted = (0.99, multiplier)
        value = ratio_of_norms(corrected, predicted, 1e-3)
        out.append({"precursor_unit_multiplier": multiplier,
                    "flux_relative_defect": 0.01,
                    "precursor_defect": 0.0,
                    "rtol": 1e-3, "combined_norm": value,
                    "passes_norm_test": value <= 1.0})
    return out


def self_check():
    n, precursor = exact_state(0.0)
    assert math.isclose(n, 1.0, rel_tol=1e-13)
    assert math.isclose(precursor, BETA / (GENERATION_TIME * DECAY), rel_tol=1e-13)
    assert precursor_integral(0.1, -DECAY) == 0.1
    slow, _ = eigenvalues()
    state = (1.0, (BETA / GENERATION_TIME) / (slow + DECAY))
    next_state = step("lagged-ftm", state, 0.1, slow)
    for old, new in zip(state, next_state):
        assert math.isclose(new, math.exp(slow * 0.1) * old, rel_tol=1e-13)
    raw = ratio_of_norms((1.0, 1.0), (0.99, 1.0), 1e-3)
    common_scale = ratio_of_norms((1e6, 1e6), (0.99e6, 1e6), 1e-3)
    assert math.isclose(raw, common_scale, rel_tol=1e-13)
    probes = norm_probes()
    assert not probes[0]["passes_norm_test"] and probes[1]["passes_norm_test"]


def assessment():
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
        check=True, capture_output=True, text=True).stdout.strip()
    windows = []
    for name, start, end, widths in (
            ("prompt_jump_and_tail", 0.0, 10.0, (0.02, 0.01, 0.005, 0.001)),
            ("tail_with_exact_initial_state_and_previous_sample", 0.5, 10.0,
             (0.1, 0.05, 0.02))):
        windows.append({
            "name": name, "start_s": start, "end_s": end,
            "rows": [{"width_s": width,
                      **{method: probe_window(method, width, start, end)
                         for method in ("backward-euler", "lagged-ftm")}}
                     for width in widths]})
    return {
        "scope": "Independent analytic probes; no GARNET solver or timing measurements",
        "source_revision": revision,
        "python": platform.python_version(),
        "available_packages": {name: importlib.util.find_spec(name) is not None
                               for name in ("numpy", "scipy", "cupy")},
        "point_kinetics": {
            "beta": BETA, "decay_per_s": DECAY,
            "generation_time_s": GENERATION_TIME, "reactivity": REACTIVITY,
            "eigenvalues_per_s": eigenvalues(), "windows": windows,
            "limitations": [
                "One delayed family, fixed coefficients, no spatial shape change or feedback",
                "Fixed widths and errors sampled at their own endpoints, no adaptive controller",
                "Tail starts from exact state/history; does not include prompt-jump integration error",
                "FTM includes exponential precursor integration; zero frequency is not BE precursors",
                "No BDF or IQS comparison and no CPU/GPU speedup is measured"]},
        "raw_error_norm_unit_sensitivity": norm_probes(),
        "illustrative_cost_model": [
            {"neutronics_fraction": 0.97,
             "attempt_count_reduction": reduction,
             "cost_per_attempt_multiplier": 1.2,
             "hypothetical_total_speedup": 1 / (0.03 + 0.97 * 1.2 / reduction)}
            for reduction in (1, 2, 4)],
        "self_checks": "passed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    self_check()
    rendered = json.dumps(assessment(), indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(f"Analytic probes and self-checks written to {args.output}")
