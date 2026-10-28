"""Small, auditable static branch-XS validation helpers.

All reactivities use rho = 1 - 1/k. Uncertainties supplied here are absolute
one-standard-error estimates, never Serpent relative errors. Statistical
screening uses normal intervals, not a claim of confidence from repeated runs.
Independent endpoint estimates are assumed explicitly where errors propagate.
No function fits curves, clips physical inputs, or infers missing coverage.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _number(value, name, *, positive=False, nonnegative=False):
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else " and nonnegative" if nonnegative else ""))
    return result


def reactivity(k: float) -> float:
    """Return dimensionless reactivity for a positive finite eigenvalue."""
    return 1.0 - 1.0 / _number(k, "k", positive=True)


def reactivity_change(k_from, k_to, *, k_from_standard_error=0.0,
                      k_to_standard_error=0.0):
    """Signed change in pcm; first-order independent-endpoint error propagation."""
    k0 = _number(k_from, "k_from", positive=True)
    k1 = _number(k_to, "k_to", positive=True)
    s0 = _number(k_from_standard_error, "k_from_standard_error", nonnegative=True)
    s1 = _number(k_to_standard_error, "k_to_standard_error", nonnegative=True)
    return {"value_pcm": 1e5 * (reactivity(k1) - reactivity(k0)),
            "standard_error_pcm": 1e5 * math.hypot(s0 / k0**2, s1 / k1**2),
            "uncertainty_assumption": "independent endpoint k estimates; first-order propagation"}


def centered_temperature_coefficient(k_lower, k_upper, t_lower, t_upper, *,
                                     k_lower_standard_error=0.0,
                                     k_upper_standard_error=0.0):
    """Secant coefficient in pcm/K using actual endpoint separation.

    Call this a centered coefficient only when endpoints bracket the intended
    reference temperature symmetrically. No unprovided nominal T is inferred.
    """
    lo = _number(t_lower, "t_lower", positive=True)
    hi = _number(t_upper, "t_upper", positive=True)
    if hi <= lo:
        raise ValueError("t_upper must exceed t_lower")
    delta = reactivity_change(k_lower, k_upper,
        k_from_standard_error=k_lower_standard_error,
        k_to_standard_error=k_upper_standard_error)
    return {"value_pcm_per_K": delta["value_pcm"] / (hi - lo),
            "standard_error_pcm_per_K": delta["standard_error_pcm"] / (hi - lo),
            "temperature_midpoint_K": (hi + lo) / 2,
            "temperature_separation_K": hi - lo,
            "uncertainty_assumption": delta["uncertainty_assumption"]}


def _screen(difference, standard_error, target, confidence_z):
    z = _number(confidence_z, "confidence_z", positive=True)
    if target is None:
        return {"status": "descriptive", "signed_difference": float(difference),
                "absolute_difference": float(abs(difference)),
                "standard_error": float(standard_error), "target": None,
                "confidence_z": z}
    target = _number(target, "target", nonnegative=True)
    uncertainty = z * standard_error
    status = "pass" if abs(difference) + uncertainty <= target else (
        "fail" if abs(difference) - uncertainty > target else "inconclusive")
    return {"status": status, "signed_difference": float(difference),
            "absolute_difference": float(abs(difference)),
            "standard_error": float(standard_error), "target": target,
            "confidence_z": z}


def coefficient_convergence(full, half, *, relative_target=0.05, confidence_z=1.96):
    """Compare independently sampled full/half-step coefficient records.

    Pass requires the entire discrepancy interval below a conservative lower
    bound on the relative tolerance. Weak/unresolved reference coefficients are
    inconclusive. Shared-endpoint/correlated coefficient estimates are not
    supported: recompute with their covariance outside this helper.
    ``relative_target=None`` reports descriptive discrepancies with no pass.
    If records include endpoint geometry, both must have the same midpoint and a 2:1
    full/half separation; bare coefficient records leave that check to callers.
    """
    geometry_keys = ("temperature_midpoint_K", "temperature_separation_K")
    has_geometry = any(key in record for record in (full, half) for key in geometry_keys)
    if has_geometry:
        if not all(key in record for record in (full, half) for key in geometry_keys):
            raise ValueError("both coefficients must supply midpoint and separation")
        fm = _number(full["temperature_midpoint_K"], "full midpoint", positive=True)
        hm = _number(half["temperature_midpoint_K"], "half midpoint", positive=True)
        fs = _number(full["temperature_separation_K"], "full separation", positive=True)
        hs = _number(half["temperature_separation_K"], "half separation", positive=True)
        if not math.isclose(fm, hm, rel_tol=1e-12, abs_tol=1e-10) or not math.isclose(fs, 2 * hs, rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError("full/half steps must share a midpoint and have a 2:1 separation")
    f = _number(full["value_pcm_per_K"], "full coefficient")
    h = _number(half["value_pcm_per_K"], "half coefficient")
    sf = _number(full["standard_error_pcm_per_K"], "full error", nonnegative=True)
    sh = _number(half["standard_error_pcm_per_K"], "half error", nonnegative=True)
    r = None if relative_target is None else _number(relative_target, "relative_target", positive=True)
    z = _number(confidence_z, "confidence_z", positive=True)
    sd = math.hypot(sf, sh)
    lower_reference = max(0., abs(h) - z * sh)
    lower_tolerance = None if r is None else r * lower_reference
    upper_tolerance = None if r is None else r * (abs(h) + z * sh)
    status = "descriptive" if r is None else "inconclusive"
    if r is not None and lower_reference > 0:
        if abs(f - h) + z * sd <= lower_tolerance:
            status = "pass"
        elif abs(f - h) - z * sd > upper_tolerance:
            status = "fail"
    return {"status": status, "signed_difference_pcm_per_K": f - h,
            "standard_error_pcm_per_K": sd, "relative_target": r,
            "nominal_target_pcm_per_K": None if r is None else r * abs(h),
            "target_interval_pcm_per_K": [lower_tolerance, upper_tolerance],
            "reference_resolved": bool(lower_reference > 0), "confidence_z": z,
            "uncertainty_assumption": "independent full/half coefficients; conservative interval screening"}


def normalized_power_rms(predicted, reference, *, weights=None, target=0.01):
    """RMS difference of powers normalized to their weighted mean (fraction).

    Arrays must have identical ordered physical bins and positive weighted
    normalization. Weights are explicit bin weights (uniform by default).
    Native bin Monte Carlo errors/covariances are not evaluated by this metric.
    ``target=None`` reports a descriptive metric without acceptance; the default
    0.01 is a historical application policy, not a generic physics tolerance.
    """
    p, r = np.asarray(predicted, dtype=float), np.asarray(reference, dtype=float)
    if p.ndim != 1 or p.size == 0 or p.shape != r.shape:
        raise ValueError("power arrays must have identical nonempty one-dimensional bins")
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != p.shape or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("weights must be finite, positive and match the bins")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(r)) or np.any(p < 0) or np.any(r < 0):
        raise ValueError("powers must be finite and nonnegative")
    mp, mr = np.average(p, weights=w), np.average(r, weights=w)
    if mp <= 0 or mr <= 0:
        raise ValueError("power normalization must be positive")
    rms = float(np.sqrt(np.average((p / mp - r / mr)**2, weights=w)))
    target = None if target is None else _number(target, "target", nonnegative=True)
    status = "descriptive" if target is None else "pass" if rms <= target else "fail"
    return {"status": status, "rms_fraction": rms,
            "target_fraction": target, "bin_count": int(p.size),
            "normalization": "each array divided by its weighted mean",
            "uncertainty_assessment": "not performed; fixed ordered physical bins required"}


def compare_reconstruction_chain(k_interpolated, k_exact, k_ce, *,
                                 k_ce_standard_error=0.0, target_pcm=20.0,
                                 power_interpolated=None, power_exact=None,
                                 power_ce=None, power_weights=None, power_target=0.01):
    """Separate interpolation error from exact-state GARNET/CE model bias.

    Historical defaults are 20 pcm and 1% power RMS, applied only to
    interpolated versus exact-state GARNET. Pass ``target_pcm=None`` and
    ``power_target=None`` for a metric-only report; any unassessed requested
    metric prevents an overall pass. CE comparisons remain descriptive.
    """
    interpolation = reactivity_change(k_exact, k_interpolated)
    assessment = _screen(interpolation["value_pcm"], 0., target_pcm, 1.96)
    comparisons = {
        "interpolated_vs_exact": {**interpolation, **assessment},
        "exact_vs_ce": reactivity_change(k_ce, k_exact, k_from_standard_error=k_ce_standard_error),
        "interpolated_vs_ce": reactivity_change(k_ce, k_interpolated, k_from_standard_error=k_ce_standard_error)}
    powers = (power_interpolated, power_exact, power_ce)
    if any(item is not None for item in powers):
        if not all(item is not None for item in powers):
            raise ValueError("supply all three power arrays together")
        comparisons["interpolated_vs_exact"]["power"] = normalized_power_rms(
            power_interpolated, power_exact, weights=power_weights, target=power_target)
        for name, value in (("exact_vs_ce", power_exact), ("interpolated_vs_ce", power_interpolated)):
            metric = normalized_power_rms(value, power_ce, weights=power_weights, target=None)
            metric["status"] = "descriptive"
            metric.pop("target_fraction")
            comparisons[name]["power"] = metric
    statuses = [assessment["status"]]
    if powers[0] is not None:
        statuses.append(comparisons["interpolated_vs_exact"]["power"]["status"])
    status = ("fail" if "fail" in statuses else
              "descriptive" if "descriptive" in statuses else assessment["status"])
    return {"status": status, "comparisons": comparisons,
            "gate_scope": "interpolated GARNET versus exact-state GARNET only"}


def interval_rod_worth(k_start, k_end, position_start_cm, position_end_cm, *,
                       k_start_standard_error=0.0, k_end_standard_error=0.0):
    """Signed interval-average d(rho)/dx in pcm/cm; x increases insertion."""
    a = _number(position_start_cm, "position_start_cm", nonnegative=True)
    b = _number(position_end_cm, "position_end_cm", nonnegative=True)
    if b <= a:
        raise ValueError("position_end_cm must exceed position_start_cm")
    delta = reactivity_change(k_start, k_end, k_from_standard_error=k_start_standard_error,
                             k_to_standard_error=k_end_standard_error)
    return {"value_pcm_per_cm": delta["value_pcm"] / (b - a),
            "standard_error_pcm_per_cm": delta["standard_error_pcm"] / (b - a),
            "interval_cm": [a, b], "quantity": "signed interval-average differential worth",
            "uncertainty_assumption": delta["uncertainty_assumption"]}


def assess_rod_worth(predicted: Sequence[dict], reference: Sequence[dict], *,
                     required_intervals, reference_peak=None, peak_standard_error=0.0,
                     peak_is_resolved=False, relative_target=0.05, confidence_z=1.96):
    """Assess only explicitly covered intervals using an externally resolved peak.

    ``peak_is_resolved`` asserts independent position-resolution checks for the
    supplied reference peak; a maximum over sparse central checks is insufficient.
    No global/full-travel claim is made by this helper. Missing coverage or an
    unresolved peak yields inconclusive, including for perfect agreement.
    ``relative_target=None`` returns descriptive interval metrics and coverage
    without accepting the comparison, regardless of peak resolution.
    """
    def interval(item):
        if len(item) != 2:
            raise ValueError("intervals require two coordinates")
        a, b = (_number(v, "interval coordinate", nonnegative=True) for v in item)
        if b <= a:
            raise ValueError("intervals must have increasing coordinates")
        return (a, b)
    required = [interval(item) for item in required_intervals]
    if not required or len(set(required)) != len(required):
        raise ValueError("required_intervals must be nonempty and unique")
    def indexed(records):
        result = {}
        for record in records:
            key = interval(record["interval_cm"])
            if key in result:
                raise ValueError("duplicate interval")
            result[key] = (_number(record["value_pcm_per_cm"], "worth"),
                           _number(record["standard_error_pcm_per_cm"], "worth error", nonnegative=True))
        return result
    p, r = indexed(predicted), indexed(reference)
    ratio = None if relative_target is None else _number(relative_target, "relative_target", positive=True)
    z = _number(confidence_z, "confidence_z", positive=True)
    sp = _number(peak_standard_error, "peak_standard_error", nonnegative=True)
    peak = None if reference_peak is None else _number(reference_peak, "reference_peak", positive=True)
    resolved = bool(peak_is_resolved and peak is not None and peak > z * sp)
    missing = [list(key) for key in required if key not in p or key not in r]
    reports = []
    for key in required:
        if key not in p or key not in r:
            continue
        pv, ps = p[key]
        rv, rs = r[key]
        difference, se = pv - rv, math.hypot(ps, rs)
        target = None if ratio is None else ratio * peak if peak is not None else 0.
        report = _screen(difference, se, target, z)
        if ratio is not None and not resolved:
            report["status"] = "inconclusive"
        elif ratio is not None:
            # Conservative bounds account for uncertainty in the peak scale too.
            if abs(difference) + z * se <= ratio * (peak - z * sp):
                report["status"] = "pass"
            elif abs(difference) - z * se > ratio * (peak + z * sp):
                report["status"] = "fail"
            else:
                report["status"] = "inconclusive"
        reports.append({"interval_cm": list(key), **report})
    states = [item["status"] for item in reports]
    status = "descriptive" if ratio is None else "fail" if "fail" in states else (
        "inconclusive" if missing or not resolved or "inconclusive" in states else "pass")
    return {"status": status, "interval_reports": reports, "missing_intervals_cm": missing,
            "required_intervals_cm": [list(key) for key in required],
            "reference_peak_pcm_per_cm": peak, "peak_standard_error_pcm_per_cm": sp,
            "peak_resolved": resolved, "relative_target": ratio,
            "scope": "listed required intervals only; no global or pointwise worth qualification"}
