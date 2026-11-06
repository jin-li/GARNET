"""Reference solutions of the GARNET vs FEMFFUSION report (data/reference/<case>.json).

Steady cases: a three-level limit (k(h) = k_inf + C h^p fitted exactly) of
k_eff and, at the fitted order, of the node powers. The primary series is
FEMFFUSION's degree-3 refinements 0/1/2 when present (the higher-order
discretization), else GARNET's GH200 mesh series. The uncertainty is the
largest of: half the extrapolation step and the spread over admissible fits;
the distance to FEMFFUSION's independent FE4/FE5 runs; and half the
distance to the other code's limit, counted only when that series is
asymptotic. Every limit and check is recorded in ``checks``.

Transient: P(t) of GARNET's finest two meshes at the two finest steps
(fixed-step BDF2), extrapolated in time (order 2) and then in space (order 2),
on the common output grid.

    PYTHONPATH=../.. python make_reference.py
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from make_figures import transient_power

HERE = os.path.dirname(os.path.abspath(__file__))

STEADY = {   # case: (GARNET GH200 series, its refinements, FEMFFUSION case tag)
    # IAEA: r = 8-16 are pre-asymptotic (order 1.57); r = 16-28 give order 1.98.
    "iaea3d": ("gpu-iaea-r*.json", (16, 24, 28), "iaea"),
    "langenbuch3d": ("gpu-lmw-r*.json", (8, 12, 16), "lmw"),
    "hpmr3d": ("ref-hpmr-r*.json", (3, 6, 12), "hpmr"),
}
FF_LEVELS = (0, 1, 2)          # FEMFFUSION FE3 refinements of the primary series
FF2_LEVELS = (1, 2, 3)         # FEMFFUSION FE2 refinements of the check series
FF_K_DIGITS = 7                # FEMFFUSION prints k with 7 significant digits


def _arr(x):
    return np.array(json.loads(json.dumps(x).replace("null", "NaN")), dtype=float)


def _runs(pattern, key):
    out = {}
    for f in glob.glob(os.path.join(HERE, "data", "perf", "*", pattern)):
        r = json.load(open(f))
        if r.get("k") is not None:
            out.setdefault(key(r), r)
    return out


def fit_limit(h, k):
    """k = k_inf + C h^p through three points; returns (k_inf, p)."""
    (h1, h2, h3), (k1, k2, k3) = h, k
    d1, d2 = k1 - k2, k2 - k3
    if d1 == 0 or d2 == 0 or np.sign(d1) != np.sign(d2):
        return None, None
    # solve (h1^p - h2^p)/(h2^p - h3^p) = d1/d2 for p by bisection
    f = lambda p: (h1 ** p - h2 ** p) / (h2 ** p - h3 ** p) - d1 / d2
    lo, hi = 0.2, 6.0
    if f(lo) * f(hi) > 0:
        return None, None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
    p = 0.5 * (lo + hi)
    C = d2 / (h2 ** p - h3 ** p)
    return k3 - C * h3 ** p, p


def series_limit(runs, levels, h_of, p_default, k_digits=None):
    """Three-level limit of k and node powers over ``levels`` of ``runs``.

    ``k_digits``: significant digits k is printed with (FEMFFUSION: 7, so
    0.1 pcm above k = 1). One printed digit is a floor on the uncertainty, and
    an order fitted to a last step of at most two printed digits is rounding
    noise, so ``p_default`` is used instead (IAEA FE3 r = 1 -> 2: 0.1 pcm)."""
    h = [h_of(l) for l in levels]
    k = [runs[l]["k"] for l in levels]
    quantum = (0.0 if k_digits is None
               else 10.0 ** (np.floor(np.log10(abs(k[2]))) - (k_digits - 1)))
    k_fit, p = fit_limit(h, k)
    if abs(k[2] - k[1]) <= 2 * quantum:
        k_fit, p = None, None
    k2 = k[2] + (k[2] - k[1]) / ((h[1] / h[2]) ** p_default - 1)
    estimates = [e for e in (k_fit, k2) if e is not None]
    k_inf = k_fit if k_fit is not None else k2
    p_use = p if p is not None else p_default
    unc = max(0.5 * abs(k_inf - k[2]), max(estimates) - min(estimates), quantum)
    fine, coarse = _arr(runs[levels[2]]["node_power"]), _arr(runs[levels[1]]["node_power"])
    node = fine + (fine - coarse) / ((h[1] / h[2]) ** p_use - 1)
    return dict(k=k_inf, uncertainty=unc, order=p, order_used=p_use, node=node,
                levels=[dict(level=l, unknowns=runs[l]["mesh"]["unknowns"], k=runs[l]["k"])
                        for l in levels])


def best_femffusion(tag, top=3):
    """Fallback primary when no degree-3 three-level series exists (HP-MR:
    FE3 r2 is out of reach): FEMFFUSION's finest run, its uncertainty the
    spread of the ``top`` finest runs (any degree) about it."""
    runs = {}
    for f in glob.glob(os.path.join(HERE, "data", "perf", "*", f"*ff*-{tag}-fe*-r*.json")):
        r = json.load(open(f))
        if r.get("k") is not None and r.get("node_power") is not None:
            runs.setdefault((r["mesh"]["degree"], r["mesh"]["refinements"]), r)
    # Equal DoFs (HP-MR FE2 r2 and FE4 r1): the higher degree is the finer run.
    finest = sorted(runs.values(), key=lambda r: (r["mesh"]["dofs"], r["mesh"]["degree"]),
                    reverse=True)[:top]
    if len(finest) < top:
        return None
    best = finest[0]
    return dict(k=best["k"], uncertainty=max(abs(r["k"] - best["k"]) for r in finest),
                order=None, node=_arr(best["node_power"]),
                run=dict(code="femffusion", degree=best["mesh"]["degree"],
                         refinements=best["mesh"]["refinements"]),
                label=f"FEMFFUSION finest run, FE{best['mesh']['degree']} "
                      f"r{best['mesh']['refinements']} ({best['mesh']['dofs']:,} DoFs)",
                levels=[dict(level=f"FE{r['mesh']['degree']} r{r['mesh']['refinements']}",
                             unknowns=r["mesh"]["dofs"], k=r["k"]) for r in finest])


def _json_nodes(node):
    return json.loads(json.dumps(np.where(np.isfinite(node), node, np.nan).tolist())
                      .replace("NaN", "null"))


def steady_reference(case):
    pattern, glevels, tag = STEADY[case]
    g = _runs(pattern, lambda r: r["mesh"]["refine"])
    garnet = (series_limit(g, glevels, lambda r: 1.0 / r, 2.0)
              if all(l in g for l in glevels) else None)
    ff = _runs(f"*ff*-{tag}-fe3-r*.json", lambda r: r["mesh"]["refinements"])
    femff = (series_limit(ff, FF_LEVELS, lambda r: 0.5 ** r, 3.0, k_digits=FF_K_DIGITS)
             if all(l in ff for l in FF_LEVELS) else None)
    ff2 = _runs(f"ff-{tag}-fe2-r*.json", lambda r: r["mesh"]["refinements"])
    fe2 = (series_limit(ff2, FF2_LEVELS, lambda r: 0.5 ** r, 4.0, k_digits=FF_K_DIGITS)
           if all(l in ff2 for l in FF2_LEVELS) else None)
    if femff is None:
        femff = best_femffusion(tag)
    # Primary: the candidate with the smallest own uncertainty.
    candidates = [c for c in (femff, garnet) if c is not None]
    if not candidates:
        return None
    primary = min(candidates, key=lambda c: c["uncertainty"])
    # Uncertainty: the primary limit's own, the spread of the independent
    # high-degree FEMFFUSION runs (FE4/FE5) about it, and the other code's
    # limit -- the latter only when its sequence is asymptotic (GARNET's
    # finite volumes are second order: observed order within 1.5-2.5).
    unc = primary["uncertainty"]
    checks = []
    for f in sorted(glob.glob(os.path.join(HERE, "data", "perf", "*", f"ffref-{tag}-*.json"))):
        r = json.load(open(f))
        if r.get("k") and r["mesh"]["degree"] >= 4:
            checks.append(dict(source=f"FEMFFUSION FE{r['mesh']['degree']} r{r['mesh']['refinements']}",
                               dofs=r["mesh"]["dofs"], k=r["k"],
                               dk_pcm=1e5 * (r["k"] - primary["k"])))
            if primary is femff:
                unc = max(unc, abs(r["k"] - primary["k"]))
    if fe2 is not None:     # the degree-2 limit: an independent FEMFFUSION series
        checks.append(dict(source="FEMFFUSION FE2 limit (r = 1, 2, 3)", k=fe2["k"],
                           dk_pcm=1e5 * (fe2["k"] - primary["k"]),
                           uncertainty_pcm=1e5 * fe2["uncertainty"], order=fe2["order"]))
        if primary is femff:
            unc = max(unc, abs(fe2["k"] - primary["k"]))
    if femff and garnet:
        other = garnet if primary is femff else femff
        asymptotic = (other is femff or
                      (garnet["order"] is not None and 1.5 <= garnet["order"] <= 2.5))
        if asymptotic and other["uncertainty"] < 5 * primary["uncertainty"] + 1e-5:
            unc = max(unc, 0.5 * abs(femff["k"] - garnet["k"]))
    for name, s in (("FEMFFUSION " + ("finest run" if femff and "label" in femff else "FE3 limit"), femff),
                    ("GARNET GH200 limit", garnet)):
        if s is not None:
            checks.append(dict(source=name, k=s["k"], dk_pcm=1e5 * (s["k"] - primary["k"]),
                               uncertainty_pcm=1e5 * s["uncertainty"], order=s["order"]))
    if "label" in primary:
        source = primary["label"] + " (spread of the three finest FEMFFUSION runs)"
    else:
        which = "FEMFFUSION FE3, refinements 0/1/2" if primary is femff else f"GARNET GH200, r = {glevels}"
        source = (f"{which} (three-level limit"
                  + (f", observed order {primary['order']:.2f}" if primary["order"] is not None
                     else f", nominal order {primary['order_used']:g}: last step within"
                          " the printed digits" if "order_used" in primary else "")
                  + ")")
    return dict(case=case, k=primary["k"], uncertainty_pcm=1e5 * unc,
                observed_order=primary["order"], node_power=_json_nodes(primary["node"]),
                source=source, levels=primary["levels"], checks=checks,
                reference_run=primary.get("run"))


TRANSIENT_LEVELS = ((4, 50), (6, 80), (8, 100))   # (refine, nz); r12 exceeds the GH200
TRANSIENT_DT = (0.025, 0.0125)


def transient_reference():
    """P(t) limit: BDF2 step extrapolation (order 2) per mesh, then a
    three-level space fit per output time, on the 0.025 s grid."""
    runs = {}
    for f in glob.glob(os.path.join(HERE, "data", "perf", "*", "gputr-*bdf2-dt*.json")):
        r = json.load(open(f))
        if r["args"].get("adaptive_rtol") is None and r["args"].get("step_solver",
                                                                         "fixed-point") == "fixed-point":
            runs[(r["mesh"]["refine"], r["mesh"]["nz"], r["args"]["dt"])] = r
    grid = np.round(np.arange(0.0, 2.0 + 1e-9, TRANSIENT_DT[0]), 10)
    per_mesh, k0 = [], None
    for rz in TRANSIENT_LEVELS:
        coarse, fine = (runs.get(rz + (dt,)) for dt in TRANSIENT_DT)
        if coarse is None or fine is None:
            return None
        Pc = np.interp(grid, *transient_power(coarse))
        Pf = np.interp(grid, *transient_power(fine))
        per_mesh.append((Pf + (Pf - Pc) / 3.0, Pf - Pc))
        k0 = fine["k"]
    h = [1.0 / r for r, _ in TRANSIENT_LEVELS]
    P = np.empty_like(grid)
    orders = []
    for i in range(grid.size):
        vals = [m[0][i] for m in per_mesh]
        lim, p = fit_limit(h, vals)
        if lim is None:                       # not monotone at this time: order 2, finest pair
            lim = vals[2] + (vals[2] - vals[1]) / ((h[1] / h[2]) ** 2 - 1)
        else:
            orders.append(p)
        P[i] = lim
    space_step = np.max(np.abs(P / per_mesh[2][0] - 1.0))
    time_step = np.max(np.abs(per_mesh[2][1] / 3.0 / per_mesh[2][0]))
    return dict(case="hpmr3d-transient", t=grid.tolist(), P=P.tolist(), k0=k0,
                uncertainty_rel=max(space_step, time_step) / 2,
                space_correction_rel=space_step, time_correction_rel=time_step,
                observed_space_order=(float(np.median(orders)) if orders else None),
                source="GARNET GH200, BDF2 dt = 0.025/0.0125 s extrapolated in time, "
                       f"then in space over (r, nz) = {TRANSIENT_LEVELS}")


def main():
    os.makedirs(os.path.join(HERE, "data", "reference"), exist_ok=True)
    tr = transient_reference()
    if tr is not None:
        with open(os.path.join(HERE, "data", "reference", "hpmr3d-transient.json"), "w") as fh:
            json.dump(tr, fh, indent=1)
        print(f"hpmr3d-transient: P(2 s) = {tr['P'][-1]:.6f}, space corr {tr['space_correction_rel']:.1e}, "
              f"time corr {tr['time_correction_rel']:.1e}, order {tr['observed_space_order']}")
    for case in STEADY:
        ref = steady_reference(case)
        if ref is None:
            print(f"{case}: reference levels missing")
            continue
        with open(os.path.join(HERE, "data", "reference", f"{case}.json"), "w") as fh:
            json.dump(ref, fh, indent=1)
        print(f"{case}: k_ref = {ref['k']:.7f} +- {ref['uncertainty_pcm']:.2f} pcm ({ref['source']}); "
              + "; ".join(f"{c['source']} {c['dk_pcm']:+.2f}" for c in ref['checks']))


if __name__ == "__main__":
    main()
