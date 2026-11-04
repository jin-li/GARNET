"""Figures, tables and summary.json of the GARNET vs FEMFFUSION performance report.

Inputs (written by collect.sh from the Slurm results):

- ``data/perf/<job>/<tag>.json`` (copied by collect.sh): one file per run, from
  tools/assess_garnet_vs_femffusion.py (GARNET) or tools/run_femffusion.py
  (FEMFFUSION). Both carry ``k``, ``node_power`` in FEMFFUSION's layout and
  normalization, ``timings``, ``counters``, ``mesh``, ``memory``,
  ``environment`` and, for the transient, ``transient.t``/``transient.P``.
- ``data/reference/<case>.json``: the named reference: ``k``, ``node_power``
  (steady) or ``t``/``P`` (transient), ``source`` and ``uncertainty_pcm``.

Every run is placed on its platform -- FEMFFUSION FE2, FEMFFUSION FE3 (one
core), GARNET one core, GARNET GH200 -- and scored against the reference:
|dk| in pcm and the max and RMS relative node-power error over the nodes that
produce power (steady), or the max relative power error on the reference time
grid and at the end (transient). The time-to-accuracy tables take, per
platform, the cheapest MEASURED run meeting each target -- no interpolation.
"""

from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = {
    "iaea3d": "IAEA-3D",
    "langenbuch3d": "3-D Langenbuch ($t=0$)",
    "hpmr3d": "HP-MR 3-D (hex-homogenized, 11 groups)",
    "hpmr3d-transient": "HP-MR 3-D, +0.5\\,\\$ insertion",
}
# (|dk| pcm, max node-power error) for steady; (max |P/P_ref - 1|, ...) transient
TARGETS = {"steady": [(10.0, 1e-2), (1.0, 1e-3)], "transient": [(1e-2,), (1e-3,)]}
PLATFORMS = ["FEMFFUSION FE2", "FEMFFUSION FE3", "GARNET CPU", "GARNET GH200"]
STYLE = {"FEMFFUSION FE2": ("tab:orange", "s"), "FEMFFUSION FE3": ("tab:red", "D"),
         "GARNET CPU": ("tab:blue", "o"), "GARNET GH200": ("tab:green", "^")}


def _arr(x):
    return np.array(json.loads(json.dumps(x).replace("null", "NaN")), dtype=float)


def platform(run):
    if run["environment"].get("code") == "femffusion":
        return f"FEMFFUSION FE{run['mesh']['degree']}"
    return "GARNET GH200" if "GH200" in str(run.get("device", "")) else (
        "GARNET CPU" if str(run.get("device", "")).startswith("cpu") else
        f"GARNET {run.get('device')}")


def mesh_label(run):
    m = run["mesh"]
    if run["environment"].get("code") == "femffusion":
        return f"FE{m['degree']} r{m['refinements']}"
    return f"r{m['refine']}" + (f" z{m['nz']}" if m.get("nz") else "")


# Tags of runs that are not comparison rows: tuning sweeps and reference runs.
NOT_ROWS = ("tune-", "fftune-", "fftrtune-", "ffref-", "ref-", "gputr-ref-")


def _is_reference_run(run, ref):
    """The run a reference IS (FEMFFUSION's finest-run fallback) is not a row."""
    rr = (ref or {}).get("reference_run")
    return bool(rr) and run["environment"].get("code") == rr["code"] and \
        run["mesh"].get("degree") == rr["degree"] and \
        run["mesh"].get("refinements") == rr["refinements"]


def load_runs(case):
    out = []
    for f in sorted(glob.glob(os.path.join(HERE, "data", "perf", "*", "*.json"))):
        name = os.path.basename(f)
        if name.startswith(NOT_ROWS) or name.endswith("-alone.json"):    # tuning, refs, controls
            continue
        run = json.load(open(f))
        if run.get("case") == case:
            out.append(run)
    return out


def score_steady(run, ref):
    out = dict(dk_pcm=1e5 * (run["k"] - ref["k"]))
    if "node_power" in run and "node_power" in ref:
        p, q = _arr(run["node_power"]), _arr(ref["node_power"])
        fuel = q > 1e-5
        rel = np.abs(p[fuel] - q[fuel]) / q[fuel]
        out.update(power_max=float(np.max(rel)), power_rms=float(np.sqrt(np.mean(rel ** 2))))
    return out


def transient_power(run):
    """(t, P) with P FEMFFUSION's kappa*Sigma_f-weighted power ratio.

    GARNET records before the tool recorded it (no ``P_fission_source``) hold
    TransientResult.power, the nu*Sigma_f-weighted source, which carries the
    inserted 1 + eps(t). Dividing by it is exact up to the spectral factor
    R(t)/R(0) (kappa- over nu-weighted rate), measured at |R/R0 - 1| < 4e-8
    over this transient, so those records are converted, not rerun."""
    t = np.asarray(run["transient"]["t"], dtype=float)
    P = np.asarray(run["transient"]["P"], dtype=float)
    if run["environment"].get("code") != "femffusion" and "P_fission_source" not in run["transient"]:
        ins = run.get("insertion") or {}
        eps, ramp = ins.get("epsilon", 0.0), ins.get("ramp", 2.0)
        P = P / (1.0 + eps * np.minimum(np.maximum(t, 0.0) / ramp, 1.0))
    return t, P


def score_transient(run, ref):
    """Error at the run's own output times (t > 0) against a cubic spline of
    the dense reference -- the solver's accuracy, not the error of
    interpolating between its steps."""
    from scipy.interpolate import CubicSpline
    t, P = transient_power(run)
    keep = (t > 0) & (t <= ref["t"][-1] + 1e-12)
    Pr = CubicSpline(ref["t"], ref["P"])(t[keep])
    rel = np.abs(P[keep] / Pr - 1.0)
    return dict(power_max=float(rel.max()), power_end=float(abs(P[-1] / ref["P"][-1] - 1.0)),
                dk_pcm=1e5 * (run["k"] - ref["k0"]) if "k0" in ref else None)


FF_SCHEMES = {"Semi-Implicit-Euler": "SI-Euler", "Semi-Implicit-Exponential": "SI-Exp",
              "Implicit-Exponential": "I-Exp"}


def time_step(run):
    """(scheme, step) labels of a transient run. GARNET: BDF2 with the
    fixed-point step (the tool's default, and its only step before
    ``step_solver`` was recorded) or the monolithic one ("mono"); an adaptive
    run is aBDF<max order> (per-step error tolerance), and its step is the
    range of accepted steps."""
    if run["environment"].get("code") == "femffusion":
        prm = run["prm"]
        s = prm.get("Distributed_Time_Scheme", "Implicit-Exponential")
        return FF_SCHEMES.get(s, s), f"{float(prm['Time_Delta']):g}"
    a = run["args"]
    order = a["time_scheme"].upper()
    if a.get("adaptive_rtol") is not None:
        d = np.diff(np.asarray(run["transient"]["t"], dtype=float))
        e = int(round(math.log10(a["adaptive_rtol"])))
        lo = f"{d.min():.1e}".replace("e-0", "e-")
        return f"a{order} ($10^{{{e}}}$)", f"{lo}--{d.max():.2g}"
    mono = (run.get("step_solver") or a.get("step_solver")) == "monolithic"
    return order + (" mono" if mono else ""), f"{a['dt']:g}"


def transient_counters(run):
    """Time steps of the march and its inner iterations: GARNET's Krylov
    iterations and rejected steps; FEMFFUSION's GMRES iterations, one solve
    per output time (its first, at t = 0, is a null step)."""
    if run["environment"].get("code") == "femffusion":
        tr = run["transient"]
        return dict(steps=len(tr["t"]) - 1, inners=int(sum(tr["its"])), rejected=0)
    c = run.get("counters") or {}
    return dict(steps=c.get("steps"), inners=c.get("inners"), rejected=c.get("rejected") or 0)


def cost(run):
    """Matched cost scopes. ``solve``: the solver's own timed work -- GARNET's
    synchronized solve_seconds (for a transient: the initial eigen-solve plus
    the march), FEMFFUSION's CPU time from "Matrices assembled" to the
    eigenvalue (plus, for a transient, its transient CPU time). ``wall``: time
    to solution from a cold start -- FEMFFUSION's whole process; GARNET's
    build + solver construction + solve, plus on the GPU one warm-up run (an
    upper bound on the CuPy kernel compilation: it also does two outer
    iterations or time steps). NumPy compiles nothing, so the CPU warm-up --
    run by early records -- is not counted."""
    t = run["timings"]
    if run["environment"].get("code") == "femffusion":
        solve = t.get("solve")
        if t.get("cpu_transient") is not None and solve is not None:
            solve += t["cpu_transient"]
        wall = t.get("wall")
    else:
        solve = t.get("solve")
        gpu = "cuda" in str(run.get("device", "")) or "GH200" in str(run.get("device", ""))
        parts = ("build", "setup", "solve") + (("warmup",) if gpu else ())
        wall = sum(t.get(k) or 0.0 for k in parts)
    return dict(solve=solve, wall=wall, setup=t.get("setup"),
                assembly=t.get("assembly"), warmup=t.get("warmup"))


def rows_for(case):
    ref_path = os.path.join(HERE, "data", "reference", f"{case}.json")
    if not os.path.exists(ref_path):
        return [], None
    ref = json.load(open(ref_path))
    transient = case.endswith("transient")
    rows = []
    for run in load_runs(case):
        if _is_reference_run(run, ref):
            continue
        s = (score_transient if transient else score_steady)(run, ref)
        extra = {}
        if transient:
            scheme, step = time_step(run)
            extra = dict(scheme=scheme, step=step)
        rows.append(dict(platform=platform(run), mesh=mesh_label(run),
                         unknowns=run["mesh"].get("unknowns"), k=run["k"],
                         counters=(transient_counters(run) if transient else
                                   run.get("counters", {})),
                         memory=run.get("memory", {}), **extra, **s, **cost(run)))
    return rows, ref


def meets(row, target, transient):
    if transient:
        return row["power_max"] <= target[0]
    return abs(row["dk_pcm"]) <= target[0] and row.get("power_max", math.inf) <= target[1]


def time_to_accuracy(rows, transient, key="solve"):
    out = {}
    for target in TARGETS["transient" if transient else "steady"]:
        best = {}
        for p in PLATFORMS:
            ok = [r for r in rows if r["platform"] == p and meets(r, target, transient)
                  and r.get(key) is not None]
            best[p] = min(ok, key=lambda r: r[key]) if ok else None
        out[target] = best
    return out


# --- outputs ------------------------------------------------------------------

def _fmt_t(x):
    return "--" if x is None else (f"{x:.3g}" if x < 100 else f"{x:.0f}")


def write_tta_table(case, tta, transient):
    lines = ["\\begin{tabular}{llrr" + ("r" if transient else "rr") + "rr}", "\\hline",
             "Target & Platform & Mesh & Unknowns & " +
             ("max $|P/P_\\mathrm{ref}-1|$" if transient else "$|\\Delta k|$ (pcm) & max $\\Delta P$") +
             " & Solve (s) & Wall (s) \\\\", "\\hline"]
    for target, best in tta.items():
        label = (f"{target[0]:g}" if transient else f"{target[0]:g} pcm, {100 * target[1]:g}\\%")
        for p in PLATFORMS:
            r = best[p]
            if r is None:
                cells = [label, p, "--", "--", "--"] + ([] if transient else ["--"]) + ["--", "--"]
            else:
                acc = [f"{r['power_max']:.1e}"] if transient else \
                      [f"{abs(r['dk_pcm']):.2f}", f"{100 * r.get('power_max', math.nan):.2f}\\%"]
                cells = [label, p, r["mesh"], f"{r['unknowns']:,}"] + acc + \
                        [_fmt_t(r["solve"]), _fmt_t(r["wall"])]
            lines.append(" & ".join(cells) + " \\\\")
            label = ""
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
    with open(os.path.join(HERE, "tables", f"tta_{case}.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def plot_pareto(case, rows, transient, ref):
    """Error vs solve time per platform: every run faint, the Pareto front
    (runs no cheaper run of the platform matches) as a line. Errors below the
    reference's own uncertainty are not resolved: they are drawn at that
    floor, hollow, over a grey band; the targets are dashed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if transient:
        panels = [("power_max", "max $|P/P_\\mathrm{ref}-1|$", ref.get("uncertainty_rel", 0.0),
                   [t[0] for t in TARGETS["transient"]])]
    else:
        panels = [("dk_pcm", "$|\\Delta k|$ (pcm)", ref.get("uncertainty_pcm", 0.0),
                   [t[0] for t in TARGETS["steady"]]),
                  ("power_max", "max node-power error", None, [t[1] for t in TARGETS["steady"]])]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.8), squeeze=False)
    for ax, (key, label, floor, targets) in zip(axes[0], panels):
        if not floor:                       # node powers: resolution of the reference fit
            floor = 1e-5
        for p in PLATFORMS:
            pts = sorted((r["solve"], abs(r[key])) for r in rows
                         if r["platform"] == p and r.get(key) is not None and r.get("solve"))
            if not pts:
                continue
            c, m = STYLE[p]
            # every run, faint; the Pareto front (no cheaper run is as
            # accurate) as the platform's line
            ax.loglog([t for t, _ in pts], [max(v, floor) for _, v in pts], ls="none",
                      marker=m, color=c, ms=4, alpha=0.3)
            front, best = [], math.inf
            for t, v in pts:
                if v < best:
                    front.append((t, v))
                    best = v
            x = [t for t, _ in front]
            y = [max(v, floor) for _, v in front]
            ax.loglog(x, y, color=c, lw=1.2, label=p)
            for xi, (_, v), yi in zip(x, front, y):
                ax.plot(xi, yi, marker=m, color=c, ms=6,
                        mfc=c if v >= floor else "white", mec=c)
        ax.axhspan(ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else floor / 10, floor,
                   color="0.85", zorder=0)
        for t in targets:
            ax.axhline(t, color="0.4", ls="--", lw=0.8)
        ax.set_xlabel("solve time (s)")
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", f"pareto_{case}.pdf"))
    if os.environ.get("FIG_PNG"):            # quick-look copies for review
        fig.savefig(os.path.join(HERE, "figures", f"pareto_{case}.png"), dpi=80)
    plt.close(fig)


def _counters(r):
    c = r.get("counters") or {}
    if "matvecs" in c:
        return f"{c.get('outers') or '--'} / {c.get('matvecs') or '--'} mv"
    if "steps" in c:
        return f"{c['steps']} steps / {c.get('inners', '--')}"
    return f"{c.get('outers', '--')} / {c.get('inners', '--')}"


def _memory_gb(r):
    mem = r.get("memory") or {}
    return (mem.get("gpu_pool_bytes") or mem.get("max_rss_bytes") or 0) / 1e9


def _platform_order(r):
    return PLATFORMS.index(r["platform"]) if r["platform"] in PLATFORMS else 9


def write_runs_table(case, rows, transient):
    if transient:
        return write_transient_runs_table(case, rows)
    head = ("Platform & Mesh & Unknowns & $k$ & $\\Delta k$ (pcm) & max $\\Delta P$ & "
            "RMS $\\Delta P$ & Outers / inners & Solve (s) & Wall (s) & Memory (GB) \\\\")
    lines = ["\\begin{tabular}{llrrrrrrrrr}", "\\hline", head, "\\hline"]
    for r in sorted(rows, key=lambda r: (_platform_order(r), r["unknowns"] or 0, r["solve"] or 0)):
        acc = [f"{r['k']:.7f}", f"{r['dk_pcm']:+.2f}", f"{100 * r.get('power_max', math.nan):.2f}\\%",
               f"{100 * r.get('power_rms', math.nan):.2f}\\%"]
        lines.append(" & ".join([r["platform"], r["mesh"].replace("_", " "),
                                 f"{r['unknowns']:,}"] + acc +
                                [_counters(r), _fmt_t(r["solve"]), _fmt_t(r["wall"]),
                                 f"{_memory_gb(r):.1f}"]) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    with open(os.path.join(HERE, "tables", f"runs_{case}.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_transient_runs_table(case, rows):
    """The transient's runs (about a hundred): a longtable that breaks across
    pages, rows grouped under a heading per platform, footnotesize cells, and
    the caption \\transientrunscaption that sections/runs.tex defines (it
    also explains the column names)."""
    cols = "lrllrrrrrrr"
    head = ("Mesh & Unknowns & Scheme & $\\Delta t$ (s) & max err & end err & Steps & "
            "Inners & Solve (s) & Wall (s) & Mem.\\ (GB) \\\\")
    spec = "".join(">{\\footnotesize}" + c for c in cols)
    lines = [f"\\begin{{longtable}}{{{spec}}}",
             "\\caption{\\transientrunscaption}\\label{tab:runs-transient}\\\\",
             "\\hline", head, "\\hline", "\\endfirsthead",
             "\\hline", head, "\\hline", "\\endhead",
             "\\hline", "\\endfoot", "\\hline", "\\endlastfoot"]
    last = None
    for r in sorted(rows, key=lambda r: (_platform_order(r), r["unknowns"] or 0, r["scheme"],
                                         r["solve"] or 0)):
        if r["platform"] != last:
            if last is not None:
                lines.append("\\noalign{\\smallskip}")
            lines.append(f"\\multicolumn{{{len(cols)}}}{{l}}{{\\textbf{{{r['platform']}}}}} \\\\")
            last = r["platform"]
        c = r["counters"]
        steps = f"{c['steps']}" + (f" ({c['rejected']} rej.)" if c.get("rejected") else "")
        inners = "--" if c.get("inners") is None else f"{c['inners']:,}"
        lines.append(" & ".join([r["mesh"], f"{r['unknowns']:,}", r["scheme"], r["step"],
                                 f"{r['power_max']:.1e}", f"{r['power_end']:.1e}", steps, inners,
                                 _fmt_t(r["solve"]), _fmt_t(r["wall"]),
                                 f"{_memory_gb(r):.1f}"]) + " \\\\")
    lines.append("\\end{longtable}")
    with open(os.path.join(HERE, "tables", f"runs_{case}.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_ratio_table(summary):
    """Time-to-accuracy (solve) per target and the ratios between platforms."""
    lines = ["\\begin{tabular}{llrrrrr}", "\\hline",
             "Case & Target & \\ffn{} best (s) & \\garnet{} CPU (s) & \\garnet{} GH200 (s) & "
             "\\ffn{}/CPU & \\ffn{}/GH200 \\\\", "\\hline"]
    for case, d in summary.items():
        for target, best in d["tta_rows"].items():
            ff = [b for p, b in best.items() if p.startswith("FEMFFUSION") and b]
            ffb = min(ff, key=lambda b: b["solve"]) if ff else None
            cpu, gpu = best.get("GARNET CPU"), best.get("GARNET GH200")
            def t(b):
                return "--" if b is None else _fmt_t(b["solve"])
            def ratio(a, b):
                if a is None or b is None:
                    return "--"
                # two significant figures, never in e-notation (130, not 1.3e+02)
                return f"{float(format(a['solve'] / b['solve'], '.2g')):g}"
            lines.append(" & ".join([CASES[case] if target == list(d["tta_rows"])[0] else "",
                                     target, t(ffb), t(cpu), t(gpu), ratio(ffb, cpu),
                                     ratio(ffb, gpu)]) + " \\\\")
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    with open(os.path.join(HERE, "tables", "tta_summary.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_reference_table():
    lines = ["\\begin{tabular}{lp{7.3cm}rr}", "\\hline",
             "Case & Reference & Value & Uncertainty \\\\", "\\hline"]
    for case in CASES:
        path = os.path.join(HERE, "data", "reference", f"{case}.json")
        if not os.path.exists(path):
            continue
        ref = json.load(open(path))
        if case.endswith("transient"):
            lines.append(f"{CASES[case]} & {ref['source']} & $P(2\\,\\mathrm{{s}})$ = "
                         f"{ref['P'][-1]:.5f} & {ref['uncertainty_rel']:.0e} \\\\")
            continue
        lines.append(f"{CASES[case]} & {ref['source']} & $k$ = {ref['k']:.7f} & "
                     f"{ref['uncertainty_pcm']:.2f} pcm \\\\")
        for c in ref.get("checks", []):
            if c["source"].startswith(("FEMFFUSION FE", "GARNET", "FEMFFUSION finest")) \
                    and abs(c["dk_pcm"]) > 0:
                lines.append(f" & \\quad check: {c['source']} & {c['k']:.7f} & "
                             f"{c['dk_pcm']:+.2f} pcm \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    with open(os.path.join(HERE, "tables", "references.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    summary = {}
    for case in CASES:
        rows, ref = rows_for(case)
        if not rows:
            continue
        transient = case.endswith("transient")
        tta = time_to_accuracy(rows, transient)
        write_tta_table(case, tta, transient)
        write_runs_table(case, rows, transient)
        plot_pareto(case, rows, transient, ref)
        summary[case] = dict(reference=dict(source=ref.get("source"),
                                            k=ref.get("k", ref.get("k0"))),
                             rows=rows,
                             tta_rows={(f"{t[0]:g}" if transient else
                                        f"{t[0]:g} pcm, {100 * t[1]:g}\\%"):
                                       {p: (r and dict(mesh=r["mesh"], solve=r["solve"], wall=r["wall"]))
                                        for p, r in b.items()} for t, b in tta.items()},
                             time_to_accuracy={str(t): {p: (r and dict(mesh=r["mesh"],
                                                                       solve=r["solve"],
                                                                       wall=r["wall"]))
                                                        for p, r in b.items()}
                                               for t, b in tta.items()})
    write_ratio_table(summary)
    write_reference_table()
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
