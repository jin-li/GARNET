"""GARNET side of the GARNET vs FEMFFUSION performance report.

    PYTHONPATH=. python tools/assess_garnet_vs_femffusion.py iaea3d --mesh 2 \
        --device cpu --repeats 3 --output iaea-r2-cpu.json
    PYTHONPATH=. python tools/assess_garnet_vs_femffusion.py hpmr3d --mesh 4 --nz 50 \
        --device gpu --output hpmr-r4.json
    PYTHONPATH=. python tools/assess_garnet_vs_femffusion.py hpmr3d-transient --mesh 2 \
        --nz 30 --dt 0.01 --time-scheme bdf2 --dollars 0.5 --output t.json

Cases (all diffusion, all posed exactly as FEMFFUSION's inputs pose them):

- ``iaea3d``: :func:`garnet.benchmarks.iaea.build_iaea` (FEMFFUSION
  examples/3D_IAEA), ``--mesh`` cells per 20 cm node.
- ``langenbuch3d``: :func:`garnet.benchmarks.langenbuch.build_langenbuch` at
  t = 0 with ``excise_corners=True`` (examples/3D_Langenbuch).
- ``hpmr3d``: the hex-homogenized 3-D HP-MR, 11-group ENDF/B-VIII.0, drums
  withdrawn (every drum hexagon the library's drum beryllium; the same list as
  tools/export_femffusion.py), ``--mesh`` = triangles-per-hex refinement,
  ``--nz`` axial layers (multiple of 10). FEMFFUSION's input comes from
  tools/export_femffusion.py.
- ``hpmr3d-transient``: an external reactivity insertion on the same core:
  every fissile nu*Sigma_f scaled by 1 + eps(t), eps rising linearly to
  rho/(1 - rho), rho = ``--dollars`` * beta, over [0, ``--ramp``] s and held
  after (FEMFFUSION's Read_XML_File interpolation is the same linear ramp).
  Delayed spectrum = chi, no feedback. Records P(t)/P(0) as FEMFFUSION defines
  it -- the kappa*Sigma_f-weighted fission rate (its Total Power) -- and, as
  ``P_fission_source``, GARNET's own TransientResult.power (the
  nu*Sigma_f-weighted source, which carries the inserted 1 + eps(t)).

Node powers are written in FEMFFUSION's layout and normalization: power density
(Sigma_f-weighted) per 20 cm node [z][y][x] for the Cartesian cases (GARNET's
first axis is FEMFFUSION's y, see tests/validation/test_femffusion_equivalence.py)
and per FEMFFUSION plane and hexagon [plane][hex] for the HP-MR, divided by the
core-volume average power density (FEMFFUSION's ``norm``). Holes are null.

Timing: ``build`` is the builder, ``setup`` the solver constructor (operator
assembly, device-synchronized), ``solve`` the solver's own synchronized
``solve_seconds``. A warm-up solve (2 outers, or 2 time steps) runs first and
is reported as ``warmup`` -- on a GPU it carries the CuPy kernel compilation.
``--repeats N`` rebuilds the solver N times and keeps the fastest solve. Memory:
process peak RSS and the CuPy pool after the runs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_femffusion as ff  # noqa: E402  (tools/ is not a package)

from garnet.backend import asnumpy, synchronize  # noqa: E402
from garnet.benchmarks.hpmr import PITCH, build_hpmr3d  # noqa: E402
from garnet.benchmarks.hpmr_transient_bench import SUBSWEEPS  # noqa: E402
from garnet.benchmarks.iaea import build_iaea  # noqa: E402
from garnet.benchmarks.langenbuch import build_langenbuch  # noqa: E402
from garnet.hexraster import hex_round  # noqa: E402
from garnet.solver import DiffusionEigenSolver  # noqa: E402
from garnet.transient import TransientSolver  # noqa: E402
from garnet.tri import TriDiffusionEigenSolver, TriGroupOperator  # noqa: E402

CASES = ("iaea3d", "langenbuch3d", "hpmr3d", "hpmr3d-transient")
FF_PLANES = 10            # FEMFFUSION's HP-MR planes (20 cm)


def _power_weight(m):
    return m.kappa_fission if m.kappa_fission is not None else m.nu_sigma_f


# --- problems -------------------------------------------------------------

def build(case, mesh, nz):
    """Solver inputs plus the node map: (problem dict, node_of_cell, node layout)."""
    if case == "iaea3d":
        p = build_iaea(mesh)
        return dict(grid=p.grid, materials=p.materials, material_map=p.material_map,
                    active=p.active, bc=p.bc, mask_bc=p.mask_bc,
                    solver=DiffusionEigenSolver), ("cartesian", mesh, (17, 17, 19))
    if case == "langenbuch3d":
        p = build_langenbuch(mesh, excise_corners=True)
        mats, mmap = p.problem_at(0.0)
        return dict(grid=p.grid, materials=mats, material_map=mmap, active=p.active,
                    bc=p.bc, mask_bc=p.mask_bc,
                    solver=DiffusionEigenSolver), ("cartesian", mesh, (11, 11, 10))
    if case.startswith("hpmr3d"):
        if nz % FF_PLANES:
            raise SystemExit("--nz must be a multiple of 10")
        # absorber="smeared" paints each drum hexagon as one material (the
        # DRUM_BE slot); the benchmark's list puts plain drum beryllium there.
        p = build_hpmr3d(refine=mesh, nz=nz, absorber="smeared",
                         materials=ff.hpmr_endfb8_builtin(three_d=True))
        prob = dict(grid=p.grid, materials=ff.hpmr_materials(),
                    material_map=p.material_map,
                    active=p.active, bc=p.bc, mask_bc=p.mask_bc,
                    solver=TriDiffusionEigenSolver)
        return prob, ("hex", p.raster, nz)
    raise SystemExit(f"unknown case {case}")


def _hex_index_of_cells(raster):
    """FEMFFUSION hexagon index (0-based) of every 2-D triangle, -1 outside."""
    order = {(R, C): i for i, (R, C, _) in
             enumerate(s for row in ff.hpmr_rows() for s in row)}
    h = raster.side
    av = np.array([h * math.sqrt(3.0) / 2, h * 0.5])
    bv = np.array([0.0, h])
    na, nb, _ = raster.material_map.shape
    out = np.full((na, nb, 2), -1, dtype=np.int64)
    for a, b, t in zip(*np.nonzero(raster.material_map > 0)):
        O = (raster.origin[0] + a - 1) * bv + (raster.origin[1] + b - 1) * av
        x, y = O + (av + bv) * (1.0 if t == 0 else 2.0) / 3.0
        R = y / (PITCH * math.sqrt(3.0) / 2.0)
        C, R = hex_round(x / PITCH - R / 2.0, R)
        out[a, b, t] = order[(R, C)]
    return out


def node_powers(prob, flux, layout):
    """Node power density over the core-average density, FEMFFUSION's layout."""
    mats, mmap, active = prob["materials"], prob["material_map"], prob["active"]
    w = np.stack([_power_weight(m) for m in mats])                 # (M, G)
    dens = np.einsum("g...,...g->...", flux, w[mmap])              # power per unit volume
    act = np.ones(mmap.shape, bool) if active is None else np.asarray(active)
    dens = np.where(act, dens, 0.0)
    core_mean = dens.sum() / act.sum()                             # uniform cell volumes
    kind = layout[0]
    if kind == "cartesian":
        _, r, (nx, ny, nz) = layout
        node_p = dens.reshape(nx, r, ny, r, nz, r).sum(axis=(1, 3, 5))
        node_v = act.reshape(nx, r, ny, r, nz, r).sum(axis=(1, 3, 5))
        node = np.where(node_v > 0, node_p / np.maximum(node_v, 1) / core_mean, np.nan)
        return node.transpose(2, 0, 1)                             # [z][y][x]
    _, raster, nz = layout
    hex_of = _hex_index_of_cells(raster)
    n_hex = int(hex_of.max()) + 1
    per = nz // FF_PLANES
    node_p = np.zeros((FF_PLANES, n_hex))
    node_v = np.zeros((FF_PLANES, n_hex))
    inside = hex_of >= 0
    for k in range(nz):
        plane = k // per
        np.add.at(node_p[plane], hex_of[inside], dens[..., k][inside])
        np.add.at(node_v[plane], hex_of[inside], act[..., k][inside])
    return np.where(node_v > 0, node_p / np.maximum(node_v, 1) / core_mean, np.nan)


# --- runs -------------------------------------------------------------------

def _solver(prob, device, precond_degree, xp_sync=True):
    cls = prob["solver"]
    t0 = time.perf_counter()
    s = cls(prob["grid"], prob["materials"], prob["material_map"], bc=prob["bc"],
            device=device, active=prob["active"], mask_bc=prob["mask_bc"],
            precond_degree=precond_degree)
    if xp_sync:
        synchronize(s.xp)
    return s, time.perf_counter() - t0


def run_steady(args, prob, layout):
    solve_kw = dict(tol_k=args.tol_k, tol_source=args.tol_source,
                    anderson_depth=args.anderson_depth)
    s, _ = _solver(prob, args.device, args.precond_degree)
    t0 = time.perf_counter()
    s.solve(max_outer=2, tol_k=0.0, tol_source=0.0)
    synchronize(s.xp)
    warmup = time.perf_counter() - t0
    best, setups, solves = None, [], []
    for _ in range(args.repeats):
        s, setup = _solver(prob, args.device, args.precond_degree)
        res = s.solve(**solve_kw)
        if not res.converged:
            raise RuntimeError(f"not converged: {res}")
        setups.append(setup)
        solves.append(res.solve_seconds)
        if best is None or res.solve_seconds < best.solve_seconds:
            best = res
    node = node_powers(prob, asnumpy(best.flux), layout)
    return dict(k=float(best.k_eff), converged=bool(best.converged),
                counters=dict(outers=int(best.outer_iterations),
                              inners=int(best.inner_iterations)),
                timings=dict(setup=min(setups), setup_all=setups, solve=min(solves),
                             solve_all=solves, warmup=warmup),
                node_power=_json_array(node), device=best.device)


def _insertion_problem_at(prob, eps, ramp):
    base = list(prob["materials"])
    cache = {}

    def problem_at(t):
        factor = 1.0 + eps * min(max(t, 0.0) / ramp, 1.0)
        if factor not in cache:
            cache[factor] = ff.scale_production(base, factor)
        return cache[factor], prob["material_map"]
    return problem_at


class _KappaPower:
    """Records R(t) = sum kappa*Sigma_f phi / sum nuSigma_f(0) phi at each
    accepted step (on_step), so the nu-weighted P(t) GARNET reports converts
    to FEMFFUSION's kappa-weighted one: P_kappa = P_nu / (1 + eps(t)) * R(t)/R(0),
    nuSigma_f(t) being (1 + eps(t)) nuSigma_f(0) in every fissile cell."""

    def __init__(self, prob):
        mats, mmap = prob["materials"], prob["material_map"]
        act = np.ones(mmap.shape, bool) if prob["active"] is None else np.asarray(prob["active"])
        kap = np.stack([m.kappa_fission if m.kappa_fission is not None else m.nu_sigma_f
                        for m in mats])[mmap]                       # (..., G)
        nu = np.stack([m.nu_sigma_f for m in mats])[mmap]
        self.w = [np.moveaxis(np.where(act[..., None], a, 0.0), -1, 0) for a in (kap, nu)]
        self.dev = None
        self.times, self.ratios = [], []

    def ratio(self, flux):
        try:
            import cupy
            xp = cupy.get_array_module(flux if not isinstance(flux, list) else flux[0])
        except ImportError:
            xp = np
        phi = xp.stack(list(flux)) if isinstance(flux, list) else flux
        if self.dev is None or self.dev[0] is not xp:
            self.dev = (xp, [xp.asarray(w) for w in self.w])
        wk, wn = self.dev[1]
        return float(xp.sum(wk * phi)) / float(xp.sum(wn * phi))

    def __call__(self, t, flux, power):
        self.times.append(float(t))
        self.ratios.append(self.ratio(flux))


def run_transient(args, prob, layout):
    kin = ff.hpmr_benchmark_kinetics()
    eps = ff.insertion_epsilon(args.dollars, kin)
    problem_at = _insertion_problem_at(prob, eps, args.ramp)
    recorder = {"cur": None}

    def on_step(t, flux, power):
        if recorder["cur"] is not None:
            recorder["cur"](t, flux, power)

    def make():
        t0 = time.perf_counter()
        s = TransientSolver(prob["grid"], problem_at, kin, bc=prob["bc"],
                            active=prob["active"], mask_bc=prob["mask_bc"],
                            group_operator=TriGroupOperator,
                            eig_solver=TriDiffusionEigenSolver, on_step=on_step,
                            precond_degree=args.precond_degree, device=args.device)
        synchronize(s.xp)
        return s, time.perf_counter() - t0

    kw = dict(dt=args.dt, tol_step=args.tol_step, time_scheme=args.time_scheme,
              step_solver=args.step_solver,
              steady_kwargs=dict(tol_k=args.tol_k, tol_source=args.tol_source))
    if args.step_solver == "fixed-point":
        # As tuned on this 11-group set (hpmr_transient_bench): rebalanced
        # Picard, Anderson depth 1, 6 Gauss-Seidel scatter subsweeps.
        kw.update(rebalance=True, anderson_depth=args.step_anderson_depth,
                  max_sweeps=args.max_sweeps, scatter_subsweeps=args.subsweeps)
    else:
        # As in examples/hpmr_bdf_cpu_benchmark.py (the HP-MR BDF studies).
        kw["multigroup_kwargs"] = dict(scatter_sweeps=3, rtol=args.mg_rtol, inner_rtol=1e-3,
                                       precond_degree=args.precond_degree)
    if args.adaptive_rtol is not None:
        kw["adaptive_bdf"] = dict(rtol=args.adaptive_rtol,
                                  automatic_order=args.automatic_order)
    # Warm up on a GPU only: there the first steps compile the CuPy kernels.
    # NumPy has nothing to warm, and two CPU steps (with the initial
    # eigen-solve) cost as much as a sizeable fraction of the run.
    warmup = 0.0
    if args.device not in ("cpu", "numpy"):
        s, _ = make()
        t0 = time.perf_counter()
        s.solve(t_end=2 * args.dt, **{**kw, "adaptive_bdf": None})
        synchronize(s.xp)
        warmup = time.perf_counter() - t0
    best, setups, solves = None, [], []
    for _ in range(args.repeats):
        s, setup = make()
        rec = recorder["cur"] = _KappaPower(prob)
        res = s.solve(t_end=args.t_end, verbose=args.verbose, **kw)
        recorder["cur"] = None
        setups.append(setup)
        solves.append(res.solve_seconds)
        if best is None or res.solve_seconds < best.solve_seconds:
            best, best_rec = res, rec
    times = np.asarray(best.times)
    r0 = best_rec.ratio(best.steady.flux)
    ratio = np.interp(times[1:], best_rec.times, best_rec.ratios)
    factor = 1.0 + eps * np.minimum(np.maximum(times[1:], 0.0) / args.ramp, 1.0)
    p_kappa = np.concatenate(([1.0], np.asarray(best.power)[1:] / factor * ratio / r0))
    return dict(k=float(best.k0), converged=True, insertion=dict(
                    dollars=args.dollars, epsilon=eps, ramp=args.ramp),
                step_solver=args.step_solver,
                counters=dict(steps=len(best.times) - 1,
                              inners=int(best.total_inner_iterations),
                              sweeps=int(sum(best.step_iterations)),
                              rejected=int(best.rejected_steps)),
                timings=dict(setup=min(setups), setup_all=setups, solve=min(solves),
                             solve_all=solves, warmup=warmup),
                transient=dict(t=[float(v) for v in best.times],
                               P=[float(v) for v in p_kappa],
                               P_fission_source=[float(v) for v in best.power],
                               spectral_ratio=[float(v) / r0 for v in best_rec.ratios],
                               orders=[int(o) for o in best.time_orders]),
                device=best.device)


# --- bookkeeping --------------------------------------------------------------

def _json_array(a):
    return [_json_array(v) for v in a] if np.ndim(a) > 1 else \
        [None if not np.isfinite(v) else float(v) for v in a]


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.machine()


def environment(gpu: bool):
    info = dict(code="garnet", python=platform.python_version(), machine=platform.machine(),
                node=platform.node(), cpu=_cpu_model(), numpy=np.__version__,
                started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    commit = os.environ.get("GARNET_COMMIT")
    if not commit:
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True, cwd=os.path.dirname(__file__)).stdout.strip()
        except OSError:
            commit = None
    info["commit"] = commit or None
    try:
        import scipy
        info["scipy"] = scipy.__version__
    except ImportError:
        pass
    if not gpu:
        return info
    try:
        import cupy
        props = cupy.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        info.update(cupy=cupy.__version__,
                    gpu=name.decode() if isinstance(name, bytes) else name,
                    cuda_driver=cupy.cuda.runtime.driverGetVersion(),
                    cuda_runtime=cupy.cuda.runtime.runtimeGetVersion())
    except Exception:
        info.update(gpu=None)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--mesh", type=int, required=True,
                        help="cells per 20 cm node (Cartesian) or hex refinement (HP-MR)")
    parser.add_argument("--nz", type=int, default=20, help="HP-MR axial layers")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tol-k", type=float, default=1e-8)
    parser.add_argument("--tol-source", type=float, default=1e-7)
    parser.add_argument("--anderson-depth", type=int, default=8)
    parser.add_argument("--precond-degree", type=int, default=0)
    # transient
    parser.add_argument("--dollars", type=float, default=0.5)
    parser.add_argument("--ramp", type=float, default=2.0)
    parser.add_argument("--t-end", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--time-scheme", default="bdf2")
    parser.add_argument("--tol-step", type=float, default=1e-7)
    parser.add_argument("--step-solver", choices=["fixed-point", "monolithic"],
                        default="fixed-point")
    parser.add_argument("--mg-rtol", type=float, default=1e-9,
                        help="monolithic step: multigroup Krylov rtol")
    parser.add_argument("--subsweeps", type=int, default=SUBSWEEPS)
    parser.add_argument("--step-anderson-depth", type=int, default=1)
    parser.add_argument("--max-sweeps", type=int, default=4000)
    parser.add_argument("--adaptive-rtol", type=float, default=None)
    parser.add_argument("--automatic-order", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.adaptive_rtol is not None and args.step_solver != "monolithic":
        parser.error("--adaptive-rtol needs --step-solver monolithic (adaptive_bdf supports "
                     "only the monolithic step)")

    wall0 = time.perf_counter()
    t0 = time.perf_counter()
    prob, layout = build(args.case, args.mesh, args.nz)
    build_seconds = time.perf_counter() - t0
    G = prob["materials"][0].n_groups
    cells = int(np.count_nonzero(prob["active"])) if prob["active"] is not None \
        else int(np.prod(prob["grid"].shape))
    gpu = args.device not in ("cpu", "numpy")
    out = dict(environment=environment(gpu), args=vars(args), case=args.case,
               mesh=dict(refine=args.mesh, nz=args.nz if args.case.startswith("hpmr") else None,
                         cells=cells, groups=G, unknowns=cells * G))
    run = run_transient if args.case == "hpmr3d-transient" else run_steady
    out.update(run(args, prob, layout))
    out["timings"]["build"] = build_seconds
    out["timings"]["wall"] = time.perf_counter() - wall0
    out["memory"] = dict(max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    if gpu:
        import cupy
        out["memory"]["gpu_pool_bytes"] = int(cupy.get_default_memory_pool().total_bytes())
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=1)
    t = out["timings"]
    print(f"RESULT {args.case} mesh={args.mesh} nz={out['mesh']['nz']} device={out['device']} "
          f"unknowns={out['mesh']['unknowns']} k={out['k']:.7f} "
          f"counters={out['counters']} setup={t['setup']:.3f}s solve={t['solve']:.3f}s "
          f"warmup={t['warmup']:.3f}s wall={t['wall']:.1f}s")


if __name__ == "__main__":
    main()
