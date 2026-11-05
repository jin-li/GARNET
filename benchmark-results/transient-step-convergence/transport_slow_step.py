"""O13: the SDP1, SDPN and S_N transient steps before and after the fix.

    python transport_slow_step.py <code root> <case> <solver> <tol_step> <reps> <out.jsonl>

<code root> holds the garnet package to time (the fix or a checkout before
it). Cases: inf, bl20, biblis2, sn20; solvers: diffusion, sdp1, sp3, sdp3,
sdp3-gmres, sn. Summary and power histories: transport-before-after.json.
"""
import json
import sys
import time
from pathlib import Path

code_root, case, solver, tol, reps, out = sys.argv[1:7]
sys.path.insert(0, code_root)
sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "tests" / "verification"))
import numpy as np
import garnet
assert garnet.__file__.startswith(code_root), garnet.__file__
from garnet import (Grid, Kinetics, Material, TransientSolver, TransientSDP1Solver,
                    TransientSDPNSolver, TransientSPNSolver, TransientSNSolver)
tol, reps = float(tol), int(reps)

def infinite_medium():
    from test_kinetics_reference import GRID, KIN, SIGMA_A, BETA, _one_group, _step_problem
    pert = _one_group(0.002 * BETA.sum() * SIGMA_A)
    return dict(grid=GRID, problem=_step_problem(_one_group(), pert), kin=KIN, bc="reflective",
                t_end=4.0, dt=0.05, steady=None)

def bl(n=20):
    from test_sdpn_symmetric import _bl_problem
    grid, mats, mmap, bc = _bl_problem(n)
    fuel = mats[0]
    pert = [Material(name="fuel*", diffusion=fuel.diffusion, sigma_a=fuel.sigma_a * 0.995,
                     nu_sigma_f=fuel.nu_sigma_f, total=fuel.total, chi=fuel.chi), mats[1]]
    kin = Kinetics(velocities=[2.2e5], beta=[0.0065], decay=[0.08])
    return dict(grid=grid, problem=lambda t: (mats if t <= 0 else pert, mmap), kin=kin, bc=bc,
                t_end=0.5, dt=0.01, steady=dict(tol_k=1e-10, tol_source=1e-9))

def biblis(r=2):
    from garnet.benchmarks.biblis import build_biblis
    p = build_biblis(r)
    m1 = p.materials[1]
    sa = np.array(m1.sigma_a, float); sa[1] *= 0.997
    pert = list(p.materials); pert[1] = Material(name="m1*", diffusion=m1.diffusion, sigma_a=sa,
        nu_sigma_f=m1.nu_sigma_f, sigma_s=m1.sigma_s, chi=m1.chi)
    kin = Kinetics(velocities=[1.25e7, 2.5e5], beta=[0.0065], decay=[0.08])
    return dict(grid=p.grid, problem=lambda t: (p.materials if t <= 0 else pert, p.material_map),
                kin=kin, bc=p.bc, t_end=0.2, dt=0.01, steady=dict(tol_k=1e-10, tol_source=1e-9),
                active=p.active, mask_bc=p.mask_bc)

def sn_core(n=20):
    from test_sdpn_symmetric import _bl_problem
    d = bl(n); d["bc"] = "vacuum"; d["t_end"] = 0.2
    return d

CASES = dict(inf=infinite_medium, bl20=bl, biblis2=biblis, sn20=sn_core)
c = CASES[case]()
extra = {k: c[k] for k in ("active", "mask_bc") if k in c}
def make():
    args = (c["grid"], c["problem"], c["kin"])
    if solver == "diffusion":
        return TransientSolver(*args, bc=c["bc"], device="cpu", **extra), dict(time_scheme="backward-euler")
    if solver == "sdp1":
        return TransientSDP1Solver(*args, bc=c["bc"], device="cpu", **extra), {}
    if solver == "sp3":
        return TransientSPNSolver(*args, order=1, bc=c["bc"], device="cpu", **extra), {}
    if solver == "sdp3":
        return TransientSDPNSolver(*args, order=3, bc=c["bc"], device="cpu", **extra), {}
    if solver == "sdp3-gmres":
        return TransientSDPNSolver(*args, order=3, bc=c["bc"], device="cpu",
                                   linear_solver="gmres", **extra), {}
    if solver == "sn":
        return TransientSNSolver(*args, bc=c["bc"], device="cpu", n_polar=2, n_azi=4), {}
times, res = [], None
for _ in range(reps):
    s, kw = make()
    if c["steady"]:
        kw["steady_kwargs"] = c["steady"]
    t0 = time.perf_counter()
    res = s.solve(t_end=c["t_end"], dt=c["dt"], tol_step=tol, **kw)
    times.append(time.perf_counter() - t0)
rec = dict(code="before" if "before" in code_root else "after", case=case, solver=solver, tol_step=tol,
           k0=float(res.k0), power=[float(p) for p in res.power], inners=int(res.total_inner_iterations),
           sweeps=(None if res.step_iterations is None else int(sum(res.step_iterations))),
           seconds=times, linsolve=getattr(s, "_linsolve", None).__name__ if hasattr(s, "_linsolve") else None)
with open(out, "a") as f:
    f.write(json.dumps(rec) + "\n")
print(f"{rec['code']:6s} {case:7s} {solver:9s} tol {tol:.0e} inners {rec['inners']:7d} best {min(times):.2f}s P(end) {rec['power'][-1]:.9f}", flush=True)
