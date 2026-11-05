"""O15: HP-MR 2-D refine 2, 11 groups, slow-symmetric drum ramp (step 07j case).
python hpmr.py <code root> <out.jsonl> <label:scheme:auto:rtol> ..."""
import json, sys, time
root, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, root); sys.path.insert(1, root + "/examples")
import numpy as np
import garnet
assert garnet.__file__.startswith(root), garnet.__file__
from hpmr_bdf_cpu_benchmark import _run, _trajectory_frames
from garnet.benchmarks.hpmr_transient_bench import build_case
from garnet.tri import TriDiffusionEigenSolver
problem, kinetics = build_case(2, nz=0, groups="11")
steady = TriDiffusionEigenSolver(problem.grid, problem.materials, problem.material_map, bc=problem.bc,
                                 active=problem.active, mask_bc=problem.mask_bc, mix_material=problem.mix_material,
                                 mix_weight=problem.mix_weight, precond_degree=1, device="cpu").solve(tol_k=1e-11, tol_source=1e-10)
frames = _trajectory_frames(problem, 2, 5, False, samples=10)
duration = 0.8
for spec in sys.argv[3:]:
    label, scheme, auto, rtol = spec.split(":")
    rtol = float(rtol)
    res, wall = _run(problem, kinetics, steady, frames, duration, 8, scheme, adaptive_bdf={
        "rtol": rtol, "min_dt": duration / 64 / 4096, "max_dt": duration / 4,
        "automatic_order": auto == "auto", "rejection_strategy": "half", "reject_max_factor": 0.5})
    o = list(res.time_orders)
    rec = dict(code="before" if "before" in root else "after", label=label, scheme=scheme, auto=auto == "auto",
               rtol=rtol, times=[float(x) for x in res.times], power=[float(x) for x in res.power],
               steps=len(res.times) - 1, rejected=int(res.rejected_steps), orders=o,
               inner=int(res.total_inner_iterations), seconds=wall)
    open(out, "a").write(json.dumps(rec) + "\n")
    print(f"{rec['code']:6s} {label:10s} rtol {rtol:g}: steps {rec['steps']} rej {rec['rejected']} "
          f"orders {np.bincount(o, minlength=6)[1:].tolist()} inner {rec['inner']} {wall:.1f}s", flush=True)
