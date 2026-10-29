"""GPU assessment of Cartesian ADF diffusion on the provisional Bernal case.

    PYTHONPATH=. python tools/assess_adf_gpu.py bernal --refinements 1 2 4 6 8 \
        --devices gpu cpu --repeats 3 --output bernal-gpu.json
    PYTHONPATH=. python tools/assess_adf_gpu.py contrast --refinement 4 --device gpu \
        --output contrast-gpu.json

``bernal`` separates the ADF effect from the solver change: ADF off with CG
and with GMRES, explicit unity factors with GMRES (must equal ADF off), and
Table 1 factors with GMRES and BiCGStab. Each configuration runs ``--repeats``
times and the fastest solve is kept (benchmark skill, best of N). Rows carry k,
pcm against the published VALKIN value (a nodal method, not truth),
iterations, setup and solve seconds, and peak device memory; matching CPU and
GPU runs are compared for k and normalized assembly fission production.

``contrast`` is the preconditioner robustness study: factors raised to an
exponent (0 = unity, 1 = Table 1, larger = stronger contrast) and random
per-node factors, against Neumann preconditioner degree and Krylov method.
Failed or unconverged solves are recorded, not dropped.

Setup and solve are wrapped in NVTX ranges, so ``nsys profile -t cuda,nvtx``
on a single configuration counts host/device transfers per phase.
The material map is the unverified figure interpretation; see
docs/bernal_adf_benchmark.md. These are code-performance and consistency
measurements, not a benchmark reproduction.
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np

from examples.bernal_adf_benchmark import TABLE, VALKIN, provisional_map, run
from garnet.profiling import nvtx_range

CONFIGS = {
    # name: (adf_on, linear_solver, factor_exponent)
    "off-cg": (False, "cg", 1.0),
    "off-gmres": (False, "gmres", 1.0),
    "unity-gmres": (True, "gmres", 0.0),
    "on-gmres": (True, "gmres", 1.0),
    "on-bicgstab": (True, "bicgstab", 1.0),
}


def device_memory():
    """(pool bytes held, device bytes in use) or (None, None) without CuPy."""
    try:
        import cupy
        free, total = cupy.cuda.runtime.memGetInfo()
        return cupy.get_default_memory_pool().total_bytes(), total - free
    except Exception:
        return None, None


def reset_device_memory():
    try:
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def measured(base, refinement, device, label, repeats, **options):
    """Best-of-N run with memory and failure bookkeeping."""
    best, times = None, []
    for _ in range(repeats):
        reset_device_memory()
        _, used_before = device_memory()
        try:
            with nvtx_range(f"{label} r{refinement} {device}"):
                row = run(base, refinement, device=device, **options)
        except Exception as exc:          # record, do not hide, a failed configuration
            return dict(label=label, refinement=refinement, requested_device=device,
                        failed=True, error=f"{type(exc).__name__}: {exc}", **_plain(options))
        pool, used_after = device_memory()
        row.update(pool_bytes=pool,
                   device_bytes_delta=None if used_after is None else used_after - used_before)
        times.append(row["solve_seconds"])
        if best is None or row["solve_seconds"] < best["solve_seconds"]:
            best = row
    best.update(label=label, requested_device=device, failed=False, solve_seconds_all=times,
                **_plain(options))
    return best


def _plain(options):
    return {k: v for k, v in options.items() if k != "node_factors"}


def print_row(r):
    if r.get("failed"):
        print(f"{r['label']:28s} r={r['refinement']} {r['requested_device']:3s}  FAILED  {r['error']}",
              flush=True)
        return
    print(f"{r['label']:28s} r={r['refinement']} {r['device']:>6s}  k={r['k_eff']:.7f}  "
          f"dk={r['pcm_vs_valkin']:+8.1f} pcm vs VALKIN  outers={r['outer_iterations']:4d}  "
          f"inners={r['inner_iterations']:7d}  setup={r['setup_seconds']:7.2f}s  "
          f"t={r['solve_seconds']:8.2f}s  res={r['relative_eigen_residual']:.1e}", flush=True)


def bernal(args):
    base = provisional_map()
    rows = []
    for refinement in args.refinements:
        for device in args.devices:
            if device == "cpu" and refinement > args.cpu_max_refinement:
                continue
            for name in args.configs:
                adf_on, solver, exponent = CONFIGS[name]
                row = measured(base, refinement, device, name, args.repeats, adf_on=adf_on,
                               linear_solver=solver, factor_exponent=exponent,
                               precond_degree=args.precond_degree)
                rows.append(row)
                print_row(row)
    ok = [r for r in rows if not r["failed"]]
    checks = []
    for r in ok:
        if r["label"] == "unity-gmres":
            off = [o for o in ok if o["label"] == "off-gmres" and o["refinement"] == r["refinement"]
                   and o["requested_device"] == r["requested_device"]]
            if off:
                checks.append(dict(check="unity ADF equals ADF off", refinement=r["refinement"],
                                   device=r["requested_device"],
                                   abs_dk=abs(r["k_eff"] - off[0]["k_eff"]),
                                   passed=abs(r["k_eff"] - off[0]["k_eff"]) <= 2e-8))
        if r["requested_device"] == "gpu":
            cpu = [c for c in ok if c["label"] == r["label"] and c["refinement"] == r["refinement"]
                   and c["requested_device"] == "cpu"]
            if cpu:
                a = np.asarray(r["normalized_assembly_fission_production"])
                b = np.asarray(cpu[0]["normalized_assembly_fission_production"])
                checks.append(dict(check="GPU equals CPU", label=r["label"],
                                   refinement=r["refinement"],
                                   abs_dk=abs(r["k_eff"] - cpu[0]["k_eff"]),
                                   max_production_diff=float(np.abs(a - b).max()),
                                   gpu_speedup=cpu[0]["solve_seconds"] / r["solve_seconds"],
                                   passed=abs(r["k_eff"] - cpu[0]["k_eff"]) <= 2e-8))
    for c in checks:
        print(json.dumps(c), flush=True)
    return dict(rows=rows, checks=checks)


def random_node_factors(seed, max_ratio):
    """Per-base-node, per-side, per-group factors with ratios up to max_ratio."""
    rng = np.random.default_rng(seed)
    half = 0.5 * np.log(max_ratio)
    return np.exp(rng.uniform(-half, half, size=(2, 1210, 6)))


def contrast(args):
    base = provisional_map()
    variants = [(f"table^{e:g}", dict(factor_exponent=e)) for e in args.exponents]
    variants += [(f"random<= {args.random_ratio:g} seed {seed}",
                  dict(node_factors=random_node_factors(seed, args.random_ratio)))
                 for seed in args.random_seeds]
    rows = []
    for name, factor_options in variants:
        for solver in args.solvers:
            for degree in args.degrees:
                row = measured(base, args.refinement, args.device, f"{name} {solver} d{degree}",
                               args.repeats, adf_on=True, linear_solver=solver,
                               precond_degree=degree, **factor_options)
                row.update(variant=name)
                rows.append(row)
                print_row(row)
    return dict(rows=rows, table_factor_ratio_max=float(TABLE[:, 7:].max() / TABLE[:, 7:].min()))


def environment():
    info = dict(python=platform.python_version(), machine=platform.machine(),
                numpy=np.__version__, started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    try:
        import cupy
        props = cupy.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        info.update(cupy=cupy.__version__,
                    gpu=name.decode() if isinstance(name, bytes) else name)
    except Exception as exc:
        info.update(gpu=None, gpu_error=f"{type(exc).__name__}: {exc}")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("bernal")
    b.add_argument("--refinements", nargs="+", type=int, default=[1, 2, 4, 6, 8])
    b.add_argument("--devices", nargs="+", choices=["gpu", "cpu"], default=["gpu", "cpu"])
    b.add_argument("--cpu-max-refinement", type=int, default=4)
    b.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    b.add_argument("--precond-degree", type=int, default=0)
    c = sub.add_parser("contrast")
    c.add_argument("--refinement", type=int, default=4)
    c.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    c.add_argument("--exponents", nargs="+", type=float, default=[0, 0.5, 1, 2, 4])
    c.add_argument("--random-seeds", nargs="*", type=int, default=[1, 2])
    c.add_argument("--random-ratio", type=float, default=4.0)
    c.add_argument("--solvers", nargs="+", choices=["gmres", "bicgstab"], default=["gmres", "bicgstab"])
    c.add_argument("--degrees", nargs="+", type=int, default=[0, 1, 2, 3])
    for p in (b, c):
        p.add_argument("--repeats", type=int, default=3)
        p.add_argument("--output", help="write JSON results to this file")
    args = parser.parse_args()
    result = dict(mode=args.mode, arguments={k: v for k, v in vars(args).items()},
                  environment=environment(), reference_k_valkin=VALKIN,
                  notes="provisional Bernal map; best-of-N solve time; tolerances "
                        "tol_k=1e-9, tol_source=1e-8, inner_rtol_floor=1e-11")
    result.update(bernal(args) if args.mode == "bernal" else contrast(args))
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")


if __name__ == "__main__":
    main()
