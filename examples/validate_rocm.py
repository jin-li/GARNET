"""Validate an actual CuPy/HIP installation and write a reproducible JSON report.

Run from the repository root: python examples/validate_rocm.py --sizes 8 12
Exit codes: 0 validated, 1 numerical/runtime failure, 2 HIP unavailable.
Timings include fresh solver construction and solve; the first run is reported
separately from at least three warmed runs. No speedup is required.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ndgpu import kernels
from ndgpu.backend import asnumpy, get_backend, synchronize
from ndgpu.grid import Grid
from ndgpu.materials import Kinetics, Material
from ndgpu.solver import DiffusionEigenSolver, SP3EigenSolver
from ndgpu.transient import TransientSolver

EIGEN_KW = dict(tol_k=1e-9, tol_source=1e-8, max_outer=500)
TOLERANCES = dict(k_abs=2e-7, normalized_flux_rel_l2=2e-5,
                  power_abs=2e-5, temperature_abs_K=2e-3)


def error(exc):
    return f"{type(exc).__name__}: {exc}"


def json_safe(value):
    """Keep a failed numerical run serializable as strict JSON."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def problem(size, *, cube=False):
    """Fixed heterogeneous two-group square or cube, in cm, no random inputs."""
    grid = Grid((size, size, size if cube else 1), (60.0, 60.0, 60.0 if cube else 1.0))
    fuel = Material([1.4, .4], [.01, .08], [.008, .12],
                    [[0, .02], [0, 0]], name="validation fuel")
    reflector = Material([1.6, .5], [.003, .025], [0, 0],
                         [[0, .025], [0, 0]], name="validation reflector")
    mmap = np.ones(grid.shape, dtype=np.int64)
    lo, hi = size // 4, size - size // 4
    mmap[lo:hi, lo:hi, slice(lo, hi) if cube else slice(None)] = 0
    return grid, [fuel, reflector], mmap


def factory(kind, size, device, hpmr_refine):
    if kind == "hpmr_thermal":
        from ndgpu.benchmarks.hpmr import build_hpmr2d
        from ndgpu.benchmarks.hpmr_thermal import build_hpmr_coupling
        from ndgpu.coupling import CoupledSolver
        p = build_hpmr2d(refine=hpmr_refine, drum_angle_deg=120,
                         absorber="polar", samples=6)
        ctx = build_hpmr_coupling(p, device=device, eigen_kwargs=EIGEN_KW)
        return lambda: CoupledSolver(ctx).solve(tol=2e-6, max_iter=30,
                                                anderson_depth=4)
    grid, mats, mmap = problem(size, cube=kind == "diffusion3d")
    kwargs = dict(bc=("zero-flux", "zero-flux",
                      "zero-flux" if kind == "diffusion3d" else "reflective"), device=device)
    if kind == "transient":
        kin = Kinetics([2e7, 2e5], [.0065], [.08])
        # A deterministic 0.1% fission increase after the initial eigenpair.
        perturbed = Material(mats[0].diffusion, mats[0].sigma_a,
                             mats[0].nu_sigma_f * 1.001, mats[0].sigma_s)
        def at(t):
            return ([mats[0] if t == 0 else perturbed, mats[1]], mmap)
        solver = TransientSolver(grid, at, kin, **kwargs)
        return lambda: solver.solve(t_end=.003, dt=.001, tol_step=1e-8,
                                     steady_kwargs=EIGEN_KW)
    cls = DiffusionEigenSolver if kind in ("diffusion", "diffusion3d") else SP3EigenSolver
    solver = cls(grid, mats, mmap, **kwargs)
    return lambda: solver.solve(**EIGEN_KW)


def summarize(result, xp, kind):
    flux = result.flux
    if not isinstance(flux, xp.ndarray):
        raise TypeError(f"expected {xp.__name__}.ndarray output, got {type(flux)}")
    host = asnumpy(flux)
    if not np.all(np.isfinite(host)) or np.linalg.norm(host) == 0:
        raise ValueError("nonfinite or zero flux")
    info = dict(array_type=f"{type(flux).__module__}.{type(flux).__name__}",
                flux_shape=list(host.shape))
    if kind == "transient":
        if not np.all(np.isfinite(result.power)):
            raise ValueError("nonfinite transient power")
        info.update(k_eff=float(result.k0), converged=bool(result.steady.converged),
                    power=result.power.tolist(), times=result.times.tolist(),
                    step_iterations=result.step_iterations,
                    inner_iterations=int(result.total_inner_iterations))
        if len(result.times) != 4:
            raise ValueError("expected exactly three transient steps")
    else:
        info.update(k_eff=float(result.k_eff), converged=bool(result.converged))
        if kind == "hpmr_thermal":
            if not np.all(np.isfinite(result.temperature)):
                raise ValueError("nonfinite coupled temperature")
            info.update(coupling_iterations=result.iterations,
                        coupling_residuals=result.residual_history,
                        temperature_min_K=float(result.temperature.min()),
                        temperature_max_K=float(result.temperature.max()),
                        neutronics_converged=bool(result.neutronics.converged))
            info["converged"] &= info["neutronics_converged"]
        else:
            info.update(outer_iterations=result.outer_iterations,
                        inner_iterations=result.inner_iterations,
                        final_source_error=float(result.source_error_history[-1]))
    if not np.isfinite(info["k_eff"]):
        raise ValueError("nonfinite eigenvalue")
    return info


def compare(reference, candidate, kind):
    a, b = asnumpy(reference.flux), asnumpy(candidate.flux)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    k_a = reference.k0 if kind == "transient" else reference.k_eff
    k_b = candidate.k0 if kind == "transient" else candidate.k_eff
    diffs = dict(k_abs=float(abs(k_a - k_b)),
                 normalized_flux_rel_l2=float(np.linalg.norm(a - b)))
    if kind == "transient":
        diffs["power_abs"] = float(np.max(np.abs(reference.power - candidate.power)))
    if kind == "hpmr_thermal":
        diffs["temperature_abs_K"] = float(np.max(np.abs(
            reference.temperature - candidate.temperature)))
    return dict(differences=diffs,
                passed=all(np.isfinite(v) and v <= TOLERANCES[k]
                           for k, v in diffs.items()))


def benchmark(kind, size, device, xp, repeats, hpmr_refine, reference=None):
    runs, first = [], None
    for index in range(repeats + 1):
        synchronize(xp)
        start = time.perf_counter()
        result = factory(kind, size, device, hpmr_refine)()
        synchronize(xp)
        elapsed = time.perf_counter() - start
        info = summarize(result, xp, kind)
        info["wall_seconds"] = elapsed
        if reference is not None:
            info["comparison"] = compare(reference, result, kind)
        if first is None:
            first = result
        runs.append(info)
    passed = all(r["converged"] and r.get("comparison", {}).get("passed", True)
                 for r in runs)
    return first, dict(passed=passed, initial_run=runs[0], warm_runs=runs[1:],
                       warm_median_seconds=float(np.median(
                           [r["wall_seconds"] for r in runs[1:]])))


def graph_probe(xp, size, *, force_stream_error=False):
    from ndgpu.linalg import PCGWorkspace, pcg
    grid, mats, mmap = problem(size)
    solver = DiffusionEigenSolver(grid, mats, mmap, device="gpu",
        bc=("zero-flux", "zero-flux", "reflective"))
    op = solver.ops[0]
    b = xp.linspace(.1, 1., int(np.prod(grid.shape)), dtype=xp.float64).reshape(grid.shape)
    workspace = PCGWorkspace.like(b, operator_out=True)
    stream_class = xp.cuda.Stream
    def unavailable_stream(*args, **kwargs):
        raise NotImplementedError("validation-injected unsupported stream capture")
    try:
        if force_stream_error:
            xp.cuda.Stream = unavailable_stream
        x, iterations = pcg(op.apply, b, xp.zeros_like(b), op.inv_diag, xp,
                            rtol=1e-9, check_every=4, graph_block=4,
                            workspace=workspace)
    finally:
        xp.cuda.Stream = stream_class
    synchronize(xp)
    residual = float(xp.linalg.norm(op.apply(x) - b) / xp.linalg.norm(b))
    path_checked = (not kernels.fused_enabled() or workspace.graph_error is not None
                    or workspace.graph_replays > 0)
    return dict(requested=True, forced_stream_error=force_stream_error,
                capture_or_fallback_exercised=path_checked,
                captures=workspace.graph_captures,
                replays=workspace.graph_replays, fallback_reason=workspace.graph_error,
                iterations=iterations, relative_residual=residual,
                passed=bool(path_checked and np.isfinite(residual) and residual < 1e-7))


def sparse_probe(xp):
    from cupyx.scipy.sparse import csr_matrix
    dense = np.array([[3., -1., 0.], [-1., 4., -1.], [0., -1., 2.]])
    vector = np.array([1., 2., 3.])
    value = csr_matrix(xp.asarray(dense)) @ xp.asarray(vector)
    synchronize(xp)
    difference = float(np.max(np.abs(asnumpy(value) - dense @ vector)))
    return dict(status="supported", max_abs_error=difference,
                passed=bool(difference < 1e-12))


def dsa_probe(xp, triangular):
    """Exercise the real accelerator factory with both sides of the bus."""
    from types import SimpleNamespace
    from scipy.sparse import csr_matrix
    from ndgpu.sn import SNTransportSolver
    from ndgpu.tri_sn import TriSNTransportSolver
    cls = TriSNTransportSolver if triangular else SNTransportSolver
    dense = np.array([[2., -1.], [-1., 2.]])
    b = np.array([1., 0.])
    expected = np.linalg.solve(dense, b)
    owner = SimpleNamespace(xp=xp, dsa_on_device=True,
                            dsa_rtol=1e-10, dsa_maxiter=500)
    solve = cls._make_diff_solver(owner, csr_matrix(dense), *([True] if triangular else []))
    runs = {}
    for backend, rhs in (("host", b), ("device", xp.asarray(b))):
        out = solve(rhs)
        synchronize(xp)
        correct_type = isinstance(out, np.ndarray if backend == "host" else xp.ndarray)
        difference = float(np.max(np.abs(asnumpy(out) - expected)))
        residual = float(np.linalg.norm(dense @ asnumpy(out) - b))
        runs[backend] = dict(array_type=f"{type(out).__module__}.{type(out).__name__}",
                             max_abs_error=difference, residual_l2=residual,
                             passed=bool(correct_type and difference < 1e-6 and residual < 1e-6))
    return dict(status="supported", runs=runs, passed=all(r["passed"] for r in runs.values()))


def trisn_probe(xp):
    from ndgpu.tri import TriGrid
    from ndgpu.tri_sn import TriSNTransportSolver
    grid = TriGrid((3, 3, 2), side=10.)
    material = Material([1.], [.08], [.12], total=[.3])
    kwargs = dict(n_polar=1, n_azi=4, engine="levels", bc="vacuum",
                  acceleration="si", outer_acceleration="power")
    cpu = TriSNTransportSolver(grid, material, device="cpu", graphs=False,
                               **kwargs).solve(tol_k=1e-9, tol_source=1e-8)
    rows = {}
    for graphs in (False, True):
        solver = TriSNTransportSolver(grid, material, device="gpu", graphs=graphs, **kwargs)
        synchronize(xp)
        result = solver.solve(tol_k=1e-9, tol_source=1e-8)
        # Public TriSNResult intentionally returns host flux; verify the actual
        # levels sweep returns device-resident arrays as well.
        sweep_flux, _ = solver._sweep_dev(0, xp.ones(solver.N, dtype=xp.float64))
        synchronize(xp)
        comparison = compare(cpu, result, "trisn")
        rows["graphs_on" if graphs else "graphs_off"] = dict(
            converged=bool(result.converged), k_eff=float(result.k_eff),
            graphs_requested=graphs, graphs_active=solver.graphs_active,
            graph_fallback_reason=solver._graph_error,
            sweep_array_type=f"{type(sweep_flux).__module__}.{type(sweep_flux).__name__}",
            comparison=comparison,
            passed=bool(cpu.converged and result.converged and comparison["passed"]
                        and isinstance(sweep_flux, xp.ndarray)))
    return dict(status="supported", runs=rows, passed=all(r["passed"] for r in rows.values()),
                limitation="Small vacuum levels smoke case with source iteration; "
                "public eigen flux is host-resident by API. DSA factories tested separately.")


def event_profile_probe(xp):
    """Exercise event profiling through the public coupled-transient API."""
    from ndgpu import HexLattice, ThermalMaterial
    fuel = Material([1.2], [.025], [.030], [[0.]], name="profile fuel")
    core = (HexLattice(pitch=10., refine=1).set_disk(0, fuel)
            .set_boundary("reflective")
            .set_kinetics(Kinetics([220000.], [.0065], [.08])).build())
    core.configure_thermal({"profile fuel": ThermalMaterial(
        conductivity=.25, sink_coeff=.02, sink_temperature=650.,
        heat_capacity=2.5)}, total_power=1., ambient_temperature=650.)
    result = core.coupled_transient(
        t_end=.01, dt=.01, dt_thermal=.01,
        device="cpu" if xp is np else "gpu", profile=True)
    seconds = result.phase_seconds
    return dict(status="supported", seconds=seconds,
                passed=bool(seconds and all(np.isfinite(v) and v >= 0
                                          for v in seconds.values())
                            and np.allclose(result.power, 1., atol=1e-6)
                            and isinstance(result.flux, xp.ndarray)))


def optional(fn):
    try:
        return fn()
    except (ImportError, NotImplementedError) as exc:
        return dict(status="unsupported", error=error(exc))
    except Exception as exc:
        return dict(status="failed", error=error(exc))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/rocm-validation.json"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 12],
                        help="square cells per side, minimum 4 (default: 8 12)")
    parser.add_argument("--repeats", type=int, default=3, help="warm fresh runs, minimum 3")
    parser.add_argument("--cube-sizes", type=int, nargs="*", default=[],
                        help="optional 3D diffusion benchmark cells per side (minimum 4)")
    parser.add_argument("--hpmr-refine", type=int, default=1,
                        help="HP-MR polar mesh refinement (default: 1, smoke test only)")
    args = parser.parse_args(argv)
    if min(args.sizes + args.cube_sizes) < 4 or args.repeats < 3 or args.hpmr_refine < 1:
        parser.error("sizes >= 4, repeats >= 3 and hpmr-refine >= 1 required")
    report = dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  status="failed", parameters=dict(sizes=args.sizes, repeats=args.repeats,
                  cube_sizes=args.cube_sizes, hpmr_refine=args.hpmr_refine,
                  dtype="float64", eigen_kwargs=EIGEN_KW,
                  transient_steps=3, dt_seconds=.001), tolerances=TOLERANCES,
                  timing_policy="construction + solve; synchronized; fresh identical starts; "
                  "initial run separate; warmed process/kernel caches; no speedup assertion",
                  scope_limitations=["Synthetic small grids test backend equivalence, not reactor accuracy.",
                      "HP-MR uses built-in two-group placeholder materials and a coarse polar mesh; "
                      "thermal coupling is an internal smoke test, not a mesh-converged benchmark.",
                      "DSA factories, CSR and TriSN levels are optional separately reported probes; "
                      "their failures do not certify those paths or invalidate the four core workload gates."],
                  software=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__),
                  environment={key: os.environ.get(key) for key in (
                      "ROCM_HOME", "HIP_PATH", "CUPY_CACHE_DIR", "OMP_NUM_THREADS",
                      "OPENBLAS_NUM_THREADS", "HSA_OVERRIDE_GFX_VERSION")},
                  cases=[])
    exit_code = 1
    previous = kernels.fused_enabled()
    try:
        xp = get_backend("gpu")  # Never accept the automatic NumPy fallback.
        report["software"]["cupy"] = xp.__version__
        if not bool(xp.cuda.runtime.is_hip):
            raise RuntimeError("ROCm validation requires cupy.cuda.runtime.is_hip == True")
        probe = xp.arange(4, dtype=xp.float64)
        if not isinstance(probe, xp.ndarray):
            raise TypeError("GPU backend did not return CuPy arrays")
        synchronize(xp)
        props = xp.cuda.runtime.getDeviceProperties(xp.cuda.runtime.getDevice())
        report["hardware"] = {str(k): (v.decode(errors="replace") if isinstance(v, bytes)
                                       else v) for k, v in props.items()}
        report["software"].update(is_hip=True,
            runtime_version=xp.cuda.runtime.runtimeGetVersion(),
            driver_version=xp.cuda.runtime.driverGetVersion())
        for package in ("ndgpu", "scipy"):
            try:
                report["software"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                report["software"][package] = "not installed as distribution"
        case_specs = [(size, kind) for size in args.sizes
                      for kind in ("diffusion", "sp3", "transient", "hpmr_thermal")
                      if kind != "hpmr_thermal" or size == args.sizes[0]]
        case_specs += [(size, "diffusion3d") for size in args.cube_sizes]
        for size, kind in case_specs:
            case = dict(kind=kind, size=size if kind != "hpmr_thermal" else None)
            report["cases"].append(case)
            try:
                kernels.set_fused(False)
                ref, case["cpu"] = benchmark(kind, size, "cpu", np,
                    args.repeats, args.hpmr_refine)
                for fused in (False, True):
                    kernels.set_fused(fused)
                    _, case["gpu_fused" if fused else "gpu_unfused"] = benchmark(
                        kind, size, "gpu", xp, args.repeats, args.hpmr_refine, ref)
                case["passed"] = all(case[k]["passed"] for k in
                                     ("cpu", "gpu_unfused", "gpu_fused"))
            except Exception as exc:
                case.update(passed=False, error=error(exc))
            print(f"{kind} size={case['size']}: {'PASS' if case['passed'] else 'FAIL'}", flush=True)
        report["pcg_graph"] = {}
        for fused in (False, True):
            kernels.set_fused(fused)
            report["pcg_graph"]["fused" if fused else "unfused"] = optional(
                lambda: graph_probe(xp, args.sizes[0]))
        report["pcg_graph"]["injected_stream_failure"] = optional(
            lambda: graph_probe(xp, args.sizes[0], force_stream_error=True))
        report["sparse_csr"] = optional(lambda: sparse_probe(xp))
        report["dsa"] = {"sn": optional(lambda: dsa_probe(xp, False)),
                         "trisn": optional(lambda: dsa_probe(xp, True))}
        report["trisn_levels"] = optional(lambda: trisn_probe(xp))
        report["coupled_event_profile"] = optional(lambda: event_profile_probe(xp))
        from ndgpu.profiling import operator_profile
        grid, mats, mmap = problem(args.sizes[0])
        solver = DiffusionEigenSolver(grid, mats, mmap, device="gpu")
        report["operator_profile"] = optional(lambda: operator_profile(
            solver.ops[0], xp, grid.shape, n_repeat=20))
        passed = all(c["passed"] for c in report["cases"])
        passed &= all(p.get("passed", False) for p in report["pcg_graph"].values())
        # Missing sparse support is recorded, while a wrong supported answer fails.
        passed &= report["sparse_csr"].get("passed", True)
        report["status"] = "passed" if passed else "failed"
        exit_code = 0 if passed else 1
    except Exception as exc:
        report["error"] = error(exc)
        if "cases" in report and not report["cases"]:
            report["status"] = "hip_unavailable"
            exit_code = 2
    finally:
        kernels.set_fused(previous)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(json_safe(report), indent=2,
                                          default=str, allow_nan=False) + "\n")
    print(f"{report['status']}: {args.output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
