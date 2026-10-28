"""Matched-error FTM / backward Euler / adaptive BDF prototype gate.

Run from a checkout with NumPy and SciPy available::

    python3 examples/exponential_transient_benchmark.py \
        --repeats 3 --output benchmark-results/exponential-prototype/results.json

Default cases are manufactured homogeneous one-group diffusion, with delayed
neutrons, and the actual split conduction/feedback coupler at two heat
capacities and thermal exchange widths. These isolate accuracy and software
overhead; they do not measure full-core throughput. The separate
``run_lra2d_cpu(time_scheme="ftm", ...)`` driver supports follow-up tests of
heterogeneous 2-D, fully implicit feedback on CPU.

Fixed BE, lagged FTM, zero-frequency FTM, adaptive BE and automatic BDF1--5
are compared on identical physical models. Coupled cases also force endpoint
heat quadrature in an FTM control. All curves use the same piecewise-linear
output interpolation. Speed ratios are reported only among rows passing the
declared power, energy and temperature budgets. No equal-rtol speedup is used.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.integrate import solve_ivp

from garnet import DiffusionEigenSolver, Grid, Kinetics, Material, TransientSolver
from garnet.backend import asnumpy, synchronize, get_backend
from garnet.coupling import CoupledSolver, CouplingContext, coupled_transient
from garnet.feedback import ThermalFeedback
from garnet.thermal import ThermalMaterial


def configurations(widths, rtols, coupled):
    rows = []
    for scheme in ("backward-euler", "ftm"):
        for width in widths:
            rows.append({"name": f"{scheme}-h{width:g}", "scheme": scheme, "dt": width})
    rows.append({"name": "zero-frequency-control", "scheme": "ftm",
                 "dt": widths[len(widths)//2], "ftm_kwargs": {"frequency": 0.0}})
    if coupled:
        rows.extend({"name": f"ftm-endpoint-heat-h{h:g}", "scheme": "ftm",
                     "dt": h, "power_quadrature": "endpoint"} for h in widths)
    for order in (1, 5):
        for rtol in rtols:
            rows.append({"name": f"adaptive-bdf{order}-rtol{rtol:g}",
                         "scheme": f"bdf{order}", "dt": 1e-5,
                         "adaptive": {"rtol": rtol, "automatic_order": order > 1}})
    return rows


def step_options(config, max_dt):
    options = {"time_scheme": config["scheme"], "tol_step": 1e-10,
               "step_solver": "monolithic", "multigroup_kwargs": {"rtol": 1e-11},
               "max_feedback_iterations": 30}
    if "ftm_kwargs" in config:
        options["ftm_kwargs"] = config["ftm_kwargs"]
    if "adaptive" in config:
        options["adaptive_bdf"] = {**config["adaptive"], "max_dt": max_dt}
    return options


def uniform_case(name, cells, device):
    coupled = name.startswith("coupled-")
    duration, velocity, absorption, beta, decay = 1.0, 2.2e5, 0.03, 0.0065, 0.08
    generation = 1/(velocity*absorption)
    capacity = 0.5 if name == "coupled-fast" else 10.0
    doppler = 0.03 if coupled else 0.0
    reference_T, sink_T, heat_source, sink = 600.0, 590.0, 10.0, 1.0
    grid = Grid(shape=(cells, cells, 1), size=(float(cells), float(cells), 1.0))
    base = Material(diffusion=[1.3], sigma_a=[absorption], nu_sigma_f=[0.035])
    initial_materials = [base]
    kinetics = Kinetics(velocities=[velocity], beta=[beta], decay=[decay])
    mapping = np.zeros(grid.shape, dtype=int)

    def rho(t):
        return 0.002 if name == "uniform-step" else 0.002*min(max(t/0.2, 0.0), 1.0)

    @lru_cache(maxsize=8192)
    def materials_at_strength(strength):
        return [base.replace(sigma_a=base.sigma_a*(1-strength))]

    def problem_at(t):
        return (initial_materials if t <= 0 else materials_at_strength(rho(t)), mapping)

    def rhs(t, y):
        n, precursor, temperature, _energy = y
        effective_absorption = absorption*(1-rho(t))*(
            1 + doppler*(np.sqrt(temperature)-np.sqrt(reference_T)))
        reactivity = 1-effective_absorption/absorption
        return [(reactivity-beta)*n/generation + decay*precursor,
                beta*n/generation-decay*precursor,
                ((heat_source*n-sink*(temperature-sink_T))/capacity if coupled else 0.0),
                n]

    initial = [1.0, beta/(generation*decay), reference_T, 0.0]
    reference = solve_ivp(rhs, (0, duration), initial, method="Radau",
                          rtol=3e-11, atol=1e-12, dense_output=True)
    tighter = solve_ivp(rhs, (0, duration), initial, method="Radau",
                        rtol=1e-11, atol=3e-13, dense_output=True)
    if not reference.success or not tighter.success:
        raise RuntimeError("independent point-kinetics/thermal reference failed")
    output_times = np.linspace(0, duration, 4001)
    reference_error = float(np.max(np.abs(
        reference.sol(output_times)[0]/tighter.sol(output_times)[0]-1)))
    if reference_error > 1e-8:
        raise RuntimeError(f"reference is not converged: {reference_error:g}")

    if coupled:
        context = CouplingContext(
            grid, initial_materials, mapping,
            [ThermalMaterial(1.0, sink_coeff=sink, sink_temperature=sink_T,
                             heat_capacity=capacity)],
            ThermalFeedback(t_ref=[reference_T], doppler=[doppler]),
            total_power=heat_source*grid.n_cells, solver_cls=DiffusionEigenSolver,
            kinetics=kinetics, device=device)
        steady = CoupledSolver(context).solve(tol=1e-10)
    else:
        steady = DiffusionEigenSolver(grid, initial_materials, mapping,
                                      bc="reflective", device=device).solve(
                                          tol_k=1e-11, tol_source=1e-10)

    def run(config, thermal_width):
        options = step_options(config, thermal_width if coupled else 0.1)
        dt = config.get("schedule", config["dt"])
        if coupled:
            return coupled_transient(
                context, t_end=duration, dt=dt, dt_thermal=thermal_width,
                problem_at=problem_at, initial_coupled_steady=steady,
                transient_kwargs=options, profile=True,
                power_quadrature=config.get("power_quadrature", "scheme"))
        return TransientSolver(grid, problem_at, kinetics, bc="reflective", device=device).solve(
            t_end=duration, dt=dt, initial_steady=steady, **options)

    return run, reference.sol, output_times, reference_error


def summarize(result, reference, output_times, thermal_width):
    coupled = hasattr(result, "counters")
    interpolated = np.interp(output_times, result.times, result.power)
    expected = reference(output_times)
    endpoint_expected = reference(result.times)[0]
    source_weights = np.asarray(result.step_source_weights)
    energy = float(np.dot(result.power[1:], source_weights))
    row = {
        "march_seconds": float(result.seconds if coupled else result.solve_seconds),
        "accepted": len(result.times)-1, "rejected": result.rejected_steps,
        "max_power_error": float(np.max(np.abs(interpolated/expected[0]-1))),
        "max_endpoint_power_error": float(np.max(np.abs(result.power/endpoint_expected-1))),
        "relative_energy_error": float(abs(energy/expected[3, -1]-1)),
        "max_exchange_temperature_error_k": 0.0,
        "inner_iterations": (result.counters["neutron_inner_iterations"] if coupled
                             else result.total_inner_iterations),
        "outer_iterations": (result.counters["neutron_total_outer_iterations"] if coupled
                             else sum(result.step_iterations)+sum(result.rejected_step_iterations)),
        "operator_rebuilds": (result.counters["neutron_operator_rebuilds"] if coupled
                              else result.operator_rebuilds),
        "ftm_fallbacks": sum(bool(reason) for reason in result.ftm_fallback_reasons),
        "frequency_range": ([float(min(result.step_frequencies)), float(max(result.step_frequencies))]
                            if result.step_frequencies else None),
        "thermal_updates": result.counters.get("thermal_steps", 0) if coupled else 0,
    }
    if coupled:
        exchanges = np.arange(thermal_width, output_times[-1]+thermal_width/2, thermal_width)
        actual = np.interp(exchanges, result.times, result.mean_temperature)
        row["max_exchange_temperature_error_k"] = float(np.max(np.abs(
            actual-reference(exchanges)[2])))
    return row


def benchmark_case(args, case, thermal_width=None):
    started = time.perf_counter()
    run, reference, output_times, reference_error = uniform_case(case, args.cells, args.device)
    setup_seconds = time.perf_counter()-started
    widths = [h for h in args.widths if thermal_width is None or
              (h <= thermal_width and np.isclose(round(thermal_width/h)*h, thermal_width))]
    if not widths:
        raise ValueError("no fixed width aligns with the requested thermal exchange")
    configs = configurations(widths, args.rtols, thermal_width is not None)
    if args.replay_adaptive_grids:
        replays = []
        for guide in [config for config in configs if "adaptive" in config]:
            result = run(guide, thermal_width)
            replays.append({"name": f"ftm-replay-{guide['name']}", "scheme": "ftm",
                            "dt": guide["dt"], "schedule": np.diff(result.times).tolist(),
                            "guide": guide["name"],
                            "guide_generation_seconds": float(
                                result.seconds if thermal_width is not None else result.solve_seconds)})
        configs.extend(replays)
    records = {config["name"]: {"configuration": config, "samples": []} for config in configs}
    xp = get_backend(args.device)
    if args.device == "gpu":
        for config in configs:
            run(config, thermal_width)  # warm kernels and allocator before timing
        synchronize(xp)
    for repeat in range(args.repeats):
        order = list(configs)
        random.Random(103+repeat).shuffle(order)
        for config in order:
            record = records[config["name"]]
            synchronize(xp)
            start = time.perf_counter()
            try:
                result = run(config, thermal_width)
                synchronize(xp)
                application_seconds = time.perf_counter()-start
                row = summarize(result, reference, output_times, thermal_width)
                row["application_seconds"] = application_seconds
                record["samples"].append(row)
                record["times"] = asnumpy(result.times).tolist()
                record["power"] = asnumpy(result.power).tolist()
                record["step_source_weights"] = list(result.step_source_weights)
                record["step_frequencies"] = list(result.step_frequencies)
            except (RuntimeError, ValueError) as error:
                record.setdefault("failures", []).append(str(error))
    for record in records.values():
        samples = record["samples"]
        record["passes_accuracy"] = len(samples) == args.repeats and all(
            sample["max_power_error"] <= args.power_budget
            and (thermal_width is None or sample["relative_energy_error"] <= args.energy_budget)
            and sample["max_exchange_temperature_error_k"] <= args.temperature_budget
            for sample in samples)
        record["median_march_seconds"] = (float(np.median([
            sample["march_seconds"] for sample in samples])) if samples else None)
    best = {}
    for family in ("backward-euler", "ftm", "bdf1", "bdf5"):
        valid = [record for record in records.values() if record["passes_accuracy"]
                 and record["configuration"]["scheme"] == family
                 and "ftm_kwargs" not in record["configuration"]
                 and "power_quadrature" not in record["configuration"]
                 and "guide" not in record["configuration"]]
        best[family] = min(valid, key=lambda row: row["median_march_seconds"]) if valid else None
    ftm = best["ftm"]
    speedups = {family: (best[family]["median_march_seconds"]/ftm["median_march_seconds"]
                         if ftm is not None and best[family] is not None else None)
                for family in ("backward-euler", "bdf1", "bdf5")}
    replay_diagnostics = [{"configuration": record["configuration"]["name"],
                           "passes_accuracy": record["passes_accuracy"],
                           "replay_only_median_seconds": record["median_march_seconds"],
                           "with_guide_generation_seconds": (
                               record["median_march_seconds"]
                               + record["configuration"]["guide_generation_seconds"]
                               if record["median_march_seconds"] is not None else None)}
                          for record in records.values() if "guide" in record["configuration"]]
    return {"case": case, "thermal_width": thermal_width, "setup_and_reference_seconds": setup_seconds,
            "reference_power_refinement_error": reference_error,
            "records": list(records.values()),
            "best_passing_configuration": {family: (record["configuration"]["name"] if record else None)
                                            for family, record in best.items()},
            "ftm_speedup_over_best_passing": speedups,
            "replay_diagnostics": replay_diagnostics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs="+", choices=("uniform-step", "uniform-ramp", "coupled-slow", "coupled-fast"),
                        default=["uniform-step", "uniform-ramp", "coupled-slow", "coupled-fast"])
    parser.add_argument("--widths", type=float, nargs="+", default=[0.02, 0.01, 0.005, 0.002, 0.001])
    parser.add_argument("--rtols", type=float, nargs="+", default=[1e-3, 1e-4])
    parser.add_argument("--thermal-widths", type=float, nargs="+", default=[0.02, 0.005])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--replay-adaptive-grids", action="store_true",
                        help="diagnostic FTM replays; report grid-generation cost separately")
    parser.add_argument("--power-budget", type=float, default=0.005)
    parser.add_argument("--energy-budget", type=float, default=0.005)
    parser.add_argument("--temperature-budget", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.cells < 1:
        parser.error("repeats and cells must be positive")
    if any(not np.isfinite(v) or v <= 0 for v in
           [*args.widths, *args.rtols, *args.thermal_widths,
            args.power_budget, args.energy_budget, args.temperature_budget]):
        parser.error("widths, tolerances and budgets must be finite and positive")
    if any(not np.isclose(round(1/h)*h, 1.0, rtol=1e-12, atol=1e-14)
           for h in args.widths+args.thermal_widths):
        parser.error("fixed widths and thermal widths must divide the one-second horizon")
    root = Path(__file__).resolve().parents[1]
    sources = ["garnet/_exponential.py", "garnet/timescheme.py", "garnet/transient.py",
               "garnet/coupling.py", "examples/exponential_transient_benchmark.py"]
    report = {"python": platform.python_version(), "numpy": np.__version__,
              "scipy": scipy.__version__, "device": args.device,
              "source_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                                 check=True, capture_output=True, text=True).stdout.strip(),
              "source_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                for name in sources},
              "arguments": sys.argv[1:],
              "thread_controls": {key: os.environ.get(key) for key in
                                  ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
              "scope": "Uniform one-group diffusion mechanism/cost study; not full-core throughput",
              "output_interpolation": "piecewise-linear for every method",
              "budgets": {"power": args.power_budget, "energy": args.energy_budget,
                          "temperature_k": args.temperature_budget}, "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in args.case:
        for thermal_width in (args.thermal_widths if case.startswith("coupled-") else [None]):
            result = benchmark_case(args, case, thermal_width)
            report["cases"].append(result)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            print(case, "thermal_dt=", thermal_width,
                  "best=", result["best_passing_configuration"],
                  "FTM speed ratios=", result["ftm_speedup_over_best_passing"], flush=True)


if __name__ == "__main__":
    main()
