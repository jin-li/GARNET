"""Neutron-only prompt-event and exponential-growth work comparison.

python3 tools/assess_exponential_events.py

Prescribed graded meshes isolate resolved start-up; they are not an adaptive
FTM controller. Optional frequency correction re-solves the actual FV endpoint.
All additional solves and capped frequency defects are recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from garnet import DiffusionEigenSolver, Grid, Kinetics, Material, TransientSolver


ROOT = Path(__file__).resolve().parents[1]
CASES = {"prompt-rise": (0.002, 1.0), "superprompt-rise": (0.01, 0.5),
         "shutdown": (-0.05, 0.2)}


def graded_widths(duration, maximum, *, initial=1e-5, growth=1.1):
    """Fixed geometric mesh, constructed without using a reference solution."""
    widths, time, width = [], 0.0, initial
    while time < duration-1e-14:
        width = min(width, maximum, duration-time)
        widths.append(width)
        time += width
        width *= growth
    return widths


def configurations():
    rows = [{"name": f"BE-h{h:g}", "scheme": "backward-euler", "dt": h}
            for h in (0.001, 0.00025, 0.00005)]
    for h in (0.01, 0.005, 0.002, 0.001):
        rows.extend({"name": f"{scheme}-uniform-h{h:g}", "scheme": scheme, "dt": h}
                    for scheme in ("crank-nicolson", "ftm2"))
        rows.append({"name": f"BE-graded-h{h:g}", "scheme": "backward-euler",
                     "dt": h, "graded": True})
        for label, scheme, options in (
                ("zero-frequency-ftm2", "ftm2", {"frequency": 0.0, "startup_be_steps": 2}),
                ("ftm2", "ftm2", {"startup_be_steps": 2}),
                ("corrected-ftm2", "ftm2", {"startup_be_steps": 2, "frequency_iterations": 4}),
                ("corrected-ftm1", "ftm", {"frequency_iterations": 4})):
            rows.append({"name": f"{label}-graded-h{h:g}", "scheme": scheme,
                         "dt": h, "graded": True, "ftm_kwargs": options})
    rows.append({"name": "ftm2-uniform-BE-startup-h0.01", "scheme": "ftm2", "dt": 0.01,
                 "ftm_kwargs": {"startup_be_steps": 2}})
    for order in (1, 5):
        for rtol in ((1e-4, 1e-5, 1e-6) if order == 1 else
                     (1e-4, 1e-5, 1e-6, 1e-7)):
            rows.append({"name": f"adaptive-bdf{order}-rtol{rtol:g}", "scheme": f"bdf{order}",
                         "dt": 1e-5, "adaptive": {"rtol": rtol, "automatic_order": order > 1}})
    return rows


def case(name, cells):
    rho, duration = CASES[name]
    beta, decay, velocity, absorption = 0.0065, 0.08, 2.2e5, 0.03
    generation = 1/(velocity*absorption)
    a = (rho-beta)/generation
    gap = np.sqrt((a+decay)**2+4*decay*beta/generation)
    slow, fast = (a-decay+gap)/2, (a-decay-gap)/2
    fraction = (rho/generation-fast)/(slow-fast)

    def exact(times):
        return fraction*np.exp(slow*times)+(1-fraction)*np.exp(fast*times)

    grid = Grid((cells, cells, 1), (float(cells), float(cells), 1.0))
    base = Material(diffusion=[1.3], sigma_a=[absorption], nu_sigma_f=[0.035])
    initial, final = [base], [base.replace(sigma_a=[absorption*(1-rho)])]
    problem = lambda t: (initial if t <= 0 else final, None)
    kin = Kinetics(velocities=[velocity], beta=[beta], decay=[decay])
    steady = DiffusionEigenSolver(grid, initial, bc="reflective", device="cpu").solve(
        tol_k=1e-12, tol_source=1e-11)
    output_times = np.linspace(0, duration, 8001)
    expected = exact(output_times)
    records = []
    for config in configurations():
        schedule = graded_widths(duration, config["dt"]) if config.get("graded") else config["dt"]
        options = {"ftm_kwargs": config["ftm_kwargs"]} if "ftm_kwargs" in config else {}
        if "adaptive" in config:
            options["adaptive_bdf"] = {**config["adaptive"], "max_dt": 0.02, "min_dt": 1e-11}
        try:
            result = TransientSolver(grid, problem, kin, bc="reflective", device="cpu").solve(
                t_end=duration, dt=schedule, time_scheme=config["scheme"], initial_steady=steady,
                step_solver="monolithic", tol_step=1e-10, multigroup_kwargs={"rtol": 1e-11}, **options)
        except (RuntimeError, ValueError) as error:
            records.append({"configuration": config, "status": "failed", "error": str(error)})
            print(name, config["name"], "failed:", str(error), flush=True)
            continue
        interpolated = np.interp(output_times, result.times, result.power)
        endpoint_error = float(np.max(np.abs(result.power/exact(result.times)-1)))
        output_error = float(np.max(np.abs(interpolated/expected-1)))
        defects = [d for d in result.step_frequency_log_defects if d is not None]
        tolerance = config.get("ftm_kwargs", {}).get("frequency_tolerance", 1e-6)
        row = {"configuration": config, "status": "completed",
               "max_endpoint_relative_power_error": endpoint_error,
               "max_common_grid_relative_power_error": output_error,
               "max_relative_power_error": max(endpoint_error, output_error),
               "accepted_steps": len(result.times)-1, "rejected_steps": result.rejected_steps,
               "neutron_block_solves": sum(result.feedback_iterations)+sum(result.rejected_feedback_iterations),
               "fgmres_iterations": sum(result.step_iterations)+sum(result.rejected_step_iterations),
               "pcg_iterations": result.total_inner_iterations,
               "frequency_refinements": sum(result.step_frequency_refinements),
               "maximum_final_frequency_log_defect": max(defects, default=None),
               "frequency_steps_above_tolerance": sum(d > tolerance for d in defects),
               "startup_BE_steps": result.ftm_fallback_reasons.count("startup-backward-euler"),
               "unsafe_frequency_BE_steps": result.ftm_fallback_reasons.count("frequency-bound"),
               "minimum_power": float(np.min(result.power)),
               "wrong_direction_increments": int(np.count_nonzero(np.diff(result.power)*rho < -1e-10))}
        assert row["neutron_block_solves"] == row["accepted_steps"]+row["rejected_steps"]+row["frequency_refinements"]
        records.append(row)
        print(name, config["name"], f"error={100*row['max_relative_power_error']:.5f}%",
              f"solves={row['neutron_block_solves']}", f"PCG={row['pcg_iterations']}", flush=True)
    return {"reactivity": rho, "duration_s": duration, "beta": beta, "decay_per_s": decay,
            "generation_time_s": generation, "initial_power": 1.0, "final_reference_power": float(exact(duration)),
            "reference": "Closed-form two-mode point kinetics, exactly matching the uniform FV problem",
            "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT/"benchmark-results/exponential-second-order/chunk2-events.json")
    args = parser.parse_args()
    sources = ["garnet/_exponential.py", "garnet/timescheme.py", "garnet/transient.py",
               "tools/assess_exponential_events.py"]
    report = {"scope": "Neutrons and delayed neutrons only; no heat solve",
              "mesh": {"cells_per_side": args.cells, "graded_initial_dt_s": 1e-5,
                       "graded_growth_factor": 1.1, "graded_mesh_is_adaptive": False},
              "error_sampling": "Maximum of own endpoints and common-grid linear output interpolation",
              "timing_claim": None,
              "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
              "cases": {}}
    for name in args.cases:
        report["cases"][name] = case(name, args.cells)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
