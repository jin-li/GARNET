"""Review exponential-theta advice using measured work and a scalar probe.

Run with GARNET installed in the active environment::

    python3 tools/assess_exponential_proposal.py

The ramp comparison executes the existing FV/FGMRES solver, including its
Crank--Nicolson scheme. The frequency-corrector experiment is a separate
one-family point-kinetics calculation using production scalar coefficients;
it does not implement a corrected/per-group scheme in the spatial solver.
No wall-time speedup is claimed from this single-run work comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

from garnet._exponential import ExponentialStep, population_frequency


ROOT = Path(__file__).resolve().parents[1]


def corrected_frequency_probe():
    beta, decay, generation, rho = 0.0065, 0.08, 1/6600, 0.002
    a, production = (rho-beta)/generation, beta/generation
    precursor0 = beta/(generation*decay)
    gap = math.sqrt((a+decay)**2 + 4*decay*production)
    slow, fast = (a-decay+gap)/2, (a-decay-gap)/2
    fraction = (a+decay*precursor0-fast)/(slow-fast)

    def exact(t):
        return fraction*math.exp(slow*t) + (1-fraction)*math.exp(fast*t)

    def frozen_step(n, precursor, width, frequency):
        coefficients = ExponentialStep(width, frequency)
        retention, integral, _ = coefficients.precursor(decay)
        new_n = ((coefficients.history_factor*n/width + decay*retention*precursor)
                 / (coefficients.diagonal_factor/width-a-decay*production*integral))
        return new_n, retention*precursor + production*integral*new_n

    rows = []
    assert math.isclose(exact(0), 1.0, abs_tol=1e-14)
    for width in (0.02, 0.01, 0.005, 0.002, 0.001):
        for method in ("backward-euler", "lagged-ftm", "corrected-ftm"):
            n, precursor, previous = 1.0, precursor0, None
            solves, error, largest_exponent, last_defect = 0, 0.0, 0.0, 0.0
            for step in range(round(1/width)):
                if method == "backward-euler":
                    new_n = ((n+width*decay*precursor/(1+decay*width))
                             / (1-width*a-width*width*decay*production/(1+decay*width)))
                    new_C = (precursor+width*production*new_n)/(1+decay*width)
                    solves += 1
                else:
                    frequency = (0.0 if previous is None else
                                 population_frequency(n, previous, width))
                    for iteration in range(100):
                        largest_exponent = max(largest_exponent, abs(frequency*width))
                        new_n, new_C = frozen_step(n, precursor, width, frequency)
                        solves += 1
                        observed = population_frequency(new_n, n, width)
                        last_defect = abs(observed-frequency)*width
                        if method == "lagged-ftm" or last_defect < 1e-11:
                            break
                        frequency = observed
                    else:
                        raise RuntimeError("point-kinetics frequency correction did not converge")
                assert all(math.isfinite(v) and v > 0 for v in (new_n, new_C))
                previous, n, precursor = n, new_n, new_C
                error = max(error, abs(n/exact((step+1)*width)-1))
            rows.append({"method": method, "width_s": width,
                         "accepted_steps": round(1/width), "linear_subsolves": solves,
                         "max_endpoint_relative_population_error": error,
                         "maximum_abs_frequency_times_width": largest_exponent,
                         "last_frequency_iteration_log_defect": last_defect})
    return {"scope": "Independent one-family point kinetics; not a spatial-solver corrector benchmark",
            "beta": beta, "decay_per_s": decay, "generation_time_s": generation,
            "reactivity": rho, "duration_s": 1.0,
            "corrector_log_tolerance": 1e-11, "corrector_max_iterations": 100,
            "error_sampling": "Each run's own step endpoints against the closed-form solution",
            "rows": rows}


def ramp_work_comparison():
    path = ROOT / "examples/exponential_transient_benchmark.py"
    spec = importlib.util.spec_from_file_location("ftm_work_comparison", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    run, reference, output_times, reference_error = benchmark.uniform_case("uniform-ramp", 4, "cpu")
    configurations = [{"name": f"{scheme}-h{width:g}", "scheme": scheme, "dt": width}
                      for scheme in ("backward-euler", "crank-nicolson", "ftm")
                      for width in (0.02, 0.01, 0.005, 0.002, 0.001)]
    configurations.extend({"name": f"adaptive-bdf{order}-rtol{rtol:g}",
                           "scheme": f"bdf{order}", "dt": 1e-5,
                           "adaptive": {"rtol": rtol, "automatic_order": order > 1}}
                          for order in (1, 5) for rtol in (1e-4, 1e-5))
    records = []
    for config in configurations:
        result = run(config, None)
        metrics = benchmark.summarize(result, reference, output_times, None)
        calls = sum(result.feedback_iterations)+sum(result.rejected_feedback_iterations)
        assert calls == metrics["accepted"]+metrics["rejected"]
        records.append({"configuration": config, "metrics": metrics,
                        "coupled_block_solves": calls})
    return {"case": "uniform-ramp", "scope": "Actual GARNET FV/FGMRES; work counts, not repeated timing",
            "reference_power_refinement_error": reference_error,
            "power_budget": 1e-3, "output_interpolation": "Common linear output interpolation for all methods",
            "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "benchmark-results/exponential-prototype")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = ["garnet/_exponential.py", "garnet/timescheme.py", "garnet/transient.py",
                    "examples/exponential_transient_benchmark.py", "tools/assess_exponential_proposal.py"]
    report = {"source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                for name in source_paths},
              "point_corrector_probe": corrected_frequency_probe(),
              "ramp_work_comparison": ramp_work_comparison()}
    (args.output_dir / "proposal-review.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    with (args.output_dir / "proposal-error-work.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["configuration", "maximum_power_error_percent",
                                                   "block_solves", "fgmres_iterations", "pcg_iterations"])
        writer.writeheader()
        for row in report["ramp_work_comparison"]["records"]:
            m = row["metrics"]
            writer.writerow({"configuration": row["configuration"]["name"],
                             "maximum_power_error_percent": 100*m["max_power_error"],
                             "block_solves": row["coupled_block_solves"],
                             "fgmres_iterations": m["outer_iterations"],
                             "pcg_iterations": m["inner_iterations"]})
            print(row["configuration"]["name"],
                  f"error={100*m['max_power_error']:.6f}%",
                  f"block solves={row['coupled_block_solves']}",
                  f"PCG={m['inner_iterations']}")


if __name__ == "__main__":
    main()
