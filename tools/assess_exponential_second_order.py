"""Small neutron-only FTM2 work comparison on the smooth diffusion ramp.

Run with GARNET installed: python3 tools/assess_exponential_second_order.py
Uses the existing independent Radau reference and actual FV/FGMRES solver.
Single-run counts assess numerical work; timings are not a speedup claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / (
        "benchmark-results/exponential-second-order/chunk1-neutrons.json"))
    args = parser.parse_args()
    path = ROOT / "examples/exponential_transient_benchmark.py"
    spec = importlib.util.spec_from_file_location("ftm2_benchmark", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    run, reference, output_times, reference_error = benchmark.uniform_case(
        "uniform-ramp", 4, "cpu")
    configurations = [
        {"name": f"{scheme}-h{width:g}", "scheme": scheme, "dt": width}
        for scheme in ("backward-euler", "crank-nicolson", "ftm", "ftm2")
        for width in (0.02, 0.01, 0.005, 0.002, 0.001)]
    configurations.extend(
        {"name": f"adaptive-bdf{order}-rtol{rtol:g}", "scheme": f"bdf{order}",
         "dt": 1e-5, "adaptive": {"rtol": rtol, "automatic_order": order > 1}}
        for order in (1, 5) for rtol in (1e-4, 1e-5))
    rows = []
    for config in configurations:
        result = run(config, None)
        metrics = benchmark.summarize(result, reference, output_times, None)
        row = {"configuration": config,
               "max_relative_power_error": metrics["max_power_error"],
               "max_endpoint_relative_power_error": metrics["max_endpoint_power_error"],
               "accepted_steps": metrics["accepted"], "rejected_steps": metrics["rejected"],
               "neutron_block_solves": sum(result.feedback_iterations)
                                       + sum(result.rejected_feedback_iterations),
               "fgmres_iterations": metrics["outer_iterations"],
               "pcg_iterations": metrics["inner_iterations"],
               "ftm_fallbacks": metrics["ftm_fallbacks"]}
        rows.append(row)
        print(config["name"], f"error={100*row['max_relative_power_error']:.6f}%",
              f"block solves={row['neutron_block_solves']}",
              f"PCG={row['pcg_iterations']}", flush=True)
    sources = ["garnet/_exponential.py", "garnet/timescheme.py", "garnet/transient.py",
               "examples/exponential_transient_benchmark.py",
               "tools/assess_exponential_second_order.py"]
    report = {"case": "uniform-ramp", "scope": "Neutrons and delayed neutrons only",
              "reference": "Independent Radau; common-grid linear output interpolation",
              "reference_refinement_error": reference_error, "power_error_budget": 1e-3,
              "timing_claim": None,
              "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                for name in sources}, "records": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
