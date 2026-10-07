#!/usr/bin/env python3
"""Validate Serpent multigroup constants in a reflective one-cell reactor.

Run from the repository with ``python examples/serpent-xs/point-reactor/run.py``.
GPU validation is explicitly requested with ``--device gpu``; add
``--require-rocm`` to require a HIP build of CuPy. CPU validation always runs
first. Reports contain source basenames and hashes, never absolute input paths.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import scipy

# Support direct execution from an uninstalled source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ndgpu import DiffusionEigenSolver, Grid
from ndgpu import kernels
from ndgpu.backend import device_name, get_backend, is_rocm
from ndgpu.serpent_xs import read_serpent_xs


def normalized(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("flux must be finite, nonnegative, and have positive sum")
    return values / values.sum()


def dense_reference(record):
    """Build the independent balance directly from Serpent estimates."""
    absorption = record.estimate("INF_RABSXS").mean.reshape(record.n_groups)
    scatter = record.estimate("INF_SP0").mean.reshape(record.n_groups, record.n_groups)
    production = record.estimate("INF_NSF").mean.reshape(record.n_groups)
    chi = normalized(record.estimate("INF_CHIT").mean)
    matrix = np.diag(absorption + scatter.sum(axis=1)) - scatter.T
    shape = np.linalg.solve(matrix, chi)
    return float(production @ shape), normalized(shape)


def balance_residual(record, k, flux):
    absorption = record.estimate("INF_RABSXS").mean.reshape(record.n_groups)
    scatter = record.estimate("INF_SP0").mean.reshape(record.n_groups, record.n_groups)
    production = record.estimate("INF_NSF").mean.reshape(record.n_groups)
    chi = normalized(record.estimate("INF_CHIT").mean)
    loss = (np.diag(absorption + scatter.sum(axis=1)) - scatter.T) @ flux
    return float(np.linalg.norm(loss - chi * (production @ flux) / k)
                 / np.linalg.norm(loss))


def estimate_pair(record, name):
    estimate = record.estimate(name)
    # ANA_KEFF has total, prompt, and delayed pairs; only total is reported.
    mean = float(estimate.mean.reshape(-1)[0])
    relative_error = float(estimate.relative_error.reshape(-1)[0])
    return {"mean": mean, "relative_error": relative_error,
            "absolute_error": abs(mean) * relative_error}


def comparison(k, flux, dense_k, dense_flux, reference_k, reference_flux):
    dense_k_error = abs(k - dense_k)
    dense_flux_error = float(np.max(np.abs(flux - dense_flux)))
    flux_delta = flux - reference_flux
    flux_max = float(np.max(np.abs(flux_delta)))
    flux_l2 = float(np.linalg.norm(flux_delta) / np.linalg.norm(reference_flux))
    k_error = abs(k - reference_k["mean"])
    k_limit = max(1e-5, 3 * reference_k["absolute_error"])
    sigma = reference_k["absolute_error"]
    checks = {
        "dense_k": {"absolute_error": dense_k_error, "limit": 1e-8,
                    "passed": dense_k_error <= 1e-8},
        "dense_flux": {"max_absolute_error": dense_flux_error, "limit": 1e-8,
                       "passed": dense_flux_error <= 1e-8},
        "serpent_k": {"absolute_error": k_error, "limit": k_limit,
                      "reference_only_sigma_units": (k - reference_k["mean"]) / sigma
                      if sigma > 0 else None, "passed": k_error <= k_limit},
        "serpent_flux": {"max_absolute_error": flux_max,
                         "max_absolute_limit": 1e-4, "relative_l2_error": flux_l2,
                         "relative_l2_limit": 5e-4,
                         "passed": flux_max <= 1e-4 and flux_l2 <= 5e-4},
    }
    return checks


def solve_case(record, device, fused, report, spectra):
    label = "cpu" if device == "cpu" else f"gpu_fusion_{'on' if fused else 'off'}"
    previous = kernels.set_fused(fused)
    try:
        solver = DiffusionEigenSolver(Grid((1, 1, 1), (1., 1., 1.)),
                                     record.to_diffusion_material(diffusion="cmm"),
                                     bc="reflective", device=device, dtype=np.float64)
        result = solver.solve(tol_k=1e-11, tol_source=1e-10, max_outer=2000)
        if device == "gpu":
            import cupy
            if solver.xp is not cupy or not isinstance(result.flux, cupy.ndarray):
                raise RuntimeError("GPU validation requires CuPy solver and flux arrays")
        flux = normalized(result.flux_numpy)
        spectra[label] = flux
        checks = comparison(result.k_eff, flux, report["dense"]["k_eff"],
                            spectra["dense"], report["references"]["IMP_KEFF"],
                            spectra["serpent"])
        residual = balance_residual(record, result.k_eff, flux)
        checks["balance"] = {"relative_residual": residual, "limit": 1e-8,
                             "passed": bool(np.isfinite(residual) and residual <= 1e-8)}
        informational = {}
        for name in ("ANA_KEFF", "INF_KINF"):
            reference = report["references"][name]
            delta = result.k_eff - reference["mean"]
            informational[name] = {"signed_delta": delta,
                                   "reference_only_sigma_units": delta / reference["absolute_error"]
                                   if reference["absolute_error"] > 0 else None}
        report["solves"][label] = {
            "k_eff": result.k_eff, "converged": result.converged,
            "outer_iterations": result.outer_iterations,
            "inner_iterations": result.inner_iterations,
            "solve_seconds": result.solve_seconds, "device": result.device,
            "fusion_enabled": fused, "checks": checks, "informational": informational,
            "passed": bool(result.converged and all(c["passed"] for c in checks.values())),
        }
    finally:
        kernels.set_fused(previous)


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def sanitize_nonfinite(value):
    """Keep failed diagnostics writable even when a solver produces NaNs."""
    found = False
    if isinstance(value, (np.ndarray, np.generic)):
        value = json_value(value)
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            cleaned[key], invalid = sanitize_nonfinite(item)
            found |= invalid
        return cleaned, found
    if isinstance(value, (list, tuple)):
        cleaned = []
        for item in value:
            result, invalid = sanitize_nonfinite(item)
            cleaned.append(result)
            found |= invalid
        return cleaned, found
    if isinstance(value, float) and not np.isfinite(value):
        return None, True
    return value, False


def write_case(output, record, report, spectra):
    stem = Path(report["source_file"]).stem
    (output / f"{stem}.json").write_text(json.dumps(report, indent=2, default=json_value,
                                                  allow_nan=False) + "\n")
    bounds = record.energy_bounds_mev
    columns = ["group", "energy_high_mev", "energy_low_mev", *spectra]
    with (output / f"{stem}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        for g in range(record.n_groups):
            writer.writerow([g + 1, bounds[g], bounds[g + 1],
                             *(values[g] for values in spectra.values())])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("results/serpent-point-reactor"))
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--require-rocm", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"requested_device": args.device, "require_rocm": args.require_rocm,
               "software": {"python": sys.version.split()[0], "numpy": np.__version__,
                            "scipy": scipy.__version__},
               "solver": {"class": "DiffusionEigenSolver", "dtype": "float64",
                          "grid_shape": [1, 1, 1], "grid_size_cm": [1., 1., 1.],
                          "boundary": "reflective", "diffusion": "cmm",
                          "tol_k": 1e-11, "tol_source": 1e-10,
                          "max_outer": 2000, "inner_rtol_floor": 1e-10},
               "validation_limits": {"dense_k_absolute": 1e-8,
                                     "dense_flux_max_absolute": 1e-8,
                                     "balance_relative_residual": 1e-8,
                                     "serpent_k": "max(1e-5, 3*reference_absolute_error)",
                                     "serpent_flux_max_absolute": 1e-4,
                                     "serpent_flux_relative_l2": 5e-4},
               "normalization": "group flux sums to one",
               "uncertainty_note": "Sigma units use reference-only uncertainty; no combined covariance is inferred.",
               "cases": [], "errors": []}

    def error(exc):
        message = str(exc)
        for path in (args.input_dir, args.output_dir, Path(__file__).resolve().parents[3]):
            message = message.replace(str(path.resolve()), "<path>")
        return {"type": type(exc).__name__, "message": message}

    loaded = []
    paths = sorted(args.input_dir.glob("*_res.m"))
    if not paths:
        summary["errors"].append({"type": "InputError", "message": "No *_res.m inputs found"})
    for path in paths:
        try:
            loaded.append((read_serpent_xs(path), path.name))
        except Exception as exc:
            report = {"source_file": path.name, "passed": False, "error": error(exc)}
            summary["cases"].append(report)
            (args.output_dir / f"{path.stem}.json").write_text(json.dumps(report, indent=2) + "\n")

    cases = []
    for record, filename in sorted(loaded, key=lambda item: item[0].n_groups):
        report = {"source_file": filename, "source_sha256": record.source_sha256,
                  "serpent_version": record.version, "universe": record.universe,
                  "record_index": record.record_index, "n_groups": record.n_groups,
                  "solves": {}, "errors": []}
        spectra = {}
        try:
            report["TOT_LEAKRATE"] = estimate_pair(record, "TOT_LEAKRATE")
            if "B1_CALCULATION" in record.fields:
                report["B1_CALCULATION"] = record.fields["B1_CALCULATION"]
            if report["TOT_LEAKRATE"]["mean"] != 0:
                raise ValueError("Reflective point-reactor comparison requires zero TOT_LEAKRATE; finite leakage is unsupported")
            dense_k, dense_flux = dense_reference(record)
            spectra.update(serpent=normalized(record.estimate("INF_FLX").mean), dense=dense_flux)
            residual = balance_residual(record, dense_k, dense_flux)
            report["dense"] = {"k_eff": dense_k, "relative_balance_residual": residual,
                               "balance_limit": 1e-8,
                               "passed": bool(np.isfinite(residual) and residual <= 1e-8)}
            reference_flux = record.estimate("INF_FLX")
            report["INF_FLX"] = {"mean": reference_flux.mean,
                                 "relative_error": reference_flux.relative_error,
                                 "absolute_error": reference_flux.absolute_error}
            report["references"] = {name: estimate_pair(record, name)
                                    for name in ("IMP_KEFF", "ANA_KEFF", "INF_KINF")}
            report["balance_diagnostics"] = record.balance_diagnostics()
            solve_case(record, "cpu", False, report, spectra)
        except Exception as exc:
            report["errors"].append(error(exc))
        cases.append((record, report, spectra))

    # Complete every CPU reference before initializing the requested GPU.
    gpu_ready = False
    if args.require_rocm and args.device != "gpu":
        summary["errors"].append({"type": "ArgumentError", "message": "--require-rocm requires --device gpu"})
    elif args.device == "gpu":
        try:
            xp = get_backend("gpu")
            if args.require_rocm and not is_rocm(xp):
                raise RuntimeError("--require-rocm requires a HIP build of CuPy")
            probe = xp.ones(1, dtype=xp.float64)
            if not kernels.is_cupy(xp) or not isinstance(probe, xp.ndarray):
                raise RuntimeError("GPU preflight requires actual CuPy device arrays")
            xp.cuda.get_current_stream().synchronize()
            summary["gpu_device"] = device_name(xp)
            summary["gpu_is_hip"] = is_rocm(xp)
            summary["software"]["cupy"] = xp.__version__
            gpu_ready = True
        except Exception as exc:
            summary["errors"].append(error(exc))
    for record, report, spectra in cases:
        if gpu_ready and "cpu" in report["solves"]:
            for fused in (False, True):
                try:
                    solve_case(record, "gpu", fused, report, spectra)
                except Exception as exc:
                    report["errors"].append(error(exc))
        expected = 3 if args.device == "gpu" else 1
        report["passed"] = bool(not report["errors"] and len(report["solves"]) == expected
                                and report.get("dense", {}).get("passed", False)
                                and all(s["passed"] for s in report["solves"].values()))
        report, invalid = sanitize_nonfinite(report)
        if invalid:
            report["passed"] = False
            report["errors"].append({"type": "NonfiniteDiagnostic",
                                     "message": "Nonfinite diagnostics were replaced with null"})
        write_case(args.output_dir, record, report, spectra)
        summary["cases"].append(report)
    summary["cases"].sort(key=lambda case: case.get("n_groups", 0))
    summary["passed"] = bool(summary["cases"] and not summary["errors"]
                             and all(case["passed"] for case in summary["cases"]))
    serialized = json.dumps(summary, indent=2, default=json_value, allow_nan=False)
    (args.output_dir / "summary.json").write_text(serialized + "\n")
    print(serialized)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
