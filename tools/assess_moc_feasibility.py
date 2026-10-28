"""Inventory existing S_N sweep payloads for the separate 2D/1D MOC study.

Run from the repository root::

    OPENBLAS_NUM_THREADS=1 python3 -m tools.assess_moc_feasibility

This runs the existing CPU levels engine. It neither implements MOC nor
predicts GPU timings. Byte counts describe selected logical array payloads,
not process RSS, CUDA allocator use, or total solver memory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import scipy

from garnet.benchmarks import build_hpmr2d, build_hpmr3d
from garnet.tri_sn import TriSNTransportSolver


ROOT = Path(__file__).resolve().parents[1]


def payload_bytes(value):
    """Sum logical array payloads; count each listed buffer independently."""
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(payload_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(payload_bytes(v) for v in value)
    return 0


def inventory(nz, scheme):
    builder = build_hpmr3d if nz else build_hpmr2d
    kwargs = {"nz": nz} if nz else {}
    p = builder(refine=2, absorber="polar", **kwargs)
    solver = TriSNTransportSolver(
        p.grid, p.materials, p.material_map, active=p.active,
        mix_material=p.mix_material, mix_weight=p.mix_weight,
        engine="levels", device="cpu", n_polar=2, n_azi=8, scheme=scheme,
    )
    group_table = solver._lv_tables(0)
    widths = np.asarray([len(level[0]) for level in solver._levels])
    buffers = solver._bufs
    # NumPy uses broadcast/reduce scratch; the fused CUDA implementation
    # replaces these arrays with empty placeholders. Subtract only those
    # known allocations, without claiming this is a CUDA measurement.
    scratch = payload_bytes(buffers.get("psiw")) + payload_bytes(buffers.get("mk_w"))
    if scheme == "scb":
        scratch += sum(payload_bytes(work[4]) for work in buffers["work"])
    shared = (payload_bytes(solver._levels) + payload_bytes(buffers)
              + payload_bytes(solver._w_xp) + payload_bytes(solver._act_flat_dev)
              + payload_bytes(getattr(solver, "_ac_dev", None)))
    fused_shared = shared - scratch
    one_group = payload_bytes(group_table)
    source = solver._act_flat.astype(float)
    phi, psi = solver._sweep_dev(0, source)
    active = solver._act_flat
    assert int(widths.sum()) == solver.M * int(active.sum())
    assert np.all(np.isfinite(phi)) and np.all(phi[active] > 0)
    assert np.all(np.isfinite(psi))
    return {
        "refine": 2, "nz": nz or 1, "scheme": scheme,
        "shape": list(p.grid.shape), "allocated_cells": solver.N,
        "active_cells": int(active.sum()), "groups": solver.G,
        "ordinates": solver.M, "sequential_levels": len(solver._levels),
        "level_width_min": int(widths.min()),
        "level_width_median": float(np.median(widths)),
        "level_width_max": int(widths.max()),
        "angular_flux_bytes_one_group": payload_bytes(buffers["psi"]),
        "cpu_unfused_shared_payload_bytes": shared,
        "fused_cuda_shared_payload_estimate_bytes": fused_shared,
        "group_table_bytes_per_group": one_group,
        "fused_cuda_sweep_payload_estimate_bytes_two_groups": fused_shared + 2 * one_group,
        "fused_cuda_sweep_payload_estimate_bytes_eleven_groups": fused_shared + 11 * one_group,
        "cpu_fixed_source_sweep_finite_and_positive": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/qa/moc-feasibility-2026-10-26.json")
    args = parser.parse_args()
    files = ["garnet/sn.py", "garnet/tri_sn.py", "garnet/kernels.py",
             "garnet/materials.py", "garnet/benchmarks/hpmr.py",
             "tools/assess_moc_feasibility.py"]
    record = {
        "study": "2D/1D MOC feasibility: existing S_N allocation inventory",
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__,
            "cupy_importable": importlib.util.find_spec("cupy") is not None,
        },
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                          for f in files},
        "limitations": [
            "No MOC implementation or GPU timing is measured.",
            "S_N level counts use the current 2D/3D quadrature APIs; equal n_polar "
            "values do not imply matching polar quadrature nodes.",
            "Payload counts cover level tables, sweep buffers, masks, active indices, "
            "weights and cached per-group sweep tables only.",
            "Excluded: cross sections, sources, DSA/CMFD matrices and workspaces, "
            "host setup arrays, CUDA graphs and allocator overhead.",
            "Each listed logical array counts independently; tiny NumPy weight "
            "aliases can differ from CUDA allocation ownership.",
            "Eleven-group numbers extrapolate table layout from two-group "
            "placeholder data; they are not an eleven-group physics solve.",
        ],
        "cases": [inventory(nz, scheme) for nz in (0, 10, 20)
                  for scheme in ("step", "scb")],
        "amdahl_scenarios": [
            {"baseline_sweep_fraction": f, "sweep_speedup": r,
             "extra_axial_and_coupling_fraction": c,
             "new_to_old_outer_iteration_ratio": k,
             "end_to_end_speedup": 1 / (k * (1 - f + f / r + c))}
            for f, r, c, k in [(0.8, 3, 0.05, 1), (0.9, 5, 0.05, 1),
                                (0.9, 5, 0.05, 2), (0.5, 5, 0.05, 1),
                                (0.9, 10, 0.05, 1)]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(args.output)
    for row in record["cases"]:
        size = row["fused_cuda_sweep_payload_estimate_bytes_eleven_groups"] / 2**20
        print(f"{row['scheme']:4s} nz={row['nz']:2d}: "
              f"{row['active_cells']:6d} active cells, "
              f"{row['sequential_levels']:3d} levels, "
              f"{size:.2f} MiB selected payload at 11 groups")


if __name__ == "__main__":
    main()
