"""C5G7 UO2/MOX colorsets with isolated-assembly ADFs; see docs/c5g7_colorset_adf.md.

Usage: python examples/c5g7_colorset_adf.py --cells-per-pin 4 \
           --nodes 1 2 17 --device cpu --output colorset.json

A fine-mesh, pin-cell-resolved diffusion solution of each colorset is the
reference. Assembly cross sections and ADFs come from isolated, reflected
single-assembly solves (the practical lattice-code workflow). Coarse
homogenized solves with and without those ADFs are compared against the
reference. Reference cross sections and reference discontinuity factors on
every coarse node face, generated from the colorset solution itself, check
the whole equivalence pipeline: whenever those factors are admissible
(positive), the coarse solve must reproduce the reference. Everything is
diffusion; nothing here is a transport validation.
"""
import argparse
import json
import time

import numpy as np

from garnet import DiffusionEigenSolver, Grid
from garnet.adf import assembly_face_factors, expand_region_side_factors, region_side_traces
from garnet.backend import asnumpy
from garnet.benchmarks.c5g7 import (ASSEMBLY_PITCH, MOX_LATTICE, N_PIN, UO2_LATTICE,
                                    build_c5g7_2d)
from garnet.sph import flux_weighted_homogenize

# Pin-cell material order of build_c5g7_2d (water reflector follows).
PIN_CHARS = ["U", "m", "o", "x", "G", "F"]
LATTICES = {"U": UO2_LATTICE, "M": MOX_LATTICE}
# Assembly layouts, layout[row][col] with x = col and y = row; every outer
# face reflects. checker3 puts MOX at the centre and corners, so the centre
# MOX sees four UO2 neighbours and each corner MOX only two.
LAYOUTS = {
    "checker2": ["UM", "MU"],
    "checker3": ["MUM", "UMU", "MUM"],
    "island3": ["UUU", "UMU", "UUU"],
}
FINE_TOL = dict(tol_k=1e-10, tol_source=1e-9, inner_rtol_floor=1e-11)
COARSE_TOL = dict(tol_k=1e-10, tol_source=1e-9, inner_rtol_floor=1e-11)


def pin_materials():
    """Pin-cell-homogenized C5G7 materials and their fission cross sections."""
    problem = build_c5g7_2d(cells_per_pin=1)
    n = len(PIN_CHARS)
    return problem.materials[:n], np.asarray(problem.fission_xs)[:n]


def fine_problem(layout, s):
    """Fine material map, assembly region map and grid for a layout."""
    layout = [list(row) for row in layout]
    ny, nx = len(layout), len(layout[0])
    n = N_PIN * s
    idx = {c: i for i, c in enumerate(PIN_CHARS)}
    mmap = np.zeros((nx * n, ny * n, 1), dtype=int)
    regions = np.zeros_like(mmap)
    for row, kinds in enumerate(layout):
        for col, kind in enumerate(kinds):
            pins = np.array([[idx[c] for c in line] for line in LATTICES[kind]])  # [py, px]
            block = np.kron(pins.T, np.ones((s, s), dtype=int))                    # [x, y]
            mmap[col * n:(col + 1) * n, row * n:(row + 1) * n, 0] = block
            regions[col * n:(col + 1) * n, row * n:(row + 1) * n, 0] = row * nx + col
    grid = Grid(mmap.shape, (nx * ASSEMBLY_PITCH, ny * ASSEMBLY_PITCH, 1.0))
    return grid, mmap, regions


def solve_fine(layout, s, device):
    materials, fission = pin_materials()
    grid, mmap, regions = fine_problem(layout, s)
    solver = DiffusionEigenSolver(grid, materials, mmap, bc="reflective", device=device)
    result = solver.solve(**FINE_TOL)
    if not result.converged:
        raise RuntimeError(f"fine solve failed for {layout}: {result}")
    return dict(solver=solver, result=result, materials=materials, fission=fission,
                grid=grid, mmap=mmap, regions=regions)


def node_map(layout, nodes, cells_per_node=1):
    """Coarse-node index of every cell, numbered x-fastest over the whole core."""
    ny, nx = len(layout), len(layout[0])
    ix = np.arange(nx * nodes * cells_per_node) // cells_per_node
    iy = np.arange(ny * nodes * cells_per_node) // cells_per_node
    return (iy[None, :] * nx * nodes + ix[:, None])[..., None]


def homogenize(fine, regions=None):
    """Flux-weighted region cross sections, Sigma_f and average flux, (R, G).

    ``regions`` defaults to the assemblies; any coarse-node map may be given.
    """
    regions = fine["regions"] if regions is None else regions
    flux = fine["result"].flux_numpy
    volume = float(np.prod(fine["grid"].spacing))
    mats, region_flux, region_volume = flux_weighted_homogenize(
        flux, fine["materials"], fine["mmap"], regions, cell_volume=volume)
    rates = np.zeros_like(region_flux)
    sigma_f = fine["fission"][fine["mmap"]]                      # (*shape, G)
    for g in range(flux.shape[0]):
        np.add.at(rates[:, g], regions.ravel(), (sigma_f[..., g] * flux[g]).ravel() * volume)
    return mats, rates / (region_flux * region_volume[:, None]), region_flux


def assembly_library(s, device):
    """Isolated, reflected UO2 and MOX assemblies: XS, Sigma_f and side ADFs.

    With reflective faces the homogeneous solution is flat, so each side's ADF
    is the side-averaged heterogeneous surface flux over the assembly average.
    The discrete surface flux at a zero-current boundary is the boundary
    cell's flux, consistent with the two-point stencil.
    """
    library = {}
    for kind in LATTICES:
        fine = solve_fine([kind], s, device)
        mats, sigma_f, region_flux = homogenize(fine)
        flux = fine["result"].flux_numpy[..., 0]                 # (G, nx, ny)
        sides = [flux[:, 0, :], flux[:, -1, :], flux[:, :, 0], flux[:, :, -1]]
        adf = np.ones((flux.shape[0], 6))
        for side, values in enumerate(sides):
            adf[:, side] = values.mean(axis=1) / region_flux[0]
        library[kind] = dict(material=mats[0], sigma_f=sigma_f[0], adf=adf,
                             k_inf=fine["result"].k_eff,
                             solve_seconds=fine["result"].solve_seconds)
    return library


def reference_factors(fine, regions, mats, region_flux, coarse_spacing):
    """Discrete-consistent reference DFs on every side of every coarse node.

    ``regions`` maps fine cells to coarse nodes. For every node side with an
    interface, the homogeneous two-point trace is phi_bar - J_out h / (2 D),
    from the reference node-average flux and outgoing current density; the
    factor is the heterogeneous reference surface flux divided by that trace.
    Exterior sides are unity. With the node's reference cross sections these
    factors make the reference node fluxes an exact coarse solution, but only
    if every trace is positive: on large nodes the two-point drop J h / (2 D)
    can exceed the average flux, and no admissible factor exists. Returns
    (factors, min_trace_ratio); the ratio is the smallest homogeneous trace
    over its node-average flux, and the factors are usable only if it is > 0.
    """
    trace, current, area = region_side_traces(fine["solver"].ops, fine["result"].flux,
                                              regions)
    G, R = trace.shape[:2]
    D = np.array([m.diffusion for m in mats])                    # (R, G)
    factors = np.ones((G, R, 6))
    min_ratio = np.inf
    for side in range(6):
        has = area[:, side] > 0
        if not has.any():
            continue
        half = 0.5 * coarse_spacing[side // 2]
        j_out = current[:, has, side] / area[has, side]
        homogeneous = region_flux[has].T - j_out * half / D[has].T
        min_ratio = min(min_ratio, float((homogeneous / region_flux[has].T).min()))
        factors[:, has, side] = trace[:, has, side] / homogeneous
    return factors, min_ratio


def solve_coarse(layout, nodes, mats, material_map, regions, side_factors, device):
    """Homogenized solve with ``nodes`` x ``nodes`` cells per assembly.

    ``material_map`` indexes ``mats``; ``regions`` and ``side_factors``
    (G, R, 6) define the ADFs, applied only where ``regions`` changes.
    """
    ny, nx = len(layout), len(layout[0])
    grid = Grid(material_map.shape, (nx * ASSEMBLY_PITCH, ny * ASSEMBLY_PITCH, 1.0))
    adf = None
    if side_factors is not None:
        adf = assembly_face_factors(regions, expand_region_side_factors(regions, side_factors))
    solver = DiffusionEigenSolver(grid, mats, material_map, bc="reflective", device=device,
                                  adf=adf, linear_solver="gmres")
    result = solver.solve(**COARSE_TOL)
    if not result.converged:
        raise RuntimeError(f"coarse solve failed: {result}")
    return solver, result


def observables(solver, flux, regions, sigma_f):
    """Assembly fission rates and interface currents, normalized to unit
    total fission production. ``sigma_f`` is per cell, (G, *shape)."""
    flux = asnumpy(flux)
    volume = float(np.prod(solver.grid.spacing))
    production = sum(float(np.sum(asnumpy(solver.fields.nu_sigma_f[g]) * flux[g]))
                     for g in range(flux.shape[0])) * volume
    flux = flux / production
    rate = np.zeros(int(regions.max()) + 1)
    np.add.at(rate, regions.ravel(), (sigma_f * flux).sum(axis=0).ravel() * volume)
    xp = solver.xp
    _, current, _ = region_side_traces(solver.ops, xp.asarray(flux), regions)
    return rate, current.sum(axis=0)


def compare(row, k_ref, rate_ref, current_ref, rate, current):
    power, power_ref = rate / rate.mean(), rate_ref / rate_ref.mean()
    err = 100 * (power - power_ref) / power_ref
    scale = np.abs(current_ref).max()
    row.update(dpcm=1e5 * (row["k_eff"] - k_ref),
               power_max_pct=float(np.abs(err).max()), power_rms_pct=float(np.sqrt(np.mean(err**2))),
               current_max_rel=float(np.abs(current - current_ref).max() / scale) if scale > 0 else 0.0,
               assembly_power=power.tolist())
    return row


def study(layout_name, s, nodes, device, library=None):
    layout = LAYOUTS[layout_name]
    library = library or assembly_library(s, device)
    t0 = time.perf_counter()
    fine = solve_fine(layout, s, device)
    fine_seconds = time.perf_counter() - t0
    k_ref = fine["result"].k_eff
    sigma_f_fine = np.moveaxis(fine["fission"][fine["mmap"]], -1, 0)
    rate_ref, current_ref = observables(fine["solver"], fine["result"].flux, fine["regions"],
                                        sigma_f_fine)
    kinds = [kind for row in layout for kind in row]
    lib_mats = [library[k]["material"] for k in kinds]
    lib_sigma_f = np.array([library[k]["sigma_f"] for k in kinds])
    lib_adf = np.stack([library[k]["adf"] for k in kinds], axis=1)      # (G, R, 6)
    fine_per_assembly = N_PIN * s

    rows = []
    for n in nodes:
        assemblies = node_map(layout, 1, n)                     # coarse cell -> assembly
        cases = [("library XS, no ADF", lib_mats, assemblies, lib_sigma_f, None, None),
                 ("library XS + ADF", lib_mats, assemblies, lib_sigma_f, lib_adf, None)]
        if fine_per_assembly % n == 0:
            # Node-level equivalence: reference XS on every coarse node, with
            # and without reference DFs on every coarse face.
            nodes_fine = node_map(layout, n, fine_per_assembly // n)
            nodes_coarse = node_map(layout, n)
            ref_mats, ref_sigma_f, ref_flux = homogenize(fine, nodes_fine)
            spacing = (ASSEMBLY_PITCH / n, ASSEMBLY_PITCH / n, 1.0)
            factors, min_ratio = reference_factors(fine, nodes_fine, ref_mats, ref_flux, spacing)
            cases.append(("reference XS, no DF", ref_mats, nodes_coarse, ref_sigma_f, None, None))
            cases.append(("reference XS + DF", ref_mats, nodes_coarse, ref_sigma_f,
                          factors if min_ratio > 0 else None, min_ratio))
        for label, mats, material_map, sigma_f, factors, min_ratio in cases:
            row = dict(case=label, nodes=n)
            if min_ratio is not None:
                row["min_trace_ratio"] = min_ratio
                if factors is None:
                    rows.append(dict(row, admissible=False))
                    continue
            solver, result = solve_coarse(layout, n, mats, material_map, material_map,
                                          factors, device)
            rate, current = observables(solver, result.flux, assemblies,
                                        np.moveaxis(sigma_f[material_map], -1, 0))
            row.update(k_eff=result.k_eff, outer_iterations=result.outer_iterations,
                       inner_iterations=result.inner_iterations,
                       solve_seconds=result.solve_seconds)
            rows.append(compare(row, k_ref, rate_ref, current_ref, rate, current))
    return dict(layout=layout_name, cells_per_pin=s, device=device,
                fine_cells=int(np.prod(fine["grid"].shape)), k_ref=k_ref,
                fine_outer_iterations=fine["result"].outer_iterations,
                fine_inner_iterations=fine["result"].inner_iterations,
                fine_seconds=fine_seconds, reference_power=(rate_ref / rate_ref.mean()).tolist(),
                rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells-per-pin", nargs="+", type=int, default=[4])
    p.add_argument("--nodes", nargs="+", type=int, default=[1, 2, 17])
    p.add_argument("--layouts", nargs="+", choices=list(LAYOUTS), default=list(LAYOUTS))
    p.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    p.add_argument("--output", help="write JSON results to this file")
    args = p.parse_args()
    out = dict(notes="diffusion-only equivalence study; pin-cell homogenized C5G7 fine mesh; "
                     "pcm = 1e5 (k - k_ref); power is Sigma_f rate per assembly, mean 1",
               device=args.device, libraries=[], studies=[])
    for s in args.cells_per_pin:
        library = assembly_library(s, args.device)
        out["libraries"].append({"cells_per_pin": s, **{
            k: dict(k_inf=v["k_inf"], adf=v["adf"].tolist(), solve_seconds=v["solve_seconds"])
            for k, v in library.items()}})
        for name in args.layouts:
            res = study(name, s, args.nodes, args.device, library)
            out["studies"].append(res)
            print(f"\n{name}  s={s}  fine cells={res['fine_cells']}  k_ref={res['k_ref']:.7f}  "
                  f"fine t={res['fine_seconds']:.2f}s")
            for r in res["rows"]:
                if r.get("admissible") is False:
                    print(f"  {r['case']:22s} N={r['nodes']:2d}  inadmissible: min homogeneous "
                          f"trace / average = {r['min_trace_ratio']:.3f}")
                    continue
                print(f"  {r['case']:22s} N={r['nodes']:2d}  k={r['k_eff']:.7f}  "
                      f"dk={r['dpcm']:+8.1f} pcm  Pmax={r['power_max_pct']:6.2f}%  "
                      f"Prms={r['power_rms_pct']:6.2f}%  Jmax={r['current_max_rel']:.3f}  "
                      f"outers={r['outer_iterations']:3d}  inners={r['inner_iterations']:6d}  "
                      f"t={r['solve_seconds']:.2f}s", flush=True)
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(out, fh, indent=2)
            fh.write("\n")


if __name__ == "__main__":
    main()
