"""Pose the hex-homogenized 3-D HP-MR to FEMFFUSION, for the GARNET vs FEMFFUSION report.

    PYTHONPATH=. python tools/export_femffusion.py OUT_DIR [--dollars 0.5]

Writes, into OUT_DIR:

- ``hpmr.msh``: the radial core as a Gmsh 2.2 mesh with three quadrilaterals
  per hexagon, the layout ``Geometry_Type = Hexagonal`` expects. A quad's
  physical tag is its hexagon's index 1..55, counted row by row from the
  bottom (constant y), left to right -- the order of ``Geometry_Points``.
  Centres are GARNET's own site coordinates (``hex_site_xy``), pointy-top
  hexagons of flat-to-flat ``PITCH``.
- ``hpmr.xml``: one ``<mix>`` per hexagon and axial plane (no
  ``<Composition>``; FEMFFUSION's identity map then gives mix id
  ``plane * 55 + hex - 1``, the user index of an extruded cell). Fuel and the
  central cell become axial reflector in the 20 cm end planes, as in
  :func:`garnet.benchmarks.hpmr.build_hpmr3d`. The drums are withdrawn: every
  drum hexagon is the library's drum beryllium, no B4C. (Smeared over the
  hexagon by area, :func:`garnet.benchmarks.hpmr.smeared_drum_material`, the
  arc makes the hexagon black to thermal neutrons -- Sigma_a ~ 10 /cm, a
  diffusion length of ~0.5 mm -- which neither code resolves on any practical
  mesh, so the inserted drums are no benchmark for discretization cost.)
  The mixes carry the kinetics (``nprecursors``, ``Beta``, ``Lambda``,
  ``ChiD`` = chi, ``Velocities``): the library's 11 speeds and six families,
  with the delayed spectrum equal to chi; see
  :func:`garnet.femffusion.write_material_xml` for why they sit in the XML.
- ``hpmr.prm``: the steady input (diffusion, vacuum on every face).
- ``hpmr_final.xml`` and ``hpmr_transient.prm``: an external reactivity
  insertion. Every fissile material's nu*Sigma_f is scaled by (1 + eps), with
  eps = rho / (1 - rho) and rho = ``--dollars`` * beta, so the final state is
  exactly rho above critical (scaling all production scales k). Sigma_f, the
  power weight, is unchanged. FEMFFUSION ramps every cross section linearly
  from hpmr.xml to hpmr_final.xml over [0, Time_End]
  (``Type_Perturbation = Read_XML_File``), so here only nu*Sigma_f moves.
- ``layout.json``: the hexagon order, sites, kinds and the insertion, so
  FEMFFUSION's per-hexagon power maps can be mapped back onto GARNET's sites.

FE degree, refinements, tolerances, solver and time step are left to the run
(command-line overrides of tools/run_femffusion.py).
"""

from __future__ import annotations

import argparse
import json
import math
import os

from garnet.benchmarks.hpmr import (AXIAL_REFLECTOR, AXIAL_REFLECTOR_HEIGHT, BE_REFLECTOR,
                                   CENTRAL, DRUM_ARC_AREA_FRACTION, DRUM_BE, FUEL,
                                   HPMR_BE_SITES, HPMR_DRUM_SITES, HPMR_FUEL_SITES, PITCH,
                                   TOTAL_HEIGHT, smeared_drum_material)
from garnet.benchmarks.hpmr_thermal import hpmr_endfb8_builtin, hpmr_kinetics_11g
from garnet.femffusion import write_material_xml
from garnet.hexraster import hex_site_xy
from garnet.materials import Kinetics

KIND_MATERIAL = {"fuel": FUEL, "central": CENTRAL, "be": BE_REFLECTOR, "drum": DRUM_BE}


def hpmr_rows():
    """Hexagon sites in FEMFFUSION order: rows of constant R (y), bottom first,
    each sorted by C (x). Returns a list of rows of (R, C, kind)."""
    kind = {(0, 0): "central"}
    kind.update({s: "fuel" for s in HPMR_FUEL_SITES})
    kind.update({s: "be" for s in HPMR_BE_SITES})
    kind.update({s: "drum" for s in HPMR_DRUM_SITES})
    rows = []
    for R in sorted({r for r, _ in kind}):
        cs = sorted(c for r, c in kind if r == R)
        if cs != list(range(cs[0], cs[-1] + 1)):
            raise ValueError(f"row R={R} has a gap; Geometry_Points needs contiguous rows")
        rows.append([(R, c, kind[(R, c)]) for c in cs])
    return rows


def write_hex_msh(rows, pitch, path):
    """Gmsh 2.2 ASCII: three counter-clockwise quads per pointy-top hexagon."""
    ri, rc = pitch / 2.0, pitch / math.sqrt(3.0)
    offsets = [(0.0, 0.0), (0.0, -rc), (ri, -rc / 2), (ri, rc / 2),
               (0.0, rc), (-ri, rc / 2), (-ri, -rc / 2)]
    nodes, index, quads = [], {}, []

    def node(x, y):
        key = (round(x, 9), round(y, 9))
        if key not in index:
            index[key] = len(nodes) + 1
            nodes.append(key)
        return index[key]

    tag = 0
    for row in rows:
        for R, C, _ in row:
            tag += 1
            cx, cy = hex_site_xy(R, C, pitch)
            p = [node(cx + dx, cy + dy) for dx, dy in offsets]
            quads += [(tag, p[0], p[1], p[2], p[3]), (tag, p[0], p[3], p[4], p[5]),
                      (tag, p[0], p[5], p[6], p[1])]
    with open(path, "w") as fh:
        fh.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n")
        fh.write(f"{len(nodes)}\n")
        fh.writelines(f"{i} {x:.9f} {y:.9f} 0\n" for i, (x, y) in enumerate(nodes, 1))
        fh.write(f"$EndNodes\n$Elements\n{len(quads)}\n")
        fh.writelines(f"{i} 3 2 {t} {i} {a} {b} {c} {d}\n"
                      for i, (t, a, b, c, d) in enumerate(quads, 1))
        fh.write("$EndElements\n")
    return len(nodes), len(quads)


def plane_materials(rows, mats, n_planes):
    """The material of every (plane, hexagon), plane-major from the bottom."""
    dz = TOTAL_HEIGHT / n_planes
    out = []
    for k in range(n_planes):
        zc = (k + 0.5) * dz
        axial = zc < AXIAL_REFLECTOR_HEIGHT or zc > TOTAL_HEIGHT - AXIAL_REFLECTOR_HEIGHT
        for row in rows:
            for _, _, kind in row:
                m = KIND_MATERIAL[kind]
                if axial and m in (FUEL, CENTRAL):
                    m = AXIAL_REFLECTOR
                out.append(mats[m])
    return out


#: B4C area fraction smeared into the drum hexagons in the benchmark: drums out.
BENCHMARK_DRUM_FRACTION = 0.0


def hpmr_materials(drum_fraction=BENCHMARK_DRUM_FRACTION):
    """11-group 3-D material list (MATERIAL_NAMES_3D order) with the drum body
    smeared at ``drum_fraction``; 0 keeps the library's drum beryllium as is."""
    mats = hpmr_endfb8_builtin(three_d=True)
    if drum_fraction:
        mats[DRUM_BE] = smeared_drum_material(mats, drum_fraction)
    return mats


def insertion_epsilon(dollars, kinetics):
    """Production scale eps of an insertion of ``dollars``: rho = eps / (1 + eps)."""
    rho = dollars * float(kinetics.beta.sum())
    return rho / (1.0 - rho)


def scale_production(mats, factor):
    """Copies of ``mats`` with nu*Sigma_f of the fissile ones times ``factor``."""
    return [m.replace(nu_sigma_f=m.nu_sigma_f * factor) if m.is_fissile else m
            for m in mats]


def hpmr_benchmark_kinetics():
    """The library kinetics with the delayed spectrum equal to chi (both codes)."""
    k = hpmr_kinetics_11g(strict=True)
    return Kinetics(velocities=k.velocities, beta=k.beta, decay=k.decay)


def _prm(entries):
    return "".join(f"set {key:<24s}= {value}\n" for key, value in entries)


def export_hpmr(out, n_planes=10, dollars=0.5, ramp=2.0):
    os.makedirs(out, exist_ok=True)
    out = os.path.abspath(out)
    rows = hpmr_rows()
    n_hex = sum(len(r) for r in rows)
    n_nodes, n_quads = write_hex_msh(rows, PITCH, os.path.join(out, "hpmr.msh"))
    mats = hpmr_materials()
    kinetics = hpmr_benchmark_kinetics()
    per_cell = plane_materials(rows, mats, n_planes)
    write_material_xml(per_cell, os.path.join(out, "hpmr.xml"), kinetics=kinetics)
    longest = max(len(r) for r in rows)
    steady = [
        ("Dimension", 3), ("N_Eigenvalues", 1), ("Energy_Groups", mats[FUEL].n_groups),
        ("Transport_Appr", "Diffusion"), ("FE_Degree", 2), ("N_Refinements", 0),
        ("Geometry_Type", "Hexagonal"),
        ("Mesh_Size", f"{longest} {len(rows)} {n_planes}"),
        ("Geometry_Points", " ".join(f"1 {len(r)}" for r in rows)),
        ("Cell_Pitch_x", f"{longest}* {PITCH}"), ("Cell_Pitch_y", f"{len(rows)}* {PITCH}"),
        ("Cell_Pitch_z", f"{n_planes}* {TOTAL_HEIGHT / n_planes}"),
        # Hexagonal 3-D meshes give every boundary face id 1; code 2 = vacuum
        # (Marshak, D dphi/dn = -phi/2), GARNET's vacuum.
        ("Boundary_Conditions", "2 2 2 2 2 2"),
        ("Mesh_Filename", os.path.join(out, "hpmr.msh")),
        ("XSEC_Type", "XML"), ("XSECS_Filename", os.path.join(out, "hpmr.xml")),
        ("Output_Filename", os.path.join(out, "hpmr.out")),
        ("Output_Flag", "false"), ("Print_Grid_Flag", "false"),
        ("Solver_Type", "bifpam"), ("EPS_Tolerance", "1e-8"),
    ]
    with open(os.path.join(out, "hpmr.prm"), "w") as fh:
        fh.write("# Hex-homogenized 3-D HP-MR, written by tools/export_femffusion.py\n")
        fh.write(_prm(steady))
    layout = dict(pitch=PITCH, n_planes=n_planes, height=TOTAL_HEIGHT, n_hex=n_hex,
                  n_nodes=n_nodes, n_quads=n_quads,
                  rows=[[[R, C, kind] for R, C, kind in row] for row in rows],
                  drum_fraction=BENCHMARK_DRUM_FRACTION,
                  drum_arc_area_fraction=DRUM_ARC_AREA_FRACTION, dollars=dollars, ramp=ramp,
                  beta=float(kinetics.beta.sum()),
                  epsilon=insertion_epsilon(dollars, kinetics))
    if dollars:
        final = plane_materials(rows, scale_production(mats, 1.0 + layout["epsilon"]),
                                n_planes)
        write_material_xml(final, os.path.join(out, "hpmr_final.xml"), kinetics=kinetics)
        transient = [(k, v) for k, v in steady if k != "Output_Filename"] + [
            ("Output_Filename", os.path.join(out, "hpmr_transient.out")),
            ("Transient", "true"), ("Type_Perturbation", "Read_XML_File"),
            ("Read_XML_Filename", os.path.join(out, "hpmr_final.xml")),
            # FEMFFUSION's default Implicit-Exponential scheme does not hold an
            # unperturbed transient on this input (P drops ~4% in a 0.1 s step,
            # in proportion to the step); both semi-implicit schemes hold it.
            ("Distributed_Time_Scheme", "Semi-Implicit-Euler"),
            ("Time_Delta", "0.01"), ("Time_End", f"{ramp}")]
        with open(os.path.join(out, "hpmr_transient.prm"), "w") as fh:
            fh.write(f"# HP-MR: +{dollars} $ external reactivity ramp over [0, Time_End], "
                     "written by tools/export_femffusion.py\n")
            fh.write(_prm(transient))
    with open(os.path.join(out, "layout.json"), "w") as fh:
        json.dump(layout, fh, indent=1)
    return layout


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", help="output directory")
    parser.add_argument("--planes", type=int, default=10,
                        help="axial planes over 200 cm (multiple of 10)")
    parser.add_argument("--dollars", type=float, default=0.5,
                        help="external reactivity at the end of the ramp, $ (0: no transient)")
    parser.add_argument("--ramp", type=float, default=2.0, help="ramp duration, s")
    args = parser.parse_args()
    if args.planes % 10:
        parser.error("--planes must be a multiple of 10 (20 cm axial reflectors)")
    layout = export_hpmr(args.out, args.planes, args.dollars, args.ramp)
    print(json.dumps({k: v for k, v in layout.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
