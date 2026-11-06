"""FEMFFUSION side of the GARNET vs FEMFFUSION performance report.

    PYTHONPATH=. python tools/run_femffusion.py iaea3d \
        --prm ~/FEMFFUSION/examples/3D_IAEA/IAEA_FE3.prm --fe-degree 3 --refinements 1 \
        --set Solver_Type=bifpam --set EPS_Tolerance=1e-9 --workdir runs/iaea-fe3-r1 \
        --repeats 3 --output iaea-fe3-r1.json
    PYTHONPATH=. python tools/run_femffusion.py hpmr3d-transient \
        --prm OUT/hpmr_transient.prm --fe-degree 2 --set Time_Delta=0.02 ...

Runs ``femffusion.exe`` (single process, one thread: FEMFFUSION's MPI path is
not usable yet, see the report) on a copy of a base ``.prm`` with overrides:
``FE_Degree``, ``N_Refinements``, ``N_Eigenvalues = 1``, ``Output_Flag =
false``, the output file in ``--workdir``, input paths made absolute (the
examples give them relative to the FEMFFUSION checkout), and any ``--set
Key=Value``. ``--opt`` passes raw command-line options (e.g. ``-precond_type
gs-cgcheb``). The run is wrapped in ``/usr/bin/time -v`` and optionally pinned
with ``taskset``.

Recorded, in the schema of tools/assess_garnet_vs_femffusion.py:

- ``k`` (``K0``), DoFs and active cells from stdout;
- ``timings``: FEMFFUSION's own process-CPU milestones (``grid`` = until
  "Grid Done", ``assembly`` = "Grid Done" to "Matrices assembled", ``solve`` =
  "Matrices assembled" to "The eigenvalues are", which includes the degree-1
  P_Init pre-solve), the transient's CPU time, and the wall time of the whole
  process (``wall``, from /usr/bin/time; reading the input and building the
  mesh happen before FEMFFUSION's timer starts);
- ``counters``: outer iterations, inner iterations, mat-vec products and
  preconditioner applications as FEMFFUSION reports them (not comparable with
  GARNET's counts);
- ``memory``: peak RSS from /usr/bin/time and FEMFFUSION's "Max Memory";
- ``node_power``: the .out "Neutron Power" map -- power density over the
  core-volume average, [z][y][x] for the Cartesian cases (holes null) and
  [plane][hex] for the HP-MR;
- ``transient``: time, total power, step widths, CPU time and solver
  iterations per step.

``--repeats N`` runs the process N times and keeps the fastest solve.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import time

CASES = ("iaea3d", "langenbuch3d", "hpmr3d", "hpmr3d-transient")
PATH_KEYS = ("XSECS_Filename", "Mesh_Filename", "Bar_Filename", "PREC_Filename",
             "Read_XML_Filename", "Read_XS_Filename", "Geometry_Filename",
             "Triangulation_Filename")
DEFAULT_SRC = os.path.expanduser(os.environ.get("FEMFFUSION_DIR", "~/FEMFFUSION"))


def read_prm(path):
    """Ordered (key, value) pairs of the ``set Key = Value`` lines."""
    out = []
    for line in open(path):
        m = re.match(r"\s*set\s+(\S+)\s*=\s*(.*?)\s*$", line.split("#")[0])
        if m:
            out.append([m.group(1), m.group(2)])
    return out


def write_prm(entries, path):
    with open(path, "w") as fh:
        fh.writelines(f"set {k} = {v}\n" for k, v in entries)


def _set(entries, key, value):
    for e in entries:
        if e[0] == key:
            e[1] = str(value)
            return
    entries.append([key, str(value)])


def prepare(args, workdir):
    entries = read_prm(args.prm)
    for e in entries:
        if e[0] in PATH_KEYS and e[1].strip() and not os.path.isabs(e[1]):
            e[1] = os.path.join(args.src, e[1])
    _set(entries, "FE_Degree", args.fe_degree)
    _set(entries, "N_Refinements", args.refinements)
    _set(entries, "N_Eigenvalues", 1)
    _set(entries, "Output_Flag", "false")
    _set(entries, "Print_Grid_Flag", "false")
    _set(entries, "Output_Filename", os.path.join(workdir, "run.out"))
    for kv in args.set:
        key, value = kv.split("=", 1)
        _set(entries, key.strip(), value.strip())
    prm = os.path.join(workdir, "run.prm")
    write_prm(entries, prm)
    return prm, dict(entries)


# --- parsing ------------------------------------------------------------------

def _num(pattern, text, cast=float, last=True):
    found = re.findall(pattern, text)
    return cast(found[-1] if last else found[0]) if found else None


def parse_stdout(text):
    flt = r"([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"
    grid = _num(rf"Grid Done\. Time = {flt}", text)
    assembled = _num(rf"Matrices assembled and sent to .*? Time = {flt}", text)
    eig = _num(rf"The eigenvalues are:\s*Time = {flt}", text)
    return dict(
        k=_num(rf"K0 : {flt}", text),
        dofs=_num(r"Number of Total DoFs:\s*(\d+)", text, int),
        cells=_num(r"Number of active cells:\s*(\d+)", text, int),
        femffusion_max_memory_mb=_num(rf"Max Memory {flt} MB", text),
        counters=dict(outers=_num(r"converged in (\d+)\s+iterations", text, int),
                      inners=_num(r"(\d+) inner iterations", text, int),
                      matvecs=_num(r"(\d+) matrix-vector multiplications", text, int),
                      precond_applications=_num(r"(\d+) preconditioner applications",
                                                text, int)),
        timings=dict(grid=grid,
                     assembly=None if None in (grid, assembled) else assembled - grid,
                     solve=None if None in (assembled, eig) else eig - assembled,
                     cpu_static=eig))


def parse_time_v(text):
    wall = re.search(r"Elapsed \(wall clock\) time .*?: (?:(\d+):)?(\d+):([\d.]+)", text)
    seconds = None
    if wall:
        h, m, s = wall.groups()
        seconds = 3600 * int(h or 0) + 60 * int(m) + float(s)
    return dict(wall=seconds,
                user=_num(r"User time \(seconds\): ([\d.]+)", text),
                system=_num(r"System time \(seconds\): ([\d.]+)", text),
                max_rss_bytes=(_num(r"Maximum resident set size \(kbytes\): (\d+)",
                                    text, int) or 0) * 1024)


def _section_numbers(lines, start):
    """Numbers of the block after line ``start`` up to the next header/blank pair."""
    vals = []
    for line in lines[start + 1:]:
        tok = line.split()
        if not tok:
            if vals:
                break
            continue
        try:
            vals += [float(t) for t in tok]
        except ValueError:
            break
    return vals


def parse_out(path):
    lines = open(path).read().splitlines()
    out = {}
    # 7 significant digits here (stdout's "K0 :" has 6, ~1 pcm resolution)
    for n, line in enumerate(lines):
        if line.strip().startswith("The Eigenvalues are"):
            out["k"] = float(lines[n + 1].split()[0])
            break
    try:
        i = lines.index("Neutron Power")
    except ValueError:
        i = None
    if i is not None:
        # "Plane k" headers (3-D), then rows of values in assembly order with
        # holes left blank; the map ends at the next text line.
        planes = []
        for line in lines[i + 1:]:
            if line.startswith("Plane "):
                planes.append([])
                continue
            if not line.strip():
                continue
            try:
                vals = [float(t) for t in line.split()]
            except ValueError:
                break
            if not planes:              # 2-D: one implicit plane
                planes.append([])
            planes[-1] += vals
        out["power_planes"] = planes
    for header, key in (("Time vector", "t"), ("Total Power vector", "P"),
                        ("Delta t", "dt"), ("CPU Time", "cpu"), ("Solver Its", "its")):
        if header in lines:
            out.setdefault("transient", {})[key] = _section_numbers(lines, lines.index(header))
    return out


def node_layout(case, entries, planes):
    """FEMFFUSION's per-assembly list -> [z][y][x] (Cartesian) or [plane][hex]."""
    if case.startswith("hpmr"):
        return planes
    pts = [int(v) for v in entries["Geometry_Points"].split()]
    rows = list(zip(pts[0::2], pts[1::2]))
    nx = max(hi for _, hi in rows)
    grid = []
    for plane in planes:
        it = iter(plane)
        grid.append([[next(it) if lo <= x + 1 <= hi else None for x in range(nx)]
                     for lo, hi in rows])
    return grid


# --- run --------------------------------------------------------------------

def run_once(args, prm, workdir):
    cmd = ["/usr/bin/time", "-v", args.binary, "-f", prm] + args.opt
    if args.taskset is not None:
        cmd = ["taskset", "-c", str(args.taskset)] + cmd
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    # Run against the build's own toolchain (gcc 12.3 runtime, OpenMPI,
    # OpenBLAS), not whatever the calling environment resolves first.
    ldp = args.build_info.get("ld_library_path")
    if ldp:
        env["LD_LIBRARY_PATH"] = ldp + (":" + env["LD_LIBRARY_PATH"]
                                        if env.get("LD_LIBRARY_PATH") else "")
    out_file = os.path.join(workdir, "run.out")
    if os.path.exists(out_file):
        os.remove(out_file)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=args.src, env=env, capture_output=True, text=True)
    wall_outer = time.perf_counter() - t0
    with open(os.path.join(workdir, "run.log"), "w") as fh:
        fh.write(proc.stdout)
        fh.write("\n--- stderr ---\n")
        fh.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"femffusion.exe failed ({proc.returncode}); see {workdir}/run.log")
    res = parse_stdout(proc.stdout)
    res["process"] = parse_time_v(proc.stderr)
    res["process"]["wall_outer"] = wall_outer
    out = parse_out(out_file)
    if "k" in out:
        res["k_stdout"], res["k"] = res["k"], out.pop("k")
    res.update(out)
    return res


def environment(args):
    info = dict(code="femffusion", node=platform.node(), machine=platform.machine(),
                started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), binary=args.binary)
    try:
        with open("/proc/cpuinfo") as fh:
            info["cpu"] = next(l.split(":", 1)[1].strip() for l in fh
                               if l.startswith("model name"))
    except (OSError, StopIteration):
        info["cpu"] = platform.machine()
    if args.build_info:
        info["build"] = args.build_info
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--prm", required=True, help="base .prm")
    parser.add_argument("--fe-degree", type=int, required=True)
    parser.add_argument("--refinements", type=int, default=0)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--opt", nargs=argparse.REMAINDER, default=[],
                        help="raw femffusion.exe options (must come last)")
    parser.add_argument("--src", default=DEFAULT_SRC, help="FEMFFUSION checkout")
    parser.add_argument("--binary", default=None)
    parser.add_argument("--taskset", default=None, help="CPU list for taskset -c")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--keep", action="store_true", help="keep run.out/run.log")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    args.src = os.path.abspath(os.path.expanduser(args.src))
    args.binary = os.path.abspath(args.binary or os.path.join(args.src, "build-release",
                                                              "femffusion.exe"))
    build_info = os.path.join(os.path.dirname(args.binary), "build-info.json")
    args.build_info = json.load(open(build_info)) if os.path.exists(build_info) else {}
    workdir = os.path.abspath(args.workdir)
    os.makedirs(workdir, exist_ok=True)
    prm, entries = prepare(args, workdir)

    runs = [run_once(args, prm, workdir) for _ in range(args.repeats)]
    key = (lambda r: r["timings"]["solve"]) if args.case != "hpmr3d-transient" else \
        (lambda r: r["process"]["wall"])
    best = min(runs, key=lambda r: key(r) if key(r) is not None else float("inf"))
    G = int(entries.get("Energy_Groups", 2))
    out = dict(environment=environment(args), args=vars(args), case=args.case,
               prm=entries,
               mesh=dict(degree=args.fe_degree, refinements=args.refinements,
                         cells=best["cells"], groups=G, dofs=best["dofs"],
                         unknowns=best["dofs"]),
               k=best["k"], converged=best["k"] is not None,
               counters=best["counters"],
               timings=dict(best["timings"], wall=best["process"]["wall"],
                            user=best["process"]["user"], system=best["process"]["system"],
                            solve_all=[r["timings"]["solve"] for r in runs],
                            wall_all=[r["process"]["wall"] for r in runs]),
               memory=dict(max_rss_bytes=best["process"]["max_rss_bytes"],
                           femffusion_max_memory_mb=best["femffusion_max_memory_mb"]),
               device="cpu (1 core)")
    if "power_planes" in best:
        out["node_power"] = node_layout(args.case, entries, best["power_planes"])
    if "transient" in best:
        out["transient"] = best["transient"]
        out["timings"]["cpu_transient"] = (best["transient"].get("cpu") or [None])[-1]
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=1)
    if not args.keep:
        for name in ("run.out",):
            p = os.path.join(workdir, name)
            if os.path.exists(p) and os.path.getsize(p) > 50e6:
                os.remove(p)
    t = out["timings"]
    print(f"RESULT {args.case} FE{args.fe_degree} r{args.refinements} dofs={out['mesh']['dofs']} "
          f"k={out['k']} counters={out['counters']} assembly={t['assembly']} "
          f"solve={t['solve']} wall={t['wall']}")


if __name__ == "__main__":
    main()
