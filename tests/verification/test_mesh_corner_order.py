"""Independent Robin rectangle and reentrant-corner diffusion controls.

PDE: -D Laplacian(u) + absorption*u = production*u/k; lateral
D*normal_derivative(u) + gamma*u = 0, reflecting z. For a rectangle,
u=cos(qx*(x-Lx/2))*cos(qy*(y-Ly/2)); q*tan(q*L/2)=gamma/D
on each axis. Thus lambda=absorption+D*(qx**2+qy**2), k=production/lambda.
This is the continuum Robin law, not an extrapolated zero-flux box.

FEM independently assembles conforming P1 triangles, consistent area mass,
and exact edge mass gamma*length/6*[[2,1],[1,2]]. No FV face weights
enter the reference. The L shape is [0,L]^2 minus [L/2,L]^2.
Its corner control does not establish a cause for any heterogeneous core order.

CLI: python this_file.py --output NEW_FILE.json. Output is exclusive-create.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pytest
import scipy
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from ndgpu import Material, MeshDiffusionEigenSolver, assemble_mesh_3d

LENGTH = 12.0
D = 1.0
ABSORPTION = 0.02
PRODUCTION = 0.04
GAMMA = 0.5
DEPTH = 0.123  # one z cell; distinguish z and lateral face distances


def rectangle_exact(lengths=(LENGTH, LENGTH), gamma=GAMMA):
    q = [brentq(lambda q: q*np.tan(q*length/2)-gamma/D,
                0, np.nextafter(np.pi/length, 0), xtol=1e-14)
         if gamma else 0.0 for length in lengths]
    eigenvalue = ABSORPTION+D*sum(x*x for x in q)
    return eigenvalue, PRODUCTION/eigenvalue


def planar_grid(n, shape):
    if n < 2 or n % 2 or shape not in ("rectangle", "lshape"):
        raise ValueError("even n >= 2 and rectangle/lshape required")
    ids, points, quads = {}, [], []

    def node(i, j):
        if (i, j) not in ids:
            ids[i, j] = len(points)
            points.append((LENGTH*i/n, LENGTH*j/n))
        return ids[i, j]

    for i in range(n):
        for j in range(n):
            if shape == "lshape" and i >= n//2 and j >= n//2:
                continue
            quads.append([node(i, j), node(i+1, j), node(i+1, j+1), node(i, j+1)])
    return np.asarray(points), np.asarray(quads)


def fem_forms(n, shape):
    points, quads = planar_grid(n, shape)
    triangles = np.concatenate((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]))
    xy = points[triangles]
    twice_area = ((xy[:, 1, 0]-xy[:, 0, 0])*(xy[:, 2, 1]-xy[:, 0, 1])
                  -(xy[:, 2, 0]-xy[:, 0, 0])*(xy[:, 1, 1]-xy[:, 0, 1]))
    assert np.all(twice_area > 0)
    area = twice_area/2
    gradients = np.stack((xy[:, [1, 2, 0], 1]-xy[:, [2, 0, 1], 1],
                          xy[:, [2, 0, 1], 0]-xy[:, [1, 2, 0], 0]), axis=2)
    gradients /= twice_area[:, None, None]
    stiffness = area[:, None, None]*np.einsum("tik,tjk->tij", gradients, gradients)
    mass = area[:, None, None]*(np.ones((3, 3))+np.eye(3))/12
    rows = np.broadcast_to(triangles[:, :, None], stiffness.shape).ravel()
    cols = np.broadcast_to(triangles[:, None, :], stiffness.shape).ravel()
    size = (len(points), len(points))
    K = coo_matrix((stiffness.ravel(), (rows, cols)), shape=size).tocsr()
    M = coo_matrix((mass.ravel(), (rows, cols)), shape=size).tocsr()
    edges = np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                                    triangles[:, [2, 0]])), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all((counts == 1) | (counts == 2))
    boundary = edges[counts == 1]
    lengths = np.linalg.norm(points[boundary[:, 1]]-points[boundary[:, 0]], axis=1)
    values = lengths[:, None, None]*np.array([[2, 1], [1, 2]])/6
    rows = np.broadcast_to(boundary[:, :, None], values.shape).ravel()
    cols = np.broadcast_to(boundary[:, None, :], values.shape).ravel()
    B = coo_matrix((values.ravel(), (rows, cols)), shape=size).tocsr()
    return points, K, M, B


def relative_residual(left, right):
    return float(np.linalg.norm(left-right)/(np.linalg.norm(left)+np.linalg.norm(right)))


def fem_level(n, shape, gamma=GAMMA):
    points, K, M, B = fem_forms(n, shape)
    A = D*K+ABSORPTION*M+gamma*B
    eigenvalues, vectors = eigsh(A, M=M, k=1, sigma=0, which="LM", tol=1e-11,
                                v0=np.ones(len(points)))
    lam, u = float(eigenvalues[0]), vectors[:, 0]
    return {"n": n, "unknowns": len(points), "lambda": lam,
            "k": PRODUCTION/lam, "residual": relative_residual(A@u, lam*(M@u))}


def fv_problem(n, shape, gamma=GAMMA):
    points, quads = planar_grid(n, shape)
    count = len(points)
    coords = np.concatenate((np.column_stack((points, np.zeros(count))),
                             np.column_stack((points, np.full(count, DEPTH)))))
    cells = np.concatenate((quads, quads+count), axis=1)
    mesh = assemble_mesh_3d(coords, cells, [0]*len(cells), normal_distances=True)
    # On these orthogonal hexes centroid-to-face distance identifies z faces.
    assert not np.isclose(LENGTH/(2*n), DEPTH/2)
    bc = ["reflective" if np.isclose(distance, DEPTH/2, rtol=0, atol=1e-12)
          else ("vacuum" if gamma == GAMMA else gamma) for _, _, distance in mesh.bfaces]
    assert bc.count("reflective") == 2*mesh.n_cells
    mat = Material([D], [ABSORPTION], [PRODUCTION])
    return MeshDiffusionEigenSolver(mesh, mat, boundary_conditions=bc, device="cpu")


def fv_level(n, shape, gamma=GAMMA):
    solver = fv_problem(n, shape, gamma)
    result = solver.solve(tol_k=1e-12, tol_source=1e-10, inner_rtol_floor=1e-12)
    assert result.converged
    u = result.flux[0]
    lam = PRODUCTION/float(result.k_eff)
    right = lam*solver.grid.integration_weights.reshape(solver.grid.shape)*u
    return {"n": n, "unknowns": solver.grid.n_cells, "lambda": lam,
            "k": float(result.k_eff),
            "residual": relative_residual(solver.ops[0].apply(u), right)}


@pytest.mark.parametrize("shape,area,perimeter", [("rectangle", 144., 48.), ("lshape", 108., 48.)])
def test_independent_geometry_and_forms(shape, area, perimeter):
    points, K, M, B = fem_forms(8, shape)
    one = np.ones(len(points))
    assert one@M@one == pytest.approx(area)
    assert one@B@one == pytest.approx(perimeter)
    np.testing.assert_allclose(K@one, 0, atol=1e-13)
    # Exact P1 volume/boundary integrals of x^2 distinguish consistent mass
    # from lumping and detect missing notch edges or an incorrect L footprint.
    x = points[:, 0]
    expected_x2 = LENGTH**4/3*(1 if shape == "rectangle" else 9/16)
    assert x@M@x == pytest.approx(expected_x2)
    expected_edge_x2 = LENGTH**3*(5/3 if shape == "rectangle" else 31/24)
    assert x@B@x == pytest.approx(expected_edge_x2)
    assert x@K@x == pytest.approx(area)
    solver = fv_problem(8, shape)
    assert solver.grid.mesh.area.sum()/DEPTH == pytest.approx(area)
    assert sum(S for (_, S, _), spec in zip(solver.grid.mesh.bfaces,
               solver.grid.boundary_conditions) if spec != "reflective")/DEPTH == pytest.approx(perimeter)


def test_fundamental_robin_rectangle():
    exact_lambda, exact_k = rectangle_exact()
    assert exact_lambda > ABSORPTION
    for method in (fem_level, fv_level):
        levels = [method(n, "rectangle") for n in (8, 16, 32)]
        errors = np.array([abs(level["k"]-exact_k) for level in levels])
        assert np.all(errors[1:] < errors[:-1])
        assert np.all((errors[:-1]/errors[1:] > 3.5) & (errors[:-1]/errors[1:] < 4.5))
        assert max(level["residual"] for level in levels) < 1e-8
        # Wrong gamma or a Dirichlet interpretation cannot pass this accuracy.
        assert errors[-1] < 5e-4
    assert fem_level(16, "rectangle", gamma=0)["k"] == pytest.approx(PRODUCTION/ABSORPTION, rel=1e-10)
    assert fv_level(16, "lshape", gamma=0)["k"] == pytest.approx(PRODUCTION/ABSORPTION, rel=1e-9)


def test_lshape_independent_refinement():
    reference = [fem_level(n, "lshape") for n in (16, 32, 64)]
    # Nested conforming Rayleigh-Ritz spaces give upper bounds on lambda.
    assert reference[0]["lambda"] > reference[1]["lambda"] > reference[2]["lambda"]
    assert reference[2]["k"]-reference[1]["k"] < reference[1]["k"]-reference[0]["k"]
    fv = [fv_level(n, "lshape") for n in (8, 16, 32)]
    errors = [abs(level["k"]-reference[-1]["k"]) for level in fv]
    assert errors[2] < errors[1] < errors[0]
    assert max(level["residual"] for level in reference+fv) < 1e-8


def diagnostics(levels, reference):
    differences = [float(np.log2(abs((a["k"]-b["k"])/(b["k"]-c["k"]))))
                   for a, b, c in zip(levels, levels[1:], levels[2:])]
    if reference is None:
        return {"successive_difference_orders": differences}
    errors = [abs(level["k"]-reference) for level in levels]
    return {"absolute_k_errors": errors,
            "orders_against_reference": [float(np.log2(a/b)) for a, b in zip(errors, errors[1:])],
            "successive_difference_orders": differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fem-max", type=int, choices=(128, 256), default=128)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new immutable path")
    root = Path(__file__).resolve().parents[2]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = {"controls": {"D": D, "absorption": ABSORPTION, "production": PRODUCTION,
                          "gamma": GAMMA, "length": LENGTH, "z_depth": DEPTH},
              "provenance": {"python": sys.version, "executable": sys.executable,
                  "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
                  "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                  "dirty": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
                  "sources_sha256": {str(p.relative_to(root)): sha(p) for p in
                      [Path(__file__).resolve(), *sorted((root/"ndgpu").rglob("*.py"))]},
                  "environment_sha256": {str(p): sha(p) for p in
                      [Path(sys.executable), root/"flake.lock", root/"uv.lock"] if p.exists()},
                  "threads": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "MKL_NUM_THREADS", "NUMEXPR_MAX_THREADS", "NUMEXPR_NUM_THREADS"]}},
              "qualifications": ["Homogeneous geometry control; does not prove the cause of core refinement orders.",
                  "L-shape reference is finite-resolution conforming FEM, not a fitted continuum limit.",
                  "FEM last-level difference is a refinement diagnostic, not a rigorous error bound.",
                  "No parameter fitting; both methods use the same continuum Robin law."], "shapes": {}}
    report["provenance"]["observed_process_threads"] = len(list(Path("/proc/self/task").iterdir()))
    for shape in ("rectangle", "lshape"):
        fem = [fem_level(n, shape) for n in (16, 32, 64, 128, 256) if n <= args.fem_max]
        fv = [fv_level(n, shape) for n in (8, 16, 32, 64)]
        exact = rectangle_exact()[1] if shape == "rectangle" else None
        report["shapes"][shape] = {"fem": fem, "fv": fv, "analytic_k": exact, "analytic_lambda": rectangle_exact()[0] if exact else None,
            "fem_reference_k": fem[-1]["k"], "fem_reference_last_change": abs(fem[-1]["k"]-fem[-2]["k"]),
            "fem_diagnostics": diagnostics(fem, exact),
            "fv_diagnostics": diagnostics(fv, exact if exact else fem[-1]["k"])}
    for relative, expected in report["provenance"]["sources_sha256"].items():
        if sha(root/relative) != expected:
            raise RuntimeError(f"source changed during calculation: {relative}")
    report["provenance"]["observed_process_threads_after"] = len(list(Path("/proc/self/task").iterdir()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({shape: {key: value for key, value in data.items() if key not in ("fem", "fv")}
                      for shape, data in report["shapes"].items()}, indent=2))


if __name__ == "__main__":
    main()
