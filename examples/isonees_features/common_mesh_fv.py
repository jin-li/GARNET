"""Reflected common-mesh FV eigenproblem with an analytical homogeneous limit."""
from pathlib import Path
import sys

# Permit the documented direct invocation from an uninstalled checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from ndgpu import Material, MeshDiffusionEigenSolver
from ndgpu.nodal_mesh import build_rectilinear_mesh_3d


def solve():
    mesh, topology = build_rectilinear_mesh_3d(
        [0., 20., 50.], [0., 10.], [0., 10.],
        active_mask=np.ones((1, 1, 2), dtype=bool),
        tags=np.zeros((1, 1, 2), dtype=int))
    material = Material([1.2], [.03], [.035], [[0.]], [1.])
    solver = MeshDiffusionEigenSolver(mesh, [material], mesh.cell_tag,
                                    boundary_conditions='reflective', device='cpu')
    result = solver.solve(tol_k=1e-11, tol_source=1e-10)
    return mesh, topology, result


def main():
    mesh, _, result = solve()
    if not result.converged:
        raise RuntimeError('FV source iteration did not converge')
    print(f'FV k = {result.k_eff:.10f}; analytical k = {.035/.03:.10f}')
    print(f'cells = {mesh.n_cells}; volume = {mesh.area.sum():.1f} cm^3')


if __name__ == '__main__':
    main()
