"""Experimental CPU NEM reference/ADF reconstruction with a known flux jump.

The two synthetic regional means define the desired interface correction.
They are manufactured inputs, not transport measurements or validation data.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from ndgpu.nem_geometry import CuboidNodalGrid
from ndgpu.nem_higher_order import NEMHigherOrderDiffusionSolver
from ndgpu.nem_reference import qualify_nem_reference_factors


def solve():
    grid = CuboidNodalGrid([
        [[0., 20.], [0., 10.], [0., 10.]],
        [[20., 50.], [0., 10.], [0., 10.]],
    ], 'reflective', leakage_boundary='primary_rule')
    materials = ([1.2], [.03], [[0.]], [.035], [1.])
    reference_solver = NEMHigherOrderDiffusionSolver(grid, *materials, degree=8)
    # With zero current and k=nuSigmaF/Sigma_a, each local profile is constant.
    # Equal heterogeneous traces then give different regional ADFs.
    factors = qualify_nem_reference_factors(
        reference_solver, k_eff=.035/.03, means=np.array([[1.], [1.1]]),
        currents=np.zeros((reference_solver.n_faces, 1)),
        heterogeneous_faces=np.ones((2, 6, 1)), fuel_mask=np.ones(2, dtype=bool),
        degrees=(4, 6, 8))
    solver = NEMHigherOrderDiffusionSolver(grid, *materials, factors.factors, degree=8)
    forward = solver.solve_eigenvalue(rtol=1e-11)
    return solver, factors, forward


def main():
    _, factors, forward = solve()
    print(f'experimental CPU NEM/ADF k = {forward.k_eff:.10f}')
    print(f'regional mean ratio = {forward.means[1, 0]/forward.means[0, 0]:.10f}')
    print(f'local corrected-trace relative error = {factors.corrected_trace_relative_error:.3g}')


if __name__ == '__main__':
    main()
