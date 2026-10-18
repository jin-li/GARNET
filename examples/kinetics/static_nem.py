"""Synthetic static kinetics example; no reactor data or Serpent run required.

Run from the repository root: python examples/kinetics/static_nem.py
The unequal reflected nodes have identical one-group material, so physical
Lambda=1/(v*nuSigmaF) and beta_eff equals the supplied raw fractions.
"""
import numpy as np
from ndgpu.nem_geometry import CuboidNodalGrid
from ndgpu.nem_higher_order import NEMHigherOrderDiffusionSolver
from ndgpu.nem_kinetics import solve_nem_adjoint, project_nem_kinetics


def main():
    grid = CuboidNodalGrid([
        [[0.,20.],[0.,10.],[0.,10.]],
        [[20.,50.],[0.,10.],[0.,10.]],
    ], 'reflective', leakage_boundary='primary_rule')
    solver = NEMHigherOrderDiffusionSolver(
        grid, [1.2], [.03], [[0.]], [.035], [1.], degree=8)
    beta = np.array([.0002,.0013,.0012,.0026,.0009,.0003])
    decay = [.0124,.0305,.111,.301,1.14,3.01]
    forward = solver.solve_eigenvalue(rtol=1e-12)
    adjoint = solve_nem_adjoint(solver, forward)
    result = project_nem_kinetics(solver, forward, adjoint,
        inverse_velocities=np.full((2,1), 1/220000.),
        delayed_production=np.broadcast_to((.035*beta)[None,:,None,None], (2,6,1,1)),
        decay=decay)
    print(f'k = {result.k_eff:.10f}')
    print(f'physical Lambda = {result.generation_time:.10g} s')
    print(f'discrete lifetime k*Lambda = {result.neutron_lifetime:.10g} s')
    print(f'beta_eff = {result.beta_total:.8f}; families = {result.beta}')
    print(f'low-frequency one-family lambda = {result.lambda_low_frequency:.8g} 1/s')


if __name__ == '__main__':
    main()
