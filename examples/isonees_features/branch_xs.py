"""Synthetic two-coordinate branch library; no Serpent or reactor data needed."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from garnet.branch_xs import (BenchmarkModeratorPath, BranchState, BranchLibrary,
                             build_nem_solver)
from garnet.branch_validation import compare_reconstruction_chain
from garnet.nem_geometry import CuboidNodalGrid


def solve():
    path = BenchmarkModeratorPath()
    grid = CuboidNodalGrid([
        [[0., 10.], [0., 10.], [0., 10.]],
        [[10., 30.], [0., 10.], [0., 10.]],
    ], 'reflective', leakage_boundary='primary_rule')
    points = [('nominal',600.,573.15), ('fuel-low',500.,573.15),
              ('fuel-high',700.,573.15), ('water-cold',600.,558.15),
              ('water-hot',600.,588.15)]
    states = []
    for name, tf, tm in points:
        rho = path.evaluate(tm)
        absorption = .03 + .0002*(np.sqrt(tf)-np.sqrt(600.)) + .02*(rho-.72683)
        states.append(BranchState(name, dict(configuration='synthetic',family='thermal',
             fuel_temperature=tf,moderator_density=rho), ('left','right'), [20.,0.],
             np.full((2,1),1.2), np.full((2,1),absorption), np.zeros((2,1,1)),
             np.full((2,1,1),.035), np.ones((2,1)),
             exterior_mask=grid.neighbors < 0,
             metadata=dict(geometry_family='two-cuboid-synthetic',
                           nuclear_data_family='manufactured',
                           moderator_path_id=path.path_id),
             qualification='synthetic-fixture'))
    elements = [('nominal', f, m) for f in ('fuel-low','fuel-high')
                for m in ('water-cold','water-hot')]
    library = BranchLibrary(states, simplices=elements)
    state = library.evaluate_temperatures(640.,578.15,path,
               configuration='synthetic',family='thermal')
    solver = build_nem_solver(state,grid,degree=4)
    result = solver.solve_eigenvalue(rtol=1e-11)
    # A reflected homogeneous one-group model has k=nuSigma_f/Sigma_a,
    # independent of cuboid sizes. This checks interpolation through the solver.
    exact = .035/state.sigma_a[0,0]
    report = compare_reconstruction_chain(result.k_eff,exact,exact)
    return library,state,solver,result,exact,report


def main():
    _,state,_,result,exact,report = solve()
    print('Sources:', dict(zip(state.contributing_ids,state.weights)))
    print(f'k={result.k_eff:.12f}; analytic k={exact:.12f}')
    print('Interpolation fixture:',report['status'])
    print('Qualification:',state.qualification)


if __name__ == '__main__':
    main()
