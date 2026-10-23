"""Synthetic two-family raw regional contract and matching CPU NEM adjoint.

The JSON is generated in a temporary directory; no Serpent or ACE data is used.
"""
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from garnet.kinetics_data import read_regional_kinetics
from garnet.nem_geometry import CuboidNodalGrid
from garnet.nem_higher_order import NEMHigherOrderDiffusionSolver
from garnet.nem_kinetics import solve_nem_adjoint, project_nem_kinetics


RAW_BETA = np.array([.002, .004])
DECAY = np.array([.08, .3])
SPEED = 220000.  # cm/s
NU_SIGMA_F = .035  # 1/cm


def raw_contract():
    """One synthetic material region, explicitly mapped onto two physical nodes."""
    return dict(
        schema_version=1, quantity_kind='raw_delayed_production',
        units=dict(energy='MeV', inverse_velocity='s/cm', production='1/cm', decay='1/s'),
        region_ids=['synthetic_fuel'], family_ids=['slow', 'fast'],
        energy_bounds=[20., 0.], inverse_velocities=[[1/SPEED]], decay=DECAY.tolist(),
        total_production=[[[NU_SIGMA_F]]],
        delayed_production=(NU_SIGMA_F*RAW_BETA).reshape(1, 2, 1, 1).tolist(),
        provenance={'synthetic_definition': hashlib.sha256(
            b'GARNET synthetic one-group two-family example v1').hexdigest()},
        approximations=['Manufactured one-group material; constant raw family fractions.'])


def solve():
    grid = CuboidNodalGrid([
        [[0., 20.], [0., 10.], [0., 10.]],
        [[20., 50.], [0., 10.], [0., 10.]],
    ], 'reflective', leakage_boundary='primary_rule')
    solver = NEMHigherOrderDiffusionSolver(
        grid, [1.2], [.03], [[0.]], [NU_SIGMA_F], [1.], degree=8)
    with TemporaryDirectory(prefix='garnet-synthetic-') as directory:
        path = Path(directory)/'raw_kinetics.json'
        path.write_text(json.dumps(raw_contract()), encoding='utf-8')
        data = read_regional_kinetics(path)
        inputs = data.for_nodes(['synthetic_fuel']*2, [20., 0.], solver.fission)
    forward = solver.solve_eigenvalue(rtol=1e-12)
    adjoint = solve_nem_adjoint(solver, forward)
    parameters = project_nem_kinetics(solver, forward, adjoint, **inputs)
    return forward, adjoint, parameters


def main():
    _, _, result = solve()
    print(f'k = {result.k_eff:.10f}; beta_eff = {result.beta_total:.8f}')
    print(f'physical Lambda = {result.generation_time:.10g} s')
    print(f'discrete lifetime k*Lambda = {result.neutron_lifetime:.10g} s')
    print(f'low-frequency one-family lambda = {result.lambda_low_frequency:.10g} 1/s')


if __name__ == '__main__':
    main()
