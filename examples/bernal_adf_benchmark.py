"""Provisional Bernal et al. simple 3-D reactor; see docs/bernal_adf_benchmark.md.

Usage: python examples/bernal_adf_benchmark.py --refinements 1 2
A replacement integer material map may be supplied with --map (NumPy .npy,
shape (11,11,10), with 0 for excised cells and 1..6 for paper materials).
"""
import argparse
import json
import time

import numpy as np

from garnet import DiffusionEigenSolver, Grid, Material
from garnet.adf import assembly_face_factors
from garnet.backend import asnumpy, synchronize

# Bernal et al. Table 1: D1, D2, absorption, production, downscatter, ADF.
TABLE = np.array([
    [1.384393, .340606, .006168, .032008, .002802, .032638, .021623, .906938, 1.245334],
    [1.403033, .338023, .007356, .054724, .004650, .076756, .019370, .904180, 1.509856],
    [1.396900, .337844, .007383, .061280, .005254, .075934, .019822, .915384, 1.610692],
    [1.230190, .265137, .001713, .034154, 0, 0, .028561, 1, 1],
    [1.779660, .373901, .000735, .013441, 0, 0, .029508, 1, 1],
    [1.519700, .247519, .000399, .010641, 0, 0, .044684, 1, 1],
])
VALKIN = 1.040374
PARCS = 1.041186


def provisional_map():
    """Figure-derived hypothesis; rod heights and hidden cells are uncertain."""
    m = np.zeros((11, 11, 10), dtype=int)
    for x in range(11):
        for y in range(11):
            a, b = abs(x - 5), abs(y - 5)
            if a == b == 5:
                continue
            m[x, y, 0] = 4
            m[x, y, 9] = 5
            m[x, y, 1:9] = 6 if max(a, b) == 5 else (2 if max(a, b) == 4 else 3)
            if a == b == 0:
                m[x, y, 2:9] = 1
            elif (a, b) == (2, 2):
                # Figure 6 appears to show this interior column as six
                # nodes tall, versus four for the two outer columns.
                m[x, y, 3:9] = 1
            elif (a, b) in ((3, 0), (0, 3)):
                m[x, y, 5:9] = 1
    return m


def make_problem(base_map, refinement, adf_on, axial_factors=False, device='cpu',
                 linear_solver=None, precond_degree=0, factor_exponent=1.0, node_factors=None):
    """Build the solver. With ``adf_on``, Table 1 factors raised to
    ``factor_exponent`` are applied (0 gives explicit unity factors), unless
    ``node_factors`` (2, 1210, 6) supplies factors per base node and side.
    The default solver is GMRES with ADFs and CG without.
    """
    base_map = np.asarray(base_map)
    if base_map.shape != (11, 11, 10) or not np.issubdtype(base_map.dtype, np.integer):
        raise ValueError('material map must be an integer array with shape (11,11,10)')
    if np.any((base_map < 0) | (base_map > 6)) or np.count_nonzero(base_map) != 1170:
        raise ValueError('map must contain 1170 active cells and material numbers 1..6')
    if refinement < 1:
        raise ValueError('refinement must be positive')
    materials = [Material(t[:2], t[2:4], t[4:6], [[0., t[6]], [0., 0.]],
                          name=f'Bernal material {i+1}') for i, t in enumerate(TABLE)]
    def refine(a):
        for axis in range(3):
            a = np.repeat(a, refinement, axis=axis)
        return a
    m = refine(base_map)
    active = m != 0
    # Inactive cells must not contain a fission source: the mask decouples
    # leakage, while the solver's source iteration still visits every cell.
    mmap = np.where(active, m - 1, 5)
    # Every original node is its own region, regardless of material number.
    regions = refine(np.arange(base_map.size).reshape(base_map.shape))
    side = np.ones((2, 6, *m.shape))
    if adf_on and node_factors is not None:
        side = np.moveaxis(np.asarray(node_factors)[:, regions, :], -1, 1)
    elif adf_on:
        for g in range(2):
            cell_factor = TABLE[mmap, 7 + g] ** factor_exponent
            side[g, :4] = cell_factor
            if axial_factors:
                side[g, 4:] = cell_factor
    adf = assembly_face_factors(regions, side) if adf_on else None
    grid = Grid(m.shape, (220., 220., 200.))
    solver = DiffusionEigenSolver(grid, materials, mmap, active=active,
        bc='zero-flux', mask_bc='zero-flux', device=device, adf=adf,
        linear_solver=linear_solver or ('gmres' if adf_on else 'cg'),
        precond_degree=precond_degree)
    return solver, m


def run(base_map, refinement, adf_on, axial_factors=False, device='cpu', **options):
    """Solve one configuration; ``options`` are passed to make_problem."""
    t0 = time.perf_counter()
    solver, m = make_problem(base_map, refinement, adf_on, axial_factors, device, **options)
    synchronize(solver.xp)
    setup_seconds = time.perf_counter() - t0
    result = solver.solve(tol_k=1e-9, tol_source=1e-8, inner_rtol_floor=1e-11)
    if not result.converged:
        raise RuntimeError(f'benchmark failed to converge: {result}')
    flux = result.flux
    source = sum(solver.nu_sigma_f[g] * flux[g] for g in range(2)) / result.k_eff
    residuals, sources = [], []
    for g in range(2):
        rhs = solver.chi[g] * source
        for gf in range(2):
            scatter = solver.sigma_s[gf][g]
            if gf != g and scatter is not None:
                rhs = rhs + scatter * flux[gf]
        residuals.append(asnumpy(solver.ops[g].apply(flux[g]) - rhs).ravel())
        sources.append(asnumpy(rhs).ravel())
    relative_residual = np.linalg.norm(np.concatenate(residuals)) / np.linalg.norm(np.concatenate(sources))
    # nuSigma_f * flux is fission neutron production, not physical power.
    production = np.zeros(m.shape)
    for g in range(2):
        production += asnumpy(solver.fields.nu_sigma_f[g]) * result.flux_numpy[g]
    production[m == 0] = 0
    # Integrate back to original radial nodes, summing the full axial column.
    n = refinement
    assembly = production.reshape(11, n, 11, n, 10, n).sum(axis=(1, 3, 4, 5))
    assembly /= assembly[assembly > 0].mean()
    return {'refinement': n, 'adf': adf_on, 'axial_factors': axial_factors,
            'k_eff': result.k_eff,
            'pcm_vs_valkin': 1e5 * (result.k_eff - VALKIN) / VALKIN,
            'pcm_vs_parcs': 1e5 * (result.k_eff - PARCS) / PARCS,
            'relative_eigen_residual': relative_residual,
            'outer_iterations': result.outer_iterations, 'inner_iterations': result.inner_iterations,
            'setup_seconds': setup_seconds, 'solve_seconds': result.solve_seconds,
            'cells': int(m.size), 'active_cells': int(np.count_nonzero(m)), 'device': solver.device, 'normalized_assembly_fission_production': assembly.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--refinements', nargs='+', type=int, default=[1, 2, 3, 4])
    p.add_argument('--map', help='validated base material map, .npy')
    p.add_argument('--axial-factors', action='store_true', help='sensitivity: apply tabulated factors axially too')
    p.add_argument('--device', choices=['cpu', 'gpu', 'auto'], default='cpu')
    p.add_argument('--output', help='write JSON results to this file')
    args = p.parse_args()
    base = np.load(args.map, allow_pickle=False) if args.map else provisional_map()
    result = {'map': args.map or 'provisional figure interpretation',
              'notes': 'zero flux on all exterior faces; chi=(1,0) as in paper Eq. 3; radial ADF only by default; production is not power',
              'device': args.device,
              'runs': [run(base, n, enabled, args.axial_factors, args.device) for n in args.refinements for enabled in (False, True)]}
    text = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, 'w') as out:
            out.write(text + '\n')
    else:
        print(text)


if __name__ == '__main__':
    main()
