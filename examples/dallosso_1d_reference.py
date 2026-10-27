"""Heterogeneous diffusion comparison with Dall'Osso PHYSOR 2014 Fig. 2.

The reported 1.04125 reference is not exactly reproduced under the assumptions
below. See docs/dallosso_1d_reference.md for geometry and unresolved details.
"""
import argparse
import json

import numpy as np
from scipy.linalg import det, expm, solve, svd
from scipy.optimize import brentq

from garnet import DiffusionEigenSolver, Grid, Material, assembly_face_factors

# Table 1: D1,D2, absorption 1,2, nu*fission 1,2, downscatter.
TABLE = np.array([[1.3200, .3830, .0090, .0800, .0060, .1040, .0170],
                  [1.4343, .3800, .0144, .2590, .0107, .4200, .0120],
                  [1.4343, .3800, .0130, .2037, .0068, .2660, .0120],
                  [1.6000, .3200, .0006, .0240, 0., 0., .0330],
                  [.9600, .3100, .0057, .1700, 0., 0., .00007]])
# Material indices U, M1, M2, M1, W gap, B baffle, W reflector.
LENGTHS = np.array([31.5, 3.15, 14.7, 3.15, .525, 3.15, 17.325])
INDICES = np.array([0, 1, 2, 1, 3, 4, 3])
PUBLISHED_K = 1.04125


def analytical_k():
    """Piecewise-exact two-group diffusion by independent multi-shooting.

    Each 4x4 transfer matrix advances both fluxes and currents across one
    homogeneous segment. Splitting long layers improves conditioning; the
    determinant vanishes when all transfers and the four BCs are compatible.
    """
    segments = []
    for length, material in zip(LENGTHS, INDICES):
        n = int(np.ceil(length / 2.))
        segments.extend([(length/n, material)] * n)

    def system(k):
        count = len(segments)
        matrix = np.zeros((4*(count+1), 4*(count+1)))
        for i, (width, material) in enumerate(segments):
            d1, d2, a1, a2, p1, p2, scatter = TABLE[material]
            ode = np.array([[0, -1/d1, 0, 0],
                            [p1/k-a1-scatter, 0, p2/k, 0],
                            [0, 0, 0, -1/d2], [scatter, 0, -a2, 0]])
            matrix[4*i:4*i+4, 4*i:4*i+4] = -expm(ode*width)
            matrix[4*i:4*i+4, 4*i+4:4*i+8] = np.eye(4)
        matrix[-4, 1] = matrix[-3, 3] = 1.
        matrix[-2, 4*count] = matrix[-1, 4*count+2] = 1.
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    k = brentq(lambda value: det(system(value)), 1.03, 1.04, xtol=1e-13)
    _, _, vh = svd(system(k))
    state = vh[-1].reshape(-1, 4)
    state /= state[0, 0]
    if np.min(state[:-1, [0, 2]]) <= 0:
        raise RuntimeError('analytical root does not have positive group fluxes')
    return k


def _heterogeneous_solver(refinement):
    if refinement < 1:
        raise ValueError('refinement must be positive')
    n = 140 * refinement
    grid = Grid((n, 1, 1), (73.5, 1., 1.))
    x = grid.cell_centers(0)
    layer = np.searchsorted(np.cumsum(LENGTHS), x)
    mmap = INDICES[layer].reshape(grid.shape)
    materials = [Material([d1, d2], [a1, a2], [p1, p2], [[0., scatter], [0., 0.]])
                 for d1, d2, a1, a2, p1, p2, scatter in TABLE]
    solver = DiffusionEigenSolver(grid, materials, mmap, device='cpu',
                                  bc=(('reflective', 'zero-flux'), 'reflective', 'reflective'))
    return solver


def run(refinement):
    solver = _heterogeneous_solver(refinement)
    grid = solver.grid
    result = solver.solve(tol_k=1e-11, tol_source=1e-10, inner_rtol_floor=1e-12)
    if not result.converged:
        raise RuntimeError(f'solve failed: {result}')
    production = sum(solver.nu_sigma_f[g] * result.flux_numpy[g] for g in range(2)).ravel()
    # Five 10.5 cm fuel segments in Fig. 2; nu Sigma_f production is not power.
    fission = production[:100*refinement].reshape(5, 20*refinement).mean(axis=1)
    fission /= fission.mean()
    source = sum(solver.nu_sigma_f[g] * result.flux_numpy[g] for g in range(2)) / result.k_eff
    residual, rhses = [], []
    for g in range(2):
        rhs = solver.chi[g]*source
        if g == 1:
            rhs = rhs + solver.sigma_s[0][1]*result.flux_numpy[0]
        residual.append(solver.ops[g].apply(result.flux_numpy[g])-rhs)
        rhses.append(rhs)
    relative_residual = np.linalg.norm(np.stack(residual))/np.linalg.norm(np.stack(rhses))
    return dict(refinement=refinement, dx_cm=grid.spacing[0], k=result.k_eff,
                reported_paper_k=PUBLISHED_K,
                delta_k_pcm=1e5*(result.k_eff-PUBLISHED_K),
                normalized_fission_production=fission.tolist(),
                paper_normalized_power=[1.333, 1.232, .979, .905, .551],
                relative_eigen_residual=float(relative_residual))


def reference_factor_comparison(refinement):
    """Match this exact fine-mesh heterogeneous solution using reference DFs.

    The five 10.5 cm fuel nodes use flux-weighted XS and a local two-group
    fixed-current solve; the reflector remains fully heterogeneous on the same
    fine mesh, which preserves the physical outer boundary without introducing
    an unprovided boundary factor. This is reference equivalence, not a
    prediction using independent assembly-library data.
    """
    fine = _heterogeneous_solver(refinement)
    grid = fine.grid
    ref = fine.solve(tol_k=1e-11, tol_source=1e-10, inner_rtol_floor=1e-11)
    if not ref.converged:
        raise RuntimeError(f'heterogeneous solve failed: {ref}')
    flux = ref.flux_numpy.reshape(2, -1)
    cells = 20 * refinement
    h = grid.spacing[0]
    interfaces = np.arange(1, 6) * cells
    currents = np.zeros((6, 2))
    traces = np.zeros((6, 2))
    for g in range(2):
        currents[1:, g] = fine.ops[g].face_currents(ref.flux_numpy[g])[0].ravel()[interfaces-1]
        D = fine.fields.diffusion[g].ravel()
        traces[1:, g] = flux[g, interfaces-1] - currents[1:, g] * h / (2*D[interfaces-1])
    regions = np.minimum(np.arange(grid.shape[0]) // cells, 5).reshape(grid.shape)
    side = np.ones((2, 6, *grid.shape))
    hom_materials = []
    for region in range(5):
        sl = slice(region*cells, (region+1)*cells)
        coeff = []
        for field in (fine.fields.diffusion, fine.fields.sigma_a, fine.fields.nu_sigma_f):
            coeff.append([float(np.dot(field[g].ravel()[sl], flux[g, sl]) / flux[g, sl].sum())
                          for g in range(2)])
        scatter = float(np.dot(fine.sigma_s[0][1].ravel()[sl], flux[0, sl]) / flux[0, sl].sum())
        D, a, p = coeff
        hom_materials.append(Material(D, a, p, [[0., scatter], [0., 0.]]))
        lap = np.diag(np.full(cells, 2.))
        lap += np.diag(np.full(cells-1, -1.), 1) + np.diag(np.full(cells-1, -1.), -1)
        lap[0, 0] -= 1
        lap[-1, -1] -= 1
        eye = np.eye(cells)
        matrix = np.block([[D[0]/h**2*lap + (a[0]+scatter-p[0]/ref.k_eff)*eye, -p[1]/ref.k_eff*eye],
                           [-scatter*eye, D[1]/h**2*lap+a[1]*eye]])
        rhs = np.zeros((2, cells))
        rhs[:, 0] += currents[region] / h
        rhs[:, -1] -= currents[region+1] / h
        homogeneous = solve(matrix, rhs.ravel()).reshape(2, cells)
        for g in range(2):
            if region:
                trace = homogeneous[g, 0] + currents[region, g]*h/(2*D[g])
                side[g, 0][regions == region] = traces[region, g] / trace
            trace = homogeneous[g, -1] - currents[region+1, g]*h/(2*D[g])
            side[g, 1][regions == region] = traces[region+1, g] / trace
    if not np.all(np.isfinite(side) & (side > 0)):
        raise RuntimeError('non-positive reference factors on this mesh')
    materials = hom_materials + [Material([d1, d2], [a1, a2], [p1, p2], [[0., sc], [0., 0.]])
                                 for d1, d2, a1, a2, p1, p2, sc in TABLE]
    original_map = INDICES[np.searchsorted(np.cumsum(LENGTHS), grid.cell_centers(0))].reshape(grid.shape)
    homogenized_map = np.where(regions < 5, regions, 5 + original_map)
    adf = assembly_face_factors(regions, side)
    runs = []
    for enabled in (False, True):
        s = DiffusionEigenSolver(grid, materials, homogenized_map,
                                 bc=(('reflective', 'zero-flux'), 'reflective', 'reflective'),
                                 device='cpu', adf=adf if enabled else None,
                                 linear_solver='gmres' if enabled else 'cg')
        result = s.solve(tol_k=1e-11, tol_source=1e-10, inner_rtol_floor=1e-11)
        if not result.converged:
            raise RuntimeError(f'homogenized solve failed: {result}')
        result_flux = result.flux_numpy.reshape(2, -1)
        result_flux *= flux[0, :5*cells].sum() / result_flux[0, :5*cells].sum()
        ref_avg = flux[:, :5*cells].reshape(2, 5, cells).mean(axis=2)
        avg = result_flux[:, :5*cells].reshape(2, 5, cells).mean(axis=2)
        J = np.array([s.ops[g].face_currents(result_flux[g].reshape(grid.shape))[0].ravel()[interfaces-1]
                      for g in range(2)]).T
        runs.append(dict(reference_factors=enabled, k=result.k_eff,
                         delta_k_pcm=1e5*(result.k_eff-ref.k_eff),
                         max_region_flux_relative_error=float(np.max(np.abs(avg/ref_avg-1))),
                         max_interface_current_absolute_error=float(np.max(np.abs(J-currents[1:])))))
    return dict(refinement=refinement, fine_reference_k=ref.k_eff, runs=runs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--refinements', nargs='+', type=int, default=[1, 2, 4, 8])
    p.add_argument('--reference-factors', action='store_true',
                   help='also run reference-factor equivalence at each requested resolution')
    args = p.parse_args()
    print(json.dumps({'analytical_interpreted_k': analytical_k(),
                      'heterogeneous': [run(n) for n in args.refinements],
                      'reference_factor_comparison': ([reference_factor_comparison(n) for n in args.refinements]
                                                      if args.reference_factors else [])}, indent=2))


if __name__ == '__main__':
    main()
