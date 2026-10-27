"""Two-group, six-layer analytical diffusion reference-DF verification.

This synthetic geometry uses material data published by Dall'Osso (PHYSOR
2014), but is not the paper's benchmark geometry. Full definition and
limitations are in docs/adf_slab_reference.md.
"""
import argparse
import json

import numpy as np
from scipy.linalg import expm, solve, svd
from scipy.optimize import brentq

from garnet import DiffusionEigenSolver, Grid, Material, assembly_face_factors

# D1,D2, Sigma_a1,a2, nuSigma_f1,f2, Sigma_s1->2; cm and cm^-1.
MATERIALS = np.array([[1.3200, .3830, .0090, .0800, .0060, .1040, .0170],
                      [1.4343, .3800, .0144, .2590, .0107, .4200, .0120],
                      [1.4343, .3800, .0130, .2037, .0068, .2660, .0120],
                      [1.6000, .3200, .0006, .0240, 0., 0., .0330]])
# width and material index, two layers per 10 cm region.
LAYERS = [(4., 0), (6., 3), (6., 1), (4., 2), (5., 0), (5., 3)]


def _matrix(k, material):
    d1, d2, a1, a2, p1, p2, scatter = material
    # y=(phi1,J1,phi2,J2, integral(phi1),integral(phi2))
    return np.array([[0, -1/d1, 0, 0, 0, 0],
                     [p1/k-a1-scatter, 0, p2/k, 0, 0, 0],
                     [0, 0, 0, -1/d2, 0, 0],
                     [scatter, 0, -a2, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0]])


def _transfer(k):
    result = np.eye(6)
    for width, material in LAYERS:
        result = expm(_matrix(k, MATERIALS[material]) * width) @ result
    return result


def heterogeneous_reference():
    def boundary_determinant(k):
        return np.linalg.det(_transfer(k)[np.ix_([1, 3], [0, 2])])
    # Bracket contains the positive fundamental eigenvalue of this case.
    k = brentq(boundary_determinant, .9, .99, xtol=1e-14)
    boundary = _transfer(k)[np.ix_([1, 3], [0, 2])]
    _, _, vt = svd(boundary)
    initial_flux = vt[-1] / vt[-1, 0]
    state = np.array([initial_flux[0], 0., initial_flux[1], 0., 0., 0.])
    edge_flux = [state[[0, 2]].copy()]
    edge_current = [state[[1, 3]].copy()]
    integrals = []
    min_flux = min(initial_flux)
    for i, (width, material) in enumerate(LAYERS):
        state[4:] = 0.
        for x in np.linspace(0, width, 101):
            min_flux = min(min_flux, np.min((expm(_matrix(k, MATERIALS[material])*x) @ state)[[0, 2]]))
        state = expm(_matrix(k, MATERIALS[material]) * width) @ state
        integrals.append(state[4:].copy())
        if i % 2:
            edge_flux.append(state[[0, 2]].copy())
            edge_current.append(state[[1, 3]].copy())
    if min_flux <= 0:
        raise RuntimeError('selected eigenvalue does not have positive group fluxes')
    integrals = np.asarray(integrals).reshape(3, 2, 2)
    avg = integrals.sum(axis=1) / 10.
    weights = integrals / integrals.sum(axis=1)[:, None, :]
    coefficients = []
    for r in range(3):
        layer_xs = np.array([MATERIALS[LAYERS[2*r+j][1]] for j in range(2)])
        w1, w2 = weights[r, :, 0], weights[r, :, 1]
        coefficients.append([np.dot(layer_xs[:, i], w1 if i in (0, 2, 4, 6) else w2)
                             for i in range(7)])
    return dict(k=k, avg=avg, coeff=np.asarray(coefficients),
                edge_flux=np.asarray(edge_flux), edge_current=np.asarray(edge_current))


def reference_factors(ref, n):
    """Fixed-current independent homogeneous multigroup solves in each region."""
    factors = np.ones((3, 2, 2))  # region, side x-/x+, group
    region_flux = []
    for r, (d1, d2, a1, a2, p1, p2, sc) in enumerate(ref['coeff']):
        h = 10. / n
        lap = np.diag(np.full(n, 2.))
        if n > 1:
            lap += np.diag(np.full(n-1, -1.), 1) + np.diag(np.full(n-1, -1.), -1)
        lap[0, 0] -= 1
        lap[-1, -1] -= 1
        eye = np.eye(n)
        matrix = np.block([[d1/h**2*lap + (a1+sc-p1/ref['k'])*eye, -p2/ref['k']*eye],
                           [-sc*eye, d2/h**2*lap+a2*eye]])
        rhs = np.zeros((2, n))
        rhs[:, 0] += ref['edge_current'][r] / h
        rhs[:, -1] -= ref['edge_current'][r+1] / h
        phi = solve(matrix, rhs.ravel()).reshape(2, n)
        region_flux.append(phi)
        for g, D in enumerate((d1, d2)):
            left = phi[g, 0] + ref['edge_current'][r, g] * h / (2*D)
            right = phi[g, -1] - ref['edge_current'][r+1, g] * h / (2*D)
            if r:
                factors[r, 0, g] = ref['edge_flux'][r, g] / left
            if r < 2:
                factors[r, 1, g] = ref['edge_flux'][r+1, g] / right
    if not np.all(np.isfinite(factors) & (factors > 0)):
        raise RuntimeError('reference factors are non-positive')
    return factors, np.asarray(region_flux)


def run(n, enabled):
    if n < 1:
        raise ValueError('refinement must be positive')
    ref = heterogeneous_reference()
    factors, expected = reference_factors(ref, n) if enabled else (None, None)
    grid = Grid((3*n, 1, 1), (30., 1., 1.))
    regions = np.repeat(np.arange(3), n).reshape(grid.shape)
    side = np.ones((2, 6, *grid.shape))
    if enabled:
        for r in range(3):
            for g in range(2):
                side[g, 0][regions == r] = factors[r, 0, g]
                side[g, 1][regions == r] = factors[r, 1, g]
    adf = assembly_face_factors(regions, side) if enabled else None
    materials = [Material([d1, d2], [a1, a2], [p1, p2], [[0, sc], [0, 0]])
                 for d1, d2, a1, a2, p1, p2, sc in ref['coeff']]
    solver = DiffusionEigenSolver(grid, materials, regions, bc='reflective', device='cpu',
                                  adf=adf, linear_solver='gmres' if enabled else 'cg')
    result = solver.solve(tol_k=1e-12, tol_source=1e-11, inner_rtol_floor=1e-13)
    if not result.converged:
        raise RuntimeError(f'benchmark failed to converge: {result}')
    flux = result.flux_numpy.reshape(2, 3, n)
    flux *= ref['avg'][:, 0].sum() / flux[0].mean(axis=1).sum()
    averages = flux.mean(axis=2).T
    currents = np.array([solver.ops[g].face_currents(flux[g].reshape(grid.shape))[0].ravel()[[n-1, 2*n-1]]
                         for g in range(2)]).T
    production = sum(solver.nu_sigma_f[g]*flux[g].reshape(grid.shape) for g in range(2)) / result.k_eff
    residual = []
    for g in range(2):
        rhs = solver.chi[g]*production
        if g == 1:
            rhs = rhs + solver.sigma_s[0][1]*flux[0].reshape(grid.shape)
        residual.append(solver.ops[g].apply(flux[g].reshape(grid.shape))-rhs)
    relative_residual = np.linalg.norm(np.stack(residual)) / np.linalg.norm(production)
    return dict(refinement=n, reference_factors=enabled, k=result.k_eff,
                reference_k=ref['k'], delta_k_pcm=1e5*(result.k_eff-ref['k'])/ref['k'],
                region_flux=averages.tolist(), reference_region_flux=ref['avg'].tolist(),
                region_flux_max_relative_error=float(np.max(np.abs(averages/ref['avg']-1))),
                currents=currents.tolist(), reference_currents=ref['edge_current'][1:-1].tolist(),
                current_max_abs_error=float(np.max(np.abs(currents-ref['edge_current'][1:-1]))),
                relative_eigen_residual=float(relative_residual),
                maximum_cell_flux_error_vs_reference_reconstruction=(float(np.max(np.abs(flux-expected.transpose(1, 0, 2))))
                                                                    if enabled else None),
                factors=None if factors is None else factors.tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--refinements', nargs='+', type=int, default=[4, 8, 16])
    args = p.parse_args()
    print(json.dumps([run(n, enabled) for n in args.refinements for enabled in (False, True)], indent=2))


if __name__ == '__main__':
    main()
