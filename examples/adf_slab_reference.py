"""Reference-DF verification against an analytical heterogeneous slab.

Install GARNET first (python -m pip install -e .).
Run: python examples/adf_slab_reference.py --refinements 1 2 4 8
The reference is one-group diffusion, not particle transport. Full case
specification and derivation: docs/adf_slab_reference.md.
"""
import argparse
import json

import numpy as np
from scipy.linalg import expm, solve
from scipy.optimize import brentq

from garnet import DiffusionEigenSolver, Grid, Material, assembly_face_factors

# Two layers per 10 cm region. (width cm, D cm, Sigma_a cm^-1, nu Sigma_f cm^-1)
LAYERS = np.array([[4., 1.3, .008, .014], [6., .25, .065, .035],
                   [6., .4, .035, .04], [4., 1., .012, .025],
                   [5., 1.5, .009, .013], [5., .18, .08, .055]])


def _transfer(k, layer, state, width=None):
    length, D, absorption, production = layer
    matrix = np.array([[0, -1 / D, 0], [production / k - absorption, 0, 0], [1, 0, 0]])
    return expm(matrix * (length if width is None else width)) @ state


def heterogeneous_reference():
    """Solve continuous piecewise-exact diffusion with reflective boundaries.

    phi(0)=1 sets normalization. Transfer matrices propagate phi, current,
    and an integral of phi independently of the GARNET discretization.
    The interval brackets only the fundamental eigenvalue for this case.
    """
    def end_current(k):
        state = np.array([1., 0., 0.])
        for layer in LAYERS:
            state = _transfer(k, layer, state)
        return state[1]
    k = brentq(end_current, .95, 1.05, xtol=1e-14)
    state = np.array([1., 0., 0.])
    edge_flux = [state[0]]
    edge_current = [state[1]]
    integrals = []
    min_flux = 1.
    for i, layer in enumerate(LAYERS):
        initial = state.copy()
        initial[2] = 0.
        for x in np.linspace(0, layer[0], 101):
            min_flux = min(min_flux, _transfer(k, layer, initial, x)[0])
        state = _transfer(k, layer, initial)
        integrals.append(state[2])
        if i % 2 == 1:
            edge_flux.append(state[0])
            edge_current.append(state[1])
    if min_flux <= 0:
        raise RuntimeError('selected eigenvalue is not a positive fundamental mode')
    integrals = np.asarray(integrals).reshape(3, 2)
    pairs = LAYERS.reshape(3, 2, 4)
    widths = pairs[:, :, 0].sum(axis=1)
    total = integrals.sum(axis=1)
    # Reaction-rate weighted cross sections; D is a stated choice, also
    # flux-weighted. D is not determined by reaction-rate preservation.
    avg = total / widths
    coeff = (pairs[:, :, 1:] * integrals[:, :, None]).sum(axis=1) / total[:, None]
    return dict(k=k, avg=avg, coeff=coeff, edge_flux=np.array(edge_flux),
                edge_current=np.array(edge_current), widths=widths)


def reference_factors(ref, n):
    """Construct factors consistent with the discrete homogeneous stencil.

    With k and the two boundary currents fixed from the heterogeneous
    reference, solve each homogeneous region's independent tridiagonal FV
    equations. Its face traces yield heterogeneous/homogeneous factors. These
    are reference factors for this exact case and resolution, not transferable
    isolated-assembly factors.
    """
    face_sides = np.ones((3, 2))
    region_flux = []
    for region in range(3):
        D, a, p = ref['coeff'][region]
        h = ref['widths'][region] / n
        lap = np.diag(np.full(n, 2.))
        if n > 1:
            lap += np.diag(np.full(n-1, -1.), 1) + np.diag(np.full(n-1, -1.), -1)
        lap[0, 0] -= 1
        lap[-1, -1] -= 1
        matrix = D / h**2 * lap + (a - p / ref['k']) * np.eye(n)
        rhs = np.zeros(n)
        rhs[0] += ref['edge_current'][region] / h
        rhs[-1] -= ref['edge_current'][region + 1] / h
        phi = solve(matrix, rhs, assume_a='sym')
        region_flux.append(phi)
        left_trace = phi[0] + ref['edge_current'][region] * h / (2 * D)
        right_trace = phi[-1] - ref['edge_current'][region+1] * h / (2 * D)
        if region > 0:
            face_sides[region, 0] = ref['edge_flux'][region] / left_trace
        if region < 2:
            face_sides[region, 1] = ref['edge_flux'][region+1] / right_trace
    if np.any(face_sides <= 0) or not np.all(np.isfinite(face_sides)):
        raise RuntimeError('non-positive reference factors')
    return face_sides, np.asarray(region_flux)


def run(n, enabled):
    if n < 1:
        raise ValueError('refinement must be a positive integer')
    ref = heterogeneous_reference()
    factors, expected_cells = reference_factors(ref, n)
    grid = Grid((3*n, 1, 1), (float(ref['widths'].sum()), 1., 1.))
    regions = np.repeat(np.arange(3), n).reshape(grid.shape)
    side = np.ones((1, 6, *grid.shape))
    for r in range(3):
        side[0, 0][regions == r] = factors[r, 0]
        side[0, 1][regions == r] = factors[r, 1]
    adf = assembly_face_factors(regions, side) if enabled else None
    materials = [Material([D], [a], [p]) for D, a, p in ref['coeff']]
    solver = DiffusionEigenSolver(grid, materials, regions, bc='reflective',
                                  device='cpu', adf=adf,
                                  linear_solver='gmres' if enabled else 'cg')
    result = solver.solve(tol_k=1e-12, tol_source=1e-11, inner_rtol_floor=1e-13)
    if not result.converged:
        raise RuntimeError(f'solve failed: {result}')
    flux = result.flux_numpy[0].ravel()
    flux *= ref['avg'].sum() / flux.reshape(3, n).mean(axis=1).sum()
    averages = flux.reshape(3, n).mean(axis=1)
    currents = solver.ops[0].face_currents(flux.reshape(grid.shape))[0].ravel()[np.array([n-1, 2*n-1])]
    rhs = solver.nu_sigma_f[0] * flux.reshape(grid.shape) / result.k_eff
    relative_residual = np.linalg.norm(solver.ops[0].apply(flux.reshape(grid.shape)) - rhs) / np.linalg.norm(rhs)
    return dict(refinement=n, reference_factors=enabled, k=result.k_eff,
                reference_k=ref['k'], delta_k_pcm=1e5 * (result.k_eff-ref['k']) / ref['k'],
                region_flux=averages.tolist(), reference_region_flux=ref['avg'].tolist(),
                region_flux_max_relative_error=float(np.max(np.abs(averages/ref['avg']-1))),
                currents=currents.tolist(), reference_currents=ref['edge_current'][1:-1].tolist(),
                current_max_abs_error=float(np.max(np.abs(currents-ref['edge_current'][1:-1]))),
                relative_eigen_residual=float(relative_residual),
                maximum_cell_flux_error_vs_reference_reconstruction=(float(np.max(np.abs(flux - expected_cells.ravel())))
                                                                    if enabled else None),
                factors=factors.tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--refinements', nargs='+', type=int, default=[1, 2, 4, 8])
    args = p.parse_args()
    print(json.dumps([run(n, enabled) for n in args.refinements for enabled in (False, True)], indent=2))


if __name__ == '__main__':
    main()
