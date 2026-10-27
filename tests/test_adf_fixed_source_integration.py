"""Cartesian ADF adjoints through the fork's coupled fixed-source solve."""

import numpy as np
import pytest
from scipy.linalg import eig

from garnet import DiffusionEigenSolver, Grid, Material
from garnet import solver as solver_module


def test_three_group_adf_forward_and_adjoint_match_dense_pencil(monkeypatch):
    grid = Grid((3, 1, 1), (9., 1., 1.))
    materials = [Material(
        diffusion=[1.2, .7, .3], sigma_a=[.02, .04, .08],
        nu_sigma_f=[.01, .04, .13], chi=[.8, .15, .05],
        sigma_s=[[0., .08, .02], [.03, 0., .07], [.02, .04, 0.]]),
        Material(diffusion=[.9, .5, .2], sigma_a=[.03, .05, .09],
        nu_sigma_f=[.02, .05, .11], chi=[.7, .2, .1],
        sigma_s=[[0., .06, .03], [.04, 0., .05], [.01, .06, 0.]])]
    regions = np.array([0, 1, 0]).reshape(grid.shape)
    factors = tuple((np.ones((3, *(n - (axis == i) for i, n in enumerate(grid.shape)))),
                     np.ones((3, *(n - (axis == i) for i, n in enumerate(grid.shape)))))
                    for axis in range(3))
    factors[0][0][..., 0, 0] = [[.7, 1.3], [1.4, .8], [.6, 1.2]]
    factors[0][1][..., 0, 0] = [[1.2, .8], [.9, 1.3], [1.5, .7]]
    solver = DiffusionEigenSolver(grid, materials, regions, bc='reflective',
        device='cpu', adf=factors, linear_solver='gmres', precond_degree=1)
    # Assemble the independent loss/production pencil by material identity;
    # no solver source or residual helper is used here.
    loss, production = np.zeros((9, 9)), np.zeros((9, 9))
    for g in range(3):
        rows = slice(3*g, 3*g+3)
        loss[rows, rows] = solver.ops[g].assemble().toarray()
        for source in range(3):
            cols = slice(3*source, 3*source+3)
            if source != g:
                loss[rows, cols] = -np.diag([
                    materials[i].sigma_s[source, g] for i in regions.ravel()])
            production[rows, cols] = np.diag([
                materials[i].chi[g] * materials[i].nu_sigma_f[source]
                for i in regions.ravel()])
    eigenvalues = eig(production, loss, right=False)
    expected = eigenvalues[np.argmax(eigenvalues.real)].real
    coupled_calls = []
    original = solver_module.fgmres
    def track_coupled(*args, **kwargs):
        if args[1].shape == (3, *grid.shape):
            coupled_calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(solver_module, 'fgmres', track_coupled)
    for adjoint in (False, True):
        coupled_calls.clear()
        result = solver.solve(adjoint=adjoint, tol_k=1e-10, tol_source=1e-9,
                              tol_balance=1e-9, tol_fixed_source=1e-10)
        assert result.converged
        assert coupled_calls, 'exercise the coupled Krylov fixed-source path'
        assert result.k_eff == pytest.approx(expected, rel=2e-9)
        phi = result.flux_numpy.ravel()
        a, f = (loss.T, production.T) if adjoint else (loss, production)
        residual = np.linalg.norm(a @ phi - f @ phi / result.k_eff) / np.linalg.norm(a @ phi)
        assert residual < 2e-9
        assert result.balance_error < 1e-9
