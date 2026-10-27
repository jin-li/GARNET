"""Whole-problem reference-DF verification against analytical diffusion."""
import numpy as np
import pytest

from garnet import DiffusionEigenSolver, Grid, Material
from examples.adf_slab_reference import LAYERS, heterogeneous_reference, run


def test_analytical_reference_matches_refined_heterogeneous_fv():
    reference = heterogeneous_reference()
    assert reference['k'] == pytest.approx(0.9964473442616343, abs=2e-12)
    errors = []
    for cells_per_cm in (2, 4, 8):
        n = 30 * cells_per_cm
        grid = Grid((n, 1, 1), (30., 1., 1.))
        centers = (np.arange(n) + .5) / cells_per_cm
        mmap = np.searchsorted(np.cumsum(LAYERS[:, 0]), centers).reshape(grid.shape)
        mats = [Material([d], [a], [p]) for _, d, a, p in LAYERS]
        result = DiffusionEigenSolver(grid, mats, mmap, bc='reflective', device='cpu').solve(
            tol_k=1e-11, tol_source=1e-10, inner_rtol_floor=1e-12)
        assert result.converged
        errors.append(abs(result.k_eff - reference['k']))
    assert 3.8 < errors[0] / errors[1] < 4.2
    assert 3.8 < errors[1] / errors[2] < 4.2
    assert errors[-1] < 5.4e-5


@pytest.mark.parametrize('n', [1, 2, 4, 8])
def test_reference_factors_preserve_reaction_rates_and_interface_currents(n):
    baseline, corrected = run(n, False), run(n, True)
    assert abs(baseline['delta_k_pcm']) > 1000
    assert abs(corrected['delta_k_pcm']) < 1e-5
    assert corrected['region_flux_max_relative_error'] < 1e-9
    assert corrected['current_max_abs_error'] < 1e-9
    assert corrected['maximum_cell_flux_error_vs_reference_reconstruction'] < 1e-9
    assert corrected['relative_eigen_residual'] < 1e-9


def test_two_group_analytical_reference_and_factor_limitations():
    from examples.adf_slab_2g_reference import (LAYERS as LAYERS_2G, MATERIALS,
                                             heterogeneous_reference as reference_2g,
                                             reference_factors, run as run_2g)
    ref = reference_2g()
    assert ref['k'] == pytest.approx(0.9437610266, abs=3e-10)
    # On these meshes the independently reconstructed homogeneous trace changes
    # sign, so no positive heterogeneous / homogeneous factor exists.
    for n in (1, 2):
        with pytest.raises(RuntimeError, match='non-positive'):
            reference_factors(ref, n)
    for n in (4, 8, 16):
        baseline, corrected = run_2g(n, False), run_2g(n, True)
        assert abs(baseline['delta_k_pcm']) > 400
        assert abs(corrected['delta_k_pcm']) < 1e-4
        assert corrected['region_flux_max_relative_error'] < 1e-8
        assert corrected['current_max_abs_error'] < 1e-9
        assert corrected['relative_eigen_residual'] < 1e-9
    # Independent GARNET heterogeneous calculation approaches the transfer
    # solution as the layer-aligned grid is refined.
    errors = []
    for per_cm in (2, 4, 8):
        grid = Grid((30 * per_cm, 1, 1), (30., 1., 1.))
        positions = (np.arange(grid.shape[0]) + .5) / per_cm
        boundaries = np.cumsum([width for width, _ in LAYERS_2G])
        layer = np.searchsorted(boundaries, positions)
        mmap = np.array([LAYERS_2G[i][1] for i in layer]).reshape(grid.shape)
        mats = [Material([d1, d2], [a1, a2], [p1, p2], [[0, sc], [0, 0]])
                for d1, d2, a1, a2, p1, p2, sc in MATERIALS]
        solution = DiffusionEigenSolver(grid, mats, mmap, bc='reflective', device='cpu').solve(
            tol_k=1e-11, tol_source=1e-10, inner_rtol_floor=1e-12)
        assert solution.converged
        errors.append(abs(solution.k_eff - ref['k']))
    assert 3.7 < errors[0] / errors[1] < 4.3
    assert 3.7 < errors[1] / errors[2] < 4.3
