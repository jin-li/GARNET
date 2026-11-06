"""Wielandt-shifted k-eigenvalue iteration: ``solve(wielandt_shift=...)``.

The shift changes the iteration, not the eigenproblem. Every test therefore
compares the shifted solve with the plain power iteration, both converged far
below the compared digits, and asks for the agreement the benchmark protocol
demands of equivalent configurations (|dk| <= 2e-8): forward and adjoint, and
for each angular family that shares the outer loop (diffusion, SP3, SPN, the
non-symmetric SDP3 block, triangular prisms, the cylindrical r-z metric).

The upscatter core is the case the shift was built for: one Gauss-Seidel
group sweep per outer leaves the thermal upscatter to the outer loop, whose
per-outer change then understates the error. The shifted outer solves the
whole multigroup system (upscatter included) each time.
"""

import numpy as np
import pytest

from garnet import CylindricalModel, Grid, Material
from garnet.benchmarks.iaea import build_iaea
from garnet.solver import (DiffusionEigenSolver, SDP3EigenSolver,
                           SP3EigenSolver, SP5EigenSolver)
from garnet.tri import TriDiffusionEigenSolver, TriGrid

TIGHT = dict(tol_k=1e-11, tol_source=1e-9, inner_rtol_floor=1e-12, max_outer=4000)
EQUIVALENT = 2e-8

# Three groups with thermal upscatter (sigma_s[from][to]; [2][1] is upscatter),
# a fissile core in a moderating reflector -- a miniature of the HP-MR set.
FUEL = Material(name="fuel", diffusion=[1.5, 0.9, 0.4],
                sigma_a=[0.009, 0.03, 0.10], nu_sigma_f=[0.006, 0.04, 0.16],
                chi=[0.95, 0.05, 0.0],
                sigma_s=[[0.0, 0.030, 0.002], [0.0, 0.0, 0.060], [0.0, 0.012, 0.0]])
MODERATOR = Material(name="moderator", diffusion=[1.3, 0.8, 0.3],
                     sigma_a=[0.001, 0.002, 0.010], nu_sigma_f=[0.0, 0.0, 0.0],
                     chi=[1.0, 0.0, 0.0],
                     sigma_s=[[0.0, 0.040, 0.001], [0.0, 0.0, 0.080], [0.0, 0.030, 0.0]])


def _upscatter_core():
    grid = Grid((12, 12, 8), (60.0, 60.0, 40.0))
    mmap = np.ones(grid.shape, dtype=int)
    mmap[2:10, 2:10, 1:7] = 0
    return grid, [FUEL, MODERATOR], mmap


def _shape(r):
    phi = r.flux_numpy
    return phi / np.linalg.norm(phi)


def _assert_same_eigenpair(plain, shifted, shape_tol=1e-6):
    assert plain.converged and shifted.converged
    assert shifted.multigroup_iterations > 0
    assert abs(shifted.k_eff - plain.k_eff) <= EQUIVALENT
    assert np.max(np.abs(_shape(shifted) - _shape(plain))) <= shape_tol


@pytest.fixture(scope="module")
def iaea():
    return build_iaea(1)


def _solver(cls, p, **kw):
    return cls(p.grid, p.materials, p.material_map, bc=p.bc, active=p.active,
               mask_bc=p.mask_bc, device="cpu", **kw)


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
@pytest.mark.parametrize("adjoint", [False, True])
def test_diffusion_eigenpair_unchanged(iaea, adjoint, mode):
    s = _solver(DiffusionEigenSolver, iaea)
    plain = s.solve(adjoint=adjoint, **TIGHT)
    shifted = s.solve(adjoint=adjoint, wielandt_shift=0.1, wielandt_mode=mode,
                      **TIGHT)
    _assert_same_eigenpair(plain, shifted)


@pytest.mark.parametrize("cls", [SP3EigenSolver, SP5EigenSolver, SDP3EigenSolver])
def test_block_solvers_eigenvalue_unchanged(iaea, cls):
    s = _solver(cls, iaea)
    plain = s.solve(**TIGHT)
    shifted = s.solve(wielandt_shift=0.1, **TIGHT)
    _assert_same_eigenpair(plain, shifted, shape_tol=1e-5)


def test_sdp3_adjoint_matches_forward(iaea):
    # The non-symmetric SDP3 block solves its adjoint on the transpose.
    s = _solver(SDP3EigenSolver, iaea)
    forward = s.solve(**TIGHT)
    adjoint = s.solve(adjoint=True, wielandt_shift=0.1, **TIGHT)
    assert adjoint.converged
    assert abs(adjoint.k_eff - forward.k_eff) <= EQUIVALENT


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_upscatter_core_converges_where_plain_stops_early(mode):
    grid, mats, mmap = _upscatter_core()
    s = DiffusionEigenSolver(grid, mats, mmap, bc="vacuum", device="cpu")
    reference = s.solve(**TIGHT)
    assert reference.converged
    shifted = s.solve(wielandt_shift=0.3, wielandt_mode=mode)   # default tolerances
    assert shifted.converged
    assert abs(shifted.k_eff - reference.k_eff) <= 1e-7
    assert shifted.outer_iterations < 30
    _assert_same_eigenpair(reference, s.solve(wielandt_shift=0.3, wielandt_mode=mode,
                                              **TIGHT))


@pytest.mark.parametrize("adjoint", [False, True])
def test_upscatter_core_on_triangular_prisms(adjoint):
    grid = TriGrid(shape=(8, 8, 2, 6), side=5.0, height=40.0)
    mmap = np.ones(grid.shape, dtype=int)
    mmap[2:6, 2:6, :, 1:5] = 0
    s = TriDiffusionEigenSolver(grid, [FUEL, MODERATOR], mmap, bc="vacuum",
                                device="cpu")
    plain = s.solve(adjoint=adjoint, **TIGHT)
    shifted = s.solve(adjoint=adjoint, wielandt_shift=0.3, **TIGHT)
    _assert_same_eigenpair(plain, shifted)


def test_cylindrical_model_passes_the_shift_through():
    # r-z grids weight the source by the cell metric (rhs_weight).
    mat = Material(diffusion=[1.3], sigma_a=[0.030], nu_sigma_f=[0.035])

    def steady(**kw):
        return (CylindricalModel(radius=80.0, height=160.0, cells=(10, 20))
                .fill(mat).set_boundary(r=("reflective", "zero-flux"), z="zero-flux")
                .steady(device="cpu", **TIGHT, **kw))

    assert abs(steady(wielandt_shift=0.2).k_eff - steady().k_eff) <= EQUIVALENT


def test_options_are_validated(iaea):
    s = _solver(DiffusionEigenSolver, iaea)
    for bad in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="wielandt_shift"):
            s.solve(wielandt_shift=bad)
    with pytest.raises(ValueError, match="wielandt_mode"):
        s.solve(wielandt_shift=0.1, wielandt_mode="rayleigh")
    with pytest.raises(ValueError, match="only with wielandt_shift"):
        s.solve(multigroup_kwargs={"restart": 10})
    with pytest.raises(ValueError, match="unknown multigroup_kwargs"):
        s.solve(wielandt_shift=0.1, multigroup_kwargs={"tolerance": 1e-3})
    with pytest.raises(ValueError, match="inner_rtol"):
        s.solve(wielandt_shift=0.1, multigroup_kwargs={"inner_rtol": 1.5})
