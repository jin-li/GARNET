"""Validation-only setup and optional frozen-package equivalence checks.

Set GARNET_FROZEN_PACKAGE to the prechange directory containing ``garnet`` to
run the independent snapshot comparison as part of a local validation audit.
"""
from dataclasses import replace
import importlib
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from garnet.corrected_mesh import CorrectedMeshGroupOperator, _validate_corrected_mesh
from garnet.corrected_mesh_moments import CorrectedMeshMomentOperator
from garnet.mesh import MeshGrid
from garnet.spn import _SPN_C, _SPN_G, _SDPN_C, _SDPN_G
from test_corrected_mesh_moments import mesh_grid


@pytest.fixture(scope="module")
def frozen():
    root = os.environ.get("GARNET_FROZEN_PACKAGE")
    if root is None:
        return None
    package = types.ModuleType("_frozen_corrected_validation")
    package.__path__ = [str(Path(root) / "garnet")]
    sys.modules[package.__name__] = package
    return tuple(importlib.import_module(package.__name__ + "." + module)
                 for module in ("corrected_mesh", "corrected_mesh_moments", "spn"))


def assert_operator_equal(actual, expected, state):
    np.testing.assert_array_equal(actual.host_matrix.toarray(), expected.host_matrix.toarray())
    np.testing.assert_array_equal(actual.apply(state), expected.apply(state))
    for a, b in zip(actual.face_currents(state), expected.face_currents(state)):
        np.testing.assert_array_equal(a, b)
    for attr in ("diag", "inv_diag", "rhs_weight", "tpfa_inv_diag"):
        np.testing.assert_array_equal(getattr(actual, attr), getattr(expected, attr))
    np.testing.assert_array_equal(actual.tpfa_apply(state), expected.tpfa_apply(state))


@pytest.mark.parametrize("bc", ["reflective", "vacuum"])
@pytest.mark.parametrize("family", ["scalar", "sp3", "sdp1"])
def test_frozen_distinct_material_operators(frozen, bc, family):
    if frozen is None:
        pytest.skip("set GARNET_FROZEN_PACKAGE for frozen prechange comparison")
    scalar, moments, spn = frozen
    mesh = mesh_grid(3)
    grid = MeshGrid(mesh, bc)
    n = grid.n_cells
    # Distinct group coefficients and signed removal guard against accidental
    # cross-group material reuse and a newly imposed positivity restriction.
    operators = []
    for group in (0, 1):
        D = np.linspace(.4 + group*.3, 1.3 + group*.7, n).reshape(grid.shape)
        st = np.linspace(1.1 + group*.1, 2.1 + group*.4, n).reshape(grid.shape)
        rem = np.linspace(-.03, .17 + group*.2, n).reshape(grid.shape)
        if family == "scalar":
            actual = CorrectedMeshGroupOperator(np, grid, D, rem, bc=bc)
            expected = scalar.CorrectedMeshGroupOperator(np, grid, D, rem, bc=bc)
            shape = grid.shape
        else:
            c, g = (_SPN_C, _SPN_G) if family == "sp3" else (_SDPN_C, _SDPN_G)
            old_c, old_g = (spn._SPN_C, spn._SPN_G) if family == "sp3" else (spn._SDPN_C, spn._SDPN_G)
            kw = dict(bc=bc, coeffs=c, boundary_g=g[1], theta=np.linspace(.01, .04, n).reshape(grid.shape))
            actual = CorrectedMeshMomentOperator(np, grid, D, st, rem, 1, **kw)
            kw.update(coeffs=old_c, boundary_g=old_g[1])
            expected = moments.CorrectedMeshMomentOperator(np, grid, D, st, rem, 1, **kw)
            shape = (2,) + grid.shape
        state = np.random.default_rng(83 + group).normal(size=shape)
        assert_operator_equal(actual, expected, state)
        operators.append(actual)
    assert not np.array_equal(operators[0].host_matrix.toarray(), operators[1].host_matrix.toarray())


@pytest.mark.parametrize("invalid", ["D_size", "D_zero", "D_nan", "rem_size", "rem_nan",
                                    "centroid", "face_centers", "bface_normals", "unit", "distance",
                                    "face_df", "boundary_df"])
def test_lightweight_validation_retains_errors(frozen, invalid):
    mesh = mesh_grid(2)
    D, rem, kw = np.ones(4), np.zeros(4), {}
    if invalid == "D_size": D = D[:-1]
    elif invalid == "D_zero": D[0] = 0
    elif invalid == "D_nan": D[0] = np.nan
    elif invalid == "rem_size": rem = rem[:-1]
    elif invalid == "rem_nan": rem[0] = np.nan
    elif invalid == "centroid": mesh = replace(mesh, centroid=mesh.centroid[:, :1])
    elif invalid == "face_centers": mesh = replace(mesh, face_centers=None)
    elif invalid == "bface_normals": mesh = replace(mesh, bface_normals=np.full_like(mesh.bface_normals, np.nan))
    elif invalid == "unit": mesh = replace(mesh, face_normals=mesh.face_normals*2)
    elif invalid == "distance": mesh = replace(mesh, face_centers=mesh.face_centers-10*mesh.face_normals)
    elif invalid == "face_df": kw["face_df"] = np.full((len(mesh.faces), 2), 1.1)
    elif invalid == "boundary_df": kw["boundary_df"] = np.full(len(mesh.bfaces), 1.1)
    grid = MeshGrid(mesh, "reflective")
    expected = {
        "D_size": "diffusion/removal require one finite value per cell; diffusion must be positive",
        "D_zero": "diffusion/removal require one finite value per cell; diffusion must be positive",
        "D_nan": "diffusion/removal require one finite value per cell; diffusion must be positive",
        "rem_size": "diffusion/removal require one finite value per cell; diffusion must be positive",
        "rem_nan": "diffusion/removal require one finite value per cell; diffusion must be positive",
        "centroid": "corrected mesh requires finite 2D or 3D cell centroids",
        "face_centers": "corrected mesh requires finite face_centers with shape (4, 2)",
        "bface_normals": "corrected mesh requires finite bface_normals with shape (8, 2)",
        "unit": "corrected mesh requires unit normals",
        "distance": "corrected mesh requires positive signed normal distances",
        "face_df": "corrected mesh discontinuity factors are not implemented",
        "boundary_df": "corrected mesh discontinuity factors are not implemented",
    }[invalid]
    with pytest.raises(ValueError) as current:
        _validate_corrected_mesh(grid, D, rem, **kw)
    assert str(current.value) == expected
    if frozen is not None:
        scalar, _, _ = frozen
        with pytest.raises(ValueError) as previous:
            scalar.CorrectedMeshGroupOperator(np, grid, D, rem, reconstruction_policy="supplied",
                                              gradient_stencils=[{} for _ in range(4)], **kw)
        assert str(current.value) == str(previous.value)


def test_moment_setup_avoids_discarded_scalar_assembly(monkeypatch):
    import garnet.corrected_mesh as scalar

    def unexpected(*args, **kwargs):
        pytest.fail("moment validation must not assemble a discarded scalar operator")

    monkeypatch.setattr(scalar.CorrectedMeshGroupOperator, "__init__", unexpected)
    grid = MeshGrid(mesh_grid(2), "reflective")
    op = CorrectedMeshMomentOperator(np, grid, np.ones(4), np.ones(4), np.zeros(4),
                                    1, bc="reflective")
    assert op.host_matrix.shape == (8, 8)
