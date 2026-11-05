"""Interactions between deferred correction and corrected-trace mesh APIs."""

import numpy as np
import pytest

from garnet.materials import Material
from garnet.mesh import UnstructuredDiffusionSolver, assemble_mesh, assemble_mesh_3d


MAT = Material(name="m", diffusion=[1.0], sigma_a=[0.1], nu_sigma_f=[0.3])


def skew_mesh():
    return assemble_mesh([(0., 0.), (3., 0.), (3., 2.), (0., 2.)],
                         [(0, 1, 2), (0, 2, 3)], [0, 0])


def solver(mesh, **kwargs):
    return UnstructuredDiffusionSolver(mesh, [MAT], np.zeros(mesh.n_cells, int),
                                       device="cpu", **kwargs)


@pytest.mark.parametrize("alpha", [.5, np.inf])
def test_explicit_unit_factors_preserve_deferred_correction(alpha):
    mesh = skew_mesh()
    ordinary = solver(mesh, alpha_boundary=alpha)
    unit = solver(mesh, alpha_boundary=alpha, face_df=np.ones((1, len(mesh.faces), 2)),
                  boundary_df=np.ones((1, len(mesh.bfaces))))
    assert ordinary._noc is not None and unit._noc is not None
    phi = np.array([0.7, 1.2])
    np.testing.assert_allclose(unit.ops[0].apply(phi), ordinary.ops[0].apply(phi))
    np.testing.assert_allclose(unit._correction(0, phi), ordinary._correction(0, phi))
    assert unit.solve(tol_k=1e-10).k_eff == pytest.approx(
        ordinary.solve(tol_k=1e-10).k_eff, abs=1e-10)


@pytest.mark.parametrize("factors", ["unequal_face", "equal_face", "boundary"])
def test_skew_nonunit_factors_require_explicit_tpfa_opt_out(factors):
    mesh = skew_mesh()
    kwargs = ({"boundary_df": np.full((1, len(mesh.bfaces)), 1.2)}
              if factors == "boundary" else
              {"face_df": np.array([[[1.2, .8 if factors == "unequal_face" else 1.2]]])})
    with pytest.raises(ValueError, match="nonorthogonal correction.*discontinuity factors"):
        solver(mesh, **kwargs)
    approximate = solver(mesh, nonorthogonal_correction=False, **kwargs)
    assert approximate._noc is None
    # The explicit opt-out still retains the corrected-trace conservative
    # interior current, even with unequal factors on the two sides.
    if factors != "boundary":
        reflective = solver(mesh, alpha_boundary=0, nonorthogonal_correction=False, **kwargs)
        phi = np.array([.7, 1.2])
        leakage = reflective.ops[0].apply(phi) - mesh.area * .1 * phi
        assert leakage.sum() == pytest.approx(0, abs=1e-14)


def test_normal_geometry_enables_3d_correction_and_legacy_mode_remains_explicit():
    coords = [(0., 0., 0.), (2., 0., 0.), (0., 2., 0.),
              (.2, .2, 1.), (1., 1., -2.)]
    cells = [(0, 1, 2, 3), (0, 2, 1, 4)]
    normal = assemble_mesh_3d(coords, cells, [0, 0], normal_distances=True)
    np.testing.assert_allclose(normal.face_distances, [[.25, .5]])
    assert solver(normal)._noc is not None
    legacy = assemble_mesh_3d(coords, cells, [0, 0], normal_distances=False)
    assert legacy.face_distances is None and legacy.face_normal is None
    assert solver(legacy)._noc is None
    assert legacy.faces == normal.faces
