"""Independent radial forcing oracle for coupled SP3 and SDP1 equations.

Circular exact fields are compared with material-wise analytic triangle
averages on inscribed polygon meshes. Geometry defects are O(dtheta**2).
The native boundary tables are exercised, but their transport derivation is
not established by this manufactured mathematical diffusion problem.
"""
import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from ndgpu.corrected_mesh_moments import CorrectedMeshMomentOperator
from ndgpu.mesh import MeshGrid
from ndgpu.spn import _SPN_C, _SPN_G, _SDPN_C, _SDPN_G
from test_corrected_mesh_radial_mms import RADII, polar_mesh, radial_cell_averages


def mathematical_coefficients(family):
    """Literal U-form PDE: -div(D grad U) + (rem*C0 + st*Ct) U = q.

    These explicit rational coefficients define the oracle independently of
    operator internals and the constructor coefficient/boundary tables.
    """
    if family == "sp3":
        return (np.array([[1., -2/3], [-2/3, 4/9]]),
                np.array([[0., 0.], [0., 5/9]]),
                np.array([[1/2, -1/8], [-1/8, 7/24]]))
    return (np.array([[1., -2/3], [-6/7, 4/7]]),
            np.array([[0., 0.], [0., 5/7]]),
            np.array([[1/2, -1/8], [-2/7, 23/42]]))


@pytest.mark.parametrize("family", ["sp3", "sdp1"])
def test_curved_radial_coupled_moment_mms(family):
    D0 = np.array([.7, 1800., .3])
    sigma_t = 1 / (3 * D0)
    diffusion = np.stack((D0, 1 / (7 * sigma_t)))
    removal = .13
    forcing = np.array([1., -.35])
    c0, ct, G = mathematical_coefficients(family)
    # At r=R: outward current = R*f/2 = G*b. Both off-diagonal
    # boundary entries are active, and the exact moment traces are nonzero.
    b = np.linalg.solve(G, RADII[-1] * forcing / 2)
    np.testing.assert_allclose(G @ b, RADII[-1] * forcing / 2, atol=1e-15)
    coeffs, native_g = ((_SPN_C, _SPN_G) if family == "sp3"
                        else (_SDPN_C, _SDPN_G))
    errors, moment_errors = [], []
    for refinement in (1, 2, 4):
        mesh, coords, angles = polar_mesh(refinement)
        tags = mesh.cell_tag
        n = mesh.n_cells
        grid = MeshGrid(mesh, "vacuum")
        op = CorrectedMeshMomentOperator(
            np, grid, D0[tags], sigma_t[tags], np.full(n, removal), 1,
            bc="vacuum", coeffs=coeffs, boundary_g=native_g[1])
        # U_m = b_m + f_m integral_r^R s/(2 D_m(s)) ds. The analytic
        # triangle averages and source never call op.apply or read op reaction.
        exact = b[:, None] + forcing[:, None] * np.stack([
            radial_cell_averages(mesh, coords, d) for d in diffusion])
        reaction = removal*c0[:, :, None] + ct[:, :, None]*sigma_t[tags]
        rhs = (forcing[:, None]
               + np.einsum("mki,ki->mi", reaction, exact)) * mesh.area
        solution = spsolve(op.host_matrix.tocsc(), rhs.ravel()).reshape(2, n)
        per_moment = np.sqrt(np.sum(mesh.area * (solution - exact)**2, axis=1))
        error = np.linalg.norm(per_moment)
        errors.append(error)
        moment_errors.append(per_moment)
        np.testing.assert_allclose(op.apply(solution).reshape(2, n), rhs,
                                   rtol=2e-8, atol=2e-10)
        interior, boundary = op.face_currents(solution)
        balance = np.einsum("mki,ki->mi", reaction, solution) * mesh.area
        for k, (i, j, _, _) in enumerate(mesh.faces):
            balance[:, i] += interior[:, k]
            balance[:, j] -= interior[:, k]
        for k, (i, _, _) in enumerate(mesh.bfaces):
            balance[:, i] += boundary[:, k]
        np.testing.assert_allclose(balance, rhs, rtol=2e-8, atol=2e-10)
        # Each moment's interior contributions cancel, leaving its reaction
        # and boundary current equal to that moment's integrated source.
        expected_outflow = np.sum(rhs - np.einsum(
            "mki,ki->mi", reaction, solution)*mesh.area, axis=1)
        np.testing.assert_allclose(boundary.sum(axis=1), expected_outflow,
                                   rtol=2e-9, atol=2e-10)
        # The circular exact field does not satisfy the homogeneous law on
        # chords. At a chord midpoint its Robin defect is:
        # G*(b + f*R^2*sin(a)^2/(4D_outer)) - f*R*cos(a)/2.
        half_angle = np.pi / angles
        trace_midpoint = (b + forcing * RADII[-1]**2
                          * np.sin(half_angle)**2 / (4*diffusion[:, -1]))
        robin_defect = G @ trace_midpoint - forcing*RADII[-1]*np.cos(half_angle)/2
        print(f"{family} unknowns={2*n} weighted_L2={error:.9e} "
              f"moment_L2={per_moment.tolist()} "
              f"boundary_Robin_defect={robin_defect.tolist()} "
              f"outflow={boundary.sum(axis=1).tolist()}")
    ratios = np.asarray(errors[:-1]) / errors[1:]
    moment_ratios = np.asarray(moment_errors[:-1]) / moment_errors[1:]
    print(f"{family} error_ratios={ratios.tolist()} orders={np.log2(ratios).tolist()} "
          f"moment_orders={np.log2(moment_ratios).tolist()}")
    assert np.all(ratios > 3.), f"coupled radial MMS failed: {errors}, {ratios}"
    assert np.all(moment_ratios > 3.), f"moment refinement failed: {moment_errors}"
