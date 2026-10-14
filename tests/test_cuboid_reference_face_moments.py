"""Prescribed affine CURRENT sign moments; no Serpent flux moments or fitting."""
import numpy as np
import pytest

from ndgpu import Material, CuboidReferenceError
from ndgpu import coupled_neumann_cuboid_reference as lu_reference
from ndgpu import coupled_neumann_cuboid_reference_modal as reference
from ndgpu.mesh import MeshGrid, MeshGroupOperator
from ndgpu.nodal_mesh import build_rectilinear_mesh_3d

TANGENTS = ((1, 2), (0, 2), (1, 2), (0, 2), (0, 1), (0, 1))


def case():
    material = Material([1., .6], [.2, .3], [.015, .01], [[0, .04], [.015, 0]], [.8, .2])
    widths = np.array([1.7, 2.3, .9])
    currents = np.column_stack((1+.1*np.arange(6), .7+.08*np.arange(6)))
    moments = np.zeros((6, 2, 2))
    moments[:, 0] = currents*np.array([.03, -.02, .015, -.025, .02, -.01])[:, None]
    moments[:, 1] = currents*np.array([-.02, .025, -.03, .02, -.015, .025])[:, None]
    return material, widths, currents, moments


def profile(side, point, widths, currents, moments):
    # Independent physical formula, expressed in native signed coordinates.
    a, b = TANGENTS[side]
    area = widths[a]*widths[b]
    return (currents[side]/area+4*moments[side, 0]*point[a]/(area*widths[a])+
            4*moments[side, 1]*point[b]/(area*widths[b]))


def test_continuous_piecewise_sign_integrals_are_native_totals_and_moments():
    _, widths, currents, moments = case()
    q, w = np.polynomial.legendre.leggauss(2)
    recovered = np.zeros_like(moments)
    totals = np.zeros_like(currents)
    for side, (a, b) in enumerate(TANGENTS):
        # Split both signed coordinates at zero: Gaussian integration is exact
        # on each polynomial piece, independently of the discrete solver grid.
        for sa in (-1, 1):
            for sb in (-1, 1):
                for qa, wa in zip(q, w):
                    for qb, wb in zip(q, w):
                        point = np.zeros(3)
                        point[a] = sa*widths[a]/4+qa*widths[a]/4
                        point[b] = sb*widths[b]/4+qb*widths[b]/4
                        integral = profile(side, point, widths, currents, moments)*wa*wb*widths[a]*widths[b]/16
                        totals[side] += integral
                        recovered[side, 0] += sa*integral
                        recovered[side, 1] += sb*integral
    np.testing.assert_allclose(totals, currents, rtol=5e-16, atol=2e-16)
    np.testing.assert_allclose(recovered, moments, rtol=2e-14, atol=2e-16)


def test_mixed_moments_match_independent_dense_mesh_operator_and_affine_boundary_integrals():
    material, widths, currents, moments = case()
    k, counts = 1.1, (2, 4, 2)
    snapshot = moments.copy()
    actual = reference(material, k, currents, widths, subdivisions=counts,
                       net_inward_current_sign_moments=moments)
    uniform = reference(material, k, currents, widths, subdivisions=counts)
    edges = [np.linspace(0, width, n+1) for width, n in zip(widths, counts)]
    mesh, topology = build_rectilinear_mesh_3d(*edges, active_mask=np.ones(tuple(reversed(counts)), bool),
                                               tags=np.zeros(tuple(reversed(counts)), int))
    grid = MeshGrid(mesh, 'reflective')
    ops = [MeshGroupOperator(np, grid, np.full(grid.shape, material.diffusion[g]),
                            np.full(grid.shape, material.removal[g])) for g in range(2)]
    scatter = material.sigma_s.copy()
    np.fill_diagonal(scatter, 0)
    coupling = scatter.T+np.outer(material.chi, material.nu_sigma_f)/k
    n = mesh.n_cells
    dense = np.empty((2*n, 2*n))
    for column in range(2*n):
        basis = np.eye(2*n)[:, column].reshape(2, n)
        dense[:, column] = (np.array([ops[g].apply(basis[g].reshape(grid.shape)).ravel() for g in range(2)])-
                            mesh.area*(coupling@basis)).ravel()
    rhs = np.zeros((2, n))
    expected_density = []
    q, weights = np.polynomial.legendre.leggauss(2)
    for (owner, area, distance), side in zip(mesh.bfaces, topology['boundary_sides']):
        a, b = TANGENTS[side]
        center = mesh.centroid[owner]-widths/2
        average = np.zeros(2)
        for qa, wa in zip(q, weights):
            for qb, wb in zip(q, weights):
                point = center.copy()
                point[a] += qa*topology['widths'][owner, a]/2
                point[b] += qb*topology['widths'][owner, b]/2
                average += profile(side, point, widths, currents, moments)*wa*wb/4
        rhs[:, owner] += area*average
        expected_density.append(average)
    expected = np.linalg.solve(dense, rhs.ravel()).reshape(2, n)
    np.testing.assert_allclose(actual.cell_mean_flux, expected, rtol=3e-12, atol=1e-13)
    np.testing.assert_allclose(actual.boundary_profiles.inward_current_density, expected_density, rtol=2e-15, atol=2e-16)
    assert np.max(abs(actual.cell_mean_flux-uniform.cell_mean_flux)) > 1e-4
    np.testing.assert_allclose(actual.volume_mean_flux, uniform.volume_mean_flux, rtol=2e-13)
    np.testing.assert_array_equal(moments, snapshot)
    diagnostics = actual.diagnostics
    assert diagnostics['boundary_current_normalization_passed'] is True
    assert diagnostics['boundary_current_normalization_relative_error'] < 2e-15
    np.testing.assert_allclose(diagnostics['recovered_net_inward_currents'], currents, rtol=2e-15)
    np.testing.assert_allclose(diagnostics['recovered_net_inward_current_sign_moments'], moments, rtol=2e-13, atol=2e-16)
    assert diagnostics['normwise_backward_error'] < 1e-14
    assert diagnostics['integrated_group_balance_max_relative'] < 1e-13
    # Recover currents from actual FV cell/boundary traces as a separate check.
    boundary = actual.boundary_profiles
    density_from_trace = ((boundary.flux-actual.cell_mean_flux[:, boundary.owners].T)*material.diffusion/
                          boundary.distances[:, None])
    for side, (a, b) in enumerate(TANGENTS):
        mask = boundary.sides == side
        totals = (density_from_trace[mask]*boundary.areas[mask, None]).sum(axis=0)
        np.testing.assert_allclose(totals, currents[side], rtol=2e-13, atol=2e-14)
        for tangent, axis in enumerate((a, b)):
            signs = np.sign(boundary.centers[mask, axis]-widths[axis]/2)
            recovered = (density_from_trace[mask]*boundary.areas[mask, None]*signs[:, None]).sum(axis=0)
            np.testing.assert_allclose(recovered, moments[side, tangent], rtol=2e-11, atol=2e-14)


def test_zero_moments_are_bitwise_uniform_on_odd_unequal_grids():
    material, widths, currents, _ = case()
    absent = reference(material, 1.1, currents, widths, subdivisions=(3, 5, 1))
    zero = reference(material, 1.1, currents, widths, subdivisions=(3, 5, 1),
                     net_inward_current_sign_moments=np.zeros((6, 2, 2)))
    for name in ('cell_mean_flux', 'face_mean_flux', 'volume_mean_flux'):
        np.testing.assert_array_equal(getattr(absent, name), getattr(zero, name))
    for name in ('flux', 'inward_current_density', 'owners', 'sides'):
        np.testing.assert_array_equal(getattr(absent.boundary_profiles, name), getattr(zero.boundary_profiles, name))
    old_lu = lu_reference(material, 1.1, currents, widths, subdivisions=(3, 5, 1))
    np.testing.assert_allclose(zero.cell_mean_flux, old_lu.cell_mean_flux, rtol=2e-12)
    assert zero.diagnostics['boundary_current_normalization_passed'] is True


def test_zero_moments_preserve_negative_boundary_classification():
    material, currents = Material([1], [1], [0]), np.array([-5, 0, 6, 0, 0, 0])[:, None]
    for function, options in ((lu_reference, {}), (reference, {}),
                              (reference, dict(net_inward_current_sign_moments=np.zeros((6, 2, 1))))):
        with pytest.raises(CuboidReferenceError, match='no clipping') as caught:
            function(material, 1, currents, [1, 1, 1], subdivisions=1, **options)
        assert caught.value.diagnostics['minimum_cell_mean_flux'] == pytest.approx(1)
        assert caught.value.diagnostics['minimum_boundary_trace_flux'] == pytest.approx(-1.5)


def test_only_nonzero_moment_tangents_require_even_resolved_counts():
    material, widths, currents, _ = case()
    moments = np.zeros((6, 2, 2))
    moments[0, 0] = [.01, -.01]  # W,y; x3 and z1 are allowed.
    moments[5, 1] = [.01, .01]  # top,y; normal z1 remains allowed.
    result = reference(material, 1.1, currents, widths, subdivisions=(3, 4, 1),
                       net_inward_current_sign_moments=moments)
    assert result.diagnostics['boundary_current_normalization_passed'] is True
    for side, tangent in ((0, 1), (1, 0), (4, 0)):
        bad = moments.copy()
        bad[side, tangent, 0] = 1e-20  # exact nonzero; no resolution/size floor
        with pytest.raises(ValueError, match='even tangential'):
            reference(material, 1.1, currents, widths, subdivisions=(3, 4, 1),
                      net_inward_current_sign_moments=bad)


@pytest.mark.parametrize('moments', [np.ones((6, 2)), np.ones((6, 3, 2)), np.ones((6, 2, 1)),
                                   np.full((6, 2, 2), np.nan), np.full((6, 2, 2), np.inf)])
def test_invalid_current_sign_moments_rejected(moments):
    material, widths, currents, _ = case()
    with pytest.raises(ValueError, match='net_inward_current_sign_moments'):
        reference(material, 1.1, currents, widths, subdivisions=4,
                  net_inward_current_sign_moments=moments)
