"""Small analytic/manufactured CPU tests for fixed-k coupled cuboid references."""
import copy

import numpy as np
import pytest

from ndgpu import (CoupledCuboidReference, CuboidBoundaryProfiles, CuboidReferenceError,
                   Material, coupled_neumann_cuboid_reference)
from ndgpu.serpent_adf import cuboid_flat_source_reconstruction


def material_mode(groups=2):
    """Manufacture a positive mode M*v=alpha²*diag(D)*v with genuine coupling."""
    alpha = 0.4
    diffusion = np.linspace(1.2, 0.8, groups)
    vector = np.linspace(1.0, 0.7, groups)
    scatter = np.full((groups, groups), 0.03)
    np.fill_diagonal(scatter, 0)
    if groups > 1:
        scatter[0, 1] = 0.05
    chi = np.arange(groups, 0, -1, dtype=float)
    chi /= chi.sum()
    production = np.linspace(0.025, 0.015, groups)
    k = 1.1
    coupling = scatter.T+np.outer(chi, production)/k
    removal = (coupling@vector)/vector+alpha**2*diffusion
    absorption = removal-scatter.sum(axis=1)
    material = Material(diffusion, absorption, production, scatter, chi)
    np.testing.assert_allclose((np.diag(material.removal)-coupling)@vector,
                               alpha**2*diffusion*vector, rtol=1e-13, atol=1e-15)
    return material, k, alpha, vector


def cosh_inputs(material, alpha, vector, widths):
    currents = np.zeros((6, material.n_groups))
    currents[0] = currents[2] = (material.diffusion*vector*alpha*np.sinh(alpha*widths[0]/2)*widths[1]*widths[2])
    mean = vector*np.sinh(alpha*widths[0]/2)/(alpha*widths[0]/2)
    return currents, mean


def exact_cell_means(result, alpha, vector, widths):
    h = widths[0]/result.subdivisions[0]
    center = result.cell_centers[:, 0]-widths[0]/2
    shape = (np.sinh(alpha*(center+h/2))-np.sinh(alpha*(center-h/2)))/(alpha*h)
    return vector[:, None]*shape


def check_boundary_currents(result, material, supplied):
    boundary = result.boundary_profiles
    recovered_density = ((boundary.flux-result.cell_mean_flux[:, boundary.owners].T)*material.diffusion[None, :]/
                         boundary.distances[:, None])
    recovered = np.array([(recovered_density[boundary.sides == side]*boundary.areas[boundary.sides == side, None]).sum(axis=0)
                          for side in range(6)])
    np.testing.assert_allclose(recovered, supplied, rtol=2e-12, atol=2e-14)


def test_one_group_cosh_refines_to_scalar_zero_uniform_source_limit():
    material = Material([1.2], [0.3], [0.0])
    alpha = np.sqrt(material.removal[0]/material.diffusion[0])
    widths, vector = np.array([3.0, 1.2, 0.7]), np.array([1.0])
    currents, mean = cosh_inputs(material, alpha, vector, widths)
    scalar = cuboid_flat_source_reconstruction(mean, material.diffusion, material.removal, currents, widths)
    np.testing.assert_allclose(scalar.uniform_source, 0, atol=3e-16)
    np.testing.assert_allclose(scalar.face_flux[[0, 2], 0], np.cosh(alpha*widths[0]/2), rtol=1e-13)
    cell_errors, face_errors = [], []
    for n in (8, 16, 32):
        result = coupled_neumann_cuboid_reference(material, 1.0, currents, widths, subdivisions=(n, 1, 1),
                                                  native_mean_flux=mean)
        np.testing.assert_allclose(result.volume_mean_flux, mean, rtol=1e-12)
        cell_errors.append(float(np.max(abs(result.cell_mean_flux-exact_cell_means(result, alpha, vector, widths)))))
        face_errors.append(float(np.max(abs(result.face_mean_flux-scalar.face_flux))))
        check_boundary_currents(result, material, currents)
        assert result.diagnostics['normwise_backward_error'] < 1e-14
        assert result.diagnostics['integrated_group_balance_max_relative'] < 1e-13
        assert result.diagnostics['independent_mesh_operator_action_relative_error'] < 1e-13
        assert result.discrete_positive is True
        assert result.continuous_positivity_certified is False
        assert isinstance(result, CoupledCuboidReference)
        assert isinstance(result.boundary_profiles, CuboidBoundaryProfiles)
    # Three resolved analytic levels test physical convergence separately from LU residual.
    assert all(3.0 < cell_errors[i]/cell_errors[i+1] < 5.0 for i in range(2))
    assert all(3.0 < face_errors[i]/face_errors[i+1] < 5.0 for i in range(2))
    assert face_errors[-1] > 100*result.diagnostics['normwise_backward_error']


def test_coupled_two_group_manufactured_positive_mode_and_face_convergence():
    material, k, alpha, vector = material_mode()
    widths = np.array([3.0, 0.9, 1.1])
    currents, mean = cosh_inputs(material, alpha, vector, widths)
    face_target = np.broadcast_to(mean, (6, 2)).copy()
    face_target[[0, 2]] = vector*np.cosh(alpha*widths[0]/2)
    cell_errors, face_errors = [], []
    for n in (8, 16, 32):
        result = coupled_neumann_cuboid_reference(material, k, currents, widths, subdivisions=(n, 2, 1))
        np.testing.assert_allclose(result.volume_mean_flux, mean, rtol=2e-12)
        # The chosen positive coupled mode has the same spatial shape in both groups.
        np.testing.assert_allclose(result.cell_mean_flux[0]/vector[0], result.cell_mean_flux[1]/vector[1], rtol=2e-12)
        cell_errors.append(float(np.max(abs(result.cell_mean_flux-exact_cell_means(result, alpha, vector, widths)))))
        face_errors.append(float(np.max(abs(result.face_mean_flux-face_target))))
        check_boundary_currents(result, material, currents)
        assert result.diagnostics['normwise_backward_error'] < 1e-14
        assert result.diagnostics['integrated_group_balance_max_relative'] < 1e-12
    assert all(3.0 < cell_errors[i]/cell_errors[i+1] < 5.0 for i in range(2))
    assert all(3.0 < face_errors[i]/face_errors[i+1] < 5.0 for i in range(2))


def test_native_mean_is_diagnostic_only_and_inputs_unchanged():
    material, k, alpha, vector = material_mode()
    widths = np.array([2.0, 1.0, 1.5])
    currents, mean = cosh_inputs(material, alpha, vector, widths)
    snapshots = copy.deepcopy((material, currents, widths))
    plain = coupled_neumann_cuboid_reference(material, k, currents, widths, subdivisions=3)
    diagnostic = coupled_neumann_cuboid_reference(material, k, currents, widths, subdivisions=3,
                                                   native_mean_flux=1.2*mean)
    np.testing.assert_array_equal(plain.cell_mean_flux, diagnostic.cell_mean_flux)
    np.testing.assert_array_equal(plain.face_mean_flux, diagnostic.face_mean_flux)
    assert plain.native_mean_compatibility is None
    info = diagnostic.native_mean_compatibility
    assert info['native_mean_imposed'] is False
    np.testing.assert_allclose(info['computed_relative_difference'], -1/6, atol=2e-13)
    assert np.linalg.norm(info['compatibility_defect_density']) > 0
    for key in ('diffusion', 'sigma_a', 'nu_sigma_f', 'sigma_s', 'chi'):
        np.testing.assert_array_equal(getattr(material, key), getattr(snapshots[0], key))
    np.testing.assert_array_equal(currents, snapshots[1])
    np.testing.assert_array_equal(widths, snapshots[2])


def test_all_spatial_mode_condition_matches_dense_nonsymmetric_operator():
    material, k, alpha, vector = material_mode()
    widths, counts = np.array([2.1, 1.3, 0.8]), (3, 2, 1)
    currents, _ = cosh_inputs(material, alpha, vector, widths)
    result = coupled_neumann_cuboid_reference(material, k, currents, widths, subdivisions=counts)
    laplacians = []
    for n, h in zip(counts, widths/np.array(counts)):
        laplacian = np.zeros((n, n))
        for i in range(n-1):
            laplacian[i, i] += 1/h**2
            laplacian[i+1, i+1] += 1/h**2
            laplacian[i, i+1] -= 1/h**2
            laplacian[i+1, i] -= 1/h**2
        laplacians.append(laplacian)
    nx, ny, nz = counts
    spatial = (np.kron(np.eye(nz), np.kron(np.eye(ny), laplacians[0]))+
               np.kron(np.eye(nz), np.kron(laplacians[1], np.eye(nx)))+
               np.kron(laplacians[2], np.eye(ny*nx)))
    scatter = material.sigma_s.copy()
    np.fill_diagonal(scatter, 0)
    matrix = np.diag(material.removal)-scatter.T-np.outer(material.chi, material.nu_sigma_f)/k
    volume = float(np.prod(widths/np.array(counts)))
    full = volume*(np.kron(np.diag(material.diffusion), spatial)+np.kron(matrix, np.eye(nx*ny*nz)))
    singular = np.linalg.svd(full, compute_uv=False)
    assert result.diagnostics['condition_number_2'] == pytest.approx(np.linalg.cond(full), rel=2e-13)
    assert result.diagnostics['minimum_singular_value'] == pytest.approx(singular[-1], rel=2e-13)
    assert result.diagnostics['maximum_singular_value'] == pytest.approx(singular[0], rel=2e-13)
    assert result.diagnostics['spatial_modes_checked'] == 6
    assert not np.allclose(matrix, matrix.T)


def test_nonconstant_resonance_rejected_even_though_uniform_matrix_invertible():
    # M=.1-2.1=-2, lambda_(1,0,0)=2 on Lx2,Nx2,hx1.
    material = Material([1.0], [0.1], [2.1])
    with pytest.raises(CuboidReferenceError, match='condition_limit') as caught:
        coupled_neumann_cuboid_reference(material, 1, np.ones((6, 1)), [2, 1, 1], subdivisions=(2, 1, 1))
    diagnostics = caught.value.diagnostics
    assert diagnostics['uniform_mode_condition_number_2'] == 1
    assert diagnostics['minimum_singular_spatial_mode'] == (1, 0, 0)
    assert diagnostics['condition_number_2'] > 1e12
    assert diagnostics['spatial_modes_checked'] == 2


def test_singular_uniform_matrix_rejected():
    material = Material([1], [0.1], [0.1])
    with pytest.raises(CuboidReferenceError, match='singular') as caught:
        coupled_neumann_cuboid_reference(material, 1, np.ones((6, 1)), [1, 1, 1], subdivisions=2)
    assert caught.value.diagnostics['minimum_singular_spatial_mode'] == (0, 0, 0)
    assert np.isinf(caught.value.diagnostics['condition_number_2'])


def test_condition_limit_checks_full_operator_not_just_uniform_matrix():
    material = Material([1], [0.1], [0])
    with pytest.raises(CuboidReferenceError, match='condition_limit') as caught:
        coupled_neumann_cuboid_reference(material, 1, np.ones((6, 1)), [1, 1, 1], subdivisions=3, condition_limit=10)
    assert caught.value.diagnostics['uniform_mode_condition_number_2'] == 1
    assert caught.value.diagnostics['condition_number_2'] > 10


@pytest.mark.parametrize('currents, cell_positive', [([-1, 0, -1, 0, 0, 0], False),
                                                    ([-5, 0, 6, 0, 0, 0], True),
                                                    ([0, 0, 0, 0, 0, 0], False)])
def test_negative_cell_or_boundary_and_zero_flux_rejected_without_clipping(currents, cell_positive):
    material = Material([1], [1], [0])
    supplied = np.array(currents, dtype=float)[:, None]
    before = supplied.copy()
    with pytest.raises(CuboidReferenceError, match='no clipping') as caught:
        coupled_neumann_cuboid_reference(material, 1, supplied, [1, 1, 1], subdivisions=1)
    diagnostics = caught.value.diagnostics
    assert (diagnostics['minimum_cell_mean_flux'] > 0) == cell_positive
    assert diagnostics['nonpositive_boundary_group_bins'] > 0
    if cell_positive:
        assert diagnostics['minimum_cell_mean_flux'] == pytest.approx(1)
        assert diagnostics['minimum_boundary_trace_flux'] == pytest.approx(-1.5)
        assert diagnostics['nonpositive_cell_group_bins'] == 0
    np.testing.assert_array_equal(supplied, before)


@pytest.mark.parametrize('updates', [
    dict(k=0), dict(k=np.nan), dict(k=True), dict(k=[1]),
    dict(widths=[1, 0, 1]), dict(widths=[1, np.inf, 1]), dict(widths=[1, 1]),
    dict(net_inward_current=np.ones((4, 1))), dict(net_inward_current=np.full((6, 1), np.nan)),
    dict(subdivisions=0), dict(subdivisions=True), dict(subdivisions=2.5),
    dict(subdivisions=(2, 3)), dict(subdivisions=(2, 1.0, 3)),
    dict(condition_limit=np.inf), dict(condition_limit=0.5), dict(rtol=0), dict(rtol=np.nan), dict(rtol=1),
    dict(native_mean_flux=[0]), dict(native_mean_flux=[np.inf]), dict(native_mean_flux=[1, 2]),
    dict(material=None),
])
def test_invalid_input_shapes_values_and_controls(updates):
    arguments = dict(material=Material([1], [1], [0]), k=1, net_inward_current=np.ones((6, 1)),
                     widths=[1, 1, 1], subdivisions=2)
    arguments.update(updates)
    with pytest.raises(ValueError):
        coupled_neumann_cuboid_reference(**arguments)


@pytest.mark.parametrize('name', ['diffusion', 'sigma_a', 'sigma_s', 'nu_sigma_f', 'chi'])
def test_mutated_nonfinite_material_rejected(name):
    material = Material([1], [1], [0])
    getattr(material, name).flat[0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        coupled_neumann_cuboid_reference(material, 1, np.ones((6, 1)), [1, 1, 1])


def test_generic_three_group_mode_and_nonzero_diagonal_scattering_ignored():
    material, k, alpha, vector = material_mode(3)
    widths = np.array([2.0, 1.0, 0.8])
    currents, mean = cosh_inputs(material, alpha, vector, widths)
    plain = coupled_neumann_cuboid_reference(material, k, currents, widths, subdivisions=(3, 2, 2))
    np.fill_diagonal(material.sigma_s, [0.2, 0.3, 0.4])
    with_diagonal = coupled_neumann_cuboid_reference(material, k, currents, widths, subdivisions=(3, 2, 2))
    np.testing.assert_allclose(with_diagonal.cell_mean_flux, plain.cell_mean_flux, rtol=2e-13)
    np.testing.assert_allclose(plain.volume_mean_flux, mean, rtol=2e-12)
    assert plain.cell_mean_flux.shape == (3, 12)
    assert plain.face_mean_flux.shape == (6, 3)
    assert plain.boundary_profiles.flux.shape[1] == 3
    assert plain.diagnostics['spatial_modes_checked'] == 12


def test_negative_absorption_rejected_even_when_outscatter_keeps_removal_positive():
    material = Material([1, 1], [0.1, 0.1], [0, 0], [[0, 0.5], [0.5, 0]])
    material.sigma_a[0] = -0.1
    assert material.removal[0] > 0
    with pytest.raises(ValueError, match='nonnegative'):
        coupled_neumann_cuboid_reference(material, 1, np.ones((6, 2)), [1, 1, 1])
