"""DCT/LU equivalence and independent analytic/positivity controls, CPU only."""
import copy

import numpy as np
import pytest

from ndgpu import (Material, CuboidReferenceError, CoupledCuboidReference,
                   coupled_neumann_cuboid_reference as lu_reference,
                   coupled_neumann_cuboid_reference_modal as modal_reference)
import ndgpu.cuboid_reference as lu_module
from ndgpu.serpent_adf import cuboid_flat_source_reconstruction


def positive_mode(groups, widths):
    """Genuine scatter/fission coupling and nonuniform traces along all axes."""
    diffusion, vector = np.linspace(1.3, 0.8, groups), np.linspace(1.0, 0.7, groups)
    scatter = np.full((groups, groups), 0.02)
    np.fill_diagonal(scatter, 0)
    if groups > 1:
        scatter[0, 1] = 0.04
    chi = np.arange(groups, 0, -1, dtype=float)
    chi /= chi.sum()
    production, k, alpha = np.linspace(0.02, 0.01, groups), 1.1, 0.4
    coupling = scatter.T+np.outer(chi, production)/k
    removal = coupling@vector/vector+alpha**2*diffusion
    material = Material(diffusion, removal-scatter.sum(axis=1), production, scatter, chi)
    even, odd = np.array([1., .4, .25]), np.array([.2, -.15, .1])
    currents = np.zeros((6, groups))
    mean = np.zeros(groups)
    for axis, (minus, plus) in enumerate(((0, 2), (1, 3), (4, 5))):
        t = alpha*widths[axis]/2
        area = np.prod(np.delete(widths, axis))
        currents[minus] = diffusion*vector*alpha*(even[axis]*np.sinh(t)-odd[axis]*np.cosh(t))*area
        currents[plus] = diffusion*vector*alpha*(even[axis]*np.sinh(t)+odd[axis]*np.cosh(t))*area
        mean += vector*even[axis]*np.sinh(t)/t
    return material, k, currents, mean


def recovered_currents(result, diffusion):
    boundary = result.boundary_profiles
    density = ((boundary.flux-result.cell_mean_flux[:, boundary.owners].T)*diffusion[None]/boundary.distances[:, None])
    return np.array([(density[boundary.sides == side]*boundary.areas[boundary.sides == side, None]).sum(axis=0)
                     for side in range(6)])


@pytest.mark.parametrize('groups,counts', [(2, (3, 4, 2)), (3, (4, 3, 2))])
def test_modal_matches_lu_fields_profiles_means_currents_and_diagnostics(groups, counts):
    widths = np.array([2.3, 1.6, 2.7])
    material, k, currents, native = positive_mode(groups, widths)
    snapshot = copy.deepcopy((material, currents, widths, native))
    expected = lu_reference(material, k, currents, widths, subdivisions=counts, native_mean_flux=1.03*native)
    actual = modal_reference(material, k, currents, widths, subdivisions=counts, native_mean_flux=1.03*native)
    assert isinstance(actual, CoupledCuboidReference)
    np.testing.assert_allclose(actual.cell_mean_flux, expected.cell_mean_flux, rtol=3e-12, atol=1e-13)
    np.testing.assert_allclose(actual.face_mean_flux, expected.face_mean_flux, rtol=3e-12, atol=1e-13)
    np.testing.assert_allclose(actual.volume_mean_flux, native, rtol=2e-12)
    np.testing.assert_allclose(actual.volume_mean_flux, expected.volume_mean_flux, rtol=2e-12)
    np.testing.assert_allclose(actual.cell_centers, expected.cell_centers, rtol=2e-15, atol=2e-15)
    np.testing.assert_array_equal(actual.cell_volumes, expected.cell_volumes)
    for name in ('owners', 'sides'):
        np.testing.assert_array_equal(getattr(actual.boundary_profiles, name), getattr(expected.boundary_profiles, name))
    for name in ('areas', 'distances', 'centers', 'inward_current_density', 'flux'):
        np.testing.assert_allclose(getattr(actual.boundary_profiles, name), getattr(expected.boundary_profiles, name), rtol=3e-12, atol=2e-14)
    np.testing.assert_allclose(recovered_currents(actual, material.diffusion), currents, rtol=1e-11, atol=2e-13)
    for key in ('condition_number_2', 'minimum_singular_value', 'maximum_singular_value',
                'minimum_singular_spatial_mode', 'uniform_mode_condition_number_2', 'spatial_modes_checked'):
        assert actual.diagnostics[key] == expected.diagnostics[key]
    for key in ('normwise_backward_error', 'integrated_group_balance_max_relative'):
        assert actual.diagnostics[key] < 1e-13
        assert expected.diagnostics[key] < 1e-13
    for key in ('minimum_cell_mean_flux_per_group', 'minimum_boundary_trace_flux_per_group',
                'integrated_group_balance_relative'):
        np.testing.assert_allclose(actual.diagnostics[key], expected.diagnostics[key], rtol=3e-12, atol=2e-14)
    for key in ('native_mean_flux', 'compatibility_defect_density', 'computed_relative_difference'):
        np.testing.assert_allclose(actual.native_mean_compatibility[key], expected.native_mean_compatibility[key], rtol=1e-10, atol=2e-13)
    assert actual.native_mean_compatibility['native_mean_imposed'] is False
    assert actual.discrete_positive is True
    assert actual.continuous_positivity_certified is False
    assert actual.diagnostics['transform_workers'] == 1
    assert actual.diagnostics['mesh_object_constructed'] is False
    for key in ('diffusion', 'sigma_a', 'sigma_s', 'nu_sigma_f', 'chi'):
        np.testing.assert_array_equal(getattr(material, key), getattr(snapshot[0], key))
    for given, before in zip((currents, widths, native), snapshot[1:]):
        np.testing.assert_array_equal(given, before)


def test_modal_default_grid_does_not_construct_mesh_or_use_sparse_lu(monkeypatch):
    def prohibited(*args, **kwargs):
        pytest.fail('Modal backend must not build Mesh/mesh operators or call sparse LU')
    monkeypatch.setattr(lu_module, 'build_rectilinear_mesh_3d', prohibited)
    monkeypatch.setattr(lu_module, 'MeshGroupOperator', prohibited)
    monkeypatch.setattr(lu_module, 'splu', prohibited)
    widths = np.array([2.3, 1.6, 2.7])
    material, k, currents, native = positive_mode(2, widths)
    result = modal_reference(material, k, currents, widths)
    assert result.subdivisions == (8, 8, 8)
    assert result.diagnostics['spatial_modes_checked'] == 512  # crosses modal solve chunk boundary
    np.testing.assert_allclose(result.volume_mean_flux, native, rtol=2e-12)
    np.testing.assert_allclose(recovered_currents(result, material.diffusion), currents, rtol=2e-11, atol=2e-13)


def test_scalar_cosh_refinement_and_old_zero_uniform_source_limit():
    material = Material([1.2], [.3], [0])
    widths, alpha = np.array([3., 1.2, .7]), .5
    currents = np.zeros((6, 1))
    currents[0] = currents[2] = material.diffusion*alpha*np.sinh(alpha*widths[0]/2)*widths[1]*widths[2]
    mean = np.array([np.sinh(alpha*widths[0]/2)/(alpha*widths[0]/2)])
    scalar = cuboid_flat_source_reconstruction(mean, material.diffusion, material.removal, currents, widths)
    np.testing.assert_allclose(scalar.uniform_source, 0, atol=3e-16)
    cell_errors, face_errors = [], []
    for n in (8, 16, 32):
        result = modal_reference(material, 1, currents, widths, subdivisions=(n, 1, 1))
        h = widths[0]/n
        x = result.cell_centers[:, 0]-widths[0]/2
        exact = (np.sinh(alpha*(x+h/2))-np.sinh(alpha*(x-h/2)))/(alpha*h)
        cell_errors.append(float(np.max(abs(result.cell_mean_flux[0]-exact))))
        face_errors.append(float(np.max(abs(result.face_mean_flux-scalar.face_flux))))
        np.testing.assert_allclose(result.volume_mean_flux, mean, rtol=1e-12)
        assert result.diagnostics['normwise_backward_error'] < 1e-14
    assert all(3 < cell_errors[i]/cell_errors[i+1] < 5 for i in range(2))
    assert all(3 < face_errors[i]/face_errors[i+1] < 5 for i in range(2))
    assert face_errors[-1] > 100*result.diagnostics['normwise_backward_error']


def test_nonfissile_reflector_like_currents_reject_same_negative_field_as_lu():
    material = Material([1, .5], [.2, .3], [0, 0], [[0, .1], [.04, 0]])
    currents = np.zeros((6, 2))
    currents[0], currents[2] = [-100, -60], [101, 61]
    # Integrated implied means are positive; that is insufficient for field positivity.
    matrix = np.diag(material.removal)-material.sigma_s.T
    assert np.all(np.linalg.solve(matrix, currents.sum(axis=0)) > 0)
    failures = []
    for function in (lu_reference, modal_reference):
        with pytest.raises(CuboidReferenceError, match='no clipping') as caught:
            function(material, 1, currents, [2.1, 1.3, .8], subdivisions=(4, 3, 2))
        failures.append(caught.value.diagnostics)
    for key in ('minimum_cell_mean_flux_per_group', 'minimum_boundary_trace_flux_per_group'):
        np.testing.assert_allclose(failures[0][key], failures[1][key], rtol=2e-12, atol=1e-12)
    assert failures[1]['nonpositive_cell_group_bins'] > 0
    assert failures[1]['nonpositive_boundary_group_bins'] > 0


def test_nonconstant_resonance_rejected_even_for_zero_rhs_in_resonant_mode():
    material = Material([1], [.1], [2.1])  # uniform M=-2, condition1; Nx2,Lx2 lambda1=2
    for function in (lu_reference, modal_reference):
        with pytest.raises(CuboidReferenceError, match='condition_limit') as caught:
            function(material, 1, np.ones((6, 1)), [2, 1, 1], subdivisions=(2, 1, 1))
        diagnostics = caught.value.diagnostics
        assert diagnostics['uniform_mode_condition_number_2'] == 1
        assert diagnostics['minimum_singular_spatial_mode'] == (1, 0, 0)
        assert diagnostics['condition_number_2'] > 1e12


def test_constant_mode_singular_rejected():
    material = Material([1], [.1], [.1])
    with pytest.raises(CuboidReferenceError, match='singular') as caught:
        modal_reference(material, 1, np.ones((6, 1)), [1, 1, 1], subdivisions=2)
    assert caught.value.diagnostics['minimum_singular_spatial_mode'] == (0, 0, 0)


@pytest.mark.parametrize('currents,cell_positive', [([-1, 0, -1, 0, 0, 0], False),
                                                  ([-5, 0, 6, 0, 0, 0], True),
                                                  ([0, 0, 0, 0, 0, 0], False)])
def test_cell_boundary_and_zero_positivity_rejection(currents, cell_positive):
    material = Material([1], [1], [0])
    with pytest.raises(CuboidReferenceError, match='no clipping') as caught:
        modal_reference(material, 1, np.array(currents)[:, None], [1, 1, 1], subdivisions=1)
    diagnostics = caught.value.diagnostics
    assert (diagnostics['minimum_cell_mean_flux'] > 0) == cell_positive
    assert diagnostics['nonpositive_boundary_group_bins'] > 0
    if cell_positive:
        assert diagnostics['minimum_cell_mean_flux'] == pytest.approx(1)
        assert diagnostics['minimum_boundary_trace_flux'] == pytest.approx(-1.5)


@pytest.mark.parametrize('updates', [dict(k=np.nan), dict(k=0), dict(k=True), dict(widths=[1, np.inf, 1]),
    dict(widths=[1, 0, 1]), dict(widths=[1, 1]), dict(net_inward_current=np.full((6, 1), np.nan)),
    dict(net_inward_current=np.ones((6, 2))), dict(subdivisions=(2, 3)), dict(subdivisions=(2, 1., 3)),
    dict(subdivisions=False), dict(condition_limit=np.inf), dict(condition_limit=.5), dict(rtol=0),
    dict(rtol=np.nan), dict(native_mean_flux=[np.inf]), dict(native_mean_flux=[0]), dict(material=None)])
def test_invalid_nonfinite_inputs_and_controls(updates):
    arguments = dict(material=Material([1], [1], [0]), k=1, net_inward_current=np.ones((6, 1)), widths=[1, 1, 1])
    arguments.update(updates)
    with pytest.raises(ValueError):
        modal_reference(**arguments)


def test_nonfinite_material_rejected():
    material = Material([1], [1], [0])
    material.sigma_s[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        modal_reference(material, 1, np.ones((6, 1)), [1, 1, 1])
