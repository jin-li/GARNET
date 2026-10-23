"""Public factor-extension/NEM integration on a fabricated two-node spectrum.

These tests establish whole-face coupling and conservation, not nuclear accuracy
or admission of ratios inferred from a transport/reference calculation. There is
no imposed volume source: the eigenproblem contains its usual fission source.
"""
import numpy as np
import pytest
from numpy.polynomial import legendre as leg
from numpy.testing import assert_allclose, assert_array_equal

from garnet.nem_geometry import CuboidNodalGrid
from garnet.nem_higher_order import NEMHigherOrderDiffusionSolver
from garnet.reflector_interface_factors import extend_reflector_interface_factors
from test_nem_higher_order import independent_pencil, recovered_vector

PAIRS = ((0, 2), (1, 3), (4, 5))


def extend(grid, baseline, fuel_mask, reflector_ratio):
    # Manufactured trace data exercise the public ratio API. Their small order
    # indicators are fixture inputs, not measured evidence of convergence.
    homogeneous = np.full_like(baseline, 2.)
    heterogeneous = homogeneous.copy()
    heterogeneous[1, 0] *= reflector_ratio
    change_over_mean = np.full_like(baseline, .001)
    factor_change = np.full_like(baseline, .002)
    arguments = (baseline, fuel_mask, grid.neighbors, heterogeneous, homogeneous,
                 change_over_mean, factor_change)
    before = [value.copy() for value in arguments]
    factors = extend_reflector_interface_factors(*arguments, order_tolerance=.005)
    for actual, previous in zip(arguments, before):
        assert_array_equal(actual, previous)
    return factors


@pytest.fixture(scope='module', params=[14, 16], ids=['P14', 'P16'])
def problem(request):
    degree = request.param
    bounds = [[[0., 1.4], [0., 1.1], [0., .8]],
              [[1.4, 2.3], [0., 1.1], [0., .8]]]
    # Transverse reflection makes this a one-dimensional heterogeneous slab.
    # Both x ends are vacuum, with no boundary drive or fitted mean constraint.
    preliminary = CuboidNodalGrid(bounds)
    boundary = ['vacuum' if preliminary.face_axes[face] == 0 else 'reflective'
                for face in preliminary.boundary_faces]
    grid = CuboidNodalGrid(bounds, boundary, leakage_boundary='primary_rule')
    diffusion = np.array([[.9, .45], [1.2, .7]])
    absorption = np.array([[.03, .06], [.015, .03]])
    scattering = np.array([[[0., .04], [.002, 0.]],
                           [[0., .025], [.001, 0.]]])
    nu_fission = np.array([[.08, .12], [0., 0.]])
    chi = np.array([[1., 0.], [1., 0.]])
    materials = (diffusion, absorption, scattering, nu_fission, chi)
    fuel = np.array([True, False])
    baseline = np.ones((2, 6, 2), dtype=np.float64)
    baseline[0, 2] = [.95, 1.08]
    unity = extend(grid, baseline, fuel, np.ones(2))
    nonunity = extend(grid, baseline, fuel, np.array([.85, 1.12]))
    results = {}
    for name, factors in [('fuel', baseline), ('unity', unity), ('reflector', nonunity)]:
        solver = NEMHigherOrderDiffusionSolver(grid, *materials,
            node_face_factors=factors, degree=degree, rtol=1e-10)
        assert_array_equal(solver.node_face_factors, factors)
        results[name] = solver.solve_eigenvalue(rtol=1e-10, max_iterations=1000)
    return degree, grid, materials, fuel, baseline, unity, nonunity, results


def test_unity_reflector_reproduces_fuel_only_complete_eigenstate(problem):
    degree, grid, materials, fuel, baseline, unity, nonunity, results = problem
    assert_array_equal(unity, baseline)
    assert unity[fuel].tobytes() == baseline[fuel].tobytes()
    reference, actual = results['fuel'], results['unity']
    assert actual.k_eff == reference.k_eff
    assert actual.iterations == reference.iterations
    for name in ('means', 'moments', 'currents', 'side_face_means', 'corrected_face_means'):
        assert_array_equal(getattr(actual, name), getattr(reference, name))
    assert_array_equal(actual.diagnostics['profile_minima'], reference.diagnostics['profile_minima'])
    assert actual.diagnostics['full_eigen_residual'] == reference.diagnostics['full_eigen_residual']


def test_nonunity_ratio_changes_public_coupling_without_changing_fuel_or_exterior(problem):
    degree, grid, materials, fuel, baseline, unity, factors, results = problem
    assert factors[fuel].tobytes() == baseline[fuel].tobytes()
    exterior = grid.neighbors < 0
    assert_array_equal(factors[exterior], np.ones_like(factors[exterior]))
    assert_array_equal(factors[1, 0], [.85, 1.12])
    assert not factors.flags.writeable
    actual, reference = results['reflector'], results['fuel']
    assert abs(actual.k_eff-reference.k_eff) > 1e-5
    for name in ('means', 'currents', 'side_face_means'):
        assert np.max(abs(getattr(actual, name)-getattr(reference, name))) > 1e-5
    # Each owner retains its factor: the uncorrected traces jump, while the
    # physically corrected interface scalar flux remains continuous.
    assert np.max(abs(actual.side_face_means[0, 2]-actual.side_face_means[1, 0])) > 1e-4
    assert_allclose(factors[0, 2]*actual.side_face_means[0, 2],
                    factors[1, 0]*actual.side_face_means[1, 0], rtol=3e-12, atol=2e-13)


def physical_coefficients(result, degree):
    """Recover Legendre polynomials from public means, moments and endpoints."""
    count, groups = result.means.shape
    coefficients = np.zeros((count, 3, groups, degree+1))
    coefficients[..., 0] = result.means[:, None, :]
    for order in range(1, degree-1):
        coefficients[..., order] = result.moments[:, :, order-1, :]*2*(2*order+1)
    for axis, (minus, plus) in enumerate(PAIRS):
        coefficients[:, axis, :, degree-1] = ((result.side_face_means[:, plus]-result.side_face_means[:, minus])/2
                                              -coefficients[:, axis, :, 1:degree-1:2].sum(axis=-1))
        coefficients[:, axis, :, degree] = ((result.side_face_means[:, plus]+result.side_face_means[:, minus])/2
                                            -coefficients[:, axis, :, :degree:2].sum(axis=-1))
    return coefficients


def test_independent_dense_eigenpencil_and_physical_interface_balances(problem):
    degree, grid, materials, fuel, baseline, unity, factors, results = problem
    result = results['reflector']
    diffusion, absorption, scattering, nu_fission, chi = materials
    # The existing test-only full uncondensed pencil uses independent Gauss
    # integrations and never calls NEM's condensed assembler/local responses.
    matrix, fission = independent_pencil(grid, *materials, degree, factors)
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(matrix, fission))
    index = np.argmax(eigenvalues.real)
    assert abs(eigenvalues[index].imag) < 1e-12
    assert_allclose(result.k_eff, eigenvalues[index].real, rtol=2e-9, atol=2e-11)
    dense = eigenvectors[:, index]
    assert np.max(abs(dense.imag)) < 1e-12
    dense = dense.real
    dense_mean = dense[:4].reshape(2, 2)
    production = np.sum(grid.volumes[:, None]*nu_fission*dense_mean)
    dense /= production
    state = recovered_vector(result, degree)
    assert_allclose(state, dense, rtol=2e-7, atol=3e-11)
    assert np.max(abs(matrix@state-fission@state/result.k_eff)) < 1e-9

    coefficients = physical_coefficients(result, degree)
    outgoing_integrated = np.zeros((2, 6, 2))
    independent_minima = np.empty((2, 3, 2))
    for node in range(2):
        for axis, sides in enumerate(PAIRS):
            for group in range(2):
                polynomial = coefficients[node, axis, group]
                roots = leg.legroots(leg.legder(polynomial))
                points = [-1., 1.]+[root.real for root in roots
                                     if abs(root.imag) < 1e-9 and -1 <= root.real <= 1]
                independent_minima[node, axis, group] = np.min(leg.legval(points, polynomial))
                for end, side in zip((-1., 1.), sides):
                    trace = leg.legval(end, polynomial)
                    assert_allclose(trace, result.side_face_means[node, side, group], rtol=1e-13, atol=1e-13)
                    global_current = -2*diffusion[node, group]/grid.widths[node, axis]*leg.legval(end, leg.legder(polynomial))
                    face = grid.node_faces[node, side]
                    assert_allclose(global_current, result.currents[face, group], rtol=2e-9, atol=1e-12)
                    outgoing_integrated[node, side, group] = end*global_current*grid.node_face_areas[node, side]
    assert np.all(result.means > 0)
    assert np.all(result.side_face_means > 0)
    assert np.all(result.corrected_face_means > 0)
    assert np.all(independent_minima > 0)
    assert_allclose(independent_minima, result.diagnostics['profile_minima'], rtol=1e-11, atol=1e-12)
    assert_allclose(outgoing_integrated[0, 2], -outgoing_integrated[1, 0], rtol=2e-10, atol=1e-12)

    # Direct groupwise reaction/production equations, integrated over each
    # node, with no imposed volume source. Scattering uses source-row order.
    for node in range(2):
        mean = result.means[node]
        reaction = absorption[node]*mean
        for group in range(2):
            other = 1-group
            reaction[group] += scattering[node, group, other]*mean[group]-scattering[node, other, group]*mean[other]
        source = chi[node]*np.dot(nu_fission[node], mean)/result.k_eff
        assert_allclose(grid.volumes[node]*(reaction-source)+outgoing_integrated[node].sum(axis=0),
                        np.zeros(2), atol=2e-10)
    assert np.all(nu_fission[1] == 0)  # Reflector node is explicitly nonfission.
    for key in ('volume_balance_error', 'weak_balance_error', 'current_balance_error',
                'mean_balance_error', 'moment_balance_error', 'interface_balance_error',
                'boundary_balance_error', 'full_eigen_residual'):
        assert 0 <= result.diagnostics[key] <= 1e-10
