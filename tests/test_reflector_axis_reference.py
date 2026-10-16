"""Independent continuous-axis checks with one CPU thread set by the runner."""
import numpy as np
import pytest

from ndgpu.reflector_axis_reference import (
    solve_reflector_axis_reference as solve, ReflectorAxisReferenceError,
    NonpositiveReflectorAxisError,
)
from ndgpu.nem_polynomial_axis import NEMPolynomialAxisResponse


def test_manufactured_coupled_four_group_quadratic():
    D = np.array([1.2, .8, .6, .3]); L = 2.3
    R = np.diag([.4, .5, .6, .7]) - np.diag([.1, .12, .15], -1)
    m = np.array([1., 2., .8, 1.3])
    # phi = m + a*t + b*(t²-1/12); its mean is exactly m.
    a = np.array([.2, -.1, .15, -.25]); b = np.array([.3, -.2, .1, .15])
    J = -D/L*np.stack((a-b, a+b))
    ell0 = np.array([.2, -.1, .05, -.02])
    ellmom = np.stack((-R@a/12, -R@b/60))
    result = solve(D, R, L, m, J, ell0, ellmom)
    t = np.linspace(-.5, .5, 61)
    expected = m + t[:, None]*a + (t[:, None]**2-1/12)*b
    np.testing.assert_allclose(result.evaluate(t), expected, atol=2e-9)
    np.testing.assert_allclose(result.mean, m, atol=1e-10)
    np.testing.assert_allclose(result.delta, R@m + (J[1]-J[0])/L + ell0)
    np.testing.assert_allclose(result.evaluate([-.5, .5], 1), -L*J/D, atol=1e-10)
    for g in range(4):
        candidates = [-.5, .5]
        if b[g] != 0 and abs(-a[g]/(2*b[g])) <= .5:
            candidates.append(-a[g]/(2*b[g]))
        v = m[g]+a[g]*np.array(candidates)+b[g]*(np.array(candidates)**2-1/12)
        assert result.minima[g] == pytest.approx(min(v), abs=2e-9)
    assert result.diagnostics['full_profile_positive']
    assert not result.diagnostics['physical_admission']
    with pytest.raises(ValueError):
        result.face_means[0, 0] = 3
    with pytest.raises(TypeError):
        result.diagnostics['converged'] = False


def test_scalar_cosh_and_fixed_source_shift():
    D, r, L, m, amplitude = 1.2, .7, 2., 3., .4
    k = L*np.sqrt(r/D)
    average_cosh = 2*np.sinh(k/2)/k
    J = -D/L*amplitude*k*np.array([[-np.sinh(k/2)], [np.sinh(k/2)]])
    result = solve([D], [[r]], L, [m], J, [.2], [[0.], [0.]])
    t = np.linspace(-.5, .5, 100)
    expected = m + amplitude*(np.cosh(k*t)-average_cosh)
    np.testing.assert_allclose(result.evaluate(t)[:, 0], expected, atol=2e-9)
    assert result.minima[0] == pytest.approx(m+amplitude*(1-average_cosh), abs=2e-9)
    assert result.minimum_locations[0] == pytest.approx(0, abs=1e-8)
    shifted = solve([D], [[r]], L, [m], J, [123.2], [[0.], [0.]])
    np.testing.assert_allclose(shifted.evaluate(t), result.evaluate(t), atol=1e-10)
    np.testing.assert_allclose(shifted.delta-result.delta, [123.])


def test_scalar_quadratic_and_native_mean_scaling():
    D, r, L, m, a, b = 1.3, .45, 1.7, 2., .3, .8
    J = -D/L*np.array([[a-b], [a+b]])
    ell0 = np.array([.12])
    moments = np.array([[-r*a/12], [-r*b/60]])
    result = solve([D], [[r]], L, [m], J, ell0, moments)
    t = np.linspace(-.5, .5, 31)
    expected = m+a*t+b*(t*t-1/12)
    np.testing.assert_allclose(result.evaluate(t)[:, 0], expected, atol=1e-10)
    scale = 1e-8
    small = solve([D], [[r]], L, [m*scale], J*scale, ell0*scale, moments*scale)
    np.testing.assert_allclose(small.evaluate(t)/scale, result.evaluate(t), atol=1e-10)
    np.testing.assert_allclose(small.delta/scale, result.delta, atol=1e-12)


def test_converged_polynomial_response_independent_comparison():
    D = np.array([1.1, .6]); R = np.array([[.7, -.05], [-.2, .4]])
    L = 3.; m = np.array([1.2, .9])
    J = np.array([[.05, -.03], [-.04, .02]])
    ell = np.array([[.005, -.004], [.002, .001]])
    continuous = solve(D, R, L, m, J, [.03, -.02], ell)
    errors = []
    t = np.linspace(-.5, .5, 81)
    for degree in (4, 8, 16):
        moments = np.zeros((degree-2, 2)); moments[:2] = ell
        polynomial = NEMPolynomialAxisResponse(D, R, L, degree).solve(m, J, leakage_moments=moments)
        values = np.polynomial.legendre.legval(2*t, polynomial.coefficients.T).T
        errors.append(np.max(abs(values-continuous.evaluate(t))))
    assert errors[-1] < 2e-9
    assert errors[-1] < errors[0]/100


def test_negative_field_is_preserved_and_rejected_by_default():
    # phi=1+12(t²-1/12)=12t², then shift via linear term to a negative interior minimum.
    a, b = 1., 12.
    J = -np.array([[a-b], [a+b]])
    args = ([1.], [[1.]], 1., [1.], J, [0.], [[-a/12], [-b/60]])
    with pytest.raises(NonpositiveReflectorAxisError) as caught:
        solve(*args)
    result = solve(*args, require_positive=False)
    np.testing.assert_allclose(result.evaluate([-.5, 0, .5]), caught.value.result.evaluate([-.5, 0, .5]))
    assert result.minima[0] == pytest.approx(-1/48, abs=1e-9)
    assert result.minimum_locations[0] == pytest.approx(-1/24, abs=1e-9)
    assert result.diagnostics['nonpositive_profile_count'] == 1
    assert result.diagnostics['diagnostic_only']
    assert not result.diagnostics['physical_admission']


@pytest.mark.parametrize('replacement', [
    {'diffusion': [0.]}, {'native_mean': [-1.]}, {'currents': [0., 0.]},
    {'transverse_leakage_moments': [[0.]]}, {'loss': [[np.nan]]},
    {'loss': [[-1.]]}, {'loss': [[0.]]}, {'loss': [[1e-14]]},
    {'loss': [[1e14]]}, {'width': np.inf}, {'tol': 0}, {'max_nodes': 4},
    {'require_positive': 'yes'}, {'loss': np.array([[1.+.1j]])},
])
def test_invalid_inputs(replacement):
    args = dict(diffusion=[1.], loss=[[1.]], width=1., native_mean=[1.],
                currents=[[0.], [0.]], transverse_leakage_mean=[0.],
                transverse_leakage_moments=[[0.], [0.]])
    args.update(replacement)
    with pytest.raises(ReflectorAxisReferenceError):
        solve(**args)


@pytest.mark.parametrize('R', [
    [[1., -1.], [-1., 1.]],  # singular conservative scattering
    [[1., .1], [0., 1.]],   # fission-like production sign
    [[1., 0.], [-2., 1.]],  # negative column removal
    [[1., -.99999999999999], [-1., 1.]],  # nearly singular
])
def test_invalid_coupled_loss(R):
    with pytest.raises(ReflectorAxisReferenceError):
        solve([1., 1.], R, 1., [1., 1.], np.zeros((2, 2)), [0., 0.], np.zeros((2, 2)))


def test_nonconvergence_is_rejected():
    with pytest.raises(ReflectorAxisReferenceError, match='did not converge'):
        solve([1.], [[1.]], 20., [5.], [[.1], [-.1]], [0.], [[0.], [0.]], max_nodes=5)


def test_stiff_coupled_derivative_zero_resolution_without_mesh_runaway():
    # Synthetic multiscale reflector: high derivative states cross zero inside
    # the interval. Unbalanced first-order states exhaust 10000 nodes at 1e-10.
    kwargs = dict(
        diffusion=[3., .45, .44, .29],
        loss=[[.01, 0, 0, 0], [-.0095, .0134, -.0053, 0],
              [0, -.0048, .122, -.175], [0, -.00008, -.046, .34]],
        width=21.5, native_mean=[400., 300., 20., 1.],
        currents=[[80., 16., 2.2, .45], [40., 12., .45, 0]],
        transverse_leakage_mean=[-2., .3, -.25, 0],
        transverse_leakage_moments=[[-2.5, .03, .5, .4], [-.7, -.04, .06, .06]],
        require_positive=False, max_nodes=10000)
    coarse = solve(**kwargs, tol=1e-8)
    fine = solve(**kwargs, tol=1e-10)
    points = np.linspace(-.5, .5, 1001)
    scale = np.array(kwargs['native_mean'])
    assert np.max(abs(fine.evaluate(points)-coarse.evaluate(points))/scale) < 1e-6
    assert fine.diagnostics['nodes'] < 10000
    assert fine.diagnostics['solver_rms_residual'] <= 1e-10
    assert fine.diagnostics['first_order_residual'] < 1e-8
    assert fine.diagnostics['mean_balance_error'] < 1e-10
    assert fine.diagnostics['current_balance_error'] < 1e-10
    assert fine.diagnostics['zeroth_balance_error'] < 1e-10
    assert fine.diagnostics['negative_profile_count'] == 2
    assert fine.minima[-1] < -13
    with pytest.raises(NonpositiveReflectorAxisError):
        solve(**{**kwargs, 'require_positive': True}, tol=1e-10)
