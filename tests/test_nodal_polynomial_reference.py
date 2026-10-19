"""Independent quadrature/manufactured checks of the local axis reference."""
import numpy as np
import pytest

from ndgpu.nodal_polynomial_reference import (
    NodalPolynomialReferenceError,
    _stationary_points,
    coupled_nodal_polynomial_axis_reference,
    quadratic_neighbor_leakage_moments,
)


def basis_at(t):
    t = np.asarray(t)
    return np.array([np.ones_like(t),t,3*t*t-.25,
                     t*(t*t-.25),(t*t-.05)*(t*t-.25)])


def derivative_at(t):
    t = np.asarray(t)
    return np.array([np.zeros_like(t),np.ones_like(t),6*t,
                     3*t*t-.25,4*t**3-.6*t])


def manufactured(coefficients, D, M, width):
    """Inputs from Gauss integrals of the actual polynomial, not block rows."""
    t, weights = np.polynomial.legendre.leggauss(10)
    t, weights = t/2,weights/2
    values = coefficients@basis_at(t)
    second = coefficients@np.array([np.zeros_like(t),np.zeros_like(t),
                                   np.full_like(t,6),6*t,12*t*t-.6])
    tests = np.array([t,3*t*t-.25])
    moments = (values*weights)@tests.T
    mean = values@weights
    leakage = (D[:,None]/width**2*((second*weights)@tests.T)-M@moments).T
    derivative = coefficients@derivative_at(np.array([-.5,.5]))
    inward = np.array([-D/width*derivative[:,0],D/width*derivative[:,1]])
    return mean,inward,leakage,moments.T


def problem(groups):
    D = np.linspace(1.1,.4,groups)
    M = np.diag(np.linspace(.4,.2,groups))
    for g in range(groups):
        for j in range(groups):
            if g != j:
                M[g,j] = -.01*(g+1)/(j+1)
    coefficients = np.column_stack((np.linspace(2.,1.2,groups),
                                   np.linspace(.1,-.12,groups),
                                   np.linspace(.08,-.1,groups),
                                   np.linspace(.05,.04,groups),
                                   np.linspace(-.03,.02,groups)))
    return D,M,2.4,coefficients


@pytest.mark.parametrize('groups',[2,4])
def test_nonsymmetric_multigroup_manufactured_quartic(groups):
    D,M,width,coefficients = problem(groups)
    mu,inward,leakage,moments = manufactured(coefficients,D,M,width)
    result = coupled_nodal_polynomial_axis_reference(D,M,width,mu,inward,leakage)
    np.testing.assert_allclose(result.basis_coefficients,coefficients,rtol=2e-10,atol=1e-12)
    np.testing.assert_allclose(result.moments,moments,rtol=1e-11,atol=1e-13)
    np.testing.assert_allclose(result.face_means,(coefficients@basis_at([-.5,.5])).T,rtol=1e-12)
    t = np.linspace(-.5,.5,37)
    np.testing.assert_allclose(np.polynomial.polynomial.polyval(t,result.power_coefficients.T),coefficients@basis_at(t),rtol=1e-12)
    for key in ('scaled_backward_error','physical_weak_balance_error',
                'physical_current_balance_error','physical_mean_balance_error',
                'physical_moment_balance_error'):
        assert result.diagnostics[key] < 1e-12
    np.testing.assert_allclose(result.diagnostics['axis_balance_without_transverse_mean'],M@mu-(inward[0]+inward[1])/width)
    assert result.diagnostics['minimum_polynomial_flux'] > 0
    raw_ratio = abs(result.diagnostics['physical_weak_residual'])/result.diagnostics['physical_weak_residual_scale']
    assert result.diagnostics['physical_weak_balance_error'] == pytest.approx(np.max(raw_ratio),rel=1e-14)


def test_unequal_neighbor_interval_means_independent_quadrature():
    center_width,minus_width,plus_width = 2.,1.25,3.5
    p0 = np.array([.8,-.2,1.5,-2.])
    p1 = np.array([-.375,.21,-.7,.8])
    p2 = np.array([5/11,-.42,.5,-.2])
    t,w = np.polynomial.legendre.leggauss(8)
    def average(lo,hi):
        nodes = (lo+hi)/2+(hi-lo)/2*t
        return (p0[:,None]+p1[:,None]*nodes+p2[:,None]*(3*nodes**2-.25))@(w/2)
    minus = average(-.5-minus_width/center_width,-.5)
    center = average(-.5,.5)
    plus = average(.5,.5+plus_width/center_width)
    result = quadratic_neighbor_leakage_moments(center,minus,plus,center_width,minus_width,plus_width)
    np.testing.assert_allclose(result,np.array([p1/12,p2/20]),rtol=1e-13,atol=1e-15)
    assert not np.allclose(result[1],p2/60)
    # Swapping neighbors and their widths reverses only the odd moment.
    reflected = quadratic_neighbor_leakage_moments(center,plus,minus,center_width,plus_width,minus_width)
    np.testing.assert_allclose(reflected,result*np.array([[-1],[1]]),rtol=1e-13,atol=1e-15)


@pytest.mark.parametrize('multiple',[-10.,-42.])
def test_actual_kernel_spatial_resonances(multiple):
    D,width = .7,1.3
    with pytest.raises(NodalPolynomialReferenceError,match='singular|condition') as exc:
        coupled_nodal_polynomial_axis_reference([D],[[multiple*D/width**2]],width,[1.],np.zeros((2,1)),np.zeros((2,1)))
    assert exc.value.diagnostics['condition_number_2_scaled'] > 1e12


@pytest.mark.parametrize('coefficients,reason',[
    (np.array([[1.,3.,0.,0.,0.]]),'whole-face'),
    (np.array([[1.,0.,5.,0.,0.]]),'polynomial profile'),
])
def test_negative_profiles_preserved(coefficients,reason):
    D,M,width = np.array([1.]),np.array([[.3]]),1.7
    mu,inward,leakage,_ = manufactured(coefficients,D,M,width)
    with pytest.raises(NodalPolynomialReferenceError,match=reason) as exc:
        coupled_nodal_polynomial_axis_reference(D,M,width,mu,inward,leakage)
    result = exc.value.result
    assert result is not None
    np.testing.assert_allclose(result.basis_coefficients,coefficients,atol=1e-12)
    assert result.diagnostics['minimum_polynomial_flux'] < 0
    if reason == 'polynomial profile':
        assert np.all(result.face_means > 0)
        assert result.diagnostics['minimum_polynomial_t'] == pytest.approx(0.,abs=1e-13)
        assert result.diagnostics['minimum_polynomial_flux'] == pytest.approx(-.25)


@pytest.mark.parametrize('power,expected',[
    ([2.,0.,0.,0.,0.],[]),
    ([2.,.2,0.,0.,0.],[]),
    ([1.,0.,1.,0.,0.],[0.]),
    ([.0101,-.004,.06,-.4,1.],[.1]),  # triple derivative root
    ([1.,.002,-.015,0.,.25],[-.2,.1]),  # double derivative root
])
def test_degenerate_and_repeated_stationary_points(power,expected):
    actual = _stationary_points(np.asarray(power))
    if not expected:
        assert actual.size == 0
    for root in expected:
        assert np.min(abs(actual-root)) < 1e-6
    assert np.all((actual > -.5)&(actual < .5))


def test_constant_and_linear_profiles():
    D = np.array([1.,.2]);M = np.zeros((2,2));width = 2.
    coefficients = np.array([[1.3,0.,0.,0.,0.],[2.,.2,0.,0.,0.]])
    mu,inward,leakage,_ = manufactured(coefficients,D,M,width)
    result = coupled_nodal_polynomial_axis_reference(D,M,width,mu,inward,leakage)
    np.testing.assert_allclose(result.basis_coefficients,coefficients,atol=1e-12)
    np.testing.assert_allclose(result.diagnostics['profile_minima'],[1.3,1.9],atol=1e-12)


@pytest.mark.parametrize('length,flux,rate',[(1e30,1e70,1e-40),(1e-30,1e-70,1e40)])
def test_physical_normalization_invariance(length,flux,rate):
    D,M,width,coefficients = problem(4)
    mu,inward,leakage,_ = manufactured(coefficients,D,M,width)
    result = coupled_nodal_polynomial_axis_reference(D*rate*length**2,M*rate,width*length,mu*flux,inward*rate*length*flux,leakage*rate*flux)
    np.testing.assert_allclose(result.basis_coefficients/flux,coefficients,rtol=2e-10,atol=1e-12)


def test_group_permutation():
    D,M,width,coefficients = problem(4)
    mu,inward,leakage,_ = manufactured(coefficients,D,M,width)
    result = coupled_nodal_polynomial_axis_reference(D,M,width,mu,inward,leakage)
    perm = np.array([2,0,3,1])
    reordered = coupled_nodal_polynomial_axis_reference(D[perm],M[np.ix_(perm,perm)],width,mu[perm],inward[:,perm],leakage[:,perm])
    np.testing.assert_allclose(reordered.basis_coefficients,result.basis_coefficients[perm],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(reordered.face_means,result.face_means[:,perm],rtol=1e-12)
    np.testing.assert_allclose(reordered.moments,result.moments[:,perm],rtol=1e-12)


def test_condition_is_full_scaled_dense_svd(monkeypatch):
    original = np.linalg.solve
    matrices = []
    def capture(matrix,rhs):
        matrices.append(matrix.copy())
        return original(matrix,rhs)
    monkeypatch.setattr(np.linalg,'solve',capture)
    D,M,width,coefficients = problem(2)
    mu,inward,leakage,_ = manufactured(coefficients,D,M,width)
    result = coupled_nodal_polynomial_axis_reference(D,M,width,mu,inward,leakage)
    assert matrices[0].shape == (8,8)
    assert result.diagnostics['condition_number_2_scaled'] == pytest.approx(np.linalg.cond(matrices[0],2),rel=1e-13)
    with pytest.raises(NodalPolynomialReferenceError,match='condition'):
        coupled_nodal_polynomial_axis_reference(D,M,width,mu,inward,leakage,condition_limit=1.)


def test_strict_residual_gate_retains_fields():
    D,M,width,coefficients = problem(2)
    mu,inward,leakage,_ = manufactured(coefficients,D,M,width)
    with pytest.raises(NodalPolynomialReferenceError,match='balance tolerance') as exc:
        coupled_nodal_polynomial_axis_reference(D,M,width,mu,inward,leakage,rtol=1e-20)
    assert exc.value.result is not None
    assert exc.value.diagnostics['physical_weak_balance_error'] > 1e-20


@pytest.mark.parametrize('replace',[
    {'diffusion':[0.]},{'diffusion':[]},{'mean':[-1.]},{'coupling':[[np.nan]]},
    {'coupling':[[1.,2.]]},{'coupling':[[1j]]},{'net_inward_density':[[0.,0.]]},
    {'transverse_leakage_moments':[[0.]]},{'width':0.},{'width':[1.]},
    {'width':np.inf},{'rtol':0.},{'condition_limit':.5},
])
def test_malformed_axis_inputs(replace):
    args = dict(diffusion=[1.],coupling=[[.3]],width=1.,mean=[1.],
                net_inward_density=np.zeros((2,1)),transverse_leakage_moments=np.zeros((2,1)))
    args.update(replace)
    with pytest.raises(ValueError):
        coupled_nodal_polynomial_axis_reference(**args)


@pytest.mark.parametrize('replace',[
    {'center_avg':[]},{'minus_avg':[1.,2.]},{'plus_avg':[np.nan]},
    {'center_width':0.},{'minus_width':-1.},{'plus_width':np.inf},
])
def test_malformed_neighbor_inputs(replace):
    args = dict(center_avg=[1.],minus_avg=[1.],plus_avg=[1.],center_width=1.,minus_width=1.,plus_width=1.)
    args.update(replace)
    with pytest.raises(ValueError):
        quadratic_neighbor_leakage_moments(**args)


def test_cpu_only():
    class DeviceArray:
        __cuda_array_interface__ = {}
    with pytest.raises(TypeError,match='CPU'):
        coupled_nodal_polynomial_axis_reference(DeviceArray(),[[.3]],1.,[1.],np.zeros((2,1)),np.zeros((2,1)))
    with pytest.raises(TypeError,match='CPU'):
        quadratic_neighbor_leakage_moments(DeviceArray(),[1.],[1.],1.,1.,1.)
