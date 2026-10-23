import numpy as np
import pytest
from numpy.polynomial import legendre as leg
from numpy.polynomial import polynomial as poly
from garnet.nem_polynomial_axis import NEMPolynomialAxisResponse, NEMPolynomialAxisError, profile_minimum
from garnet.nodal_polynomial_reference import coupled_nodal_polynomial_axis_reference


@pytest.mark.parametrize('degree',[4,6,8,10,12,14,16])
@pytest.mark.parametrize('groups',[2,4,7,14,23])
def test_manufactured_full_degree_coupled(degree,groups):
    rng=np.random.default_rng(100+degree+groups)
    D=np.linspace(.7,1.4,groups); L=2.3
    C=np.diag(np.linspace(.2,.8,groups))-.03*rng.random((groups,groups))
    c=rng.normal(size=(groups,degree+1))/(1+np.arange(degree+1))**2
    c[:,0]=3
    # Independent strong-form manufacture, integrated by Gauss quadrature.
    u,w=leg.leggauss(2*degree+3); w=w/2
    values=leg.legval(u,c.T)
    second=leg.legval(u,leg.legder(c.T,2))*4
    strong=C@values-D[:,None]/L**2*second
    tests=leg.legvander(u,degree-2)[:,1:].T/2
    qm=(strong*w)@tests.T
    currents=(-D[:,None]/L*leg.legval([-1.,1.],leg.legder(c.T))*2).T
    response=NEMPolynomialAxisResponse(D,C,L,degree)
    got=response.solve(c[:,0],currents,moment_source=qm.T)
    np.testing.assert_allclose(got.coefficients,c,rtol=1e-9,atol=2e-12)
    np.testing.assert_allclose(got.face_means,leg.legval([-1.,1.],c.T).T,atol=2e-12)
    # Permuting groups and rescaling rate/current units preserves solution.
    p=np.arange(groups)[::-1]
    alt=NEMPolynomialAxisResponse(D[p]*1e5,C[p][:,p]*1e5,L,degree)
    other=alt.solve(c[p,0],currents[:,p]*1e5,moment_source=qm[p].T*1e5)
    np.testing.assert_allclose(other.coefficients,got.coefficients[p],atol=2e-12)


@pytest.mark.parametrize('groups',[2,4])
def test_existing_quartic_reference_parity(groups):
    D=np.linspace(.5,1.1,groups);mu=np.linspace(3,4,groups);L=2.1
    M=np.diag(np.linspace(.2,.5,groups))-.02*np.ones((groups,groups))
    inward=np.stack((np.linspace(-.04,.03,groups),np.linspace(.01,.05,groups)))
    ell=np.arange(2*groups).reshape(2,groups)*.003
    old=coupled_nodal_polynomial_axis_reference(D,M,L,mu,inward,ell)
    new=NEMPolynomialAxisResponse(D,M,L).solve(mu,np.stack((inward[0],-inward[1])),leakage_moments=ell)
    np.testing.assert_allclose(new.face_means,old.face_means,atol=2e-13)
    np.testing.assert_allclose(new.moments,old.moments,atol=2e-13)
    u=np.linspace(-1,1,231)
    for g in range(groups):
        np.testing.assert_allclose(leg.legval(u,new.coefficients[g]),poly.polyval(u/2,old.power_coefficients[g]),atol=2e-13)


@pytest.mark.parametrize('shift',[10,42])
def test_quartic_singular_resonance_rejected(shift):
    with pytest.raises(NEMPolynomialAxisError):
        NEMPolynomialAxisResponse(np.ones(2),-shift*np.eye(2),1)


@pytest.mark.parametrize('root_multiplicity',[1,2,3,4,5,6,7,8,9,10,11])
def test_stationary_repeated_roots_and_dense_extrema(root_multiplicity):
    # Integrate derivative with an interior repeated root and a separate simple
    # root. Odd/even multiplicities test flat extrema and stationary inflections.
    derivative=poly.polypow(np.array([-.137,1.]),root_multiplicity)
    if len(derivative)<12:
        derivative=poly.polymul(derivative,[.41,1.])
    power=poly.polyint(derivative);power[0]=-.1
    c=leg.poly2leg(power)
    minimum,where=profile_minimum(c)
    candidates=[-1.,1.,.137]
    if root_multiplicity<11:candidates.append(-.41)
    exact=min(poly.polyval(np.array(candidates),power))
    assert abs(minimum-exact)<1e-11
    assert -.5<=where<=.5
    assert minimum<=min(leg.legval(np.linspace(-1,1,20001),c))+1e-12


@pytest.mark.parametrize('degree',[4,6,8,10,12,14,16])
def test_negative_profile_retained_with_signed_sources(degree):
    response=NEMPolynomialAxisResponse([1.],[[.1]],1.,degree)
    q=np.zeros((degree-2,1));q[0]=4
    result=response.solve([1.],[[0.],[0.]],moment_source=q)
    assert result.diagnostics['negative_profile_count']==1
    assert result.diagnostics['minimum_profile_flux']<0


@pytest.mark.parametrize('degree',[True,3,5,18,4.5])
def test_invalid_degree(degree):
    with pytest.raises(ValueError):NEMPolynomialAxisResponse([1.],[[1.]],1.,degree)


def test_zero_constant_profiles_and_validation():
    assert profile_minimum([0.,0.])==(0.,-.5)
    assert profile_minimum([-2.])==(-2.,-.5)
    with pytest.raises(ValueError):NEMPolynomialAxisResponse([-1.],[[1.]],1.)
    with pytest.raises(ValueError):NEMPolynomialAxisResponse([1.],[[np.nan]],1.)
    with pytest.raises(ValueError):NEMPolynomialAxisResponse([1.],[[1.]],1.).solve([1.],[[0.],[0.]],moment_source=np.ones((3,1)))
