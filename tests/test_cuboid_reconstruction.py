"""Independent Neumann and limiting-problem checks of frozen reconstruction."""
import numpy as np
import pytest
from ndgpu.serpent_adf import cuboid_flat_source_reconstruction as reconstruct
from ndgpu.serpent_adf import cuboid_flat_source_factors


def currents(widths, face_densities):
    x,y,z=widths
    areas=np.array([y*z,x*z,y*z,x*z,x*y,x*y])
    return np.asarray(face_densities)*areas[:,None]


def test_zero_currents_and_source_defect():
    mean=np.array([2.,3.]);D=np.array([.3,1.]);r=np.array([0.,.1])
    result=reconstruct(mean,D,r,np.zeros((6,2)),[3.,4.,5.],physical_source=[0.,.25])
    np.testing.assert_allclose(result.face_flux,np.broadcast_to(mean,(6,2)))
    np.testing.assert_allclose(result.uniform_source,[0.,.3])
    np.testing.assert_allclose(result.source_defect,[0.,.05])
    np.testing.assert_allclose(result.minimum_flux,mean)


def test_manufactured_affine_zero_removal_all_axes():
    widths=[2.,3.,4.];D=.7;mean=5.;gradient=np.array([.03,-.02,.04])
    inward=np.array([-D*gradient[0],-D*gradient[1],D*gradient[0],D*gradient[1],-D*gradient[2],D*gradient[2]])[:,None]
    result=reconstruct([mean],[D],[0.],currents(widths,inward),widths)
    expected=mean+np.array([-gradient[0],-1.5*gradient[1],gradient[0],1.5*gradient[1],-2*gradient[2],2*gradient[2]])
    np.testing.assert_allclose(result.face_flux[:,0],expected,atol=1e-14)
    np.testing.assert_allclose(result.minimum_flux,[mean-np.dot(np.abs(gradient),np.array(widths)/2)])
    assert abs(result.uniform_source[0])<1e-15


def test_manufactured_quadratic_zero_removal():
    L=6.;D=.4;c=.01;mean=3.
    # phi=mean+c*(x^2-L^2/12), centred x; inward currents=D*c*L both ends.
    inward=np.array([D*c*L,0,D*c*L,0,0,0])[:,None]
    result=reconstruct([mean],[D],[0.],currents([L,2,3],inward),[L,2,3])
    np.testing.assert_allclose(result.face_flux[[0,2],0],mean+c*L*L/6)
    np.testing.assert_allclose(result.minimum_flux,[mean-c*L*L/12])
    np.testing.assert_allclose(result.uniform_source,[-2*D*c])


@pytest.mark.parametrize('removal',[.001,.08,.7])
def test_independent_neumann_finite_volume_refinement(removal):
    L,D,mean=4.,.7,3.;incoming=[.025,-.015]
    inward=np.array([incoming[0],0,incoming[1],0,0,0])[:,None]
    result=reconstruct([mean],[D],[removal],currents([L,2,3],inward),[L,2,3])
    errors=[]
    for n in [8,16,32,64,128]:
        h=L/n;w=D/h
        A=np.diag(np.full(n,removal*h))
        for i in range(n-1):
            A[i,i]+=w;A[i+1,i+1]+=w;A[i,i+1]-=w;A[i+1,i]-=w
        q=removal*mean-sum(incoming)/L
        b=np.full(n,q*h);b[0]+=incoming[0];b[-1]+=incoming[1]
        phi=np.linalg.solve(A,b)
        assert abs(phi.mean()-mean)<2e-10
        trace=np.array([phi[0]+h*incoming[0]/(2*D),phi[-1]+h*incoming[1]/(2*D)])
        errors.append(np.linalg.norm(trace-result.face_flux[[0,2],0]))
    assert all(a/b>3.8 for a,b in zip(errors,errors[1:]))
    assert errors[-1]<1e-5


def test_small_and_large_optical_size_and_factor_scaling():
    flux=np.array([3.,3.,3.]);D=np.ones(3);r=np.array([0.,1e-16,1e6])
    inward=np.full((6,3),.001)
    result=reconstruct(flux,D,r,currents([2,3,4],inward),[2,3,4])
    np.testing.assert_allclose(result.face_flux[:,0],result.face_flux[:,1],rtol=1e-14)
    assert np.all(np.isfinite(result.face_flux)) and np.all(result.minimum_flux>0)
    factors,_=cuboid_flat_source_factors(result.face_flux*1.2,flux,D,r,currents([2,3,4],inward),[2,3,4])
    np.testing.assert_allclose(factors,1.2)


def test_reject_negative_interior_even_if_faces_positive():
    # Three quadratic contributions can dip below zero at the centre while
    # every face average remains positive.
    inward=np.full((6,1),1.)
    with pytest.raises(ValueError,match='nonpositive'):
        reconstruct([.4],[1.],[0.],currents([2,2,2],inward),[2,2,2])


@pytest.mark.parametrize('which',['mean','diffusion','removal','widths','currents'])
def test_invalid_inputs(which):
    args=[np.array([2.]),np.array([1.]),np.array([.1]),np.zeros((6,1)),np.ones(3)]
    index={'mean':0,'diffusion':1,'removal':2,'currents':3,'widths':4}[which]
    args[index].flat[0]=np.nan
    with pytest.raises(ValueError,match='finite'):reconstruct(*args)


def test_positive_face_means_do_not_imply_positive_affine_field():
    gradient=1.2
    face_densities=np.array([-gradient,-gradient,gradient,gradient,-gradient,gradient])[:,None]
    # Every known exact affine face mean is 0.4 or 1.6; minimum is -0.8.
    with pytest.raises(ValueError,match='nonpositive'):
        reconstruct([1.],[1.],[0.],currents([1,1,1],face_densities),[1,1,1])


def test_declared_uncertainty_margin_rejects_nominal_positive_solution():
    with pytest.raises(ValueError,match='uncertain'):
        reconstruct([.1],[1.],[.2],np.zeros((6,1)),[1,1,1],minimum_flux_margin=.2)
    assert reconstruct([.1],[1.],[.2],np.zeros((6,1)),[1,1,1],minimum_flux_margin=.05).minimum_flux[0] == .1
