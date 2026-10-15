"""Opt-in auxiliary-source references; independent dense and analytic controls."""
import numpy as np
import pytest
from ndgpu import Material, CuboidReferenceError
from ndgpu import coupled_neumann_cuboid_reference_modal as reference

TANGENTS=((1,2),(0,2),(1,2),(0,2),(0,1),(0,1))


def material():
    return Material([1.,.6],[.2,.3],[.015,.01],[[0,.04],[.015,0]],[.8,.2])


def dense_problem(m,k,widths,counts,currents,moments):
    """Hand-assembled graph and affine subface integrals, no modal helpers."""
    nx,ny,nz=counts; total=nx*ny*nz; h=np.asarray(widths)/counts; volume=np.prod(h)
    lap=np.zeros((total,total)); rhs=np.zeros((m.n_groups,total))
    index=lambda x,y,z: x+nx*(y+ny*z)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                xyz=np.array([x,y,z]); i=index(x,y,z)
                for a in range(3):
                    if xyz[a]+1 < counts[a]:
                        other=xyz.copy(); other[a]+=1; j=index(*other)
                        for row,col,sign in ((i,i,1),(j,j,1),(i,j,-1),(j,i,-1)):
                            lap[row,col]+=sign/h[a]**2
                for side,axis,at in ((0,0,0),(2,0,nx-1),(1,1,0),(3,1,ny-1),(4,2,0),(5,2,nz-1)):
                    if xyz[axis] != at: continue
                    a,b=TANGENTS[side]; area=widths[a]*widths[b]
                    coordinate=(xyz+.5)*h-np.asarray(widths)/2
                    density=(currents[side]+4*moments[side,0]*coordinate[a]/widths[a]+
                             4*moments[side,1]*coordinate[b]/widths[b])/area
                    rhs[:,i]+=density*volume/h[axis]
    scatter=m.sigma_s.copy(); np.fill_diagonal(scatter,0)
    matrix=np.diag(m.removal)-scatter.T-np.outer(m.chi,m.nu_sigma_f)/k
    operator=volume*(np.kron(np.diag(m.diffusion),lap)+np.kron(matrix,np.eye(total)))
    return operator,rhs,matrix,volume


def test_forced_dense_solution_direct_residuals_mean_current_and_sign_moments():
    m=material(); k=1.1; widths=np.array([1.7,2.3,.9]); counts=(2,4,2)
    currents=np.column_stack((1+.1*np.arange(6),.7+.08*np.arange(6)))
    moments=np.zeros((6,2,2)); moments[:,0]=.03*currents; moments[:,1]=-.02*currents
    options=dict(subdivisions=counts,net_inward_current_sign_moments=moments)
    old=reference(m,k,currents,widths,**options)
    native=old.volume_mean_flux*np.array([1.15,.9])
    result=reference(m,k,currents,widths,native_mean_flux=native,preserve_native_mean=True,**options)
    operator,rhs,matrix,volume=dense_problem(m,k,widths,counts,currents,moments)
    delta=matrix@native-currents.sum(axis=0)/np.prod(widths)
    expected=np.linalg.solve(operator,(rhs+delta[:,None]*volume).ravel()).reshape(2,-1)
    np.testing.assert_allclose(result.cell_mean_flux,expected,rtol=2e-13,atol=2e-14)
    np.testing.assert_allclose(result.cell_mean_flux-old.cell_mean_flux,
        np.broadcast_to((native-old.volume_mean_flux)[:,None],expected.shape),atol=5e-14)
    np.testing.assert_allclose(result.volume_mean_flux,native,rtol=2e-15)
    boundary=result.boundary_profiles
    np.testing.assert_array_equal(boundary.inward_current_density,old.boundary_profiles.inward_current_density)
    density=(boundary.flux-result.cell_mean_flux[:,boundary.owners].T)*m.diffusion/boundary.distances[:,None]
    for side,(a,b) in enumerate(TANGENTS):
        mask=boundary.sides==side
        np.testing.assert_allclose((density[mask]*boundary.areas[mask,None]).sum(axis=0),currents[side],atol=6e-14)
        for tangent,axis in enumerate((a,b)):
            recovered=(density[mask]*boundary.areas[mask,None]*np.sign(boundary.centers[mask,axis]-widths[axis]/2)[:,None]).sum(axis=0)
            np.testing.assert_allclose(recovered,moments[side,tangent],atol=4e-14)
    physical=(operator@result.cell_mean_flux.ravel()).reshape(2,-1)-rhs
    auxiliary=physical-delta[:,None]*volume; d=result.diagnostics
    np.testing.assert_allclose(d['delta_uniform_auxiliary_source_density'],delta)
    np.testing.assert_allclose(d['physical_algebra_residual_l2'],np.linalg.norm(physical),rtol=2e-13)
    assert np.linalg.norm(auxiliary)<1e-12 and d['auxiliary_algebra_residual_l2']<1e-12
    np.testing.assert_allclose(d['physical_integrated_group_balance_residual'],physical.sum(axis=1),atol=1e-13)
    np.testing.assert_allclose(d['physical_integrated_group_balance_residual'],delta*np.prod(widths),atol=1e-13)
    assert d['physical_algebra_residual_l2']>.1 and d['auxiliary_normwise_backward_error']<1e-14
    assert result.native_mean_compatibility['native_mean_imposed'] is True
    assert result.continuous_positivity_certified is False


def test_scalar_manufactured_uniform_source_cosh_refines_at_prescribed_mean():
    m=Material([1.],[.4],[0.]); widths=np.array([2.,1.,1.])
    alpha=np.sqrt(.4); amplitude=.3; offset=2.
    currents=np.zeros((6,1)); currents[[0,2],0]=amplitude*alpha*np.sinh(alpha)
    native=np.array([offset+amplitude*np.sinh(alpha)/alpha]); errors=[]
    for nx in (8,16,32):
        r=reference(m,1,currents,widths,subdivisions=(nx,1,1),native_mean_flux=native,preserve_native_mean=True)
        h=2/nx; center=r.cell_centers[:,0]-1
        exact=offset+amplitude*np.cosh(alpha*center)*np.sinh(alpha*h/2)/(alpha*h/2)
        errors.append(np.max(abs(r.cell_mean_flux[0]-exact)))
        np.testing.assert_allclose(r.volume_mean_flux,native,rtol=2e-15)
        np.testing.assert_allclose(r.diagnostics['delta_uniform_auxiliary_source_density'],[.4*offset],atol=3e-16)
    assert errors[1]<.4*errors[0] and errors[2]<.4*errors[1]


def test_default_false_is_exactly_unchanged_with_and_without_native_mean():
    m=material(); currents=np.full((6,2),.1)
    for native in (None,[2.,3.]):
        options=dict(subdivisions=(3,5,2),native_mean_flux=native)
        omitted=reference(m,1.1,currents,[1.,2.,3.],**options)
        explicit=reference(m,1.1,currents,[1.,2.,3.],preserve_native_mean=False,**options)
        for key in ('cell_mean_flux','face_mean_flux','volume_mean_flux'):
            np.testing.assert_array_equal(getattr(omitted,key),getattr(explicit,key))
        np.testing.assert_array_equal(omitted.boundary_profiles.flux,explicit.boundary_profiles.flux)
        assert omitted.diagnostics['normwise_backward_error']==explicit.diagnostics['normwise_backward_error']


def test_negative_unforced_intermediate_does_not_reject_positive_final_field():
    m=Material([1.],[1.],[0.]); currents=np.full((6,1),-.02)
    with pytest.raises(CuboidReferenceError,match='Nonpositive'):
        reference(m,1,currents,[1,1,1],subdivisions=4)
    result=reference(m,1,currents,[1,1,1],subdivisions=4,native_mean_flux=[2.],preserve_native_mean=True)
    assert result.cell_mean_flux.min()>1.9 and result.boundary_profiles.flux.min()>1.9
    np.testing.assert_allclose(result.volume_mean_flux,[2.],atol=3e-16)


def test_final_negative_trace_and_nonconstant_resonance_still_reject():
    with pytest.raises(CuboidReferenceError,match='Nonpositive'):
        reference(Material([1],[1],[0]),1,np.array([-5,0,6,0,0,0])[:,None],[1,1,1],
            subdivisions=1,native_mean_flux=[1.],preserve_native_mean=True)
    for production in (2.1,.1):  # nonconstant resonance; singular uniform block
        with pytest.raises(CuboidReferenceError,match='singular|condition_limit'):
            reference(Material([1],[.1],[production]),1,np.zeros((6,1)),[2,1,1],subdivisions=(2,1,1),
                native_mean_flux=[1.],preserve_native_mean=True)


@pytest.mark.parametrize('native',[None,[-1.],[0.],[np.nan],[np.inf],[1.,2.]])
def test_missing_invalid_native_rejected(native):
    with pytest.raises(ValueError,match='native_mean_flux'):
        reference(Material([1],[1],[0]),1,np.zeros((6,1)),[1,1,1],native_mean_flux=native,preserve_native_mean=True)


@pytest.mark.parametrize('flag',[1,0,None,'true',np.bool_(True)])
def test_flag_must_be_explicit_python_bool(flag):
    with pytest.raises(ValueError,match='preserve_native_mean'):
        reference(Material([1],[1],[0]),1,np.zeros((6,1)),[1,1,1],native_mean_flux=[1.],preserve_native_mean=flag)
