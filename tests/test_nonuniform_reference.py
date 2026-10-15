"""Independent small FV/dense systems, analytic refinement and failure controls."""
import copy
import numpy as np
import pytest
from ndgpu import Material, CuboidReferenceError
from ndgpu import coupled_neumann_cuboid_reference_modal as uniform_reference
from ndgpu.nonuniform_reference import coupled_neumann_tensor_reference as reference
from ndgpu.mesh import MeshGrid, MeshGroupOperator
from ndgpu.nodal_mesh import build_rectilinear_mesh_3d


def boundary_shapes(edges,g):
    nx,ny,nz=[len(e)-1 for e in edges]
    return [(nz,ny,g),(nz,nx,g),(nz,ny,g),(nz,nx,g),(ny,nx,g),(ny,nx,g)]


def zero_currents(edges,g):
    return [np.zeros(shape) for shape in boundary_shapes(edges,g)]


def test_arbitrary_subfaces_against_independent_actual_fv_matrix_and_condition_bounds():
    edges=[np.array([-.2,.1,.9,1.7]),np.array([-1.,-.2,1.]),np.array([2.,2.4,3.3])]
    m=Material([1.,.7],[.3,.4],[.02,.01],[[0,.03],[.01,0]],[.9,.1]);native=np.array([3.,2.]);k=1.1
    rng=np.random.default_rng(4);densities=[rng.normal(0,.015,size=s) for s in boundary_shapes(edges,2)]
    snapshot=copy.deepcopy((edges,densities,native))
    result=reference(m,k,edges,native,inward_current_density=densities)
    shape=tuple(len(e)-1 for e in edges[::-1])
    mesh,topology=build_rectilinear_mesh_3d(*edges,active_mask=np.ones(shape,bool),tags=np.zeros(shape,int))
    grid=MeshGrid(mesh,'reflective');c=mesh.n_cells
    operators=[MeshGroupOperator(np,grid,np.full(grid.shape,m.diffusion[g]),np.full(grid.shape,m.removal[g])) for g in range(2)]
    scatter=m.sigma_s.copy();np.fill_diagonal(scatter,0)
    coupling=scatter.T+np.outer(m.chi,m.nu_sigma_f)/k
    matrix=np.diag(m.removal)-coupling
    dense=np.empty((2*c,2*c));basis=np.eye(2*c)
    for column in range(2*c):
        v=basis[:,column].reshape(2,c)
        dense[:,column]=(np.array([operators[g].apply(v[g].reshape(grid.shape)).ravel() for g in range(2)])-mesh.area*(coupling@v)).ravel()
    rhs=np.zeros((2,c));face_counters=np.zeros(6,int)
    # Factory face order differs across sides but each side keeps x/y/z tensor order.
    for (owner,area,distance),side in zip(mesh.bfaces,topology['boundary_sides']):
        density=densities[side].reshape(-1,2)[face_counters[side]];face_counters[side]+=1
        rhs[:,owner]+=area*density
    total=rhs.sum(axis=1);delta=matrix@native-total/mesh.area.sum()
    target_rhs=rhs+delta[:,None]*mesh.area
    expected=np.linalg.solve(dense,target_rhs.ravel()).reshape(2,c)
    np.testing.assert_allclose(result['cell_mean_flux'],expected,rtol=2e-13,atol=2e-14)
    np.testing.assert_allclose(result['cell_centers'],mesh.centroid,atol=3e-16)
    np.testing.assert_allclose(result['cell_volumes'],mesh.area,rtol=3e-16)
    physical=(dense@result['cell_mean_flux'].ravel()).reshape(2,c)-rhs
    np.testing.assert_allclose(result['physical_integrated_residual'],physical,atol=2e-14)
    np.testing.assert_allclose(result['auxiliary_integrated_residual'],physical-delta[:,None]*mesh.area,atol=2e-14)
    d=result['diagnostics'];singular=np.linalg.svd(dense,compute_uv=False)
    assert singular[-1]>=d['operator_sigma_min_lower_bound']*(1-1e-13)
    assert singular[0]<=d['operator_sigma_max_upper_bound']*(1+1e-13)
    assert singular[0]>=d['operator_sigma_max_lower_bound']*(1-1e-13)
    assert singular[0]/singular[-1]<=d['condition_number_2_upper_bound']*(1+1e-13)
    actual_backward=np.linalg.norm(result['auxiliary_integrated_residual'])/(singular[0]*np.linalg.norm(expected)+np.linalg.norm(target_rhs))
    assert actual_backward<=d['normwise_backward_error_upper_bound']*(1+1e-13)
    np.testing.assert_allclose(result['volume_mean_flux'],native,rtol=2e-14)
    np.testing.assert_allclose(d['physical_integrated_group_balance_residual'],delta*mesh.area.sum(),atol=2e-14)
    assert d['boundary_current_and_sign_moment_recovery_passed']
    for old,new in zip(snapshot[0],edges):np.testing.assert_array_equal(old,new)
    for old,new in zip(snapshot[1],densities):np.testing.assert_array_equal(old,new)
    np.testing.assert_array_equal(snapshot[2],native)
    integrals=[density*f['areas'][...,None] for density,f in zip(densities,result['boundary_profiles'])]
    integrated=reference(m,k,edges,native,inward_current_integral=integrals)
    np.testing.assert_allclose(integrated['cell_mean_flux'],result['cell_mean_flux'],rtol=2e-14,atol=2e-14)
    for face in result['boundary_profiles']:
        side=face['side'];axis=(0,1,0,1,2,2)[side]
        np.testing.assert_array_equal(face['centers'][...,axis],np.full(face['areas'].shape,edges[axis][-1 if side in (2,3,5) else 0]))
        trace=(result['cell_mean_flux'][:,face['owners']].transpose(1,2,0)+face['distances'][...,None]*face['inward_current_density']/m.diffusion)
        np.testing.assert_array_equal(face['flux'],trace)


def test_uniform_grid_agrees_with_original_native_mean_api_fields_and_traces():
    widths=np.array([1.7,2.3,.9]);counts=(3,2,4)
    edges=[np.linspace(0,w,n+1) for w,n in zip(widths,counts)]
    m=Material([1.,.6],[.2,.3],[.015,.01],[[0,.04],[.015,0]],[.8,.2])
    totals=np.full((6,2),.1);native=[3.,2.]
    areas=[widths[1]*widths[2],widths[0]*widths[2],widths[1]*widths[2],widths[0]*widths[2],widths[0]*widths[1],widths[0]*widths[1]]
    j=[np.broadcast_to(totals[s]/areas[s],shape).copy() for s,shape in enumerate(boundary_shapes(edges,2))]
    old=uniform_reference(m,1.1,totals,widths,subdivisions=counts,native_mean_flux=native,preserve_native_mean=True)
    new=reference(m,1.1,edges,native,inward_current_density=j)
    for key in ('cell_mean_flux','face_mean_flux','volume_mean_flux'):
        np.testing.assert_allclose(new[key],getattr(old,key),rtol=2e-13,atol=2e-14)
    for face in new['boundary_profiles']:
        mask=old.boundary_profiles.sides==face['side']
        np.testing.assert_allclose(face['flux'].reshape(-1,2),old.boundary_profiles.flux[mask],rtol=2e-13,atol=2e-14)
    np.testing.assert_allclose(new['diagnostics']['delta_uniform_auxiliary_source_density'],old.diagnostics['delta_uniform_auxiliary_source_density'],rtol=2e-14,atol=2e-14)


def test_nonuniform_manufactured_cosh_cell_means_refine_with_fixed_uniform_source():
    m=Material([1.],[.4],[0.]);alpha=np.sqrt(.4);offset=2.;amplitude=.3
    native=[offset+amplitude*np.sinh(alpha)/alpha];errors=[]
    for n in (8,16,32):
        x=2*np.linspace(0,1,n+1)**1.3;edges=[x,np.array([0.,1.]),np.array([0.,1.])]
        j=zero_currents(edges,1);j[0][:]=j[2][:]=amplitude*alpha*np.sinh(alpha)
        result=reference(m,1,edges,native,inward_current_density=j)
        exact=offset+amplitude*(np.sinh(alpha*(x[1:]-1))-np.sinh(alpha*(x[:-1]-1)))/(alpha*np.diff(x))
        errors.append(np.sqrt(np.sum((result['cell_mean_flux'][0]-exact)**2*np.diff(x))/2))
        np.testing.assert_allclose(result['volume_mean_flux'],native,rtol=1e-13)
        np.testing.assert_allclose(result['diagnostics']['delta_uniform_auxiliary_source_density'],[.8],atol=3e-16)
    assert errors[1]<.4*errors[0] and errors[2]<.4*errors[1]


def test_exact_sign_integration_for_subface_crossing_native_midpoint():
    edges=[np.array([0.,1.]),np.array([0.,.2,.7,1.]),np.array([0.,1.])]
    j=zero_currents(edges,1);j[0][0,:,0]=[.01,.02,.03]
    r=reference(Material([1],[1],[0]),1,edges,[2.],inward_current_density=j)
    expected=.01*(-.2)+.02*(-.1)+.03*.3
    np.testing.assert_allclose(r['diagnostics']['input_integrated_sign_moments'][0,0,0],expected,atol=1e-17)
    assert r['diagnostics']['boundary_current_and_sign_moment_recovery_passed']


@pytest.mark.parametrize('edges',[[[0,1],[0,0],[0,1]],[[0,1],[0,np.nan],[0,1]],[[0,1],[0,1]],[[0,1],[1,0],[0,1]]])
def test_invalid_edges_rejected(edges):
    with pytest.raises(ValueError):reference(Material([1],[1],[0]),1,edges,[1.],inward_current_density=[])


@pytest.mark.parametrize('kind',['missing','both','shape','nonfinite','mean'])
def test_invalid_boundary_or_mean_rejected(kind):
    edges=[np.array([0.,1.])]*3;j=zero_currents(edges,1);kwargs=dict(inward_current_density=j);native=[1.]
    if kind=='missing':kwargs={}
    if kind=='both':kwargs['inward_current_integral']=j
    if kind=='shape':j[0]=np.zeros((1,1))
    if kind=='nonfinite':j[0][0,0,0]=np.inf
    if kind=='mean':native=[0.]
    with pytest.raises(ValueError):reference(Material([1],[1],[0]),1,edges,native,**kwargs)


def test_singular_uniform_and_nonconstant_blocks_rejected_despite_prescribed_mean():
    edges=[np.array([0.,.6,2.]),np.array([0.,1.]),np.array([0.,1.])];j=zero_currents(edges,1)
    for production in (.1,.1+1/.6+1/1.4):
        with pytest.raises(CuboidReferenceError,match='conditioned|Singular'):
            reference(Material([1],[.1],[production]),1,edges,[1.],inward_current_density=j)


def test_final_cell_and_trace_positivity_rejected_without_clipping():
    edges=[np.array([0.,1.])]*3;j=zero_currents(edges,1);j[0][:]=-3
    with pytest.raises(CuboidReferenceError,match='Nonpositive') as failure:
        reference(Material([1],[1],[0]),1,edges,[1.],inward_current_density=j)
    assert failure.value.diagnostics['minimum_cell_mean_flux_per_group'][0]>0
    assert failure.value.diagnostics['minimum_boundary_trace_flux_per_group'][0]<0
    edges=[np.array([0.,.5,1.]),np.array([0.,1.]),np.array([0.,1.])];j=zero_currents(edges,1);j[0][:]=-10;j[2][:]=10
    with pytest.raises(CuboidReferenceError,match='Nonpositive') as failure:
        reference(Material([1],[.1],[0]),1,edges,[1.],inward_current_density=j)
    assert failure.value.diagnostics['minimum_cell_mean_flux_per_group'][0]<0
