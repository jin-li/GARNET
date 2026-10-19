import numpy as np
import pytest
from ndgpu.nem import NEMDiffusionSolver
from ndgpu.nem_higher_order import NEMHigherOrderDiffusionSolver
from ndgpu.nem_geometry import CuboidNodalGrid
from ndgpu.nem_reference import reconstruct_nem_reference, qualify_nem_reference_factors
from test_nem import tensor_grid, materials


@pytest.mark.parametrize('degree',[None,4,8,16])
@pytest.mark.parametrize('policy',['one_sided','primary_rule'])
def test_reference_recovers_complete_general_kernel_eigenstate(degree,policy):
    base=tensor_grid(([0.,.7,1.9],[0.,1.3],[0.,.8]))
    grid=CuboidNodalGrid(base.bounds,leakage_boundary=policy)
    mat=materials(grid,2)
    kernel=np.array([[[.83,.51],[.17,.49]],[[.62,.37],[.38,.63]]])*mat[3][:,None,:]
    factors=np.ones((2,6,2));factors[0,2]=[.91,1.07];factors[1,0]=[1.05,.94]
    cls=NEMDiffusionSolver if degree is None else NEMHigherOrderDiffusionSolver
    kwargs={} if degree is None else {'degree':degree}
    solver=cls(grid,*mat,factors,fission_matrix=kernel,**kwargs)
    fwd=solver.solve_eigenvalue(rtol=1e-12,max_iterations=2000)
    got=reconstruct_nem_reference(solver,k_eff=fwd.k_eff,means=fwd.means,currents=fwd.currents)
    inward=-fwd.currents[grid.node_faces]*grid.normal_signs[None,:,None]*grid.node_face_areas[:,:,None]
    native=reconstruct_nem_reference(solver,k_eff=fwd.k_eff,means=fwd.means,
                                     net_inward_integrated=inward)
    np.testing.assert_allclose(native.face_means,got.face_means,rtol=1e-12,atol=1e-13)
    np.testing.assert_allclose(got.face_means,fwd.side_face_means,rtol=2e-9,atol=1e-11)
    np.testing.assert_allclose(got.moments,fwd.moments,rtol=2e-8,atol=1e-11)
    np.testing.assert_allclose(got.auxiliary_mean_source,0,atol=2e-11)
    np.testing.assert_array_equal(got.production,kernel)
    # A factor connects the two homogeneous side traces, not their raw values.
    np.testing.assert_allclose(got.face_means[0,2]*factors[0,2],
                               got.face_means[1,0]*factors[1,0],rtol=2e-9)
    assert got.diagnostics['auxiliary_source_is_physical'] is False
    assert got.diagnostics['ADF_qualification'] is False
    for array in (got.face_means,got.moments,got.coefficients,got.production,got.coupling):
        assert not array.flags.writeable


def test_reference_reports_prescribed_mean_balance_without_changing_production():
    grid=tensor_grid(([0.,1.],[0.,1.],[0.,1.]))
    solver=NEMDiffusionSolver(grid,*materials(grid,2))
    mean=np.array([[1.,2.]])
    got=reconstruct_nem_reference(solver,k_eff=1.3,means=mean,currents=np.zeros((solver.n_faces,2)))
    expected=(solver.R[0]-solver.fission[0]/1.3)@mean[0]
    np.testing.assert_allclose(got.auxiliary_mean_source[0],expected)
    np.testing.assert_allclose(got.face_means,np.broadcast_to(mean[:,None,:],(1,6,2)),atol=1e-13)
    np.testing.assert_array_equal(got.production,solver.fission)
    assert np.max(abs(got.auxiliary_mean_source))>0


def test_native_shared_face_disagreement_is_preserved_and_reported():
    grid=tensor_grid(([0.,.7,1.9],[0.,1.3],[0.,.8]))
    solver=NEMDiffusionSolver(grid,*materials(grid,2))
    inward=np.zeros((2,6,2))
    inward[0,2]=-.03*grid.node_face_areas[0,2]
    inward[1,0]=.05*grid.node_face_areas[1,0]
    got=reconstruct_nem_reference(solver,k_eff=1.,means=np.ones((2,2)),
                                 net_inward_integrated=inward)
    np.testing.assert_allclose(got.diagnostics['shared_face_current_density_max_abs_disagreement'],.02)
    # Independent nodal mean balance retains each side; averaging would change it.
    reaction=np.einsum('ngh,nh->ng',solver.R-solver.fission,np.ones((2,2)))
    np.testing.assert_allclose(got.auxiliary_mean_source[:,0]-reaction[:,0],[.03/.7,-.05/1.2])


def test_negative_reference_profiles_are_retained_without_ADF_admission():
    grid=tensor_grid(([0.,1.],[0.,1.],[0.,1.]))
    solver=NEMDiffusionSolver(grid,*materials(grid,2))
    got=reconstruct_nem_reference(solver,k_eff=1.,means=np.ones((1,2)),
                                 currents=np.ones((solver.n_faces,2))*100)
    assert got.diagnostics['negative_profile_count']>0
    assert np.min(got.face_means)<0
    assert got.diagnostics['ADF_qualification'] is False


def test_general_kernel_reflector_inclusive_factor_prescription_and_rejections():
    base=tensor_grid(([0.,.7,1.9,2.8],[0.,.8,2.,3.],[0.,.9,2.2,3.4]))
    grid=CuboidNodalGrid(base.bounds,leakage_boundary='primary_rule')
    n=grid.n_nodes
    mat=materials(grid,2)
    fuel=np.all(grid.neighbors>=0,axis=1)
    assert fuel.sum()==1
    mat[3][~fuel]=0
    kernel=np.broadcast_to(np.array([[.8,.5],[.2,.5]]),(n,2,2))*mat[3][:,None,:]
    solver=NEMHigherOrderDiffusionSolver(grid,*mat,fission_matrix=kernel,degree=8)
    means=1+np.arange(n)[:,None]*np.array([[.001,.002]])
    currents=np.sin(np.arange(solver.n_faces*2)).reshape(-1,2)*1e-4
    reference=reconstruct_nem_reference(solver,k_eff=1.,means=means,currents=currents)
    internal=grid.neighbors>=0
    expected=np.ones((n,6,2));expected[internal]=1.03
    het=reference.face_means*expected
    het[~internal]=-.1  # Exterior ratios have no role in this prescription.
    args=dict(k_eff=1.,means=means,currents=currents,heterogeneous_faces=het,
              fuel_mask=fuel,degrees=(4,6,8))
    got=qualify_nem_reference_factors(solver,**args)
    np.testing.assert_allclose(got.factors,expected,rtol=1e-15)
    assert not got.factors.flags.writeable
    assert np.max(got.face_change_over_mean)>0
    assert np.max(got.factor_relative_change_bound)>0
    assert np.max(abs(got.references[-1].auxiliary_mean_source))>0
    assert got.corrected_trace_relative_error<=3e-16
    with pytest.raises(ValueError,match='order'):
        qualify_nem_reference_factors(solver,**args,order_tolerance=1e-20)
    bad=dict(args,degrees=(4,6,16))
    with pytest.raises(ValueError,match='degree'):
        qualify_nem_reference_factors(solver,**bad)
    bad=dict(args,heterogeneous_faces=het.copy())
    bad['heterogeneous_faces'][fuel,0]=-1
    with pytest.raises(ValueError,match='heterogeneous'):
        qualify_nem_reference_factors(solver,**bad)


@pytest.mark.parametrize('fuel', [True,False])
def test_factor_qualification_rejects_negative_fuel_or_reflector_denominator(fuel):
    grid=tensor_grid(([0.,.7,1.9],[0.,1.3],[0.,.8]))
    solver=NEMHigherOrderDiffusionSolver(grid,*materials(grid,2),degree=8)
    with pytest.raises(ValueError,match='fuel reference' if fuel else 'reflector positivity'):
        qualify_nem_reference_factors(solver,k_eff=1.,means=np.ones((2,2)),
            currents=np.ones((solver.n_faces,2))*100,heterogeneous_faces=np.ones((2,6,2)),
            fuel_mask=np.full(2,fuel),degrees=(4,6,8),order_tolerance=1e6)


@pytest.mark.parametrize('bad',['k','mean','current','changed'])
def test_reference_rejects_invalid_inputs_and_changed_cached_solver(bad):
    grid=tensor_grid(([0.,1.],[0.,1.],[0.,1.]))
    solver=NEMDiffusionSolver(grid,*materials(grid,2))
    k=1.;mean=np.ones((1,2));current=np.zeros((solver.n_faces,2))
    if bad=='k': k=0
    if bad=='mean': mean[0,0]=-1
    if bad=='current': current=np.ones((1,6,2))
    if bad=='changed': solver.fission=solver.fission*1.01
    with pytest.raises(ValueError): reconstruct_nem_reference(solver,k_eff=k,means=mean,currents=current)
