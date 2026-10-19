"""Tiny computational eigenpairs, independent dense reference; no native data."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy.linalg import eig
from ndgpu.coupled_moments import CoupledMomentOperator,OrthogonalMesh
from ndgpu.coupled_moment_solver import CoupledMomentEigenSolver
from ndgpu.moment_materials import MomentMaterial

# Reuse the independent synthetic face-unknown fixture, never external XS.
spec=importlib.util.spec_from_file_location('coupled_oracle_fixture',Path(__file__).with_name('test_coupled_moments.py'))
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)


def dense(op):
    F=np.column_stack([op.fission_action(np.eye(op.size)[:,i]) for i in range(op.size)])
    A=fixtures.oracle(op.mesh,op.materials,op.family)
    values,vectors=eig(F,A)
    i=np.argmax(values.real);u=vectors[:,i].real
    normalization=sum(v*np.dot(m.nu_sigma_f,p) for v,m,p in zip(op.mesh.volumes,op.materials,op.scalar_flux(u)))
    return values[i].real,u/normalization


@pytest.mark.parametrize('family',['sp3','sdp1'])
@pytest.mark.parametrize('heterogeneous',[False,True])
def test_dense_eigenpair_and_fresh_initial(family,heterogeneous):
    op=CoupledMomentOperator(fixtures.mesh() if heterogeneous else OrthogonalMesh([2.5],[[0]]),
            [fixtures.material(),fixtures.material(1)] if heterogeneous else [fixtures.material(isotropic=True)],family=family)
    solver=CoupledMomentEigenSolver(op);result=solver.solve()
    expected,u=dense(op)
    assert result.converged and result.numerical_converged and result.reconstructed_nonnegative
    assert result.k_eff==pytest.approx(expected,rel=2e-9)
    np.testing.assert_allclose(result.moments.ravel(),u,rtol=2e-8,atol=1e-10)
    np.testing.assert_allclose(result.power,np.stack([m.kappa_fission*p for m,p in zip(op.materials,result.scalar_flux)]))
    assert result.full_residual<1e-9
    assert not result.provenance['physical_admission'] and not result.provenance['scattering_RHS']
    before=result.moments.copy();initial=np.zeros(op.shape);initial[:,0,:]=2
    initial_before=initial.copy();again=solver.solve(initial=initial)
    np.testing.assert_array_equal(initial,initial_before);np.testing.assert_array_equal(result.moments,before)
    assert again.k_eff==pytest.approx(result.k_eff,rel=1e-9)
    with pytest.raises(ValueError):result.scalar_flux.setflags(write=True)
    with pytest.raises(TypeError):result.provenance['physical_admission']=True
    if not heterogeneous:
        m=op.materials[0];assert result.k_eff==pytest.approx(m.nu_sigma_f@np.linalg.solve(m.loss[0],m.chi),rel=1e-10)


def test_permutation_invariance():
    m=[fixtures.material(),fixtures.material(1)];grid=fixtures.mesh();a=CoupledMomentEigenSolver(CoupledMomentOperator(grid,m)).solve()
    def perm(x):
        return MomentMaterial(x.total[::-1],x.chi[::-1],x.nu_sigma_f[::-1],x.kappa_fission[::-1],{}, {},
            {k:v[::-1,::-1] for k,v in x.loss.items()}, {}, {},dict(x.provenance),{})
    from ndgpu.coupled_moments import InteriorFace,BoundaryFace
    grid=OrthogonalMesh([3,2],[[2],[0]],(InteriorFace(1,0,1.7,(.7,),(1,),.7,1.3),),(BoundaryFace(1,.8,(-.4,),(-1,),.4,'vacuum'),))
    b=CoupledMomentEigenSolver(CoupledMomentOperator(grid,[perm(m[1]),perm(m[0])])).solve()
    assert a.converged and b.converged
    assert a.k_eff==pytest.approx(b.k_eff,rel=1e-10)
    np.testing.assert_allclose(b.scalar_flux,a.scalar_flux[::-1,::-1],rtol=1e-9)


def test_inner_and_outer_failure_and_independent_full_residual():
    op=CoupledMomentOperator(fixtures.mesh(),[fixtures.material(),fixtures.material(1)])
    solver=CoupledMomentEigenSolver(op)
    r=solver.solve(max_inner=1)
    assert not r.converged and r.reason=='inner_failure_or_full_linear_residual'
    assert r.history[0]['inner_iterations']==1
    r=solver.solve(max_outer=1)
    assert not r.converged and r.reason=='outer_iteration_limit'
    # Mutated sparse assembly cannot trick the independent matrix-free residual.
    op.assembled=op.assembled*1.1
    r=solver.solve()
    assert not r.converged and r.reason=='inner_failure_or_full_linear_residual'
    assert r.history[0]['linear_residual']>.01


@pytest.mark.parametrize('initial',[np.ones(3),np.full(4,np.nan),np.array([-1.,1,0,0]),np.array([0.,0,2,2])])
def test_invalid_initial(initial):
    solver=CoupledMomentEigenSolver(CoupledMomentOperator(OrthogonalMesh([1],[[0]]),[fixtures.material()]))
    with pytest.raises(ValueError):solver.solve(initial=initial)


def test_zero_fission_negative_power_and_invalid_controls():
    m=fixtures.material();grid=OrthogonalMesh([1],[[0]])
    zero=MomentMaterial(m.total,m.chi,np.zeros(2),m.kappa_fission,{}, {},dict(m.loss),{}, {},dict(m.provenance),{})
    with pytest.raises(ValueError,match='fission source'):CoupledMomentEigenSolver(CoupledMomentOperator(grid,[zero])).solve()
    negative=MomentMaterial(m.total,m.chi,m.nu_sigma_f,-m.kappa_fission,{}, {},dict(m.loss),{}, {},dict(m.provenance),{})
    r=CoupledMomentEigenSolver(CoupledMomentOperator(grid,[negative])).solve()
    assert not r.converged and not r.reconstructed_nonnegative
    assert r.reason=='negative_reconstructed_scalar_or_power'
    solver=CoupledMomentEigenSolver(CoupledMomentOperator(grid,[m]))
    for kwargs in ({'max_inner':0},{'max_outer':1.5},{'inner_tolerance':np.nan},{'source_tolerance':-1}):
        with pytest.raises(ValueError):solver.solve(**kwargs)


@pytest.mark.parametrize('family',['sp3','sdp1'])
def test_existing_isotropic_common_basis_small_mesh_algebra(family):
    from ndgpu import Grid
    from ndgpu.spn import SDPNGroupOperator,_SPN_C,_SDPN_C
    from ndgpu.coupled_moments import InteriorFace
    m=fixtures.material(isotropic=True)
    grid=OrthogonalMesh([1,1],[[.5],[1.5]],(InteriorFace(0,1,1,(1,),(1,),.5,.5),))
    op=CoupledMomentOperator(grid,[m,m],family=family)
    structured=Grid(shape=(2,1,1),size=(2,1,1))
    u=np.array([[[1.,2.],[-.1,-.2]],[[3.,1.],[-.2,-.1]]])
    expected=np.zeros_like(u)
    for g in range(2):
        constant=lambda x:np.full(structured.shape,x)
        old=SDPNGroupOperator(np,structured,constant(1/(3*m.total[g])),constant(m.total[g]),
            constant(m.loss[0][g,g]),1,bc='reflective',coeffs=_SPN_C if family=='sp3' else _SDPN_C)
        input_group=u[:,:,g].T.reshape(2,2,1,1)
        expected[:,:,g]=old.apply(input_group).reshape(2,2).T
    np.testing.assert_allclose(op.action(u),expected,atol=1e-15)
    result=CoupledMomentEigenSolver(op).solve(initial=u)
    assert result.converged and result.reconstructed_nonnegative
    assert np.any(u[:,1,:]<0)  # signed higher U is permitted


def test_negative_reconstructed_solution_preserved_without_clip():
    m=fixtures.material()
    # Explicit synthetic invalid physical spectrum exercises the output gate;
    # the native reader would reject this input before adapter construction.
    bad=MomentMaterial(m.total,np.array([1.,-.4]),m.nu_sigma_f,m.kappa_fission,{}, {},dict(m.loss),{}, {},dict(m.provenance),{})
    r=CoupledMomentEigenSolver(CoupledMomentOperator(OrthogonalMesh([1],[[0]]),[bad])).solve()
    assert not r.converged and not r.reconstructed_nonnegative
    assert r.reason=='negative_reconstructed_scalar_or_power'
    assert np.any(r.scalar_flux<0) and np.any(r.power<0)
