"""Tiny synthetic native SP7 checks; no external XS or physical admission."""
from dataclasses import replace

import numpy as np
import pytest

from garnet.coupled_moments import CoupledMomentOperator, _family_tables
from garnet.corrected_native_moments import CorrectedNativeMomentOperator
from garnet.coupled_moment_solver import CoupledMomentEigenSolver
from garnet.mesh import MeshGrid
from garnet.moment_materials import MomentMaterial
from garnet.spn import _SPN_C, _SPN_G, _SDPN_C, _SDPN_G
from test_corrected_mesh import mesh_grid
from test_corrected_native_moments import orthogonal, partial

WEIGHTS=np.array([1.,-2/3,8/15,-16/35])


def material(which=0,groups=2):
    total=np.arange(groups)*.3+1.3+.1*which
    rng=np.random.default_rng(400+which)
    losses={0:np.diag(np.arange(groups)*.2+.5+.05*which)-.03*(np.ones((groups,groups))-np.eye(groups))}
    for order in range(1,8):
        losses[order]=np.diag(total+.03*order)+rng.uniform(-.035,.035,(groups,groups))
    chi=np.arange(groups,0,-1,dtype=float);chi/=chi.sum()
    return MomentMaterial(total,chi,np.arange(groups)*.05+.3,np.arange(groups)*5.+20,
        {},{},losses,{}, {},{'orientation_contract':'from-major'}, {})


def operator(corrected,*,bc='reflective',groups=2,mats=None,skew=0.):
    mesh=mesh_grid(2,skew)
    mats=mats if mats is not None else [material(i%2,groups) for i in range(mesh.n_cells)]
    if corrected:return CorrectedNativeMomentOperator(MeshGrid(mesh,bc),mats,family='sp7')
    return CoupledMomentOperator(orthogonal(mesh,bc),mats,family='sp7')


@pytest.mark.parametrize('corrected',[False,True])
def test_table_identity_four_moment_scalar_and_fission(corrected):
    op=operator(corrected)
    assert op.shape==(4,4,2) and op.required_orders==tuple(range(8))
    tables,boundary=_family_tables('sp7')
    np.testing.assert_array_equal(tables,np.asarray(_SPN_C[3]))
    np.testing.assert_array_equal(boundary,np.asarray(_SPN_G[3]))
    for actual,expected in zip(op.C,_SPN_C[3]):np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(op.B,np.kron(_SPN_G[3],np.eye(2)))
    np.testing.assert_array_equal(op.c1[0],WEIGHTS)
    for coefficient in op.C[1:]:np.testing.assert_array_equal(coefficient[0],np.zeros(4))
    state=np.random.default_rng(7).normal(size=op.shape)
    expected=state[:,0]-2*state[:,1]/3+8*state[:,2]/15-16*state[:,3]/35
    np.testing.assert_array_equal(op.scalar_flux(state),expected)
    # Isolate the fourth moment so omitting it cannot pass by cancellation.
    state[:,:3]=0;state[:,3]=1.
    np.testing.assert_array_equal(op.scalar_flux(state),np.full((4,2),-16/35))
    volumes=op.volumes if corrected else op.mesh.volumes
    expected=np.stack([v*WEIGHTS[:,None]*m.chi[None,:]*(m.nu_sigma_f@p)
                       for v,m,p in zip(volumes,op.materials,op.scalar_flux(state))])
    np.testing.assert_allclose(op.fission_action(state),expected,atol=2e-16)


@pytest.mark.parametrize('family,order',[('sp3',1),('sdp1',1),('sp5',2)])
def test_existing_family_tables_and_scalar_arithmetic_unchanged(family,order):
    c,g=_family_tables(family)
    np.testing.assert_array_equal(c,(_SDPN_C if family=='sdp1' else _SPN_C)[order])
    np.testing.assert_array_equal(g,(_SDPN_G if family=='sdp1' else _SPN_G)[order])
    mesh=mesh_grid(2,0.)
    op=CoupledMomentOperator(orthogonal(mesh,'reflective'),[material()]*4,family=family)
    x=np.random.default_rng(33).normal(size=op.shape)
    expected=x[:,0,:]-2*x[:,1,:]/3
    if family=='sp5':expected=expected+8*x[:,2,:]/15
    np.testing.assert_array_equal(op.scalar_flux(x),expected)


@pytest.mark.parametrize('corrected',[False,True])
def test_one_group_reflective_analytic_eigenvalue(corrected):
    m=material(groups=1)
    op=operator(corrected,mats=[m]*4,groups=1)
    result=CoupledMomentEigenSolver(op).solve(preconditioner='ilu')
    # Infinite homogeneous reflection removes leakage: k = nuSigmaF / L0.
    expected=float(m.nu_sigma_f[0]*m.chi[0]/m.loss[0][0,0])
    assert result.converged and result.k_eff==pytest.approx(expected,rel=2e-10)
    assert result.full_residual<=1e-9 and np.min(result.scalar_flux)>0
    assert result.provenance['moments_per_cell']==4
    assert result.provenance['required_orders']==tuple(range(8))
    assert result.provenance['family']=='sp7'
    assert result.provenance['fallback'] is False and result.provenance['regularization'] is False


@pytest.mark.parametrize('corrected',[False,True])
def test_missing_native_order_seven_refused(corrected):
    m=material();loss=dict(m.loss);del loss[7]
    with pytest.raises(ValueError,match=r'0\.\.7'):
        operator(corrected,mats=[replace(m,loss=loss)]*4)


@pytest.mark.parametrize('corrected',[False,True])
def test_mixed_energy_anisotropic_action_and_physical_balance(corrected):
    op=operator(corrected,skew=.31 if corrected else 0.)
    x=np.random.default_rng(27).normal(size=op.shape)
    volumes=op.volumes if corrected else op.mesh.volumes
    assert np.linalg.norm(op.diffusion[0]@op.diffusion[1]-op.diffusion[1]@op.diffusion[0])>1e-6
    np.testing.assert_allclose(op.assembled@x.ravel(),op.action(x).ravel(),atol=4e-14)
    # Independently form every even-loss reaction and odd-loss diffusion block.
    reaction=np.stack([v*sum(np.asarray(c)@x[i]@m.loss[2*k].T for k,c in enumerate(_SPN_C[3]))
                       for i,(v,m) in enumerate(zip(volumes,op.materials))])
    for D,m in zip(op.diffusion,op.materials):
        for k in range(4):
            np.testing.assert_allclose(D[k*2:(k+1)*2,k*2:(k+1)*2],np.linalg.solve((4*k+3)*m.loss[2*k+1],np.eye(2)),atol=3e-16)
    scalar=x[:,0]-2*x[:,1]/3+8*x[:,2]/15-16*x[:,3]/35
    physical=np.stack([v*(m.loss[0]@p) for v,m,p in zip(volumes,op.materials,scalar)])
    np.testing.assert_allclose(reaction[:,0],physical,atol=3e-16)
    accumulated=reaction.copy()
    if corrected:
        interior,boundary=op.face_currents(x)
        np.testing.assert_array_equal(boundary,np.zeros_like(boundary))
        for (i,j,*_),current in zip(op.grid.mesh.faces,interior):
            accumulated[i]+=current;accumulated[j]-=current
    else:
        for i,j,t in op.faces:
            current=(t@(x[i]-x[j]).ravel()).reshape(4,2)
            accumulated[i]+=current;accumulated[j]-=current
    np.testing.assert_allclose(op.action(x),accumulated,atol=3e-14)
    # Interior currents cancel for each moment/group; the zeroth sum is L0 phi.
    np.testing.assert_allclose(op.action(x)[:,0].sum(axis=0),physical.sum(axis=0),atol=3e-14)


def test_corrected_sp7_affine_interface_currents_independent():
    mesh=partial(mesh_grid(4,.35))
    normal=np.array([1.,-.35])/np.hypot(1,.35)
    tangent=np.array([.35,1.])/np.hypot(1,.35)
    signed=mesh.centroid@normal-.5/np.hypot(1,.35)
    pair=[material(),material(1)];mats=[pair[int(s>0)] for s in signed]
    op=CorrectedNativeMomentOperator(MeshGrid(mesh,'reflective'),mats,family='sp7')
    current=np.linspace(-.3,.7,8);ht=np.linspace(.2,-.4,8);base=np.linspace(1.,2.,8)
    gradients=np.stack([np.linalg.solve(D,current)[:,None]*normal+ht[:,None]*tangent for D in op.diffusion])
    state=np.stack([base+np.linalg.solve(D,current)*sn+ht*tn for D,sn,tn in zip(op.diffusion,signed,mesh.centroid@tangent)])
    np.testing.assert_allclose(op.gradients(state.ravel()),gradients,atol=6e-13)
    currents,boundary=op.face_currents(state.ravel());checked=0
    for k,((i,j,area,_),center,n) in enumerate(zip(mesh.faces,mesh.face_centers,mesh.face_normals)):
        np.testing.assert_allclose(currents[k].ravel(),-area*op.diffusion[i]@gradients[i]@n,atol=6e-14)
        if mats[i] is not mats[j] and signed[i]*signed[j]<0:
            np.testing.assert_allclose(state[i]+gradients[i]@(center-mesh.centroid[i]),state[j]+gradients[j]@(center-mesh.centroid[j]),atol=6e-14)
            checked+=1
    assert checked==4
    assert boundary.shape[0]==0
    np.testing.assert_allclose(op.assembled@state.ravel(),op.action(state.ravel()),atol=6e-14)


@pytest.mark.parametrize('family',['sp1','sp9','SP7','unknown',None])
def test_invalid_families_still_refused(family):
    with pytest.raises(ValueError,match='Use sp3'):_family_tables(family)
