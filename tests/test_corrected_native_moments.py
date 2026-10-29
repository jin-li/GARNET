"""Synthetic numerical checks only: no native-data physical admission."""
from dataclasses import replace
from itertools import product
import numpy as np
import pytest
from scipy.linalg import eigvals
from garnet.mesh import Mesh, MeshGrid
from garnet.moment_materials import MomentMaterial
from garnet.corrected_native_moments import CorrectedNativeMomentOperator as Operator
from garnet.coupled_moments import CoupledMomentOperator, OrthogonalMesh, InteriorFace, BoundaryFace
from garnet.coupled_moment_solver import CoupledMomentEigenSolver
from garnet.corrected_mesh_moments import CorrectedMeshMomentOperator
from garnet.spn import _SPN_C, _SPN_G, _SDPN_C, _SDPN_G
from test_corrected_mesh import mesh_grid
from test_coupled_moments import material, oracle


def op(mesh, mats=None, family='sp3', bc='vacuum', **kwargs):
    return Operator(MeshGrid(mesh,bc), mats or [material()]*mesh.n_cells, family=family, **kwargs)


def orthogonal(mesh, bc='vacuum'):
    return OrthogonalMesh(mesh.area, mesh.centroid,
        tuple(InteriorFace(int(i),int(j),a,tuple(c),tuple(n),*d) for (i,j,a,_),c,n,d in
              zip(mesh.faces,mesh.face_centers,mesh.face_normals,mesh.face_distances)),
        tuple(BoundaryFace(int(i),a,tuple(c),tuple(n),d,bc) for (i,a,d),c,n in
              zip(mesh.bfaces,mesh.bface_centers,mesh.bface_normals)))


def partial(mesh, keep=()):
    keep = np.asarray(keep,dtype=int)
    return replace(mesh,bfaces=[mesh.bfaces[k] for k in keep],bface_centers=mesh.bface_centers[keep],
                   bface_normals=mesh.bface_normals[keep],bface_tags=mesh.bface_tags[keep])


@pytest.mark.parametrize('family',['sp3','sdp1'])
@pytest.mark.parametrize('bc',['vacuum','reflective'])
def test_orthogonal_native_full_equality(family,bc):
    mesh=mesh_grid(2,0);mats=[material(i%2) for i in range(mesh.n_cells)]
    a=op(mesh,mats,family,bc);b=CoupledMomentOperator(orthogonal(mesh,bc),mats,family=family)
    np.testing.assert_allclose(a.assembled.toarray(),b.assembled.toarray(),atol=1e-14)
    np.testing.assert_allclose(a.assembled.toarray(),oracle(orthogonal(mesh,bc),mats,family),atol=1e-14)
    u=np.arange(a.size)*.23-1
    np.testing.assert_allclose(a.action(u),b.action(u),atol=1e-14)
    np.testing.assert_allclose(a.fission_action(u),b.fission_action(u),atol=1e-14)


def representable(g,i):
    total=np.arange(g)*.3+1.2+.02*i
    loss={0:np.eye(g)*(.23+.01*i),1:np.diag(total*.8),2:np.diag(total),3:np.diag(total)}
    if g==2:loss[0][0,1]=-.03;loss[0][1,0]=-.07
    return MomentMaterial(total,np.ones(g)/g,np.full(g,.3),np.ones(g)*20,{}, {},loss,{}, {},
                          {'orientation_contract':'from-major'}, {})


@pytest.mark.parametrize('family',['sp3','sdp1'])
@pytest.mark.parametrize('g',[1,2])
def test_representable_legacy_reduction_nonaffine_skew_vacuum(family,g):
    mesh=mesh_grid(3,.41);mats=[representable(g,i) for i in range(mesh.n_cells)]
    a=op(mesh,mats,family);n=mesh.n_cells
    c,G=(_SPN_C,_SPN_G) if family=='sp3' else (_SDPN_C,_SDPN_G)
    # Non-affine field exercises overdetermined LS scalar-row scaling.
    u=np.random.default_rng(25).normal(size=a.shape)
    expected=np.zeros_like(u)
    for group in range(g):
        st=np.array([m.total[group] for m in mats]);rem=np.array([m.loss[0][group,group] for m in mats])
        D=np.array([1/(3*m.loss[1][group,group]) for m in mats])
        old=CorrectedMeshMomentOperator(np,a.grid,D,st,rem,1,bc='vacuum',coeffs=c,boundary_g=G[1])
        state=u[:,:,group].T.reshape((2,)+a.grid.shape)
        expected[:,:,group]=old.apply(state).reshape(2,n).T
        for other in range(g):
            if other != group:
                coupling=np.array([m.loss[0][group,other] for m in mats])
                expected[:,:,group]+=mesh.area[:,None]*coupling[:,None]*(u[:,:,other]@np.asarray(c[1][0]).T)
    np.testing.assert_allclose(a.action(u),expected,atol=2e-14)


@pytest.mark.parametrize('family',['sp3','sdp1'])
def test_noncommuting_piecewise_affine_skew_interface(family):
    mesh=partial(mesh_grid(4,.35));normal=np.array([1.,-.35])/np.hypot(1,.35)
    tangent=np.array([.35,1.])/np.hypot(1,.35)
    s=mesh.centroid@normal-.5/np.hypot(1,.35)
    mats=[material(int(v>0)) for v in s];a=op(mesh,mats,family)
    D0,D1=a.diffusion[0],a.diffusion[3]
    assert np.linalg.norm(D0@D1-D1@D0)>1e-5
    current=np.array([.3,-.1,.7,.2]);ht=np.array([.2,.4,-.3,.1]);base=np.array([2,3,1,2])
    H=np.stack([np.linalg.solve(D,current)[:,None]*normal+ht[:,None]*tangent for D in a.diffusion])
    u=np.stack([base+np.linalg.solve(D,current)*sn+ht*tn for D,sn,tn in zip(a.diffusion,s,mesh.centroid@tangent)])
    np.testing.assert_allclose(a.gradients(u.ravel()),H,atol=2e-13)
    currents,_=a.face_currents(u.ravel());checked=0
    for k,((i,j,area,_),center,n) in enumerate(zip(mesh.faces,mesh.face_centers,mesh.face_normals)):
        np.testing.assert_allclose(currents[k].ravel(),-area*a.diffusion[i]@H[i]@n,atol=2e-14)
        if mats[i] is not mats[j] and s[i]*s[j]<0:
            left=u[i]+H[i]@(center-mesh.centroid[i]);right=u[j]+H[j]@(center-mesh.centroid[j])
            np.testing.assert_allclose(left,right,atol=2e-14)
            np.testing.assert_allclose(a.diffusion[i]@H[i]@n,a.diffusion[j]@H[j]@n,atol=2e-14);checked+=1
    assert checked==4


@pytest.mark.parametrize('family',['sp3','sdp1'])
def test_coupled_vacuum_affine_identity(family):
    mesh=mesh_grid(3,.35);k=0;mesh=partial(mesh,[k]);a=op(mesh,family=family)
    n=mesh.bface_normals[0];t=np.array([-n[1],n[0]]);trace=np.array([1.1,.8,-.3,.4])
    H=-np.linalg.solve(a.diffusion[0],a.B@trace)[:,None]*n+np.array([.4,-.1,.2,.7])[:,None]*t
    u=trace+(mesh.centroid-mesh.bface_centers[0])@H.T
    np.testing.assert_allclose(a.gradients(u.ravel()),np.broadcast_to(H,(mesh.n_cells,)+H.shape),atol=2e-13)
    interior,boundary=a.face_currents(u.ravel());area=mesh.bfaces[0][1]
    np.testing.assert_allclose(boundary[0].ravel(),area*a.B@trace,atol=2e-14)
    np.testing.assert_allclose(boundary[0].ravel(),-area*a.diffusion[0]@H@n,atol=2e-14)
    if family=='sdp1':assert not np.allclose(a.B,a.B.T)


def cube3d():
    # One valid 3D cube; six boundary equations span its gradient patch.
    normals=np.vstack((np.eye(3),-np.eye(3)))
    return Mesh(np.array(list(product((-1.,1.),repeat=3))),[tuple(range(8))],np.array([0]),
                np.zeros((1,3)),np.array([8.]),[],[(0,4.,1.)]*6,
                face_centers=np.empty((0,3)),face_normals=np.empty((0,3)),
                bface_centers=normals,bface_normals=normals)


@pytest.mark.parametrize('mesh',[mesh_grid(3,.35),cube3d()])
@pytest.mark.parametrize('bc',['reflective','vacuum'])
def test_reflective_constant_conservation_independent_csr_dense(mesh,bc):
    a=op(mesh,bc=bc);constant=np.broadcast_to(np.array([1,2,3,4]),(mesh.n_cells,4)).ravel()
    currents=a.face_currents(constant)
    if bc=='reflective':
        for values in currents:np.testing.assert_allclose(values,0,atol=3e-14)
    rng=np.random.default_rng(42);u=rng.normal(size=a.size)
    result=a.action(u);reaction=np.stack([r@x for r,x in zip(a.reaction,u.reshape(-1,4))])
    _,boundary=a.face_currents(u)
    np.testing.assert_allclose((result.reshape(-1,4)-reaction).sum(axis=0),boundary.reshape(-1,4).sum(axis=0),atol=3e-14)
    dense=np.column_stack([a.action(e) for e in np.eye(a.size)])
    np.testing.assert_allclose(dense,a.assembled.toarray(),atol=2e-14)
    rhs=rng.normal(size=a.size);solution=np.linalg.solve(dense,rhs)
    np.testing.assert_allclose(a.action(solution),rhs,atol=3e-13)
    a.assembled=a.assembled*1.1
    np.testing.assert_allclose(a.action(u),result,atol=0)


def test_group_permutation_and_material_immutability():
    mesh=mesh_grid(2,.35);mats=[material(i%2) for i in range(mesh.n_cells)];a=op(mesh,mats,family='sdp1')
    def perm(m):
        return MomentMaterial(m.total[::-1],m.chi[::-1],m.nu_sigma_f[::-1],m.kappa_fission[::-1],{}, {},
                              {k:v[::-1,::-1] for k,v in m.loss.items()}, {}, {},dict(m.provenance),{})
    b=op(mesh,[perm(m) for m in mats],family='sdp1');indices=np.arange(a.size).reshape(a.shape)[:,:,::-1].ravel()
    np.testing.assert_allclose(b.assembled.toarray(),a.assembled.toarray()[indices][:,indices],atol=2e-14)
    u=np.arange(a.size)*.1
    np.testing.assert_allclose(b.fission_action(u[indices]),a.fission_action(u)[indices],atol=1e-14)
    with pytest.raises(ValueError):a.materials[0].loss[0].setflags(write=True)


def test_signed_singular_reaction_leakage_and_explicit_failures():
    mesh=mesh_grid(2,.3);m=material();loss=dict(m.loss);loss[0]=np.zeros((2,2));loss[2]=np.zeros((2,2))
    singular=replace(m,loss=loss);a=op(mesh,[singular]*mesh.n_cells)
    assert np.linalg.matrix_rank(a.assembled.toarray())==a.size
    loss=dict(m.loss);loss[0]=-np.eye(2)
    signed=op(mesh,[replace(m,loss=loss)]*mesh.n_cells)
    assert signed.reaction[0][0,0]<0
    for bad in (np.zeros((2,2)),np.diag([1,1e-14])):
        loss=dict(m.loss);loss[1]=bad
        with pytest.raises(ValueError,match='L1'):op(mesh,[replace(m,loss=loss)]*mesh.n_cells)
    with pytest.raises(ValueError,match='rank-deficient'):op(partial(mesh_grid(1,.3)))
    with pytest.raises(ValueError,match='ill-conditioned'):op(mesh,condition_limit=1.01)
    # Guard must run before geometry validation and patch/material allocation.
    fake=MeshGrid(mesh);fake.n_cells=5001
    with pytest.raises(ValueError,match='20000'):Operator(fake,[m]*5001)
    with pytest.raises(ValueError,match='boundaries'):op(mesh,bc='zero-flux')
    with pytest.raises(ValueError,match='unknowns'):a.action(np.ones(3))


@pytest.mark.parametrize('family',['sp3','sdp1'])
@pytest.mark.parametrize('preconditioner',['none','ilu'])
def test_small_eigenvalue_independent_dense_pencil_and_provenance(family,preconditioner):
    mesh=mesh_grid(2,0);mats=[material(isotropic=True)]*mesh.n_cells;a=op(mesh,mats,family)
    A=oracle(orthogonal(mesh),mats,family)
    # Independently build integrated fission from explicit state reconstruction.
    c1=np.array([[1,-2/3],[-2/3,4/9]]) if family=='sp3' else np.array([[1,-2/3],[-6/7,4/7]])
    F=np.zeros_like(A)
    for i,(v,m) in enumerate(zip(mesh.area,mats)):
        block=np.kron(c1[:,0,None]@np.array([[1.,-2/3]]),np.outer(m.chi,m.nu_sigma_f))*v
        F[i*4:i*4+4,i*4:i*4+4]=block
    expected=max(eigvals(F,A).real)
    result=CoupledMomentEigenSolver(a).solve(max_inner=1000,preconditioner=preconditioner)
    assert result.converged and result.reconstructed_nonnegative
    assert result.k_eff==pytest.approx(expected,rel=2e-9)
    assert result.full_residual<1e-9
    assert not result.provenance['physical_admission'] and not result.provenance['scattering_RHS']
    assert result.provenance['preconditioner']==preconditioner
    metadata=result.provenance['preconditioner_metadata']
    assert metadata['matrix_nnz']==a.assembled.nnz
    assert metadata['setup_seconds']>=0
    if preconditioner=='ilu':
        assert metadata['drop_tol']==1e-6 and metadata['fill_factor']==20
        assert metadata['superlu_nnz']>0 and 'not an explicit' in metadata['superlu_nnz_caveat']
        default=CoupledMomentEigenSolver(a).solve()
        assert default.provenance['preconditioner']=='none'
        assert default.converged
        np.testing.assert_allclose(result.moments,default.moments,rtol=2e-8,atol=1e-10)
        assert result.k_eff==pytest.approx(default.k_eff,rel=2e-9)
    hashes=result.provenance['implementation_sha256']
    assert {'corrected_native_moments.py','corrected_mesh.py','mesh.py'}<=hashes.keys()
    a.assembled=a.assembled*1.1
    failed=CoupledMomentEigenSolver(a).solve(preconditioner=preconditioner)
    assert not failed.converged and failed.reason=='inner_failure_or_full_linear_residual'


def test_ilu_factor_once_without_explicit_lu_materialization(monkeypatch):
    import garnet.coupled_moment_solver as driver
    a=op(mesh_grid(2,0),[material(isotropic=True)]*4)
    original=driver.spilu;calls=[]
    class Factor:
        def __init__(self,factor):self.factor=factor
        @property
        def nnz(self):return self.factor.nnz
        def solve(self,x):return self.factor.solve(x)
        @property
        def L(self):raise AssertionError('L must not be materialized')
        @property
        def U(self):raise AssertionError('U must not be materialized')
    def factor(matrix,**kwargs):
        calls.append((matrix,kwargs))
        assert matrix.format=='csc' and not np.shares_memory(matrix.data,a.assembled.data)
        return Factor(original(matrix,**kwargs))
    monkeypatch.setattr(driver,'spilu',factor)
    result=driver.CoupledMomentEigenSolver(a).solve(preconditioner='ilu')
    assert result.converged and result.iterations>1
    assert len(calls)==1
    assert calls[0][1]==dict(drop_tol=1e-6,fill_factor=20)


def test_unknown_preconditioner_and_ilu_factor_failure_no_fallback(monkeypatch):
    import garnet.coupled_moment_solver as driver
    a=op(mesh_grid(2,0),[material(isotropic=True)]*4)
    solver=driver.CoupledMomentEigenSolver(a)
    with pytest.raises(ValueError,match='preconditioner'):solver.solve(preconditioner='automatic')
    def fail(*args,**kwargs):raise RuntimeError('synthetic singular factor')
    def forbidden(*args,**kwargs):raise AssertionError('GMRES must not run after factor failure')
    monkeypatch.setattr(driver,'spilu',fail);monkeypatch.setattr(driver,'gmres',forbidden)
    with pytest.raises(RuntimeError,match='ILU factorization failed; no solver fallback') as caught:
        solver.solve(preconditioner='ilu')
    assert 'synthetic singular factor' in str(caught.value.__cause__)
