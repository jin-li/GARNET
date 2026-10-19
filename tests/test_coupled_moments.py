"""Independent tiny dense face-unknown reference, no external data or solves."""
import numpy as np
import pytest
from scipy.linalg import eigvals
from ndgpu.moment_materials import MomentMaterial
from ndgpu.coupled_moments import BoundaryFace,InteriorFace,OrthogonalMesh,CoupledMomentOperator


def material(which=0,isotropic=False):
    total=np.array([1.3,1.8])
    losses={0:np.array([[.6,-.12],[-.18,.9]]),1:np.array([[1.1,.08],[-.04,1.5]]),
            2:np.array([[1.2,-.03],[.06,1.6]]),3:np.array([[1.25,.04],[-.05,1.7]])}
    if which:
        losses={k:a+np.array([[.2,-.07],[.1,.3]]) for k,a in losses.items()}
    if isotropic:
        losses={0:np.diag([.5,.8]),1:np.diag(total),2:np.diag(total),3:np.diag(total)}
    return MomentMaterial(total,np.array([.8,.2]),np.array([.3,.4]),np.array([20,30]),
         {k:np.zeros((2,2)) for k in losses},{k:np.diag(total)-a for k,a in losses.items()},losses,{}, {},
         {'orientation_contract':'from-major','physical_admission':False},{})


def mesh(vacuum=True):
    return OrthogonalMesh([2.,3.],[[0.],[2.]],(InteriorFace(0,1,1.7,(.7,),(1.,),.7,1.3),),
         (BoundaryFace(0,.8,(-.4,),(-1.,),.4,'vacuum' if vacuum else 'reflective'),
          BoundaryFace(1,1.,(2.6,),(1.,),.6,'reflective')))


def tables(family):
    if family=='sp3':return (np.array([[1,-2/3],[-2/3,4/9]]),np.array([[0,0],[0,5/9]]),np.array([[.5,-1/8],[-1/8,7/24]]))
    return (np.array([[1,-2/3],[-6/7,4/7]]),np.array([[0,0],[0,5/7]]),np.array([[.5,-1/8],[-2/7,23/42]]))


def oracle(mesh,materials,family):
    # Keep face U unknowns and enforce current continuity directly, then take
    # their Schur complement. No production harmonic/vacuum formula reused.
    n=len(materials);g=2;q=4
    vacuum=[f for f in mesh.boundary if f.kind=='vacuum']
    count=len(mesh.interior)+len(vacuum);H=np.zeros(((n+count)*q,)*2)
    c1,c2,marshak=tables(family);B=np.kron(marshak,np.eye(g))
    Ds=[]
    for i,(v,m) in enumerate(zip(mesh.volumes,materials)):
        s=slice(i*q,(i+1)*q)
        H[s,s]=v*(np.kron(c1,m.loss[0])+np.kron(c2,m.loss[2]))
        D=np.zeros((q,q));D[:g,:g]=np.linalg.solve(3*m.loss[1],np.eye(g));D[g:,g:]=np.linalg.solve(7*m.loss[3],np.eye(g));Ds.append(D)
    for k,f in enumerate(mesh.interior):
        z=slice((n+k)*q,(n+k+1)*q)
        for i,d in ((f.left,f.dleft),(f.right,f.dright)):
            s=slice(i*q,(i+1)*q);C=Ds[i]/d
            H[s,s]+=f.area*C;H[s,z]-=f.area*C
            H[z,s]-=C;H[z,z]+=C
    for k,f in enumerate(vacuum,start=len(mesh.interior)):
        z=slice((n+k)*q,(n+k+1)*q);s=slice(f.cell*q,(f.cell+1)*q);C=Ds[f.cell]/f.distance
        H[s,s]+=f.area*C;H[s,z]-=f.area*C;H[z,s]-=C;H[z,z]+=C+B
    N=n*q
    return H[:N,:N]-H[:N,N:]@np.linalg.solve(H[N:,N:],H[N:,:N]) if count else H


@pytest.mark.parametrize('family',['sp3','sdp1'])
@pytest.mark.parametrize('vacuum',[False,True])
def test_independent_noncommuting_face_unknown_oracle(family,vacuum):
    m=(material(),material(1));grid=mesh(vacuum);op=CoupledMomentOperator(grid,m,family=family)
    assert np.linalg.norm(m[0].loss[1]@m[1].loss[1]-m[1].loss[1]@m[0].loss[1])>.001
    expected=oracle(grid,m,family)
    np.testing.assert_allclose(op.assembled.toarray(),expected,rtol=2e-14,atol=2e-14)
    u=np.arange(8.)*.3-1
    np.testing.assert_allclose(op.action(u),expected@u,rtol=2e-14,atol=2e-14)
    np.testing.assert_array_equal(op.action(u.reshape(2,2,2)).ravel(),op.action(u))
    scalar=u.reshape(2,2,2)[:,0]-2*u.reshape(2,2,2)[:,1]/3
    np.testing.assert_array_equal(op.scalar_flux(u),scalar)
    c1,_,_=tables(family)
    expected_f=np.stack([v*c1[:,0,None]*x.chi*(x.nu_sigma_f@p) for v,x,p in zip(grid.volumes,m,scalar)])
    np.testing.assert_allclose(op.fission_action(u),expected_f.ravel())


def test_integrated_face_conservation_and_no_neighbor_transpose():
    op=CoupledMomentOperator(mesh(False),(material(),material(1)))
    u=np.arange(8.).reshape(2,4)
    leak=op.action(u.ravel()).reshape(2,4)-np.stack([a@x for a,x in zip(op.reaction,u)])
    np.testing.assert_allclose(leak.sum(axis=0),0,atol=1e-15)
    T=op.faces[0][2];assert not np.allclose(T,T.T)
    np.testing.assert_allclose(leak[0],T@(u[0]-u[1]))


@pytest.mark.parametrize('family',['sp3','sdp1'])
def test_group_and_cell_permutation(family):
    m=[material(),material(1)];op=CoupledMomentOperator(mesh(),m,family=family)
    def perm(x):
        return MomentMaterial(x.total[::-1],x.chi[::-1],x.nu_sigma_f[::-1],x.kappa_fission[::-1],{}, {},
            {k:a[::-1,::-1] for k,a in x.loss.items()}, {}, {},dict(x.provenance),{})
    swapped=OrthogonalMesh([3,2],[[2],[0]],(InteriorFace(1,0,1.7,(.7,),(1,),.7,1.3),),
        (BoundaryFace(1,.8,(-.4,),(-1,),.4,'vacuum'),))
    b=CoupledMomentOperator(swapped,[perm(m[1]),perm(m[0])],family=family)
    indices=np.arange(8).reshape(2,2,2)[::-1,:,::-1].ravel()
    np.testing.assert_allclose(b.assembled.toarray(),op.assembled.toarray()[indices][:,indices])
    u=np.arange(8.)
    np.testing.assert_allclose(b.fission_action(u[indices]),op.fission_action(u)[indices])


@pytest.mark.parametrize('family',['sp3','sdp1'])
def test_isotropic_reduction_and_homogeneous_k_infinity(family):
    m=material(isotropic=True);grid=OrthogonalMesh([2.5],[[0.]])
    op=CoupledMomentOperator(grid,[m],family=family)
    c1,c2,_=tables(family)
    np.testing.assert_allclose(op.diffusion[0],np.diag(np.r_[1/(3*m.total),1/(7*m.total)]))
    np.testing.assert_allclose(op.assembled.toarray()/2.5,np.kron(c1,m.loss[0])+np.kron(c2,np.diag(m.total)))
    F=np.column_stack([op.fission_action(np.eye(4)[:,i]) for i in range(4)])
    values=eigvals(F,op.assembled.toarray())
    expected=m.nu_sigma_f@np.linalg.solve(m.loss[0],m.chi)
    assert max(values.real)==pytest.approx(expected,rel=1e-13)
    eigenvalues,eigenvectors=np.linalg.eig(np.linalg.solve(op.assembled.toarray(),F))
    u=eigenvectors[:,np.argmax(eigenvalues.real)].real
    np.testing.assert_allclose(op.action(u),op.fission_action(u)/expected,atol=1e-14)


@pytest.mark.parametrize('bad',[
    OrthogonalMesh([0],[[0]]),
    OrthogonalMesh([1,1],[[0],[2]],(InteriorFace(0,1,1,(.7,),(1,),.8,1.3),)),
    OrthogonalMesh([1],[[0]],boundary=(BoundaryFace(0,1,(1,),(1,),1,'zero'),)),
    OrthogonalMesh([1],[[np.nan]]),
    OrthogonalMesh([1],[[0]],boundary=(BoundaryFace(2,1,(1,),(1,),1,'vacuum'),)),
])
def test_bad_geometry_refused(bad):
    with pytest.raises(ValueError):CoupledMomentOperator(bad,[material()]*len(bad.volumes))


def test_losses_unknowns_and_family_refuse():
    m=material();grid=OrthogonalMesh([1],[[0]])
    for value in (np.zeros((2,2)),np.diag([1,1e-14]),np.full((2,2),np.nan)):
        losses=dict(m.loss);losses[1]=value
        bad=MomentMaterial(m.total,m.chi,m.nu_sigma_f,m.kappa_fission,{}, {},losses,{}, {},dict(m.provenance),{})
        with pytest.raises(ValueError):CoupledMomentOperator(grid,[bad])
    with pytest.raises(ValueError):CoupledMomentOperator(grid,[m],family='sp5')
    op=CoupledMomentOperator(grid,[m])
    with pytest.raises(ValueError):op.action([np.nan]*4)
    with pytest.raises(ValueError):op.scalar_flux(np.ones(3))


def test_duplicate_face_and_nonfinite_vectors_refused():
    f=InteriorFace(0,1,1,(1,),(1,),1,1)
    with pytest.raises(ValueError,match='Duplicate'):
        CoupledMomentOperator(OrthogonalMesh([1,1],[[0],[2]],(f,f)),[material(),material()])
    m=material()
    bad=MomentMaterial(m.total,np.array([np.nan,0]),m.nu_sigma_f,m.kappa_fission,{}, {},dict(m.loss),{}, {},dict(m.provenance),{})
    with pytest.raises(ValueError,match='material vectors'):
        CoupledMomentOperator(OrthogonalMesh([1],[[0]]),[bad])


def test_energy_bounds_mismatch_and_missing_moments_refused():
    m=material();p=dict(m.provenance);p['energy_bounds_mev']=np.array([20,1,0])
    a=MomentMaterial(m.total,m.chi,m.nu_sigma_f,m.kappa_fission,{}, {},dict(m.loss),{}, {},p,{})
    with pytest.raises(ValueError,match='Energy boundary'):
        CoupledMomentOperator(mesh(),[a,m])
    loss=dict(m.loss);del loss[3]
    b=MomentMaterial(m.total,m.chi,m.nu_sigma_f,m.kappa_fission,{}, {},loss,{}, {},dict(m.provenance),{})
    with pytest.raises(ValueError,match='0..3'):
        CoupledMomentOperator(OrthogonalMesh([1],[[0]]),[b])
