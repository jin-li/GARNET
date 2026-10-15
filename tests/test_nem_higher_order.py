"""Independent uncondensed Legendre physical pencil and quartic parity."""
import numpy as np
import pytest
from numpy.polynomial import legendre as leg
from ndgpu.nem import NEMDiffusionSolver
from ndgpu.nem_higher_order import NEMHigherOrderDiffusionSolver
from ndgpu.nem_geometry import CuboidNodalGrid
from test_nem import tensor_grid,materials,independent_weights,PAIRS


def independent_pencil(grid,D,sa,ss,nuf,chi,P,factors):
    N,G=D.shape;K=P-2;F=len(grid.face_nodes);ng=N*G
    coff=ng;joff=ng+N*3*P*G;size=joff+F*G
    A=np.zeros((size,size));Q=np.zeros_like(A)
    c=lambda i,a,j,g:coff+((i*3+a)*P+j-1)*G+g
    jidx=lambda f,g:joff+f*G+g
    u,w=leg.leggauss(2*P+5);w=w/2
    basis=leg.legvander(u,P).T
    dd=np.stack([leg.legval(u,leg.legder(np.eye(P+1)[j],2))*4 for j in range(P+1)])
    dends=np.stack([leg.legval([-1.,1.],leg.legder(np.eye(P+1)[j]))*2 for j in range(P+1)])
    moments=(basis[1:K+1]/2*w)@basis.T
    second=(basis[1:K+1]/2*w)@dd.T
    # Exact zero mean/mutual orthogonality; independently integrated nonzero
    # overlaps retained. Prevent zero quadrature noise filling sparse nulls.
    moments[abs(moments)<1e-12]=0;second[abs(second)<1e-11]=0
    for i in range(N):
        R=np.zeros((G,G))
        for g in range(G):
            R[g,g]=sa[i,g]+sum(ss[i,g,h] for h in range(G) if h!=g)
            for h in range(G):
                if h!=g:R[g,h]=-ss[i,h,g]
        Fmat=np.outer(chi[i],nuf[i])
        A[i*G:(i+1)*G,i*G:(i+1)*G]=R
        Q[i*G:(i+1)*G,i*G:(i+1)*G]=Fmat
        for axis,(sm,sp) in enumerate(PAIRS):
            L=grid.widths[i,axis];fm,fp=grid.node_faces[i,[sm,sp]]
            samples,weights=independent_weights(grid,i,axis)
            for g in range(G):
                A[i*G+g,jidx(fp,g)]+=1/L;A[i*G+g,jidx(fm,g)]-=1/L
                for n in range(1,K+1):
                    row=c(i,axis,n,g)
                    for jj in range(1,P+1):
                        for h in range(G):
                            A[row,c(i,axis,jj,h)]+=R[g,h]*moments[n-1,jj]
                            Q[row,c(i,axis,jj,h)]+=Fmat[g,h]*moments[n-1,jj]
                        A[row,c(i,axis,jj,g)]-=D[i,g]/L**2*second[n-1,jj]
                    if n<=2:
                        for k,sample in enumerate(samples):
                            for transverse,(tm,tp) in enumerate(PAIRS):
                                if transverse!=axis:
                                    ww=weights[n-1,k]/grid.widths[i,transverse]
                                    A[row,jidx(grid.node_faces[sample,tp],g)]+=ww
                                    A[row,jidx(grid.node_faces[sample,tm],g)]-=ww
                for sideidx,face in enumerate((fm,fp)):
                    row=c(i,axis,K+sideidx+1,g)
                    for jj in range(1,P+1):A[row,c(i,axis,jj,g)]=-D[i,g]/L*dends[jj,sideidx]
                    A[row,jidx(face,g)]=-1
    bindex={int(f):k for k,f in enumerate(grid.boundary_faces)}
    for f,(low,high) in enumerate(grid.face_nodes):
        axis=grid.face_axes[f];sm,sp=PAIRS[axis]
        for g in range(G):
            row=jidx(f,g)
            if low>=0 and high>=0:
                owners=((low,sp,1.,1.),(high,sm,-1.,-1.))
            else:
                k=bindex[f];node,side=grid.boundary_nodes[k],grid.boundary_sides[k]
                A[row,jidx(f,g)]=grid.normal_signs[side]
                owners=((node,side,-grid.boundary_alpha[k],1. if side==sp else -1.),)
            for node,side,sign,uend in owners:
                factor=factors[node,side,g] if low>=0 and high>=0 else 1.
                A[row,node*G+g]+=sign*factor
                for jj in range(1,P+1):A[row,c(node,axis,jj,g)]+=sign*factor*uend**jj
    return A,Q


def recovered_vector(r,P):
    N,G=r.means.shape;K=P-2;c=np.zeros((N,3,P,G))
    c[:,:,:K]=r.moments*(2*(2*np.arange(1,K+1)+1))[None,None,:,None]
    for a,(sm,sp) in enumerate(PAIRS):
        c[:,a,P-2]=(r.side_face_means[:,sp]-r.side_face_means[:,sm])/2-c[:,a,0:K:2].sum(axis=1)
        c[:,a,P-1]=(r.side_face_means[:,sp]+r.side_face_means[:,sm])/2-r.means-c[:,a,1:K:2].sum(axis=1)
    return np.r_[r.means.ravel(),c.ravel(),r.currents.ravel()]


@pytest.mark.parametrize('P',[4,6,8,14])
@pytest.mark.parametrize('G',[2,4])
def test_full_uncondensed_signed_sources_factors_robin(P,G):
    grid=tensor_grid(([0.,.7,1.9],[0.,1.3],[0.,.8]))
    D,sa,ss,nuf,chi=materials(grid,G);N=grid.n_nodes;K=P-2
    factors=np.ones((N,6,G));factors[0,2]=np.linspace(.9,1.1,G);factors[1,0]=np.linspace(1.05,.95,G)
    rng=np.random.default_rng(503);q0=.4+rng.random((N,G));qm=rng.normal(size=(N,3,K,G))*.001
    drive=rng.normal(size=(len(grid.boundary_faces),G))*.01
    solver=NEMHigherOrderDiffusionSolver(grid,D,sa,ss,nuf,chi,factors,degree=P)
    result=solver.solve_fixed_source(q0,qm,drive)
    A,Q=independent_pencil(grid,D,sa,ss,nuf,chi,P,factors)
    rhs=np.zeros(len(A));rhs[:N*G]=q0.ravel()
    for i in range(N):
        for a in range(3):
            offset=N*G+(i*3+a)*P*G;rhs[offset:offset+K*G]=qm[i,a].ravel()
    rhs[-solver.n_faces*G:].reshape(-1,G)[grid.boundary_faces]=drive
    expected=np.linalg.solve(A,rhs)
    np.testing.assert_allclose(recovered_vector(result,P),expected,rtol=2e-8,atol=3e-12)
    assert solver._loss_matrix.shape==(G*(N+len(grid.face_nodes)),)*2
    assert result.moments.shape==(N,3,K,G)
    assert solver._backend_evidence['polynomial_degree']==P


@pytest.mark.parametrize('boundary_policy',['one_sided','primary_rule'])
def test_degree4_matches_existing_solver(boundary_policy):
    base=tensor_grid(([0.,.7,2.],[0.,1.3],[0.,.8]))
    grid=CuboidNodalGrid(base.bounds,leakage_boundary=boundary_policy)
    mat=materials(grid,4);q0=np.ones((grid.n_nodes,4));qm=np.arange(grid.n_nodes*24).reshape(grid.n_nodes,3,2,4)*.00001
    old=NEMDiffusionSolver(grid,*mat).solve_fixed_source(q0,qm)
    new=NEMHigherOrderDiffusionSolver(grid,*mat,degree=4).solve_fixed_source(q0,qm)
    for name in ('means','moments','currents','side_face_means'):
        np.testing.assert_allclose(getattr(old,name),getattr(new,name),rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(old.diagnostics['profile_minima'],new.diagnostics['profile_minima'],atol=1e-12)


@pytest.mark.parametrize('P',[6,8,14])
def test_all_moment_fission_against_independent_eigenpencil(P):
    grid=tensor_grid(([0.,.7,1.9],[0.,1.3],[0.,.8]));G=2;N=grid.n_nodes
    mat=materials(grid,G);factors=np.ones((N,6,G))
    solver=NEMHigherOrderDiffusionSolver(grid,*mat,degree=P)
    A,Q=independent_pencil(grid,*mat,P,factors)
    values,vectors=np.linalg.eig(np.linalg.solve(A,Q));idx=np.argmax(values.real)
    assert abs(values[idx].imag)<1e-12
    result=solver.solve_eigenvalue(rtol=1e-10,max_iterations=1000)
    assert abs(result.k_eff-values[idx].real)<1e-10
    state=recovered_vector(result,P);res=A@state-Q@state/result.k_eff
    assert np.max(abs(res))<1e-9
    assert np.max(abs(result.moments[:,:,2:]))>1e-10
    assert result.diagnostics['all_spatial_fission_moments']
    assert abs(solver._production(result)-1)<1e-12


def test_degree_and_moment_count_mutation_rejected():
    grid=tensor_grid(([0.,1.],[0.,1.],[0.,1.]));s=NEMHigherOrderDiffusionSolver(grid,*materials(grid,2),degree=6)
    s.degree=8
    with pytest.raises(ValueError,match='changed'):s.solve_fixed_source(np.ones((1,2)))
    s.degree=6;s.n_moments=2
    with pytest.raises(ValueError,match='changed'):s.solve_fixed_source(np.ones((1,2)))
