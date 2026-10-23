"""Algebra checks for generic common-trace factors, not Serpent qualification."""

import numpy as np
import pytest

from garnet import MeshGrid, MeshGroupOperator, MeshSP3EigenSolver, MeshSDP1EigenSolver
from garnet.linalg import bicgstab
from test_mesh_moments import rectangles, boxes, materials


def pencil(mesh, mats, cm, family, specs, ff, bf):
    # Literal worktree order-1 tables: U1=3 phi2; phi0=U0-2/3 U1.
    c0 = np.array([[1, -2/3], [-2/3, 4/9]] if family == 'spn' else
                  [[1, -2/3], [-6/7, 4/7]])
    ct = np.array([[0, 0], [0, 5/9 if family == 'spn' else 5/7]])
    g = np.array([[1/2, -1/8], [-1/8, 7/24]] if family == 'spn' else
                 [[1/2, -1/8], [-2/7, 23/42]])
    G, N = mats[0].n_groups, mesh.n_cells
    A = np.zeros((G*2*N, G*2*N)); F = np.zeros_like(A)
    def ix(group, moment, cell):
        return (group*2+moment)*N+cell
    for group in range(G):
        D = np.array([[mats[m].diffusion[group] for m in cm],
                      [1/(7*mats[m].sigma_t[group]) for m in cm]])
        for n, mat_index in enumerate(cm):
            mat, V = mats[mat_index], mesh.area[n]
            for i in range(2):
                for j in range(2):
                    A[ix(group,i,n),ix(group,j,n)] += V*(c0[i,j]*mat.removal[group]+ct[i,j]*mat.sigma_t[group])
                    for source in range(G):
                        F[ix(group,i,n),ix(source,j,n)] += V*c0[i,0]*mat.chi[group]*mat.nu_sigma_f[source]*c0[0,j]
                        if source != group:
                            A[ix(group,i,n),ix(source,j,n)] -= V*c0[i,0]*mat.sigma_s[source,group]*c0[0,j]
        for k, ((left,right,S,d), distances) in enumerate(zip(mesh.faces,mesh.face_distances)):
            fl, fr = ff[group,k]
            for i in range(2):
                # Solve two trace equations, rather than using production weights.
                al, ar = D[i,left]/distances[0], D[i,right]/distances[1]
                equations = np.array([[fl,-fr],[al,ar]])
                for column, ul, ur in ((left,1.,0.),(right,0.,1.)):
                    tl, tr = np.linalg.solve(equations,[0.,al*ul+ar*ur])
                    current = S*al*(ul-tl)
                    A[ix(group,i,left),ix(group,i,column)] += current
                    A[ix(group,i,right),ix(group,i,column)] -= current
        for k, ((n,S,d),spec) in enumerate(zip(mesh.bfaces,specs)):
            if spec == 'reflective':
                continue
            aa = np.diag(D[:,n]/d)
            if spec == 'vacuum':
                traces = np.linalg.solve(aa+bf[group,k]*g,aa)
                K = S*aa@(np.eye(2)-traces)
            else:
                K = S*aa
            for i in range(2):
                for j in range(2):
                    A[ix(group,i,n),ix(group,j,n)] += K[i,j]
    return A,F


@pytest.mark.parametrize('M,g', [(1,[[.5]]),(2,[[.5,-.125],[-.125,7/24]]),
                                  (2,[[.5,-.125],[-2/7,23/42]])])
def test_corrected_boundary_trace_equations_and_scalar_limit(M,g):
    mesh = rectangles([2.],3.)
    grid = MeshGrid(mesh,['vacuum','reflective','vacuum','zero-flux'])
    Ds = [np.full(grid.shape,.7 if i == 0 else 1/7) for i in range(M)]
    factors = np.array([.4,1.3,1.8,.9])
    blocks = grid.boundary_blocks(np,None,Ds,g,boundary_df=factors)
    u = np.arange(1.,M+1)
    direct = np.zeros(M)
    for (n,S,d),spec,f in zip(mesh.bfaces,grid.boundary_conditions,factors):
        if spec == 'vacuum':
            aa = np.diag([D.item()/d for D in Ds])
            trace = np.linalg.solve(aa+f*np.array(g),aa@u)
            current = aa@(u-trace)
            np.testing.assert_allclose(current,f*np.array(g)@trace)
            direct += S*current
    np.testing.assert_allclose(blocks[0][1][0]@u,direct)
    if M == 1:
        scalar = MeshGroupOperator(np,grid,Ds[0],np.zeros(grid.shape),boundary_df=factors)
        stripped = MeshGroupOperator(np,grid,Ds[0],np.zeros(grid.shape),
                                     bc=grid.moment_boundary_conditions(strip_vacuum=True),boundary_df=factors)
        np.testing.assert_allclose((scalar.diag-stripped.diag).item(),direct.item())


@pytest.mark.parametrize('family,cls', [('spn',MeshSP3EigenSolver),('sdpn',MeshSDP1EigenSolver)])
@pytest.mark.parametrize('make_mesh', [rectangles,boxes])
def test_dense_multigroup_eigenpair_physical_flux_and_full_moment_residual(family,cls,make_mesh):
    mesh, mats, cm = make_mesh(), materials(), [0,1]
    specs = [('reflective','vacuum','zero-flux')[k%3] for k in range(len(mesh.bfaces))]
    ff = np.array([[[.7,1.4]],[[1.8,.9]]])
    bf = np.linspace(.6,1.5,2*len(mesh.bfaces)).reshape(2,-1)
    solver = cls(mesh,mats,cm,device='cpu',boundary_conditions=specs,
                 marshak_vacuum=True,symmetrize=False,moment_df_model='common_trace',
                 moment_face_df=ff,moment_boundary_df=bf)
    assert solver._linsolve is bicgstab
    assert all(not op.symmetric for op in solver.ops)
    A,F = pencil(mesh,mats,cm,family,specs,ff,bf)
    probe = np.random.default_rng(32).normal(size=A.shape[0]).reshape(2,2,2,1,1)
    rows = []
    for group in range(2):
        row = solver.ops[group].apply(probe[group])
        source = 1-group
        row -= solver._rhs(group,solver.sigma_s[source][group]*solver._phi(probe[source])*solver._src_weight)
        rows.append(row.ravel())
    np.testing.assert_allclose(np.concatenate(rows),A@probe.ravel(),rtol=3e-13,atol=3e-13)
    result = solver.solve(tol_k=1e-11,tol_source=1e-10,inner_rtol_floor=1e-13)
    assert result.converged
    vals,vecs = np.linalg.eig(np.linalg.solve(A,F))
    index = np.argmax(vals.real)
    assert abs(result.k_eff-vals[index].real) < 2e-10
    state = np.concatenate([u.ravel() for u in solver.state])
    assert np.linalg.norm(A@state-F@state/result.k_eff)/np.linalg.norm(A@state) < 2e-9
    ref = vecs[:,index].real.reshape(2,2,2)
    phi = ref[:,0]-2/3*ref[:,1]
    np.testing.assert_allclose(result.flux.reshape(2,2)/result.flux[0,0,0,0],phi/phi[0,0],rtol=2e-8,atol=2e-9)


@pytest.mark.parametrize('cls', [MeshSP3EigenSolver,MeshSDP1EigenSolver])
@pytest.mark.parametrize('marshak', [False,True])
def test_unit_factors_recover_default_including_similarity(cls,marshak):
    mesh=rectangles(); kw=dict(device='cpu',marshak_vacuum=marshak,bc='vacuum' if marshak else 'reflective')
    default=cls(mesh,materials(),[0,1],**kw)
    unit=cls(mesh,materials(),[0,1],moment_df_model='common_trace',**kw)
    u=np.arange(4.).reshape(2,2,1,1)+.7
    for ref,op in zip(default.ops,unit.ops):
        np.testing.assert_array_equal(op.apply(u),ref.apply(u))
        np.testing.assert_array_equal(op.inv_diag,ref.inv_diag)
        assert op.symmetric == ref.symmetric
        assert op.phi0_weights == ref.phi0_weights
        assert op.src_weights == ref.src_weights


@pytest.mark.parametrize('kwargs,match', [
    ({'moment_face_df':np.ones((2,1,2))},'require moment_df_model'),
    ({'moment_df_model':'epdf'},'supports only'),
    ({'face_df':np.ones((2,1,2))},'ambiguous'),
    ({'moment_df_model':'common_trace','moment_face_df':np.zeros((2,1,2))},'finite positive'),
    ({'moment_df_model':'common_trace','moment_boundary_df':np.ones((2,1))},'shape'),
    ({'moment_df_model':'common_trace','moment_face_df':np.array([[[.7,1.4]],[[1.,1.]]]),'linear_solver':'cg'},'nonsymmetric'),
])
def test_api_gates(kwargs,match):
    with pytest.raises(ValueError,match=match):
        MeshSP3EigenSolver(rectangles(),materials(),[0,1],device='cpu',marshak_vacuum=True,**kwargs)


@pytest.mark.parametrize('cls', [MeshSP3EigenSolver,MeshSDP1EigenSolver])
def test_independent_face_traces_conservation_rescaling_reversal_and_subfaces(cls):
    from dataclasses import replace
    mesh=rectangles()
    ff=np.array([[[.7,1.4]],[[1.8,.9]]])
    kw=dict(device='cpu',bc='reflective',symmetrize=False,moment_df_model='common_trace')
    solver=cls(mesh,materials(),[0,1],moment_face_df=ff,**kw)
    u=np.array([[1.2,3.4],[.8,-.2]]).reshape(2,2,1,1)
    for group,op in enumerate(solver.ops):
        Ds=[np.array([m.diffusion[group] for m in materials()]),
            np.array([1/(7*m.sigma_t[group]) for m in materials()])]
        for moment,spatial in enumerate(op.moments):
            al,ar=Ds[moment]/mesh.face_distances[0]
            fl,fr=ff[group,0]
            ul,ur=u[moment].ravel()
            tl,tr=np.linalg.solve([[fl,-fr],[al,ar]],[0,al*ul+ar*ur])
            current=mesh.faces[0][2]*al*(ul-tl)
            assert abs(fl*tl-fr*tr)<1e-14
            np.testing.assert_allclose(al*(ul-tl),-ar*(ur-tr),atol=2e-15)
            removal=np.array([m.removal[group] for m in materials()])
            total=np.array([m.sigma_t[group] for m in materials()])
            divisor=9 if cls is MeshSP3EigenSolver else 7
            reaction=mesh.area*(removal if moment==0 else (4*removal+5*total)/divisor)
            leakage=spatial.apply(u[moment]).ravel()-reaction*u[moment].ravel()
            np.testing.assert_allclose(leakage,[current,-current],atol=3e-15)
    i,j,S,d=mesh.faces[0]
    split=replace(mesh,faces=[(i,j,.3*S,d),(i,j,.7*S,d)],
                  face_distances=np.repeat(mesh.face_distances,2,axis=0))
    reverse=replace(mesh,faces=[(j,i,S,d)],face_distances=mesh.face_distances[:,::-1])
    alternatives=[cls(mesh,materials(),[0,1],moment_face_df=19*ff,**kw),
                  cls(split,materials(),[0,1],moment_face_df=np.repeat(ff,2,axis=1),**kw),
                  cls(reverse,materials(),[0,1],moment_face_df=ff[...,::-1],**kw)]
    for other in alternatives:
        for op,ref in zip(other.ops,solver.ops):
            np.testing.assert_allclose(op.apply(u),ref.apply(u),rtol=3e-15,atol=3e-15)


def test_boundary_factor_limits_and_nonmultiplicative_elimination():
    mesh=rectangles([2.],3.); grid=MeshGrid(mesh,'vacuum')
    D=[np.full(grid.shape,.7),np.full(grid.shape,.2)]
    g=np.array([[.5,-.125],[-.125,7/24]])
    def block(f):
        return grid.boundary_blocks(np,None,D,g,np.full(len(mesh.bfaces),f))[0][1][0]
    small=1e-10
    np.testing.assert_allclose(block(small)/small,sum(S*g for n,S,d in mesh.bfaces),rtol=2e-9,atol=1e-9)
    np.testing.assert_allclose(block(1e10),sum(S*np.diag([.7/d,.2/d]) for n,S,d in mesh.bfaces),rtol=2e-9,atol=1e-9)
    assert not np.allclose(block(2),2*block(1))


@pytest.mark.parametrize('kwargs,match', [({'bc':'vacuum'},'marshak_vacuum'),
                                         ({'bc':.2,'marshak_vacuum':True},'supports only'),
                                         ({'order':2,'bc':'reflective'},'order-1')])
def test_opt_in_boundary_and_order_gates(kwargs,match):
    from garnet import MeshSPNEigenSolver
    with pytest.raises(ValueError,match=match):
        MeshSPNEigenSolver(rectangles(),materials(),[0,1],moment_df_model='common_trace',device='cpu',**kwargs)


def test_unequal_factors_commute_with_supported_sdp_similarity():
    mesh=rectangles(); mats=materials(); ff=np.array([[[.7,1.4]],[[1.8,.9]]])
    solver=MeshSDP1EigenSolver(mesh,mats,[0,1],bc='reflective',device='cpu',
                              moment_df_model='common_trace',moment_face_df=ff)
    assert solver._symmetrize
    A,F=pencil(mesh,mats,[0,1],'sdpn',['reflective']*len(mesh.bfaces),ff,
               np.ones((2,len(mesh.bfaces))))
    # Literal diagonal similarity of the SDP1 c table, independently applied
    # to the entire multigroup pencil, including scalar-flux/source weights.
    r=np.tile(np.repeat([1.,np.sqrt(7)/3],2),2)
    transformed_A=r[:,None]*A/r[None,:]
    transformed_F=r[:,None]*F/r[None,:]
    result=solver.solve(tol_k=1e-11,tol_source=1e-10,inner_rtol_floor=1e-13)
    assert result.converged
    state=np.concatenate([u.ravel() for u in solver.state])
    residual=transformed_A@state-transformed_F@state/result.k_eff
    assert np.linalg.norm(residual)/np.linalg.norm(transformed_A@state)<2e-9
    physical=(state/r).reshape(2,2,2)
    np.testing.assert_allclose(result.flux.reshape(2,2),physical[:,0]-2/3*physical[:,1],rtol=3e-14)
    plain=MeshSDP1EigenSolver(mesh,mats,[0,1],bc='reflective',device='cpu',symmetrize=False,
                             moment_df_model='common_trace',moment_face_df=ff)
    plain_result=plain.solve(tol_k=1e-11,tol_source=1e-10,inner_rtol_floor=1e-13)
    assert plain_result.converged
    assert abs(result.k_eff-plain_result.k_eff)<2e-10
    np.testing.assert_allclose(result.flux,plain_result.flux,rtol=2e-8,atol=2e-9)
