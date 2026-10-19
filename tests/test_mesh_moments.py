"""Independent CPU checks of the DF-off integrated common-mesh operators."""

import numpy as np
import pytest

from ndgpu import (Grid, Material, MeshGrid, MeshGroupOperator,
                   MeshDiffusionEigenSolver, MeshSP1EigenSolver,
                   MeshSP3EigenSolver, MeshSDP1EigenSolver,
                   MeshSPNEigenSolver, MeshSDPNEigenSolver,
                   UnstructuredDiffusionSolver, assemble_mesh, assemble_mesh_3d)
from ndgpu.solver import SPNEigenSolver, SDPNEigenSolver
from ndgpu.spn import SDPNGroupOperator


def rectangles(widths=(1., 2.), height=1.):
    edges = np.r_[0., np.cumsum(widths)]
    coords = np.array([(x, y) for x in edges for y in (0., height)])
    cells = [(2*i, 2*i+2, 2*i+3, 2*i+1) for i in range(len(widths))]
    return assemble_mesh(coords, cells, [0]*len(cells))


def boxes(widths=(1., 2.), height=3., depth=4.):
    edges = np.r_[0., np.cumsum(widths)]
    coords = np.array([(x,y,z) for x in edges for z in (0.,depth) for y in (0.,height)])
    cells = [(4*i,4*i+4,4*i+5,4*i+1,4*i+2,4*i+6,4*i+7,4*i+3)
             for i in range(len(widths))]
    return assemble_mesh_3d(coords,cells,[0]*len(cells),normal_distances=True)


def materials():
    return [Material([.8, .3], [.07, .11], [.08, .16],
                     [[0, .03], [.01, 0]], [.85, .15], total=[.6, 1.2]),
            Material([1.7, .6], [.12, .08], [.04, .19],
                     [[0, .05], [.015, 0]], [.9, .1], total=[.4, .8])]


def dense_pencil(mesh, mats, cm, family, specs):
    """Assemble directly from FV face currents and literal order-1 tables.

    Does not use operator.apply/diag/coupling or production coefficient tables.
    Unknown order is (group, moment, cell).
    """
    if family == "diffusion":
        c0, ct, g = np.ones((1, 1)), np.zeros((1, 1)), np.array([[.5]])
    elif family == "spn":
        c0 = np.array([[1, -2/3], [-2/3, 4/9]])
        ct = np.array([[0, 0], [0, 5/9]])
        g = np.array([[1/2, -1/8], [-1/8, 7/24]])
    else:
        c0 = np.array([[1, -2/3], [-6/7, 4/7]])
        ct = np.array([[0, 0], [0, 5/7]])
        g = np.array([[1/2, -1/8], [-2/7, 23/42]])
    G, M, N = mats[0].n_groups, len(c0), mesh.n_cells
    A, F = np.zeros((G*M*N, G*M*N)), np.zeros((G*M*N, G*M*N))
    def ix(group, moment, cell):
        return (group*M + moment)*N + cell
    for group in range(G):
        D = np.array([[mats[m].diffusion[group] if i == 0 else
                       1/(7*mats[m].sigma_t[group]) for m in cm] for i in range(M)])
        for n, mat_index in enumerate(cm):
            mat, V = mats[mat_index], mesh.area[n]
            for i in range(M):
                for j in range(M):
                    A[ix(group,i,n),ix(group,j,n)] += V*(c0[i,j]*mat.removal[group] + ct[i,j]*mat.sigma_t[group])
                    for source in range(G):
                        F[ix(group,i,n),ix(source,j,n)] += V*c0[i,0]*mat.chi[group]*mat.nu_sigma_f[source]*c0[0,j]
                        if source != group:
                            A[ix(group,i,n),ix(source,j,n)] -= V*c0[i,0]*mat.sigma_s[source,group]*c0[0,j]
        for face, distances in zip(mesh.faces, mesh.face_distances):
            left, right, S, _ = face
            for i in range(M):
                w = S/(distances[0]/D[i,left] + distances[1]/D[i,right])
                for row, column, sign in ((left,left,1), (right,right,1), (left,right,-1), (right,left,-1)):
                    A[ix(group,i,row),ix(group,i,column)] += sign*w
        for (n,S,d), spec in zip(mesh.bfaces, specs):
            if spec == "reflective":
                continue
            if spec == "vacuum":
                # Directly eliminate traces in the two simultaneous boundary equations.
                aa = np.diag(D[:,n]/d)
                trace = np.linalg.solve(aa+g, aa)
                K = S*aa@(np.eye(M)-trace)
            else:
                K = np.diag(S*D[:,n]/d)
            for i in range(M):
                for j in range(M):
                    A[ix(group,i,n),ix(group,j,n)] += K[i,j]
    return A, F


def test_unequal_volume_resistance_conservation_and_action():
    mesh = rectangles()
    grid = MeshGrid(mesh, "reflective")
    D, r = np.array([.8, 1.7]), np.array([.07, .12])
    op = MeshGroupOperator(np, grid, D.reshape(grid.shape), r.reshape(grid.shape))
    w = 1/(.5/.8 + 1/1.7)
    expected = np.array([[w+.07, -w], [-w, w+.24]])
    phi = np.array([1.2, 3.4]).reshape(grid.shape)
    np.testing.assert_allclose(op.apply(phi).ravel(), expected@phi.ravel())
    leakage = op.apply(phi).ravel() - mesh.area*r*phi.ravel()
    assert abs(leakage.sum()) < 1e-14
    np.testing.assert_allclose(op.diag.ravel(), expected.diagonal())
    # Piecewise affine solution with equal current at an unequal-D interface.
    current = .27
    jump = current*(.5/.8 + 1/1.7)
    affine = np.array([2., 2.+jump]).reshape(grid.shape)
    np.testing.assert_allclose((op.apply(affine).ravel()-mesh.area*r*affine.ravel()), [-current,current])


@pytest.mark.parametrize("family,solver_cls", [("diffusion",MeshDiffusionEigenSolver), ("spn",MeshSP3EigenSolver), ("sdpn",MeshSDP1EigenSolver)])
@pytest.mark.parametrize("anderson_depth", [1, 5])
def test_dense_multigroup_eigenpencil_and_full_residual(family, solver_cls, anderson_depth):
    mesh, mats, cm = rectangles(), materials(), [0,1]
    specs = ["reflective" if i % 3 == 0 else "vacuum" for i in range(len(mesh.bfaces))]
    kw = {} if family == "diffusion" else dict(marshak_vacuum=True, symmetrize=False)
    solver = solver_cls(mesh, mats, cm, boundary_conditions=specs, device="cpu", **kw)
    result = solver.solve(tol_k=1e-11, tol_source=1e-10, inner_rtol_floor=1e-13,
                          anderson_depth=anderson_depth)
    assert result.converged
    A,F = dense_pencil(mesh, mats, cm, family, specs)
    probe = np.random.default_rng(419).normal(size=A.shape[0])
    probe_state = probe.reshape((2,) + solver.state[0].shape)
    action = []
    for group in range(2):
        row = solver.ops[group].apply(probe_state[group])
        for source in range(2):
            if source != group:
                scatter = solver.sigma_s[source][group]*solver._phi(probe_state[source])
                row = row - solver._rhs(group,scatter*solver._src_weight)
        action.append(row.ravel())
    np.testing.assert_allclose(np.concatenate(action),A@probe,rtol=2e-13,atol=2e-13)
    vals, vecs = np.linalg.eig(np.linalg.solve(A,F))
    index = np.argmax(vals.real)
    assert abs(result.k_eff - vals[index].real) < 2e-10
    state = np.concatenate([u.ravel() for u in solver.state])
    residual = A@state - F@state/result.k_eff
    assert np.linalg.norm(residual)/np.linalg.norm(A@state) < 2e-9
    ref = vecs[:,index].real
    np.testing.assert_allclose(state/state[0], ref/ref[0], rtol=2e-8, atol=2e-9)
    fission = sum(mats[m].nu_sigma_f[g]*result.flux[g,n,0,0]*mesh.area[n]
                  for g in range(2) for n,m in enumerate(cm))
    assert abs(fission/mesh.area.sum()-1) < 2e-12


@pytest.mark.parametrize("bc", ["vacuum", "reflective"])
def test_order_zero_matches_independent_unstructured_diffusion(bc):
    mesh, mats = rectangles(), materials()
    old = UnstructuredDiffusionSolver(mesh,mats,[0,1],alpha_boundary=.5 if bc == "vacuum" else 0, device="cpu").solve(tol_k=1e-10,tol_source=1e-9)
    new = MeshSP1EigenSolver(mesh,mats,[0,1],bc=bc,marshak_vacuum=True,device="cpu").solve(tol_k=1e-10,tol_source=1e-9)
    diff = MeshDiffusionEigenSolver(mesh,mats,[0,1],bc=bc,device="cpu").solve(tol_k=1e-10,tol_source=1e-9)
    assert old.converged and new.converged and diff.converged
    np.testing.assert_allclose([new.k_eff,diff.k_eff],old.k_eff,rtol=2e-9)


@pytest.mark.parametrize("family,order", [("spn",1),("sdpn",1),("spn",2),("sdpn",2),("sdpn",3)])
@pytest.mark.parametrize("boundary", ["vacuum","zero-flux","reflective"])
def test_uniform_cartesian_operator_and_eigenvalue_agreement(family, order, boundary):
    n, width, height = 3, 2., 4.
    mesh = rectangles([width]*n, height)
    # y and the unused z direction are reflective on both representations.
    specs = []
    for owner,S,d in mesh.bfaces:
        specs.append(boundary if S == height else "reflective")
    mesh_cls, cart_cls = ((MeshSPNEigenSolver,SPNEigenSolver) if family == "spn" else
                         (MeshSDPNEigenSolver,SDPNEigenSolver))
    kw = dict(marshak_vacuum=True, symmetrize=False, device="cpu")
    m = mesh_cls(mesh,materials(),[0,1,0],order=order,boundary_conditions=specs,**kw)
    # Existing general classes select order through their class attribute.
    cart = type("OrderSolver", (cart_cls,), {"_order":order})(Grid((n,1,1),(n*width,height,1)),materials(),
          material_map=np.array([0,1,0]).reshape(n,1,1),bc=(boundary,"reflective","reflective"),**kw)
    u = np.arange((order+1)*n,dtype=float).reshape((order+1,n,1,1))*.13+.5
    for a,b in zip(m.ops,cart.ops):
        np.testing.assert_allclose(a.apply(u),width*height*b.apply(u),rtol=2e-13,atol=2e-13)
    mr, cr = [s.solve(tol_k=1e-9,tol_source=1e-8) for s in (m,cart)]
    assert mr.converged and cr.converged
    assert abs(mr.k_eff-cr.k_eff) < 2e-8


@pytest.mark.parametrize("M,g", [(1,[[.5]]),(2,[[.5,-.125],[-.125,7/24]])])
def test_one_cell_multiple_mixed_faces_direct_trace_elimination(M,g):
    mesh = rectangles([2.],3.)
    specs = ("vacuum","reflective","vacuum","zero-flux")
    grid = MeshGrid(mesh,specs)
    Ds = [np.full(grid.shape,.7 if i == 0 else 1/7) for i in range(M)]
    u = np.arange(1.,M+1)
    blocks = grid.boundary_blocks(np,None,Ds,g)
    assert len(blocks) == 1 and len(blocks[0][0][0]) == 1
    direct = np.zeros(M)
    for (owner,S,d),spec in zip(mesh.bfaces,specs):
        if spec == "vacuum":
            A = np.diag([D.item()/d for D in Ds])
            surface = np.linalg.solve(A+g,A@u)
            direct += S*A@(u-surface)
    np.testing.assert_allclose(blocks[0][1][0]@u,direct)
    if M == 1:
        scalar = MeshGroupOperator(np,grid,Ds[0],np.zeros(grid.shape))
        stripped = MeshGroupOperator(np,grid,Ds[0],np.zeros(grid.shape),bc=grid.moment_boundary_conditions(strip_vacuum=True))
        np.testing.assert_allclose(scalar.diag-stripped.diag,direct.item())
    # Coupled assembly includes the Dirichlet face, strips only the vacuum,
    # and applies the aggregated block once; no-interior-face gather is valid.
    op = SDPNGroupOperator(np,grid,Ds[0],np.full(grid.shape,1.),np.full(grid.shape,.2),
                            order=M-1,coeffs={M-1:[np.eye(M).tolist()]*M},
                            op_cls=MeshGroupOperator,boundary_g=g,bc=None)
    state = u.reshape((M,)+grid.shape)
    stripped_action = np.stack([o.apply(state[i]) for i,o in enumerate(op.moments)])
    np.testing.assert_allclose((op.apply(state)-stripped_action).ravel(),direct)


@pytest.mark.parametrize("kwargs,match", [({"active":[True,True]},"masks"),
    ({"hybrid_mask":[True,True]},"masks"),({"df":np.ones(2)},"DF-off"),
    ({"bcf":np.ones(2)},"DF-off"),({"spn_precondition":2},"preconditioning"),
    ({"order":3,"symmetrize":True},"congruence"),
    ({"marshak_vacuum":True,"linear_solver":"cg"},"nonsymmetric")])
def test_unsupported_capabilities_rejected(kwargs,match):
    with pytest.raises(ValueError,match=match):
        MeshSDPNEigenSolver(rectangles(),materials(),[0,1],device="cpu",**kwargs)


def test_adjoint_and_invalid_geometry_rejected():
    solver = MeshDiffusionEigenSolver(rectangles(),materials(),[0,1],device="cpu")
    with pytest.raises(ValueError,match="adjoints"):
        solver.solve(adjoint=True)
    mesh = rectangles()
    mesh.face_distances[0,0] = 0
    with pytest.raises(ValueError,match="face_distances"):
        MeshGrid(mesh)


def test_one_cell_independent_diffusion_solver_empty_faces():
    solver = UnstructuredDiffusionSolver(rectangles([2.]),[materials()[0]],[0],device="cpu")
    assert solver.solve(tol_k=1e-9,tol_source=1e-8).converged


def test_three_dimensional_normal_distances_opt_in_preserves_legacy_default():
    coords = np.array([[0,0,0],[2,0,0],[0,3,0],[0,0,4]],dtype=float)
    old = assemble_mesh_3d(coords,[(0,1,2,3)],[0])
    mesh = assemble_mesh_3d(coords,[(0,1,2,3)],[0],normal_distances=True)
    assert old.face_distances is None
    assert mesh.face_distances.shape == (0,2)
    # Right tetrahedron centroid has normal distance .5 from x=0.
    np.testing.assert_allclose(sorted(b[2] for b in mesh.bfaces),sorted([.5,.75,1.,3/np.sqrt(61)]))
    op = MeshDiffusionEigenSolver(mesh,materials()[0],device="cpu")
    assert op.solve(tol_k=1e-9,tol_source=1e-8).converged


@pytest.mark.parametrize("family,solver_cls", [("diffusion",MeshDiffusionEigenSolver),
    ("spn",MeshSP3EigenSolver),("sdpn",MeshSDP1EigenSolver)])
def test_unequal_three_dimensional_multigroup_pencil_and_scalar_flux(family,solver_cls):
    mesh, mats, cm = boxes(), materials(), [0,1]
    np.testing.assert_allclose(mesh.area,[12.,24.])
    np.testing.assert_allclose(mesh.face_distances,[[.5,1.]])
    # Each owner has five exposed faces with all three boundary classes.
    classes = ("reflective","vacuum","zero-flux")
    specs = [classes[i%3] for i in range(len(mesh.bfaces))]
    for owner in (0,1):
        assert len({s for f,s in zip(mesh.bfaces,specs) if f[0] == owner}) == 3
    kw = {} if family == "diffusion" else dict(marshak_vacuum=True,symmetrize=False)
    solver = solver_cls(mesh,mats,cm,boundary_conditions=specs,device="cpu",**kw)
    result = solver.solve(tol_k=1e-11,tol_source=1e-10,inner_rtol_floor=1e-13)
    assert result.converged
    A,F = dense_pencil(mesh,mats,cm,family,specs)
    vals,vecs = np.linalg.eig(np.linalg.solve(A,F))
    index = np.argmax(vals.real)
    assert abs(result.k_eff-vals[index].real) < 2e-10
    state = np.concatenate([s.ravel() for s in solver.state])
    assert np.linalg.norm(A@state-F@state/result.k_eff)/np.linalg.norm(A@state) < 2e-9
    reference = vecs[:,index].real.reshape((2,-1,2))
    phi = reference[:,0] if family == "diffusion" else reference[:,0]-2/3*reference[:,1]
    np.testing.assert_allclose(result.flux.reshape(2,2)/result.flux[0,0,0,0],
                               phi/phi[0,0],rtol=2e-8,atol=2e-9)
    # Full source balance after projection back to scalar flux: F contains the
    # physical scalar fission density and reaction rows are integrated once.
    source = sum(solver.nu_sigma_f[g]*result.flux[g] for g in range(2))
    np.testing.assert_allclose(np.sum(source*solver.integration_weights),mesh.area.sum(),rtol=2e-12)


@pytest.mark.parametrize('solver_cls', [MeshDiffusionEigenSolver,MeshSP3EigenSolver,MeshSDP1EigenSolver])
def test_integer_like_float_boundary_owners_preserve_operator_action(solver_cls):
    from dataclasses import replace
    mesh=boxes()
    floating=replace(mesh,bfaces=[(float(owner),area,distance) for owner,area,distance in mesh.bfaces])
    kwargs={} if solver_cls is MeshDiffusionEigenSolver else {'marshak_vacuum':True,'symmetrize':False}
    expected=solver_cls(mesh,materials(),[0,1],device='cpu',**kwargs)
    actual=solver_cls(floating,materials(),[0,1],device='cpu',**kwargs)
    for op,ref in zip(actual.ops,expected.ops):
        shape=actual._initial_state()[0].shape
        probe=np.arange(np.prod(shape),dtype=float).reshape(shape)+1
        np.testing.assert_allclose(op.apply(probe),ref.apply(probe))
