"""Independent conservative scalar-DF checks on the common mesh path."""
from dataclasses import replace

import numpy as np
import pytest

from ndgpu import (MeshGrid, MeshGroupOperator, MeshDiffusionEigenSolver,
                   MeshSP1EigenSolver, UnstructuredDiffusionSolver,
                   Material, assemble_mesh)
from ndgpu.linalg import pcg


def mesh_two():
    coords = np.array([(x, y) for x in (0., 1., 3.) for y in (0., 2.)])
    return assemble_mesh(coords, [(0, 2, 3, 1), (2, 4, 5, 3)], [0, 0])


def mats():
    return [Material([.8, .3], [.07, .11], [.08, .16],
                     [[0, .03], [.01, 0]], [.85, .15], total=[.6, 1.2]),
            Material([1.7, .6], [.12, .08], [.04, .19],
                     [[0, .05], [.015, 0]], [.9, .1], total=[.4, .8])]


def operator(mesh, factors, boundary=None, specs="reflective", removal=(.07, .12)):
    grid = MeshGrid(mesh, specs)
    return MeshGroupOperator(np, grid, np.array([.8, 1.7]).reshape(grid.shape),
                             np.asarray(removal).reshape(grid.shape),
                             face_df=factors, boundary_df=boundary)


def matrix(op):
    return np.column_stack([op.apply(v.reshape(op.grid.shape)).ravel() for v in np.eye(2)])


def test_literal_asymmetric_matrix_resistance_and_integrated_conservation():
    mesh = mesh_two()
    op = operator(mesh, [[1.2, .7]])
    w = 2 / (1.2*.5/.8 + .7*1/1.7)
    expected = np.array([[.14+1.2*w, -.7*w], [-1.2*w, .48+.7*w]])
    np.testing.assert_allclose(matrix(op), expected, atol=1e-14)
    assert not op.symmetric
    phi = np.array([2.7, .4])
    current = w * (1.2*phi[0] - .7*phi[1])
    leakage = op.apply(phi.reshape(op.grid.shape)).ravel() - mesh.area*np.array([.07,.12])*phi
    np.testing.assert_allclose(leakage, [current, -current], atol=1e-14)
    assert abs(leakage.sum()) < 1e-14
    # A constant corrected trace f_i*phi_i produces exactly zero current.
    homogeneous = (3 / np.array([1.2, .7])).reshape(op.grid.shape)
    pure = operator(mesh, [[1.2, .7]], removal=(0., 0.))
    np.testing.assert_allclose(pure.apply(homogeneous), 0, atol=2e-14)


def test_refined_subfaces_have_same_total_current():
    mesh = mesh_two()
    i,j,S,d = mesh.faces[0]
    split = replace(mesh, faces=[(i,j,.3*S,d),(i,j,.7*S,d)],
                    face_distances=np.repeat(mesh.face_distances, 2, axis=0))
    np.testing.assert_allclose(matrix(operator(split, [[1.2,.7],[1.2,.7]])),
                               matrix(operator(mesh, [[1.2,.7]])), atol=1e-14)


@pytest.mark.parametrize("normal", [False, True])
def test_unit_df_preserves_legacy_action_and_cg(normal):
    mesh = mesh_two()
    if not normal:
        mesh = replace(mesh, face_distances=None)
    a = operator(mesh, None)
    b = operator(mesh, np.ones((1,2)), np.ones(len(mesh.bfaces)))
    np.testing.assert_array_equal(matrix(a), matrix(b))
    assert a.symmetric and b.symmetric
    D = np.array([.8,1.7])
    d = mesh.faces[0][3]
    w = (2*D[0]*D[1]/((D[0]+D[1])*d)*2 if not normal else 2/(.5/.8+1/1.7))
    np.testing.assert_allclose(matrix(a), [[.14+w,-w],[-w,.48+w]])
    solver = MeshDiffusionEigenSolver(mesh,mats(),[0,1],device="cpu",face_df=np.ones((2,1,2)))
    assert solver._linsolve is pcg


def test_missing_normal_distances_uses_equal_split_resistance():
    mesh = replace(mesh_two(), face_distances=None)
    op = operator(mesh, [[1.2,.7]])
    d = mesh.faces[0][3]
    w = 2/(1.2*d/2/.8 + .7*d/2/1.7)
    np.testing.assert_allclose(matrix(op), [[.14+1.2*w,-.7*w],[-1.2*w,.48+.7*w]])


def test_mixed_boundary_law_direct_trace_elimination():
    mesh = mesh_two()
    specs = ["reflective", "zero-flux", "vacuum", .23, "vacuum", "reflective"]
    factors = np.array([.4, 1.8, .7, 1.3, 2.1, .9])
    op = operator(mesh, [[1.2,.7]], factors, specs)
    base = operator(mesh, [[1.2,.7]])
    extra = np.zeros(2)
    for (owner,S,d),spec,f in zip(mesh.bfaces,specs,factors):
        D = [.8,1.7][owner]
        if spec == "reflective":
            current = 0
        elif spec == "zero-flux":
            current = D/d
        else:
            alpha = .5 if spec == "vacuum" else spec
            # D/d*(phi_c-phi_s) = alpha*f*phi_s, with phi_c=1.
            trace = (D/d)/(D/d+alpha*f)
            current = D/d*(1-trace)
        extra[owner] += S*current
    np.testing.assert_allclose(matrix(op)-matrix(base), np.diag(extra),atol=1e-14)


@pytest.mark.parametrize("name,value", [
    ("face_df", np.ones((2,1))), ("face_df", np.zeros((2,1,2))),
    ("face_df", np.full((2,1,2),np.nan)), ("face_df", np.full((2,1,2),np.inf)),
    ("boundary_df", np.ones((2,5))), ("boundary_df", -np.ones((2,6))),
    ("boundary_df", np.full((2,6),np.nan)), ("boundary_df", np.full((2,6),np.inf))])
def test_invalid_df_shapes_and_factors(name,value):
    with pytest.raises(ValueError,match=name):
        MeshDiffusionEigenSolver(mesh_two(),mats(),[0,1],device="cpu",**{name:value})


def test_nonsymmetric_auto_solver_explicit_cg_rejection_and_moment_gate():
    factors = np.array([[[1.2,.7]],[[.9,1.1]]])
    solver = MeshDiffusionEigenSolver(mesh_two(),mats(),[0,1],face_df=factors,device="cpu")
    assert solver._linsolve is not pcg
    assert all(not op.symmetric for op in solver.ops)
    for cls in (MeshDiffusionEigenSolver, UnstructuredDiffusionSolver):
        with pytest.raises(ValueError, match="CG.*nonsymmetric"):
            cls(mesh_two(),mats(),[0,1],face_df=factors,linear_solver="cg",device="cpu")
    with pytest.raises(ValueError,match="diffusion-only"):
        MeshSP1EigenSolver(mesh_two(),mats(),[0,1],face_df=factors,device="cpu")
    with pytest.raises(ValueError,match="diffusion-only"):
        MeshSP1EigenSolver(mesh_two(),mats(),[0,1],boundary_df=np.ones((2,6)),device="cpu")


def test_dense_nonuniform_multigroup_df_eigenpencil_and_legacy_adapter():
    mesh,materials = mesh_two(),mats()
    factors = np.array([[[1.2,.7]],[[.9,1.1]]])
    boundary = np.array([[.8,1.1,.7,1.2,1.3,.6],[1.1,.9,1.3,.8,.7,1.2]])
    solver = MeshDiffusionEigenSolver(mesh,materials,[0,1],face_df=factors,
                                    boundary_df=boundary,bc="vacuum",device="cpu")
    result = solver.solve(tol_k=1e-11,tol_source=1e-10,inner_rtol_floor=1e-13)
    assert result.converged
    A,F = np.zeros((4,4)),np.zeros((4,4))
    for g in range(2):
        D = [m.diffusion[g] for m in materials]
        fl,fr = factors[g,0]
        w = 2/(fl*.5/D[0] + fr*1/D[1])
        A[2*g:2*g+2,2*g:2*g+2] += [[w*fl,-w*fr],[-w*fl,w*fr]]
        for n,m in enumerate(materials):
            row = 2*g+n
            A[row,row] += mesh.area[n]*m.removal[g]
            for source in range(2):
                F[row,2*source+n] += mesh.area[n]*m.chi[g]*m.nu_sigma_f[source]
                if source != g:
                    A[row,2*source+n] -= mesh.area[n]*m.sigma_s[source,g]
        for b,(n,S,d) in enumerate(mesh.bfaces):
            # Direct trace elimination, independent of production boundary formula.
            a = D[n]/d
            trace = a/(a+.5*boundary[g,b])
            A[2*g+n,2*g+n] += S*a*(1-trace)
    eig,vec = np.linalg.eig(np.linalg.solve(A,F))
    index = np.argmax(eig.real)
    np.testing.assert_allclose(result.k_eff,eig[index].real,rtol=2e-9)
    phi = result.flux.ravel()
    np.testing.assert_allclose(phi/phi[0],vec[:,index].real/vec[0,index].real,rtol=2e-8)
    assert np.linalg.norm(A@phi-F@phi/result.k_eff)/np.linalg.norm(A@phi) < 2e-9
    assert abs(phi[0]-phi[1]) > .01
    old = UnstructuredDiffusionSolver(mesh,materials,[0,1],face_df=factors,boundary_df=boundary,device="cpu")
    for g in range(2):
        probe = np.array([.3,2.1])
        np.testing.assert_allclose(old.ops[g].apply(probe),solver.ops[g].apply(probe.reshape(mesh.n_cells,1,1)).ravel())
    legacy = old.solve(tol_k=1e-10,tol_source=1e-9)
    assert legacy.converged
    np.testing.assert_allclose(legacy.k_eff,result.k_eff,rtol=3e-9)


def test_single_cell_scalar_df_and_no_interior_faces():
    coords = np.array([[0.,0.],[1.,0.],[1.,2.],[0.,2.]])
    mesh = assemble_mesh(coords,[(0,1,2,3)],[0])
    material = mats()[0]
    factors = np.array([[.7,1.1,.8,1.2],[1.1,.9,1.3,.8]])
    solver = MeshDiffusionEigenSolver(mesh,material,face_df=np.ones((2,0,2)),
                                    boundary_df=factors,device="cpu")
    assert solver._linsolve is pcg
    for g,op in enumerate(solver.ops):
        D = material.diffusion[g]
        loss = material.removal[g]*mesh.area[0]
        for b,(_,S,d) in enumerate(mesh.bfaces):
            trace = (D/d)/(D/d+.5*factors[g,b])
            loss += S*(D/d)*(1-trace)
        np.testing.assert_allclose(op.apply(np.ones(op.grid.shape)).ravel(),[loss])
        assert op.symmetric
