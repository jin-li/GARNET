"""Integration checks for the opt-in corrected common-mesh eigen drivers."""
import numpy as np
import pytest
from ndgpu import Material, MeshDiffusionEigenSolver, MeshSP3EigenSolver, MeshSDP1EigenSolver
from ndgpu.mesh import assemble_mesh


def square_mesh(shear=0.):
    coords = [(x + shear*y, y) for y in (0., .4, 1.) for x in (0., .3, 1.)]
    cells = [(j*3+i, j*3+i+1, (j+1)*3+i+1, (j+1)*3+i) for j in range(2) for i in range(2)]
    return assemble_mesh(coords, cells, [0]*4)


@pytest.mark.parametrize('solver_cls',[MeshDiffusionEigenSolver,MeshSP3EigenSolver,MeshSDP1EigenSolver])
def test_corrected_reflective_homogeneous_limit(solver_cls):
    material = Material([.7],[.03],[.035],[[.2]],[1.],total=[.23])
    solver = solver_cls(square_mesh(.25),[material],[0]*4,
                        boundary_conditions='reflective',device='cpu',mesh_discretization='corrected')
    result=solver.solve(tol_k=1e-8,tol_source=1e-7)
    assert result.converged
    assert abs(result.k_eff-.035/.03)<2e-7
    assert all(not op.symmetric for op in solver.ops)


@pytest.mark.parametrize('solver_cls',[MeshDiffusionEigenSolver,MeshSP3EigenSolver,MeshSDP1EigenSolver])
def test_corrected_orthogonal_eigenvalue_matches_tpfa(solver_cls):
    mats=[Material([.7],[.03],[.035],[[.2]],[1.],total=[.23]),
          Material([.4],[.06],[.07],[[.2]],[1.],total=[.26])]
    kw={} if solver_cls is MeshDiffusionEigenSolver else {'marshak_vacuum':True,'symmetrize':False}
    common=dict(boundary_conditions='vacuum',device='cpu',**kw)
    a=solver_cls(square_mesh(),mats,[0,1,0,1],**common)
    b=solver_cls(square_mesh(),mats,[0,1,0,1],mesh_discretization='corrected',**common)
    ra=a.solve(tol_k=1e-8,tol_source=1e-7)
    rb=b.solve(tol_k=1e-8,tol_source=1e-7)
    assert ra.converged and rb.converged
    assert abs(ra.k_eff-rb.k_eff)<2e-7


def test_corrected_cg_and_unknown_discretization_rejected():
    material=Material([1.],[.1],[.12])
    with pytest.raises(ValueError,match='CG'):
        MeshDiffusionEigenSolver(square_mesh(),material,mesh_discretization='corrected',linear_solver='cg')
    with pytest.raises(ValueError,match='mesh_discretization'):
        MeshDiffusionEigenSolver(square_mesh(),material,mesh_discretization='other')
