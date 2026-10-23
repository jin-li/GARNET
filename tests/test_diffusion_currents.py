"""Face-law, geometry and actual-operator tests without eigenvalue solves."""
from copy import deepcopy

import numpy as np
import pytest

from garnet.diffusion_currents import export_diffusion_face_currents
from garnet.grid import Grid
from garnet.materials import Material
from garnet.nodal_mesh import build_rectilinear_mesh_3d
from garnet.solver import DiffusionEigenSolver, MeshDiffusionEigenSolver, MeshSP3EigenSolver


def materials():
    return [Material([1.2,.3],[.04,.12],[.1,.02],[[0,.02],[.01,0]]),
            Material([.4,1.8],[.09,.2],[.03,.01],[[0,.01],[.03,0]])]


def mesh_solver(bc='vacuum'):
    mesh,geometry=build_rectilinear_mesh_3d([-2.,-1.,2.],[0.,2.,5.],[1.,2.,6.],
        active_mask=np.ones((2,2,2),dtype=bool),tags=np.zeros((2,2,2),dtype=int))
    solver=MeshDiffusionEigenSolver(mesh,materials(),np.arange(mesh.n_cells)%2,bc=bc,device='cpu')
    solver.state=[np.arange(1.,mesh.n_cells+1).reshape(solver.grid.shape),
                  np.arange(mesh.n_cells+1.,1.,-1).reshape(solver.grid.shape)]
    return solver,geometry


def test_variable_widths_diffusion_true_centers_and_independent_matrix():
    solver,geometry=mesh_solver()
    before=[s.copy() for s in solver.state]
    result=export_diffusion_face_currents(solver,geometry=geometry,flux=np.stack(solver.state))
    mesh=solver.grid.mesh
    n=mesh.n_cells
    for g in range(2):
        phi=solver.state[g].ravel()
        D=solver.fields.diffusion[g].ravel()
        R=solver.fields.removal[g].ravel()
        matrix=np.diag(R*mesh.area)
        for f,(i,j,S,separation) in enumerate(mesh.faces):
            di,dj=mesh.face_distances[f]
            w=S/(di/D[i]+dj/D[j])
            matrix[i,i]+=w;matrix[j,j]+=w
            matrix[i,j]-=w;matrix[j,i]-=w
            assert result['interior']['integrated_current'][g,f]==pytest.approx(w*(phi[i]-phi[j]))
        for f,(i,S,d) in enumerate(mesh.bfaces):
            matrix[i,i]+=S*.5*D[i]/(d*.5+D[i])
        np.testing.assert_allclose(result['integrated_loss'][g],matrix@phi,rtol=2e-14,atol=2e-14)
        np.testing.assert_allclose(result['operator_integrated_loss'][g],matrix@phi,rtol=2e-14,atol=2e-14)
    owners=result['interior']['owner'];neighbors=result['interior']['neighbor']
    midpoint=(mesh.centroid[owners]+mesh.centroid[neighbors])/2
    assert np.max(abs(midpoint-result['interior']['center']))>.1
    # Native Cartesian interval endpoints, not centroid midpoints.
    assert result['interior']['center'][0,0]==pytest.approx(-1.)
    np.testing.assert_array_equal(result['interior']['neighbor_integrated_current'],-result['interior']['integrated_current'])
    for old,new in zip(before,solver.state):np.testing.assert_array_equal(old,new)


def test_vacuum_boundary_law_and_centers():
    solver,geometry=mesh_solver()
    result=export_diffusion_face_currents(solver,geometry=geometry)
    b=result['boundary'];D=np.stack([d.ravel() for d in solver.fields.diffusion]);phi=np.stack([s.ravel() for s in solver.state])
    expected=.5*D[:,b['owner']]*phi[:,b['owner']]/(b['normal_distance']*.5+D[:,b['owner']])
    np.testing.assert_allclose(b['current_density'],expected,rtol=2e-15)
    np.testing.assert_allclose(b['center'],solver.grid.mesh.centroid[b['owner']]+b['normal']*b['normal_distance'][:,None])


def test_reflective_boundary_zero_and_constant_state_reaction_only():
    solver,geometry=mesh_solver('reflective')
    solver.state=[np.full(solver.grid.shape,3.) for g in range(2)]
    result=export_diffusion_face_currents(solver,geometry=geometry)
    np.testing.assert_array_equal(result['boundary']['integrated_current'],0.)
    np.testing.assert_array_equal(result['interior']['integrated_current'],0.)
    np.testing.assert_allclose(result['integrated_loss'],result['cell_reaction'],rtol=0,atol=0)


def test_structured_density_operator_has_volume_once():
    grid=Grid((3,2,1),(9.,4.,5.))
    bc=('reflective','vacuum','zero-flux','vacuum','reflective','vacuum')
    solver=DiffusionEigenSolver(grid,materials(),np.indices(grid.shape).sum(axis=0)%2,bc=bc,device='cpu')
    solver.state=[np.arange(1.,7.).reshape(grid.shape),np.arange(8.,2.,-1).reshape(grid.shape)]
    result=export_diffusion_face_currents(solver,boundary_conditions=bc)
    np.testing.assert_allclose(result['integrated_loss'],
        np.stack([op.apply(s).ravel() for op,s in zip(solver.ops,solver.state)])*grid.cell_volume,rtol=1e-14)
    assert result['diagnostics']['operator_form']=='density'
    with pytest.raises(ValueError,match='must be explicit'):
        export_diffusion_face_currents(solver)
    with pytest.raises(ValueError,match='differs from actual diffusion operator'):
        export_diffusion_face_currents(solver,boundary_conditions='reflective')


def test_malformed_geometry_missing_coverage_or_wrong_centroid_rejected():
    solver,geometry=mesh_solver()
    with pytest.raises(ValueError,match='requires widths'):
        export_diffusion_face_currents(solver)
    for key in ('widths','face_sides','boundary_sides'):
        changed=deepcopy(geometry)
        if key=='widths':changed[key][0,0]*=1.1
        elif key=='face_sides':changed[key][0,1]=changed[key][0,0]
        else:changed[key][0]=(int(changed[key][0])+2)%4
        with pytest.raises(ValueError):export_diffusion_face_currents(solver,geometry=changed)
    solver.grid.mesh.centroid[1,1]+=.1
    with pytest.raises(ValueError,match='face centers'):
        export_diffusion_face_currents(solver,geometry=geometry)


def test_wrong_flux_nonunity_adf_moment_and_nonfinite_coefficients_rejected():
    solver,geometry=mesh_solver()
    with pytest.raises(ValueError,match='flux differs'):
        export_diffusion_face_currents(solver,geometry=geometry,flux=np.stack(solver.state)*2)
    for name in ('face_df','boundary_df'):
        old=getattr(solver,name).copy();getattr(solver,name).flat[0]=1.01
        with pytest.raises(ValueError,match='ADF off'):
            export_diffusion_face_currents(solver,geometry=geometry)
        setattr(solver,name,old)
    solver.fields.diffusion[0].flat[0]=np.nan
    with pytest.raises(ValueError,match='diffusion coefficients'):
        export_diffusion_face_currents(solver,geometry=geometry)
    moment=MeshSP3EigenSolver(solver.grid.mesh,materials(),np.arange(8)%2,bc='vacuum',device='cpu')
    with pytest.raises(ValueError,match='Only scalar diffusion'):
        export_diffusion_face_currents(moment,geometry=geometry)


def test_operator_mismatch_detected_without_mutating_operators():
    solver,geometry=mesh_solver()
    solver.ops[0]._flat.diag[0]*=1.01
    with pytest.raises(ValueError,match='differs from actual diffusion operator'):
        export_diffusion_face_currents(solver,geometry=geometry)


def test_legacy_equal_distance_geometry_is_not_invented_for_unequal_cells():
    solver,geometry=mesh_solver()
    solver.grid.mesh.face_distances=None
    # Half centroid spacing would put the interface at the wrong physical plane.
    with pytest.raises(ValueError,match='interior half widths'):
        export_diffusion_face_currents(solver,geometry=geometry)


def test_exported_geometry_and_currents_do_not_alias_solver_inputs():
    solver,geometry=mesh_solver()
    original_distances=solver.grid.mesh.face_distances.copy()
    original_state=[s.copy() for s in solver.state]
    result=export_diffusion_face_currents(solver,geometry=geometry)
    result['interior']['normal_distances'][:]=99.
    result['interior']['integrated_current'][:]=99.
    result['boundary']['center'][:]=99.
    np.testing.assert_array_equal(solver.grid.mesh.face_distances,original_distances)
    for expected,actual in zip(original_state,solver.state):
        np.testing.assert_array_equal(actual,expected)
