"""Independent analytical checks for the opt-in corrected scalar stencil."""
from dataclasses import replace

import numpy as np
import pytest

from garnet.mesh import MeshGrid, MeshGroupOperator, assemble_mesh
from garnet.corrected_mesh import CorrectedMeshGroupOperator
from garnet.linalg import gmres


def mesh_grid(n=5, shear=.35):
    coords = np.array([(x+shear*y, y) for y in np.linspace(0, 1, n+1)
                       for x in np.linspace(0, 1, n+1)])
    cells = [(j*(n+1)+i, j*(n+1)+i+1, (j+1)*(n+1)+i+1, (j+1)*(n+1)+i)
             for j in range(n) for i in range(n)]
    return assemble_mesh(coords, cells, np.zeros(n*n, dtype=int))


def operator(mesh, D=1., removal=.2, bc="reflective", **kw):
    grid = MeshGrid(mesh, bc)
    return CorrectedMeshGroupOperator(np, grid, np.broadcast_to(D, (grid.n_cells,)).copy(),
                                      np.full(grid.shape, removal), **kw)


def test_linear_conservative_and_reflective_constant():
    mesh = mesh_grid()
    op = operator(mesh, D=np.linspace(.4, 3., mesh.n_cells))
    rng = np.random.default_rng(301)
    u, v = rng.normal(size=(2, mesh.n_cells))
    np.testing.assert_allclose(op.apply((2*u-3*v)).ravel(),
                               2*op.apply(u).ravel()-3*op.apply(v).ravel(), atol=2e-14)
    np.testing.assert_allclose(op.apply(np.ones(mesh.n_cells)).ravel(), .2*mesh.area, atol=1e-14)
    interior, boundary = op.face_currents(u)
    expected = .2*mesh.area*u
    for (i, j, _, _), current in zip(mesh.faces, interior):
        expected[i] += current
        expected[j] -= current
    for (i, _, _), current in zip(mesh.bfaces, boundary):
        expected[i] += current
    np.testing.assert_allclose(op.apply(u).ravel(), expected, atol=1e-14)
    assert abs((op.apply(u).ravel()-.2*mesh.area*u).sum()) < 3e-14
    assert not op.symmetric
    np.testing.assert_allclose(op.diag.ravel(), op.host_matrix.diagonal())


@pytest.mark.parametrize("bc", ["reflective", "vacuum", "zero-flux", .31])
def test_orthogonal_agreement(bc):
    mesh = mesh_grid(shear=0.)
    D = np.linspace(.5, 2., mesh.n_cells)
    op = operator(mesh, D, bc=bc)
    ref = MeshGroupOperator(np, op.grid, D, np.full(mesh.n_cells, .2))
    u = np.linspace(-.2, 3., mesh.n_cells)
    np.testing.assert_allclose(op.apply(u), ref.apply(u), rtol=1e-13, atol=1e-13)


def test_piecewise_linear_interface_current_away_from_boundary():
    # The interface x=.5 has continuous trace and D*du/dx; du/dy is shared.
    mesh = mesh_grid(8, shear=0.)
    x, y = mesh.centroid.T
    D = np.where(x < .5, .7, 3.1)
    q, tangent = 1.3, -.4
    u = 2. + q*(x-.5)/D + tangent*y
    op = operator(mesh, D)
    current, _ = op.face_currents(u)
    checked = 0
    for k, (i, j, area, _) in enumerate(mesh.faces):
        if D[i] == D[j] or not (.2 < y[i] < .8):
            continue
        normal = mesh.face_normals[k]
        expected = -area*np.dot([q, D[i]*tangent], normal)
        np.testing.assert_allclose(current[k], expected, atol=1e-13)
        # Verify the actual gradient, not only cancellation in the flux.
        for owner in (i, j):
            grad = sum((vector*u[col] for col, vector in op.gradient_stencils[owner].items()))
            np.testing.assert_allclose(grad, [q/D[owner], tangent], atol=1e-13)
        checked += 1
    assert checked == 4


def test_skew_interface_current_with_supplied_linear_gradient_oracle():
    mesh = mesh_grid(6)
    x, y = mesh.centroid.T
    # A material interface parallel to the slanted grid edges has normal n.
    normal = np.array([1., -.35])/np.hypot(1, .35)
    tangent = np.array([.35, 1.])/np.hypot(1, .35)
    coordinate = mesh.centroid@normal
    interface = .5/np.hypot(1, .35)
    D = np.where(coordinate < interface, .6, 2.7)
    u = 3.+1.2*(coordinate-interface)/D+.3*(mesh.centroid@tangent)
    exact = 1.2/D[:, None]*normal+.3*tangent
    # Fixed linear oracle: the anchor is nonzero and its coefficient is fixed.
    stencils = [{0: g/u[0]} for g in exact]
    op = operator(mesh, D, reconstruction_policy="supplied", gradient_stencils=stencils)
    currents, _ = op.face_currents(u)
    checked = 0
    for k, (i, j, area, _) in enumerate(mesh.faces):
        if D[i] != D[j]:
            expected = -area*1.2*np.dot(normal, mesh.face_normals[k])
            np.testing.assert_allclose(currents[k], expected, atol=2e-14)
            checked += 1
    assert checked == 6


def test_skew_manufactured_solution_converges_with_dense_independent_solve():
    errors = []
    for n in (4, 8, 16):
        mesh = mesh_grid(n, shear=.4)
        op = operator(mesh, removal=.3, bc="zero-flux")
        x, y = mesh.centroid.T
        X, Y = x-.4*y, y
        exact = np.sin(np.pi*X)*np.sin(np.pi*Y)
        source = np.pi**2*((2+.4**2)*exact + .8*np.cos(np.pi*X)*np.cos(np.pi*Y))+.3*exact
        # Build the dense matrix by basis actions, independently of stored CSR.
        dense = np.column_stack([op.apply(e).ravel() for e in np.eye(mesh.n_cells)])
        rhs = source*mesh.area
        solution = np.linalg.solve(dense, rhs)
        errors.append(np.sqrt(np.sum(mesh.area*(solution-exact)**2)))
        if n == 4:
            result = gmres(op.apply, rhs.reshape(op.grid.shape), np.zeros(op.grid.shape),
                           op.inv_diag, np, rtol=1e-11, maxiter=1000)
            np.testing.assert_allclose(result[0].ravel(), solution, atol=1e-9)
    assert errors[0]/errors[1] > 3.
    assert errors[1]/errors[2] > 3.


def test_rank_deficiency_and_reconstruction_policy_are_explicit():
    mesh = mesh_grid(2)
    # Removing all boundary equations and vertical neighbors leaves only x rows.
    keep = [k for k, normal in enumerate(mesh.face_normals) if abs(normal[0]) > .9]
    thin = replace(mesh, faces=[mesh.faces[k] for k in keep],
                   face_centers=mesh.face_centers[keep], face_normals=mesh.face_normals[keep],
                   face_distances=mesh.face_distances[keep], bfaces=[],
                   bface_centers=np.empty((0, 2)), bface_normals=np.empty((0, 2)))
    with pytest.raises(ValueError, match="rank-deficient"):
        operator(thin)
    with pytest.raises(ValueError, match="requires gradient_stencils"):
        operator(mesh, reconstruction_policy="coupled_moments")
    with pytest.raises(ValueError, match="requires finite face_centers"):
        operator(replace(mesh, face_centers=None))
    with pytest.raises(ValueError, match="discontinuity factors"):
        operator(mesh, face_df=np.full((len(mesh.faces), 2), 1.1))


@pytest.mark.parametrize("alpha", [0., .31, .5, np.inf])
def test_skew_boundary_trace_elimination_against_affine_current(alpha):
    mesh = mesh_grid(4)
    k = next(k for k, normal in enumerate(mesh.bface_normals) if normal[0] < -.9)
    owner, area, _ = mesh.bfaces[k]
    normal = mesh.bface_normals[k]
    tangent = np.array([-normal[1], normal[0]])
    grad = .4*tangent + (0. if alpha == 0 else -.7)*normal
    trace = 0. if alpha == 0 or np.isinf(alpha) else -1.3*np.dot(grad, normal)/alpha
    center = mesh.bface_centers[k]
    u = trace+(mesh.centroid-center)@grad
    anchor = int(np.argmax(np.abs(u)))
    stencils = [{anchor: grad/u[anchor]} for _ in mesh.cells]
    op = operator(mesh, 1.3, bc=alpha, reconstruction_policy="supplied", gradient_stencils=stencils)
    _, currents = op.face_currents(u)
    np.testing.assert_allclose(currents[k], -area*1.3*np.dot(grad, normal), atol=3e-15)
