"""Independent coupled-current and dense-pencil checks."""
from dataclasses import replace

import numpy as np
import pytest

from ndgpu.corrected_mesh_moments import CorrectedMeshMomentOperator
from ndgpu.mesh import MeshGrid, MeshGroupOperator, assemble_mesh
from ndgpu.spn import SDPNGroupOperator, _SPN_C, _SPN_G, _SDPN_C, _SDPN_G
from ndgpu.linalg import gmres


def mesh_grid(n=4, shear=.35):
    coords = np.array([(x+shear*y, y) for y in np.linspace(0, 1, n+1)
                       for x in np.linspace(0, 1, n+1)])
    cells = [(j*(n+1)+i, j*(n+1)+i+1, (j+1)*(n+1)+i+1, (j+1)*(n+1)+i)
             for j in range(n) for i in range(n)]
    return assemble_mesh(coords, cells, np.zeros(n*n, dtype=int))


def operator(mesh, family="spn", bc="vacuum", D=.8, **kw):
    grid = MeshGrid(mesh, bc)
    c, g = (_SPN_C, _SPN_G) if family == "spn" else (_SDPN_C, _SDPN_G)
    return CorrectedMeshMomentOperator(np, grid, np.full(grid.shape, D),
                                      np.full(grid.shape, 1.2), np.full(grid.shape, .13),
                                      1, bc=bc, coeffs=c, boundary_g=g[1], **kw)


@pytest.mark.parametrize("family", ["spn", "sdpn"])
def test_orthogonal_independent_dense_pencil_and_legacy(family):
    mesh = mesh_grid(3, 0.)
    specs = ["reflective" if i % 3 == 0 else "zero-flux" if i % 3 == 1 else "vacuum"
             for i in range(len(mesh.bfaces))]
    op = operator(mesh, family, specs)
    n = mesh.n_cells
    if family == "spn":
        c0 = np.array([[1., -2/3], [-2/3, 4/9]])
        ct = np.array([[0., 0.], [0., 5/9]])
        g = np.array([[.5, -1/8], [-1/8, 7/24]])
    else:
        c0 = np.array([[1., -2/3], [-6/7, 4/7]])
        ct = np.array([[0., 0.], [0., 5/7]])
        g = np.array([[.5, -1/8], [-2/7, 23/42]])
    D = [.8, 1/(7*1.2)]
    dense = np.kron(.13*c0+1.2*ct, np.diag(mesh.area))
    for (i, j, area, _), (di, dj) in zip(mesh.faces, mesh.face_distances):
        for m in range(2):
            ix = np.array([m*n+i, m*n+j])
            dense[np.ix_(ix, ix)] += area*D[m]/(di+dj)*np.array([[1., -1.], [-1., 1.]])
    for (i, area, d), spec in zip(mesh.bfaces, specs):
        A = np.diag(np.array(D)/d)
        K = (np.zeros((2, 2)) if spec == "reflective" else area*A if spec == "zero-flux"
             else area*A@(np.eye(2)-np.linalg.solve(A+g, A)))
        ix = [i, n+i]
        dense[np.ix_(ix, ix)] += K
    columns = np.column_stack([op.apply(e).ravel() for e in np.eye(2*n)])
    np.testing.assert_allclose(columns, dense, atol=2e-14)
    np.testing.assert_allclose(op.host_matrix.toarray(), dense, atol=2e-14)
    np.testing.assert_allclose(op.diag.ravel(), dense.diagonal())
    c = _SPN_C if family == "spn" else _SDPN_C
    old = SDPNGroupOperator(np, op.grid, np.full(op.grid.shape, .8),
                            np.full(op.grid.shape, 1.2), np.full(op.grid.shape, .13),
                            1, bc=specs, coeffs=c, boundary_g=g, op_cls=MeshGroupOperator)
    state = np.arange(2*n).reshape((2,)+op.grid.shape)/n
    np.testing.assert_allclose(op.apply(state), old.apply(state), atol=2e-14)


@pytest.mark.parametrize("family", ["spn", "sdpn"])
def test_skew_affine_coupled_marshak_gradient_and_current(family):
    mesh = mesh_grid(5)
    k = next(k for k, ((i, _, _), normal) in enumerate(zip(mesh.bfaces, mesh.bface_normals))
             if normal[0] < -.9 and .3 < mesh.centroid[i, 1] < .7)
    # Retain one physical boundary constraint. Interior neighbor patches remain
    # full rank, allowing a global affine field satisfying this homogeneous law.
    mesh = replace(mesh, bfaces=[mesh.bfaces[k]], bface_centers=mesh.bface_centers[[k]],
                   bface_normals=mesh.bface_normals[[k]], bface_tags=mesh.bface_tags[[k]])
    op = operator(mesh, family)
    G = np.asarray((_SPN_G if family == "spn" else _SDPN_G)[1])
    D = np.array([.8, 1/(7*1.2)])
    normal = mesh.bface_normals[0]
    tangent = np.array([-normal[1], normal[0]])
    trace = np.array([1.1, -.3])
    gradients = -(G@trace/D)[:, None]*normal + np.array([.4, -.7])[:, None]*tangent
    state = trace[:, None] + gradients@(mesh.centroid-mesh.bface_centers[0]).T
    for row in op.gradient_stencils:
        actual = sum(vector*state.ravel()[col] for col, vector in row.items())
        np.testing.assert_allclose(actual, gradients, atol=4e-13)
    interior, boundary = op.face_currents(state)
    for f, ((_, _, area, _), nf) in enumerate(zip(mesh.faces, mesh.face_normals)):
        np.testing.assert_allclose(interior[:, f], -area*D*(gradients@nf), atol=2e-14)
    area = mesh.bfaces[0][1]
    np.testing.assert_allclose(boundary[:, 0], area*G@trace, atol=2e-14)
    np.testing.assert_allclose(boundary[:, 0], -area*D*(gradients@normal), atol=2e-14)


def test_skew_reflective_constants_conservation_and_gmres_dense_solution():
    mesh = mesh_grid(4)
    op = operator(mesh, "sdpn", "reflective")
    state = np.broadcast_to(np.array([1.3, -.7])[:, None, None, None], (2,)+op.grid.shape)
    interior, boundary = op.face_currents(state)
    np.testing.assert_allclose(interior, 0., atol=3e-15)
    np.testing.assert_allclose(boundary, 0., atol=3e-15)
    rng = np.random.default_rng(71)
    state = rng.normal(size=(2,)+op.grid.shape)
    interior, boundary = op.face_currents(state)
    c = np.asarray(_SDPN_C[1])
    reaction = np.einsum('mk,ki->mi', .13*c[0]+1.2*c[1], state.reshape(2, -1))*mesh.area
    for f, (i, j, _, _) in enumerate(mesh.faces):
        reaction[:, i] += interior[:, f]
        reaction[:, j] -= interior[:, f]
    np.testing.assert_allclose(op.apply(state).reshape(2, -1), reaction, atol=2e-14)
    dense = np.column_stack([op.apply(e).ravel() for e in np.eye(state.size)])
    rhs = rng.normal(size=state.shape)
    expected = np.linalg.solve(dense, rhs.ravel())
    actual, info = gmres(op.apply, rhs, np.zeros_like(rhs), op.inv_diag, np,
                         rtol=1e-11, maxiter=1000)
    np.testing.assert_allclose(actual.ravel(), expected, rtol=1e-8, atol=1e-8)
    assert not op.symmetric


def test_unsupported_modes_and_missing_geometry_fail_explicitly():
    mesh = mesh_grid(2)
    for kw, message in ((dict(symmetrize=True), 'symmetrize'),
                        (dict(moment_df_model='common_trace'), 'discontinuity'),
                        (dict(hybrid_confine=True), 'hybrid')):
        with pytest.raises(ValueError, match=message):
            operator(mesh, **kw)
    with pytest.raises(ValueError, match='face_centers'):
        operator(replace(mesh, face_centers=None))
    with pytest.raises(ValueError, match='order 1'):
        CorrectedMeshMomentOperator(np, MeshGrid(mesh), np.ones(4), np.ones(4), np.ones(4), 2)


def test_coupled_patch_rank_failure_is_explicit():
    mesh = mesh_grid(2, 0.)
    keep = [k for k, normal in enumerate(mesh.face_normals) if normal[0] > .9]
    thin = replace(mesh, faces=[mesh.faces[k] for k in keep],
                   face_centers=mesh.face_centers[keep], face_normals=mesh.face_normals[keep],
                   face_distances=mesh.face_distances[keep], bfaces=[],
                   bface_centers=np.empty((0, 2)), bface_normals=np.empty((0, 2)))
    with pytest.raises(ValueError, match='rank-deficient coupled gradient patch'):
        operator(thin)


def test_three_dimensional_orthogonal_boundary_and_time_reaction():
    from ndgpu.mesh import assemble_mesh_3d
    coords = np.array([(x, y, z) for z in (0., 3.) for y in (0., 2.) for x in (0., 1.)])
    mesh = assemble_mesh_3d(coords, [(0, 1, 3, 2, 4, 5, 7, 6)], [0], normal_distances=True)
    op = operator(mesh, 'sdpn', theta=.4)
    old = SDPNGroupOperator(np, op.grid, np.full(op.grid.shape, .8),
                            np.full(op.grid.shape, 1.2), np.full(op.grid.shape, .13),
                            1, bc='vacuum', coeffs=_SDPN_C, boundary_g=_SDPN_G[1],
                            theta=.4, op_cls=MeshGroupOperator)
    state = np.array([1.3, -.7]).reshape((2,)+op.grid.shape)
    np.testing.assert_allclose(op.apply(state), old.apply(state), atol=2e-14)
    np.testing.assert_allclose(op.inv_diag, old.inv_diag, atol=2e-14)
