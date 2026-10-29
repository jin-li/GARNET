"""Independent dense checks of fixed TPFA ILU, optionally on an actual GPU."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse.linalg import spilu
from scipy.sparse import csc_matrix

from garnet.mesh import _MeshGroupOperator
from garnet.mesh_preconditioner import _assemble_tpfa, build_tpfa_ilu


@pytest.fixture(params=["cpu", "gpu"])
def xp(request):
    if request.param == "cpu":
        return np
    if os.environ.get("GARNET_TEST_GPU") != "1":
        pytest.skip("set GARNET_TEST_GPU=1 to require an actual GPU")
    import cupy
    assert cupy.cuda.runtime.getDeviceCount() > 0
    return cupy


def ell_operator(matrix, xp, wrapped=False):
    """Encode a literal dense matrix without using production assembly."""
    n = len(matrix)
    nbr = np.broadcast_to(np.arange(n)[:, None], (n, n)).copy()
    weights = -matrix.T.copy()
    np.fill_diagonal(weights, 0)
    with np.errstate(divide="ignore"):
        op = _MeshGroupOperator(xp, xp.asarray(nbr), xp.asarray(weights),
                                xp.asarray(np.diag(matrix)))
    return (SimpleNamespace(_flat=op, diag=op.diag.reshape(n, 1, 1))
            if wrapped else op)


@pytest.mark.parametrize("wrapped", [False, True])
def test_scalar_dense_and_repeated_linear_application(xp, wrapped):
    matrix = np.array([[4., -1., 0., -2.], [-3., 7., -1., 0.],
                       [0., -2., 6., -1.], [-1., 0., -2., 8.]])
    op = ell_operator(matrix, xp, wrapped)
    precondition = build_tpfa_ilu(op, xp)
    shape = op.diag.shape
    rhs = xp.asarray(np.arange(1., 5.)).reshape(shape)
    result = precondition(rhs)
    assert isinstance(result, xp.ndarray)
    assert result.shape == shape
    np.testing.assert_allclose(np.asarray(result) if xp is np else result.get(),
                               np.linalg.solve(matrix, np.arange(1., 5.)).reshape(shape),
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        (precondition(3 * rhs) - 3 * result).ravel().get() if xp is not np
        else (precondition(3 * rhs) - 3 * result).ravel(), 0., atol=1e-12)
    assert precondition.nnz >= matrix.shape[0] * 2
    with pytest.raises(ValueError, match="shape"):
        precondition(rhs.reshape(2, 2))
    with pytest.raises(TypeError, match="backend"):
        precondition(rhs.tolist())


def test_nonidentity_row_and_column_permutations(xp):
    # Zero diagonal forces row pivoting; the irregular graph forces COLAMD.
    matrix = np.array([[0., 3., 0., 0., 1.], [2., 0., 4., 0., 0.],
                       [0., 5., 6., 1., 0.], [0., 0., 2., 7., 3.],
                       [4., 0., 0., 2., 8.]])
    factor = spilu(csc_matrix(matrix), drop_tol=1e-6, fill_factor=20)
    assert np.any(factor.perm_r != np.arange(5))
    assert np.any(factor.perm_c != np.arange(5))
    solve = build_tpfa_ilu(ell_operator(matrix, xp), xp)
    result = solve(xp.asarray([1., -3., 2., 7., 5.]))
    result = result if xp is np else result.get()
    np.testing.assert_allclose(result, np.linalg.solve(matrix, [1., -3., 2., 7., 5.]),
                               rtol=1e-12, atol=1e-12)


def test_moments_coupling_and_marshak_dense(xp):
    a = np.array([[4., -1., 0.], [-2., 6., -1.], [0., -3., 8.]])
    b = np.array([[5., -2., 0.], [-1., 7., -2.], [0., -1., 6.]])
    c01, c10 = np.array([.2, -.4, .3]), np.array([-.7, .1, -.2])
    boundary = np.array([[[.8, -.1], [-.3, .5]], [[.4, -.2], [-.1, .7]]])
    op = SimpleNamespace(
        moments=[ell_operator(a, xp, True), ell_operator(b, xp, True)],
        coupling={(0, 1): xp.asarray(c01).reshape(3, 1, 1),
                  (1, 0): xp.asarray(c10).reshape(3, 1, 1)},
        marshak=[((xp.asarray([0, 2]), 0, 0), xp.asarray(boundary))])
    expected = np.block([[a, np.diag(c01)], [np.diag(c10), b]])
    for cell, block in zip([0, 2], boundary):
        for i in range(2):
            for j in range(2):
                expected[3*i+cell, 3*j+cell] += block[i, j]
    assembled, shape = _assemble_tpfa(op)
    np.testing.assert_allclose(assembled.toarray(), expected)
    assert shape == (2, 3, 1, 1)
    rhs = np.arange(1., 7.)
    result = build_tpfa_ilu(op, xp)(xp.asarray(rhs).reshape(shape))
    result = result if xp is np else result.get()
    np.testing.assert_allclose(result.ravel(), np.linalg.solve(expected, rhs),
                               rtol=1e-12, atol=1e-12)


def test_unsupported_operator_and_singular_factorization_are_explicit():
    with pytest.raises(TypeError, match="mesh ELL"):
        build_tpfa_ilu(object(), np)
    with pytest.raises(RuntimeError, match="singular"):
        build_tpfa_ilu(ell_operator(np.zeros((2, 2)), np), np)


def test_real_mesh_moment_operator_action(xp):
    from garnet import MeshGrid, MeshGroupOperator, assemble_mesh
    from garnet.spn import SDPNGroupOperator

    coords = np.array([(x, y) for x in (0., 1., 3., 4.) for y in (0., 1.)])
    mesh = assemble_mesh(coords, [(0, 2, 3, 1), (2, 4, 5, 3), (4, 6, 7, 5)],
                         [0, 0, 0])
    grid = MeshGrid(mesh, "vacuum")
    diffusion = xp.asarray([.8, 1.2, .6]).reshape(grid.shape)
    total = xp.asarray([.9, .7, 1.1]).reshape(grid.shape)
    removal = xp.asarray([.08, .12, .06]).reshape(grid.shape)
    op = SDPNGroupOperator(xp, grid, diffusion, total, removal, 1,
                           bc="vacuum", op_cls=MeshGroupOperator,
                           boundary_g=[[.5, -.125], [-2/7, 23/42]])
    matrix, shape = _assemble_tpfa(op)
    for column in range(matrix.shape[1]):
        basis = xp.zeros(shape)
        basis.reshape(-1)[column] = 1
        action = op.apply(basis)
        action = action if xp is np else action.get()
        np.testing.assert_allclose(matrix.toarray()[:, column], action.ravel(),
                                   rtol=1e-13, atol=1e-13)
    rhs = xp.arange(1., matrix.shape[0] + 1).reshape(shape)
    solution = build_tpfa_ilu(op, xp)(rhs)
    residual = op.apply(solution) - rhs
    residual = residual if xp is np else residual.get()
    np.testing.assert_allclose(residual, 0., atol=2e-13)


@pytest.mark.parametrize('family', ['scalar', 'sp3', 'sdp1'])
def test_corrected_ilu_skew_heterogeneous_action_and_true_residual(family):
    from garnet import (Material, MeshDiffusionEigenSolver, MeshSP3EigenSolver,
                        MeshSDP1EigenSolver, assemble_mesh)
    from garnet.linalg import gmres
    from garnet.mesh_preconditioner import build_corrected_ilu

    # Nonuniform sheared cells and heterogeneous material exercise corrected
    # stencils; moment vacuum boundaries retain their full coupled blocks.
    coords = [(x + .35*y, y) for y in (0., .4, 1.) for x in (0., .3, 1.)]
    cells = [(j*3+i, j*3+i+1, (j+1)*3+i+1, (j+1)*3+i)
             for j in range(2) for i in range(2)]
    mesh = assemble_mesh(coords, cells, [0]*4)
    materials = [Material([.7], [.03], [.035], [[.2]], [1.], total=[.23]),
                 Material([.4], [.06], [.07], [[.2]], [1.], total=[.26])]
    cls = dict(scalar=MeshDiffusionEigenSolver, sp3=MeshSP3EigenSolver,
               sdp1=MeshSDP1EigenSolver)[family]
    kwargs = {} if family == 'scalar' else dict(marshak_vacuum=True, symmetrize=False)
    solver = cls(mesh, materials, [0, 1, 0, 1], device='cpu',
                 boundary_conditions='vacuum', mesh_discretization='corrected',
                 mesh_preconditioner='corrected_ilu', **kwargs)
    op, precondition = solver.ops[0], solver.preconds[0]
    shape = op.diag.shape
    matrix = op.host_matrix.toarray()
    for basis in np.eye(matrix.shape[0]):
        np.testing.assert_allclose(op.host_matrix @ basis,
                                   op.apply(basis.reshape(shape)).ravel(), atol=2e-14)
    rhs = np.arange(1., matrix.shape[0]+1).reshape(shape)
    expected = np.linalg.solve(matrix, rhs.ravel()).reshape(shape)
    np.testing.assert_allclose(precondition(rhs), expected, rtol=1e-12, atol=1e-12)
    solution, _ = gmres(op.apply, rhs, np.zeros(shape), op.inv_diag, np,
                        rtol=1e-11, maxiter=30, precond=precondition)
    np.testing.assert_allclose(solution, expected, rtol=1e-11, atol=1e-11)
    assert np.linalg.norm(op.apply(solution)-rhs)/np.linalg.norm(rhs) <= 1e-11
    assert precondition.shape == shape
    assert precondition.matrix_nnz == op.host_matrix.nnz
    assert precondition.nnz > 0 and precondition.setup_seconds > 0
    with pytest.raises(ValueError, match='shape'):
        precondition(rhs.ravel())
    with pytest.raises(TypeError, match='backend'):
        precondition(rhs.tolist())
    with pytest.raises(ValueError, match='dimensions'):
        build_corrected_ilu(op.host_matrix, (matrix.shape[0]+1,), np)


def test_corrected_ilu_shared_permutations_and_explicit_failure(xp):
    from garnet.mesh_preconditioner import build_corrected_ilu
    matrix = np.array([[0., 3., 0., 0., 1.], [2., 0., 4., 0., 0.],
                       [0., 5., 6., 1., 0.], [0., 0., 2., 7., 3.],
                       [4., 0., 0., 2., 8.]])
    precondition = build_corrected_ilu(csc_matrix(matrix), (5,), xp)
    rhs = xp.asarray([1., -3., 2., 7., 5.])
    result = precondition(rhs)
    np.testing.assert_allclose(result if xp is np else result.get(),
                               np.linalg.solve(matrix, [1., -3., 2., 7., 5.]),
                               rtol=1e-12, atol=1e-12)
    with pytest.raises(RuntimeError, match='singular'):
        build_corrected_ilu(csc_matrix((2, 2)), (2,), xp)
