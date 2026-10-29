"""Independent checks of the opt-in corrected lower-moment block ILU."""

import os

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spilu

from garnet.mesh_preconditioner import build_corrected_block_ilu


@pytest.fixture(params=['cpu', 'gpu'])
def xp(request):
    if request.param == 'cpu':
        return np
    if os.environ.get('GARNET_TEST_GPU') != '1':
        pytest.skip('set GARNET_TEST_GPU=1 to require an actual GPU')
    import cupy
    assert cupy.cuda.runtime.getDeviceCount() > 0
    return cupy


def block_problem():
    # Both scalar blocks require nonidentity row and column permutations.
    a00 = np.array([[0., 3., 0., 0., 1.], [2., 0., 4., 0., 0.],
                    [0., 5., 6., 1., 0.], [0., 0., 2., 7., 3.],
                    [4., 0., 0., 2., 8.]])
    a11 = a00.T.copy()
    a10 = np.array([[.1, 0., -.3, 0., .4], [0., .2, 0., -.5, 0.],
                    [.7, 0., -.1, 0., 0.], [0., .9, 0., .3, 0.],
                    [0., 0., .6, 0., -.2]])
    # Deliberately nonzero A01, omitted by this preconditioner.
    matrix = np.block([[a00, np.eye(5)*.8], [a10, a11]])
    lower = np.block([[a00, np.zeros((5, 5))], [a10, a11]])
    return matrix, lower


def test_independent_lower_reference_permutations_linearity_and_shape(xp):
    matrix, lower = block_problem()
    for block in (matrix[:5, :5], matrix[5:, 5:]):
        factor = spilu(sparse.csc_matrix(block), drop_tol=1e-6, fill_factor=20)
        assert np.any(factor.perm_r != np.arange(5))
        assert np.any(factor.perm_c != np.arange(5))
    shape = (2, 5, 1, 1)
    precondition = build_corrected_block_ilu(sparse.csr_matrix(matrix), shape, xp)
    rhs = xp.asarray(np.arange(1., 11.)).reshape(shape)
    other = xp.asarray(np.linspace(-3., 2., 10)).reshape(shape)
    original = rhs.copy()
    result = precondition(rhs)
    host = lambda a: a if xp is np else a.get()
    assert isinstance(result, xp.ndarray) and result.shape == shape
    np.testing.assert_allclose(host(result).ravel(), np.linalg.solve(lower, host(rhs).ravel()),
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(host(precondition(2*rhs - 3*other)),
                               host(2*result - 3*precondition(other)), atol=1e-12)
    np.testing.assert_array_equal(host(rhs), host(original))
    assert precondition.coupling_nnz == np.count_nonzero(matrix[5:, :5])
    assert precondition.matrix_nnz == np.count_nonzero(matrix)
    assert precondition.nnz == sum(b['nnz'] for b in precondition.block_stats)
    assert [b['block'] for b in precondition.block_stats] == ['A00', 'A11']
    assert all(b['setup_seconds'] > 0 for b in precondition.block_stats)
    with pytest.raises(ValueError, match='shape'):
        precondition(rhs.ravel())
    with pytest.raises(TypeError, match='backend'):
        precondition(host(rhs).tolist())
    if xp is not np:
        with pytest.raises(TypeError, match='backend'):
            precondition(host(rhs))


@pytest.mark.parametrize('block', [0, 1])
def test_singular_diagonal_blocks_fail_explicitly(block):
    matrix = np.eye(6)
    matrix[block*3:(block+1)*3, block*3:(block+1)*3] = 0
    with pytest.raises(RuntimeError, match=f'A{block}{block}.*singular'):
        build_corrected_block_ilu(sparse.csr_matrix(matrix), (2, 3), np)


def test_invalid_shapes_and_backend():
    matrix = sparse.eye(6, format='csr')
    for shape in [(6,), (3, 2), (2, 0)]:
        with pytest.raises(ValueError, match='two-moment'):
            build_corrected_block_ilu(matrix, shape, np)
    with pytest.raises(ValueError, match='dimensions'):
        build_corrected_block_ilu(matrix, (2, 4), np)
    with pytest.raises(TypeError, match='NumPy or CuPy'):
        build_corrected_block_ilu(matrix, (2, 3), object())


def test_cpu_neither_block_extracts_explicit_factors(monkeypatch):
    import scipy.sparse.linalg

    calls = []

    class InternalFactorOnly:
        def __init__(self, matrix, **kwargs):
            self.factor = spilu(matrix, **kwargs)
            self.nnz = self.factor.nnz
            calls.append(matrix.shape)

        def solve(self, rhs):
            return self.factor.solve(rhs)

        @property
        def L(self):
            raise AssertionError('CPU must not materialize L')

        @property
        def U(self):
            raise AssertionError('CPU must not materialize U')

    monkeypatch.setattr(scipy.sparse.linalg, 'spilu', InternalFactorOnly)
    matrix, lower = block_problem()
    precondition = build_corrected_block_ilu(sparse.csr_matrix(matrix), (2, 5), np)
    rhs = np.arange(1., 11.).reshape(2, 5)
    np.testing.assert_allclose(precondition(rhs).ravel(), np.linalg.solve(lower, rhs.ravel()),
                               rtol=1e-12, atol=1e-12)
    assert calls == [(5, 5), (5, 5)]
    assert precondition.nnz_kind == 'superlu_reported_not_explicit_csc_count'


def skew_problem():
    from garnet import Material, assemble_mesh

    coords = [(x + .35*y, y) for y in (0., .4, 1.) for x in (0., .3, 1.)]
    cells = [(j*3+i, j*3+i+1, (j+1)*3+i+1, (j+1)*3+i)
             for j in range(2) for i in range(2)]
    mesh = assemble_mesh(coords, cells, [0]*4)
    materials = [Material([.7], [.03], [.035], [[.2]], [1.], total=[.23]),
                 Material([.4], [.06], [.07], [[.2]], [1.], total=[.26])]
    return mesh, materials


@pytest.mark.parametrize('family', ['sp3', 'sdp1'])
@pytest.mark.parametrize('boundary', ['reflective', 'vacuum'])
def test_skew_heterogeneous_moment_solve_and_true_residual(family, boundary):
    from garnet import MeshSP3EigenSolver, MeshSDP1EigenSolver
    from garnet.linalg import gmres

    mesh, materials = skew_problem()
    cls = dict(sp3=MeshSP3EigenSolver, sdp1=MeshSDP1EigenSolver)[family]
    solver = cls(mesh, materials, [0, 1, 0, 1], device='cpu',
                 boundary_conditions=boundary, mesh_discretization='corrected',
                 mesh_preconditioner='corrected_block_ilu', marshak_vacuum=True,
                 symmetrize=False)
    op, precondition = solver.ops[0], solver.preconds[0]
    matrix = op.host_matrix.toarray()
    n = matrix.shape[0]//2
    a10 = matrix[n:, :n]
    if boundary == 'vacuum':
        # Marshak gradient elimination contributes genuine off-cell coupling.
        assert np.count_nonzero(a10 - np.diag(np.diag(a10))) > 0
    lower = matrix.copy()
    lower[:n, n:] = 0
    rhs = np.arange(1., matrix.shape[0]+1).reshape(op.diag.shape)
    np.testing.assert_allclose(precondition(rhs).ravel(), np.linalg.solve(lower, rhs.ravel()),
                               rtol=1e-12, atol=1e-12)
    expected = np.linalg.solve(matrix, rhs.ravel()).reshape(rhs.shape)
    solution, _ = gmres(op.apply, rhs, np.zeros_like(rhs), op.inv_diag, np,
                        rtol=1e-11, maxiter=60, precond=precondition)
    np.testing.assert_allclose(solution, expected, rtol=1e-10, atol=1e-10)
    assert np.linalg.norm(op.apply(solution)-rhs)/np.linalg.norm(rhs) <= 1e-11
    assert precondition.group == 0


@pytest.mark.parametrize('family, kwargs', [('scalar', {}), ('spn', {'order': 0}),
                                           ('spn', {'order': 2}), ('sdpn', {'order': 3})])
def test_solver_rejects_unsupported_scalar_and_orders(family, kwargs):
    from garnet import MeshDiffusionEigenSolver, MeshSPNEigenSolver, MeshSDPNEigenSolver

    mesh, materials = skew_problem()
    cls = dict(scalar=MeshDiffusionEigenSolver, spn=MeshSPNEigenSolver,
               sdpn=MeshSDPNEigenSolver)[family]
    with pytest.raises(ValueError, match='only order-1 mesh SP3/SDP1'):
        cls(mesh, materials, [0, 1, 0, 1], device='cpu',
            mesh_discretization='corrected', mesh_preconditioner='corrected_block_ilu', **kwargs)


def test_solver_requires_corrected_discretization():
    from garnet import MeshSP3EigenSolver

    mesh, materials = skew_problem()
    with pytest.raises(ValueError, match='only available for corrected meshes'):
        MeshSP3EigenSolver(mesh, materials, [0, 1, 0, 1], device='cpu',
                           mesh_preconditioner='corrected_block_ilu')
