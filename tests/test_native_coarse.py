"""Bounded two-level algebra and native-action checks; no physical admission."""
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import gmres, spilu

import garnet.native_coarse as coarse_module
import garnet.coupled_moment_solver as driver
from garnet.native_coarse import NativeCoarsePreconditioner as Coarse, connected_aggregates
from garnet.corrected_native_moments import CorrectedNativeMomentOperator
from garnet.mesh import MeshGrid
from test_corrected_mesh import mesh_grid
from test_native_sp5 import material


def line(n):
    return np.arange(n, dtype=float)[:, None], [(i, i+1) for i in range(n-1)]


def test_mapping_deterministic_ties_fixed_budget_and_connected():
    centers, edges = line(257)
    a = connected_aggregates(centers, edges)
    b = connected_aggregates(centers, reversed(edges + edges))
    np.testing.assert_array_equal(a, b)
    assert len(np.unique(a)) == 128
    assert not a.flags.writeable
    # First farthest seed is cell 256, then 128, then 64 (ties choose low).
    assert [a[i] for i in (0, 256, 128, 64)] == [0, 1, 2, 3]
    # Cell 1 is exactly between seed cells 0 and 2; lower ordinal wins.
    assert a[1] == a[0]
    for owner in np.unique(a):
        members = np.flatnonzero(a == owner)
        assert members[-1]-members[0]+1 == len(members)


def test_mapping_components_seeded_before_farthest_and_budget_rejected():
    centers, edges = line(260)
    edges.remove((129, 130))
    ids = connected_aggregates(centers, edges)
    assert ids[0] == 0 and ids[130] == 1
    assert not set(ids[:130]) & set(ids[130:])
    with pytest.raises(ValueError, match='components exceed'):
        connected_aggregates(np.arange(129.)[:, None], [])
    np.testing.assert_array_equal(connected_aggregates([[0.], [0.]], []), [0, 1])


@pytest.mark.parametrize('centers,edges', [([[0.], [0.]], [(0, 1)]),
    ([[0.], [np.nan]], []), ([[0.], [1.]], [(0, 2)]),
    ([[0.], [1.]], [(0, 0)]), ([[0.], [1.]], [(False, 1)]),
    ([[0.], [1.]], [(0., 1)])])
def test_invalid_geometry_rejected(centers, edges):
    with pytest.raises(ValueError):
        connected_aggregates(centers, edges)


@pytest.mark.parametrize('moments,groups', [(1, 1), (2, 3), (3, 2)])
def test_layout_adjoint_duplicate_safe_bounded_coarse_and_fixed_formula(moments, groups):
    n, q = 5, moments*groups
    rng = np.random.default_rng(41)
    dense = rng.normal(size=(n*q, n*q))*.01 + np.diag(np.arange(1., n*q+1))
    # Explicit duplicate entries and unsorted columns in every row.
    indices = np.tile(np.r_[np.arange(n*q)[::-1], 0], n*q)
    data = np.concatenate([np.r_[row[::-1], .125] for row in dense])
    matrix = sparse.csr_matrix((data, indices, np.arange(n*q+1)*(n*q+1)), shape=dense.shape)
    dense[:, 0] += .125
    mapping = np.array([0, 1, 0, 1, 2])
    smoother = lambda r: .2*r
    before = [getattr(matrix, key).tobytes() for key in ('data', 'indices', 'indptr')]
    c = Coarse(matrix, (n, moments, groups), mapping, smoother, max_rows=2, max_nnz=n*q+2)
    P = np.zeros((n*q, 3*q))
    for cell in range(n):
        for m in range(moments):
            for g in range(groups):
                P[cell*q+groups*m+g, mapping[cell]*q+groups*m+g] = 1
    z, r = rng.normal(size=3*q), rng.normal(size=n*q)
    np.testing.assert_array_equal(c.prolong(z), P@z)
    np.testing.assert_allclose(c.restrict(r), P.T@r, atol=1e-15)
    assert np.dot(c.prolong(z), r) == pytest.approx(np.dot(z, c.restrict(r)))
    np.testing.assert_allclose(c.coarse_matrix, P.T@dense@P, rtol=2e-15, atol=1e-15)
    np.testing.assert_allclose(c.coarse_action(z), P.T@(matrix@(P@z)), rtol=2e-15, atol=1e-14)
    np.testing.assert_allclose(c.apply(matrix@c.prolong(z)), c.prolong(z), rtol=2e-14, atol=1e-14)
    qr = P@np.linalg.solve(P.T@dense@P, P.T@r)
    expected = qr+smoother(r-matrix@qr)
    actual = c.apply(r)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-14)
    np.testing.assert_array_equal(c.apply(r), actual)
    np.testing.assert_allclose(c.apply(2*r+matrix@c.prolong(z)), 2*actual+c.apply(matrix@c.prolong(z)), atol=2e-14)
    assert before == [getattr(matrix, key).tobytes() for key in ('data', 'indices', 'indptr')]
    assert c.matrix is matrix
    assert c.metadata['csr_chunks'] > 1
    assert c.metadata['largest_chunk_nnz'] <= n*q+2
    assert c.metadata['coarse_solves'] == c.metadata['extra_a_matvecs'] == 5
    assert c.metadata['applications'] == 5
    assert c.metadata['setup_coarse_solves'] == 1
    assert c.metadata['condition_estimate_1'] >= 1
    assert len(c.metadata['aggregate_sha256']) == len(c.metadata['coarse_sha256']) == 64


def test_extended_precision_accumulation_before_single_cast():
    # Same coarse pair straddles chunks; float64 accumulation loses the unit.
    a = sparse.csr_matrix((np.array([1e16, 1., -1e16, 2.]),
        np.array([0, 0, 0, 1]), np.array([0, 3, 4])), shape=(2, 2))
    c = Coarse(a, (2, 1, 1), [0, 0], lambda r: r, max_rows=1, max_nnz=1)
    assert c.coarse_matrix[0, 0] == 3.
    assert c.metadata['largest_chunk_nnz'] == 3  # one indivisible row


@pytest.mark.parametrize('family', ['sp3', 'sp5'])
def test_independent_native_action_and_reused_group_factors(family):
    mesh = mesh_grid(2, .2)
    op = CorrectedNativeMomentOperator(MeshGrid(mesh, 'vacuum'),
        [material(i % 2, groups=2) for i in range(mesh.n_cells)], family=family)
    factors = []
    for g in range(op.n_groups):
        indices = np.arange(g, op.size, op.n_groups)
        factors.append((indices, spilu(driver._group_principal_block(op.assembled, op.n_groups, g).tocsc(),
                                     drop_tol=1e-6, fill_factor=10)))
    def smoother(r):
        result = np.empty(op.size)
        for indices, factor in factors:
            result[indices] = factor.solve(r[indices])
        return result
    # Explicit small connected map exercises nontrivial restriction as well.
    c = Coarse(op.assembled, op.shape, [0, 0, 1, 1], smoother, max_rows=3)
    z = np.random.default_rng(12).normal(size=c.dimension)
    np.testing.assert_allclose(c.coarse_action(z), c.restrict(op.action(c.prolong(z))), rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(c.apply(op.action(c.prolong(z))), c.prolong(z), rtol=2e-13, atol=2e-14)
    rhs = op.fission_action(np.ones(op.shape)).ravel()
    x, info = gmres(op.assembled, rhs, M=c.linear_operator, rtol=1e-11, atol=0)
    assert info == 0
    np.testing.assert_allclose(x, np.linalg.solve(op.assembled.toarray(), rhs), rtol=2e-10, atol=1e-12)
    assert np.linalg.norm(op.action(x)-rhs)/np.linalg.norm(rhs) < 1e-11
    mapped = Coarse.from_operator(op, smoother)
    assert mapped.n_aggregates == op.n_cells


@pytest.mark.parametrize('family', ['sp3', 'sp5'])
def test_actual_native_data_tiny_direct_and_independent_residual(family):
    source = os.environ.get('NDGPU_NATIVE_COARSE_TEST_XS')
    if not source:
        pytest.skip('Set NDGPU_NATIVE_COARSE_TEST_XS to audited native derivative for optional local evidence')
    from garnet.serpent_xs import read_serpent_moment_results
    records = read_serpent_moment_results(Path(source), required_orders=tuple(range(6)))
    materials = [record.to_moment_material(orientation='from-major') for record in records]
    mesh = mesh_grid(2, .15)
    op = CorrectedNativeMomentOperator(MeshGrid(mesh, 'vacuum'),
        [materials[i % len(materials)] for i in range(mesh.n_cells)], family=family)
    c = Coarse(op.assembled, op.shape, [0, 0, 1, 1], lambda r: r)
    z = np.random.default_rng(76).normal(size=c.dimension)
    np.testing.assert_allclose(c.coarse_action(z), c.restrict(op.action(c.prolong(z))), rtol=2e-12, atol=2e-12)
    # Fixed existing group factors are used by the integrated option; its true
    # action acceptance remains independent from the assembled matrix.
    rhs = op.fission_action(np.ones(op.shape)).ravel()
    factor = spilu(op.assembled.tocsc(), drop_tol=1e-6, fill_factor=10)
    c = Coarse(op.assembled, op.shape, [0, 0, 1, 1], factor.solve)
    x, info = gmres(op.assembled, rhs, M=c.linear_operator, rtol=1e-10, atol=0)
    assert info == 0
    np.testing.assert_allclose(x, np.linalg.solve(op.assembled.toarray(), rhs), rtol=1e-8, atol=1e-10)
    assert np.linalg.norm(op.action(x)-rhs)/np.linalg.norm(rhs) < 1e-10


@pytest.mark.parametrize('bad', ['singular', 'nan', 'inf'])
def test_coarse_setup_fails_closed(bad):
    matrix = sparse.eye(2, format='csr')
    matrix.data[0] = {'singular': 0., 'nan': np.nan, 'inf': np.inf}[bad]
    with pytest.raises(ValueError, match='Singular|Nonfinite'):
        Coarse(matrix, (2, 1, 1), [0, 1], lambda r: r)


def test_coarse_solve_check_failure_and_nonfinite_application(monkeypatch):
    c = Coarse(sparse.eye(2, format='csr'), (2, 1, 1), [0, 1], lambda r: r)
    np.testing.assert_array_equal(c.apply(np.zeros(2)), np.zeros(2))
    assert np.isfinite(c.metadata['max_coarse_solve_componentwise_ratio'])
    with pytest.raises(ValueError, match='Finite'):
        c.apply([np.nan, 1.])
    monkeypatch.setattr(coarse_module, 'lu_solve', lambda *args, **kwargs: np.zeros(2))
    with pytest.raises(ValueError, match='backward residual'):
        c.apply(np.ones(2))


def test_orthogonal_geometry_and_smoother_exception_propagation():
    from test_native_sp5 import operator
    op = operator(False, n=2)
    class SmootherError(Exception):
        pass
    def broken(rhs):
        raise SmootherError('original smoother failure')
    c = Coarse.from_operator(op, broken)
    assert c.n_aggregates == op.n_cells
    with pytest.raises(SmootherError, match='original smoother failure'):
        c.apply(np.ones(op.size))


@pytest.mark.parametrize('family', ['sp3', 'sp5'])
@pytest.mark.parametrize('mode', ['none', 'group_ilu'])
def test_original_b272f85_solver_numerics_bitwise(family, mode):
    """Local frozen-commit evidence; source archives may lack the git object."""
    from test_native_sp5 import operator
    baseline = subprocess.run(['git', 'show', 'b272f85:garnet/coupled_moment_solver.py'],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    if baseline.returncode:
        pytest.skip('Original b272f85 source object unavailable')
    name = 'garnet._coarse_baseline_solver'
    module = types.ModuleType(name)
    module.__file__ = driver.__file__
    sys.modules[name] = module
    try:
        exec(compile(baseline.stdout, driver.__file__, 'exec'), module.__dict__)
        op = operator(True, n=2)
        if family == 'sp3':
            op = CorrectedNativeMomentOperator(op.grid, op.materials, family=family)
        original = module.CoupledMomentEigenSolver(op).solve(preconditioner=mode)
        current = driver.CoupledMomentEigenSolver(op).solve(preconditioner=mode)
        assert original.history == current.history
        assert original.k_eff == current.k_eff and original.full_residual == current.full_residual
        assert original.reason == current.reason
        for field in ('moments', 'scalar_flux', 'power'):
            np.testing.assert_array_equal(getattr(original, field), getattr(current, field))
    finally:
        del sys.modules[name]


@pytest.mark.parametrize('family', ['sp3', 'sp5'])
def test_solver_integration_true_residual_provenance_and_default_unchanged(family, monkeypatch):
    from test_native_sp5 import operator
    op = operator(True, n=2)
    if family == 'sp3':
        op = CorrectedNativeMomentOperator(op.grid, op.materials, family=family)
    before = [getattr(op.assembled, key).tobytes() for key in ('data', 'indices', 'indptr')]
    default = driver.CoupledMomentEigenSolver(op).solve()
    explicit = driver.CoupledMomentEigenSolver(op).solve(preconditioner='none')
    assert default.k_eff == explicit.k_eff and default.history == explicit.history
    np.testing.assert_array_equal(default.moments, explicit.moments)
    setups = []
    result = driver.CoupledMomentEigenSolver(op).solve(preconditioner='group_ilu_coarse',
        group_ilu_fill_factor=10, group_ilu_setup_callback=setups.append)
    assert result.converged and default.converged
    assert len([e for e in setups if e['phase'] == 'after_factorization']) == op.n_groups
    assert result.full_residual < 1e-9
    np.testing.assert_allclose(result.k_eff, default.k_eff, rtol=1e-10)
    meta = result.provenance['preconditioner_metadata']['coarse']
    assert meta['extra_a_matvecs'] == meta['coarse_solves'] > 0
    assert 'native_coarse.py' in result.provenance['implementation_sha256']
    assert 'native_coarse.py' not in default.provenance['implementation_sha256']
    assert result.provenance['group_ilu_fill_factor'] == 10
    assert before == [getattr(op.assembled, key).tobytes() for key in ('data', 'indices', 'indptr')]
    # Corrupt assembly is still refused by independent true-action acceptance.
    op.assembled.data *= 1.3
    rejected = driver.CoupledMomentEigenSolver(op).solve(preconditioner='group_ilu_coarse')
    assert not rejected.converged and rejected.reason == 'inner_failure_or_full_linear_residual'
