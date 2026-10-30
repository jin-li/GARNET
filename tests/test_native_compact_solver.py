"""Small deferred solver/coarse equivalence; no production fit or solve claim."""
from dataclasses import replace
import weakref
import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator, spilu
import garnet.coupled_moment_solver as driver
from garnet.corrected_native_moments import CorrectedNativeMomentOperator as Operator
from garnet.native_coarse import NativeCoarsePreconditioner as Coarse
from test_native_compact import fixture


GATES = dict(inner_tolerance=1e-7, k_tolerance=1e-6,
             source_tolerance=1e-5, residual_tolerance=1e-5)


def pair(family):
    grid, materials = fixture(family)
    return (Operator(grid, materials, family=family),
            Operator(grid, materials, family=family, storage='deferred',
                     gradient_chunk_cells=1, batch_size=3))


def forbid_assembled(monkeypatch):
    original = Operator.assembled
    def get(op):
        if op.storage == 'deferred':
            raise AssertionError('deferred assembled property must never be accessed')
        return original.fget(op)
    monkeypatch.setattr(Operator, 'assembled', property(get, original.fset))


def smoother(op):
    factors = [(np.arange(group, op.size, op.n_groups),
                spilu(op.group_principal_block(group).tocsc(), drop_tol=1e-6, fill_factor=20))
               for group in range(op.n_groups)]
    def apply(rhs):
        out = np.empty(op.size)
        for indices, factor in factors:
            out[indices] = factor.solve(rhs[indices])
        return out
    return apply


@pytest.mark.parametrize('family', ['sp3', 'sp5'])
def test_streamed_projection_full_action_and_fixed_two_level_parity(family, monkeypatch):
    assembled, deferred = pair(family)
    mapping = [0, 0, 1, 1]
    fixed = smoother(assembled)
    reference = Coarse(assembled.assembled, assembled.shape, mapping, fixed, max_rows=2, max_nnz=50)
    forbid_assembled(monkeypatch)
    calls = []
    original_action = deferred.action
    def action(x):
        calls.append(1)
        return original_action(x)
    monkeypatch.setattr(deferred, 'action', action)
    compact = Coarse(deferred.linear_operator(), deferred.shape, mapping, fixed,
                     block_iterator=deferred.iter_full_blocks, max_rows=2, max_nnz=50)
    np.testing.assert_allclose(compact.coarse_matrix, reference.coarse_matrix, rtol=1e-12, atol=3e-14)
    rng = np.random.default_rng(871)
    z, rhs = rng.normal(size=compact.dimension), rng.normal(size=deferred.size)
    np.testing.assert_array_equal(compact.prolong(z), reference.prolong(z))
    np.testing.assert_array_equal(compact.restrict(rhs), reference.restrict(rhs))
    np.testing.assert_allclose(compact.coarse_action(z), compact.restrict(deferred.action(compact.prolong(z))),
                               rtol=1e-12, atol=3e-14)
    before = len(calls)
    np.testing.assert_allclose(compact.apply(rhs), reference.apply(rhs), rtol=1e-12, atol=3e-14)
    assert len(calls) == before+1
    np.testing.assert_allclose(compact.apply(deferred.action(compact.prolong(z))), compact.prolong(z),
                               rtol=1e-12, atol=3e-14)
    assert compact.metadata['full_block_chunks'] > 1 and compact.metadata['csr_chunks'] == 0
    assert compact.metadata['largest_chunk_nnz'] <= compact.q**2  # one indivisible full block
    assert compact.metadata['extra_a_matvecs'] == compact.metadata['coarse_solves'] == 2
    mapped = Coarse.from_operator(deferred, fixed)
    assert mapped.n_aggregates == deferred.n_cells
    np.testing.assert_allclose(mapped.apply(rhs), np.linalg.solve(assembled.assembled.toarray(), rhs),
                               rtol=1e-12, atol=3e-14)


def test_streamed_extended_precision_duplicate_sum_across_and_within_batches():
    # Float64 accumulation would lose the unit across these repeated pairs.
    def blocks():
        yield np.array([0, 0]), np.array([0, 0]), np.array([[[1e16]], [[1.]]])
        yield np.array([0, 1]), np.array([0, 1]), np.array([[[-1e16]], [[2.]]])
    action = LinearOperator((2, 2), matvec=lambda x: np.array([x[0], 2*x[1]]), dtype=float)
    for max_nnz in (1, 100):
        coarse = Coarse(action, (2, 1, 1), [0, 0], lambda r: r,
                        block_iterator=blocks, max_rows=100, max_nnz=max_nnz)
        assert coarse.coarse_matrix[0, 0] == 3.
        assert coarse.metadata['fine_entries_accumulated'] == 4
        assert coarse.metadata['full_block_chunks'] == (4 if max_nnz == 1 else 2)


@pytest.mark.parametrize('family', ['sp3', 'sp5'])
@pytest.mark.parametrize('mode', ['none', 'group_ilu', 'group_ilu_coarse'])
def test_solver_parity_and_explicit_campaign_acceptance_gates(family, mode, monkeypatch):
    assembled, deferred = pair(family)
    reference = driver.CoupledMomentEigenSolver(assembled).solve(preconditioner=mode, **GATES)
    forbid_assembled(monkeypatch)
    result = driver.CoupledMomentEigenSolver(deferred).solve(preconditioner=mode, **GATES)
    assert reference.converged and result.converged
    assert result.reason == reference.reason == 'converged'
    assert all(record['linear_residual'] <= 1e-7 for record in result.history)
    last = result.history[-1]
    assert last['k_change'] <= 1e-6
    assert last['source_change'] <= 1e-5
    assert result.full_residual <= 1e-5
    # GMRES stopping can differ by roundoff, but both returned pairs must satisfy
    # the same explicit campaign gates and agree well below those gates.
    np.testing.assert_allclose(result.k_eff, reference.k_eff, rtol=1e-12, atol=3e-14)
    np.testing.assert_allclose(result.moments, reference.moments, rtol=1e-12, atol=3e-14)
    meta = result.provenance['preconditioner_metadata']
    assert meta['matrix_nnz'] is None and not meta['global_matrix_materialized']
    assert result.provenance['operator_storage'] == 'deferred'
    assert result.provenance['operator_payload_accounting']['total_array_bytes'] > 0
    assert 'native_compact.py' in result.provenance['implementation_sha256']
    for name, value in GATES.items():
        assert result.provenance[name] == value
    if mode == 'group_ilu_coarse':
        assert meta['coarse']['full_block_chunks'] > 0
        assert meta['coarse']['extra_a_matvecs'] > 0


@pytest.mark.parametrize('family', ['sp3', 'sp5'])
@pytest.mark.parametrize('mode', ['group_ilu', 'group_ilu_coarse'])
def test_exact_group_providers_release_sources_and_keep_fixed_factors(family, mode, monkeypatch):
    assembled, deferred = pair(family)
    expected = [assembled.group_principal_block(g).tocsc() for g in range(deferred.n_groups)]
    expected_smoother = smoother(assembled)
    provider = deferred.group_principal_block
    calls, csr_refs, csc_refs, solves = [], [], [], []
    forbid_assembled(monkeypatch)
    def group_provider(group):
        assert all(ref() is None for ref in csr_refs+csc_refs)
        block = provider(group)
        calls.append(group); csr_refs.append(weakref.ref(block))
        return block
    monkeypatch.setattr(deferred, 'group_principal_block', group_provider)
    real_spilu = driver.spilu
    class FixedFactor:
        def __init__(self, factor): self.factor = factor
        @property
        def nnz(self): return self.factor.nnz
        def solve(self, rhs):
            solves.append(1)
            return self.factor.solve(rhs)
        @property
        def L(self): raise AssertionError('No explicit L materialization')
        @property
        def U(self): raise AssertionError('No explicit U materialization')
    def factor(block, **kwargs):
        assert all(ref() is None for ref in csr_refs)
        assert block.format == 'csc'
        np.testing.assert_allclose(block.toarray(), expected[len(csc_refs)].toarray(), rtol=1e-12, atol=3e-14)
        assert kwargs == dict(drop_tol=1e-6, fill_factor=20)
        csc_refs.append(weakref.ref(block))
        return FixedFactor(real_spilu(block, **kwargs))
    monkeypatch.setattr(driver, 'spilu', factor)
    real_gmres = driver.gmres
    applications = []
    def gmres(matrix, rhs, *, M, **kwargs):
        assert isinstance(matrix, LinearOperator)
        assert all(ref() is None for ref in csr_refs+csc_refs)
        assert calls == list(range(deferred.n_groups))
        vector = np.arange(1., deferred.size+1)
        if mode == 'group_ilu':
            np.testing.assert_allclose(M@vector, expected_smoother(vector), rtol=1e-12, atol=3e-14)
        before = len(calls)
        first = M@vector
        np.testing.assert_array_equal(M@vector, first)
        np.testing.assert_allclose(M@(2*vector), 2*first, rtol=1e-12, atol=3e-14)
        applications.append(1)
        assert len(calls) == before
        return real_gmres(matrix, rhs, M=M, **kwargs)
    monkeypatch.setattr(driver, 'gmres', gmres)
    result = driver.CoupledMomentEigenSolver(deferred).solve(preconditioner=mode, **GATES)
    assert result.converged and result.iterations > 1
    assert len(calls) == len(csc_refs) == deferred.n_groups
    assert len(applications) == result.iterations and len(solves) > deferred.n_groups
    assert all(ref() is None for ref in csr_refs+csc_refs)


@pytest.mark.parametrize('mode', ['none', 'group_ilu', 'group_ilu_coarse'])
def test_inconsistent_deferred_gmres_action_is_rejected_by_true_action(mode, monkeypatch):
    _, deferred = pair('sp5')
    forbid_assembled(monkeypatch)
    action = deferred.action
    monkeypatch.setattr(deferred, 'linear_operator', lambda: LinearOperator(
        (deferred.size, deferred.size), matvec=lambda x: 1.3*action(x), dtype=float))
    result = driver.CoupledMomentEigenSolver(deferred).solve(preconditioner=mode, **GATES)
    assert not result.converged
    assert result.reason == 'inner_failure_or_full_linear_residual'
    assert result.history[0]['gmres_info'] == 0
    assert result.history[0]['linear_residual'] > .1


def test_deferred_global_ilu_rejected_without_assembly_or_factorization(monkeypatch):
    _, deferred = pair('sp3')
    forbid_assembled(monkeypatch)
    def forbidden(*args, **kwargs): raise AssertionError('No factorization or GMRES allowed')
    monkeypatch.setattr(driver, 'spilu', forbidden)
    monkeypatch.setattr(driver, 'gmres', forbidden)
    with pytest.raises(ValueError, match='Global ILU is unsupported with deferred storage'):
        driver.CoupledMomentEigenSolver(deferred).solve(preconditioner='ilu')


@pytest.mark.parametrize('phase', ['extraction', 'factorization'])
def test_deferred_group_failure_has_no_fallback(phase, monkeypatch):
    _, deferred = pair('sp5')
    provider = deferred.group_principal_block
    original_factor = driver.spilu
    count = []
    def group(group):
        if phase == 'extraction' and group == 1: raise MemoryError('bounded extraction failure')
        return provider(group)
    def factor(block, **kwargs):
        count.append(1)
        if phase == 'factorization' and len(count) == 2: raise RuntimeError('fixed factor failure')
        return original_factor(block, **kwargs)
    def forbidden(*args, **kwargs): raise AssertionError('No GMRES after setup failure')
    monkeypatch.setattr(deferred, 'group_principal_block', group)
    monkeypatch.setattr(driver, 'spilu', factor)
    monkeypatch.setattr(driver, 'gmres', forbidden)
    forbid_assembled(monkeypatch)
    with pytest.raises((MemoryError, RuntimeError), match='group 1.*no solver fallback') as caught:
        driver.CoupledMomentEigenSolver(deferred).solve(preconditioner='group_ilu_coarse')
    assert len(caught.value.completed_groups) == 1


@pytest.mark.parametrize('family', ['sp3', 'sp5'])
@pytest.mark.parametrize('relative,accepted', [(.9e-7, True), (1.1e-7, False)])
def test_true_inner_gate_at_campaign_threshold(family, relative, accepted, monkeypatch):
    assembled, deferred = pair(family)
    matrix = assembled.assembled.toarray()
    forbid_assembled(monkeypatch)
    def gmres(operator, rhs, **kwargs):
        assert isinstance(operator, LinearOperator)
        assert kwargs['rtol'] == 1e-7 and kwargs['atol'] == 0
        # Report success in both cases: only the independent true-action gate
        # decides whether this controlled relative error is accepted.
        return (1+relative)*np.linalg.solve(matrix, rhs), 0
    monkeypatch.setattr(driver, 'gmres', gmres)
    result = driver.CoupledMomentEigenSolver(deferred).solve(max_outer=1, **GATES)
    first = result.history[0]
    np.testing.assert_allclose(first['linear_residual'], relative, rtol=1e-12, atol=1e-14)
    assert ('k_eff' in first) == accepted
    assert (result.reason == 'inner_failure_or_full_linear_residual') != accepted


@pytest.mark.parametrize('gate', ['k_tolerance', 'source_tolerance', 'residual_tolerance'])
def test_each_outer_campaign_gate_independently_rejects(gate, monkeypatch):
    grid, materials = fixture('sp5')
    loose = dict(inner_tolerance=1e-7, k_tolerance=.9, source_tolerance=.9, residual_tolerance=.9)
    initial = Operator(grid, materials, family='sp5', storage='deferred')
    first = driver.CoupledMomentEigenSolver(initial).solve(max_outer=1, **loose)
    # Scale this synthetic fission fixture so the first k is 1.01, within the
    # loose control gates; loss/diffusion/boundary coefficients stay unchanged.
    scale = 1.01/first.history[0]['k_eff']
    materials = [replace(m, nu_sigma_f=m.nu_sigma_f*scale) for m in materials]
    op = Operator(grid, materials, family='sp5', storage='deferred')
    forbid_assembled(monkeypatch)
    control = driver.CoupledMomentEigenSolver(op).solve(max_outer=1, **loose)
    assert control.converged
    thresholds = dict(loose, **{gate: GATES[gate]})
    rejected = driver.CoupledMomentEigenSolver(op).solve(max_outer=1, **thresholds)
    assert not rejected.converged and rejected.reason == 'outer_iteration_limit'
    assert rejected.history[0]['gmres_info'] == 0
    field = dict(k_tolerance='k_change', source_tolerance='source_change', residual_tolerance='full_residual')[gate]
    assert rejected.history[0][field] > GATES[gate]
    assert rejected.provenance[gate] == GATES[gate]
