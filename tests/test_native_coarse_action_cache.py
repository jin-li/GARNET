"""Numerical parity for opt-in A P caching; no production qualification."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator

import garnet.native_coarse as coarse_module
from garnet.corrected_native_moments import CorrectedNativeMomentOperator as Operator
from garnet.native_coarse import NativeCoarsePreconditioner as Coarse
from garnet.native_coarse_action_cache import ProlongationActionCache
from test_native_compact import fixture


def error_metrics(actual, expected, scale):
    delta = np.abs(actual-expected)
    normwise = np.linalg.norm(delta)/max(np.linalg.norm(expected), np.finfo(float).tiny)
    ratios = np.divide(delta, scale, out=np.zeros_like(delta), where=scale != 0)
    assert np.all(delta[scale == 0] == 0)
    return float(normwise), float(np.max(ratios))


@pytest.mark.parametrize('family', ['sp3', 'sp5'])
@pytest.mark.parametrize('batch', [1, 7, 128])
def test_native_action_preconditioner_and_exact_coarse_parity(family, batch, monkeypatch):
    grid, mats = fixture(family)
    op = Operator(grid, mats, family=family, storage='deferred',
                  gradient_chunk_cells=1, batch_size=batch)
    # Nontrivial connected aggregation on the four-cell skewed mesh.
    ids = np.array([0, 0, 1, 1], dtype=np.int64)
    ids.setflags(write=False)
    monkeypatch.setattr(coarse_module, 'operator_aggregates', lambda operator: ids)
    before = [(m.data.tobytes(), m.indices.tobytes(), m.indptr.tobytes())
              for _, _, m in op.compact.gradient_chunks]
    # Same coupled fixed smoother, deliberately not an exact inverse.
    rng = np.random.default_rng(651)
    S = .03*np.eye(op.size)+rng.normal(scale=1e-4, size=(op.size, op.size))
    original = Coarse.from_operator(op, lambda r: S@r, max_rows=7, max_nnz=37)
    cached = Coarse.from_operator(op, lambda r: S@r, max_rows=7, max_nnz=37,
        cache_prolongation_action=True, cache_max_bytes=8_000_000)
    cache = cached.prolongation_action_cache
    for name in ('aggregate_ids', 'coarse_matrix', '_lu', '_piv'):
        assert getattr(original, name).tobytes() == getattr(cached, name).tobytes()
    for key in ('full_block_chunks', 'fine_entries_accumulated', 'largest_chunk_nnz',
                'aggregate_sha256', 'coarse_sha256'):
        assert original.metadata[key] == cached.metadata[key]
    assert original.metadata['prolongation_action_cache'] == {'enabled': False}
    vectors = [rng.normal(size=cached.dimension), np.eye(1, cached.dimension)[0],
               np.ones(cached.dimension), np.r_[np.ones(cached.q), -np.ones(cached.q)],
               rng.normal(size=cached.dimension)*1e-150]
    maximum = np.zeros(4)
    # Small reference only: the implementation never constructs full fine A.
    dense_native = np.column_stack([op.linear_operator()@e for e in np.eye(op.size)])
    for z in vectors:
        actual = cache.action(z)
        expected = op.linear_operator()@cached.prolong(z)
        scale = np.abs(dense_native)@np.abs(cached.prolong(z))+np.abs(expected)
        action_errors = error_metrics(actual, expected, scale)
        assert action_errors[0] < 5e-13 and action_errors[1] < 5e-13
        rhs = rng.normal(size=op.size)*max(np.max(np.abs(z)), 1e-150)
        old, new = original.apply(rhs), cached.apply(rhs)
        # Scale includes both correction and smoother remainder paths to avoid
        # interpreting near-zero cancellation as a relative output error.
        coeff = original._solve_coarse(original.restrict(rhs))
        correction = original.prolong(coeff)
        smooth_scale = np.abs(S)@(np.abs(rhs)+np.abs(dense_native)@np.abs(correction))
        pre_errors = error_metrics(new, old, np.abs(correction)+smooth_scale+np.abs(old))
        assert pre_errors[0] < 5e-13 and pre_errors[1] < 5e-13
        maximum = np.maximum(maximum, (*action_errors, *pre_errors))
    assert cached.metadata['extra_a_matvecs'] == 0
    assert cached.metadata['cached_ap_matvecs'] == len(vectors)
    assert original.metadata['extra_a_matvecs'] == len(vectors)
    def forbidden(_):
        raise AssertionError('extra native A action used')
    cached.matrix = LinearOperator((op.size, op.size), matvec=forbidden, dtype=float)
    cached.apply(np.ones(op.size))
    assert before == [(m.data.tobytes(), m.indices.tobytes(), m.indptr.tobytes())
                      for _, _, m in op.compact.gradient_chunks]
    assert cache.matrix.shape == (op.size, cached.dimension)
    assert cache.matrix.has_sorted_indices
    assert cache.metadata['actions'] == 2*len(vectors)+1
    for array in (cache.matrix.data, cache.matrix.indices, cache.matrix.indptr):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0
        base = array
        while isinstance(base, np.ndarray):
            assert not base.flags.writeable
            base = base.base
    print(f'{family} batch={batch} max AP norm/scaled={maximum[:2].tolist()} '
          f'preconditioner norm/scaled={maximum[2:].tolist()}')


def tiny_operator():
    return SimpleNamespace(shape=(3, 1, 1),
        grid=SimpleNamespace(mesh=SimpleNamespace(faces=[(0, 1), (1, 2)])),
        compact=SimpleNamespace(gradient_chunks=()))


def test_duplicate_cancellation_across_batches_and_no_pruning():
    ids = np.array([0, 0, 1])
    cache = ProlongationActionCache(tiny_operator(), ids, 10000)
    blocks = [np.array([[[v]]]) for v in (1e16, 1., -1e16, 1e-300)]
    before = [v.tobytes() for v in blocks]
    def stream():
        for block in blocks[:3]:
            yield np.array([0]), np.array([1]), block
        yield np.array([2]), np.array([2]), blocks[-1]
    list(cache.wrap(stream)())
    cache.finalize()
    assert cache.matrix.toarray()[0, 0] == 1.
    assert cache.action(np.ones(2))[0] == 1.
    assert cache.action(np.ones(2))[2] == 1e-300
    assert cache.matrix.data.size == 6  # includes all structural zero slots
    assert before == [v.tobytes() for v in blocks]


@pytest.mark.parametrize('budget', [None, True, False, 0, -1, 1.5, '10000'])
def test_budget_invalid(budget):
    with pytest.raises(ValueError, match='explicit positive integer'):
        ProlongationActionCache(tiny_operator(), np.array([0, 0, 1]), budget)


def test_budget_rejects_before_numeric_allocation_and_counts_transients(monkeypatch):
    op, ids = tiny_operator(), np.array([0, 0, 1])
    probe = ProlongationActionCache(op, ids, 10000)
    peak = probe.metadata['estimated_cache_peak_bytes']
    assert peak == (np.dtype(np.longdouble).itemsize+16)*6+2*(4+6)*8
    real_zeros = np.zeros
    def guarded_zeros(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 3:
            raise AssertionError('numerical cache allocated before budget rejection')
        return real_zeros(shape, *args, **kwargs)
    monkeypatch.setattr(np, 'zeros', guarded_zeros)
    with pytest.raises(ValueError, match='estimated peak'):
        ProlongationActionCache(op, ids, peak-1)


def test_support_and_unexpected_stream_slots_fail_closed():
    from scipy import sparse
    op = tiny_operator()
    op.grid.mesh.faces = [(0, 1)]
    op.compact.gradient_chunks = ((0, 3, sparse.bsr_matrix(np.eye(3)+np.eye(3, k=2))),)
    with pytest.raises(ValueError, match='gradient support'):
        ProlongationActionCache(op, np.array([0, 0, 1]), 10000)
    op.compact.gradient_chunks = ()
    cache = ProlongationActionCache(op, np.array([0, 0, 1]), 10000)
    with pytest.raises(ValueError, match='outside A P cache pattern'):
        list(cache.wrap(lambda: iter([(np.array([0]), np.array([2]), np.ones((1, 1, 1)))]))())


def test_explicit_unsupported_modes_and_invalid_enable_flag():
    grid, mats = fixture('sp3')
    assembled = Operator(grid, mats, family='sp3')
    with pytest.raises(ValueError, match='deferred corrected-native'):
        Coarse.from_operator(assembled, lambda r: r, cache_prolongation_action=True, cache_max_bytes=10000)
    with pytest.raises(ValueError, match='explicit positive integer'):
        Coarse.from_operator(assembled, lambda r: r, cache_prolongation_action=True)
    with pytest.raises(ValueError, match='must be boolean'):
        Coarse.from_operator(assembled, lambda r: r, cache_prolongation_action='yes')
