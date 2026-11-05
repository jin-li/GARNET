"""Opt-in bounded A P cache for deferred corrected-native coarse correction.

Only the preconditioner uses this action; native residual acceptance is unchanged.
"""
import hashlib
from time import perf_counter

import numpy as np
from scipy import sparse


def validate_budget(value):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or value <= 0):
        raise ValueError('cache_max_bytes must be an explicit positive integer budget')
    return int(value)


class ProlongationActionCache:
    """Sorted structural slots, longdouble duplicate sums, one float64 cast.

    The budget bounds cache array peak, including a conservative possible SciPy
    payload copy. It is not a bound on total process or solver memory.
    """
    def __init__(self, operator, aggregate_ids, max_bytes):
        started = perf_counter()
        max_bytes = validate_budget(max_bytes)
        if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
            raise ValueError('Extended precision longdouble accumulation required on this platform')
        self.ids = aggregate_ids
        self.cells, moments, groups = operator.shape
        self.q = moments*groups
        self.n_aggregates = int(np.max(aggregate_ids))+1
        neighbors = [{i} for i in range(self.cells)]
        for face in operator.grid.mesh.faces:
            i, j = map(int, face[:2])
            neighbors[i].add(j); neighbors[j].add(i)
        # Check the actual frozen compact support before relying on two hops.
        for start, stop, matrix in operator.compact.gradient_chunks:
            for row in range(stop-start):
                columns = matrix.indices[matrix.indptr[row]:matrix.indptr[row+1]]
                if any(int(col) not in neighbors[start+row] for col in columns):
                    raise ValueError('Compact gradient support exceeds face-neighbor cache pattern')
        def columns(i):
            return sorted({int(aggregate_ids[k]) for j in neighbors[i] for k in neighbors[j]})
        self.indptr = np.zeros(self.cells+1, dtype=np.int64)
        for i in range(self.cells):
            self.indptr[i+1] = self.indptr[i]+len(columns(i))
        slots = int(self.indptr[-1])
        structure_bytes = self.indptr.nbytes+slots*np.dtype(np.int64).itemsize
        payload = slots*self.q*self.q*np.dtype(np.float64).itemsize
        extended = slots*self.q*self.q*np.dtype(np.longdouble).itemsize
        peak = extended+2*payload+2*structure_bytes
        if peak > max_bytes:
            raise ValueError(f'A P cache estimated peak {peak} bytes exceeds cache_max_bytes={max_bytes}')
        self.indices = np.empty(slots, dtype=np.int64)
        for i in range(self.cells):
            self.indices[self.indptr[i]:self.indptr[i+1]] = columns(i)
        del neighbors
        self._accumulator = np.zeros((slots, self.q, self.q), dtype=np.longdouble)
        self.actions = 0
        self.action_seconds = 0.
        self.accumulation_seconds = 0.
        self._metadata = dict(enabled=True, format='rectangular BSR', block_slots=slots,
            shape=[self.cells*self.q, self.n_aggregates*self.q], blocksize=[self.q, self.q],
            payload_bytes=payload, extended_accumulator_bytes=extended,
            pattern_array_bytes=structure_bytes, estimated_cache_peak_bytes=peak,
            cache_max_bytes=max_bytes, pruning=False, global_fine_matrix_materialized=False,
            pattern='self plus two-hop face adjacency; actual compact gradient support checked',
            pattern_sha256=hashlib.sha256(self.indptr.astype('<i8', copy=False).tobytes()+
                self.indices.astype('<i8', copy=False).tobytes()).hexdigest(),
            memory_caveat='cache array peak estimate includes longdouble, final float64, possible payload/structure copy; excludes Python topology and caller/operator/solver workspaces',
            pattern_setup_seconds=perf_counter()-started)

    def wrap(self, block_iterator):
        def stream():
            for rows, columns, blocks in block_iterator():
                started = perf_counter()
                rows, columns, blocks = np.asarray(rows), np.asarray(columns), np.asarray(blocks)
                if (rows.ndim != 1 or columns.shape != rows.shape or
                        rows.dtype.kind not in 'iu' or columns.dtype.kind not in 'iu' or
                        blocks.shape != (len(rows), self.q, self.q) or
                        np.any(rows < 0) or np.any(rows >= self.cells) or
                        np.any(columns < 0) or np.any(columns >= self.cells)):
                    raise ValueError('Invalid native full block stream for A P cache')
                for row, col, block in zip(rows, columns, blocks):
                    lo, hi = self.indptr[row:row+2]
                    target = self.ids[col]
                    offset = int(lo)+int(np.searchsorted(self.indices[lo:hi], target))
                    if offset >= hi or self.indices[offset] != target:
                        raise ValueError('Native block lies outside A P cache pattern')
                    # No float64 intermediate sum; repeated slots across batches
                    # retain cancellation-sensitive units in extended precision.
                    self._accumulator[offset] += block
                self.accumulation_seconds += perf_counter()-started
                yield rows, columns, blocks
        return stream

    def finalize(self):
        started = perf_counter()
        data = self._accumulator.astype(np.float64)
        del self._accumulator
        del self.ids
        if not np.all(np.isfinite(data)):
            raise ValueError('Nonfinite A P cache coefficients')
        self.matrix = sparse.bsr_matrix((data, self.indices, self.indptr),
            shape=(self.cells*self.q, self.n_aggregates*self.q), copy=False)
        # Freeze actual SciPy arrays and their owned backing arrays; avoid the
        # extra multi-GB immutable-bytes copy used for small compact stencils.
        for array in (data, self.indices, self.indptr, self.matrix.data,
                      self.matrix.indices, self.matrix.indptr):
            while isinstance(array, np.ndarray):
                array.setflags(write=False)
                array = array.base
        self.indices, self.indptr = self.matrix.indices, self.matrix.indptr
        self._metadata['retained_array_bytes'] = sum(a.nbytes for a in
            (self.matrix.data, self.matrix.indices, self.matrix.indptr))
        self._metadata['setup_seconds'] = (self._metadata['pattern_setup_seconds']+
            self.accumulation_seconds+perf_counter()-started)

    def action(self, coarse):
        self.actions += 1
        started = perf_counter()
        try:
            return self.matrix@coarse
        finally:
            self.action_seconds += perf_counter()-started

    @property
    def metadata(self):
        return dict(self._metadata, actions=self.actions, action_seconds=self.action_seconds,
                    accumulation_seconds=self.accumulation_seconds)
