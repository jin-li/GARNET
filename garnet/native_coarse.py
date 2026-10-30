"""Fixed, experimental native two-level correction; no physical admission.

The implicit transfer preserves every (moment, group) component. Restriction
is a SUM of integrated balances, never a volume average. Nothing here changes
the operator or the smoother supplied by the caller.
"""
import hashlib
import heapq
import resource
import sys
import warnings
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve, get_lapack_funcs
from scipy.sparse.linalg import LinearOperator


def connected_aggregates(centers, edges):
    """Deterministic fixed-budget (min(cells,128)) connected aggregation.

    Components are ordered by smallest cell index and seeded there first.
    Remaining seeds maximize minimum squared Euclidean distance to any seed;
    exact ties choose the smallest cell index. Graph Voronoi assignment uses
    positive centroid distances and lexicographic (distance, seed ordinal,
    predecessor cell) ties. Thus predecessor chains stay inside aggregates.
    Material identities are deliberately absent from this preconditioner map.
    """
    centers = np.asarray(centers, dtype=float)
    if (centers.ndim != 2 or centers.shape[0] < 1 or
            centers.shape[1] not in (1, 2, 3) or not np.all(np.isfinite(centers))):
        raise ValueError('Finite cell centroids in 1D, 2D or 3D required')
    n = len(centers)
    budget = min(n, 128)
    neighbors = [set() for _ in range(n)]
    for i, j in edges:
        if (isinstance(i, (bool, np.bool_)) or isinstance(j, (bool, np.bool_)) or
                not isinstance(i, (int, np.integer)) or not isinstance(j, (int, np.integer)) or
                not 0 <= i < n or not 0 <= j < n or i == j):
            raise ValueError('Invalid interior face cell indices')
        neighbors[i].add(int(j)); neighbors[j].add(int(i))
    graph = []
    for i, adjacent in enumerate(neighbors):
        row = []
        for j in sorted(adjacent):
            distance = float(np.linalg.norm(centers[j] - centers[i]))
            if not np.isfinite(distance) or distance <= 0:
                raise ValueError('Interior centroid edge lengths must be finite and positive')
            row.append((j, distance))
        graph.append(row)
    del neighbors
    seen = np.zeros(n, dtype=bool)
    seeds = []
    for root in range(n):
        if seen[root]:
            continue
        seeds.append(root)
        if len(seeds) > budget:
            raise ValueError('Mesh connected components exceed fixed aggregate budget 128')
        seen[root] = True
        stack = [root]
        while stack:
            for j, _ in graph[stack.pop()]:
                if not seen[j]:
                    seen[j] = True; stack.append(j)
    nearest = np.full(n, np.inf)
    for seed in seeds:
        distance = np.sum((centers - centers[seed])**2, axis=1)
        if not np.all(np.isfinite(distance)):
            raise ValueError('Nonfinite Euclidean seed distances')
        np.minimum(nearest, distance, out=nearest)
    selected = np.zeros(n, dtype=bool)
    selected[seeds] = True
    while len(seeds) < budget:
        # Selected cells have priority below all nonnegative distances,
        # including distinct cells at the same Euclidean coordinates.
        nearest[selected] = -1.
        seed = int(np.argmax(nearest))
        seeds.append(seed); selected[seed] = True
        distance = np.sum((centers - centers[seed])**2, axis=1)
        if not np.all(np.isfinite(distance)):
            raise ValueError('Nonfinite Euclidean seed distances')
        np.minimum(nearest, distance, out=nearest)
    distances = np.full(n, np.inf)
    owners = np.full(n, budget, dtype=np.int64)
    predecessor = np.full(n, n, dtype=np.int64)
    queue = []
    for ordinal, seed in enumerate(seeds):
        distances[seed] = 0.; owners[seed] = ordinal; predecessor[seed] = seed
        heapq.heappush(queue, (0., ordinal, seed, seed))
    while queue:
        distance, owner, previous, i = heapq.heappop(queue)
        if (distance, owner, previous) != (distances[i], owners[i], predecessor[i]):
            continue
        for j, length in graph[i]:
            candidate = (distance + length, owner, i)
            if not np.isfinite(candidate[0]) or candidate[0] <= distance:
                raise ValueError('Graph distances must increase finitely along edges')
            if candidate < (distances[j], owners[j], predecessor[j]):
                distances[j], owners[j], predecessor[j] = candidate
                heapq.heappush(queue, (*candidate, j))
    # Verify nonempty coverage and actual face connectivity independently of
    # predecessor bookkeeping (which is not itself a connectivity oracle).
    visited = np.zeros(n, dtype=bool)
    for ordinal, seed in enumerate(seeds):
        if owners[seed] != ordinal:
            raise ValueError('Aggregate lost its seed')
        stack = [seed]; visited[seed] = True; count = 0
        while stack:
            i = stack.pop(); count += 1
            for j, _ in graph[i]:
                if owners[j] == ordinal and not visited[j]:
                    visited[j] = True; stack.append(j)
        if count != np.count_nonzero(owners == ordinal):
            raise ValueError('Disconnected aggregate')
    if not np.all(visited) or np.any(owners >= budget):
        raise ValueError('Incomplete aggregate coverage')
    owners.setflags(write=False)
    return owners


def operator_aggregates(operator):
    """Use actual interior-face adjacency and cell centroids of native meshes."""
    from .corrected_native_moments import CorrectedNativeMomentOperator
    from .coupled_moments import CoupledMomentOperator
    if isinstance(operator, CorrectedNativeMomentOperator):
        mesh = operator.grid.mesh
        return connected_aggregates(mesh.centroid, ((f[0], f[1]) for f in mesh.faces))
    if isinstance(operator, CoupledMomentOperator):
        mesh = operator.mesh
        # OrthogonalMesh already stores physical centroids and interior owners;
        # use centroid separation, not an inferred index-space geometry.
        return connected_aggregates(mesh.centroids, ((f.left, f.right) for f in mesh.interior))
    raise ValueError('Native operator with explicit cell geometry required')


class NativeCoarsePreconditioner:
    """M2 r = Q r + M(r - A Q r), Q = P (P.T A P)^-1 P.T.

    ``smoother`` is an already built fixed linear callable (or LinearOperator).
    No group factors are built here, allowing research harnesses to reuse them.
    ``aggregate_ids`` is a known connected mapping for small algebraic fixtures;
    production callers should use ``from_operator`` for geometry validation.
    Matrix and smoother must remain unchanged for the lifetime of this object.
    For deferred storage, supply a float64 LinearOperator and a callable
    ``block_iterator`` yielding bounded (cell rows, cell columns, q-by-q blocks).
    Entries may repeat and are summed before one final float64 cast. The matrix
    action, including every group coupling, supplies the correction residual.
    """
    def __init__(self, matrix, shape, aggregate_ids, smoother, *, max_rows=4096,
                 max_nnz=1_000_000, block_iterator=None):
        started = perf_counter()
        if (len(shape) != 3 or any(isinstance(x, (bool, np.bool_)) or
                not isinstance(x, (int, np.integer)) or x < 1 for x in shape)):
            raise ValueError('Positive native (cells, moments, groups) shape required')
        self.shape = tuple(int(x) for x in shape)
        cells, moments, groups = self.shape
        self.q = moments*groups
        self.size = cells*self.q
        streamed = block_iterator is not None
        if streamed:
            if (not callable(block_iterator) or not isinstance(matrix, LinearOperator) or
                    matrix.shape != (self.size, self.size) or matrix.dtype != np.dtype('float64')):
                raise ValueError('Native float64 LinearOperator and callable full block iterator required')
        elif (not sparse.isspmatrix_csr(matrix) or matrix.shape != (self.size, self.size) or
                matrix.dtype != np.dtype('float64')):
            raise ValueError('Native float64 square CSR matrix required')
        for value in (max_rows, max_nnz):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError('Positive integer CSR chunk limits required')
        ids = np.asarray(aggregate_ids)
        if (ids.shape != (cells,) or ids.dtype.kind not in 'iu' or
                np.any(ids < 0) or np.max(ids) >= min(cells, 128)):
            raise ValueError('One integer aggregate id per cell within fixed budget required')
        self.aggregate_ids = np.array(ids, dtype=np.int64, copy=True)
        self.aggregate_ids.setflags(write=False)
        self.n_aggregates = int(np.max(ids)) + 1
        if len(np.unique(ids)) != self.n_aggregates:
            raise ValueError('Aggregate ids must be contiguous and nonempty')
        if not callable(smoother) and not isinstance(smoother, LinearOperator):
            raise ValueError('Fixed linear smoother callable required')
        self.matrix = matrix
        self.smoother = smoother
        self.dimension = self.n_aggregates*self.q
        if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
            raise ValueError('Extended precision longdouble accumulation required on this platform')
        coarse = np.zeros((self.dimension, self.dimension), dtype=np.longdouble)
        chunks = 0; largest_nnz = 0; workspace = 0
        fine_entries = 0
        if streamed:
            # Stream uncoalesced full-q blocks. Every duplicate pair contributes
            # in extended precision, including cancellation across batches.
            limit = max(1, min(max_rows//self.q, max_nnz//(self.q*self.q)))
            components = np.arange(self.q, dtype=np.int64)
            for cell_rows, cell_columns, blocks in block_iterator():
                cell_rows, cell_columns = np.asarray(cell_rows), np.asarray(cell_columns)
                blocks = np.asarray(blocks)
                if (cell_rows.ndim != 1 or cell_columns.shape != cell_rows.shape or
                        cell_rows.dtype.kind not in 'iu' or cell_columns.dtype.kind not in 'iu' or
                        blocks.shape != (len(cell_rows), self.q, self.q) or
                        np.any(cell_rows < 0) or np.any(cell_rows >= cells) or
                        np.any(cell_columns < 0) or np.any(cell_columns >= cells)):
                    raise ValueError('Invalid native full block stream')
                for start in range(0, len(cell_rows), limit):
                    stop = min(start+limit, len(cell_rows))
                    values = blocks[start:stop].astype(np.longdouble)
                    if not np.all(np.isfinite(values)):
                        raise ValueError('Nonfinite fine matrix coefficients')
                    coarse_rows = self.q*self.aggregate_ids[cell_rows[start:stop], None, None]+components[None, :, None]
                    coarse_columns = self.q*self.aggregate_ids[cell_columns[start:stop], None, None]+components[None, None, :]
                    np.add.at(coarse, (coarse_rows, coarse_columns), values)
                    count = values.size
                    fine_entries += count
                    chunks += 1; largest_nnz = max(largest_nnz, count)
                    workspace = max(workspace, sum(a.nbytes for a in
                        (cell_rows, cell_columns, blocks, values, coarse_rows, coarse_columns, components)))
            # The iterator owns bounded source batches; no fine matrix or global
            # row dictionary is retained by coarse construction.
        else:
            start = 0
            while start < self.size:
                stop = min(start + max_rows, self.size)
                counts = np.diff(matrix.indptr[start:stop+1]).astype(np.int64)
                cumulative = np.cumsum(counts, dtype=np.int64)
                stop = start + max(1, int(np.searchsorted(cumulative, max_nnz, side='right')))
                lo, hi = int(matrix.indptr[start]), int(matrix.indptr[stop])
                columns = matrix.indices[lo:hi]
                values = matrix.data[lo:hi].astype(np.longdouble)
                if not np.all(np.isfinite(values)):
                    raise ValueError('Nonfinite fine matrix coefficients')
                rows = np.repeat(np.arange(start, stop, dtype=np.int64), np.diff(matrix.indptr[start:stop+1]))
                coarse_rows = self.q*self.aggregate_ids[rows//self.q] + rows % self.q
                coarse_columns = self.q*self.aggregate_ids[columns//self.q] + columns % self.q
                # add.at preserves every duplicate, including repeated coarse pairs
                # within a row/chunk. Cast only after all extended precision sums.
                np.add.at(coarse, (coarse_rows, coarse_columns), values)
                chunks += 1; largest_nnz = max(largest_nnz, hi-lo)
                workspace = max(workspace, sum(a.nbytes for a in
                    (counts, cumulative, values, rows, coarse_rows, coarse_columns)))
                start = stop
            del counts, cumulative, values, rows, coarse_rows, coarse_columns, columns
            fine_entries = int(matrix.nnz)
        extended_bytes = coarse.nbytes
        self.coarse_matrix = np.array(coarse, dtype=np.float64, order='F')
        del coarse
        if not np.all(np.isfinite(self.coarse_matrix)):
            raise ValueError('Nonfinite coarse matrix')
        accumulation_seconds = perf_counter()-started
        with warnings.catch_warnings():
            warnings.simplefilter('error', LinAlgWarning)
            try:
                self._lu, self._piv = lu_factor(self.coarse_matrix, check_finite=True)
            except (LinAlgWarning, ValueError) as exc:
                raise ValueError('Singular or invalid coarse LU; no regularization or fallback') from exc
        if not np.all(np.isfinite(self._lu)) or np.any(np.diag(self._lu) == 0):
            raise ValueError('Singular or nonfinite coarse LU')
        gecon = get_lapack_funcs('gecon', (self._lu,))
        rcond, info = gecon(self._lu, float(np.linalg.norm(self.coarse_matrix, 1)), norm='1')
        if info != 0 or not np.isfinite(rcond) or rcond < 1/np.finfo(float).max:
            raise ValueError('Invalid or singular coarse condition estimate')
        self.coarse_solves = 0
        self.extra_a_matvecs = 0
        self.applications = 0
        self.setup_coarse_solves = 0
        self.max_coarse_solve_relative_residual = 0.
        self.max_coarse_solve_componentwise_ratio = 0.
        self._solve_coarse(np.arange(1., self.dimension+1.), setup=True)
        for array in (self.coarse_matrix, self._lu, self._piv):
            array.setflags(write=False)
        self._metadata = dict(coarse_dimension=self.dimension, aggregate_count=self.n_aggregates,
            aggregate_budget=min(cells, 128), aggregate_sha256=hashlib.sha256(
                self.aggregate_ids.astype('<i8', copy=False).tobytes()).hexdigest(),
            coarse_sha256=hashlib.sha256(self.coarse_matrix.tobytes(order='C')).hexdigest(),
            restriction='sum integrated balances; no volume averaging',
            coarse_layout='(M*G)*aggregate + G*moment + group',
            accumulation=('bounded full block stream, duplicate-safe longdouble; one float64 cast' if streamed else
                          'bounded CSR chunks, duplicate-safe longdouble; one float64 cast'),
            accumulation_dtype=str(np.dtype(np.longdouble)), csr_chunks=0 if streamed else chunks,
            full_block_chunks=chunks if streamed else 0,
            fine_entries_accumulated=fine_entries, coarse_nnz=int(np.count_nonzero(self.coarse_matrix)),
            accumulation_seconds=accumulation_seconds,
            largest_chunk_nnz=largest_nnz, max_rows=int(max_rows), max_nnz=int(max_nnz),
            chunk_workspace_array_bytes=workspace, extended_coarse_array_bytes=extended_bytes,
            retained_array_bytes=sum(a.nbytes for a in (self.aggregate_ids, self.coarse_matrix, self._lu, self._piv)),
            coarse_matrix_array_bytes=self.coarse_matrix.nbytes, lu_array_bytes=self._lu.nbytes,
            setup_process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform == 'darwin' else 1024),
            memory_caveat=('array bytes include bounded source block batches and accumulation arrays; exclude Python/SciPy workspaces and caller-owned action/smoother; RSS is process lifetime high-water mark' if streamed else
                'array bytes exclude Python/SciPy workspaces and caller-owned fine matrix/smoother; RSS is process lifetime high-water mark'),
            condition_estimate_1=float(1/rcond), reciprocal_condition_estimate_1=float(rcond),
            coarse_solve_check='componentwise |Ac*x-r| <= beta*(|Ac|*|x|+|r|), beta=2*d*u/(1-2*d*u), u=eps64/2; zero scale requires zero residual',
            setup_seconds=perf_counter()-started)
        self.linear_operator = LinearOperator(matrix.shape, matvec=self.apply, dtype=matrix.dtype)

    @classmethod
    def from_operator(cls, operator, smoother, **kwargs):
        started = perf_counter()
        aggregates = operator_aggregates(operator)
        mapping_seconds = perf_counter()-started
        if getattr(operator, 'storage', 'assembled') == 'deferred':
            result = cls(operator.linear_operator(), operator.shape, aggregates, smoother,
                         block_iterator=operator.iter_full_blocks, **kwargs)
        else:
            result = cls(operator.assembled, operator.shape, aggregates, smoother, **kwargs)
        result._metadata['mapping_seconds'] = mapping_seconds
        result._metadata['setup_seconds'] += mapping_seconds
        result._metadata['mapping'] = 'component-seeded farthest Euclidean, graph Dijkstra lexicographic ties'
        return result

    @property
    def metadata(self):
        return dict(self._metadata, coarse_solves=self.coarse_solves,
                    setup_coarse_solves=self.setup_coarse_solves, extra_a_matvecs=self.extra_a_matvecs,
                    applications=self.applications,
                    max_coarse_solve_relative_residual=self.max_coarse_solve_relative_residual,
                    max_coarse_solve_componentwise_ratio=self.max_coarse_solve_componentwise_ratio)

    def coarse_action(self, coarse):
        return self.coarse_matrix@np.asarray(coarse, dtype=float).reshape(self.dimension)

    def prolong(self, coarse):
        values = np.asarray(coarse, dtype=float).reshape(self.n_aggregates, self.q)
        return values[self.aggregate_ids].reshape(-1)

    def restrict(self, fine):
        values = np.asarray(fine, dtype=float).reshape(self.shape[0], self.q)
        result = np.zeros((self.n_aggregates, self.q))
        np.add.at(result, self.aggregate_ids, values)
        return result.reshape(-1)

    def _solve_coarse(self, rhs, *, setup=False):
        result = lu_solve((self._lu, self._piv), rhs, check_finite=True)
        if setup:
            self.setup_coarse_solves += 1
        else:
            self.coarse_solves += 1
        residual = np.abs(self.coarse_action(result)-rhs)
        u = np.finfo(float).eps/2
        beta = (2*self.dimension*u)/(1-2*self.dimension*u)
        if not np.all(np.isfinite(result)) or not np.all(np.isfinite(residual)):
            raise ValueError('Coarse solve backward residual check failed; no fallback')
        for start in range(0, self.dimension, 64):
            stop = min(start+64, self.dimension)
            scale = np.abs(self.coarse_matrix[start:stop])@np.abs(result)+np.abs(rhs[start:stop])
            if not np.all(np.isfinite(scale)) or np.any(residual[start:stop] > beta*scale):
                raise ValueError('Coarse solve backward residual check failed; no fallback')
            bound = beta*scale
            ratios = np.divide(residual[start:stop], bound,
                               out=np.zeros(stop-start), where=bound != 0)
            self.max_coarse_solve_componentwise_ratio = max(
                self.max_coarse_solve_componentwise_ratio, float(np.max(ratios)))
        denominator = np.linalg.norm(rhs, np.inf)
        relative = float(np.max(residual)/denominator) if denominator else float(np.max(residual))
        self.max_coarse_solve_relative_residual = max(self.max_coarse_solve_relative_residual, relative)
        return result

    def apply(self, rhs):
        rhs = np.asarray(rhs, dtype=float).reshape(self.size)
        if not np.all(np.isfinite(rhs)):
            raise ValueError('Finite preconditioner RHS required')
        self.applications += 1
        correction = self.prolong(self._solve_coarse(self.restrict(rhs)))
        self.extra_a_matvecs += 1
        remainder = rhs-self.matrix@correction
        smooth = self.smoother@remainder if isinstance(self.smoother, LinearOperator) else self.smoother(remainder)
        result = correction+np.asarray(smooth).reshape(self.size)
        if not np.all(np.isfinite(result)):
            raise ValueError('Nonfinite two-level preconditioner result')
        return result
