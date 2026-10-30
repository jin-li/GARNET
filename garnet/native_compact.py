"""Bounded deferred corrected-native prototype; no solver/coarse integration.

Gradient blocks have shape (q*dimension, q). Only one cell's normalized SVD
patch and one bounded chunk's mutable coefficients exist during construction.
Payload accounting is array storage, not process RSS or a production fit claim.
"""
from types import MappingProxyType
import numpy as np
from scipy import sparse
from .moment_materials import _immutable


def validate_group(group, count):
    if isinstance(group, (bool, np.bool_)) or not isinstance(group, (int, np.integer)) or not 0 <= group < count:
        raise ValueError('group must be an integer in [0, n_groups)')


def _frozen(array):
    """Freeze one bounded chunk, retaining its dtype (including sparse indices)."""
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class CompactNativeStorage:
    def __init__(self, op, inverse_D, distances, bd, specs, chunk_cells, batch_size):
        self.op, self.batch_size = op, batch_size
        self.q = q = op.n_groups*op.n_moments
        mesh = op.grid.mesh
        self.dim = dim = mesh.centroid.shape[1]
        identity = np.eye(q)
        # Topological indices only; equations are generated for one cell at a time.
        adjacent = [[] for _ in range(op.n_cells)]
        boundary_at = [[] for _ in range(op.n_cells)]
        faces, boundary = [], []
        for f, ((i, j, area, _), center, normal, (di, dj)) in enumerate(zip(
                mesh.faces, mesh.face_centers, mesh.face_normals, distances)):
            i, j = int(i), int(j)
            adjacent[i].append((f, j, 1)); adjacent[j].append((f, i, -1))
            T = area*op._inverse(di*inverse_D[i]+dj*inverse_D[j], 'face resistance')
            if not np.all(np.isfinite(T)):
                raise ValueError('Nonfinite corrected native face coefficient')
            faces.append((i, j, _immutable(T), _immutable(center-mesh.centroid[i]-di*normal),
                          _immutable(center-mesh.centroid[j]+dj*normal)))
        for f, ((i, area, _), center, normal, d, spec) in enumerate(zip(
                mesh.bfaces, mesh.bface_centers, mesh.bface_normals, bd, specs)):
            i = int(i)
            boundary_at[i].append(f)
            if spec == 'vacuum':
                C = op.diffusion[i]/d
                K = area*op.B@op._inverse(C+op.B, 'vacuum C+B')@C
            else:
                K = np.zeros((q, q))
            if not np.all(np.isfinite(K)):
                raise ValueError('Nonfinite corrected native boundary coefficient')
            boundary.append((i, _immutable(K), _immutable(center-mesh.centroid[i]-d*normal)))
        self.faces, self.boundary = tuple(faces), tuple(boundary)
        chunks = []
        for start in range(0, op.n_cells, chunk_cells):
            stop = min(start+chunk_cells, op.n_cells)
            data, indices, indptr = [], [], [0]
            for i in range(start, stop):
                patch = []
                def equation(blocks):
                    return np.stack(blocks, axis=2).reshape(q, q*dim)
                for f, neighbor, sign in adjacent[i]:
                    left, right = mesh.faces[f][:2]
                    vector = sign*(mesh.centroid[right]-mesh.centroid[left])
                    normal = sign*mesh.face_normals[f]
                    d = distances[f][1 if sign == 1 else 0]
                    jump = inverse_D[neighbor]@op.diffusion[i]-identity
                    A = equation([vector[k]*identity+d*normal[k]*jump for k in range(dim)])
                    patch.append((A, {neighbor: identity, i: -identity}))
                for f in boundary_at[i]:
                    normal = mesh.bface_normals[f]
                    r = mesh.bface_centers[f]-mesh.centroid[i]
                    B = op.B if specs[f] == 'vacuum' else np.zeros((q, q))
                    A = equation([normal[k]*op.diffusion[i]+r[k]*B for k in range(dim)]
                                 if specs[f] == 'vacuum' else [normal[k]*identity for k in range(dim)])
                    patch.append((A, {i: -B}))
                if not patch:
                    raise ValueError(f'rank-deficient gradient patch at cell {i}')
                A = np.concatenate([a for a, _ in patch])
                scales = np.linalg.norm(A, axis=1)
                if np.any(scales == 0) or not np.all(np.isfinite(scales)):
                    raise ValueError(f'rank-deficient gradient patch at cell {i}')
                A = A/scales[:, None]
                U, s, Vh = np.linalg.svd(A, full_matrices=False)
                if len(s) < q*dim or s[-1] <= max(1e-12, 1/op.condition_limit)*s[0]:
                    raise ValueError(f'rank-deficient/ill-conditioned gradient patch at cell {i}')
                inverse = (Vh.T/s)@U.T
                row = {}
                for k, (_, rhs) in enumerate(patch):
                    for col, value in rhs.items():
                        contribution = inverse[:, k*q:(k+1)*q]@(value/scales[k*q:(k+1)*q, None])
                        row[col] = row.get(col, 0)+contribution
                for col in sorted(row):
                    indices.append(col); data.append(row[col])
                indptr.append(len(data))
            matrix = sparse.bsr_matrix((np.asarray(data), np.asarray(indices), np.asarray(indptr)),
                                       shape=((stop-start)*q*dim, op.size))
            if not np.all(np.isfinite(matrix.data)):
                raise ValueError('Nonfinite corrected native gradient coefficient')
            # SciPy may narrow indices at construction; freeze the actual arrays.
            matrix.data = _frozen(matrix.data)
            matrix.indices = _frozen(matrix.indices)
            matrix.indptr = _frozen(matrix.indptr)
            chunks.append((start, stop, matrix))
        self.gradient_chunks = tuple(chunks)
        self._chunk_cells = chunk_cells
        self.provenance = MappingProxyType(dict(storage='deferred', physical_admission=False,
            gradient_format='rectangular BSR', gradient_chunk_cells=chunk_cells,
            batch_size=batch_size, global_matrix_materialized=False,
            solver_integration=False, coarse_integration=False, pruning=False))

    def payload_accounting(self):
        """Retained NumPy payload, with shared material arrays counted once.

        Excludes Python/SciPy objects, source mesh/materials, action workspaces,
        constructor temporaries, and requested group matrices. Not measured RSS.
        """
        gradient = sum(a.nbytes for _, _, m in self.gradient_chunks for a in (m.data, m.indices, m.indptr))
        seen = set()
        def unique_bytes(arrays):
            result = 0
            for a in arrays:
                if id(a) not in seen:
                    seen.add(id(a)); result += a.nbytes
            return result
        reaction = unique_bytes(self.op.reaction)
        diffusion = unique_bytes(self.op.diffusion)
        faces = unique_bytes(a for f in self.faces for a in f[2:])
        boundary = unique_bytes(a for f in self.boundary for a in f[1:])
        other = unique_bytes((self.op.volumes, self.op.B, *self.op.C))
        return dict(gradient_bytes=gradient, reaction_bytes=reaction, diffusion_bytes=diffusion,
                    face_bytes=faces, boundary_bytes=boundary, other_bytes=other,
                    total_array_bytes=gradient+reaction+diffusion+faces+boundary+other)

    def gradients(self, x):
        out = np.empty((self.op.n_cells, self.q, self.dim))
        for start, stop, matrix in self.gradient_chunks:
            out[start:stop] = (matrix@x.ravel()).reshape(stop-start, self.q, self.dim)
        return out

    def _currents(self, x, H, boundary=False):
        entries = self.boundary if boundary else self.faces
        for start in range(0, len(entries), self.batch_size):
            batch = entries[start:start+self.batch_size]
            owners = np.array([f[0] for f in batch])
            if boundary:
                K = np.stack([f[1] for f in batch]); t = np.stack([f[2] for f in batch])
                trace = x[owners]+np.einsum('bad,bd->ba', H[owners], t)
                yield start, owners, None, np.einsum('bak,bk->ba', K, trace)
            else:
                neighbors = np.array([f[1] for f in batch])
                T = np.stack([f[2] for f in batch])
                ti, tj = np.stack([f[3] for f in batch]), np.stack([f[4] for f in batch])
                trace = (x[owners]+np.einsum('bad,bd->ba', H[owners], ti)-x[neighbors]
                         -np.einsum('bad,bd->ba', H[neighbors], tj))
                yield start, owners, neighbors, np.einsum('bak,bk->ba', T, trace)

    def face_currents(self, x):
        H = self.gradients(x)
        result = []
        for boundary, entries in ((False, self.faces), (True, self.boundary)):
            out = np.empty((len(entries), self.q))
            for start, _, _, current in self._currents(x, H, boundary):
                out[start:start+len(current)] = current
            result.append(out.reshape(-1, self.op.n_moments, self.op.n_groups))
        return tuple(result)

    def action(self, x):
        out = np.empty_like(x)
        for start in range(0, len(x), self.batch_size):
            stop = min(start+self.batch_size, len(x))
            out[start:stop] = np.einsum('bak,bk->ba', np.stack(self.op.reaction[start:stop]), x[start:stop])
        H = self.gradients(x)
        for boundary in (False, True):
            for _, owners, neighbors, current in self._currents(x, H, boundary):
                np.add.at(out, owners, current)
                if neighbors is not None:
                    np.add.at(out, neighbors, -current)
        return out

    def _gradient_row(self, cell):
        _, _, matrix = self.gradient_chunks[cell//self._chunk_cells]
        row = cell % self._chunk_cells
        for offset in range(matrix.indptr[row], matrix.indptr[row+1]):
            yield int(matrix.indices[offset]), matrix.data[offset].reshape(self.q, self.dim, self.q)

    def _contributions(self):
        # Products retain all internal groups until the consumer selects indices.
        for i, R in enumerate(self.op.reaction):
            yield i, i, R
        for i, j, T, ti, tj in self.faces:
            for cell, t, sign in ((i, ti, 1), (j, tj, -1)):
                yield i, cell, sign*T
                yield j, cell, -sign*T
                for col, H in self._gradient_row(cell):
                    value = sign*T@np.einsum('adk,d->ak', H, t)
                    yield i, col, value
                    yield j, col, -value
        for i, K, t in self.boundary:
            yield i, i, K
            for col, H in self._gradient_row(i):
                yield i, col, K@np.einsum('adk,d->ak', H, t)

    def iter_full_blocks(self):
        rows, cols, data = [], [], []
        for i, j, block in self._contributions():
            rows.append(i); cols.append(j); data.append(block)
            if len(data) == self.batch_size:
                yield np.asarray(rows), np.asarray(cols), np.asarray(data)
                rows, cols, data = [], [], []
        if data:
            yield np.asarray(rows), np.asarray(cols), np.asarray(data)

    def group_principal_block(self, group):
        validate_group(group, self.op.n_groups)
        select = np.arange(group, self.q, self.op.n_groups)
        # Only the requested moment-by-moment group blocks are retained here.
        rows = [{} for _ in range(self.op.n_cells)]
        for ii, jj, blocks in self.iter_full_blocks():
            for i, j, block in zip(ii, jj, blocks):
                value = block[np.ix_(select, select)]
                rows[i][j] = rows[i].get(j, 0)+value
        data, indices, indptr = [], [], [0]
        for row in rows:
            for col in sorted(row):
                indices.append(col); data.append(row[col])
            indptr.append(len(data))
        n = self.op.n_cells*self.op.n_moments
        result = sparse.bsr_matrix((np.asarray(data), np.asarray(indices), np.asarray(indptr)), shape=(n, n)).tocsr()
        if not np.all(np.isfinite(result.data)):
            raise ValueError('Nonfinite corrected native group block')
        # Match assembled storage exactly: remove exact zeros, preserving every
        # nonzero signed coefficient regardless of its magnitude. This also
        # preserves the source sparsity pattern for future group ILU adapters.
        result.eliminate_zeros()
        return result
