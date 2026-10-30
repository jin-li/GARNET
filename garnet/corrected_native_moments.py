"""Small CPU full-matrix corrected FV experiment; no physical admission.

Native cell-major states are (cells, moments, groups). Full loss matrices remain on
one left-hand side. This deliberately bounded prototype is not reactor scalable.
"""
import numpy as np
from scipy import sparse
from .mesh import MeshGrid
from .corrected_mesh import _validate_corrected_mesh
from .moment_materials import MomentMaterial, _immutable
from .coupled_moments import CoupledMomentOperator, _family_tables


class CorrectedNativeMomentOperator:
    """Fixed row-normalized coupled patches, signed distances, full currents.

    ``volumes`` is the integrated-volume accessor used by the native CPU driver.
    ``storage='assembled'`` preserves the default global CSR. Explicit
    ``storage='deferred'`` retains bounded gradient chunks and supplies actions
    and exact group blocks, without solver/coarse integration or global assembly.
    ``gradient_chunk_cells`` bounds cell rows per gradient BSR chunk;
    ``batch_size`` bounds reaction/face batches and streamed full block batches.
    Deferred provenance and retained array accounting are available on ``compact``.
    Only homogeneous vacuum and reflective boundaries are supported. Local SVD
    rejects rank loss or condition above condition_limit; no regularization,
    positivity clipping, symmetric-part admission, or reaction inversion occurs.
    """
    physical_admission = False
    prototype_unknown_limit = 20_000
    _inverse = CoupledMomentOperator._inverse
    _values = CoupledMomentOperator._values
    scalar_flux = CoupledMomentOperator.scalar_flux

    def __init__(self, grid, materials, *, family='sp3', bc=None, condition_limit=1e12,
                 max_unknowns=None, storage='assembled', gradient_chunk_cells=128, batch_size=128):
        if storage not in ('assembled', 'deferred'):
            raise ValueError('storage must be assembled or deferred')
        for name, value in (('gradient_chunk_cells', gradient_chunk_cells), ('batch_size', batch_size)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        self.storage = storage
        if not isinstance(grid, MeshGrid):
            raise ValueError('Explicit MeshGrid required')
        limit = self.prototype_unknown_limit if max_unknowns is None else max_unknowns
        if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, (int, np.integer)) or limit <= 0:
            raise ValueError('max_unknowns must be a positive integer')
        self.max_unknowns = int(limit)
        materials = tuple(materials)
        if (not materials or len(materials) != grid.n_cells or
                any(not isinstance(m, MomentMaterial) for m in materials)):
            raise ValueError('One immutable MomentMaterial per cell required')
        self.n_cells, self.n_groups = grid.n_cells, materials[0].n_groups
        coefficients, boundary_g = _family_tables(family)
        self.n_moments = len(coefficients)
        self.required_orders = tuple(range(2*self.n_moments))
        g, q = self.n_groups, self.n_moments*self.n_groups
        self.shape, self.size = (self.n_cells, self.n_moments, g), self.n_cells*q
        # Before material matrices, local dense patches or global sparse expansion.
        if g < 1:
            raise ValueError('Positive native group count required')
        if self.size > self.max_unknowns:
            raise ValueError(f'Small corrected-native prototype limited to {self.max_unknowns} unknowns')
        if not np.isfinite(condition_limit) or condition_limit <= 1:
            raise ValueError('Invalid condition limit')
        self.condition_limit = condition_limit
        self.grid, self.materials, self.family = grid, materials, family
        mesh = grid.mesh
        _, _, dim, distances, bd, specs = _validate_corrected_mesh(
            grid, np.ones(self.n_cells), np.zeros(self.n_cells), bc)
        if any(s not in ('reflective', 'vacuum') for s in specs):
            raise ValueError('Native corrected boundaries require reflective or vacuum')
        # A single very wide group block can exhaust memory below the global
        # unknown limit. Bound the local SVD/block workspace separately.
        if (q*dim)**2 > 4_000_000:
            raise ValueError('Small corrected-native prototype local matrix limit exceeded')
        self.volumes = _immutable(np.asarray(mesh.area))
        self.C = tuple(_immutable(c) for c in coefficients)
        self.c1, self.c2 = self.C[:2]
        self.B = _immutable(np.kron(boundary_g, np.eye(g)))
        reaction, diffusion, inverse_D, cache = [], [], [], {}
        for v, m in zip(self.volumes, materials):
            if m.n_groups != g or m.provenance.get('orientation_contract') not in ('from-major', 'to-major'):
                raise ValueError('Group/orientation identity differs')
            if not np.array_equal(m.provenance.get('energy_bounds_mev'), materials[0].provenance.get('energy_bounds_mev')):
                raise ValueError('Energy boundary identity differs')
            if any(a.shape != (g,) or not np.all(np.isfinite(a)) for a in (m.total, m.chi, m.nu_sigma_f, m.kappa_fission)):
                raise ValueError('Invalid finite native material vectors')
            if any(k not in m.loss for k in self.required_orders):
                raise ValueError(f'Native losses 0..{self.required_orders[-1]} required')
            if any(m.loss[k].shape != (g,g) or not np.all(np.isfinite(m.loss[k])) for k in self.required_orders):
                raise ValueError('Invalid finite native loss matrices')
            if id(m) not in cache:
                D = np.zeros((q,q))
                for k in range(self.n_moments):
                    s = slice(k*g,(k+1)*g)
                    D[s,s] = self._inverse(m.loss[2*k+1], f'L{2*k+1}')/(4*k+3)
                R = sum(np.kron(c, m.loss[2*k]) for k,c in enumerate(self.C))
                cache[id(m)] = (_immutable(D), _immutable(self._inverse(D, 'D')), _immutable(R))
            D, inv, R = cache[id(m)]
            diffusion.append(D); inverse_D.append(inv); reaction.append(_immutable(v*R))
        if any(not np.all(np.isfinite(r)) for r in reaction):
            raise ValueError('Nonfinite integrated reaction')
        self.diffusion, self.reaction = tuple(diffusion), tuple(reaction)
        if storage == 'deferred':
            from .native_compact import CompactNativeStorage
            self.compact = CompactNativeStorage(self, inverse_D, distances, bd, specs,
                                                int(gradient_chunk_cells), int(batch_size))
            return
        identity = np.eye(q)
        equations = [[] for _ in materials]
        interior, boundary = [], []
        # Each patch unknown is H.ravel(), with H shaped (q, dimension).
        def gradient_equation(blocks):
            return np.stack(blocks, axis=2).reshape(q, q*dim)
        for (i,j,area,_), center, normal, (di,dj) in zip(mesh.faces, mesh.face_centers, mesh.face_normals, distances):
            r = mesh.centroid[j]-mesh.centroid[i]
            for owner, neighbor, vector, n, d in ((i,j,r,normal,dj), (j,i,-r,-normal,di)):
                jump = inverse_D[neighbor]@diffusion[owner]-identity
                A = gradient_equation([vector[k]*identity+d*n[k]*jump for k in range(dim)])
                equations[owner].append((A, {neighbor: identity, owner: -identity}))
            T = area*self._inverse(di*inverse_D[i]+dj*inverse_D[j], 'face resistance')
            interior.append((int(i),int(j),_immutable(T),_immutable(center-mesh.centroid[i]-di*normal),
                             _immutable(center-mesh.centroid[j]+dj*normal)))
        for (i,area,_), center, normal, d, spec in zip(mesh.bfaces,mesh.bface_centers,mesh.bface_normals,bd,specs):
            r = center-mesh.centroid[i]
            B = self.B if spec == 'vacuum' else np.zeros((q,q))
            # A same-material mirrored neighbor has jump=0 and r=2*d*n,
            # hence row normalization gives I*H*n=0. Multiplying by D before
            # the overdetermined fit changes its metric for coupled groups.
            A = gradient_equation([normal[k]*diffusion[i]+r[k]*B for k in range(dim)]
                                  if spec == 'vacuum' else [normal[k]*identity for k in range(dim)])
            equations[i].append((A, {i: -B}))
            if spec == 'vacuum':
                C = diffusion[i]/d
                K = area*self.B@self._inverse(C+self.B, 'vacuum C+B')@C
            else:
                K = np.zeros((q,q))
            boundary.append((int(i),_immutable(K),_immutable(r-d*normal)))
        gradients = []
        for i, patch in enumerate(equations):
            if not patch:
                raise ValueError(f'rank-deficient gradient patch at cell {i}')
            A = np.concatenate([a for a,_ in patch])
            scales = np.linalg.norm(A,axis=1)
            if np.any(scales == 0) or not np.all(np.isfinite(scales)):
                raise ValueError(f'rank-deficient gradient patch at cell {i}')
            A = A/scales[:,None]
            U,s,Vh = np.linalg.svd(A,full_matrices=False)
            if len(s) < q*dim or s[-1] <= max(1e-12,1/condition_limit)*s[0]:
                raise ValueError(f'rank-deficient/ill-conditioned gradient patch at cell {i}')
            inverse = (Vh.T/s)@U.T
            row = {}
            for k, (_,rhs) in enumerate(patch):
                for col, value in rhs.items():
                    contribution = inverse[:,k*q:(k+1)*q]@(value/scales[k*q:(k+1)*q,None])
                    row[col] = row.get(col,0)+contribution.reshape(q,dim,q)
            gradients.append(_immutable(row))
        self.gradient_stencils = tuple(gradients)
        self.faces, self.boundary = tuple(interior), tuple(boundary)
        # Assemble block dictionaries independently of runtime face/reaction sums.
        rows = [{i:r.copy()} for i,r in enumerate(reaction)]
        def trace(i,t):
            row = {col:np.einsum('adk,d->ak',h,t) for col,h in gradients[i].items()}
            row[i] = row.get(i,0)+identity
            return row
        def accumulate(target,source,weight):
            for col,value in source.items():
                target[col] = target.get(col,0)+weight@value
        for i,j,T,ti,tj in interior:
            row = {}
            accumulate(row,trace(i,ti),T); accumulate(row,trace(j,tj),-T)
            accumulate(rows[i],row,identity); accumulate(rows[j],row,-identity)
        for i,K,t in boundary:
            accumulate(rows[i],trace(i,t),K)
        # Each dictionary entry is already one unique q-by-q block. Keep its
        # indices at cell granularity rather than boxing q*q scalar COO triples.
        nblocks = sum(len(row) for row in rows)
        # Include expanded CSR entries when selecting width, since an explicit
        # unknown budget can exceed the default small-prototype bound.
        index_dtype = np.int32 if max(self.size, nblocks*q*q) <= np.iinfo(np.int32).max else np.int64
        block_data = np.empty((nblocks,q,q), dtype=float)
        block_indices = np.empty(nblocks, dtype=index_dtype)
        block_indptr = np.empty(self.n_cells+1, dtype=index_dtype)
        offset = 0
        block_indptr[0] = 0
        for i,row in enumerate(rows):
            for j in sorted(row):
                block_indices[offset] = j
                block_data[offset] = row[j]
                offset += 1
            block_indptr[i+1] = offset
        self.assembled = sparse.bsr_matrix(
            (block_data,block_indices,block_indptr), shape=(self.size,self.size)).tocsr()
        self.assembled.eliminate_zeros()
        if not np.all(np.isfinite(self.assembled.data)):
            raise ValueError('Nonfinite corrected native assembly')

    @property
    def assembled(self):
        if self.storage == 'deferred':
            raise NotImplementedError('Global assembled matrix is unsupported with deferred storage')
        return self._assembled

    @assembled.setter
    def assembled(self, value):
        if self.storage == 'deferred':
            raise NotImplementedError('Global assembled matrix is unsupported with deferred storage')
        self._assembled = value

    def linear_operator(self):
        from scipy.sparse.linalg import LinearOperator
        return LinearOperator((self.size, self.size), matvec=self.action, dtype=float)

    def group_principal_block(self, group):
        if self.storage == 'deferred':
            return self.compact.group_principal_block(group)
        from .native_compact import validate_group
        validate_group(group, self.n_groups)
        indices = np.arange(self.size).reshape(self.shape)[:, :, group].ravel()
        return self.assembled[indices][:, indices].tocsr()

    def iter_full_blocks(self):
        """Yield bounded (cell rows, cell columns, full q-by-q blocks) batches.

        Entries may repeat: consumers must sum them. Deferred only; no global
        matrix is constructed. Intended for a future coarse projection adapter.
        """
        if self.storage != 'deferred':
            raise NotImplementedError('Block streaming requires deferred storage')
        return self.compact.iter_full_blocks()

    def gradients(self,u):
        if self.storage == 'deferred':
            return self.compact.gradients(self._values(u))
        x = self._values(u)
        return np.stack([sum((np.einsum('adk,k->ad',h,x[col]) for col,h in row.items()),
                             np.zeros((self.n_moments*self.n_groups,self.grid.mesh.centroid.shape[1])))
                         for row in self.gradient_stencils])

    def face_currents(self,u):
        """Integrated interior i->j and boundary outward currents, (faces,M,G)."""
        if self.storage == 'deferred':
            return self.compact.face_currents(self._values(u))
        x, H = self._values(u), self.gradients(u)
        q = self.n_moments*self.n_groups
        interior = np.array([T@(x[i]+H[i]@ti-x[j]-H[j]@tj) for i,j,T,ti,tj in self.faces]).reshape(-1,q)
        boundary = np.array([K@(x[i]+H[i]@t) for i,K,t in self.boundary]).reshape(-1,q)
        return interior.reshape(-1,self.n_moments,self.n_groups), boundary.reshape(-1,self.n_moments,self.n_groups)

    def action(self,u):
        if self.storage == 'deferred':
            return self.compact.action(self._values(u)).reshape(np.shape(u))
        x = self._values(u)
        out = np.stack([r@v for r,v in zip(self.reaction,x)])
        interior,boundary = self.face_currents(u)
        for (i,j,*_),current in zip(self.faces,interior.reshape(-1,self.n_moments*self.n_groups)):
            out[i] += current; out[j] -= current
        for (i,*_),current in zip(self.boundary,boundary.reshape(-1,self.n_moments*self.n_groups)):
            out[i] += current
        return out.reshape(np.shape(u))

    apply = action

    def fission_action(self,u):
        phi = self.scalar_flux(u)
        result = np.stack([v*self.c1[:,0,None]*m.chi[None,:]*np.dot(m.nu_sigma_f,p)
                           for v,m,p in zip(self.volumes,self.materials,phi)])
        return result.reshape(np.shape(u))
