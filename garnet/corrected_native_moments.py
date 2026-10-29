"""Small CPU full-matrix corrected FV experiment; no physical admission.

Native cell-major states are (cells, 2, groups). Full loss matrices remain on
one left-hand side. This deliberately bounded prototype is not reactor scalable.
"""
import numpy as np
from scipy import sparse
from .mesh import MeshGrid
from .corrected_mesh import _validate_corrected_mesh
from .moment_materials import MomentMaterial, _immutable
from .coupled_moments import CoupledMomentOperator
from .spn import _SPN_C, _SDPN_C, _SPN_G, _SDPN_G


class CorrectedNativeMomentOperator:
    """Fixed row-normalized coupled patches, signed distances, full currents.

    ``volumes`` is the integrated-volume accessor used by the native CPU driver.
    Only homogeneous vacuum and reflective boundaries are supported. Local SVD
    rejects rank loss or condition above condition_limit; no regularization,
    positivity clipping, symmetric-part admission, or reaction inversion occurs.
    """
    physical_admission = False
    prototype_unknown_limit = 20_000
    _inverse = CoupledMomentOperator._inverse
    _values = CoupledMomentOperator._values
    scalar_flux = CoupledMomentOperator.scalar_flux

    def __init__(self, grid, materials, *, family='sp3', bc=None, condition_limit=1e12):
        if not isinstance(grid, MeshGrid):
            raise ValueError('Explicit MeshGrid required')
        materials = tuple(materials)
        if (not materials or len(materials) != grid.n_cells or
                any(not isinstance(m, MomentMaterial) for m in materials)):
            raise ValueError('One immutable MomentMaterial per cell required')
        self.n_cells, self.n_groups = grid.n_cells, materials[0].n_groups
        g, q = self.n_groups, 2*self.n_groups
        self.shape, self.size = (self.n_cells, 2, g), self.n_cells*q
        # Before material matrices, local dense patches or global sparse expansion.
        if g < 1:
            raise ValueError('Positive native group count required')
        if self.size > self.prototype_unknown_limit:
            raise ValueError('Small corrected-native prototype limited to 20000 unknowns')
        if family not in ('sp3', 'sdp1'):
            raise ValueError('Use sp3 or sdp1')
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
        c1, c2 = np.asarray((_SPN_C if family == 'sp3' else _SDPN_C)[1])
        self.c1, self.c2 = _immutable(c1), _immutable(c2)
        self.B = _immutable(np.kron((_SPN_G if family == 'sp3' else _SDPN_G)[1], np.eye(g)))
        reaction, diffusion, inverse_D, cache = [], [], [], {}
        for v, m in zip(self.volumes, materials):
            if m.n_groups != g or m.provenance.get('orientation_contract') not in ('from-major', 'to-major'):
                raise ValueError('Group/orientation identity differs')
            if not np.array_equal(m.provenance.get('energy_bounds_mev'), materials[0].provenance.get('energy_bounds_mev')):
                raise ValueError('Energy boundary identity differs')
            if any(a.shape != (g,) or not np.all(np.isfinite(a)) for a in (m.total, m.chi, m.nu_sigma_f, m.kappa_fission)):
                raise ValueError('Invalid finite native material vectors')
            if any(k not in m.loss for k in range(4)):
                raise ValueError('Native losses 0..3 required')
            if any(m.loss[k].shape != (g,g) or not np.all(np.isfinite(m.loss[k])) for k in range(4)):
                raise ValueError('Invalid finite native loss matrices')
            if id(m) not in cache:
                D = np.zeros((q,q))
                D[:g,:g] = self._inverse(m.loss[1], 'L1')/3
                D[g:,g:] = self._inverse(m.loss[3], 'L3')/7
                R = np.kron(c1, m.loss[0])+np.kron(c2, m.loss[2])
                cache[id(m)] = (_immutable(D), _immutable(self._inverse(D, 'D')), _immutable(R))
            D, inv, R = cache[id(m)]
            diffusion.append(D); inverse_D.append(inv); reaction.append(_immutable(v*R))
        if any(not np.all(np.isfinite(r)) for r in reaction):
            raise ValueError('Nonfinite integrated reaction')
        self.diffusion, self.reaction = tuple(diffusion), tuple(reaction)
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
            A = gradient_equation([normal[k]*diffusion[i]+r[k]*B for k in range(dim)])
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
        block_data = np.empty((nblocks,q,q), dtype=float)
        block_indices = np.empty(nblocks, dtype=np.int32)
        block_indptr = np.empty(self.n_cells+1, dtype=np.int32)
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

    def gradients(self,u):
        x = self._values(u)
        return np.stack([sum((np.einsum('adk,k->ad',h,x[col]) for col,h in row.items()),
                             np.zeros((2*self.n_groups,self.grid.mesh.centroid.shape[1])))
                         for row in self.gradient_stencils])

    def face_currents(self,u):
        """Integrated interior i->j and boundary outward currents, (faces,2,G)."""
        x, H = self._values(u), self.gradients(u)
        q = 2*self.n_groups
        interior = np.array([T@(x[i]+H[i]@ti-x[j]-H[j]@tj) for i,j,T,ti,tj in self.faces]).reshape(-1,q)
        boundary = np.array([K@(x[i]+H[i]@t) for i,K,t in self.boundary]).reshape(-1,q)
        return interior.reshape(-1,2,self.n_groups), boundary.reshape(-1,2,self.n_groups)

    def action(self,u):
        x = self._values(u)
        out = np.stack([r@v for r,v in zip(self.reaction,x)])
        interior,boundary = self.face_currents(u)
        for (i,j,*_),current in zip(self.faces,interior.reshape(-1,2*self.n_groups)):
            out[i] += current; out[j] -= current
        for (i,*_),current in zip(self.boundary,boundary.reshape(-1,2*self.n_groups)):
            out[i] += current
        return out.reshape(np.shape(u))

    apply = action

    def fission_action(self,u):
        phi = self.scalar_flux(u)
        result = np.stack([v*self.c1[:,0,None]*m.chi[None,:]*np.dot(m.nu_sigma_f,p)
                           for v,m,p in zip(self.volumes,self.materials,phi)])
        return result.reshape(np.shape(u))
