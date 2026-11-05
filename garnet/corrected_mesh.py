"""Opt-in linear nonorthogonal scalar finite-volume operator.

See docs/development/corrected-fv.md for reconstruction assumptions and limits.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from .backend import asnumpy
from .mesh import _mesh_df, MeshGroupOperator
from .stencil import face_alpha


def _sparse(rows, n):
    data, indices, indptr = [], [], [0]
    for row in rows:
        for col, value in sorted(row.items()):
            if value != 0:
                indices.append(col)
                data.append(value)
        indptr.append(len(data))
    return csr_matrix((data, indices, indptr), shape=(len(rows), n))


class _Gather:
    """Backend-independent sparse row gather, with no device scatter."""
    def __init__(self, xp, matrix, dtype):
        self.xp = xp
        self.nrows = matrix.shape[0]
        width = max(1, int(np.diff(matrix.indptr).max(initial=0)))
        indices = np.zeros((self.nrows, width), dtype=np.int32)
        values = np.zeros((self.nrows, width), dtype=dtype)
        for i in range(self.nrows):
            a, b = matrix.indptr[i:i + 2]
            indices[i, :b-a] = matrix.indices[a:b]
            values[i, :b-a] = matrix.data[a:b]
        self.indices, self.values = xp.asarray(indices), xp.asarray(values)

    def __call__(self, u):
        return self.xp.sum(self.values * u.reshape(-1)[self.indices], axis=1)


class CorrectedMeshGroupOperator:
    """Fixed linear conservative scalar FV stencil; use GMRES, not CG.

    ``gradient_stencils`` optionally supplies one dictionary per cell, mapping
    cell indices to dimension-vector coefficients of its gradient. This hook
    replaces scalar boundary reconstruction entirely and must encode the
    intended homogeneous boundary law. It is intended for future coupled
    boundary integration, not an assertion of moment compatibility.
    """
    def __init__(self, xp, grid, D, removal, bc=None, active=None,
                 mask_bc="vacuum", symmetric=True, face_df=None, boundary_df=None,
                 *, reconstruction_policy="scalar_boundary", gradient_stencils=None):
        if active is not None:
            raise ValueError("corrected mesh active masks are unsupported")
        mesh, n = grid.mesh, grid.n_cells
        self.xp, self.grid, self.symmetric = xp, grid, False
        diffusion = np.asarray(asnumpy(D), dtype=float).reshape(-1)
        removal = np.asarray(asnumpy(removal), dtype=float).reshape(-1)
        if (diffusion.size != n or removal.size != n or
                not np.all(np.isfinite(diffusion)) or np.any(diffusion <= 0) or
                not np.all(np.isfinite(removal))):
            raise ValueError("diffusion/removal require one finite value per cell; diffusion must be positive")
        for value, shape, name in ((face_df, (len(mesh.faces), 2), "face_df"),
                                   (boundary_df, (len(mesh.bfaces),), "boundary_df")):
            if np.any(_mesh_df(value, shape, name) != 1):
                raise ValueError("corrected mesh discontinuity factors are not implemented")
        if (mesh.centroid.ndim != 2 or mesh.centroid.shape[0] != n or
                mesh.centroid.shape[1] not in (2, 3) or not np.all(np.isfinite(mesh.centroid))):
            raise ValueError("corrected mesh requires finite 2D or 3D cell centroids")
        dim = mesh.centroid.shape[1]
        for name, count in (("face_centers", len(mesh.faces)), ("face_normals", len(mesh.faces)),
                            ("bface_centers", len(mesh.bfaces)), ("bface_normals", len(mesh.bfaces))):
            arr = np.asarray(getattr(mesh, name), dtype=float)
            if arr.shape != (count, dim) or not np.all(np.isfinite(arr)):
                raise ValueError(f"corrected mesh requires finite {name} with shape {(count, dim)}")
        for normals in (mesh.face_normals, mesh.bface_normals):
            if not np.allclose(np.linalg.norm(normals, axis=1), 1, rtol=1e-10, atol=1e-12):
                raise ValueError("corrected mesh requires unit normals")
        # Use signed geometric projections, rather than trusting legacy distances.
        distances = []
        for (i, j, _, _), center, normal in zip(mesh.faces, mesh.face_centers, mesh.face_normals):
            distances.append((np.dot(center-mesh.centroid[i], normal),
                              np.dot(mesh.centroid[j]-center, normal)))
        distances = np.asarray(distances).reshape(-1, 2)
        bd = np.array([np.dot(c-mesh.centroid[i], normal) for (i, _, _), c, normal
                       in zip(mesh.bfaces, mesh.bface_centers, mesh.bface_normals)])
        if (not np.all(np.isfinite(distances)) or not np.all(np.isfinite(bd)) or
                np.any(distances <= 0) or np.any(bd <= 0)):
            raise ValueError("corrected mesh requires positive signed normal distances")
        specs = grid.classify_boundary(bc)
        self.reconstruction_policy = reconstruction_policy
        if gradient_stencils is None:
            if reconstruction_policy != "scalar_boundary":
                raise ValueError("custom reconstruction policy requires gradient_stencils")
            gradients = self._reconstruct(mesh, diffusion, specs, distances)
        else:
            if reconstruction_policy != "supplied":
                raise ValueError("gradient_stencils require reconstruction_policy='supplied'")
            gradients = gradient_stencils
        if len(gradients) != n:
            raise ValueError("gradient_stencils require one dictionary per cell")
        self.gradient_stencils = []
        for row in gradients:
            checked = {}
            for col, vector in row.items():
                vec = np.asarray(vector, dtype=float)
                if int(col) != col or not 0 <= col < n or vec.shape != (dim,) or not np.all(np.isfinite(vec)):
                    raise ValueError("invalid gradient stencil entry")
                checked[int(col)] = vec.copy()
            self.gradient_stencils.append(checked)
        rows = [{i: removal[i] * mesh.area[i]} for i in range(n)]
        interior, boundary = [], []

        def trace(row, owner, tangent, scale):
            row[owner] = row.get(owner, 0.) + scale
            for col, vec in self.gradient_stencils[owner].items():
                row[col] = row.get(col, 0.) + scale * np.dot(vec, tangent)

        for (i, j, area, _), center, normal, (di, dj) in zip(
                mesh.faces, mesh.face_centers, mesh.face_normals, distances):
            row = {}
            weight = area / (di / diffusion[i] + dj / diffusion[j])
            trace(row, i, center-mesh.centroid[i]-di*normal, weight)
            trace(row, j, center-mesh.centroid[j]+dj*normal, -weight)
            interior.append(row)
            for col, value in row.items():
                rows[i][col] = rows[i].get(col, 0.) + value
                rows[j][col] = rows[j].get(col, 0.) - value
        for (i, area, _), center, normal, distance, spec in zip(
                mesh.bfaces, mesh.bface_centers, mesh.bface_normals, bd, specs):
            alpha = face_alpha(spec)
            weight = (area*diffusion[i]/distance if np.isinf(alpha) else
                      area*alpha*diffusion[i]/(diffusion[i]+alpha*distance))
            row = {}
            trace(row, i, center-mesh.centroid[i]-distance*normal, weight)
            boundary.append(row)
            for col, value in row.items():
                rows[i][col] = rows[i].get(col, 0.) + value
        self.host_matrix = _sparse(rows, n)
        dtype = np.result_type(asnumpy(D).dtype, np.float32)
        self._apply = _Gather(xp, self.host_matrix, dtype)
        self._interior = _Gather(xp, _sparse(interior, n), dtype)
        self._boundary = _Gather(xp, _sparse(boundary, n), dtype)
        self.diag = xp.asarray(self.host_matrix.diagonal(), dtype=dtype).reshape(grid.shape)
        self.inv_diag = 1 / self.diag
        self.rhs_weight = xp.asarray(grid.integration_weights, dtype=dtype)
        # A fixed TPFA basis preconditions the corrected nonsymmetric problem.
        self._tpfa = MeshGroupOperator(xp, grid, D, removal.reshape(grid.shape), bc=bc)
        self.tpfa_apply = self._tpfa.apply
        self.tpfa_inv_diag = self._tpfa.inv_diag

    @staticmethod
    def _reconstruct(mesh, D, specs, distances):
        n, dim = mesh.centroid.shape
        equations = [[] for _ in range(n)]
        for (i, j, _, _), normal, (di, dj) in zip(mesh.faces, mesh.face_normals, distances):
            r = mesh.centroid[j]-mesh.centroid[i]
            equations[i].append((r+(D[i]/D[j]-1)*dj*normal, {j: 1., i: -1.}))
            equations[j].append((-r-(D[j]/D[i]-1)*di*normal, {i: 1., j: -1.}))
        for (i, _, _), center, normal, spec in zip(
                mesh.bfaces, mesh.bface_centers, mesh.bface_normals, specs):
            alpha = face_alpha(spec)
            r = center-mesh.centroid[i]
            if np.isinf(alpha):
                equations[i].append((r, {i: -1.}))
            else:
                equations[i].append((D[i]*normal+alpha*r, {i: -alpha}))
        gradients = []
        for i, equations_i in enumerate(equations):
            if not equations_i:
                raise ValueError(f"rank-deficient gradient patch at cell {i}")
            vectors = np.asarray([e[0] for e in equations_i])
            scales = np.linalg.norm(vectors, axis=1)
            if np.any(scales == 0):
                raise ValueError(f"zero reconstruction equation at cell {i}")
            design = vectors/scales[:, None]
            singular = np.linalg.svd(design, compute_uv=False)
            if len(singular) < dim or singular[-1] <= 1e-12*singular[0]:
                raise ValueError(f"rank-deficient gradient patch at cell {i}")
            inverse = np.linalg.pinv(design, rcond=1e-12)
            row = {}
            for k, (_, rhs) in enumerate(equations_i):
                for col, value in rhs.items():
                    row[col] = row.get(col, np.zeros(dim)) + inverse[:, k]*value/scales[k]
            gradients.append(row)
        return gradients

    def apply(self, phi):
        return self._apply(phi).reshape(self.grid.shape)

    def face_currents(self, phi):
        """Integrated currents: interior i->j and boundary outward, in mesh order."""
        return self._interior(phi), self._boundary(phi)
