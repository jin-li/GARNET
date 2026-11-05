"""Fixed linear nonorthogonal order-1 SPN/SDPN finite-volume closure."""
from __future__ import annotations

import numpy as np

from .backend import asnumpy
from .corrected_mesh import CorrectedMeshGroupOperator, _Gather, _sparse
from .spn import _SDPN_C, SDPNGroupOperator
from .mesh import MeshGroupOperator
from .stencil import face_alpha


class CorrectedMeshMomentOperator:
    """Coupled gradient reconstruction and homogeneous Marshak trace elimination.

    Unknowns have shape ``(2, n_cells, 1, 1)``. Geometry and scalar material
    coefficients define a fixed linear stencil; no angular transport data are
    inferred. This stencil is generally nonsymmetric and requires GMRES.
    """

    def __init__(self, xp, grid, D1, sigma_t, removal, order, bc="zero-flux",
                 active=None, mask_bc="vacuum", op_cls=None, coeffs=None,
                 boundary_g=None, theta=None, symmetrize=False,
                 hybrid_mask=None, hybrid_mask_bc="reflective",
                 hybrid_confine=False, moment_df_model=None, moment_face_df=None,
                 moment_boundary_df=None):
        if order != 1:
            raise ValueError("corrected moment reconstruction supports order 1 only")
        if active is not None or hybrid_mask is not None or hybrid_confine:
            raise ValueError("corrected moment active/hybrid masks are unsupported")
        if symmetrize:
            raise ValueError("corrected moment reconstruction does not support symmetrize")
        if any(v is not None for v in (moment_df_model, moment_face_df, moment_boundary_df)):
            raise ValueError("corrected moment discontinuity factors are unsupported")
        self._tpfa = SDPNGroupOperator(
            xp, grid, D1, sigma_t, removal, order, bc=bc,
            op_cls=MeshGroupOperator, coeffs=coeffs, boundary_g=boundary_g, theta=theta)
        self.tpfa_apply = self._tpfa.apply
        self.tpfa_inv_diag = self._tpfa.inv_diag
        self.xp, self.grid = xp, grid
        self.M, self.order, self.symmetric = 2, 1, False
        self.hybrid_mask, self.hybrid_confine = None, False
        mesh, n = grid.mesh, grid.n_cells
        st = np.asarray(asnumpy(sigma_t), dtype=float).reshape(-1)
        rem = np.asarray(asnumpy(removal), dtype=float).reshape(-1)
        if st.size != n or not np.all(np.isfinite(st)) or np.any(st <= 0):
            raise ValueError("sigma_t requires one finite positive value per cell")
        D = np.stack([np.asarray(asnumpy(D1), dtype=float).reshape(-1), 1/(7*st)])
        # Reuse the scalar geometry/material validation, with reconstruction
        # explicitly bypassed: the coupled equations below own all gradients.
        CorrectedMeshGroupOperator(np, grid, D[0], rem, bc=bc,
                                   reconstruction_policy="supplied",
                                   gradient_stencils=[{} for _ in range(n)])
        specs = grid.classify_boundary(bc)
        c = np.asarray((coeffs if coeffs is not None else _SDPN_C)[order], dtype=float)
        self.src_weights = c[0, :, 0].tolist()
        self.phi0_weights = c[0, 0, :].tolist()
        self.time_weights = c.sum(axis=0).tolist()
        reaction = c[0, :, :, None]*rem + c[1, :, :, None]*st
        if theta is not None:
            th = np.asarray(asnumpy(theta), dtype=float)
            if th.size not in (1, n) or not np.all(np.isfinite(th)):
                raise ValueError("theta must be finite and scalar or cell-valued")
            reaction += c.sum(axis=0)[:, :, None]*th.reshape(-1)
        g = None if boundary_g is None else np.asarray(boundary_g, dtype=float)
        if g is not None and (g.shape != (2, 2) or not np.all(np.isfinite(g))):
            raise ValueError("boundary_g must be a finite 2 by 2 matrix")
        dim = mesh.centroid.shape[1]
        distances = np.asarray([(np.dot(center-mesh.centroid[i], normal),
                                 np.dot(mesh.centroid[j]-center, normal))
                                for (i, j, _, _), center, normal in
                                zip(mesh.faces, mesh.face_centers, mesh.face_normals)]).reshape(-1, 2)
        bd = np.asarray([np.dot(center-mesh.centroid[i], normal)
                         for (i, _, _), center, normal in
                         zip(mesh.bfaces, mesh.bface_centers, mesh.bface_normals)])
        equations = [[] for _ in range(n)]
        for (i, j, _, _), normal, (di, dj) in zip(mesh.faces, mesh.face_normals, distances):
            r = mesh.centroid[j]-mesh.centroid[i]
            for m in range(2):
                for owner, neighbor, vector in (
                        (i, j, r+(D[m, i]/D[m, j]-1)*dj*normal),
                        (j, i, -r-(D[m, j]/D[m, i]-1)*di*normal)):
                    lhs = np.zeros((2, dim)); lhs[m] = vector
                    equations[owner].append((lhs.ravel(), {m*n+neighbor: 1., m*n+owner: -1.}))
        boundary_matrices = []
        for (i, area, _), center, normal, d, spec in zip(
                mesh.bfaces, mesh.bface_centers, mesh.bface_normals, bd, specs):
            r = center-mesh.centroid[i]
            alpha = face_alpha(spec)
            G = (g if spec == "vacuum" and g is not None else
                 np.diag([alpha, alpha]))
            if np.isinf(alpha):
                K = area*np.diag(D[:, i]/d)
            else:
                A = np.diag(D[:, i]/d)
                K = area*G@np.linalg.solve(A+G, A)
            boundary_matrices.append(K)
            for m in range(2):
                lhs = np.zeros((2, dim))
                if np.isinf(alpha):
                    lhs[m] = r
                    rhs = {m*n+i: -1.}
                else:
                    lhs[:] = G[m, :, None]*r
                    lhs[m] += D[m, i]*normal
                    rhs = {k*n+i: -G[m, k] for k in range(2)}
                equations[i].append((lhs.ravel(), rhs))
        gradients = []
        for i, patch in enumerate(equations):
            design = np.asarray([lhs for lhs, _ in patch]).reshape(-1, 2*dim)
            scales = np.linalg.norm(design, axis=1)
            if not len(scales) or np.any(scales == 0):
                raise ValueError(f"rank-deficient coupled gradient patch at cell {i}")
            design /= scales[:, None]
            sv = np.linalg.svd(design, compute_uv=False)
            if len(sv) < 2*dim or sv[-1] <= 1e-12*sv[0]:
                raise ValueError(f"rank-deficient coupled gradient patch at cell {i}")
            inverse = np.linalg.pinv(design, rcond=1e-12)
            row = {}
            for k, (_, rhs) in enumerate(patch):
                for col, value in rhs.items():
                    row[col] = row.get(col, np.zeros((2, dim))) + inverse[:, k].reshape(2, dim)*value/scales[k]
            gradients.append(row)
        self.gradient_stencils = gradients
        rows = [{} for _ in range(2*n)]
        for m in range(2):
            for k in range(2):
                for i in range(n):
                    rows[m*n+i][k*n+i] = reaction[m, k, i]*mesh.area[i]

        def trace(row, owner, moment, tangent, weight):
            col = moment*n+owner
            row[col] = row.get(col, 0.) + weight
            for col, vector in gradients[owner].items():
                row[col] = row.get(col, 0.) + weight*np.dot(vector[moment], tangent)

        def accumulate(target, source, sign=1.):
            for col, value in source.items():
                target[col] = target.get(col, 0.) + sign*value

        interior = [[] for _ in range(2)]
        boundary = [[] for _ in range(2)]
        for (i, j, area, _), center, normal, (di, dj) in zip(
                mesh.faces, mesh.face_centers, mesh.face_normals, distances):
            for m in range(2):
                row = {}
                w = area/(di/D[m, i]+dj/D[m, j])
                trace(row, i, m, center-mesh.centroid[i]-di*normal, w)
                trace(row, j, m, center-mesh.centroid[j]+dj*normal, -w)
                interior[m].append(row)
                accumulate(rows[m*n+i], row)
                accumulate(rows[m*n+j], row, -1.)
        for (i, _, _), center, normal, d, K in zip(
                mesh.bfaces, mesh.bface_centers, mesh.bface_normals, bd, boundary_matrices):
            for m in range(2):
                row = {}
                for k in range(2):
                    trace(row, i, k, center-mesh.centroid[i]-d*normal, K[m, k])
                boundary[m].append(row)
                accumulate(rows[m*n+i], row)
        dtype = np.result_type(asnumpy(D1).dtype, np.float32)
        self.host_matrix = _sparse(rows, 2*n)
        self._apply = _Gather(xp, self.host_matrix, dtype)
        self._interior = _Gather(xp, _sparse(interior[0]+interior[1], 2*n), dtype)
        self._boundary = _Gather(xp, _sparse(boundary[0]+boundary[1], 2*n), dtype)
        self.diag = xp.asarray(self.host_matrix.diagonal(), dtype=dtype).reshape((2,)+grid.shape)
        self.inv_diag = 1/self.diag
        self.rhs_weight = xp.asarray(grid.integration_weights, dtype=dtype)

    def apply(self, state):
        return self._apply(state).reshape((2,)+self.grid.shape)

    def face_currents(self, state):
        """Return integrated i-to-j and outward currents, each moment first."""
        return (self._interior(state).reshape(2, len(self.grid.mesh.faces)),
                self._boundary(state).reshape(2, len(self.grid.mesh.bfaces)))
