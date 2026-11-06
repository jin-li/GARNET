"""Conservative 2D/3D upwind streaming plus native total collision; no solve."""
from dataclasses import dataclass, field, InitVar

import numpy as np
from scipy.sparse import coo_matrix

from .mesh import Mesh
from .native_angular_source import NativeAngularSource, _budget, _integer, _readonly


@dataclass(frozen=True)
class NativeMeshStreaming:
    """Frozen geometry bound to one NativeAngularSource's angular/energy layout.

    Equation arrays are cell-integrated [cell, angle, group]. Physical direction
    components are used unchanged in the mesh dimension. Geometry must
    explicitly supply face centers and outward/left-to-right unit normals.
    """
    mesh: InitVar[Mesh]
    source: NativeAngularSource
    max_bytes: int = field(default=256*1024**2, repr=False, compare=False)
    area: np.ndarray = field(init=False, repr=False, compare=False)
    centroid: np.ndarray = field(init=False, repr=False, compare=False)
    owners: np.ndarray = field(init=False, repr=False, compare=False)
    measures: np.ndarray = field(init=False, repr=False, compare=False)
    normals: np.ndarray = field(init=False, repr=False, compare=False)
    centers: np.ndarray = field(init=False, repr=False, compare=False)
    boundary_owners: np.ndarray = field(init=False, repr=False, compare=False)
    boundary_measures: np.ndarray = field(init=False, repr=False, compare=False)
    boundary_normals: np.ndarray = field(init=False, repr=False, compare=False)
    boundary_centers: np.ndarray = field(init=False, repr=False, compare=False)
    total: np.ndarray = field(init=False, repr=False, compare=False)
    dimension: int = field(init=False)

    def __post_init__(self, mesh):
        if not isinstance(mesh, Mesh) or not isinstance(self.source, NativeAngularSource):
            raise ValueError('Mesh and NativeAngularSource required')
        nc, nf, nb = mesh.n_cells, len(mesh.faces), len(mesh.bfaces)
        if nc == 0 or nc != self.source.cell_material_ids.size:
            raise ValueError('mesh/source cell count mismatch')
        ng = self.source.n_groups
        # Inspect shape without constructing a numeric geometry copy. Reject a
        # malformed dimension before dimension-dependent allocation estimates.
        shape = getattr(mesh.centroid, 'shape', None)
        if shape is None:
            try:
                shape = (len(mesh.centroid), len(mesh.centroid[0]))
            except (TypeError, IndexError):
                shape = ()
        if len(shape) != 2 or shape[0] != nc or shape[1] not in (2, 3):
            raise ValueError('cell centroids must have shape (n_cells, 2 or 3)')
        dim = shape[1]
        if dim == 3 and (self.quadrature.n_polar < 2 or self.quadrature.n_azimuth < 4):
            raise ValueError('3D streaming requires n_polar >= 2 and n_azimuth >= 4')
        # Guard before geometry/total copies. Includes parser and closure workspace.
        _budget(8*((8+2*dim)*nc + (16+4*dim)*nf + (12+4*dim)*nb
                   + 3*len(self.source.materials)*ng), self.max_bytes)

        def geometry(value, shape, name):
            array = np.asarray(value)
            if array.shape != shape or array.dtype.kind not in 'fiu' or not np.all(np.isfinite(array)):
                raise ValueError(f'{name} must be finite real shape {shape}')
            return _readonly(array)

        area = geometry(mesh.area, (nc,), 'cell measures')
        centroid = geometry(mesh.centroid, (nc, dim), 'cell centroids')
        if np.any(area <= 0):
            raise ValueError('cell measures must be positive')
        normals = geometry(mesh.face_normals, (nf, dim), 'interior normals')
        centers = geometry(mesh.face_centers, (nf, dim), 'interior centers')
        bnormals = geometry(mesh.bface_normals, (nb, dim), 'boundary normals')
        bcenters = geometry(mesh.bface_centers, (nb, dim), 'boundary centers')

        def parse(entries, interior):
            owners, measures = [], []
            for entry in entries:
                if len(entry) != (4 if interior else 3):
                    raise ValueError('invalid face record')
                ids = entry[:2] if interior else entry[:1]
                for owner in ids:
                    _integer(owner, 'face owner')
                    if owner >= nc:
                        raise ValueError('face owner outside mesh')
                if interior and ids[0] == ids[1]:
                    raise ValueError('interior face owners must differ')
                measure, distance = entry[-2:]
                if not np.isfinite(measure) or measure <= 0 or not np.isfinite(distance) or distance <= 0:
                    raise ValueError('face measures and distances must be finite positive')
                owners.append(ids)
                measures.append(measure)
            width = 2 if interior else 1
            return _readonly(np.asarray(owners).reshape(-1, width), np.int64), _readonly(measures)

        owners, measures = parse(mesh.faces, True)
        bowners, bmeasures = parse(mesh.bfaces, False)
        bowners = _readonly(bowners[:, 0], np.int64)
        closure = np.zeros((nc, dim))
        scale = np.zeros(nc)
        for ids, measure, normal, center in zip(owners, measures, normals, centers):
            left, right = ids
            if not np.isclose(np.linalg.norm(normal), 1, atol=2e-13, rtol=0):
                raise ValueError('interior normal must be unit')
            if np.dot(center-centroid[left], normal) <= 0 or np.dot(centroid[right]-center, normal) <= 0:
                raise ValueError('interior normal must point left-to-right across face center')
            closure[left] += measure*normal
            closure[right] -= measure*normal
            scale[left] += measure
            scale[right] += measure
        for owner, measure, normal, center in zip(bowners, bmeasures, bnormals, bcenters):
            if not np.isclose(np.linalg.norm(normal), 1, atol=2e-13, rtol=0):
                raise ValueError('boundary normal must be unit')
            if np.dot(center-centroid[owner], normal) <= 0:
                raise ValueError('boundary normal must point outward across face center')
            closure[owner] += measure*normal
            scale[owner] += measure
        if np.any(scale <= 0) or np.any(np.linalg.norm(closure, axis=1) > 2e-12*scale):
            raise ValueError('cell face geometry fails sum(area*normal)=0 closure')
        totals = []
        for material in self.source.materials:
            value = np.asarray(material.total)
            if (value.shape != (ng,) or value.dtype.kind not in 'fiu'
                    or not np.all(np.isfinite(value)) or np.any(value < 0)):
                raise ValueError('native total must be finite nonnegative shape (n_groups,)')
            totals.append(value)
        for name, value in (('area', area), ('centroid', centroid), ('owners', owners),
                            ('measures', measures), ('normals', normals), ('centers', centers),
                            ('boundary_owners', bowners), ('boundary_measures', bmeasures),
                            ('boundary_normals', bnormals), ('boundary_centers', bcenters),
                            ('total', _readonly(np.stack(totals)))):
            object.__setattr__(self, name, value)
        object.__setattr__(self, 'dimension', dim)

    @property
    def quadrature(self):
        return self.source.quadrature

    @property
    def shape(self):
        return (len(self.area), self.quadrature.n_angles, self.source.n_groups)

    def _guard(self):
        nc, na, ng = self.shape
        a = (self.quadrature.lmax+1)**2
        _budget(8*(6*nc*na*ng + nc*a*ng + na*a + 6*na*ng +
                   2*len(self.boundary_owners)*na*ng), self.max_bytes)

    def _flux(self, value):
        self._guard()
        array = np.asarray(value)
        if array.shape != self.shape or array.dtype.kind not in 'fiu':
            raise ValueError(f'angular flux must have real shape {self.shape}')
        array = np.asarray(array, dtype=float)
        return self._finite(array)

    @staticmethod
    def _finite(array):
        if not np.all(np.isfinite(array)):
            raise ValueError('nonfinite input or output (possible overflow)')
        return array

    def _incoming(self, value):
        if value is None:
            return None
        array = np.asarray(value)
        expected = (len(self.boundary_owners), self.shape[1], self.shape[2])
        if array.shape != expected or array.dtype.kind not in 'fiu':
            raise ValueError(f'incoming boundary flux must have real shape {expected}')
        # Outgoing and tangential entries are never consumed, including NaNs.
        return array

    def action(self, angular_flux):
        """L psi = V Sigma_t psi + interior upwind flux + boundary outflow."""
        flux = self._flux(angular_flux)
        result = np.empty(self.shape)
        directions = self.quadrature.directions[:, :self.dimension]
        for cell, material in enumerate(self.source.cell_material_ids):
            result[cell] = self.area[cell]*self.total[material]*flux[cell]
        for (left, right), area, normal in zip(self.owners, self.measures, self.normals):
            speed = directions @ normal
            face_flux = area*speed[:, None]*np.where(speed[:, None] >= 0, flux[left], flux[right])
            result[left] += face_flux
            result[right] -= face_flux
        for owner, area, normal in zip(self.boundary_owners, self.boundary_measures, self.boundary_normals):
            result[owner] += area*np.maximum(directions @ normal, 0)[:, None]*flux[owner]
        return self._finite(result)

    def boundary_rhs(self, incoming=None):
        """Positive inflow forcing b=-A min(Omega dot n,0) psi_in; None=vacuum."""
        self._guard()
        incoming = self._incoming(incoming)
        result = np.zeros(self.shape)
        if incoming is not None:
            for face, (owner, area, normal) in enumerate(zip(
                    self.boundary_owners, self.boundary_measures, self.boundary_normals)):
                speed = self.quadrature.directions[:, :self.dimension] @ normal
                indices = np.flatnonzero(speed < 0)
                value = np.asarray(incoming[face, indices], dtype=float)
                self._finite(value)
                result[owner, indices] -= area*speed[indices, None]*value
        return self._finite(result)

    def residual(self, angular_flux, incoming=None, *, include_scattering=False):
        """L psi - b, optionally minus V*q_native(psi), with V applied once."""
        result = self.action(angular_flux) - self.boundary_rhs(incoming)
        if include_scattering:
            result -= self.area[:, None, None]*self.source.scatter(angular_flux)
        return self._finite(result)

    def assembled(self, angle, group):
        """Independent CSR L[angle,group], built one ordinate/group at a time."""
        angle = _integer(angle, 'angle')
        group = _integer(group, 'group')
        if angle >= self.shape[1] or group >= self.shape[2]:
            raise ValueError('angle or group outside layout')
        nc, nf, nb = len(self.area), len(self.owners), len(self.boundary_owners)
        _budget(128*(nc+4*nf+nb+1), self.max_bytes)
        omega = self.quadrature.directions[angle]
        # One angle/group only. Separate multiply/add ufuncs retain the scalar
        # arithmetic order; dot/einsum can change rounding and upstream choice.
        with np.errstate(over='ignore', invalid='ignore'):
            speed = omega[0]*self.normals[:, 0] + omega[1]*self.normals[:, 1]
            boundary_speed = (omega[0]*self.boundary_normals[:, 0]
                              + omega[1]*self.boundary_normals[:, 1])
            if self.dimension == 3:
                speed = speed + omega[2]*self.normals[:, 2]
                boundary_speed = boundary_speed + omega[2]*self.boundary_normals[:, 2]
            coefficient = self.measures*speed
            boundary = self.boundary_measures*boundary_speed
        self._finite(coefficient)
        # Reject even unused negative boundary overflow, which the scalar
        # builder silently omitted. Finite coefficients keep identical behavior.
        self._finite(boundary)
        outgoing = np.flatnonzero(boundary > 0)
        end = nc + 2*nf
        count = end + len(outgoing)
        rows = np.empty(count, dtype=np.int64)
        columns = np.empty(count, dtype=np.int64)
        values = np.empty(count, dtype=np.float64)
        rows[:nc] = np.arange(nc, dtype=np.int64)
        columns[:nc] = rows[:nc]
        with np.errstate(over='ignore', invalid='ignore'):
            values[:nc] = self.area*self.total[self.source.cell_material_ids, group]
        # Preserve COO insertion order: diagonals, interleaved face (+c,-c),
        # then strictly positive boundaries in input order. Keep signed zeros.
        rows[nc:end:2] = self.owners[:, 0]
        rows[nc+1:end:2] = self.owners[:, 1]
        upstream = np.where(coefficient >= 0, self.owners[:, 0], self.owners[:, 1])
        columns[nc:end:2] = upstream
        columns[nc+1:end:2] = upstream
        values[nc:end:2] = coefficient
        values[nc+1:end:2] = -coefficient
        rows[end:] = self.boundary_owners[outgoing]
        columns[end:] = rows[end:]
        values[end:] = boundary[outgoing]
        self._finite(values)
        matrix = coo_matrix((values, (rows, columns)), shape=(nc, nc)).tocsr()
        self._finite(matrix.data)
        return matrix

    def face_currents(self, angular_flux, incoming=None):
        """Area-integrated net currents [face,group], summed with physical weights.

        Interior sign is left-to-right; boundary sign is outward. Incoming=None
        is vacuum. These are angular integrals, unlike per-ordinate equations.
        """
        flux = self._flux(angular_flux)
        incoming = self._incoming(incoming)
        nf, nb = len(self.owners), len(self.boundary_owners)
        _budget(8*(6*np.prod(self.shape) + (nf+nb)*self.shape[2] + 6*self.shape[1]*self.shape[2]), self.max_bytes)
        interior = np.empty((nf, self.shape[2]))
        boundary = np.empty((nb, self.shape[2]))
        for face, ((left, right), area, normal) in enumerate(zip(self.owners, self.measures, self.normals)):
            speed = self.quadrature.directions[:, :self.dimension] @ normal
            trace = np.where(speed[:, None] >= 0, flux[left], flux[right])
            interior[face] = area*np.einsum('n,ng->g', self.quadrature.weights*speed, trace)
        for face, (owner, area, normal) in enumerate(zip(self.boundary_owners, self.boundary_measures, self.boundary_normals)):
            speed = self.quadrature.directions[:, :self.dimension] @ normal
            trace = np.zeros((self.shape[1], self.shape[2]))
            outgoing = speed > 0
            trace[outgoing] = flux[owner, outgoing]
            if incoming is not None:
                inflow = speed < 0
                trace[inflow] = incoming[face, inflow]
                self._finite(trace[inflow])
            boundary[face] = area*np.einsum('n,ng->g', self.quadrature.weights*speed, trace)
        return self._finite(interior), self._finite(boundary)
