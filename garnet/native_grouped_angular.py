"""Opt-in one-energy-group angular operators on the reference 2D geometry.

No solve, fission, eigeniteration, or automatic operator selection is provided.
The source's canonical signed SP0..SPL, quadrature and energy grid are shared.
This is a mathematical prerequisite, not physical-array/3D qualification.
"""
from dataclasses import dataclass, field

import numpy as np

from .native_angular_source import _budget, _integer
from .native_mesh_streaming import NativeMeshStreaming

_BATCH = 512


@dataclass(frozen=True)
class GroupAngularOperators:
    """Cell-integrated Tg and Tg-Sgg, with explicit vacuum/specular boundaries.

    Caller-owned inputs must be real numeric ndarrays (strides are allowed).
    They and all previously returned fields/moment caches are excluded from
    max_bytes, as is the already-owned immutable streaming/source geometry.
    Each call guards a conservative peak of its own numeric arrays, including
    retained reflection permutations, outputs, float conversions and bounded
    validation/gather/BLAS work. This is not a process-RSS or BLAS-thread limit;
    Python objects and implementation-specific BLAS scratch are not estimated.
    No full multigroup angular array or angle-by-face cache is allocated.
    """
    streaming: NativeMeshStreaming
    boundary_conditions: tuple
    max_bytes: int = field(default=256*1024**2, repr=False, compare=False)
    reflection_permutations: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.streaming, NativeMeshStreaming):
            raise ValueError('NativeMeshStreaming required')
        s = self.streaming
        if not isinstance(self.boundary_conditions, tuple):
            raise ValueError('explicit boundary condition tuple required')
        bc = self.boundary_conditions
        if len(bc) != len(s.boundary_owners) or any(v not in ('vacuum', 'reflective') for v in bc):
            raise ValueError('one vacuum/reflective boundary condition per boundary face required')
        # At most two shared axis permutations; allow reflection construction,
        # immutable backing copies, validation vectors and boolean temporaries.
        _budget(8*(100*s.shape[1]+1), self.max_bytes)
        axes, permutations = {}, []
        directions, weights = s.quadrature.directions, s.quadrature.weights
        for spec, normal in zip(bc, s.boundary_normals):
            if spec == 'vacuum':
                permutations.append(None)
                continue
            if abs(abs(normal[0])-1) <= 2e-13 and abs(normal[1]) <= 2e-13:
                axis, tangent = 'x', 1
            elif abs(abs(normal[1])-1) <= 2e-13 and abs(normal[0]) <= 2e-13:
                axis, tangent = 'y', 0
            else:
                raise ValueError('reflective normal must align with Cartesian x or y axis')
            if axis not in axes:
                axes[axis] = s.quadrature.reflection(axis)
            permutation = axes[axis]
            speed = directions[:, :2] @ normal
            if (not np.array_equal(permutation[permutation], np.arange(s.shape[1])) or
                    not np.array_equal(weights[permutation], weights) or
                    not np.allclose(directions[permutation, tangent], directions[:, tangent], atol=2e-13, rtol=0) or
                    not np.allclose(directions[permutation, 2], directions[:, 2], atol=2e-13, rtol=0) or
                    not np.allclose(speed[permutation], -speed, atol=2e-13, rtol=0)):
                raise ValueError('quadrature does not provide exact specular index reflection')
            permutations.append(permutation)
        object.__setattr__(self, 'reflection_permutations', tuple(permutations))

    @property
    def shape(self):
        """One-group angular shape [cell, angle]."""
        return self.streaming.shape[:2]

    @property
    def allocation_estimate_bytes(self):
        """Conservative uniform per-call numeric live-byte upper estimate.

        3 single-group fields + 2 local moment fields + weighted harmonics;
        bounded 512-cell conversions/gathers (including all-group *moments*,
        never all-group angles); and reflection/face/validation temporaries.
        No nc*na*ng or nb*na term occurs.
        """
        nc, na = self.shape
        h = self.streaming.quadrature.harmonics.shape[1]
        ng = self.streaming.source.n_groups
        b = min(nc, _BATCH)
        return 8*(3*nc*na + 2*nc*h + na*h + b*(4*h*ng+8*h+8*na+16) + 100*na+1)

    def _guard(self):
        _budget(self.allocation_estimate_bytes, self.max_bytes)

    def _group(self, group):
        group = _integer(group, 'group')
        if group >= self.streaming.source.n_groups:
            raise ValueError('group outside layout')
        return group

    @staticmethod
    def _array(value, shape, name):
        # Avoid implicitly allocating a caller's nested sequence before guarding.
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype.kind not in 'fiu':
            raise ValueError(f'{name} must be a real ndarray with shape {shape}')
        for start in range(0, shape[0], _BATCH):
            if not np.all(np.isfinite(value[start:start+_BATCH])):
                raise ValueError(f'{name} must be finite')
        return value

    def _finite(self, value):
        return self._array(value, value.shape, 'operator output')

    def project_group(self, psi_g):
        """Return moments [cell,harmonic] = sum_n w[n] H[n,a] psi_g[n]."""
        self._guard()
        flux = self._array(psi_g, self.shape, 'angular flux')
        q = self.streaming.quadrature
        weighted = q.weights[:, None]*q.harmonics
        result = np.empty((self.shape[0], q.harmonics.shape[1]))
        for start in range(0, len(result), _BATCH):
            stop = min(start+_BATCH, len(result))
            rows = np.asarray(flux[start:stop], dtype=np.float64)
            result[start:stop] = rows @ weighted
        return self._finite(result)

    def _scatter(self, moments, group, *, self_only=False, exclude_self=False):
        s, q = self.streaming, self.streaming.quadrature
        result = np.empty(self.shape)
        h = q.harmonics.shape[1]
        for start in range(0, self.shape[0], _BATCH):
            stop = min(start+_BATCH, self.shape[0])
            ids = s.source.cell_material_ids[start:stop]
            # Material selection is bounded by the cell batch, even for an
            # unused/zero material. Never build a global material index vector.
            for mid in np.unique(ids):
                production = s.source.production[int(mid)]
                selected = np.flatnonzero(ids == mid)
                rows = np.asarray(moments[start:stop][selected], dtype=np.float64)
                transformed = np.zeros((len(selected), h))
                for order in range(q.lmax+1):
                    block = slice(order*order, (order+1)**2)
                    if self_only:
                        transformed[:, block] = rows[:, block]*production[order, group, group]
                    else:
                        couplings = production[order, group, :]
                        if exclude_self:
                            # Exclude before contraction: subtracting a large
                            # self term afterward can erase small cross sources.
                            # Mask only a temporary vector; raw XS stay immutable.
                            couplings = couplings.copy()
                            couplings[group] = 0.
                        transformed[:, block] = rows[:, block, :] @ couplings
                angular = transformed @ q.harmonics.T
                angular *= s.area[start:stop][selected, None]/(4*np.pi)
                result[start+selected] = angular
        return self._finite(result)

    def scatter_to_group(self, m_all, g, *, exclude_self=False):
        """V*q_g from caller-owned [cell,harmonic,from_group] moments.

        Full canonical signed orders are used without removal or clipping.
        exclude_self removes precisely the g<-g term, including higher orders.
        """
        self._guard()
        g = self._group(g)
        if not isinstance(exclude_self, (bool, np.bool_)):
            raise ValueError('exclude_self must be boolean')
        s = self.streaming
        moments = self._array(m_all, (self.shape[0], s.quadrature.harmonics.shape[1], s.source.n_groups), 'moments')
        return self._scatter(moments, g, exclude_self=exclude_self)

    def same_group_scatter(self, psi_g, g):
        """V*Sgg*psi_g, retaining every supported harmonic order."""
        self._guard()
        g = self._group(g)
        return self._scatter(self.project_group(psi_g), g, self_only=True)

    def streaming_group_action(self, psi_g, g):
        """Tg: collision + conservative upwind streaming + mixed reflection."""
        self._guard()
        g = self._group(g)
        flux = self._array(psi_g, self.shape, 'angular flux')
        s = self.streaming
        result = np.empty(self.shape)
        for cell, mid in enumerate(s.source.cell_material_ids):
            result[cell] = s.area[cell]*s.total[mid, g]*np.asarray(flux[cell], dtype=np.float64)
        directions = s.quadrature.directions[:, :2]
        for (left, right), area, normal in zip(s.owners, s.measures, s.normals):
            speed = directions @ normal
            trace = np.where(speed >= 0, flux[left], flux[right])
            current = area*speed*trace
            result[left] += current
            result[right] -= current
        for face, (owner, area, normal) in enumerate(zip(s.boundary_owners, s.boundary_measures, s.boundary_normals)):
            speed = directions @ normal
            result[owner] += area*np.maximum(speed, 0)*flux[owner]
            permutation = self.reflection_permutations[face]
            if permutation is not None:
                incoming = np.flatnonzero(speed < 0)
                result[owner, incoming] += area*speed[incoming]*flux[owner, permutation[incoming]]
        return self._finite(result)

    def group_action(self, psi_g, g):
        """(Tg - V*Sgg) psi_g; off-group scattering belongs in a future RHS."""
        self._guard()
        result = self.streaming_group_action(psi_g, g)
        result -= self.same_group_scatter(psi_g, g)
        return self._finite(result)

    def diagonal_group(self, g):
        """Streaming/collision diagonal only, with no inversion or floors.

        Matches NativeMeshTransport.diagonal[:,:,g]; it does not subtract the
        scattering diagonal or fold reflected angular couplings onto it.
        """
        self._guard()
        g = self._group(g)
        s = self.streaming
        result = np.empty(self.shape)
        for cell, mid in enumerate(s.source.cell_material_ids):
            result[cell] = s.area[cell]*s.total[mid, g]
        directions = s.quadrature.directions[:, :2]
        for (left, right), area, normal in zip(s.owners, s.measures, s.normals):
            coefficient = area*(directions @ normal)
            result[left] += np.maximum(coefficient, 0)
            result[right] += np.maximum(-coefficient, 0)
        for owner, area, normal in zip(s.boundary_owners, s.boundary_measures, s.boundary_normals):
            result[owner] += np.maximum(area*(directions @ normal), 0)
        return self._finite(result)
