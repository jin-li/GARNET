"""CPU geometry and interval-mean leakage closure for the NEM prototype.

Faces are exact, conforming whole cuboid faces. Coordinates are matched exactly;
callers should reuse shared coordinates rather than rely on geometric tolerance.
The default consecutive-neighbor one-sided quadratic closure is a prototype
choice. The explicit primary_rule option uses boundary interpolation rules
from Di Pasquale, Appendix A.2 (printed pp. 123-125), and Bandini (1990, p. 41).
"""
from __future__ import annotations

import numpy as np

from .face_conventions import NEM_AXIS_FACE_PAIRS, NEM_NORMAL_SIGNS

_SIDE_PAIRS = NEM_AXIS_FACE_PAIRS


def _readonly(values, dtype=None):
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


class CuboidNodalGrid:
    """Conforming orthogonal cuboids with physically oriented shared faces.

    ``bounds`` has shape (N,3,2), or (N,6) in the order
    ``xmin,xmax,ymin,ymax,zmin,zmax``. Nodes keep input order. Face IDs follow
    sorted ``(axis, plane, tangent bounds)`` keys. ``face_nodes[f]`` lists the
    spatially low and high owners, with -1 for the exterior. Thus the first
    owner supplies its plus face and the second supplies its minus face.

    Node sides use ``x-,y-,x+,y+,z-,z+``. Boundary conditions are ``vacuum``
    (alpha=0.5) or ``reflective`` (alpha=0), either uniformly or in
    ``boundary_faces`` order. Array attributes and returned leakage arrays are
    independent read-only copies. Disconnected cuboids and true void gaps are
    allowed; overlapping volumes and hanging/partial faces are rejected.
    ``leakage_boundary`` selects ``one_sided`` (the default) or ``primary_rule``;
    see :meth:`leakage_weights` for the boundary interpolation constraints.
    ``boundary_kinds`` records the conditions in ``boundary_faces`` order.
    """

    def __init__(self, bounds, boundary_conditions='vacuum', *, leakage_boundary='one_sided'):
        if not isinstance(leakage_boundary, str) or leakage_boundary not in ('one_sided', 'primary_rule'):
            raise ValueError('leakage_boundary must be one_sided or primary_rule')
        self.leakage_boundary = leakage_boundary
        if hasattr(bounds, '__cuda_array_interface__'):
            raise ValueError('bounds must be a CPU array')
        raw = np.asarray(bounds)
        if raw.dtype.kind not in 'iuf' or raw.ndim not in (2, 3):
            raise ValueError('bounds must be a real CPU array of shape (N,6) or (N,3,2)')
        if raw.ndim == 2 and raw.shape[1:] == (6,):
            raw = raw.reshape(-1, 3, 2)
        if raw.shape[1:] != (3, 2) or len(raw) == 0:
            raise ValueError('bounds must contain at least one cuboid with shape (N,6) or (N,3,2)')
        b = np.array(raw, dtype=float, copy=True)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            widths = b[:, :, 1] - b[:, :, 0]
            volumes = widths.prod(axis=1)
            areas = np.column_stack([widths[:, (a+1) % 3]*widths[:, (a+2) % 3]
                                     for a in range(3)])
            centers = b[:, :, 0] + widths/2
        if (not np.all(np.isfinite(b)) or not np.all(np.isfinite(widths))
                or np.any(widths <= 0) or np.any(widths/2 <= 0)
                or not np.all(np.isfinite(volumes)) or np.any(volumes <= 0)
                or not np.all(np.isfinite(areas)) or np.any(areas <= 0)
                or not np.all(np.isfinite(centers))):
            raise ValueError('cuboids must have finite bounds, positive finite widths, areas and volumes')
        self.n_nodes = len(b)
        # Sweep the x intervals, checking positive intersection in all axes.
        order = np.argsort(b[:, 0, 0], kind='stable')
        active = []
        for node in order:
            active = [j for j in active if b[j, 0, 1] > b[node, 0, 0]]
            for other in active:
                if np.all(np.minimum(b[node, :, 1], b[other, :, 1]) >
                          np.maximum(b[node, :, 0], b[other, :, 0])):
                    raise ValueError('duplicate or overlapping cuboid volumes')
            active.append(node)

        owners = {}
        planes = {}
        for node in range(self.n_nodes):
            for axis, pair in enumerate(_SIDE_PAIRS):
                tangent = [a for a in range(3) if a != axis]
                footprint = tuple(b[node, tangent, :].ravel())
                for end, side in enumerate(pair):
                    plane = b[node, axis, end]
                    key = (axis, plane, *footprint)
                    owners.setdefault(key, []).append((node, side, end))
                    planes.setdefault((axis, plane), set()).add(footprint)
        for footprints in planes.values():
            ordered = sorted(footprints)
            for i, first in enumerate(ordered):
                for second in ordered[i+1:]:
                    if (min(first[1], second[1]) > max(first[0], second[0])
                            and min(first[3], second[3]) > max(first[2], second[2])):
                        raise ValueError('nonconforming partial coplanar face overlap (hanging face)')

        keys = sorted(owners)
        nf = np.empty((self.n_nodes, 6), dtype=np.int64)
        neighbors = np.full((self.n_nodes, 6), -1, dtype=np.int64)
        face_nodes = np.full((len(keys), 2), -1, dtype=np.int64)
        axes, face_areas, face_planes = [], [], []
        boundary_faces, boundary_nodes, boundary_sides = [], [], []
        for face, key in enumerate(keys):
            axis, plane, t0, t1, u0, u1 = key
            entries = owners[key]
            if len(entries) > 2 or (len(entries) == 2 and entries[0][2] == entries[1][2]):
                raise ValueError('a whole face must have one owner or two opposite owners')
            for node, side, end in entries:
                nf[node, side] = face
                face_nodes[face, 1-end] = node
                if len(entries) == 2:
                    neighbors[node, side] = next(n for n, _, _ in entries if n != node)
                else:
                    boundary_faces.append(face)
                    boundary_nodes.append(node)
                    boundary_sides.append(side)
            axes.append(axis)
            face_areas.append((t1-t0)*(u1-u0))
            face_planes.append(plane)
        if isinstance(boundary_conditions, str):
            conditions = [boundary_conditions]*len(boundary_faces)
        else:
            try:
                conditions = list(boundary_conditions)
            except TypeError as exc:
                raise ValueError('boundary_conditions must be a string or boundary-face list') from exc
        if len(conditions) != len(boundary_faces):
            raise ValueError('boundary_conditions must match boundary_faces order and length')
        if any(not isinstance(c, str) or c not in ('vacuum', 'reflective') for c in conditions):
            raise ValueError('only vacuum and reflective boundary conditions are supported')
        for name, value, dtype in (
            ('bounds', b, float), ('widths', widths, float), ('volumes', volumes, float),
            ('face_nodes', face_nodes, np.int64), ('face_axes', axes, np.int64),
            ('face_areas', face_areas, float), ('face_planes', face_planes, float),
            ('node_faces', nf, np.int64), ('neighbors', neighbors, np.int64),
            ('boundary_faces', boundary_faces, np.int64), ('boundary_nodes', boundary_nodes, np.int64),
            ('boundary_sides', boundary_sides, np.int64),
            ('boundary_kinds', conditions, str),
            ('boundary_alpha', [.5 if c == 'vacuum' else 0. for c in conditions], float),
            ('node_face_areas', np.array(face_areas)[nf], float),
            ('normal_signs', NEM_NORMAL_SIGNS, np.int64),
        ):
            setattr(self, name, _readonly(value, dtype))
        degrees = np.empty((self.n_nodes, 3), dtype=np.int64)
        condition = np.empty((self.n_nodes, 3))
        self._leakage = {}
        boundary_kind = dict(zip(zip(boundary_nodes, boundary_sides), conditions))
        for node in range(self.n_nodes):
            for axis, (minus_side, plus_side) in enumerate(_SIDE_PAIRS):
                minus, plus = neighbors[node, [minus_side, plus_side]]
                if leakage_boundary == 'primary_rule':
                    # Three interpolation constraints: the center interval mean
                    # and one immediate interval/face point on each side. These
                    # are not physical ghost cells and never bridge a void.
                    rows = [[1., 0., 1/12]]
                    sources = [node]
                    for end, side, neighbor in ((0, minus_side, minus), (1, plus_side, plus)):
                        if neighbor >= 0:
                            lo, hi = (b[neighbor, axis] - centers[node, axis])/widths[node, axis]
                            rows.append([1., (lo+hi)/2, (lo*lo+lo*hi+hi*hi)/3])
                            sources.append(int(neighbor))
                        elif boundary_kind[node, side] == 'reflective':
                            # The mirrored center-width interval copies the
                            # center leakage mean, rather than a new unknown.
                            rows.append([1., -1. if end == 0 else 1., 13/12])
                            sources.append(node)
                        else:
                            t = -.5 if end == 0 else .5
                            rows.append([1., t, t*t])
                            sources.append(None)  # prescribed zero face leakage
                    vandermonde = np.asarray(rows)
                    if not np.all(np.isfinite(vandermonde)):
                        raise ValueError('nonfinite leakage interval-mean reconstruction')
                    cond = np.linalg.cond(vandermonde)
                    if not np.isfinite(cond) or cond > 1e12:
                        raise ValueError('ill-conditioned leakage interval-mean reconstruction')
                    target = np.array([[0., 1/12, 0.], [0., 0., 1/60]])
                    constraint_weights = np.linalg.solve(vandermonde.T, target.T).T
                    samples = list(dict.fromkeys(s for s in sources if s is not None))
                    weights = np.zeros((2, len(samples)))
                    for column, source in enumerate(sources):
                        if source is not None:
                            weights[:, samples.index(source)] += constraint_weights[:, column]
                    if not np.all(np.isfinite(weights)):
                        raise ValueError('nonfinite leakage reconstruction weights')
                    self._leakage[node, axis] = (_readonly(samples, np.int64), _readonly(weights))
                    degrees[node, axis], condition[node, axis] = 2, cond
                    continue
                samples = [node]
                if minus >= 0 and plus >= 0:
                    samples += [minus, plus]
                elif minus >= 0 or plus >= 0:
                    side = minus_side if minus >= 0 else plus_side
                    first = neighbors[node, side]
                    samples.append(first)
                    second = neighbors[first, side]
                    if second >= 0:
                        samples.append(second)
                # Exact whole-face connectivity guarantees equal tangential footprints.
                for sample in samples:
                    tangent = [a for a in range(3) if a != axis]
                    if not np.array_equal(b[sample, tangent], b[node, tangent]):
                        raise ValueError('leakage samples require compatible tangent footprints')
                degree = len(samples)-1
                with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                    interval = (b[samples, axis, :] - centers[node, axis])/widths[node, axis]
                    lo, hi = interval.T
                    vandermonde = np.column_stack((np.ones(len(samples)), (lo+hi)/2,
                                                   (lo*lo+lo*hi+hi*hi)/3))[:, :degree+1]
                if not np.all(np.isfinite(vandermonde)):
                    raise ValueError('nonfinite leakage interval-mean reconstruction')
                cond = np.linalg.cond(vandermonde)
                if not np.isfinite(cond) or cond > 1e12:
                    raise ValueError('ill-conditioned leakage interval-mean reconstruction')
                target = np.array([[0., 1/12, 0.], [0., 0., 1/60]])[:, :degree+1]
                weights = np.linalg.solve(vandermonde.T, target.T).T
                if not np.all(np.isfinite(weights)):
                    raise ValueError('nonfinite leakage reconstruction weights')
                self._leakage[node, axis] = (_readonly(samples, np.int64), _readonly(weights))
                degrees[node, axis], condition[node, axis] = degree, cond
        self.leakage_degrees = _readonly(degrees)
        self.leakage_condition = _readonly(condition)

    def leakage_weights(self, node, axis):
        """Return sample node IDs and (2,K) moment weights for interval means.

        Fit a monomial polynomial of degree at most two in the center node's
        coordinate ``t=(x-center)/width``. The resulting moments integrate
        against ``t`` and ``3*t**2-1/4`` on [-1/2,1/2]. Use both immediate
        neighbors when possible, otherwise up to two consecutive neighbors on
        the available side. Short chains reduce to linear or constant fits.

        With ``leakage_boundary='primary_rule'``, always fit a quadratic using
        immediate neighbors. An absent vacuum neighbor supplies a zero leakage
        point at the physical face; an absent reflective neighbor supplies a
        mirrored center-width interval with the center leakage value. These
        are interpolation constraints, not physical ghost cells. Zero face
        leakage approximates the zero-incoming Robin boundary; the actual
        boundary condition retains alpha=0.5. Returned IDs are physical nodes
        only: zero-point columns are dropped and repeated center coefficients
        are combined. Thus K can be 1 or 2 while leakage_degrees remains 2.
        """
        if (not isinstance(node, (int, np.integer)) or isinstance(node, (bool, np.bool_))
                or not 0 <= node < self.n_nodes):
            raise ValueError('node must be a valid integer node ID')
        if (not isinstance(axis, (int, np.integer)) or isinstance(axis, (bool, np.bool_))
                or not 0 <= axis < 3):
            raise ValueError('axis must be 0, 1 or 2')
        return self._leakage[node, axis]
