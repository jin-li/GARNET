"""Opt-in, conforming polygon geometry with area centroids and signed distances.

No coordinate welding, orientation repair, hanging-edge repair, or sliver removal
is performed. See ``docs/development/polygon-mesh.md`` for the input contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from fractions import Fraction

import numpy as np

from .mesh import Mesh


_EPS = np.finfo(float).eps


@dataclass(frozen=True)
class PolygonMeshAudit:
    n_cells: int
    n_vertices: int
    n_interior_faces: int
    n_boundary_faces: int
    total_area: float
    first_moment: np.ndarray
    minimum_area: float
    minimum_normal_distance: float


@dataclass(frozen=True)
class ClipPiece:
    """One retained polygon; callers retain the parent/material identity."""

    vertices: np.ndarray
    area: np.longdouble
    first_moment: np.ndarray


def _integer_array(value, name):
    a = np.asarray(value)
    if a.dtype.kind not in "iu" or a.dtype.kind == "b":
        raise ValueError(f"{name} must contain integers")
    if a.dtype.kind == "u" and a.size and a.max() > np.iinfo(np.int64).max:
        raise ValueError(f"{name} exceeds int64 range")
    return a.astype(np.int64, copy=False)


def _polygon_geometry(coords, flat, starts, counts):
    """Segmented O(E) local-origin shoelace, moments, and centroid-fan checks."""
    owner = np.repeat(np.arange(len(starts)), counts)
    nxt = np.arange(len(flat)) + 1
    nxt[starts + counts - 1] = starts
    origin = coords[flat[starts]]
    p = coords[flat] - origin[owner]
    q = coords[flat[nxt]] - origin[owner]
    products = p[:, 0] * q[:, 1], p[:, 1] * q[:, 0]
    cross = products[0] - products[1]
    twice = np.add.reduceat(cross, starts)
    scale = np.add.reduceat(np.abs(products[0]) + np.abs(products[1]), starts)
    # Bound grows with local vertex count, never with global coordinate size.
    error = 32 * _EPS * counts * scale
    if np.any(~np.isfinite(twice)) or np.any(twice <= error) or np.any(twice / 2 <= 0):
        raise ValueError("cell area must be finite, certified positive, and CCW")
    local_centroid = np.column_stack([
        np.add.reduceat((p[:, k] + q[:, k]) * cross, starts) / (3 * twice)
        for k in (0, 1)])
    centroid = origin + local_centroid
    if not np.all(np.isfinite(centroid)):
        raise ValueError("nonfinite area centroid")
    # Use the actual rounded Mesh centroid, not an unrepresentable local point.
    _validate_fan(coords, flat, nxt, owner, centroid, starts, counts)
    return twice / 2, centroid, owner, nxt


def _validate_fan(coords, flat, nxt, owner, centroid, starts, counts):
    a, b = coords[flat] - centroid[owner], coords[flat[nxt]] - centroid[owner]
    terms = a[:, 0] * b[:, 1], a[:, 1] * b[:, 0]
    fan = terms[0] - terms[1]
    if np.any(~np.isfinite(fan)) or np.any(fan <= 32 * _EPS * (np.abs(terms[0]) + np.abs(terms[1]))):
        raise ValueError("polygon must have a strictly positive centroid fan")
    angles = np.arctan2(fan, np.einsum("ij,ij->i", a, b))
    winding = np.add.reduceat(angles, starts)
    if np.any(np.abs(winding - 2 * np.pi) > 128 * _EPS * counts):
        raise ValueError("polygon must have one angular winding (no self-intersections)")


def assemble_polygon_mesh(coords, cells, tags, *, boundary_tags) -> Mesh:
    """Build a strict conforming 2D Mesh from CCW centroid-star polygons.

    ``cells`` is an iterable of integer vertex-index sequences. ``boundary_tags``
    must map every one-owner edge (unordered index pair) to an integer label and
    must not label any interior or absent edge. Zero is an explicit valid label.
    Coordinate identity is exact; canonicalization belongs to the caller.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.all(np.isfinite(coords)):
        raise ValueError("coords must be finite with shape (n_vertices, 2)")
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError("duplicate coordinates require explicit caller canonicalization")
    cells = [tuple(_integer_array(c, "cell indices")) for c in cells]
    counts = np.asarray([len(c) for c in cells], dtype=np.int64)
    if len(cells) == 0 or np.any(counts < 3):
        raise ValueError("at least one cell, with at least three vertices, is required")
    tags = _integer_array(tags, "tags")
    if tags.shape != (len(cells),):
        raise ValueError("one integer tag is required per cell")
    flat = np.concatenate(cells)
    if np.any(flat < 0) or np.any(flat >= len(coords)):
        raise ValueError("invalid vertex index")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    area, centroid, owner, nxt = _polygon_geometry(coords, flat, starts, counts)
    repeated = np.lexsort((flat, owner))
    if np.any((owner[repeated[1:]] == owner[repeated[:-1]]) &
              (flat[repeated[1:]] == flat[repeated[:-1]])):
        raise ValueError("repeated vertex in cell")
    u, v = flat, flat[nxt]
    delta = coords[v] - coords[u]
    length = np.hypot(delta[:, 0], delta[:, 1])
    if np.any(~np.isfinite(length)) or np.any(length <= 0):
        raise ValueError("zero or nonfinite edge length")
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    order = np.lexsort((hi, lo))
    group_start = np.r_[0, 1 + np.flatnonzero(
        (lo[order[1:]] != lo[order[:-1]]) | (hi[order[1:]] != hi[order[:-1]]))]
    multiplicity = np.diff(np.r_[group_start, len(order)])
    if np.any(multiplicity > 2):
        raise ValueError("nonmanifold edge has more than two owners")
    first = order[group_start]
    interior = multiplicity == 2
    left, right = first[interior], order[group_start[interior] + 1]
    if np.any((u[left] != v[right]) | (v[left] != u[right]) |
              (owner[left] == owner[right])):
        raise ValueError("shared edge owners must have opposite orientations")
    if not isinstance(boundary_tags, Mapping):
        raise ValueError("boundary_tags must exhaustively map boundary edges")
    labels = {}
    for edge, tag in boundary_tags.items():
        edge = _integer_array(edge, "boundary edge")
        value = _integer_array(tag, "boundary tag")
        if edge.shape != (2,) or edge[0] == edge[1] or value.ndim != 0:
            raise ValueError("boundary tags require distinct index pairs and scalar integers")
        key = tuple(sorted(edge))
        if key in labels:
            raise ValueError("duplicate unordered boundary edge label")
        labels[key] = int(value)
    boundary = first[~interior]
    boundary_keys = [(int(lo[e]), int(hi[e])) for e in boundary]
    if labels.keys() != set(boundary_keys):
        raise ValueError("boundary tags must cover exactly all boundary edges; no interior tags")
    # A CCW edge has its outward normal to its right. Distances retain signs.
    normals = np.column_stack((delta[:, 1], -delta[:, 0])) / length[:, None]
    centers = coords[u] + delta / 2
    di = np.einsum("ij,ij->i", centers - centroid[owner], normals)
    dj = np.einsum("ij,ij->i", centroid[owner[right]] - centers[left], normals[left])
    if np.any(~np.isfinite(di)) or np.any(di <= 0) or np.any(~np.isfinite(dj)) or np.any(dj <= 0):
        raise ValueError("strictly positive signed normal distances are required")
    separation = np.linalg.norm(centroid[owner[left]] - centroid[owner[right]], axis=1)
    if np.any(~np.isfinite(separation)) or np.any(separation <= 0):
        raise ValueError("invalid centroid separation")
    faces = list(zip(owner[left].tolist(), owner[right].tolist(), length[left].tolist(), separation.tolist()))
    bfaces = list(zip(owner[boundary].tolist(), length[boundary].tolist(), di[boundary].tolist()))
    return Mesh(coords=coords, cells=cells, cell_tag=tags, centroid=centroid,
                area=area, faces=faces, bfaces=bfaces,
                face_distances=np.column_stack((di[left], dj)),
                face_centers=centers[left], face_normals=normals[left],
                bface_centers=centers[boundary], bface_normals=normals[boundary],
                bface_tags=np.asarray([labels[key] for key in boundary_keys], dtype=np.int64))


def polygon_mesh_audit(mesh: Mesh) -> PolygonMeshAudit:
    """Compact geometry summary; no solver or mesh modification."""
    distances = np.r_[mesh.face_distances.ravel(), [f[2] for f in mesh.bfaces]]
    return PolygonMeshAudit(mesh.n_cells, len(mesh.coords), len(mesh.faces),
                            len(mesh.bfaces), float(np.sum(mesh.area)),
                            np.sum(mesh.area[:, None] * mesh.centroid, axis=0),
                            float(np.min(mesh.area)), float(np.min(distances)))


def _plane_values(vertices, plane):
    with np.errstate(over="ignore", invalid="ignore"):
        terms = vertices * plane[:2]
        values = terms.sum(axis=1) - plane[2]
        bound = 16 * np.finfo(np.longdouble).eps * (np.abs(terms).sum(axis=1) + abs(plane[2]))
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(bound)):
        raise ValueError("halfplane evaluation exceeds numeric range")
    uncertain = (np.abs(values) <= bound) & (values != 0)
    if np.any(uncertain):
        raise ValueError("halfplane classification is numerically ambiguous; caller must resolve geometry")
    # A rounded cancellation to zero is not proof of exact plane incidence.
    # This slow path only handles apparent incidences, using the exact binary
    # values supplied to this predicate (including constructed intersections).
    def rational(x):
        return Fraction(*x.as_integer_ratio())

    for index in np.flatnonzero((values == 0) & (bound > 0)):
        exact = sum((rational(vertices[index, k]) * rational(plane[k]) for k in (0, 1)),
                    -rational(plane[2]))
        if exact != 0:
            raise ValueError("halfplane incidence is numerically ambiguous; caller must resolve geometry")
    return values


def _clip_convex(vertices, plane):
    values = _plane_values(vertices, plane)
    output = []
    for k, p in enumerate(vertices):
        q, vp, vq = vertices[(k + 1) % len(vertices)], values[k], values[(k + 1) % len(vertices)]
        if vp >= 0:
            output.append(p)
        if (vp < 0 and vq > 0) or (vp > 0 and vq < 0):
            # The scaled ratio avoids overflow in vp-vq for opposite signs.
            magnitude = max(abs(vp), abs(vq))
            t = (vp / magnitude) / (vp / magnitude - vq / magnitude)
            r = p + t * (q - p)
            # Solve one coordinate on the plane; never snap existing vertices.
            axis = int(np.argmax(np.abs(plane[:2])))
            other = 1 - axis
            r[axis] = (plane[2] - plane[other] * r[other]) / plane[axis]
            if not np.all(np.isfinite(r)):
                raise ValueError("halfplane intersection exceeds numeric range")
            output.append(r)
    if not output:
        return np.empty((0, 2), dtype=np.longdouble)
    result = np.asarray(output, dtype=np.longdouble)
    # Remove exact adjacent duplicates only, including the cyclic pair.
    result = result[np.any(result != np.roll(result, 1, axis=0), axis=1)]
    return result


def _piece(vertices):
    if len(vertices) < 3:
        return None
    local = vertices - vertices[0]
    q = np.roll(local, -1, axis=0)
    terms = local[:, 0] * q[:, 1], local[:, 1] * q[:, 0]
    cross = terms[0] - terms[1]
    twice = np.sum(cross)
    scale = np.sum(np.abs(terms[0]) + np.abs(terms[1]))
    if twice == 0 and scale == 0:
        return None  # certified line/point contact, not a positive-area sliver
    if not np.isfinite(twice) or twice <= 32 * np.finfo(vertices.dtype).eps * len(vertices) * scale:
        raise ValueError("clipped piece has ambiguous or nonpositive area")
    centroid = vertices[0] + np.sum((local + q) * cross[:, None], axis=0) / (3 * twice)
    area, moment = twice / 2, centroid * (twice / 2)
    if area <= 0 or not np.all(np.isfinite(moment)):
        raise ValueError("clipped piece has unrepresentable area or first moment")
    return ClipPiece(vertices.copy(), area, moment)


def clip_star_polygon(vertices, *, halfplanes=((0, 1, 0), (1, -1, 0)), fan_center=None):
    """Intersect a CCW centroid-star polygon with ``a*x+b*y >= c`` planes.

    A wholly retained polygon stays one polygon. A crossing polygon is split
    into its area-centroid fan and every positive convex clipped piece is
    returned, including disconnected intersections and arbitrarily small
    certified pieces. ``fan_center`` optionally supplies another strictly
    interior kernel point; it changes only the triangulation. Output uses
    longdouble; identity remains caller-owned.
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3 or not np.all(np.isfinite(vertices)):
        raise ValueError("vertices must be a finite (n>=3, 2) array")
    if len(np.unique(vertices, axis=0)) != len(vertices):
        raise ValueError("duplicate polygon vertices")
    _, centroid, _, _ = _polygon_geometry(vertices, np.arange(len(vertices)),
                                           np.array([0]), np.array([len(vertices)]))
    if fan_center is not None:
        center = np.asarray(fan_center, dtype=float)
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError("fan_center must be a finite 2-vector strictly inside the kernel")
        flat = np.arange(len(vertices))
        _validate_fan(vertices, flat, np.roll(flat, -1), np.zeros(len(vertices), dtype=int),
                      center[None, :], np.array([0]), np.array([len(vertices)]))
        centroid = center[None, :]
    planes = np.asarray(halfplanes, dtype=np.longdouble)
    if planes.size == 0:
        planes = planes.reshape(0, 3)
    if planes.ndim != 2 or planes.shape[1] != 3 or not np.all(np.isfinite(planes)) or np.any(np.all(planes[:, :2] == 0, axis=1)):
        raise ValueError("halfplanes require finite nonzero normals and shape (n, 3)")
    vertices = vertices.astype(np.longdouble)
    if all(np.all(_plane_values(vertices, plane) >= 0) for plane in planes):
        return [_piece(vertices)]
    pieces = []
    for k in range(len(vertices)):
        triangle = np.array([centroid[0], vertices[k], vertices[(k + 1) % len(vertices)]], dtype=np.longdouble)
        for plane in planes:
            triangle = _clip_convex(triangle, plane)
            if len(triangle) < 3:
                break
        piece = _piece(triangle)
        if piece is not None:
            pieces.append(piece)
    return pieces
