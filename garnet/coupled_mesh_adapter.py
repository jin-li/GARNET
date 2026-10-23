"""Explicit CPU geometry bridge for full-group coupled moment operators.

Only closed orthogonal 2D/3D finite-volume meshes are admitted. Normals and
boundary kinds are mandatory metadata; tags, vertices and midpoint guesses are
not used. No skew/hanging-face correction, DF, solver or science admission is
provided. Successful conversion establishes geometry consistency only.
"""

from numbers import Integral

import numpy as np

from .mesh import Mesh
from .coupled_moments import BoundaryFace, InteriorFace, OrthogonalMesh, _geometry


_ROUNDOFF = 1e-12


def to_orthogonal_mesh(mesh, *, interior_normals, boundary_normals, boundary_kinds):
    """Convert ``Mesh`` using normals/kinds in the exact stored face order.

    Interior normals point from the first owner to the second; boundary normals
    point outward. ``Mesh.face_distances`` is required even for zero interior
    faces (shape ``(0, 2)``). Owners must be integer scalars (not booleans).
    Distances and measures are preserved, never repaired or inferred. Each
    cell's area-normal closure and pyramid-volume identity are checked with
    relative tolerance 1e-12 scaled to its total face measure/volume.
    """
    if not isinstance(mesh, Mesh):
        raise ValueError("Explicit Mesh required")
    c = np.asarray(mesh.centroid, dtype=float)
    v = np.asarray(mesh.area, dtype=float)
    n = mesh.n_cells
    if (n < 1 or c.ndim != 2 or c.shape[0] != n or c.shape[1] not in (2, 3)
            or not np.all(np.isfinite(c)) or v.shape != (n,)
            or not np.all(np.isfinite(v)) or np.any(v <= 0)):
        raise ValueError("Finite positive cell volumes and 2D/3D centroids required")
    dim = c.shape[1]
    if mesh.face_distances is None:
        raise ValueError("Explicit face_distances required; no midpoint fallback")
    distances = np.asarray(mesh.face_distances, dtype=float)
    if (distances.shape != (len(mesh.faces), 2)
            or not np.all(np.isfinite(distances)) or np.any(distances <= 0)):
        raise ValueError("face_distances must be finite positive with shape (n_faces, 2)")

    def normals(value, count, name):
        a = np.asarray(value, dtype=float)
        if (a.shape != (count, dim) or not np.all(np.isfinite(a))
                or not np.allclose(np.linalg.norm(a, axis=1), 1,
                                   rtol=_ROUNDOFF, atol=_ROUNDOFF)):
            raise ValueError(f"{name} must have finite unit normals with shape {(count, dim)}")
        return a

    ni = normals(interior_normals, len(mesh.faces), "interior_normals")
    nb = normals(boundary_normals, len(mesh.bfaces), "boundary_normals")
    if isinstance(boundary_kinds, str):
        raise ValueError("Supply one explicit boundary kind per boundary face")
    try:
        kinds = tuple(boundary_kinds)
    except TypeError as exc:
        raise ValueError("Supply one explicit boundary kind per boundary face") from exc
    if len(kinds) != len(mesh.bfaces) or any(k not in ("vacuum", "reflective") for k in kinds):
        raise ValueError("Boundary kinds must be vacuum/reflective, one per boundary face")

    def owner(i):
        if isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) or not 0 <= i < n:
            raise ValueError("Face owner must be an exact integer cell index, not boolean")
        return int(i)

    def positive(a):
        if not np.isscalar(a) or not np.isfinite(a) or a <= 0:
            raise ValueError("Face measures and distances must be finite positive scalars")
        return float(a)

    closure = np.zeros((n, dim))
    surface = np.zeros(n)
    pyramid = np.zeros(n)
    planes = [[] for _ in range(n)]

    def add(i, area, normal, distance):
        # Repeated planes also catch boundary/interior double counting and
        # coplanar split faces, for which this adapter supplies no hanging support.
        for previous_normal, previous_distance in planes[i]:
            if (np.allclose(normal, previous_normal, rtol=0, atol=_ROUNDOFF)
                    and abs(distance - previous_distance)
                    <= _ROUNDOFF * max(distance, previous_distance)):
                raise ValueError("Duplicate or split face plane; hanging faces unsupported")
        planes[i].append((normal.copy(), distance))
        closure[i] += area * normal
        surface[i] += area
        pyramid[i] += area * distance / dim

    interior = []
    for f, normal, (di, dj) in zip(mesh.faces, ni, distances):
        if len(f) != 4:
            raise ValueError("Interior faces require two owners, measure and centroid distance")
        i, j = owner(f[0]), owner(f[1])
        if i == j:
            raise ValueError("Interior faces require different owners")
        area, legacy_distance = positive(f[2]), positive(f[3])
        delta = c[j] - c[i]
        scale = max(di + dj, legacy_distance, np.linalg.norm(delta))
        if (not np.isfinite(scale) or not np.all(np.isfinite(delta))
                or abs(di + dj - legacy_distance) > _ROUNDOFF * scale
                or abs(np.dot(delta, normal) - (di + dj)) > _ROUNDOFF * scale
                or np.linalg.norm(delta - (di + dj) * normal) > _ROUNDOFF * scale):
            raise ValueError("Orthogonal alignment and centroid/face distances must agree")
        center = c[i] + di * normal
        if np.linalg.norm(center - (c[j] - dj * normal)) > _ROUNDOFF * scale:
            raise ValueError("Interior face centers must agree")
        add(i, area, normal, di)
        add(j, area, -normal, dj)
        interior.append(InteriorFace(i, j, area, tuple(center), tuple(normal), float(di), float(dj)))

    boundary = []
    for f, normal, kind in zip(mesh.bfaces, nb, kinds):
        if len(f) != 3:
            raise ValueError("Boundary faces require owner, measure and distance")
        i, area, distance = owner(f[0]), positive(f[1]), positive(f[2])
        add(i, area, normal, distance)
        boundary.append(BoundaryFace(i, area, tuple(c[i] + distance * normal),
                                     tuple(normal), distance, kind))
    if (not np.all(np.isfinite(surface)) or not np.all(np.isfinite(closure))
            or np.any(surface <= 0)
            or np.any(np.linalg.norm(closure, axis=1) > _ROUNDOFF * surface)):
        raise ValueError("Unexplained or missing faces: per-cell area-normal closure failed")
    if (not np.all(np.isfinite(pyramid))
            or np.any(np.abs(pyramid - v) > _ROUNDOFF * np.maximum(pyramid, v))):
        raise ValueError("Geometric face volume must equal Mesh.area")
    result = OrthogonalMesh(v, c, tuple(interior), tuple(boundary))
    _geometry(result)
    return result
