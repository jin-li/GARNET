"""Vertex-free finite-volume geometry for masked rectilinear 3D grids.

This module supplies geometry and connectivity only. Callers choose boundary
conditions for both outer-box faces and exposures next to inactive cells.
"""
from __future__ import annotations

import numpy as np

from .mesh import Mesh


SIDE_NAMES = ('x_minus', 'y_minus', 'x_plus', 'y_plus', 'z_minus', 'z_plus')
SIDE_NORMALS = np.array([[-1, 0, 0], [0, -1, 0], [1, 0, 0], [0, 1, 0],
                         [0, 0, -1], [0, 0, 1]], dtype=np.int64)
_SIDE_PAIRS = ((0, 2), (1, 3), (4, 5))


def _edges(values, name):
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size < 2 or raw.dtype.kind not in 'iuf':
        raise ValueError(f'{name} must be a real one-dimensional array with at least two edges')
    edges = raw.astype(float, copy=True)
    with np.errstate(over='ignore', invalid='ignore'):
        widths = np.diff(edges)
    if (not np.all(np.isfinite(edges)) or not np.all(np.isfinite(widths))
            or np.any(widths <= 0) or np.any(widths/2 <= 0)):
        raise ValueError(f'{name} must have finite strictly increasing edges and positive finite half-widths')
    return edges, widths


def build_rectilinear_mesh_3d(x_edges, y_edges, z_edges, *, active_mask, tags):
    """Return ``(Mesh, topology)`` for a conforming Cartesian cell grid.

    ``active_mask`` and ``tags`` have shape ``(nz, ny, nx)``. The mask must
    have boolean dtype, tags integer dtype; only active tags enter the mesh.
    Tags are labels and need not be contiguous or nonnegative. At least one
    cell must be active. Cell IDs follow active cells in deterministic C order
    (z, then y, then x); ``grid_indices`` contains their ``(iz, iy, ix)`` rows
    and ``grid_to_cell`` stores IDs or -1 for inactive cells.

    ``widths`` and ``centroids`` are cell arrays in x/y/z order. ``face_owners``
    and ``face_sides`` have shape (F,2), aligned with ``mesh.faces``. Owners
    are oriented from the lower to higher coordinate. ``boundary_owners`` and
    ``boundary_sides`` align with ``mesh.bfaces``. Side IDs follow SIDE_NAMES;
    ``side_normals`` are outward unit normals. ``boundary_outer_box`` marks
    faces outside the coordinate box; ``boundary_masked`` marks faces next to
    an inactive cell inside it. Neither category chooses a boundary condition.

    ``mesh.area`` contains cell volumes. Interior ``face_distances`` are the
    exact normal half-widths of both owners. Vertex coordinates and cell vertex
    tuples are deliberately absent: this Mesh supports the finite-volume
    operators, but not vertex-based mesh export. All returned arrays are owned
    by this result; inputs are neither modified nor retained as writable views.
    """
    x, dx = _edges(x_edges, 'x_edges')
    y, dy = _edges(y_edges, 'y_edges')
    z, dz = _edges(z_edges, 'z_edges')
    shape = (len(dz), len(dy), len(dx))
    mask = np.asarray(active_mask)
    labels = np.asarray(tags)
    if mask.dtype.kind != 'b' or mask.shape != shape:
        raise ValueError('active_mask must have boolean dtype and shape (nz, ny, nx)')
    if labels.dtype.kind not in 'iu' or labels.shape != shape:
        raise ValueError('tags must have integer dtype and shape (nz, ny, nx)')
    where = np.argwhere(mask)
    if not len(where):
        raise ValueError('active_mask must contain at least one active cell')
    ids = np.full(shape, -1, dtype=np.int64)
    ids[mask] = np.arange(len(where), dtype=np.int64)
    widths = np.column_stack((dx[where[:, 2]], dy[where[:, 1]], dz[where[:, 0]]))
    centroids = np.column_stack((x[where[:, 2]]+widths[:, 0]/2,
                                 y[where[:, 1]]+widths[:, 1]/2,
                                 z[where[:, 0]]+widths[:, 2]/2))
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        volumes = widths.prod(axis=1)
    if not np.all(np.isfinite(centroids)) or not np.all(np.isfinite(volumes)) or np.any(volumes <= 0):
        raise ValueError('Active cell centroids and positive volumes must be finite')

    faces, distances, sides = [], [], []
    boundaries, bsides, outer = [], [], []
    for axis, pair in enumerate(_SIDE_PAIRS):
        grid_axis = 2-axis
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            areas = widths[:, (axis+1)%3]*widths[:, (axis+2)%3]
        if not np.all(np.isfinite(areas)) or np.any(areas <= 0):
            raise ValueError('Active face areas must be finite and positive')
        for sign, side in zip((-1, 1), pair):
            other = where.copy()
            other[:, grid_axis] += sign
            inside = (other[:, grid_axis] >= 0) & (other[:, grid_axis] < shape[grid_axis])
            neighbor = np.full(len(where), -1, dtype=np.int64)
            neighbor[inside] = ids[tuple(other[inside].T)]
            owner = np.flatnonzero(neighbor < 0)
            boundaries.append(np.column_stack((owner, areas[owner], widths[owner, axis]/2)))
            bsides.append(np.full(len(owner), side, dtype=np.int64))
            outer.append(~inside[owner])
            if sign > 0:
                owner = np.flatnonzero(neighbor >= 0)
                j = neighbor[owner]
                di, dj = widths[owner, axis]/2, widths[j, axis]/2
                faces.append(np.column_stack((owner, j, areas[owner], di+dj)))
                distances.append(np.column_stack((di, dj)))
                sides.append(np.tile((pair[1], pair[0]), (len(owner), 1)))

    face_array = np.concatenate(faces)
    boundary_array = np.concatenate(boundaries)
    owners = face_array[:, :2].astype(np.int64)
    bowners = boundary_array[:, 0].astype(np.int64)
    face_sides, boundary_sides = np.concatenate(sides), np.concatenate(bsides)
    boundary_outer = np.concatenate(outer)
    mesh = Mesh(coords=np.empty((0, 3)), cells=[()]*len(where), cell_tag=labels[mask].copy(),
                centroid=centroids, area=volumes,
                faces=[(int(i), int(j), float(a), float(d)) for i, j, a, d in face_array],
                bfaces=[(int(i), float(a), float(d)) for i, a, d in boundary_array],
                face_distances=np.concatenate(distances))
    topology = {'grid_indices': where, 'grid_to_cell': ids,
                'widths': widths, 'centroids': centroids,
                'x_edges': x, 'y_edges': y, 'z_edges': z,
                'face_owners': owners, 'face_sides': face_sides,
                'boundary_owners': bowners, 'boundary_sides': boundary_sides,
                'boundary_outer_box': boundary_outer, 'boundary_masked': ~boundary_outer,
                'side_names': SIDE_NAMES, 'side_normals': SIDE_NORMALS.copy()}
    return mesh, topology
