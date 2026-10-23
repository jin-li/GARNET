"""Unstructured finite-volume diffusion on a Gmsh mesh.

Reads a Gmsh 2.2 ``.msh`` file -- 2D triangle/quad cells, or 3D tetrahedra,
hexahedra and prisms -- and solves the multigroup k-eigenvalue diffusion problem
on the arbitrary geometry it describes, the same job FEMFFUSION or GeN-Foam do
from a mesh rather than from a structured lattice. This is the general-geometry
escape hatch: where the structured Cartesian/hex operators need a lattice, this
one needs only cells, their vertices, and per-cell materials.

The scheme is a cell-centred two-point-flux finite volume: for a face shared by
cells i and j, the 2D coupling is A_face / (d_i / D_i + d_j / D_j), where
d_i and d_j are centroid-to-face distances projected on the face normal.
This is consistent on orthogonal meshes; skewed interfaces (including hanging
nodes) retain a conservative TPFA approximation without a nonorthogonal flux
correction. Legacy/manual and 3D meshes without these distances use the harmonic
mean coefficient times face measure over centroid separation. Boundary faces
get the Robin/albedo term (vacuum alpha = 1/2).
The connectivity is irregular, but
the within-group operator is still applied matrix-free as a pure *gather*: the
faces are stored as a row-wise ELLPACK adjacency (each cell's neighbours padded
to the maximum degree), and the apply reads each cell's neighbour fluxes and
combines them with per-slot weights -- no scatter, hence no GPU atomics, and a
coalesced write, the same access pattern that makes the structured stencils fast
on CUDA. The single NumPy/CuPy code path runs on CPU or GPU and reuses the
structured solvers' Jacobi/Neumann-PCG, putting the general-geometry track on the
GPU alongside the structured lattices rather than the earlier SciPy sparse direct
factorisation. Power iteration drives the outer fission source.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .backend import asnumpy, device_name, get_backend, synchronize
from .linalg import get_linear_solver, neumann_preconditioner, pcg
from .stencil import _face_valid, face_alpha


_VACUUM_ALPHA = 0.5


@dataclass
class Mesh:
    """Cells and faces of a 2D or 3D finite-volume mesh.

    ``area`` holds the cell measure (area in 2D, volume in 3D) and each face its
    measure (edge length in 2D, face area in 3D); the two-point-flux operator
    consumes these identically in either dimension.

    ``face_distances``, when provided, stores the normal distance from each
    face's two cell centroids to its plane, in the same i/j order as ``faces``.
    Omitting it preserves the harmonic-mean centroid-separation formula used
    by manual and 3D meshes. Optional ``face_centers``/``face_normals`` follow
    ``faces`` order, with unit normals directed from i to j. Their ``bface_*``
    counterparts follow ``bfaces`` order with outward owner normals. Boundary
    physical tags are integers in ``bface_tags`` (zero means untagged). Centers
    use the same length units as ``coords``; normals are dimensionless.
    """

    coords: np.ndarray              # (n_nodes, 2) or (n_nodes, 3)
    cells: list                     # list of node-index tuples (per cell)
    cell_tag: np.ndarray            # (n_cells,) physical/elementary tag per cell
    centroid: np.ndarray            # (n_cells, 2) or (n_cells, 3)
    area: np.ndarray                # (n_cells,) cell measure (2D area / 3D volume)
    faces: list                     # (i, j, face_measure, centroid_distance) interior
    bfaces: list                    # (i, face_measure, centroid_to_face_distance) boundary
    face_distances: np.ndarray | None = None  # (n_faces, 2) normal distances for i, j

    face_centers: np.ndarray | None = None   # (n_faces, dim)
    face_normals: np.ndarray | None = None   # unit normals from i toward j
    bface_centers: np.ndarray | None = None  # (n_bfaces, dim)
    bface_normals: np.ndarray | None = None  # unit outward owner normals
    bface_tags: np.ndarray | None = None     # physical tags; zero means untagged

    @property
    def n_cells(self) -> int:
        return len(self.cells)


class MeshGrid:
    """Shape-preserving finite-volume grid with authoritative boundary classes.

    Fields have shape ``(n_cells, 1, 1)``; these axes are storage only. Boundary
    conditions are a scalar or a sequence in ``mesh.bfaces`` order. Vacuum
    faces may represent outer or explicitly exposed void surfaces. No active
    mask or discontinuity-factor interface is inferred from this facade.
    """

    geometry = "mesh"

    def __init__(self, mesh: Mesh, boundary_conditions="vacuum"):
        self.mesh = mesh
        self.shape = (mesh.n_cells, 1, 1)
        self.n_cells = mesh.n_cells
        self.integration_weights = np.asarray(mesh.area, dtype=float).reshape(self.shape)
        if self.n_cells < 1 or not np.all(np.isfinite(self.integration_weights)) or np.any(self.integration_weights <= 0):
            raise ValueError("mesh cell volumes must be finite and positive")
        for faces, interior in ((mesh.faces, True), (mesh.bfaces, False)):
            for face in faces:
                owners = face[:2] if interior else face[:1]
                if any(int(c) != c or c < 0 or c >= self.n_cells for c in owners):
                    raise ValueError("mesh face has an invalid cell owner")
                if interior and owners[0] == owners[1]:
                    raise ValueError("interior face must join different cells")
                if any(not np.isfinite(x) or x <= 0 for x in face[2 if interior else 1:]):
                    raise ValueError("mesh face measures and distances must be finite and positive")
        if mesh.face_distances is not None:
            distances = np.asarray(mesh.face_distances)
            if distances.shape != (len(mesh.faces), 2) or not np.all(np.isfinite(distances)) or np.any(distances <= 0):
                raise ValueError("face_distances must be positive with shape (n_faces, 2)")
        self.boundary_conditions = self.classify_boundary(boundary_conditions)

    def classify_boundary(self, bc=None):
        """Resolve exactly one boundary class for each physical boundary face."""
        if bc is None:
            return self.boundary_conditions
        if isinstance(bc, (str, int, float, np.number)):
            specs = [bc] * len(self.mesh.bfaces)
        else:
            specs = list(bc)
            if len(specs) != len(self.mesh.bfaces):
                raise ValueError("mesh boundary_conditions must have one entry per boundary face")
        result = tuple(_face_valid(s) for s in specs)
        if any(np.isnan(face_alpha(s)) for s in result):
            raise ValueError("mesh boundary coefficients must not be NaN")
        return result

    def moment_boundary_conditions(self, bc=None, strip_vacuum=False):
        specs = self.classify_boundary(bc)
        return tuple("reflective" if strip_vacuum and s == "vacuum" else s for s in specs)

    def boundary_blocks(self, xp, bc, D_list, g, boundary_df=None):
        """Integrated Marshak loss, summed on host by unique cell owner.

        Trace elimination uses the same geometric normal distance for every
        moment: ``K = S f G solve(diag(D/d) + f G, diag(D/d))``.
        The optional positive scalar factor corrects every moment trace alike.
        Aggregating before
        device indexing avoids repeated-index scatter loss on GPU backends.
        """
        specs = self.classify_boundary(bc)
        M = len(D_list)
        D = [asnumpy(d).reshape(-1) for d in D_list]
        gm = np.asarray(g, dtype=D[0].dtype)
        factors = _mesh_df(boundary_df, (len(self.mesh.bfaces),), "moment_boundary_df")
        blocks = {}
        for (owner, measure, distance), spec, factor in zip(self.mesh.bfaces, specs, factors):
            owner = int(owner)  # validated integer-like mesh identifiers
            if spec != "vacuum":
                continue
            A = np.diag([d[owner] / distance for d in D])
            fg = factor * gm
            K = measure * fg @ np.linalg.solve(A + fg, A)
            blocks[owner] = blocks.get(owner, np.zeros((M, M), dtype=gm.dtype)) + K
        if not blocks:
            return []
        owners = np.asarray(sorted(blocks), dtype=np.int32)
        sl = (xp.asarray(owners), 0, 0)
        return [(sl, xp.asarray(np.stack([blocks[c] for c in owners]))) ]


def _mesh_df(value, shape, name):
    """Validate scalar diffusion factors on host, including empty face lists."""
    factors = np.ones(shape) if value is None else np.asarray(asnumpy(value), dtype=float)
    if factors.shape != shape or not np.all(np.isfinite(factors)) or np.any(factors <= 0):
        raise ValueError(f"{name} must be finite positive factors with shape {shape}")
    return factors


class MeshGroupOperator:
    """Integrated conservative TPFA loss with density-valued coefficients.

    Reactions and the RHS carry the cell volume exactly once. On orthogonal
    faces the resistance is ``d_i/D_i + d_j/D_j``; legacy geometry without
    normal distances splits the centroid separation equally (d/2 per side),
    retaining harmonic-centroid TPFA for unit factors. Scalar face factors
    enforce continuity of the corrected trace f*phi; the same face current
    enters the two rows with opposite signs. At a boundary the corrected trace
    is f_B*phi_surface; eliminating it gives S*alpha*f_B*D/(D+alpha*f_B*d).
    Reflective leakage is zero and the zero-flux limit is S*D/d.
    ``apply`` and ``diag``
    preserve the facade shape used by the shared moment and eigen drivers.
    """

    def __init__(self, xp, grid, D, removal, bc=None, active=None,
                 mask_bc="vacuum", symmetric=True, face_df=None, boundary_df=None):
        if active is not None:
            raise ValueError("mesh active masks are unsupported; expose boundary faces explicitly")
        if not symmetric:
            raise ValueError("mesh operators use the integrated symmetric form")
        self.xp = xp
        self.grid = grid
        mesh = grid.mesh
        n = grid.n_cells
        Dh = asnumpy(D).reshape(-1)
        rh = asnumpy(removal).reshape(-1)
        if Dh.size != n or rh.size != n or not np.all(np.isfinite(Dh)) or np.any(Dh <= 0):
            raise ValueError("mesh diffusion must have one finite positive value per cell")
        fi = np.asarray([f[0] for f in mesh.faces], dtype=np.int64)
        fj = np.asarray([f[1] for f in mesh.faces], dtype=np.int64)
        measure = np.asarray([f[2] for f in mesh.faces])
        factors = _mesh_df(face_df, (len(mesh.faces), 2), "face_df")
        bfactors = _mesh_df(boundary_df, (len(mesh.bfaces),), "boundary_df")
        if mesh.face_distances is None:
            distance = np.asarray([f[3] for f in mesh.faces])
            distances = np.column_stack((distance / 2, distance / 2))
        else:
            distances = np.asarray(mesh.face_distances)
        weights = measure / (factors[:, 0] * distances[:, 0] / Dh[fi]
                             + factors[:, 1] * distances[:, 1] / Dh[fj])
        if mesh.face_distances is None and np.all(factors == 1):
            # Preserve the historical harmonic expression and roundoff.
            weights = measure * 2 * Dh[fi] * Dh[fj] / ((Dh[fi] + Dh[fj]) * distance)
        left, right = weights * factors[:, 0], weights * factors[:, 1]
        self.symmetric = bool(np.array_equal(left, right))
        nbr, fmap = _build_ell(fi, fj, n)
        # A sentinel handles one-cell and other meshes with no interior faces.
        safe = np.where(fmap >= 0, fmap, len(weights))
        owner_left = np.arange(n)[None, :] == np.append(fi, -1)[safe]
        well = np.where(owner_left, np.append(right, 0)[safe], np.append(left, 0)[safe])
        diag = rh * mesh.area + np.bincount(fi, weights=left, minlength=n) + np.bincount(fj, weights=right, minlength=n)
        for (owner, area, distance), spec, factor in zip(mesh.bfaces, grid.classify_boundary(bc), bfactors):
            owner = int(owner)  # validated integer-like mesh identifiers
            alpha = face_alpha(spec)
            diag[owner] += (area * Dh[owner] / distance if np.isinf(alpha)
                            else area * alpha * factor * Dh[owner] / (distance * alpha * factor + Dh[owner]))
        self._flat = _MeshGroupOperator(xp, xp.asarray(nbr, dtype=np.int32),
                                        xp.asarray(well, dtype=D.dtype), xp.asarray(diag, dtype=D.dtype))
        self.diag = self._flat.diag.reshape(grid.shape)
        self.inv_diag = 1 / self.diag
        self.rhs_weight = xp.asarray(grid.integration_weights, dtype=D.dtype)

    def apply(self, phi):
        return self._flat.apply(phi.reshape(-1)).reshape(self.grid.shape)


# Gmsh element type -> node count, split by dimension.
_GMSH_2D = {2: 3, 3: 4}                          # triangle, quad
_GMSH_3D = {4: 4, 5: 8, 6: 6}                     # tetrahedron, hexahedron, prism


def read_gmsh(path: str, *, normal_distances=None) -> Mesh:
    """Read Gmsh 2.2 ASCII cells and physical boundary tags.

    Node IDs may be sparse or unordered. The first element tag is the physical
    tag (zero for an element without tags); remaining tags are skipped according
    to the declared count. Lower-dimensional elements label matching boundary
    node sets. ``normal_distances=None`` keeps the 2D/3D assembly defaults.
    """
    with open(path) as stream:
        lines = stream.read().splitlines()
    mi = lines.index("$MeshFormat")
    fmt = lines[mi + 1].split()
    if fmt[:2] != ["2.2", "0"]:
        raise ValueError("read_gmsh requires Gmsh 2.2 ASCII format")
    ni = lines.index("$Nodes")
    nn = int(lines[ni + 1])
    coords = np.zeros((nn, 3))
    node_index = {}
    for k in range(nn):
        t = lines[ni + 2 + k].split()
        node_id = int(t[0])
        if node_id in node_index:
            raise ValueError(f"duplicate Gmsh node ID {node_id}")
        node_index[node_id] = k
        coords[k] = [float(x) for x in t[1:4]]
    ei = lines.index("$Elements")
    ne = int(lines[ei + 1])
    elems = []
    for k in range(ne):
        t = [int(x) for x in lines[ei + 2 + k].split()]
        ntags = t[2]
        if ntags < 0 or len(t) < 3 + ntags:
            raise ValueError("invalid Gmsh element tag count")
        elems.append((t[1], t[3] if ntags else 0, t[3 + ntags:]))
    is_3d = any(kind in _GMSH_3D for kind, _, _ in elems)
    table = _GMSH_3D if is_3d else _GMSH_2D
    boundary_table = _GMSH_2D if is_3d else {1: 2}
    cells, tags, boundary_tags = [], [], {}
    for kind, physical, nodes in elems:
        nnode = table.get(kind, boundary_table.get(kind))
        if nnode is None:
            continue
        if len(nodes) != nnode:
            raise ValueError(f"Gmsh element type {kind} requires {nnode} nodes")
        try:
            nodes = tuple(node_index[n] for n in nodes)
        except KeyError as exc:
            raise ValueError(f"unknown Gmsh node ID {exc.args[0]}") from exc
        if kind in table:
            cells.append(nodes)
            tags.append(physical)
        else:
            key = tuple(sorted(nodes))
            if key in boundary_tags and boundary_tags[key] != physical:
                raise ValueError("conflicting physical tags on a boundary face")
            boundary_tags[key] = physical
    kwargs = {"boundary_tags": boundary_tags}
    if normal_distances is not None:
        kwargs["normal_distances"] = normal_distances
    if is_3d:
        return assemble_mesh_3d(coords, cells, tags, **kwargs)
    return assemble_mesh(coords[:, :2], cells, tags, **kwargs)


def _validate_mesh_geometry(mesh):
    """Reject degenerate or incorrectly oriented assembled geometry."""
    if (len(mesh.cell_tag) != mesh.n_cells or not np.all(np.isfinite(mesh.coords))
            or not np.all(np.isfinite(mesh.area)) or np.any(mesh.area <= 0)):
        raise ValueError("mesh requires one tag per cell and finite positive cell measures")
    for entries, centers, normals, interior in (
            (mesh.faces, mesh.face_centers, mesh.face_normals, True),
            (mesh.bfaces, mesh.bface_centers, mesh.bface_normals, False)):
        for face, center, normal in zip(entries, centers, normals):
            quantities = face[2:] if interior else face[1:]
            if not np.all(np.isfinite(quantities)) or np.any(np.asarray(quantities) <= 0):
                raise ValueError("mesh face measures and distances must be finite and positive")
            if not np.all(np.isfinite(normal)) or not np.isclose(np.linalg.norm(normal), 1):
                raise ValueError("mesh face normal must be a finite unit vector")
            if np.dot(center - mesh.centroid[face[0]], normal) <= 0:
                raise ValueError("mesh face normal must point outward from its owner")
            if interior and np.dot(mesh.centroid[face[1]] - center, normal) <= 0:
                raise ValueError("interior face must separate its cell centroids")
    return mesh


def _boundary_tag_map(boundary_tags):
    return {} if boundary_tags is None else {
        tuple(sorted(nodes)): int(tag) for nodes, tag in boundary_tags.items()}


def assemble_mesh(coords, cells, tags, *, normal_distances=True, boundary_tags=None) -> Mesh:
    """Build a Mesh (centroids, areas, interior/boundary faces) from polygons.

    coords : (n_nodes, 2) node coordinates. cells : list of node-index tuples.
    tags   : one integer per cell (material/assembly id). A face is a boundary
    face when only one cell borders it. ``boundary_tags`` optionally maps
    boundary node tuples (in either order) to integer physical labels.

    ``normal_distances=True`` supplies face-normal interior resistances and
    boundary distances. False retains the historical centroid-separation
    interior coupling and Euclidean centroid-to-boundary-midpoint distances.
    Both are conservative TPFA approximations on skewed interfaces; neither
    includes a nonorthogonal flux correction.

    Nonconforming (locally-refined) interfaces are supported: a one-cell edge
    (i, j) whose midpoint is itself a mesh node m, with the half-edges (i, m)
    and (m, j) each owned by another cell, is a coarse edge meeting two fine
    cells across a 2:1 hanging node. It is split into two interior faces
    coupling the coarse cell to each fine cell (each carrying its half-edge
    length) -- the standard conservative two-point-flux treatment. Purely
    conforming meshes are unaffected (their one-cell edges have no midpoint
    node, so they stay boundary faces).
    """
    coords = np.asarray(coords, dtype=float)
    cells = [tuple(c) for c in cells]
    nc = len(cells)
    centroid = np.zeros((nc, 2))
    area = np.zeros(nc)
    edge_cells = defaultdict(list)
    for c, ns in enumerate(cells):
        P = coords[list(ns)]
        # Subtract a local origin before shoelace products. Thin fitted cells
        # can have tiny areas compared with their global coordinate products.
        # The translation preserves area and avoids their cancellation.
        local = P - P[0]
        x, y = local[:, 0], local[:, 1]
        area[c] = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        centroid[c] = P.mean(0)
        m = len(ns)
        for a in range(m):
            edge_cells[tuple(sorted((ns[a], ns[(a + 1) % m])))].append(c)

    # Node lookup by (rounded) coordinate, over nodes actually used by cells,
    # for detecting a hanging-node midpoint on a one-cell edge.
    def key(p):
        return (round(float(p[0]), 6), round(float(p[1]), 6))
    used = {n for ns in cells for n in ns}
    node_at = {key(coords[n]): n for n in used}
    one_cell = {e: lst[0] for e, lst in edge_cells.items() if len(lst) == 1}

    def dist(a, b):
        return float(np.hypot(*(centroid[a] - centroid[b])))

    faces, face_distances, bfaces, consumed = [], [], [], set()
    face_centers, face_normals, bface_centers, bface_normals, bface_tags = [], [], [], [], []
    boundary_tags = _boundary_tag_map(boundary_tags)

    def add_face(i, j, edge):
        p0, p1 = coords[list(edge)]
        tangent = p1 - p0
        L = float(np.hypot(*tangent))
        if not np.isfinite(L) or L <= 0:
            raise ValueError("mesh edge length must be finite and positive")
        normal = np.array([-tangent[1], tangent[0]]) / L
        if np.dot(normal, centroid[j] - centroid[i]) < 0:
            normal = -normal
        midpoint = 0.5 * (p0 + p1)
        faces.append((i, j, L, dist(i, j)))
        face_centers.append(midpoint)
        face_normals.append(normal)
        face_distances.append((abs(float(np.dot(centroid[i] - midpoint, normal))),
                               abs(float(np.dot(centroid[j] - midpoint, normal)))))
    for e, ci in one_cell.items():
        if e in consumed:
            continue
        a, b = e
        mnode = node_at.get(key(0.5 * (coords[a] + coords[b])))
        if mnode is None or mnode in (a, b):
            continue
        e1, e2 = tuple(sorted((a, mnode))), tuple(sorted((mnode, b)))
        if e1 in one_cell and e2 in one_cell:      # coarse edge over two fine cells
            for half in (e1, e2):
                cj = one_cell[half]
                add_face(ci, cj, half)
            consumed.update((e, e1, e2))

    for e, lst in edge_cells.items():
        if e in consumed:
            continue
        p0, p1 = coords[e[0]], coords[e[1]]
        L = float(np.hypot(*(p1 - p0)))
        if len(lst) == 2:
            i, j = lst
            add_face(i, j, e)
        elif len(lst) == 1:
            i = lst[0]
            if not np.isfinite(L) or L <= 0:
                raise ValueError("mesh edge length must be finite and positive")
            tangent = p1 - p0
            normal = np.array([-tangent[1], tangent[0]]) / L
            midpoint = 0.5 * (p0 + p1)
            if np.dot(normal, midpoint - centroid[i]) < 0:
                normal = -normal
            bface_centers.append(midpoint)
            bface_normals.append(normal)
            bface_tags.append(boundary_tags.get(e, 0))
            db = (abs(float(np.dot(centroid[i] - midpoint, normal)))
                  if normal_distances else float(np.linalg.norm(centroid[i] - midpoint)))
            bfaces.append((i, L, db))
        else:
            raise ValueError(f"edge {e} is shared by {len(lst)} cells (non-manifold mesh)")
    return _validate_mesh_geometry(Mesh(coords=coords, cells=cells, cell_tag=np.asarray(tags), centroid=centroid,
               area=area, faces=faces, bfaces=bfaces,
               face_distances=(np.asarray(face_distances).reshape(-1, 2)
                               if normal_distances else None),
               face_centers=np.asarray(face_centers).reshape(-1, 2),
               face_normals=np.asarray(face_normals).reshape(-1, 2),
               bface_centers=np.asarray(bface_centers).reshape(-1, 2),
               bface_normals=np.asarray(bface_normals).reshape(-1, 2),
               bface_tags=np.asarray(bface_tags, dtype=int)))


# Local face node-orderings per 3D element type, keyed by node count. Orientation
# is irrelevant here: face areas and cell volumes are taken in magnitude and a
# face is identified by the *set* of its nodes, so any consistent enumeration of
# each element's bounding faces works. (Gmsh 4=tet, 5=hex, 6=prism/wedge.)
_TET_FACES = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
_HEX_FACES = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
              (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
_PRISM_FACES = [(0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)]
_FACE_TEMPLATES = {4: _TET_FACES, 8: _HEX_FACES, 6: _PRISM_FACES}


def _poly_area_centroid_3d(pts):
    """Area and area-weighted centroid of a planar polygon in 3D (fan triangulation)."""
    p0 = pts[0]
    atot = 0.0
    csum = np.zeros(3)
    for i in range(1, len(pts) - 1):
        cr = np.cross(pts[i] - p0, pts[i + 1] - p0)
        a = 0.5 * float(np.linalg.norm(cr))
        atot += a
        csum += a * (p0 + pts[i] + pts[i + 1]) / 3.0
    return atot, (csum / atot if atot > 0 else pts.mean(0))


def _cell_volume_centroid_3d(cell_pts, faces_local):
    """Volume and volume-weighted centroid of a star-convex polyhedron.

    Decomposes into tetrahedra from a seed point (the vertex mean) to a fan
    triangulation of every bounding face. Exact for convex tets/hexes/prisms.
    """
    g = cell_pts.mean(0)
    vtot = 0.0
    csum = np.zeros(3)
    for f in faces_local:
        fp = cell_pts[list(f)]
        p0 = fp[0]
        for i in range(1, len(fp) - 1):
            v = abs(float(np.dot(np.cross(fp[i] - g, fp[i + 1] - g), p0 - g))) / 6.0
            vtot += v
            csum += v * (g + p0 + fp[i] + fp[i + 1]) / 4.0
    return vtot, (csum / vtot if vtot > 0 else g)


def assemble_mesh_3d(coords, cells, tags, *, normal_distances=False, boundary_tags=None) -> Mesh:
    """Build a 3D Mesh (volumes, centroids, interior/boundary faces) from cells.

    coords : (n_nodes, 3) node coordinates. cells : list of node-index tuples,
    each a tetrahedron (4 nodes), prism/wedge (6) or hexahedron (8) in Gmsh
    ordering. tags : one integer per cell (material id). Two cells share a face
    when they share the same set of face nodes (conforming meshes only -- 3D
    hanging nodes are not split); a face touched by one cell is a boundary face.

    The returned Mesh carries the same two-point-flux quantities the 2D path
    does -- so `Mesh.area` holds the cell *volume* and each face its *area* and
    centroid-to-centroid (or centroid-to-face) distance -- and drives the exact
    same operator and solver.

    ``normal_distances=True`` supplies face-normal interior resistances and
    normal boundary distances. The default retains legacy centroid distances.
    ``boundary_tags`` optionally maps boundary face node tuples (any order) to
    integer physical labels. Faces must be planar; cells must be star-convex
    about their computed centroid.
    """
    coords = np.asarray(coords, dtype=float)
    cells = [tuple(c) for c in cells]
    nc = len(cells)
    centroid = np.zeros((nc, 3))
    volume = np.zeros(nc)
    face_cells = defaultdict(list)      # sorted global node key -> [(cell, global face nodes)]
    for c, ns in enumerate(cells):
        tmpl = _FACE_TEMPLATES.get(len(ns))
        if tmpl is None:
            raise ValueError(f"cell {c} has {len(ns)} nodes; expected a tet (4), "
                             "prism (6) or hex (8)")
        cell_pts = coords[list(ns)]
        volume[c], centroid[c] = _cell_volume_centroid_3d(cell_pts, tmpl)
        for fl in tmpl:
            gface = tuple(ns[k] for k in fl)
            face_cells[tuple(sorted(gface))].append((c, gface))

    def dist(a, b):
        return float(np.linalg.norm(centroid[a] - centroid[b]))

    faces, face_distances, bfaces = [], [], []
    face_centers, face_normals, bface_centers, bface_normals, bface_tags = [], [], [], [], []
    boundary_tags = _boundary_tag_map(boundary_tags)
    for face_nodes, lst in face_cells.items():
        gface = lst[0][1]
        A, fcent = _poly_area_centroid_3d(coords[list(gface)])
        fp = coords[list(gface)]
        normal = np.cross(fp[1] - fp[0], fp[2] - fp[0])
        norm = np.linalg.norm(normal)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("mesh face area must be finite and positive")
        normal /= norm
        owner = lst[0][0]
        if np.dot(normal, fcent - centroid[owner]) < 0:
            normal = -normal
        if len(lst) == 2:
            i, j = lst[0][0], lst[1][0]
            faces.append((i, j, A, dist(i, j)))
            face_centers.append(fcent)
            face_normals.append(normal)
            face_distances.append((abs(float(np.dot(centroid[i] - fcent, normal))),
                                   abs(float(np.dot(centroid[j] - fcent, normal)))))
        elif len(lst) == 1:
            i = lst[0][0]
            db = (abs(float(np.dot(centroid[i] - fcent, normal))) if normal_distances
                  else float(np.linalg.norm(centroid[i] - fcent)))
            bfaces.append((i, A, db))
            bface_centers.append(fcent)
            bface_normals.append(normal)
            bface_tags.append(boundary_tags.get(face_nodes, 0))
        else:
            raise ValueError(f"face {lst[0][1]} is shared by {len(lst)} cells "
                             "(non-manifold or non-conforming mesh)")
    return _validate_mesh_geometry(Mesh(coords=coords, cells=cells, cell_tag=np.asarray(tags),
                centroid=centroid, area=volume, faces=faces, bfaces=bfaces,
                face_distances=(np.asarray(face_distances).reshape(-1, 2)
                                if normal_distances else None),
                face_centers=np.asarray(face_centers).reshape(-1, 3),
                face_normals=np.asarray(face_normals).reshape(-1, 3),
                bface_centers=np.asarray(bface_centers).reshape(-1, 3),
                bface_normals=np.asarray(bface_normals).reshape(-1, 3),
                bface_tags=np.asarray(bface_tags, dtype=int)))


@dataclass
class MeshResult:
    k_eff: float
    flux: np.ndarray                # (G, n_cells)
    converged: bool
    outer_iterations: int
    solve_seconds: float
    device: str = "cpu (numpy)"
    inner_iterations: int = 0
    k_history: list = None


def _build_ell(face_i, face_j, n):
    """Row-wise (ELLPACK) adjacency from a symmetric face list.

    Each interior face (i, j) couples both endpoints, so it appears once in row
    i (neighbour j) and once in row j (neighbour i). The rows are padded to the
    maximum cell degree K and stored column-major, (K, n): ``nbr[s, c]`` is the
    s-th neighbour cell of c and ``fidx[s, c]`` the face feeding that slot (-1 on
    padding). Padding neighbours point at the cell itself with a zero weight, so
    they are inert. This lets the operator apply be a *gather* (read each
    neighbour's flux) with a coalesced write, instead of a *scatter* (atomic adds
    on the GPU) -- the same access pattern that makes the structured stencils
    fast on CUDA.
    """
    nf = face_i.size
    owner = np.concatenate([face_i, face_j])
    neigh = np.concatenate([face_j, face_i])
    fidx = np.concatenate([np.arange(nf), np.arange(nf)]).astype(np.int64)
    deg = np.bincount(owner, minlength=n)
    K = max(int(deg.max()) if nf else 0, 1)
    offset = np.zeros(n, dtype=np.int64)
    if n:
        offset[1:] = np.cumsum(deg)[:-1]
    order = np.argsort(owner, kind="stable")
    owner_s = owner[order]
    slot = np.arange(owner.size) - offset[owner_s]           # 0..deg-1 within each row
    nbr = np.tile(np.arange(n, dtype=np.int64), (K, 1))      # (K, n), self-padded
    fmap = np.full((K, n), -1, dtype=np.int64)
    if nf:
        nbr[slot, owner_s] = neigh[order]
        fmap[slot, owner_s] = fidx[order]
    return nbr, fmap


class _MeshGroupOperator:
    """Matrix-free within-group FV operator on an unstructured mesh.

    Exposes the same ``apply`` / ``inv_diag`` interface as the structured
    :class:`~garnet.operator.GroupOperator`, so it drops straight into the shared
    Jacobi/Neumann-PCG. The connectivity is an explicit ELLPACK adjacency
    (:func:`_build_ell`) rather than a shifted stencil, but the apply is still a
    pure *gather*: for every cell read its neighbours' flux and combine with the
    per-slot weights, ``A phi = diag*phi - sum_s w[s]*phi[nbr[s]]``. No scatter,
    so no GPU atomics and the write is coalesced. The assembled operator --
    volume-integrated resistance-weighted two-point-flux leakage, plus removal and the
    Robin boundary term on the diagonal -- is symmetric for equal face-side
    factors. Unequal scalar factors produce a conservative nonsymmetric matrix.
    """

    def __init__(self, xp, nbr, w_ell, diag):
        self.xp = xp
        self.nbr = nbr                # (K, n) int32 neighbour cell per slot
        self.w_ell = w_ell            # (K, n) coupling weight per slot (0 on padding)
        self.diag = diag              # (n,) assembled diagonal
        self.inv_diag = 1.0 / diag    # Jacobi preconditioner

    def apply(self, phi):
        gathered = phi[self.nbr]                      # (K, n) neighbour flux
        return self.diag * phi - (self.w_ell * gathered).sum(axis=0)


class UnstructuredDiffusionSolver:
    """Multigroup k-eigenvalue diffusion FV solver on an arbitrary Gmsh mesh.

    materials      : list of Material (all same group count, up- or down-scatter).
    cell_material  : (n_cells,) index into `materials` for each cell.
    alpha_boundary : Robin coefficient on every boundary face (0.5 = vacuum,
                     0 = reflective, or a custom albedo).
    device         : "auto" (GPU if present) | "gpu" | "cpu".
    precond_degree : Neumann-polynomial preconditioner degree for the inner CG
                     (0 = plain Jacobi, the default), as on the structured
                     solvers.
    face_df        : optional (G,n_interior_faces,2) corrected trace factors.
    boundary_df    : optional (G,n_boundary_faces) corrected trace factors.
    linear_solver  : None (CG for equal factors, BiCGSTAB for unequal factors),
                     "cg", "gmres", or "bicgstab".
    """

    def __init__(self, mesh: Mesh, materials, cell_material,
                 alpha_boundary=_VACUUM_ALPHA, device="auto",
                 precond_degree=0, dtype=np.float64, linear_solver=None,
                 face_df=None, boundary_df=None):
        self.xp = xp = get_backend(device)
        self.device = device_name(xp)
        self.dtype = dtype
        self.mesh = mesh
        self.mats = list(materials)
        self.cm = cm = np.asarray(cell_material)
        self.alpha = float(alpha_boundary)
        self.G = G = self.mats[0].n_groups
        n = mesh.n_cells
        use_df = face_df is not None or boundary_df is not None
        self.face_df = _mesh_df(face_df, (G, len(mesh.faces), 2), "face_df")
        self.boundary_df = _mesh_df(boundary_df, (G, len(mesh.bfaces)), "boundary_df")
        nonsymmetric = np.any(self.face_df[..., 0] != self.face_df[..., 1])
        self._linsolve = get_linear_solver(linear_solver or ("bicgstab" if nonsymmetric else "cg"))
        if nonsymmetric and self._linsolve is pcg:
            raise ValueError("CG cannot solve a nonsymmetric mesh diffusion DF operator; use bicgstab or gmres")
        df_grid = MeshGrid(mesh, self.alpha) if use_df else None

        area = np.asarray(mesh.area, dtype=dtype)
        D = [np.array([self.mats[m].diffusion[g] for m in cm], dtype=dtype) for g in range(G)]
        removal = [np.array([self.mats[m].removal[g] for m in cm], dtype=dtype) for g in range(G)]

        # Face and boundary connectivity as flat arrays (host; moved to device).
        fi = np.array([f[0] for f in mesh.faces], dtype=np.int64)
        fj = np.array([f[1] for f in mesh.faces], dtype=np.int64)
        fL = np.array([f[2] for f in mesh.faces], dtype=dtype)
        fd = np.array([f[3] for f in mesh.faces], dtype=dtype)
        face_distances = None
        if mesh.face_distances is not None:
            face_distances = np.asarray(mesh.face_distances, dtype=dtype)
            if face_distances.shape != (len(mesh.faces), 2):
                raise ValueError("face_distances must have shape (n_faces, 2)")
        bi = np.array([b[0] for b in mesh.bfaces], dtype=np.int64)
        bL = np.array([b[1] for b in mesh.bfaces], dtype=dtype)
        bd = np.array([b[2] for b in mesh.bfaces], dtype=dtype)

        # ELLPACK adjacency (geometry, shared across groups); moved to device once.
        nbr, fmap = _build_ell(fi, fj, n)
        nbr_dev = xp.asarray(nbr.astype(np.int32))
        fpad = np.where(fmap >= 0, fmap, 0)                    # safe index for padding

        self.ops = []
        for g in range(G):
            if use_df:
                op = MeshGroupOperator(xp, df_grid, xp.asarray(D[g]), xp.asarray(removal[g]),
                                       face_df=self.face_df[g], boundary_df=self.boundary_df[g])
                op._flat.symmetric = op.symmetric
                self.ops.append(op._flat)
                continue
            Dg = D[g]
            Di, Dj = Dg[fi], Dg[fj]
            if face_distances is None:
                face_w = 2.0 * Di * Dj / (Di + Dj) * fL / fd
            else:
                face_w = fL / (face_distances[:, 0] / Di + face_distances[:, 1] / Dj)
            # per-slot weights (0 where padded); diag from the same face weights.
            w_ell = np.append(face_w, 0.0)[np.where(fmap >= 0, fpad, len(face_w))].astype(dtype)
            diag = removal[g] * area
            diag = diag + np.bincount(fi, weights=face_w, minlength=n)
            diag = diag + np.bincount(fj, weights=face_w, minlength=n)
            if self.alpha != 0.0:
                bw = self.alpha * Dg[bi] * bL / (bd * self.alpha + Dg[bi])
                diag = diag + np.bincount(bi, weights=bw, minlength=n)
            self.ops.append(_MeshGroupOperator(
                xp, nbr_dev, xp.asarray(w_ell), xp.asarray(diag)))
        self.preconds = [neumann_preconditioner(op.apply, op.inv_diag,
                                                 int(precond_degree))
                         for op in self.ops]

        # Source data, device-resident.
        self.area = xp.asarray(area)
        self.nsf = [xp.asarray(np.array([self.mats[m].nu_sigma_f[g] for m in cm], dtype=dtype))
                    for g in range(G)]
        self.chi = [xp.asarray(np.array([self.mats[m].chi[g] for m in cm], dtype=dtype))
                    for g in range(G)]
        # scattering g'->g (both down- and up-scatter); lagged through the source.
        self.scat = {}
        for gf in range(G):
            for gt in range(G):
                if gt != gf:
                    col = np.array([self.mats[m].sigma_s[gf, gt] for m in cm], dtype=dtype)
                    if np.any(col):
                        self.scat[(gf, gt)] = xp.asarray(col)

    def solve(self, tol_k=1e-7, tol_source=1e-8, max_outer=1000,
              inner_rtol_floor=1e-11, anderson_depth=8) -> MeshResult:
        """Power iteration on the fission source, Anderson-accelerated.

        The fission-source map ``F -> (1/k) sum_g nuSigma_f,g A_g^-1 chi_g F`` is a
        fixed point whose plain power iteration converges at the dominance ratio;
        for a loosely-coupled core (ratio near 1) that is hundreds of outers.
        Anderson acceleration mixes a short history of source residuals (the same
        scheme used in the transient step and the SPH solve), collapsing the slow
        modes so the eigenvector converges in far fewer outers. ``anderson_depth``
        <= 1 recovers the plain power iteration.
        """
        xp, G = self.xp, self.G
        synchronize(xp)
        t0 = time.perf_counter()
        n = self.mesh.n_cells
        area = self.area
        phi = [xp.ones(n, dtype=self.dtype) for _ in range(G)]
        k = 1.0
        F = sum(self.nsf[g] * phi[g] for g in range(G))
        F = F * (n / float(xp.sum(F)))                    # normalize source, sum = n
        conv = False
        inner_total = 0
        src_err = 1.0
        k_hist = []
        hist = []                                         # (F_in, raw_iterate) for Anderson
        for outer in range(1, max_outer + 1):
            # Inner CG tolerance tracks the *outer residual* (src_err): loose early,
            # tight late. It must not be tied to tol_source -- doing so pinned the
            # inner solve loose whenever tol_source was set large (converge on k
            # only), so k stalled at the inner-solve noise floor and Anderson drifted.
            rtol = min(1e-3, max(0.1 * src_err, inner_rtol_floor))
            for g in range(G):
                q = self.chi[g] / k * F * area
                for (gf, gt), col in self.scat.items():
                    if gt == g:
                        q = q + col * phi[gf] * area
                phi[g], n_it = self._linsolve(self.ops[g].apply, q, phi[g],
                                              self.ops[g].inv_diag, xp,
                                              rtol=rtol, precond=self.preconds[g])
                inner_total += n_it

            Fn = sum(self.nsf[g] * phi[g] for g in range(G))
            total_new = float(xp.sum(Fn))
            k_new = k * total_new / n                     # F has sum n
            g_F = Fn * (n / total_new)                    # raw power iterate, sum = n
            r = g_F - F
            src_err = float(xp.sqrt(xp.sum(r * r) / xp.sum(g_F ** 2)))
            dk = abs(k_new - k)
            k = k_new
            k_hist.append(k)
            if dk < tol_k and src_err < tol_source and outer > 5:
                F = g_F
                conv = True
                break

            # Anderson update: F <- residual-minimizing mix of the recent iterates.
            F_new = g_F
            if anderson_depth > 1:
                hist.append((F, g_F))
                if len(hist) > anderson_depth:
                    hist.pop(0)
                if len(hist) >= 2:
                    res = [Gj - Sj for Sj, Gj in hist]
                    dres = [res[i] - res[-1] for i in range(len(res) - 1)]
                    m = len(dres)
                    A = np.array([[float(xp.sum(dres[i] * dres[j])) for j in range(m)]
                                  for i in range(m)])
                    b = np.array([-float(xp.sum(dres[i] * res[-1])) for i in range(m)])
                    A[np.diag_indices(m)] += 1e-12 * (np.trace(A) + 1e-300)
                    try:
                        gamma = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        gamma = None
                    if gamma is not None and np.all(np.abs(gamma) < 1e4):
                        for j in range(m):
                            F_new = F_new + float(gamma[j]) * (hist[j][1] - hist[-1][1])
            F = F_new * (n / float(xp.sum(F_new)))        # renormalize, sum = n

        synchronize(xp)
        flux = np.array([asnumpy(phi[g]) for g in range(G)])
        return MeshResult(k_eff=k, flux=flux, converged=conv, outer_iterations=outer,
                          solve_seconds=time.perf_counter() - t0, device=self.device,
                          inner_iterations=inner_total, k_history=k_hist)
