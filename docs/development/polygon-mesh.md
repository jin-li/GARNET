# Opt-in polygon mesh geometry

`garnet.polygon_mesh` provides explicit polygon geometry for a caller preparing
a conforming finite-volume mesh. It does not change `assemble_mesh`, select a
solver, construct a core, or change defaults.

```python
from garnet.polygon_mesh import assemble_polygon_mesh, polygon_mesh_audit

mesh = assemble_polygon_mesh(
    coords,                       # finite float64-compatible (n_vertices, 2)
    cells,                        # iterable of integer vertex-index sequences
    tags,                         # one integer material/region tag per cell
    boundary_tags=boundary_tags,   # {(vertex_a, vertex_b): integer physical tag}
)
audit = polygon_mesh_audit(mesh)
```

Every cell must have at least three distinct indices, counterclockwise boundary
order, positive area, and a strictly positive fan about its **true area
centroid**. The fan angles must wind once around that centroid. This admits
concave polygons whose area centroids lie strictly in their kernels; it rejects
non-star polygons, centroid-on-edge cases, reversed orientation, and
self-intersecting radial traversals. Collinear consecutive boundary edges are
allowed when the centroid fan remains strictly positive. Coordinates must be
globally unique, including unused coordinates. Indices must be in range.

The constructor uses local-origin shoelace sums and first moments, rather than
vertex averages. An edge has one boundary owner or exactly two owners with
opposite orientation. Each exposed edge needs exactly one explicit physical
tag; zero is a valid explicit tag. Interior and absent edge tags are errors.
Normal vectors point outward from the first owner, and both interior side
distances must be strictly positive with their original signs. The returned
`Mesh` includes face centers, normals, normal distances, and boundary tags.
The ordinary TPFA discretization still has its usual nonorthogonality error.

Coordinates are never rounded, welded, reordered, or repaired. A caller that
wants coordinate canonicalization must implement and audit that operation
separately, then rewrite the connectivity and physical edge labels explicitly.
There is no automatic 2:1 hanging-edge split. Missing labels on a hanging
interface therefore cause rejection. This API cannot distinguish a deliberately
labeled crack from a caller incorrectly labeling every hanging edge as exposed;
the caller owns that physical classification. It also does not perform a
global overlap search between cells with disjoint vertex IDs.

Geometry computations are segmented array operations over all cell edges.
Coordinate uniqueness and edge grouping use sorting, giving approximately
`O(V log V + E log E)` work and `O(V + E)` storage. There is no pairwise edge or
cell search. Input and returned cell/face collections retain the existing
`Mesh` Python-list interface, so production memory planning must include Python
objects and temporary sorting arrays. Million-cell peak memory and timing have
not been benchmarked by this change. A CSR-specific interface is not provided.

`polygon_mesh_audit(mesh)` returns counts, total area and first moment, minimum
cell area, and minimum normal distance. This is a geometry summary for meshes
constructed by this module, rather than an independent topology validator.

## Halfplane clipping

```python
from garnet.polygon_mesh import clip_star_polygon

pieces = clip_star_polygon(parent_vertices)
# Default region: y >= 0 and x - y >= 0.
# A plane (a, b, c) means a*x + b*y >= c:
pieces = clip_star_polygon(parent_vertices, halfplanes=[(1, 0, 1)])
# Optional explicit interior kernel point changes triangulation only:
pieces = clip_star_polygon(parent_vertices, fan_center=caller_chosen_point)
for piece in pieces:
    use(piece.vertices, piece.area, piece.first_moment)
```

Input polygon validation is the same centroid-fan/winding check as assembly.
A wholly retained polygon remains one polygon, including when concave.
A crossing polygon is triangulated about its area centroid. Each convex fan
triangle is intersected with all supplied halfplanes; every certified positive
piece is returned separately. Their interiors are disjoint. This preserves all
components of a disconnected intersection. It does not clip an arbitrary
concave polygon as though its intersection were one simple polygon.

An optional `fan_center=(x, y)` replaces the default area centroid for fan
triangulation. It must be finite and strictly inside the polygon kernel, with
a positive fan and one angular winding. This changes no polygon boundary,
area, or first moment. A caller may explicitly choose a point on a symmetry
plane to avoid tiny fan fragments when the true centroid lies just off that
plane. The caller owns any projection policy and displacement audit; the helper
does not project the point. Original polygon validation still requires a valid
area-centroid fan. The real-template cell-10861 regression demonstrates both
the original tiny fragments and an explicit projected fan center with identical
retained area and first moments. Its center projection displacement is about
`1.018e-10` in the source coordinate units, so a strict `1e-10` caller bound
would reject that projection.

Each `ClipPiece` holds its vertex coordinates, area, and first moment
`integral (x, y) dA`. The parent identity and material tag remain caller-owned;
callers should retain those alongside **all** returned pieces. Clip pieces can
share fan edges and constructed coordinates. Before assembling multiple pieces,
the caller must explicitly establish common vertex IDs and conforming edge
subdivision, and classify the resulting boundary edges. The helper does not
claim that independently clipped pieces automatically form a conforming mesh.

## Numerical policy

The mesh uses float64. Its area and fan signs must exceed conservative local
arithmetic bounds proportional to machine epsilon and the absolute product
sums. Area bounds also grow with the polygon vertex count. No fixed absolute
area cutoff is used: tiny, well-resolved polygons are valid. Area moments are
computed after subtracting a vertex origin, limiting cancellation from large
global offsets. The rounded, representable centroid must itself pass the fan
test; translating a tiny cell beyond coordinate resolution can cause rejection.

Clipping evaluates planes and constructs intersections in NumPy `longdouble`.
The returned vertices, areas, and first moments retain that dtype; precision is
platform-dependent. No input vertex is snapped or shifted. Only new edge-plane
intersections have one coordinate solved from the plane equation. Only exact
adjacent duplicate vertices are removed. There is no proximity-based merge and
no discard threshold for positive-area slivers.

A nonzero plane residual within `16 * eps(longdouble)` times the absolute
evaluation terms is ambiguous and raises `ValueError`. Apparent zero residuals
are checked with exact rational arithmetic on their binary floating-point
values to reject cancellations that merely rounded to zero. Positive clipped
areas must clear a local product-sum bound of
`32 * eps(dtype) * vertex_count`; unresolved areas raise rather than disappear.
Certified point/line contacts return no positive piece. These are conservative
floating-point checks, not a general exact geometric predicate library or a
global guarantee against every adversarial roundoff configuration. Callers
must resolve rejected near-plane cases explicitly; this module does not invent
a snapping tolerance. Casting outputs to float64 is a separate caller action
that can collapse a valid extended-precision sliver and requires revalidation.

The focused CPU tests exercise translated skew-quad area centroids, signed
two-cell face geometry and conservative flux, invalid geometry/connectivity,
exhaustive boundary classification, disconnected concave-star clipping,
analytic wedge moments, exact contacts, and retained areas of order `1e-25`.
