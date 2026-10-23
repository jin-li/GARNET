# Fitted mesh geometry and physical tags

`garnet.mesh.Mesh` supports optional face geometry for experimental fitted-mesh
finite-volume development. Existing positional constructors remain valid; the
new fields default to `None` for manually constructed meshes. Both assembly
functions populate the fields:

| Field | Shape | Meaning |
| --- | --- | --- |
| `face_centers` | `(n_faces, dim)` | Interior face midpoint (2D) or area centroid (3D) |
| `face_normals` | `(n_faces, dim)` | Unit normal from `faces[k][0]` toward `faces[k][1]` |
| `bface_centers` | `(n_bfaces, dim)` | Boundary face midpoint or area centroid |
| `bface_normals` | `(n_bfaces, dim)` | Unit normal outward from the owning cell |
| `bface_tags` | `(n_bfaces,)` | Integer physical tag, with zero for an untagged face |

Centers and distances use the length units of the input coordinates. Normals
are dimensionless; face measures are lengths in 2D and areas in 3D. Metadata
follows the existing face-list ordering. A split 2:1 interface has one center,
normal, measure, and distance pair per fine half-edge. A shared normal belongs
to the first owner; the second owner's outward normal is its negative.

`read_gmsh(path, normal_distances=None)` accepts Gmsh 2.2 ASCII linear
triangles/quads or tetrahedra/hexahedra/prisms. It maps arbitrary node IDs into
zero-based coordinate indices, honors each element's declared tag count, and
preserves the first physical tag as `cell_tag`. Physical line elements in 2D
and surface elements in 3D label boundary faces by matching their node sets.
An element with no tags receives physical tag zero. Conflicting duplicate
boundary labels raise `ValueError`. Interior surface labels do not become
boundary faces or boundary conditions.

The default distance behavior is unchanged: 2D supplies projected normal
interior distances and boundary distances; 3D retains centroid separation and
Euclidean boundary distances. Pass `normal_distances=True` to opt into projected
3D distances, or `False` for legacy 2D distances. This changes geometry consumed
by TPFA, without adding a nonorthogonal flux correction.

```python
from garnet.mesh import assemble_mesh

mesh = assemble_mesh(
    [(0, 0), (1, 0), (1, 1), (0, 1)],
    [(0, 1, 2, 3)], [17], boundary_tags={(0, 1): 31},
)
assert mesh.cell_tag.tolist() == [17]
assert mesh.bface_tags.tolist() == [31, 0, 0, 0]
```

Assemblers reject nonpositive cell/face measures or distances, non-manifold
faces, and interior faces whose centroids are not separated by the face plane.
3D faces must be planar, and cells must be star-convex about their computed
centroid. These checks are conservative prerequisites, not a general mesh
quality certification. Existing 2D vertex-mean cell centers are retained.
Only the existing 2D midpoint 2:1 hanging-node split is supported; 3D hanging
faces, curved/higher-order Gmsh elements, and Gmsh 4.x are unsupported. Geometry
metadata and physical labels do not themselves assign solver boundary classes.

## Extruded 3D transmission manufactured solution

`tests/test_corrected_mesh_extruded_mms.py` independently constructs conforming
triangular prisms without Gmsh and exercises the corrected scalar operator in
three dimensions. Polar triangles have material interfaces at radii 0.4107
and 0.4186, outer radius `R = 0.6309`, and diffusion coefficients
`D = [0.7,1800,0.3]`. Extrusion spans `[-H,H]` with `H = R` and homogeneous
Dirichlet conditions on the cylindrical side approximation and both end caps.

The circular reference is
`u(r,z) = u_rad(r) (1-z^2/H^2)`, where
`u_rad(r) = integral_r^R s/(2D(s)) ds`. Its independently specified source is
`q(r,z) = (1-z^2/H^2) + 2D u_rad(r)/H^2`.
Triangle radial quadratic averages use exact second coordinate moments; slab
averages use `mean(z^2) = (z_lo^2 + z_lo*z_hi + z_hi^2)/3`. The prism solution
average is the product of those independent averages, and the source integral
uses the corresponding exact means. Neither exact values nor source generation
uses the discrete operator action.

The angular counts are 24, 48 and 96; radial subdivisions in the three materials
are `[2,1,1]`, `[4,2,2]` and `[8,4,4]`; axial slab counts are 2, 4 and 8.
The 3D assembler's prism volumes are checked independently against triangle
area times slab width and the analytic polygon-cylinder volume. Tests also
check unit normals and positive signed face distances in both owner directions.
SciPy's sparse direct solver supplies the independent linear solve.

| Prisms | Volume-weighted L2 error | Error ratio | Observed order |
| --- | --- | --- | --- |
| 336 | 5.763041818e-2 | — | — |
| 2,880 | 1.519225150e-2 | 3.79341 | 1.92349 |
| 23,808 | 3.865609960e-3 | 3.93010 | 1.97457 |

Local conservative face balances and direct residuals pass. Numerical total
outflows are 44.7844626494, 45.1675601851 and 45.2636321416, matching the
independently integrated volume sources. The analytic vector current is
`J_xy = (x,y)(1-z^2/H^2)/2`, `J_z = 2D u_rad z/H^2`.
Its polygon side and two-cap integrated outflows are respectively
`polygon_area*4H/3` and `4/H * sum_triangle(area*D*mean(u_rad))`.
Numerical side outflows are 1.405095214, 1.142639684 and 1.074881829,
approaching analytic values 1.039914887, 1.048888272 and 1.051138841.
Cap outflows are 43.37936744, 44.02492050 and 44.18875031, approaching
43.74454776, 44.11867191 and 44.21249330. Solution and separate side/cap
outflow errors must each decrease by a factor greater than 3 per refinement.

This remains a joint geometry/discretization test: cylindrical interfaces are
inscribed vertical polygon faces, with O(angular spacing squared) trace jumps
between the material-wise quadratic extensions. Their normal currents agree.
The circular reference also has a nonzero side trace, bounded by
`R^2 sin(pi/n_angles)^2/(4D_outer)`, while end-cap traces vanish exactly.
The test reports that boundary defect and the outer polygon sagitta. It does
not assert that the circular reference exactly solves the polygonal problem.

Single-thread NumPy/SciPy runs took 21–28 seconds overall, with finest-level
times of 17.66–24.31 seconds and peak process RSS about 442 MiB. This small scalar
prism MMS provides a 3D geometry/operator verification case; it does not advance
reactor geometry, eigenvalue, coupled-moment, transport-reference or production
qualification.
