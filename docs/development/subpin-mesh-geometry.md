# Fitted mesh geometry and physical tags

`ndgpu.mesh.Mesh` supports optional face geometry for experimental fitted-mesh
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
from ndgpu.mesh import assemble_mesh

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
