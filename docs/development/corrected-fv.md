# Corrected scalar finite volume (experimental opt-in)

`garnet.corrected_mesh.CorrectedMeshGroupOperator` provides a fixed linear,
conservative nonorthogonal correction for scalar diffusion. It accepts the
`MeshGroupOperator` constructor arguments and exposes `apply`, `diag`,
`inv_diag`, and volume-valued `rhs_weight`. It always advertises
`symmetric=False`; use GMRES or another nonsymmetric solver. Existing mesh
operators and solver selection are unchanged by importing this module.

```python
from garnet.corrected_mesh import CorrectedMeshGroupOperator
op = CorrectedMeshGroupOperator(xp, grid, D, removal)
```

The mesh must provide face centers and unit normals, including boundary
faces. Interior normals point from owner i to owner j; boundary normals point
outward. Signed projections of center-to-face vectors must be positive.
Legacy centroid distances are not used. Arbitrary dimension is supported by
the algebra, but the numerical qualification in the accompanying tests is 2D.
Nonunit scalar discontinuity factors and active masks are rejected.

## Reconstruction and currents

Each cell has a least-squares gradient, assembled once as a sparse linear
combination of cell values. For a neighbor across a face with normal n,
normal distance d_j on the neighbor side, and r_ij = x_j - x_i, the equation is

```
[r_ij + (D_i / D_j - 1) d_j n] dot grad_i = u_j - u_i.
```

This relation assumes a locally planar material interface, continuous scalar
trace and normal current, and a shared tangential trace gradient. It reproduces
piecewise affine fields satisfying those interface laws when the local patch
and boundary equations admit that field. It is an approximation for curved
fields, corners, and patches crossing several interfaces. Equations are
normalized by their vector lengths before least squares. A deficient patch
(relative singular value <= 1e-12) raises an error; there is no TPFA fallback
or pseudoinverse acceptance of a missing spatial direction. Thin layers with
insufficient independent neighbor/boundary information can therefore fail.

With t_i = x_f - x_i - d_i n and t_j = x_f - x_j + d_j n, the integrated
interior current from i to j is

```
J = A [(u_i + grad_i dot t_i) - (u_j + grad_j dot t_j)]
      / (d_i / D_i + d_j / D_j).
```

Exactly the same linear current enters row i positively and row j negatively.
The reaction is removal times cell volume. Construction stores a host CSR
matrix; runtime application uses padded sparse row gathers on NumPy/CuPy,
without device scatter or flux-dependent reconstruction. There are no limiters
or clipping. Positivity, a discrete maximum principle, or positive definiteness
are not guaranteed. `host_matrix` is available for small diagnostics;
`face_currents(u)` returns `(interior, boundary)` integrated current arrays in
mesh order, with outward boundary signs. The host matrix should be treated as
read-only: mutating it does not rebuild the device gathers.

## Scalar boundaries and moment integration

For the homogeneous scalar law J/A = alpha times surface trace, reconstruction
includes the equation

```
(D n + alpha (x_f - x_i)) dot grad_i = -alpha u_i.
```

Reflective faces use n dot grad_i = 0. The homogeneous Dirichlet limit uses
(x_f - x_i) dot grad_i = -u_i. Scalar trace elimination gives the outward
integrated current

```
J = A alpha D / (D + alpha d) * (u_i + grad_i dot t_i),
t_i = x_f - x_i - d n.
```

Reflective current is zero; the Dirichlet coefficient is A D/d. Existing
`MeshGrid` boundary spellings apply: `reflective`, `vacuum` (alpha = 1/2),
`zero-flux` (zero surface scalar value), or a nonnegative scalar alpha.
Nonhomogeneous boundary data are not currently exposed.

Coupled Marshak moment boundaries require coupled gradient boundary equations.
Replacing their vacuum boundary classification by reflective would impose an
incorrect reconstruction even if a separate moment loss block were added.
This operator alone is **not qualified for coupled moment solves**. Integration
must supply the appropriate reconstruction, including coupling between moments,
and assemble its matching boundary currents. The explicit extension hook is
`reconstruction_policy="supplied", gradient_stencils=...`: one dictionary per
cell maps cell indices to dimension-vector gradient coefficients. This is a
homogeneous scalar linear hook; it does not itself represent cross-moment
coefficients. Supplying these stencils replaces all automatic scalar boundary
reconstruction, and the caller owns their consistency. Unknown policies and
stencils supplied without the explicit `supplied` policy are rejected.

## Validation scope

`tests/test_corrected_mesh.py` checks linearity, integrated conservation,
reflective constants, TPFA agreement on orthogonal meshes for reflective,
Robin, vacuum and Dirichlet boundaries, piecewise affine discontinuous-D
interface currents away from boundary patches, a skew-interface gradient
oracle, rank rejection, and a sheared-grid smooth manufactured solution solved
with an independently formed dense matrix. The manufactured refinement checks
second-order solution error reduction and compares a small GMRES solve with
the dense solution. This is scalar evidence, not a multigroup eigenvalue,
sub-pin physics, GPU performance, or coupled moment qualification.


The shared mesh eigen drivers expose this operator through
`mesh_discretization="corrected"`. Their default Krylov solver is GMRES with
a fixed incomplete-LU preconditioner based on TPFA. Setup uses SciPy SuperLU
with drop tolerance 1e-6 and fill factor 20; GPU application uses resident
triangular factors and explicit permutations. A two-sweep TPFA Neumann option
is retained for comparisons, but failed on the fitted thin-gap pin. The separate
[order-1 moment operator](corrected-mesh-moments.md) implements genuine coupled
Marshak reconstruction; the scalar supplied-gradient hook is not that coupling.
Actual AMD checks of the synthetic two-group sheared problem agree with CPU
in both fused settings. These checks establish numerical backend consistency,
not reactor accuracy or performance.

## Nonconstant circular-interface manufactured solution

`tests/test_corrected_mesh_radial_mms.py` independently constructs conforming
polar triangle meshes with NumPy, without Gmsh. It solves
`-div(D grad u) = 1` with homogeneous outer Dirichlet data, using SciPy's sparse
direct solver on the corrected operator. The circular reference is
`u(r) = integral_r^R s/(2 D(s)) ds`, so `D grad u = -x/2`. Material interfaces
are at radii 0.4107 and 0.4186; the outer radius is 0.6309. Each material's
quadratic reference is averaged analytically over each triangle using its exact
second coordinate moment, rather than sampled at the centroid.

Both angular counts (48, 96, 192) and radial subdivisions in every material
double, giving 720, 2,976 and 12,096 cells. Area-weighted L2 errors and observed
orders from a single-thread NumPy/SciPy run are:

| Diffusion coefficients (inner/layer/outer) | L2, 720 cells | L2, 2,976 cells | L2, 12,096 cells | Orders |
| --- | --- | --- | --- | --- |
| 0.7 / 3 / 0.3 | 7.215923377e-4 | 1.770795268e-4 | 4.390385342e-5 | 2.0268, 2.0120 |
| 0.7 / 1800 / 0.3 | 7.185744807e-4 | 1.766877454e-4 | 4.384534355e-5 | 2.0239, 2.0107 |

The tests require error ratios above 3 on both refinements, check the direct
solution residual and face-by-face cell balance, and compare total conservative
outflow with the polygon area. The analytic current `x/2` has exactly the same
integrated polygon outflow. Numerical outflows at these levels are approximately
1.2468952349, 1.2495706616 and 1.2502400558.

This is a joint geometry/discretization refinement: circles are approximated
by inscribed polygons. The material-wise quadratic extensions have small trace
jumps on interface chords, and the circular reference has nonzero trace along
outer boundary chords. Both defects are O(angular spacing squared), as is the
geometric sagitta. At the outer boundary the maximum trace defects are
1.418851241e-3, 3.550929508e-4 and 8.879700922e-5; the test reports interface
trace defects and sagittas too. Thus the measured error includes geometry error
and does not isolate the stencil's error on an exact circular domain. This
nonconstant, high-contrast scalar MMS provides curved-interface convergence
evidence; it does not qualify reactor eigenvalues, transport agreement,
Serpent Gd/TRC cases, or coupled moments.
