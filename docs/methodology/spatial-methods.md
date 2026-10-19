# Spatial discretization and equivalence

## Finite-volume diffusion

Integrating balance over a cell gives reaction terms weighted by volume and leakage through oriented faces. For two orthogonal neighboring cells with face area $S$, center-to-face distances $d_i,d_j$ and diffusion coefficients $D_i,D_j$, the conductance is

```{math}
G_{ij}=\frac{S}{d_i/D_i+d_j/D_j},\qquad
I_{i\to j}=G_{ij}(\phi_i-\phi_j).
```

The same face current enters the two cell equations with opposite signs. This is the discrete conservation property. Equal cell widths reduce this to harmonic-mean diffusion. Nonuniform cells require both half distances; replacing them by a guessed center distance can preserve solver convergence while changing the problem.

Cartesian operators use nearest-neighbor stencils. Cylindrical r-z operators incorporate radial areas and cell volumes, then use volume weighting when symmetry is required. Triangular operators use their explicit neighbor topology and prism extrusion in z. Unstructured finite volumes use mesh cell volumes, face areas and distances. Two-point flux accuracy depends on mesh orthogonality; arbitrary connectivity alone does not establish accuracy on strongly skewed cells.

The two-dimensional polygon assembler accepts `normal_distances=True` by
default. Explicit `normal_distances=False` retains the historical harmonic
coefficient/centroid-separation interior formula and centroid-to-boundary-
midpoint distances. The locally refined HPMR benchmark selects this historical
mode for both refined and uniform references, preserving its established
discretization. Neither mode includes a general nonorthogonal flux correction;
its refinement evidence is not a proof of consistency on arbitrary skew meshes.
The three-dimensional assembler already exposes the same explicit choice,
with its existing default `False`.


`build_rectilinear_mesh_3d` constructs a vertex-free common finite-volume mesh from strictly increasing x/y/z edges and a boolean mask. The input arrays are `(nz, ny, nx)`; active cell IDs follow C order. Centroids and widths are in x/y/z order. Mask exposures and outer-box exposures are distinct metadata, and callers choose their boundary conditions. This mesh has no vertex representation for vertex-based export.

## Discontinuity factors

A whole-face factor relates a homogenized surface value to its reference heterogeneous value. Interface continuity is imposed on the **factor-weighted trace**, while the normal current remains conservative. In NEM notation,

```{math}
f_{i,+}\phi_{i,+}^{\rm hom}=f_{j,-}\phi_{j,-}^{\rm hom}.
```

Factors are positive node-side/group data, not a material-independent correction. Their orientation, reference spectrum, production kernel, geometry, side ordering and normalization belong to their provenance. NEM expects `(N,6,G)` in x−, y−, x+, y+, z−, z+ order and requires exterior factors to be unity.

A whole-face average preserves one scalar trace per group and face. It cannot impose a subface distribution or face moments. Mesh moment factors and native face-trace developments use distinct representations; do not equate a refined submesh calculation with a whole-face ADF receiver. Transfer requires an explicit derivation and independent convergence evidence.

## SPH

Superhomogenization adjusts homogenized constants until selected regional reaction-rate or flux constraints match a reference calculation. `sph_correct`, `sph_correct_monolithic` and `sph_get_correct` provide explicit reference/coarse iterations; `flux_weighted_homogenize` and `region_average` support the reduction. A successful correction reproduces the declared target and state. It does not establish prediction after temperature, control geometry, energy partition or transport closure changes.

Record the reference solver, target normalization, region map, residual history and corrected materials. `TriReactor.with_materials` retains geometry while replacing materials in the same count/order. This hand-off is appropriate only when the corrected constants were generated for that geometry and method.

## Cuboid nodal expansion

NEM integrates the three-dimensional diffusion equation transversely to form one-dimensional equations with transverse leakage. Within a node, the transverse-integrated flux uses a Legendre polynomial expansion. The base implementation is quartic; the higher-order receiver supports degrees 4, 6, 8, 10, 12, 14 and 16. The higher-order model retains degree-minus-two weak source moments per axis. These are **spatial** moments, distinct from angular SPN or scattering moments.

For normalized local coordinate $\xi\in[-1,1]$, write $\phi_g(\xi)=\bar\phi_g+\sum_{n=1}^{p}a_{g,n}P_n(\xi)$. Endpoint traces follow $P_n(\pm1)=(\pm1)^n$ and currents follow the differentiated expansion. Weak balance tests and the two endpoint-current constraints determine local coefficients. Quadratic transverse leakage is formed from neighboring interval means with the selected boundary interpolation policy; it is not a pointwise finite-difference sample.

Local coefficient elimination leaves node means and shared face currents in a sparse condensed system. The transverse leakage closure remains explicitly prescribed; raising polynomial degree does not eliminate that approximation. `CuboidNodalGrid` records the boundary leakage policy. Neighbor/current sign conventions and nonuniform widths enter the local responses and must survive adjoint construction.

Condition estimates and scaled solves gate local and global responses. Negative means, traces and profile minima remain visible diagnostics rather than being clipped. Higher order can expose oscillation or ill-conditioning. Accepting a case requires residual, positivity and order-convergence evidence, not simply the largest available degree.

The NEM reference reconstruction uses the receiver's exact $R-C/k$, polynomial degree, geometry and leakage policy. Prescribed reference currents need not obey the receiving core's exterior Robin condition. An auxiliary mean source can close local reference balance; it is excluded from the core production and physical population. The reflector-inclusive factor qualification compares three increasing orders and their face changes, with additional profile/factor checks. Passing this local gate does not validate the receiving core.
