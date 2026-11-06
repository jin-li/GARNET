# Serpent data and common finite-volume meshes

## Select a fixed-state scalar record

This template requires your own existing Serpent result file. It does not start Serpent:

```python
from garnet import DiffusionEigenSolver, Grid
from garnet.serpent_xs import read_serpent_xs
record = read_serpent_xs("assembly_res.m", universe="0")
material = record.to_diffusion_material(diffusion="cmm")
solver = DiffusionEigenSolver(Grid((1, 1, 1), (1.0, 1.0, 1.0)),
                             material, bc="reflective", device="cpu")
result = solver.solve(tol_k=1e-10, tol_source=1e-9)
print(record.source_sha256, result.k_eff)
```

Use `read_serpent_results` to inventory multiple records, then select a unique universe or zero-based record index. Preserve the exact energy boundaries. `diffusion="inf"` is an explicit alternative field choice; missing requested data should remain an error. The reflective single cell is a zero-leakage balance test, not a finite assembly model or a delayed point-kinetics calculation.

The scalar adapter uses reduced absorption and production scattering. Retained collision fields have different meanings. Read the [scalar import contract](../serpent_xs.md) before comparing balances or interpreting uncertainties. Fixed-state reading does not interpolate temperature, density, burnup or B1 leakage.

## Inspect native ADF tallies

For a selected record with ADF output, the reader keeps native face order and returns surface arrays as `(face,group)`:

```python
surface_flux = record.adf_estimate("DF_HET_SURF_FLUX")
net_inward = record.adf_estimate("DF_SURF_NET_CURR")
print(record.n_surfaces, surface_flux.mean, net_inward.mean)
```

This block continues the data-dependent scalar-record template. Surface flux is a surface average; native partial/net currents are integrated tallies, with net inward equal to incoming minus outgoing. The reader does not divide by area or reorder faces. Establish the native-to-receiver side map explicitly, and convert integrated current to current density only at the interface that requires it. Printed `DF_SURF_DF` is not automatically qualified for a different receiver or production kernel. Reference reconstruction and whole-face factor qualification remain separate operations.

## Native moments are opt-in

The separate reader requires orders and orientation explicitly:

```python
from garnet.serpent_xs import read_serpent_moment_xs
native = read_serpent_moment_xs("assembly_res.m", universe="0",
                                required_orders=(0, 1, 2, 3))
moments = native.to_moment_material(orientation="from-major")
print(moments.loss[1])
```

This template needs a file containing the required native fields. Choose `from-major` only when that flattening contract has been independently established. Signed higher moments and native reported error metadata remain unchanged. This object feeds the specialized coupled-moment path; standard `Material` SPN/SN constructors do not automatically consume it. See the [methodology](../methodology/transport.md) for full-matrix, hybrid GPU and $B\to0$ condensation limits.

For native full-matrix SP3/SP5/SP7, use the separate experimental CPU operator
and its explicit family and per-cell material bindings. Follow the
[method selection guide](../development/native-methods.md) before selecting
spatial/energy homogenization or an angular comparator. It distinguishes
source-only, streaming and bounded transport APIs, and explains optional
compact/coarse/packed controls. None of these choices automatically qualifies
native data or transfers earlier GPU evidence to a new CPU path.

## Native angular finite volume in 2D and 3D

For explicit material regions, bind a `NativeAngularSource` to
`NativeMeshStreaming`, then use `NativeMeshTransport` or its documented fixed
preconditioners. This CPU research path retains the supplied total and native
scattering-production moments through P7. Each regional fission source uses
the supplied cross-group `chi[to]*nuSigmaF[from]` without spectrum normalization.

The streaming mesh may be 2D or 3D. The existing Gmsh loader and
`assemble_mesh_3d` accept conforming tetrahedra, triangular prisms and hexahedra.
Cell measures are volumes in 3D; each face must have its area, center and unit
normal. Boundary conditions are explicit per face: vacuum or Cartesian-axis
specular reflection, including z faces. Oblique specular reflection is unsupported.

In 3D choose at least two polar and four azimuthal quadrature nodes explicitly
for P0; its default quadrature cannot carry axial leakage. The default P7
quadrature already meets this coverage requirement. Coverage alone does not
establish angular convergence. Start with a small mesh and budget the full
cell-by-angle-by-group fields and sparse preconditioner before extending it.
Software verification does not establish physical 3D reactor accuracy.

See the [streaming contract](../development/native-mesh-streaming.md),
[iteration controls](../development/native-mesh-transport.md) and
[preconditioners](../development/native-angular-p0.md).

## Build a nonuniform masked mesh

The following example needs no external data:

```python
import numpy as np
from garnet import Material, MeshModel
from garnet.nodal_mesh import build_rectilinear_mesh_3d
active = np.ones((1, 2, 2), dtype=bool)  # nz, ny, nx
active[0, 1, 1] = False
mesh, topology = build_rectilinear_mesh_3d(
    [0.0, 10.0, 25.0], [0.0, 12.0, 20.0], [0.0, 5.0],
    active_mask=active, tags=np.zeros(active.shape, dtype=int))
fuel = Material(name="synthetic", diffusion=[1.2], sigma_a=[0.025],
                nu_sigma_f=[0.030], sigma_s=[[0.0]], chi=[1.0])
model = MeshModel(mesh).fill(fuel).set_boundary("reflective")
print(model.run(device="cpu"))
print(topology["grid_indices"])
```

The topology records exposed masked faces separately from outer-box faces; this builder example deliberately makes both reflective. If your model requires different exposure conditions, assign them through the appropriate low-level mesh boundary interface. IDs follow active `(z,y,x)` C order, while centroid and width components are `(x,y,z)`. Tags are labels, not automatically material indices.

The vertex-free result supports finite-volume connectivity, not vertex export. `MeshModel` is a diffusion builder; low-level `MeshGrid` moment solvers have their own data and factor contracts. For general Gmsh input see [unstructured mesh](https://github.com/jin-li/GARNET/blob/main/examples/unstructured_mesh.py). The synthetic [common mesh example](https://github.com/jin-li/GARNET/blob/main/examples/isonees_features/common_mesh_fv.py) compares the integrated common-mesh pathway.

## Choose geometry and skew correction explicitly

For polygon meshes, `assemble_mesh(..., normal_distances=True)` is the default.
It supplies normal distances on both sides of interior faces, plus interior
and boundary face normals and centres. `UnstructuredDiffusionSolver` uses
this geometry for its default deferred nonorthogonal correction: reconstructed
cell gradients correct the tangential part of the face leakage. On an
orthogonal mesh that contribution vanishes. The three-dimensional
`assemble_mesh_3d` retains `normal_distances=False` by default; opt into `True`
to provide this geometry for the unstructured diffusion solver. Gmsh imports
already select normal geometry for three-dimensional meshes.

`normal_distances=False` preserves the historical centroid-distance two-point
approximation, including Euclidean centroid-to-boundary-midpoint distances,
and omits correction geometry. Selecting normal distances while disabling
`nonorthogonal_correction` also leaves a two-point approximation; normal
distances alone do not establish convergence on a skew mesh.

With normal geometry and an active skew correction,
`UnstructuredDiffusionSolver` raises an error if `face_df` or `boundary_df`
contains non-unit factors. A corrected gradient law for those discontinuous
traces is not implemented. Pass `nonorthogonal_correction=False` to that
low-level solver only when deliberately choosing the conservative two-point
factor approximation, then assess its mesh error. Factors remain usable on
orthogonal meshes, where the correction is zero. See the
[spatial methodology](../methodology/spatial-methods.md) for the discretization
and evidence limits. This diffusion option does not qualify the native moment
transport pathway or add the correction to its separate `MeshGrid` operators.
Before using regional data, inspect energy-group flux support and scattering
transfer errors as well as the declared balance check. A balance pass can
reflect a wide statistical tolerance. Printed zero errors do not prove
exactness, and a 100% relative error does not identify a particle-event count.
For sparse Serpent 2.2.2 group-constant batches, increasing particles per batch
may need investigation separately from adding cycles. Retain native errors,
source histories and correction provenance; the adapter does not replace the
study's statistical or physical qualification.

Gmsh 2.2 ASCII imports can retain physical material and boundary tags. See the
[fitted mesh geometry contract](../development/subpin-mesh-geometry.md). Map each
boundary tag explicitly to a supported boundary condition; a physical label
alone does not impose a reflective or vacuum boundary. Face geometry metadata
does not add nonorthogonal correction to the existing two-point operator.

Two-dimensional cell areas are evaluated in local coordinates to avoid
subtracting large global-coordinate products for thin fitted cells. Check
material-region areas against the intended geometry separately; this numerical
stabilization does not remove polygonal approximation of curved interfaces.


## Corrected fitted-mesh solves (experimental)

Pass `mesh_discretization="corrected"` to `MeshDiffusionEigenSolver`,
`MeshSP3EigenSolver`, or `MeshSDP1EigenSolver` to use the fixed linear
nonorthogonal reconstruction. The solver selects GMRES and, by default, a fixed TPFA incomplete-LU
preconditioner, with CPU factorization and backend-resident application. CG and moment symmetrization are unsupported.
For moment solvers use `marshak_vacuum=True` to select coupled vacuum traces.
Existing default solves retain the two-point operator. The corrected path
requires oriented face geometry, a full-rank reconstruction patch, and unity
discontinuity factors; higher-order mesh moment models are not supported.

Use `tol_k=1e-6`, `tol_source=1e-5`, `tol_balance=1e-5`,
`tol_fixed_source=1e-6`, and `inner_rtol_floor=1e-7` for explicit
production-like iteration criteria. The separate fixed-source tolerance
otherwise retains its stricter default. Independent equation/current balance and spatial
convergence remain necessary before interpreting a reactor comparison.

The corrected path accepts `mesh_preconditioner="tpfa_ilu"` (default),
`"corrected_ilu"`, `"corrected_block_ilu"`, or `"tpfa_neumann"` (two fixed Jacobi sweeps).
`"corrected_ilu"` factors the actual corrected FV matrix instead of its TPFA
approximation. It changes neither the equations nor the stopping criteria;
convergence still requires the true residual checks. Small CPU tests cover
scalar and moment operators; this option does not establish reactor accuracy
or GPU validation. Its setup and factor memory can exceed the TPFA option.
For order-1 SP3/SDP1, `"corrected_block_ilu"` instead factors the two corrected
diagonal moment blocks sequentially. In the actual U-form state `[U0, U1]`
(scalar flux `phi0 = U0 - 2/3 U1`), its fixed linear application is
`z0 = ILU(A00)^-1 r0`, then `z1 = ILU(A11)^-1 (r1 - A10 z0)`.
It retains the full sparse corrected `A10`, including off-cell Marshak terms,
and adds no scaling or volume weighting. Scalar and other moment orders are
rejected. Both blocks use `drop_tol=1e-6`, `fill_factor=20`; singular factors
fail explicitly without clipping or fallback. CPU setup avoids extracting
explicit L/U copies. Each callable exposes `group`, `shape`, `setup_seconds`,
`matrix_nnz`, `nnz`, `nnz_kind`, `coupling_nnz`, and `block_stats` with per-block
timings and counts. Factor counts are SuperLU-reported counts rather than
explicit CSC counts. Small CPU tests cover reflective and coupled Marshak
boundaries; large-model memory/convergence and GPU execution remain unverified.
ILU uses SciPy setup; on CuPy its
triangular factors and permutations stay on the GPU. Factorization/backend
failures propagate. The weaker polynomial option may fail on thin high-contrast
material layers. Measure construction/factor memory before a large model.
