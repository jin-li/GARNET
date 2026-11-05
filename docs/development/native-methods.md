# Choosing corrected mesh and native moment methods

These APIs support explicit numerical experiments with fitted finite-volume
meshes and native multigroup data. They remain opt-in. Manufactured fields,
independent matrix actions, current balance and true residual checks establish
specific software and algebraic properties. They do not qualify a native
estimator, a full-core prediction, high-fidelity transport accuracy, or a large
assembly's memory and convergence behavior.

## Select the operator before selecting acceleration

| Calculation | Entry point | Data and scope |
|---|---|---|
| Scalar corrected diffusion | `CorrectedMeshGroupOperator`, or `mesh_discretization="corrected"` in the mesh diffusion solver | Scalar diffusion/removal coefficients; fixed linear nonorthogonal reconstruction. |
| Per-group corrected SP3/SDP1 | `CorrectedMeshMomentOperator`, or the corrected mesh moment solver dispatch | Two coupled pseudo-moments with scalar material coefficients; distinct from native full group matrices. |
| Native full-matrix SP3/SDP1/SP5/SP7 | `CorrectedNativeMomentOperator` | Explicit family, immutable `MomentMaterial` per cell, full native group loss matrices and family-specific coupled boundary laws; CPU research path. |
| Angular scattering action | `ProductQuadrature` and `NativeAngularSource` | Full-sphere harmonic projection and native production matrices; no streaming or eigenvalue solve by themselves. |
| Angular leakage and collision | `NativeMeshStreaming` | Two-dimensional fitted-mesh upwind streaming with physical three-dimensional ordinate components; prescribed incoming boundary data. |
| Bounded angular fixed-source/eigenvalue iteration | `NativeMeshTransport` | Explicit vacuum or axis-aligned specular boundaries, cold iteration and independent true residual gates; CPU mathematical comparator. |
| Angular P0 acceleration | `P0GalerkinTransport` or `StreamingBlockP0Transport` | Explicit alternatives to the angular solver's default diagonal preconditioner, retaining its equations and acceptance checks. |
| Native P0 reactions in NEM | Explicit `native_total` and `native_scattering_production` in the NEM solvers | Direct native total-minus-production reaction matrix; experimental nodal diffusion input, not angular transport. |

The unstructured diffusion solver's existing **deferred** skew correction is
separate from the fixed linear corrected operators above. Supplying normals
and centres alone does not select the latter. For the corrected scalar and
per-group paths, use nonsymmetric solvers and follow the supported mask, factor,
geometry and boundary restrictions in the [scalar guide](corrected-fv.md) and
[order-1 moment guide](corrected-mesh-moments.md). Native full-matrix SP5/SP7 do
not extend the per-group order-1 dispatch; they use their own
[native operator API](corrected-native-moments.md), with
`CoupledMomentEigenSolver` as its eigenvalue driver.

## Establish geometry and material bindings

Start with the [face geometry contract](subpin-mesh-geometry.md) and
[polygon geometry guide](polygon-mesh.md). Record cell ownership, material tags,
face orientation, normal distances and boundary classes. A valid geometry
contract is a prerequisite for a solve, not evidence of spatial convergence.
Corrected reconstruction may reject deficient local patches rather than
silently reverting to an uncorrected approximation.

Read native data with `read_serpent_moment_xs` and bind each derived
`MomentMaterial` to the intended cells. Choose the matrix orientation explicitly,
preserve exact group boundaries, retain signed higher moments and reported
errors, and satisfy the orders required by the chosen family. Scalar material
adapters and standard SPN/SN constructors do not automatically select the native
full-matrix equations. See the [Serpent workflow](../user/serpent-mesh.md) and
[native operator guide](corrected-native-moments.md) for layouts and admission
checks.

## Reduce spatial or energy resolution deliberately

`homogenize_moment_materials` pools an explicitly chosen collection of native
materials using supplied **volume-integrated scalar flux** weights. It does
not choose regions or multiply those weights by volume again. Incoming-group
rows of the native matrices are weighted consistently; fission emission and
power use their respective source and direct power weights. Preserve the exact
record-to-weight binding and use the returned power coefficient after rebinding.
The [spatial homogenization guide](moment-homogenization.md) explains preserved
rates, unsupported collections and provenance.

`collapse_native_energy_coefficients` in
`garnet.moment_energy_homogenization` separately aggregates contiguous energy
groups using an explicit partition covering the original grid. It does not
pool spatial regions. Keep a fine-grid control and compare the quantities of
interest after each reduction. Scalar-flux weighting of higher moments does
not preserve general angular-moment reaction rates. Derived means carry no
new uncertainty or covariance estimate; neither reduction supplies physical
admission or replaces a qualified native group-constant calculation.

## Build an angular comparator in stages

First verify projection and scattering with the
[angular-source API](native-angular-source.md), then test leakage and boundary
currents with the [streaming API](native-mesh-streaming.md). Choose angular
resolution independently: quadrature exactness for harmonic products is not
angular convergence of a streaming problem. `NativeMeshTransport` adds the
[bounded iteration](native-mesh-transport.md); record both inner and outer
termination reasons and independently recomputed residuals. Reflection is
limited to Cartesian-axis normals. Signed native kernels are retained rather
than clipped, so convergence alone does not establish physical suitability.

The [P0 acceleration guide](native-angular-p0.md) describes a discrete
Galerkin projection of the angular loss, plus the optional streaming-block
correction. This P0 coarse space is distinct from the
[native NEM P0 reaction input](native-nem-p0.md). Neither is a qualification
shortcut between nodal diffusion and angular transport.

## Treat storage and preconditioning as separate experiments

The native corrected-moment driver supports explicit fixed ILU, energy-group
ILU and two-level coarse corrections. `storage="deferred"` selects compact
operator storage instead of the default assembled representation. Optional
cached coarse prolongation actions avoid selected repeated operator work.
These controls change storage or linear-solver work; the original native
operator, fission equation and true residual acceptance still define the
answer. Compare actions and converged states with an assembled small reference
before increasing the problem size. See the
[native solver and storage guide](corrected-native-moments.md).

Packed factor storage is also explicit: the native moment driver uses
`group_ilu_storage="packed_cpu"`, while the angular P0 implementations use
`factor_storage="packed"`. These CPU backends require the documented SciPy
version and private triangular-solve ABI; unsupported versions fail rather
than selecting another backend. Packing temporarily retains both factor
representations. Array payload accounting and allocation guards are not
process memory limits; retain external memory, thread and time budgets.
No compact, coarse or packed option implies that a complete assembly fits or
converges within a particular budget.

Native full-matrix corrected moments, angular iteration and the packed/coarse
controls described here are experimental CPU paths. Existing NumPy/CuPy
corrected scalar/per-group operators and earlier independent moment GPU paths
have their own device checks. Their evidence does not establish GPU support,
performance or accuracy for the new native CPU APIs. Select a method from its
operator and data contract, then validate spatial, angular and energy
resolution separately from solver acceleration.
