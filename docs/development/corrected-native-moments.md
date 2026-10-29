# Experimental native full-matrix corrected moments

`garnet.corrected_native_moments.CorrectedNativeMomentOperator` is an explicit,
CPU research implementation for SP3, SDP1 and SP5 on a `MeshGrid`. It is separate from the
legacy per-group corrected scalar/moment dispatch. It makes no reactor
scalability, angular-transport consistency, or physical-accuracy claim. No
assembly calculation or native-data solve is part of this implementation.

Construct with one immutable `MomentMaterial` reference per cell and an explicit
`family='sp3'`, `'sdp1'` or `'sp5'`. Boundaries come from the grid (or the optional `bc`
argument), and must be homogeneous `vacuum` or `reflective`. No active mask,
discontinuity factor, fitted boundary source or GPU path is provided. States have native **cell-major** shape `(n_cells, M, n_groups)` or
its flattened equivalent. SP3/SDP1 use `M=2`, with
`U=(phi0+2*phi2, 3*phi2)` and scalar flux `U0-2*U1/3`. SP5 uses `M=3`, with
`U=(phi0+2*phi2, 3*phi2+4*phi4, 5*phi4)` and scalar flux
`U0-2*U1/3+8*U2/15`. Signed higher moments are retained.
`action`/`apply` preserve input shape; `face_currents` returns integrated
currents shaped `(n_faces, M, n_groups)` and `(n_boundary_faces, M, n_groups)`,
directed i-to-j and outward respectively. `gradients` returns
`(n_cells, M*n_groups, dimension)`. `n_moments` and `required_orders` expose the
selected layout and native loss orders; `C` contains all reaction coefficient
matrices, while `c1`/`c2` retain their existing meanings.

The `volumes` accessor supplies frozen cell measures for the native CPU eigen
driver, since `MeshGrid` stores them as `grid.mesh.area`. `assembled` is host
SciPy CSR. `action` evaluates gradients, traces, integrated currents and reaction
sums directly, independently of that CSR. Corrupting CSR is therefore detected
by the eigen driver's recomputed true residual. `fission_action` reconstructs
scalar flux and applies integrated native chi/NSF and the family source weights.
There is no second off-group scattering RHS.

Final sparse assembly copies each unique cell-to-cell `q` by `q` block into
preallocated NumPy float64 block data and typed block indices/row pointers.
Indices use int32 when the matrix dimension and expanded scalar entry count
fit, and int64 otherwise, before allocating block storage.
Sorted block columns form SciPy BSR, which is converted to the public CSR and
has explicit scalar zeros removed. This avoids Python scalar COO coordinate
and value lists; it does not change block accumulation, material caching,
gradient reconstruction, face currents, or the independent action path.
Both the intermediate block array and final CSR can coexist during conversion.
This storage change alone supplies no measured memory or scalability guarantee.

## Matrices and order

With q=MG and native full loss matrices L, the diffusion and reaction are

```
D_i = block_diag(inv(L_(2m+1),i)/(4m+3) for m=0..M-1)
R_i = volume_i * sum(C[m] kron L_(2m),i for m=0..M-1)
B   = G_family kron I_G
```

SP5 requires explicit native orders 0 through 5; SP3/SDP1 require 0 through 3.
Pass `required_orders=tuple(range(6))` to the native Serpent reader for SP5,
then construct moment materials with the established orientation. Missing orders
are refused; no scalar transport correction or synthesized high moment is used.
All matrices retain their nonsymmetry and signed entries. Finite even-order losses
are sufficient: reaction/removal matrices need not be invertible or positive.
Odd-order losses and actual elimination matrices must pass the finite condition gate
(default 1e12); there is no symmetric-positive-definite gate, clipping or
regularization. The material matrix cache reuses only identical immutable
material objects. Material-dependent gradient patches remain cell-specific.

Geometry validation reuses `_validate_corrected_mesh`, with neutral scalar
coefficients only for that helper's geometry contract. Actual diffusion is
never replaced with those coefficients. Two- or three-dimensional centroids,
actual face centers, unit normals, and positive **signed projected** normal
distances are required. A face normal n points from i to j. For H_i with q rows
and spatial columns, the cell-owned neighbor equation is

```
H_i r + d_j * (inv(D_j) @ D_i - I) @ H_i @ n = U_j - U_i
```

The opposite patch reverses indices and orientation. The inverse diffusion
must stay on the left in `inv(D_j) @ D_i`. Boundary patch equations are
`D_i @ H_i @ n = 0` for reflection and
`D_i @ H_i @ n + B @ H_i @ (x_f-x_i) = -B @ U_i` for vacuum.
Each individual scalar equation is divided by its entire Euclidean row norm,
and its RHS by the same norm. Full-rank fixed SVD least squares rejects singular
value ratios at or below `max(1e-12, 1/condition_limit)`; it does not truncate and
accept deficient patches. This row scaling matters for non-affine fields.

With tangential offsets `t_i=x_f-x_i-d_i*n` and
`t_j=x_f-x_j+d_j*n`, currents use the following exact multiplication order:

```
a_i = U_i + H_i @ t_i
J_f = area * inv(d_i*inv(D_i) + d_j*inv(D_j)) @ (a_i-a_j)
C   = D_i/d_i
J_b = area * B @ inv(C+B) @ C @ (U_i + H_i @ t_i)  # vacuum
```

Interior currents are accumulated with opposite signs on their two owners;
reflective current is zero. Tangential offsets vanish on orthogonal geometry,
recovering the existing full native `CoupledMomentOperator`. Representable
legacy scalar reduction requires **L2=L3=sigma_t**, L1=1/(3D1) and L0=removal.
Arbitrary native one-group moments do not satisfy that equivalence. For
isotropic multigroup scattering, complete legacy coupled equations match after
off-group scattering is moved to the left.

## Bounds and numerical evidence

By default the constructor rejects more than **20,000 unknowns before local
dense material or patch allocation and sparse expansion**. An explicit
`max_unknowns=<positive integer>` replaces that default numeric budget;
`max_unknowns=None` retains the class default. Boolean, noninteger and
nonpositive budgets are rejected. The selected budget is exposed as
`op.max_unknowns` for callers to record in provenance. An override does not
launch a calculation or establish that construction, an eigenproblem or a
preconditioner fits in memory. Larger resource probes need separate process
memory and time caps. The constructor additionally rejects a squared
local gradient width `(q*dimension)^2` above 4,000,000 entries. These are basic
prototype bounds, not memory/performance guarantees. Local patch SVD and Python
block dictionaries can be expensive well below them; assembly-scale storage,
preconditioning and measured scaling require separate work.

`tests/test_corrected_native_moments.py` covers full-matrix orthogonal equality,
SP3/SDP1 one-group and isotropic multigroup reduction on non-affine skew vacuum
fixtures, genuinely noncommuting piecewise-affine interface patches, coupled
vacuum trace/current identities, reflective constants and global leakage
conservation, 3D geometry, independent face action versus CSR and dense solves,
group permutation, signed removal, singular reaction with nonsingular leakage,
explicit rank/condition/size rejection, and a small eigenvalue versus an
independently assembled dense pencil. Partial-boundary affine patch fixtures
satisfy all included boundary constraints; they are not complete reactor domains.
The dense eigen oracle uses an orthogonal synthetic mesh and eliminates explicit
face unknowns, rather than using the production harmonic/vacuum formulas.

`tests/test_native_block_storage.py` additionally compares canonical CSR arrays,
gradients, actions, fission actions, and face currents exactly with the immutable
`c613ef7` source snapshot on noncommuting two-group skew meshes for both families
and reflective/vacuum boundaries. It checks the snapshot manifest and unchanged
unchanged shared type/helper hashes before loading the frozen operator and its
frozen coupled-moment helper in an isolated namespace. That local evidence
test skips when its external snapshot is absent; independent numerical oracle
tests remain available in source archives. Small fixtures also check default
budget refusal, explicit budget replacement and early invalid-budget rejection.

`CoupledMomentEigenSolver` explicitly admits this class, uses the volume
accessor, preserves its scalar/power positivity and source/k/balance checks,
and hashes this module plus the shared corrected geometry and mesh sources.
All results remain computational-only with `physical_admission=False`.

### Optional fixed CPU ILU

`CoupledMomentEigenSolver(op).solve(preconditioner='ilu')` explicitly enables a
fixed incomplete LU preconditioner. The default `preconditioner='none'` retains
unpreconditioned GMRES. The two admitted native CPU operator classes use the same
option; unknown names are refused. Each ILU solve makes one frozen CSC copy of
`op.assembled`, calls SciPy `spilu` once with **drop_tol=1e-6, fill_factor=20**,
and uses that copy and factor for every power-iteration linear solve. Other
SuperLU options, including equilibration behavior, keep their SciPy defaults.
There is no tuning, factor reuse across calls, tolerance change or fallback.
A factorization failure raises an explicit `RuntimeError` with its original
cause before GMRES is called.

`provenance['preconditioner']` records the requested mode.
`provenance['preconditioner_metadata']` records `setup_seconds` and `matrix_nnz`;
the ILU mode additionally records `drop_tol`, `fill_factor`, `superlu_nnz` and
`superlu_nnz_caveat`. Setup time includes the CSC snapshot, factorization and
linear-operator wrapper. `superlu_nnz` is **SuperLU.nnz internal factor storage**,
not an explicit CSC entry count of materialized L and U. The implementation does
not access or materialize those factors. The default mode reports zero setup
time and omits factor statistics.

Preconditioning changes only the linear solver. The true linear and eigenpair
residuals still use the independent `op.action`; the same raw scalar/power
positivity, source normalization and k/source/balance gates apply. Synthetic
SP3/SDP1 tests compare ILU with the default and an independent dense pencil,
reject corrupted CSR even when its ILU solve succeeds, check one factorization
without L/U materialization, and check explicit failure without fallback. These
checks do not establish convergence on reactor geometry or native data.

### Optional iteration progress

`solve(iteration_callback=callback)` calls `callback(event)` with a fresh
dictionary containing only scalar values. The default `None` retains the
existing numerical path. This is useful for observing long native assembly
solves without exposing mutable solver arrays or preconditioner factors.

The `phase` field identifies four events:

- `inner_start`: outer iteration, inner budget, restart length and tolerance.
- `inner_iteration`: outer iteration, one-based inner iteration and
  `scipy_preconditioned_residual_norm`. This is SciPy's monitored preconditioned
  relative residual; `used_for_acceptance=False` makes its role explicit.
- `inner_complete`: iteration count, GMRES status,
  `true_action_relative_residual` and `accepted_linear`. Acceptance still
  requires successful GMRES and the independently evaluated true action
  residual to satisfy the requested inner tolerance.
- `outer_complete`: eigenvalue, eigenvalue/source changes, full residual,
  reconstructed-field positivity, convergence flag and stopping reason.

An inner failure emits no outer completion event. An outer event is emitted
only after the source normalization and physical-field checks are reached.
Callback exceptions propagate to the caller; they do not trigger a numerical
fallback. The callback changes neither the cold inner initial guess nor the
restart, budgets, residual checks or positivity criteria. Tests compare
observed and default SP3/SP5 runs bitwise and distinguish the monitored
residual from true-action acceptance.

### Optional fixed CPU energy-group ILU

`solve(preconditioner='group_ilu')` uses fixed block Jacobi across energy
groups. For every group `g`, it extracts every moment for **all cells** from the
native `(cells, M, G)` layout, using indices `M*G*i+G*m+g` in ascending cell,
then moment order. For SP3/SDP1 this remains `[2*G*i+g, 2*G*i+G+g]`.
This is an `M*cells` square diagonal
group block, including all its spatial and moment couplings. Each block is
copied to CSC and factored exactly once with `spilu(drop_tol=1e-6,
fill_factor=group_ilu_fill_factor)`. The explicit
`solve(group_ilu_fill_factor=20)` default preserves the original budget; a
finite real value at least one is required, and booleans are rejected. This
parameter applies only to `group_ilu`; ordinary `ilu` retains its fixed value
of 20. A smaller value can reduce factor storage but can also cause breakdown,
slower convergence or stagnation. It is not a total-memory cap or an automatic
fallback. Group solves gather and scatter those same indices for every
GMRES application; factors remain frozen throughout the solve. SuperLU's other
options retain their defaults. Setup failures identify the group and raise
with their original cause; there is no solver fallback.

GMRES continues to use the original full CSR, retaining every off-group
coefficient. Only the preconditioner omits off-group coupling. This mode does
not make a full-matrix CSC snapshot; it freezes only the extracted factors.
As with the existing modes, callers must not mutate the assembled matrix
during a solve. No equations, independent `op.action` linear/outer/final
residual checks, convergence tolerances or positivity criteria change.
Both admitted native CPU operator classes support this option.

The common metadata records total setup time and full `matrix_nnz`, plus
`drop_tol`, `fill_factor`, `group_cell_ordering`, and a `groups` sequence. Each
group entry records its zero-based `group`, block `dimension`, `matrix_nnz`,
`superlu_nnz` and `setup_seconds`. Per-group setup time includes extraction,
CSC conversion and factorization. The `superlu_nnz_caveat` has the same meaning
as in monolithic ILU: reported internal storage, with no L/U materialization.
`tests/test_native_group_ilu.py` checks distinct permuted blocks, index
gather/scatter and frozen factors, exactly G setup calls across outer
iterations, one-group equivalence to monolithic ILU, small corrected SP3/SDP1
eigenpairs against an independent dense pencil, and rejection of corrupted
assembly by true action. These synthetic checks supply no native-data physical
admission, reactor convergence, runtime or memory guarantee.

Group-block extraction uses two passes over bounded row chunks instead of
materializing every selected row across all energy groups. Each chunk contains
at most 4,096 selected rows and 1,000,000 source entries, except that one larger
indivisible row is allowed. The first pass counts retained entries; the second
fills the final CSR allocation. Column selection and remapping preserve the
entries, explicit zeros, duplicates and their order. The resulting CSC block
is the same principal submatrix used by the original extraction. Transient
blocks are released after factorization; all seven factors remain available
to the preconditioner in a seven-group calculation.

This bounds extraction workspace, not total operator or factor storage. An
assembly resource probe is still required. Tests compare the extracted sparse
arrays with direct SciPy slicing, including empty rows and wider index arrays,
and check native SP3/SP5 group solves. The full GMRES matrix, incomplete-LU
parameters and independent residual checks are unchanged.

An optional `group_ilu_setup_callback` receives metadata-only events before
extraction, before factorization, after each completed factor and on setup
failure. Events identify the group, effective fill budget and completed factor
statistics. A research driver can persist them alongside process memory
measurements without materializing the factors' L and U matrices. Failed setup
raises with its original cause and completed-group evidence; it does not try
another fill budget. Callback errors also propagate. The effective group fill
budget is recorded in returned provenance and preconditioner metadata.

The existing raw native-data audit found order-1/3 invertibility with condition
numbers at most about 308. That algebraic observation does not qualify higher
angular estimators for deterministic physical use. Native balance/precision,
higher-moment estimator consistency and binary/library identity remain separate
questions. This experiment is a separate numerical route, not an assumed fix
for the previously observed reactor discrepancy.

## Experimental SP5 validation and resource scope

`tests/test_native_sp5.py` derives the transformed SP5 reaction and outward
Marshak boundary coefficients independently using exact fractions, the explicit
U-transform and half-range Legendre integrals. It checks dense face-unknown
elimination, third-moment-only scalar/fission reconstruction, homogeneous
reflective k-infinity, a compatible orthogonal scalar-SP5 reduction, noncommuting
skew affine transmission, vacuum current identities, energy permutations,
three-moment group ILU and true-residual rejection of corrupted assembled matrices.
The existing corrected scalar mesh operator supports only two moments; it does
not provide a nonorthogonal SP5 reference.

An SP5 construction or one successful group factorization does not establish
full-solve memory feasibility. A resource probe must include all group factors,
a GMRES solve, independent action and result creation. Record family, moment
count, required orders and implementation hashes with resource evidence. An SP3
construction receipt cannot qualify SP5 resources. Public default unknown limits
are unchanged; large trials require explicit process memory/time caps.

These checks establish computational behavior on their stated fixtures. They do
not establish improved reactor accuracy, higher-moment estimator consistency,
statistical precision or full-core feasibility.
