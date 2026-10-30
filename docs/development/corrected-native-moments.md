# Experimental native full-matrix corrected moments

`garnet.corrected_native_moments.CorrectedNativeMomentOperator` is an explicit,
CPU research implementation for SP3, SDP1 and SP5 on a `MeshGrid`. It is separate from the
legacy per-group corrected scalar/moment dispatch. It makes no reactor
scalability, angular-transport consistency, or physical-accuracy claim. No
assembly calculation or native-data solve is part of this implementation.

Construct with one immutable `MomentMaterial` reference per cell and an explicit
`family='sp3'`, `'sdp1'` or `'sp5'`. Boundaries come from the grid (or the optional `bc`
argument); each must be homogeneous `vacuum` or `reflective`, and mixed classes
are supported. No active mask,
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
`H_i @ n = 0` for reflection and
`D_i @ H_i @ n + B @ H_i @ (x_f-x_i) = -B @ U_i` for vacuum.
Each individual scalar equation is divided by its entire Euclidean row norm,
and its RHS by the same norm. Full-rank fixed SVD least squares rejects singular
value ratios at or below `max(1e-12, 1/condition_limit)`; it does not truncate and
accept deficient patches. This row scaling matters for non-affine fields.

The reflective row uses `I*n`, with zero RHS, to match a same-material mirrored
neighbor: `inv(D_neighbor) @ D_i - I = 0` and centroid separation `r=2*d*n`
reduce its normalized neighbor equation to `H_i @ n=0`. Although multiplying
this exact constraint by invertible D gives the same continuous Neumann
condition, it changes the metric of an overdetermined row-normalized least
squares fit for coupled groups. The former `D_i @ H_i @ n=0` rows therefore
failed exact discrete reflection folding on a skew non-affine fixture with
native non-diagonal D. Only reflective gradient rows change; the vacuum rows,
interior transmission/current formulas and exactly zero reflective boundary
current remain unchanged. This is still a soft least squares row, not a demand
that arbitrary non-affine reconstructed gradients satisfy `H_i @ n=0` exactly.
Rank/conditioning thresholds, clipping policy and solver acceptance do not change.

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
`c613ef7` source snapshot on noncommuting two-group skew vacuum meshes for both
families. Reflective cases retain exact legacy fission/material checks and
current CSR/action/face-conservation checks; their gradient/action equality
against the old reflective metric is intentionally replaced by the folding
invariant in `tests/test_native_reflection_folding.py`. That test checks the
full matrix `S A_full E = A_half`, every basis gradient/current map, zero cut
flux and rotation covariance on four skew half-domain cells mirrored into
eight full-domain cells. It uses heterogeneous noncommuting off-diagonal D,
SP3/SDP1/SP5 and oblique reflective cuts with mirrored vacuum exteriors.
These synthetic tests do not copy reactor XS or qualify a reactor calculation.
The storage oracle checks the snapshot manifest and unchanged
shared type/helper hashes before loading the frozen operator and its
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
fallback. The callback changes neither the selected inner initial guess nor the
restart, budgets, residual checks or positivity criteria. Tests compare
observed and default SP3/SP5 runs bitwise and distinguish the monitored
residual from true-action acceptance.

### Optional inner warm start

`solve(inner_warm_start=True)` explicitly enables a fixed initial-guess policy
for the existing inner GMRES calls. The option requires a Python `bool` and
defaults to `False`; default histories, callback records and numerical results
retain the original cold path. The **first outer solve is always cold**
(`x0=None`), including when the caller supplies an outer `initial` state.
Subsequent outer solves start from a newly allocated `x0=k*u`, using the latest
eigenvalue and previous source-normalized outer candidate. This reconstructs
the preceding linear solution up to normalization roundoff and approximates
the solution of the next equation. The eigen-relation `A*(k*u)=F*u` motivates
that scaling at convergence; it is not an exact identity for the new RHS
before convergence. The normalized state is never passed as mutable GMRES
workspace, and the RHS remains exactly `op.fission_action(u)`.

This policy changes only the initial guess. It preserves the frozen A and M,
restart length, iteration/resource budgets, tolerances, source normalization,
positivity, and independently recomputed `op.action` acceptance. A warm solve
with nonzero GMRES status or a failed independent residual is still rejected.
There is no restart-budget reinterpretation, retry, or fallback.

Provenance records `inner_warm_start` and the actual
`inner_initial_guess_reuse_count`; the opt-in mode also records
`initial_guess_source='previous source-normalized outer state scaled by k'`.
The legacy `initial_state_reused=False` field retains its existing meaning;
it does not describe this newly explicit within-solve inner-guess policy.
When enabled, each history entry and `inner_start`/`inner_complete` callback
record includes the requested mode, actual `inner_initial_guess_reused`,
`inner_initial_guess` (`zero` or `k*u`) and restart length. Thus an enabled
one-outer call truthfully records zero actual reuse. Callbacks remain scalar
metadata only and do not expose iterates.

For research checkpoints, the separate optional
`accepted_outer_state_callback(payload)` runs once per valid outer candidate,
after successful GMRES/independent linear acceptance, positive source
normalization and scalar/power positivity, before convergence exit or the next
outer solve. A valid candidate is not necessarily an accepted final eigenpair.
Its payload contains independent flat array copies named `linear_solution`
(raw y), `rhs` and `native_state` (normalized u), plus fresh scalar history
fields `outer`, `inner_iterations`, `gmres_info`, `linear_residual`, `k_eff`,
`k_change`, `source_change`, `full_residual`, the current convergence-gate flag
`converged`, and the opt-in warm/restart fields when enabled. No phi/power
arrays are copied. Mutating the payload cannot alter solver state or later
warm starts. Callback exceptions propagate immediately, allowing a harness to
save a checkpoint and stop before another GMRES call. No callback runs for a
failed linear solve or a candidate that fails positivity. The default `None`
adds no state-array copies and retains the numerical path. Checkpoint policy,
disk writes and resource headroom checks belong to the caller.

When this state hook is provided together with `group_ilu_coarse`, the
`inner_start`, `inner_complete` and candidate payloads also include fresh
scalar monitor fields: `coarse_aggregate_sha256`, `coarse_aggregate_count`,
`coarse_dimension`, `coarse_applications`, `coarse_solves`,
`coarse_extra_a_matvecs` and `coarse_setup_coarse_solves`. This permits a mapping
guard before the first GMRES and preserves actual counters in a checkpoint
before a callback raises. No arrays or factors enter the progress callback.
Existing progress payloads are unchanged when the state hook is absent.

`tests/test_native_inner_warm_start.py` checks the first cold solve, later
`k*u` scaling and unchanged source normalization/RHS, separate initial-guess
allocation, fixed matrix/preconditioner, failure rejection, strict flag
validation, and bitwise default comparisons with frozen `70c399c` SP3/SP5
solves for every preconditioner. Optional tiny actual-data SP3/SP5 cases use
`NDGPU_NATIVE_COARSE_TEST_XS` and compare warm/cold eigenpairs, independent
residuals, normalized source and positivity. These small tests do not qualify
assembly-scale convergence or transport accuracy. The next assembly research
gate is only a bounded two-outer qualification: one accepted cold solve, then
one warm RHS with the existing 1000-inner-step budget and externally enforced
one-CPU, 96-GiB, 3600-second caps. This option does not launch that experiment
or establish that it will succeed.

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
parameter applies to the group ILU modes; ordinary `ilu` retains its fixed value
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

### Experimental fixed native two-level correction

`solve(preconditioner='group_ilu_coarse', group_ilu_fill_factor=10)` explicitly
adds a coarse correction to the original fixed group ILU. This is a bounded
implementation experiment, not an established fix for assembly-scale SP5
stagnation. Select `group_ilu` to return to the original path. Defaults, group
factor extraction/options, full GMRES matrix identity, callbacks, and independent
`op.action` linear/outer/final acceptance remain unchanged. In particular this
option does not change the general solver tolerances; a research harness must
explicitly select its inner `1e-7`, k `1e-6` and source/balance `1e-5` gates.

For a fixed group smoother M and implicit piecewise-constant transfer P:

```
Ac = P.T A P
Q r = P solve(Ac, P.T r)
M2 r = Q r + M(r - A Q r)
```

Every aggregate retains all M*G native components, in coarse index
`(M*G)*aggregate + G*moment + group`. Restriction is a **sum of integrated
balances**, never volume averaging. Aggregates may cross material identities
only inside this preconditioner; no material, equation or fine matrix changes.
Each application uses one dense coarse LU solve, one additional original CSR
matvec and one application of the unchanged group factors. There are no adaptive
factors, aggregate-count tuning, shifts, regularization or fallback.

The geometry map has exactly `min(n_cells,128)` aggregates. It first seeds every
mesh connected component at its smallest cell index, in component order; more
than 128 components are refused. Remaining seeds maximize minimum squared
Euclidean centroid distance, with smallest-index ties. Multisource graph
Dijkstra uses interior-face adjacency and finite positive centroid edge
lengths, with lexicographic `(distance, seed ordinal, predecessor cell)` ties.
Nonempty coverage and connectivity of each aggregate are checked afterwards.
Corrected operators use `grid.mesh.centroid` and interior `faces`; orthogonal
operators use their actual `mesh.centroids` and `interior` owners, with no
inferred index-space geometry.

Coarse assembly scans bounded CSR chunks (at most 4096 rows and 1,000,000
entries, except a single larger indivisible row), accumulates every duplicate
coarse pair in `longdouble`, and casts once to float64 before nonsymmetric dense
LU. Platforms without wider `longdouble` are refused. It never forms AP, a full
fine transfer, a full sparse matrix copy, or abs(full fine A). The fine matrix
and caller-owned smoother must stay fixed for the object's lifetime.

`NativeCoarsePreconditioner.from_operator(op, smoother)` in
`garnet.native_coarse` accepts an already-built fixed linear callable or
`LinearOperator`, so a diagnostic can reuse its group factors. The lower-level
constructor accepts `(matrix, shape, aggregate_ids, smoother)` for small known
layout fixtures; callers supplying a map own its geometric connectivity check.
`prolong`, `restrict`, `coarse_action`, `apply` and `linear_operator` expose the
fixed algebra for external audits. Tests compare `coarse_action(z)` with
`restrict(op.action(prolong(z)))` independently, and check `M2(A P z)` against
`P z`. A production research harness should audit its own fixed vectors before
GMRES; no costly native action is hidden inside coarse setup/application.

Singular/nonfinite LU, invalid LAPACK `gecon` reciprocal condition estimates,
and failed coarse solves raise without adjustment. One deterministic setup
solve and every application use the fixed componentwise diagnostic
`abs(Ac*x-r) <= beta*(abs(Ac)*abs(x)+abs(r))`, where
`beta=2*d*u/(1-2*d*u)`, `d=dim(Ac)` and `u=eps64/2`. Zero scale requires exactly
zero residual; nonfinite quantities fail. Absolute coarse coefficients are
processed in dense row chunks. This is a backward-consistency diagnostic,
not a proven forward-error bound for LU or the native computational graph.
Maximum observed forward relative coarse residual and componentwise bound ratio
are recorded separately. No roundoff allowance enters true action acceptance.

Metadata under `preconditioner_metadata['coarse']` reports exact aggregate and
coarse-matrix SHA256 hashes, layout, fixed budget/dimension, setup/mapping and
accumulation times, CSR chunk/entry counts, `gecon` condition estimate, retained
and workspace array bytes, and process lifetime peak RSS (not isolated setup
memory). The LU/pivot and dense coarse arrays are retained; array counts exclude
Python/SciPy temporaries and the caller-owned fine matrix/factors. Setup solves,
application coarse solves and extra A matvecs have separate counters. Only the
opt-in mode adds `native_coarse.py` to implementation provenance.

`tests/test_native_coarse.py` covers deterministic ties, disconnected/invalid
geometry, connected coverage, native transfer/adjoint layout, duplicate-safe
bounded accumulation and explicit small P.T A P, independent SP3/SP5 action,
coarse-space inverse identity, linearity/repeatability, fixed-factor reuse,
fail-closed behavior and integrated true-residual rejection. Optional local
actual-data tiny SP3/SP5 checks read the audited derivative named by
`NDGPU_NATIVE_COARSE_TEST_XS`, compare a direct solve and independent residual,
and skip when that external input is unavailable. These checks supply no
assembly convergence, runtime, transport accuracy or physical admission claim.

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

## Experimental deferred operator storage

`CorrectedNativeMomentOperator(..., storage="deferred")` provides an opt-in
operator prototype. The default remains `storage="assembled"`. Deferred storage
uses the same material losses, normalized local SVD equations, boundary
conditions and conservative face currents. It constructs one cell's gradient
patch at a time and stores its coefficients in immutable rectangular BSR
chunks. `gradient_chunk_cells` and `batch_size` are positive integer controls,
both defaulting to 128. They bound gradient construction and action batches;
they are not limits on total process memory.

The prototype exposes:

- `linear_operator()`: the full coupled action through SciPy's `LinearOperator`.
- `group_principal_block(group)`: one CSR principal block, ordered by cell and
  retained moment. Full internal cross-group products are evaluated before the
  requested group rows and columns are selected. Exact zeros are removed, as in
  the assembled path; nonzero coefficients are never thresholded.
- `iter_full_blocks()`: bounded batches of `(cell_rows, cell_columns, blocks)`.
  Contributions can repeat, so consumers must sum duplicates. This is an
  interface used by the native coarse projection without a global matrix.
- `compact.payload_accounting()`: retained NumPy array bytes, excluding Python
  containers, source geometry/materials, action workspaces, construction
  temporaries and requested group matrices. This is not a process RSS estimate.

Accessing `.assembled` in deferred mode raises `NotImplementedError`; it never
silently allocates a global matrix. The CPU eigenvalue driver supports deferred
storage with `preconditioner="none"`, `"group_ilu"`, or `"group_ilu_coarse"`.
Global `"ilu"` rejects deferred storage explicitly. GMRES retains all coupled
terms through the full action. The driver creates and releases one exact group
source block at a time and retains the fixed factors for the entire solve.
True inner residuals are recomputed with `op.action`; source, eigenvalue,
full-residual and positivity criteria retain their existing meanings.

For `group_ilu_coarse`, the connected mapping and transfers are unchanged.
`NativeCoarsePreconditioner` streams full blocks and sums their contributions
to `P.T @ A @ P` in extended precision, then casts once to float64. The coarse
correction uses the full deferred action. Streaming and assembled construction
have different floating-point accumulation orders; matrix hashes and factor
patterns need not be identical. Reported workspace bytes include the source
batch and accumulation arrays, but not Python/SciPy temporaries.

`tests/test_native_compact_solver.py` checks assembled/deferred SP3/SP5 solve
and coarse-action equivalence, fixed-factor reuse, source-block release and
absence of assembled access. It tests true-inner acceptance on either side of
`1e-7` and rejection by each outer criterion at the campaign settings
(`k=1e-6`, source/full `1e-5`). Those settings are explicit solve arguments;
public defaults remain unchanged. A bounded assembly resource and convergence
pilot remains necessary before larger calculations.

`tests/test_native_compact.py` compares SP3/SP5 actions, currents, gradients,
fission and principal blocks with the assembled implementation and an
independent local least-squares action. It covers noncommuting group losses,
mixed boundaries, multiple geometric faces joining a cell pair, energy-group
permutations, 3D geometry, invalid/rank-deficient inputs and allocation guards.
Numerical parity assertions use explicit `rtol=1e-12`, with the stated absolute
tolerance; exact sparse zero removal has a separate pattern regression.

Deferred group construction currently retains moment-only row dictionaries
followed by BSR/CSR conversion for the requested group. Setup also retains
topological adjacency and individual reaction/face arrays. These costs, batched
stacking and scatter throughput require an actual bounded resource pilot.
Successful small-problem parity establishes the tested computational behavior;
it does not establish full-core memory fit, runtime or physical accuracy.
