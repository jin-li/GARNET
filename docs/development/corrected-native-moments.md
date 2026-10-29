# Experimental native full-matrix corrected moments

`garnet.corrected_native_moments.CorrectedNativeMomentOperator` is an explicit,
small CPU experiment for SP3 and SDP1 on a `MeshGrid`. It is separate from the
legacy per-group corrected scalar/moment dispatch. It makes no reactor
scalability, angular-transport consistency, or physical-accuracy claim. No
assembly calculation or native-data solve is part of this implementation.

Construct with one immutable `MomentMaterial` reference per cell and an explicit
`family='sp3'` or `'sdp1'`. Boundaries come from the grid (or the optional `bc`
argument), and must be homogeneous `vacuum` or `reflective`. No active mask,
discontinuity factor, fitted boundary source or GPU path is provided. States have native **cell-major** shape `(n_cells, 2, n_groups)` or
its flattened equivalent. The two moments are `U=(phi0+2*phi2, 3*phi2)`;
`scalar_flux(U)` reconstructs `U0-2*U1/3`. `action`/`apply` preserve input shape;
`face_currents` returns integrated currents with shapes `(n_faces, 2, n_groups)`
and `(n_boundary_faces, 2, n_groups)`, directed i-to-j and outward respectively.
`gradients` returns `(n_cells, 2*n_groups, dimension)`.

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

With q=2G and native full loss matrices L, the diffusion and reaction are

```
D_i = block_diag(inv(L1_i)/3, inv(L3_i)/7)
R_i = volume_i * (C1 kron L0_i + C2 kron L2_i)
B   = G_family kron I_G
```

All matrices retain their nonsymmetry and signed entries. Finite L0 and L2 are
sufficient: neither reaction nor removal needs to be invertible or positive.
L1, L3 and the actual elimination matrices must pass the finite condition gate
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
shared helper hashes before loading the frozen operator. That local evidence
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

The existing raw native-data audit found order-1/3 invertibility with condition
numbers at most about 308. That algebraic observation does not qualify higher
angular estimators for deterministic physical use. Native balance/precision,
higher-moment estimator consistency and binary/library identity remain separate
questions. This experiment is a separate numerical route, not an assumed fix
for the previously observed reactor discrepancy.
