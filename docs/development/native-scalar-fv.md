# Native scalar finite volume pilot

`garnet.native_scalar_fv` is an opt-in CPU diffusion prototype for regional
coefficients that must retain their native neutron-production balance. It uses
the existing polygon mesh and scalar FV leakage operators. It does not change
`Material`, `RegionalXS`, the existing eigenvalue drivers, or native angular
transport.

This interface is useful when reduced absorption is signed, or when replacing
native total and scattering production with an absorption-based material would
alter the reaction operator. Signed reduced absorption is a diagnostic here;
it is not a coefficient used to assemble loss.

## Discrete equations

For cell `c`, coefficient-table row `r`, destination group `g` and source group
`h`, the integrated operators are

```text
A(phi)[c,g] = leakage(D, phi)[c,g]
            + volume[c] * (T[r,g]*phi[c,g]
                           - sum_h SP0[r,h,g]*phi[c,h])
F(phi)[c,g] = volume[c] * sum_h F[r,g,h]*phi[c,h].
```

`SP0` has **source, destination** group order. `F` has **destination, source**
order. The full fission matrix is applied without a separable spectrum
approximation or spectrum normalization. The source normalization is the sum
of the integrated fission-neutron source.

The scalar group operators contain `T - diagonal(SP0)`. The coupled action
subtracts only off-diagonal production after that. Cell volume appears exactly
once; the existing leakage operators already integrate the reaction diagonal.
The native total/scattering imbalance is retained. No absorption repair,
diagonal fitting, coefficient clipping or automatic fallback is performed.

## Inputs and use

Import the prototype explicitly:

```python
from garnet.native_scalar_fv import RegionalCoefficients, NativeScalarFV
```

`RegionalCoefficients` takes unique integer `region_ids`, descending
`energy_bounds`, positive regional `D` and `T` arrays of shape `(R, G)`, and
nonnegative `SP0` and `F` arrays of shape `(R, G, G)`. Declare
`scattering_orientation="from,to"` and `fission_orientation="to,from"`.
Optional `signed_rabs` is retained as diagnostic data. Optional
`kappa_fission[R,G]` supplies energy-production coefficients. Numeric input
arrays must be finite and real. Coefficients are copied into immutable storage.

Give `NativeScalarFV` a mesh, the coefficient table and an explicit integer
`cell_region_index[N]` assigning each cell to a table row. Mesh tags are not
automatically interpreted as row indices. Select `method="tpfa"` or
`method="corrected"`, and `preconditioner="ilu"` or `"jacobi"` explicitly.
The corrected option uses the existing nonorthogonal scalar reconstruction.
Group ILU settings are fixed by the shared preconditioner helpers; intergroup
coupling remains in the full GMRES action.

Set `max_bytes`, `restart`, `max_inner_iterations` and `max_outer_iterations`
explicitly. Regional matrices are expanded only for bounded batches of cells,
controlled by `batch_cells`; the full cell-by-group-by-group matrices are not
stored. Boundary conditions are passed to the existing mesh classification.

## Iteration, results and failures

The cold power iteration uses a zero initial guess for every coupled inverse
solve. It checks the actual unpreconditioned inner residual against `1e-7`,
absolute eigenvalue change against `1e-6`, and source and full eigen residuals
against `1e-5`. Positivity checks apply to scalar flux and, when available,
integrated heat. Passing these checks establishes numerical convergence of
the declared discrete model, not agreement with transport or mesh convergence.

`solve()` returns a `PilotResult` with convergence status, reason, eigenvalue
estimate, flux, optional heat, iteration history and diagnostic metadata.
Treat a failed result as diagnostic evidence. It must not be exported as a
converged eigenpair. An eigenvalue estimate is unavailable before a finite
candidate iterate has been accepted.

Energy heat is computed separately from neutron production:

```text
heat[c] = volume[c] * sum_g kappa_fission[r,g] * flux[c,g].
```

Without `kappa_fission`, heat is unavailable; the fission-neutron source is
not substituted for it. No subcell power reconstruction is implied by a scalar
cell mean.

Positive diffusion coefficients do not guarantee that native loss is
invertible or that a corrected stencil preserves positivity. Singular loss,
factorization failures, nonfinite arithmetic, negative states and exhausted
iteration or memory budgets must remain explicit failures. A closed reflective
conservative loss operator is one possible singular case.

## Resource limits and current status

The constructor estimates setup and Krylov storage before constructing group
operators. It also records traced allocation and setup timing information.
These are estimates and measurements with stated scope; Python tracing does
not guarantee coverage of native sparse-factorization peaks. Enforce an
external process memory/time ceiling for physical calculations. The
`external_memory_cap=True` argument declares that the caller did so; it does
not install or verify a ceiling. Large states require this declaration.

`estimate_memory(mesh, coefficients, method=..., preconditioner=..., restart=...,
batch_cells=..., max_outer_iterations=...)` exposes the same pre-build estimate
without constructing FV operators or factors. Constructor diagnostics retain
the complete breakdown in `memory_estimate`. The internal `max_bytes` and
external process ceilings are unchanged by this accounting.

The estimate separates storage by lifetime:

- Each corrected group retains its host CSR matrix, three padded row gathers
  (cell loss, interior currents, boundary currents), gradient dictionaries with
  small NumPy vectors, scalar diagonals, and its TPFA basis. TPFA retains ELL
  indices/weights and diagonals. These costs scale with the group count.
- Gradient support is bounded by the owner and face neighbors. A corrected
  cell row includes those supports from its face neighbors too. The estimator
  derives row and face width allowances from this topology, including padded
  gather widths. It does not allocate cell-by-group-by-group matrices.
- Temporary row/current dictionaries, reconstruction equations, CSR-building
  Python lists and copied gradients belong to one group under construction.
  Their byte allowances are counted once, alongside all retained groups.
  They are not multiplied by ILU fill or by the group count.
- CPU ILU retains SuperLU sparse factor arrays and permutation/bookkeeping
  data. The planning allowance uses up to `20*matrix_nnz + 2*N` factor entries,
  capped at `2*N*N`, with 24 bytes per entry plus per-row bookkeeping.
  This uses the fixed helper's fill factor of 20; it is not a promise that
  SuperLU numerical pivoting or internal allocation obeys that entry count.
  Ordering/factorization workspace receives three factor-size allowances plus
  sparse conversion and per-row space. Only one group's factorization workspace
  overlaps retained factors. The setup transient is the larger of this workspace
  and the one-group Python construction allowance.
- Shared input buffers, mesh records, Krylov/state vectors, bounded regional
  batches, action gather temporaries, and 16 KiB per permitted outer-history
  entry are counted separately. History storage therefore participates in
  admission before the solver runs.

Python dictionary/object costs and native workspace are conservative planning
allowances, not allocator guarantees. Topology with a high-degree cell can
force nearly dense corrected gathers and is rejected if the estimate exceeds
`max_bytes`. A source change to retained operator fields or the fixed ILU helper
requires reviewing this estimate. Traced setup measurements and an actual
externally enforced process cap remain necessary; neither an estimate below
the cap nor a small synthetic mesh establishes a physical memory qualification.

Start with a small matched pin case. Measure setup, factor storage and peak
process memory before admitting a large assembly. Use an immutable source/data
snapshot and run physical comparisons without competing timed calculations.

The prototype has manufactured tests against independent dense operators and
eigenpairs, including full nonseparable fission, unequal volumes, shuffled
regional maps, conservation and failure cases. It has not yet been qualified
against the PRATIC Serpent references. A scalar diffusion closure remains a
different approximation from angular transport or coupled multigroup P1.
