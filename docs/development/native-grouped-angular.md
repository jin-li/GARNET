# Experimental grouped angular transport

`garnet.native_grouped_angular.GroupAngularOperators` exposes the group blocks
of the existing two-dimensional native angular equations. It provides operator
primitives only. The separate opt-in `GroupedFixedSource` prototype described
below adds a multigroup fixed-source sweep. Neither interface provides fission
power iteration or automatic solver selection.

```python
from garnet.native_grouped_angular import GroupAngularOperators

operators = GroupAngularOperators(streaming, boundary_conditions,
                                 max_bytes=operator_workspace_budget)
moments_g = operators.project_group(angular_flux_g)
loss_g = operators.group_action(angular_flux_g, group)
cross_source_g = operators.scatter_to_group(all_group_moments, group,
                                          exclude_self=True)
```

The caller supplies an existing immutable `NativeMeshStreaming` and an explicit
tuple containing one `"vacuum"` or `"reflective"` condition for each boundary
face. Angular group fields have shape `[cell, angle]`. Projected moments have
shape `[cell, harmonic]`; the coupled cache supplied to `scatter_to_group` has
shape `[cell, harmonic, from_group]`. The original source, energy grid, signed
production matrices and physical quadrature remain shared and unchanged.

`project_group` computes the sum of `weight * harmonic * angular_flux` with
the existing real spherical-harmonic basis, whose zeroth component is one.
Scattering uses all available orders, canonical `[to,from]` coefficients and
the original `1/(4*pi)` normalization. Scattering outputs are cell-integrated:
cell area is included exactly once. Excluding the target group removes its
same-group scattering term; it does not remove a physical energy group or
discard upscatter. Exclusion occurs before the moment contraction, using a
temporary coefficient vector, so subtracting a dominant same-group term does
not round away weak intergroup sources. The stored raw matrices remain unchanged.

`streaming_group_action` includes total collision, conservative interior
upwinding, vacuum outflow and reflected incoming coupling. Specular boundaries
retain their angular index permutations; all angles remain coupled within the
group. `group_action` subtracts only same-group scattering. Its result equals
the target group of the coupled loss operator applied to a field with only
that incident energy group active. Intergroup scattering belongs on the right
side of the group equation. `diagonal_group` returns the streaming and
collision diagonal used by the reference transport preconditioner; it does
not subtract scattering or replace angular reflection by a diagonal term.

The immutable wrapper shares its geometry and source. Its uniform
`allocation_estimate_bytes` bounds its own numeric outputs, local moments,
reflection permutations and bounded material/validation work. Caller-owned
fields, a previously retained coupled moment cache and the shared geometry
are excluded. There is no full multigroup angular allocation or global
angle-by-face cache in these primitives. Same-group scattering uses only the
local group moments. Material processing uses bounded batches of 512 cells
and visits the material IDs present in each batch.

This numeric guard is not a process memory-admission result. Python overhead,
vendor BLAS workspaces, caller state, Krylov storage, factors, result copies
and simultaneous call ownership require separate accounting. Streaming and
diagonal actions currently use the reference face loops; core runtime
feasibility is unproven.

Manufactured operator verification compares projection, all-group and same-group
scattering, streamed loss, exact reflection and diagonals with the existing
coupled implementation, including both XS orientations, heterogeneous cells,
upscatter, signed P7 data and mixed boundaries. Operator agreement does not
establish convergence of group iteration. These primitives have no physical-array,
core or three-dimensional qualification.

## Opt-in multigroup fixed-source prototype

`garnet.native_grouped_fixed_source.GroupedFixedSource` solves the supplied
cell-integrated equation `A psi = rhs` on the same 2D reference geometry.
It is a separate explicit entry point; existing transport solvers and physical
runs do not select it automatically.

```python
from garnet.native_grouped_fixed_source import GroupedFixedSource

solver = GroupedFixedSource(streaming, boundary_conditions,
                           inner_tolerance=1e-10, full_tolerance=1e-8,
                           restart=30, max_inner=200, max_sweeps=100,
                           max_bytes=fixed_source_workspace_budget)
result = solver.solve(cell_integrated_rhs)  # cold outer start
# Explicit alternative: solver.solve(cell_integrated_rhs, initial_flux=guess)
```

Both `rhs` and an optional `initial_flux` must be finite real numeric ndarrays
with shape `[cell, angle, group]`; signed angular fields and strided inputs
are allowed. The solver makes owned float64 copies and never modifies inputs.
The source is already cell-integrated and is not multiplied by area again.
An omitted initial field means exactly zero. A supplied field explicitly enables
a warm outer start; each group GMRES starts from that group's latest iterate.

Each full sweep visits all native energy groups in ascending index order.
For group `g`, all-angle GMRES solves `(Tg - V*Sgg) psi_g` against the external
source plus cross-group scattering from the latest coupled moment cache.
Earlier groups contribute their updated moments and later groups contribute
their previous moments. Repeated sweeps retain upscatter in both directions.
The LHS keeps exact same-group signed harmonics and specular angular coupling.
The only preconditioner is the one-group streaming/collision diagonal; no
P0 factors, coupled angular Krylov basis, XS changes, clipping, fission terms,
retries or alternative solve paths are added.

`inner_tolerance` and `full_tolerance` must be finite positive numbers; restart,
maximum inner steps and maximum sweeps must be positive integers. Effective
restart is capped by the group dimension and inner-step limit. GMRES uses
legacy callback semantics so `max_inner` counts Arnoldi steps rather than
restart cycles. Every group must have GMRES success and a directly recomputed
true group residual within the inner tolerance. The coupled cache is refreshed
from the complete terminal field after every full sweep. A separate calculation
then evaluates `A psi - rhs` group by group at that simultaneous field, using
all scattering, not the lagged group RHS. Convergence requires this complete
residual to pass `full_tolerance`; accurate group solves alone are insufficient.
Residuals use the L2 norm relative to the corresponding RHS, or the absolute
L2 norm if that RHS is exactly zero. The default zero source returns the exact
zero field without GMRES. No fission normalization or positivity admission is
part of this fixed-source mathematical problem.

The result contains an immutable copy of the raw terminal `angular_flux`,
`converged`, `reason`, `full_residual`, immutable sweep `history`, and provenance.
Iteration exhaustion retains the terminal field and its simultaneous residual.
Finite inner candidates are retained when GMRES fails its convergence gate;
no cold retry replaces them. Arithmetic failure retains the last finite
projected flux/cache pair and reports `full_residual=None`, since an earlier
residual could predate a partial sweep. History records completed groups,
inner steps, GMRES statuses, true group residuals and simultaneous residuals;
a failed inner sweep can therefore be partial.

`memory_estimate_bytes` is checked before operator reflection construction and
again before solve allocations. With `F=nc*na*ng`, `M=nc*h*ng`, `G=nc*na`,
`h=(L+1)^2`, and effective restart `r`, its byte bound is the operator's public
workspace bound plus
`8 * (5*F + 2*M + (r+1)*G + 32*G + 6*(r+1)^2 + 4*max_sweeps*(ng+4))`.
The full-field allowance includes owned source/iterate, immutable result-copy
ownership and conversion/validation margin. The remaining terms cover the
moment cache, one-group GMRES basis and conservative live workspace,
Hessenberg/rotation arrays and history numeric payload. The actual persistent
diagonal and GMRES basis are one group only. The simultaneous residual never
allocates a full coupled residual field or an angle-by-face cache.

Caller inputs, previously retained results, and existing shared immutable
geometry/source storage are outside this owned-workspace estimate. Python
object overhead and vendor BLAS scratch are not bounded by the numeric guard.
The five-full-field reserve is conservative; it does not claim these copies
are all persistently allocated. This is neither an RSS cap nor admission of a
96-GiB core calculation. Runtime, group-iteration convergence, physical arrays,
fission/eigenvalue behavior and 3D geometry remain unqualified.

Manufactured tests include an analytic reflective one-cell two-group solution,
upscatter that requires revisiting groups, and literal small dense coupled
matrices built from the Legendre angular kernel and geometric face equations.
They cover mixed reflection, both native matrix orientations, signed P7,
true residuals, zero source, explicit warm starts, caps, nonconvergence and
input/result immutability. These are mathematical tests, not physical admission.
