# Experimental one-energy-group angular operators

`garnet.native_grouped_angular.GroupAngularOperators` exposes the group blocks
of the existing two-dimensional native angular equations. It provides operator
primitives only: there is no fixed-source solver, multigroup sweep, fission
power iteration, or automatic solver selection.

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
discard upscatter.

`streaming_group_action` includes total collision, conservative interior
upwinding, vacuum outflow and reflected incoming coupling. Specular boundaries
retain their angular index permutations; all angles remain coupled within the
group. `group_action` subtracts only same-group scattering. Its result equals
the target group of the coupled loss operator applied to a field with only
that incident energy group active. Intergroup scattering belongs on the right
side of a future group equation. `diagonal_group` returns the streaming and
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
vendor BLAS workspaces, caller state, future Krylov storage, factors, result
copies and simultaneous call ownership require separate accounting. Streaming
and diagonal actions currently use the reference face loops; core runtime
feasibility is unproven.

Manufactured verification compares group projection, all-group and same-group
scattering, streamed loss, exact reflection and diagonals with the existing
coupled implementation, including both XS orientations, heterogeneous cells,
upscatter, signed P7 data and mixed boundaries. Operator agreement does not
establish convergence of group iteration. A future solver must refresh all
moments and pass the complete simultaneous residual, fission normalization,
positivity and iteration-limit checks. This prototype has no physical-array,
core or three-dimensional qualification.
