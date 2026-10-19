# Static branch libraries

The experimental `ndgpu.branch_xs` module provides immutable table states,
exact selection, bounded absolute interpolation and fresh CPU NEM operators.
It is separate from `TabulatedFeedback`: it does not clamp requests or apply
ratios to an existing solver. Run the data-free example from the checkout:

```sh
python examples/isonees_features/branch_xs.py
```

## Prepare and evaluate a library

A table state identifies the physical configuration, conditions, ordered
regions, energy boundaries and material arrays. Face factors use
`(region, 6, group)` with sides `x-, y-, x+, y+, z-, z+`. Fission matrices use
`(region, emitted_group, source_group)`. Record the reaction convention,
diffusion prescription, geometry, nuclear data and factor method in metadata.

Create segments or triangles explicitly. Use separate libraries/elements for
ARO/ARI configurations (normally labelled `aro`/`ari`), different bank
environments and different geometry or nuclear-data families. A five-point
temperature star covers a diamond in its transformed coordinates, not its
bounding rectangle. A request beyond a supplied segment or triangle raises an
error. Unvalidated interpolation remains labelled unvalidated.

`select(state_id)` returns the exact immutable state. `evaluate(conditions,
configuration=..., family=...)` returns absolute arrays and source weights.
Inspect those weights and the provenance before interpreting a prediction.
`BenchmarkModeratorPath` supplies the bounded three-knot benchmark density path.
Use `evaluate_temperatures(Tf, Tm, path, configuration=..., family=...)` to
evaluate it without treating moderator temperature as another independent
axis. Record the same `moderator_path_id` on every source state. `save(path)`
and `BranchLibrary.load(path)` persist table libraries as JSON without pickle.
Ambiguous condition lookup requires explicit `select(state_id)`. Evaluated
states cannot be reused as table entries, which would discard their lineage.
Use `build_nem_solver` with the physical grid and an explicit region-to-node
mapping when table regions are shared by multiple physical nodes. The returned
solver is new; changing the state requires constructing another one.

See the example for the executable interface and the
[methodology chapter](../methodology/branch-xs.md) for the fission-matrix and
ADF interpolation definitions. Existing fixed-state Serpent reading remains
available; importing XS alone does not create qualified ADFs.

## Qualify a new application

1. Reproduce exact table states, including physical region and face mapping.
2. Freeze the interpolation elements and hold out transport states.
3. Solve predictions using only training XS and factors.
4. Construct exact-state NDgpu solutions from the returned withheld data.
5. Report interpolation error, exact-state model error and total prediction
   error separately. Use `ndgpu.branch_validation` for consistent reactivity
   changes, coefficients, uncertainty screening and interval rod worth.

Temperature coefficients require an explicitly defined density path and
thermal-scattering treatment. Fuel Doppler changes normally hold density and
dimensions fixed. Update water in mixed reflector materials as well as in
the coolant. Preserve native errors; unresolved uncertainty is not a passed
accuracy test.

A branch library generated with uniform full-core temperature changes is not
qualified for arbitrary cellwise temperatures. This release does not add a
transient feedback hook, rod-motion dynamics, temperature/rod combinations or
depletion interpolation.
