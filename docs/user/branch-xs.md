# Static branch libraries

The experimental `garnet.branch_xs` module provides immutable table states,
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

ADFs must travel with the same spatial-reference and receiving-solver closure
and reflector mask used to derive them. Record `transverse_leakage_identity`,
`reflector_mask_sha256`, `leakage_boundary` and
`reference_current_convention` in state metadata, alongside the ADF method
and solver identity. A declared difference, or a declaration on only one
vertex, rejects an interpolation element. An explicit null also counts as a
declaration. Matching omissions retain legacy numerical interpolation; they
do not establish scientific qualification. The library compares declarations;
it does not verify that a mask hash describes the receiving geometry or that
the receiving solver implements the declared closure.

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
4. Construct exact-state GARNET solutions from the returned withheld data.
5. Report interpolation error, exact-state model error and total prediction
   error separately. Use `garnet.branch_validation` for consistent reactivity
   changes, coefficients, uncertainty screening and interval rod worth.

Cases used to choose a leakage method, reflector mask or ADF prescription are
method-development cases. Freeze those choices before blind validation, and
keep the validation transport data out of prediction and method selection.
Exact-state GARNET versus continuous-energy transport tests the model and
closure; interpolated GARNET versus exact-state GARNET tests interpolation.
Neither comparison substitutes for the other, and matching closure metadata
does not demonstrate accuracy in either comparison.

Temperature coefficients require an explicitly defined density path and
thermal-scattering treatment. Fuel Doppler changes normally hold density and
dimensions fixed. Update water in mixed reflector materials as well as in
the coolant. Preserve native errors; unresolved uncertainty is not a passed
accuracy test.

A branch library generated with uniform full-core temperature changes is not
qualified for arbitrary cellwise temperatures. This release does not add a
transient feedback hook, rod-motion dynamics, temperature/rod combinations or
depletion interpolation.

## Import returned Serpent files

`garnet.branch_io.import_returned_serpent_state` accepts a returned result file
(or several files), explicit conditions and an ordered mapping from your region
IDs to native universe names:

```python
from garnet.branch_io import import_returned_serpent_state

state, evidence = import_returned_serpent_state(
    "cold_res.m", state_id="fuel-cold",
    conditions={"configuration": "aro", "family": "thermal",
                "fuel_temperature": 500., "moderator_density": .72683},
    region_universes=[("fuel", "fuel_universe"),
                      ("reflector", "reflector_universe")],
    source_files=["input", "materials.inc"],
)
```

The mapping must cover every returned universe exactly once. Completion and
zero lost particles are required. Declared sources are hashed before and after
import; include every input, material and geometry file you want recorded.
The importer does not discover includes. Its evidence is a retrospective
snapshot: it does not attest which files the original execution consumed.
Executable and ACE content identities remain unknown. Missing collection
receipts do not require inventing replacements. Raw errors remain attached to
the native records. Importing without separately qualified factors gives unity
factors; it does not reconstruct or qualify ADFs.

## Construct a thermal star

`garnet.branch_thermal.build_thermal_star` takes five original native-backed
states in order: nominal, fuel-low, fuel-high, moderator-low, moderator-high.
Low/high mean actual temperature, so the moderator-low point has higher density.
Supply the actual `moderator_temperature` in each state's conditions or metadata,
its `moderator_path_id`, and metadata `role="nominal"` or `role="training"`.
Every source must explicitly name `geometry_family`, `nuclear_data_family`,
`scoring_identity`, `adf_prescription` and `solver_family`; matching omissions
do not establish compatibility.
Withheld roles are rejected. The builder creates four triangles in
`(sqrt(fuel_temperature), moderator_density)`; it preserves the dependent
moderator temperature as provenance rather than a third axis.

A `qualification_policy` callback must return `ThermalStarEvidence` with a
policy ID and evidence references for every state. The application must inspect
reconstruction, ADF, source and nuclear-data evidence; a qualification string
alone does not establish acceptance. For numerical diagnostics,
`diagnostic=True` permits explicitly unvalidated construction. It still rejects
incompatible geometry, nuclear-data identities, conventions or paths.

Run the self-contained manufactured example:

```sh
python examples/branch_xs/thermal_star.py
```

Keep half-step and combined transport checks outside the five training states.
A successful import or exact-state reconstruction does not establish prediction
accuracy. Different ACE routes need a controlled equivalence assessment before
they can share a qualified library.

`BranchLibrary` compares Serpent versions strictly by default. To explicitly
accept a version pairing, pass `compatible_serpent_versions=("2.2.0", "2.2.2")`
using the exact strings stored in your states' `serpent_version` metadata.
Only tuples/lists are accepted; a nonempty policy needs at least two distinct,
nonblank strings. Mixed versions must all be declared and included in that
policy. This records a user/application assumption, not code proof of version
equivalence, and leaves nuclear-data, geometry, scoring and closure checks in
force. A version difference alone does not automatically require another
Monte Carlo run; assess the available input, data and method evidence.

The policy survives library save/load; older files without it remain strict.
Source metadata and native versions remain unchanged. Evaluated metadata
records `source_serpent_versions` by contributor and
`compatible_serpent_versions`; mixed-version evaluations omit the single
`serpent_version` field rather than attributing the first vertex's version to
the whole prediction. This policy does not qualify a library or waive an
assessment of a changed nuclear-data route.

## Convergence and physical checks

Static CPU NEM supports explicit `tol_k`, `tol_source` and `tol_balance` controls:

```python
result = solver.solve_eigenvalue(
    tol_k=1e-6, tol_source=1e-5, tol_balance=1e-5,
)
```

`tol_k` controls absolute eigenvalue change; `tol_source` controls successive
normalized fission-source change across all spatial moments; `tol_balance`
controls the full residual. Constructor `rtol` controls each linear solve and
auxiliary reference recovery separately. Existing calls using only `rtol`
retain their previous stopping behavior. Numerical convergence does not replace
positive-profile, interface and physical conservation checks.

Use `garnet.branch_nem_validation.assess_nem_solution(solver, result,
balance_tolerance=1e-5)` to check a receiving solution before interpreting its
accuracy. It checks the unchanged solver configuration, finite fields,
nonnegative physical fluxes, positive total fission production, full-matrix
mean balance, corrected-face continuity and exposed Robin boundaries. It also
recomputes the degree-specific weak/moment/profile diagnostics instead of
trusting stored result diagnostics. Reconstructed profile extrema are numerical;
they are not interval-certified bounds. These checks establish numerical
acceptability, not transport accuracy or nuclear-data equivalence.
