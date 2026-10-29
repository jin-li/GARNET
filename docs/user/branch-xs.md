# Static branch libraries

The experimental `garnet.branch_xs` module provides immutable table states,
exact selection, bounded absolute interpolation and fresh CPU NEM operators.
It is separate from `TabulatedFeedback`: it does not clamp requests or apply
ratios to an existing solver. Native NEM branch states use three
Cartesian axes and six whole-node faces in the order `x-, y-, x+, y+, z-, z+`.
Generic table interpolation does not add support for arbitrary spatial topology. Run the data-free example from the checkout:

```sh
python examples/isonees_features/branch_xs.py
```

## Prepare and evaluate a library

A table state identifies the physical configuration, conditions, ordered
regions, energy boundaries and material arrays. Face factors use
`(region, 6, group)` with sides `x-, y-, x+, y+, z-, z+`. Fission matrices use
`(region, emitted_group, source_group)`. Record the reaction convention,
diffusion prescription, geometry, nuclear data and factor method in metadata.


Native Serpent `MICRO_NG`/`MICRO_E` fields identify the intermediate micro-group
structure separately from the final `MACRO_NG`/`MACRO_E` structure. GARNET
validates this optional pair and derives immutable `state.microgroup_identity`
from all retained native records, including older saved libraries. Every
printed micro-group boundary must match exactly across interpolation vertices; equal counts
or macro groups are insufficient. A Serpent version compatibility policy cannot
waive this check. Known and unknown micro-grids cannot be mixed in an interpolation element.
Unrelated states may share an archive for exact selection. Wholly unknown
synthetic/legacy states remain usable with an explicit `status: unknown`
identity; this does not establish scientific compatibility. Caller metadata
cannot override the identity recovered from retained native fields. Native-free
manual states may explicitly declare the validated identity as caller evidence.
The identity also records the current `inf` diffusion formulation.

Serpent uses intermediate micro-group constants when collapsing to macro groups.
Its manual explains that, for infinite-spectrum constants, choosing the
micro-group structure has virtually no effect beyond floating-point precision.
This identity guard conservatively records the processing structure; it does
not assert that a grid difference causes a significant physics bias. See the
[Serpent group-constant generation guide](https://serpent.vtt.fi/docs/user_guide/gc_generation.html#infinite-and-critical-spectrum-calculation).

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

For additional application-specific constraints, use the explicit metadata
namespace `compatibility_identities`, for example:

```python
metadata = {"compatibility_identities": {
    "processing_route": "local-collapse-v2",
    "composition": {"salt_recipe": "recipe-7", "option": None},
}}
```

The namespace must be a mapping of nonempty string names to JSON-compatible
identity values. The library compares the entire mapping, including nested
keys: an omitted identity differs from an explicit null. Namespace omission
also differs from an empty mapping. Keep ordinary descriptive provenance
outside it. Legacy identity keys remain checked with their historical presence
rules: closure keys distinguish omission/null, while other legacy keys compare
missing and null equally. Neither matching metadata nor custom identities
establish scientific equivalence. The namespace and explicit version policy
survive JSON save/load without a separate key registry.

Create segments or triangles explicitly. Use separate libraries/elements for
ARO/ARI configurations (normally labelled `aro`/`ari`), different bank
environments and different geometry or nuclear-data families. A five-point
temperature star covers a diamond in its transformed coordinates, not its
bounding rectangle. A request beyond a supplied segment or triangle raises an
error. Unvalidated interpolation remains labelled unvalidated.

`select(state_id)` returns the exact immutable state. `evaluate(conditions,
configuration=..., family=...)` returns absolute arrays and source weights.
Inspect those weights and the provenance before interpreting a prediction.
`TabulatedModeratorPath` supplies a caller-defined bounded density path:

```python
from garnet.branch_xs import TabulatedModeratorPath

path = TabulatedModeratorPath(
    temperatures=(300., 400., 500.),       # K, increasing
    densities=(1.0, .95, .85),             # g/cm3, strictly monotone
    path_id="my-fluid-table-v1", pressure_bar=20.,
)
```

Supply at least two positive finite knots and an explicit nonblank identity.
Densities may increase or decrease; interpolation uses shape-preserving PCHIP
and rejects extrapolation. Omit `pressure_bar` when it is unknown or not relevant;
a supplied pressure must be positive finite and is provenance only. This class
does not calculate an equation of state. `BenchmarkModeratorPath` remains the
historical 155-bar PRATIC wrapper with unchanged default knots and positional
argument order. Migrate other fluids or pressures to `TabulatedModeratorPath`;
no PRATIC geometry, material names or two-group energy structure is required.
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

## Metrics and application acceptance policy

`garnet.branch_validation` computes reactivity differences in pcm, temperature
coefficients in pcm/K, interval rod worth in pcm/cm, and normalized power RMS
as a fraction. Acceptance thresholds are caller policy. Historical defaults
remain for compatibility: 20 pcm for reconstruction interpolation, 0.01 for
power RMS, and 0.05 relative for coefficient convergence/rod worth. Choose
thresholds explicitly for a new application. To compute metrics without
acceptance:

```python
from garnet.branch_validation import compare_reconstruction_chain

report = compare_reconstruction_chain(
    k_interpolated, k_exact, k_ce,
    target_pcm=None, power_target=None,
    power_interpolated=p_interpolated, power_exact=p_exact, power_ce=p_ce,
)
assert report["status"] == "descriptive"
```

Keep three checks separate: pointwise eigenvalue and power interpolation
against the exact reconstructed solver; predicted coefficient accuracy against
continuous-energy (CE) transport or another reference over the same physical
endpoint interval; and full-step versus half-step coefficient convergence.
A relative target for coefficient accuracy is caller policy and must name its
reference. Agreement with the exact reconstructed solver, or convergence as
the interval shrinks, does not establish accuracy against CE over that interval.
Finite-interval coefficients are secants, not local derivatives.

Nominal agreement with a reference can coexist with inconclusive reference
precision. A screen that treats the prediction as deterministic and propagates
only native Monte Carlo endpoint errors is conditional: it omits uncertainty
in training XS and ADFs. Report that scope and unresolved precision separately
from nominal relative error; neither establishes application qualification.

The chain forwards `power_target` to the interpolated-versus-exact power
comparison. Transport comparisons remain descriptive. A requested metric with
no threshold prevents an overall pass; a failed assessed metric still fails.
For individual helpers, use `normalized_power_rms(..., target=None)`,
`coefficient_convergence(..., relative_target=None)` or
`assess_rod_worth(..., relative_target=None)`. Descriptive reports retain input,
physical-bin, uncertainty and coverage validation. They never qualify a result.

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
Low/high mean actual temperature; densities follow the supplied monotone path.
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

The `build_thermal_star` constructor also accepts `compatible_serpent_versions`
and forwards that policy to its `BranchLibrary`. This permits an explicitly
accepted executable-version pair without waiving training roles, nuclear-data
identity or the caller-supplied qualification policy.

## Experimental geometry-aware rod interpolation

`garnet.branch_rod.evaluate_rod_occupancy` offers a separate experimental
alternative to uniform segment weights. Use an explicit one-dimensional
`BranchLibrary` with an identity-transformed rod coordinate. It retains every
public library domain, fixed-condition, version and closure compatibility guard.
An exact anchor is returned with its original arrays, native records and
qualification; the geometry callback is not invoked for that anchor.

```python
from garnet.branch_rod import evaluate_rod_occupancy

# Application-specific geometry; return exactly one fraction per region.
# Coordinate units must match the library and this geometry implementation.
def occupancy(conditions, region_ids):
    position_cm = conditions["rod_insertion_cm"]
    return {region: geometry.absorber_fraction(region, position_cm)
            for region in region_ids}

prediction = evaluate_rod_occupancy(
    rod_library, {"rod_insertion_cm": 100.,
                  "configuration": "bank-a", "family": "rod"},
    occupancy=occupancy, geometry_id="bank-a-physical-regions-v1",
)
```

The example assumes caller-provided `geometry` and `rod_library`; it is not a
PRATIC geometry prescription. The provider receives immutable conditions and
ordered region IDs. It must return a mapping with exactly those IDs and finite
real scalar fractions in `[0, 1]`. Booleans, arrays and missing/extra keys are
rejected. Target occupancy must lie between the two endpoint occupancies in
each region, including regions with equal endpoint occupancies. No clipping or
extrapolation is performed. A region whose endpoint occupancies differ uses
its local occupancy fraction to interpolate every XS channel and the logarithm
of each face factor. A region with unchanged occupancy keeps the global rod
coordinate weight; unchanged occupancy does not imply unchanged XS in a
coupled core. Exterior face factors remain exactly unity.

Inspect `prediction.metadata["rod_occupancy"]` for the geometry identity,
source roles and IDs, endpoint/target occupancies and local right weights.
`prediction.weights` retains the original global segment weights as lineage;
it is not the per-region interpolation weight. Original source roles and native data
are unchanged; interpolated metadata has `role="prediction"`. Interpolated results carry no native record and are labelled
`experimental-rod-occupancy-unvalidated`. They cannot be inserted as original
library states. The method does not establish transport reaction-rate
conservation, full-range rod qualification or differential rod-worth accuracy.

The geometry callback is not stored by `BranchLibrary.save`; its JSON format
is unchanged. Preserve the callback source/version or hash, physical geometry,
region mapping, coordinate units, library snapshot and resulting metadata with
each study. A named geometry identity is a caller declaration, not an
independent verification of geometry correctness. Freeze the mapping and
interpolation method before withholding transport states for qualification.
