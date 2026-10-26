# Diffusion transients through the model API

`Model`, `CylindricalModel`, and `TriReactor` share the same neutron-transient
interface and return `TransientModelResult`. They also support thermal
feedback through `configure_thermal` and the coupled methods. The underlying
diffusion and time-integration solvers remain available for specialized work.

## A timed material change

```python
from garnet import CylindricalModel, Kinetics, PWR_TWO_GROUP

fuel = PWR_TWO_GROUP.replace(name="fuel")
core = (CylindricalModel(radius=60, height=120, cells=(12, 24))
        .fill(fuel).set_boundary(r="vacuum", z="vacuum")
        .set_kinetics(Kinetics([1e7, 3e5], [0.0065], [0.08])))
steady = core.steady(device="auto")
poisoned = fuel.replace(sigma_a=fuel.sigma_a * [1.0, 1.003])

run = core.transient(
    t_end=0.1, dt=0.01, time_scheme="bdf2", device="auto",
    initial_steady=steady,
    materials_at=lambda t: {"fuel": poisoned} if 0.035 <= t < 0.08 else {},
    event_times=[0.035, 0.08],
)
print(run)
print(run.power_at([0.02, 0.06, 0.1]))
```

The initial eigenvalue `k0` critically adjusts the fission source, so an
unperturbed calculation stays at `P/P0 = 1` to solver accuracy.
`materials_at(t)` accepts a partial
name/index mapping or a complete ordered list; each mapping is relative to the
base model. Return cached material objects while the constants are unchanged.

For moving assignments, `state_at(t)` returns a cached model with the same
grid, physical dimensions, boundaries, active cells and material count. Use
`at(t)` for the advanced two/four-array problem specification. These three
callbacks are mutually exclusive. The
[complete example](../examples/cylindrical_transient.py) includes a reflector
and accepts `--device cpu`, `--device gpu`, and `--adaptive`.

## Time steps, events and adaptive BDF

`dt` is a scalar fixed width, an explicit list of widths summing to `t_end`,
or the initial width for adaptive BDF. Backward Euler remains the default.
Select `time_scheme="bdf2"` through `"bdf6"` explicitly for higher orders.

`event_times` must lie strictly inside `(0, t_end)`. For fixed BDF steps, the
API splits the schedule at those times and includes a final short interval if
needed. Adaptive BDF hits the same times exactly. Both paths use the existing
solver's history-restart behavior at the event. This prevents a control jump
from being silently crossed by a long step. The callback defines the value at
the event itself; use `t >= event` when the new state begins there.

```python
adaptive = core.transient(
    t_end=0.1, dt=0.005, time_scheme="bdf3", device="auto",
    materials_at=lambda t: {"fuel": poisoned} if t >= 0.035 else {},
    event_times=[0.035],
    adaptive_bdf={"rtol": 1e-3, "min_dt": 1e-7, "max_dt": 0.01},
)
```

Supplying `adaptive_bdf` selects monolithic stepping unless `step_solver` was
explicitly supplied. If the time scheme is omitted, it uses adaptive BDF1
(backward Euler). Existing low-level `bdf_restart_times` and
`step_alignment_times` are still available. Use `event_times` or
`bdf_restart_times`, not both. Nonuniform/event schedules require BDF or FTM.
Controller tolerances govern a local defect estimate; assess time convergence
for the observable and transient of interest.

Experimental `time_scheme="ftm2"` and `"ftm-bdf2"` also use this interface
with fixed or explicit widths. The latter supplies a damped, full-state
transformed BDF2 recurrence. Both refuse `adaptive_bdf`; see the
[damping checkpoint](../benchmark-results/exponential-second-order/CHUNK3.md)
for safeguards, precursor differences and the measured limits.

Constructor settings such as `dtype` belong in `solver_kwargs`. Common
`precond_degree`, `precond_dtype`, and `linear_solver` controls are also direct
arguments. `tol_step`, `time_scheme`, `adaptive_bdf`, `linsolve_kwargs`, and
`multigroup_kwargs` go to the time integrator. Mixed preconditioning is still
opt-in and is not supported by monolithic steps.

## Results and progress

The common result provides:

| Field or method | Meaning |
|---|---|
| `steady`, `k0` | Public time-zero steady report and critical-adjustment eigenvalue |
| `times`, `power` | Accepted times in seconds and normalized `P/P0`, including t=0 |
| `step_widths`, `n_steps` | Actual accepted widths and step count, for fixed or adaptive runs |
| `flux`, `flux_numpy` | Final scalar flux as NumPy; its grid axes are preserved |
| `precursors_numpy` | Final precursor fields copied to NumPy on request |
| `peak_power`, `peak_time`, `final_power` | Recorded extrema and final power |
| `power_at(times)` | Linear interpolation inside the recorded span; no extrapolation |
| `raw` | Original solver result, including device-resident flux and precursors |

Solver telemetry such as `time_orders`, `local_errors`, `rejected_steps`,
`step_iterations`, and `initial_state_reused` is forwarded from `raw` and
appears in tab completion. `dt` retains the first recorded width for
compatibility; use `step_widths` when the schedule varies. `power_at` samples
the recorded history; it is not a high-order dense-output formula.

```python
accepted = []
run = core.transient(
    t_end=0.1, dt=0.01, device="auto",
    on_step=lambda t, flux, power: accepted.append((t, float(power))),
)
```

`on_step` runs once after each accepted neutron step and never for rejected
attempts or the initial state. The flux stays on its current device. Treat
callback arrays as read-only and copy them if retaining snapshots. Ordinary
result reports transfer the final scalar flux to NumPy; device workflows can
use `run.raw.flux` and `run.raw.precursors`.

Public `initial_steady` results must be converged forward diffusion solutions
matching the actual t=0 grid, boundary conditions, cross sections, maps, and
volume mixing. This is checked against saved metadata, including after the
original builder has been edited. Raw solver results retain their existing
caller-owned physical-compatibility contract. Reusing an initial eigenpair is
not a general transient restart/checkpoint facility.

Public steady results built with raw `xs_update`/`xs_update_at` field hooks are
rejected for automatic reuse: their effective spatial cross sections are not
captured by material metadata. The same restriction applies when the new
transient uses `xs_update_at`. Express the t=0 constants as model materials,
or use the low-level result with its explicit caller-owned compatibility check.

## Cylindrical r-z geometry

`CylindricalModel(radius, height, cells=(nr, nz))` represents a full body of
revolution. `add_region(material, r=(...), z=(...))` paints cell centres in an
annular axial region. `add_box` is an alias. `r="vacuum"` sets the cylinder
wall; `z=("reflective", "vacuum")` sets the lower and upper axial faces.
The r=0 face has zero area and is a natural symmetry axis.

`material_map` has shape `(nr, nz)` and scalar flux has shape
`(groups, nr, 1, nz)`. `radial_centers` and `axial_centers` provide coordinates.
`grid.cell_volumes` and steady `result.cell_volumes` contain full annular
volumes `π(r_outer² − r_inner²) Δz`. Thus `result.power_density(total_power=...)`
integrates to the requested full-core watts; there is no slice-power factor.
The existing scalar `Grid.cell_volume` retains its rectangular stencil meaning.

To use an existing structured benchmark definition without repainting:

```python
from garnet import Model

core = Model.from_grid(grid, materials, material_map, bc=bc, kinetics=kinetics)
run = core.transient(t_end=1.0, dt=0.01, at=problem_at, device="cpu")
```

`Model.from_grid` returns `CylindricalModel` for a cylindrical grid and copies
the material map. Cartesian input grids retain all three supplied extents and
boundary axes, including one-cell-thick directions. This preserves the
original benchmark problem rather than inferring collapsed-axis boundaries.

## Thermal coupling

The [thermal configuration](model_api.md#thermal-coupling) works for all three
grid types. `coupled_transient` and `quasistatic_transient` now accept named
`materials_at` as well as cached `state_at` and the raw `problem_at` callback.
Only one perturbation callback may be supplied. Material count/order and the
configured thermal/feedback indexing stay fixed across states.

For coupled progress, `on_step(t, flux, power, temperature)` observes an
accepted neutron step after any due thermal exchange. Temperature is held
between those exchanges. Arrays remain on their current device and should be
treated as read-only.

Adaptive neutron controls enter through
`coupled_transient(transient_kwargs={"time_scheme": "bdf2",
"adaptive_bdf": {"rtol": 1e-3}})`. Monolithic/BDF1 defaults are filled in as
for neutron-only runs. Thermal exchange times are aligned by the coupled
driver without restarting BDF history. For fixed stepping, `dt_thermal` must
remain an integer multiple of `dt`.

## Test migration and evidence

Six existing test modules were reworked where the public diffusion API can
express the same calculation: cylindrical analytics, ANL-7416, TWIGL/LMW,
point-kinetics/BDF, per-material kinetics, and triangular prompt-jump/inhour
checks. Their 58 test functions and 149 assertion statements/calls are
preserved. Specialized solver-internal tests remain active at the lower level.

Original modules are retained byte-for-byte under `tests/legacy/` and excluded
from default collection. Run them explicitly:

```bash
python -m pytest tests                         # current API and solver checks
python -m pytest tests/legacy --run-legacy      # 62 preserved low-level cases
python -m pytest tests --run-legacy             # current and preserved checks
```

The [new workflow tests](../tests/verification/test_transient_model_api.py)
cover common results, callbacks, event schedules, adaptive-step rejection,
time-zero compatibility, physical annular volumes, and full/quasi-static
cylindrical coupling. Recorded results and reproduction details are in the
[V&V report](verification_validation.md#transient-api-and-cylindrical-check).
