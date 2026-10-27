# Transients, thermal feedback and noise

## Begin with a stationary check

This complete synthetic example starts from an eigenstate and applies no perturbation. Its critical-adjusted power should remain one:

```python
import numpy as np
from garnet import Material, Model
fuel = Material(name="fuel", diffusion=[1.2], sigma_a=[0.025],
                nu_sigma_f=[0.030], sigma_s=[[0.0]], chi=[1.0])
model = (Model(size=(20.0,), cells=(8,)).fill(fuel)
         .set_boundary("reflective")
         .set_kinetics(velocities=[220000.0], beta=[0.0065], decay=[0.08]))
run = model.transient(t_end=0.1, dt=0.01, device="cpu")
print(run.k0, run.times, run.power)
assert np.allclose(run.power, 1.0, atol=1e-5)
```

Lengths use cm, velocities cm/s, decay constants s$^{-1}$ and all schedule
widths/times seconds. Delayed fractions are dimensionless.

Initialization defaults account for the arithmetic precision:
`tol_k=max(1e-8,8*eps(dtype))` and
`tol_source=max(1e-7,64*eps(dtype))`, with the solver's matching default
balance test. A nonempty `steady_kwargs` remains an explicit accuracy request
and is passed unchanged; asking FP32 to meet FP64-scale tolerances can fail.
A supplied `initial_steady` must be converged tightly enough for the transient
being interpreted. A small steady residual can cause a visible rapid
relaxation in a stiff kinetic problem, so verify its physical balance rather
than checking only the eigenvalue change.

The initial eigenvalue normalizes fission so the reference is stationary. It
does not assert that the unadjusted material is physically critical.
`materials_at(t)` accepts a partial name/index mapping relative to the base
model, or a complete list in `model.materials` order. Return cached unchanged
objects between real changes. Distinct controllable regions need distinct
initial material objects. Alternatively, `state_at(t)` returns a cached model
with compatible geometry, boundaries, active cells and material count. The
advanced `at(t)` callback accepts the low-level two/four-array problem contract;
these callbacks are mutually exclusive.

Results include accepted `times` in seconds, relative `power` ($P/P_0$),
`step_widths`, final NumPy flux, and device fields through `raw`. `power_at`
linearly interpolates the recorded power without extrapolation; it is not
high-order dense output. `on_step(t, flux, power)` observes accepted endpoints
only. Treat its device arrays as read-only and copy retained snapshots.
An `initial_steady` must match the actual initial geometry, boundaries,
materials and mixing; reusing it is not a general restart facility. Steady
results made with raw spatial XS hooks cannot be automatically reused.
See the [transient API guide](../transient_api.md) for these checks and the
cylindrical array/annular-volume contract.

`Model`, `CylindricalModel` and `TriReactor` share this diffusion-transient
interface. Cylindrical maps have shape `(nr,nz)` and scalar flux has shape
`(G,nr,1,nz)`. Its cell volumes are full annular volumes, so prescribed power
belongs to the full body of revolution. Do not apply a slice-power multiplier.

Use [TWIGL](https://github.com/jin-li/GARNET/blob/main/examples/twigl_benchmark.py) and [rod drop](https://github.com/jin-li/GARNET/blob/main/examples/transient_rod_drop.py) for perturbation patterns. At a discontinuity, align the step and use the supported BDF restart controls. Check time refinement independently of the neutron linear/source tolerances. Standalone nonuniform/adaptive schedules and coupled schedules have different constraints.

## Explicit schedules and adaptive BDF

`dt` is a scalar width, an explicit sequence summing to `t_end`, or the initial
adaptive width. `event_times` lies strictly inside the simulation interval and
aligns steps with state changes. BDF events restart history; define a new BDF
state with `t >= event`. For example, continuing the synthetic model above:

```python
poisoned = fuel.replace(sigma_a=fuel.sigma_a * 1.003)
perturbed = model.transient(
    t_end=0.1, dt=0.005, device="cpu", time_scheme="bdf2",
    materials_at=lambda t: {"fuel": poisoned} if t >= 0.035 else {},
    event_times=[0.035],
    adaptive_bdf={"rtol": 1e-3, "min_dt": 1e-7, "max_dt": 0.01},
)
print(perturbed.step_widths, perturbed.power_at([0.02, 0.06, 0.1]))
```

`adaptive_bdf` selects monolithic stepping unless overridden; if the scheme
is omitted it uses BDF1. Orders 2 through 6 are explicit choices. Local defect
tolerances do not bound accumulated error or interpolation error. Refine the
maximum width for the required output resolution as well as the state tolerance.

## Experimental frequency transforms

The diffusion schemes `ftm`, `ftm2` and `ftm-bdf2` estimate one shared growth
frequency from the accepted neutron population. They accept fixed/nonuniform
widths and explicit events, with safeguards reported in `time_orders` and
`ftm_fallback_reasons`. They are experimental; the tested TWIGL comparisons do
not establish a runtime advantage over tuned adaptive BDF.

`ftm` is first order with analytic frozen-exponential precursor production;
`ftm2` uses a transformed trapezoidal neutron step and linear transformed
precursor production, which can ring at prompt events. `ftm-bdf2` transforms
both neutron and precursor BDF2 histories, providing stiff-mode damping with
BDF1 startup/restarts. It does not guarantee positivity. Use monolithic
stepping for the supported adaptive transform workflow:

```python
transformed = model.transient(
    t_end=0.1, dt=0.005, device="cpu", time_scheme="ftm-bdf2",
    adaptive_ftm={"rtol": 1e-4, "output_rtol": 1e-4,
                  "min_dt": 1e-10, "max_dt": 0.01},
    materials_at=lambda t: {"fuel": poisoned} if t > 0.035 else {},
    event_times=[0.035],
)
```

For FTM, the event endpoint belongs to the preceding segment: use `t > event`
for the interval leaving it. This differs from the BDF callback above.
`adaptive_ftm` supports `ftm2` and `ftm-bdf2`, and refuses `adaptive_bdf`.
It accepts the two-half-step physical state after comparison with a full step;
rejected attempts roll back state and histories. `output_rtol` additionally
checks midpoint curvature against linear output in flux, precursors and power.
Its local chord check does not certify all interior times or a power jump.

The opt-in `estimator="history"` can replace the full trial with a physical
history predictor when safe; `"history-audit"` records both defects while
retaining step-doubling decisions. Startup, events and unsafe history use step
doubling. Counters include rejected and discarded work. Inspect trial,
estimator, frequency-refinement and output-error telemetry in the
[API guide](../transient_api.md). Adaptive FTM excludes thermal coupling,
implicit feedback and external error state. Fixed-step coupled transforms have
separate heat quadrature limits; neutron order does not establish thermal order.

## Thermal coupling

Build a `TriReactor`, then call `configure_thermal` with material-name mappings of `ThermalMaterial` and optional `FeedbackSpec`. The [custom reactor example](https://github.com/jin-li/GARNET/blob/main/examples/custom_tri_reactor.py) gives the full sequence, including `total_power`, sink data and ambient boundary conditions. `coupled_steady` imposes domain power and returns its temperature-dependent shape and eigenvalue.

For evolution use `core.coupled_transient(t_end=..., dt=..., dt_thermal=..., device=...)` after thermal/kinetics configuration. The fixed-schedule interface needs compatible thermal windows. Pass cached compatible frames through the documented state callback for moving controls. Report total modeled power, geometry/thickness, feedback reference temperature and units.
Adaptive BDF controls enter through `transient_kwargs`, for example
`{"time_scheme": "bdf2", "adaptive_bdf": {"rtol": 1e-3}}`. The coupled driver
aligns neutron steps with thermal exchange times without restarting history;
fixed stepping still requires `dt_thermal` to be an integer multiple of `dt`.
Adaptive FTM is not supported by this coupled path.

`profile=True` returns phase times and counters. Opt-in FP32 preconditioning retains FP64 physical state and convergence tests; inspect fallback counts and compare against native precision. Do not infer a speedup from iteration count alone. See [coupled transient](https://github.com/jin-li/GARNET/blob/main/examples/hpmr_coupled_transient.py) and [coupling details](../coupling.md).

## Quasi-static acceleration

`quasistatic_transient` is useful when the spatial shape changes slowly enough to correct on larger macro intervals. Choose `shape_dt`, residual thresholds and adjoint reuse controls explicitly. Inspect shape updates, predictor disagreement, fallback reasons and adjoint residuals. `.power` includes the changing shape normalization; `.amplitude` alone is not physical power.

Compare against a full spatial transient on representative intervals, especially rapid control motion. A fixed-shape approximation has no shape correction and needs stronger justification. [Quasi-static benchmark](https://github.com/jin-li/GARNET/blob/main/examples/hpmr_quasistatic_benchmark.py) demonstrates comparison and telemetry.

## Frequency-domain response

The [transfer-function example](https://github.com/jin-li/GARNET/blob/main/examples/noise_transfer_function.py) is the recommended first noise run. It compares a homogeneous reflected calculation against point-kinetics theory. `NoiseSolver` uses angular frequency in rad/s; convert frequency in Hz with $\omega=2\pi f$. Its complex response encodes amplitude and phase under the package's $+i\omega$ convention.

Choose nonzero frequencies and sufficiently small perturbations. Read the global kinetics shape restrictions before reusing a transient input table. A noise solution is a linear response about its reference, not a finite-amplitude trajectory. See [methodology](../methodology/time-coupling.md) for the delayed effective spectrum and DC limit.
