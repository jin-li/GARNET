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

Initialization defaults account for the arithmetic precision:
`tol_k=max(1e-8,8*eps(dtype))` and
`tol_source=max(1e-7,64*eps(dtype))`, with the solver's matching default
balance test. A nonempty `steady_kwargs` remains an explicit accuracy request
and is passed unchanged; asking FP32 to meet FP64-scale tolerances can fail.
A supplied `initial_steady` must be converged tightly enough for the transient
being interpreted. A small steady residual can cause a visible rapid
relaxation in a stiff kinetic problem, so verify its physical balance rather
than checking only the eigenvalue change.

The initial eigenvalue normalizes fission so the reference is stationary. It does not assert that the unadjusted material is physically critical. A changed `materials_at(t)` must return the full list in `model.materials` order; return cached unchanged objects between real changes. Distinct controllable regions need distinct initial material objects. The low-level geometry callback has a separate `(materials,map)` contract.

Use [TWIGL](https://github.com/jin-li/GARNET/blob/main/examples/twigl_benchmark.py) and [rod drop](https://github.com/jin-li/GARNET/blob/main/examples/transient_rod_drop.py) for perturbation patterns. At a discontinuity, align the step and use the supported BDF restart controls. Check time refinement independently of the neutron linear/source tolerances. Standalone nonuniform/adaptive schedules and coupled schedules have different constraints.

## Thermal coupling

Build a `TriReactor`, then call `configure_thermal` with material-name mappings of `ThermalMaterial` and optional `FeedbackSpec`. The [custom reactor example](https://github.com/jin-li/GARNET/blob/main/examples/custom_tri_reactor.py) gives the full sequence, including `total_power`, sink data and ambient boundary conditions. `coupled_steady` imposes domain power and returns its temperature-dependent shape and eigenvalue.

For evolution use `core.coupled_transient(t_end=..., dt=..., dt_thermal=..., device=...)` after thermal/kinetics configuration. The fixed-schedule interface needs compatible thermal windows. Pass cached compatible frames through the documented state callback for moving controls. Report total modeled power, geometry/thickness, feedback reference temperature and units.

`profile=True` returns phase times and counters. Opt-in FP32 preconditioning retains FP64 physical state and convergence tests; inspect fallback counts and compare against native precision. Do not infer a speedup from iteration count alone. See [coupled transient](https://github.com/jin-li/GARNET/blob/main/examples/hpmr_coupled_transient.py) and [coupling details](../coupling.md).

## Quasi-static acceleration

`quasistatic_transient` is useful when the spatial shape changes slowly enough to correct on larger macro intervals. Choose `shape_dt`, residual thresholds and adjoint reuse controls explicitly. Inspect shape updates, predictor disagreement, fallback reasons and adjoint residuals. `.power` includes the changing shape normalization; `.amplitude` alone is not physical power.

Compare against a full spatial transient on representative intervals, especially rapid control motion. A fixed-shape approximation has no shape correction and needs stronger justification. [Quasi-static benchmark](https://github.com/jin-li/GARNET/blob/main/examples/hpmr_quasistatic_benchmark.py) demonstrates comparison and telemetry.

## Frequency-domain response

The [transfer-function example](https://github.com/jin-li/GARNET/blob/main/examples/noise_transfer_function.py) is the recommended first noise run. It compares a homogeneous reflected calculation against point-kinetics theory. `NoiseSolver` uses angular frequency in rad/s; convert frequency in Hz with $\omega=2\pi f$. Its complex response encodes amplitude and phase under the package's $+i\omega$ convention.

Choose nonzero frequencies and sufficiently small perturbations. Read the global kinetics shape restrictions before reusing a transient input table. A noise solution is a linear response about its reference, not a finite-amplitude trajectory. See [methodology](../methodology/time-coupling.md) for the delayed effective spectrum and DC limit.
