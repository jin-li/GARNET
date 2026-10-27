# Shared-frequency exponential diffusion prototype

Date: 2026-10-25. Decision: **retain as an experimental reference; do not
promote this first-order, lagged-frequency variant as a general acceleration.**

The prototype works in the actual diffusion solver, including multigroup
precursor elimination and thermal coupling. The primary comparisons are
backward Euler and adaptive BDF, as requested. The measured results do not
support a general speedup over those methods. IQS and GPU performance have
not been evaluated in this study.

The [follow-up proposal review](../../docs/exponential_proposal_review.md)
adds CN/error-versus-work results, an isolated frequency-correction probe,
and a more precise distinction between exponential BE and exponential theta.

## What was implemented

- `TransientSolver.solve(time_scheme="ftm")`: one frozen frequency shared by
  all cells and energy groups, normally estimated from two accepted neutron
  populations and the actual previous width.
- Separate time-diagonal and physical history-source multipliers,
  `1 + omega*h` and `exp(omega*h)`. BE/BDF reuse their existing time
  coefficients without allocating a duplicate RHS diagonal.
- Analytic precursor integration for the assumed exponential end-of-step
  source, retaining total/prompt/delayed-spectrum conventions and material
  kinetics. Stable scalar coefficients handle `omega + lambda -> 0`.
- A configurable exponent bound (`abs(omega*h) <= 0.5` by default), with
  actual BE precursor/flux/heat coefficients outside the bound. A nonpositive
  or nonfinite total neutron population causes a failure requesting a smaller
  width. These guards are not an LTE estimator or a proof of unconditional
  positivity/stability.
- Exponential heat-source quadrature in the split conduction coupler and
  the fully implicit LRA adiabatic driver. Absolute temperature is not
  exponentially transformed. Endpoint quadrature remains available as an
  ablation control.
- Explicit nonuniform coupled schedules, validated to end at every thermal
  exchange, allowing diagnostic replays of adaptive BE/BDF grids.
- Accepted frequencies, heat-source weights, BE-fallback reasons and operator
  rebuild counts, alongside the existing Krylov/feedback/rejection counters.

This is the minimal shared-frequency prototype from the
[development plan](../../docs/exponential_transformation_plan.md), not a
reproduction of SIMULATE's full local/group-frequency and frequency-correction
algorithms. FTM has no independent adaptive controller in this implementation.
`adaptive_bdf` remains BDF-only; fixed or supplied nonuniform FTM schedules are
supported. Face-coupling reuse under time-diagonal changes remains future work.
For FTM, `bdf_restart_times` clears population history before the first step
leaving an aligned event, matching adaptive BDF's convention; the legacy
fixed-BDF restart convention is unchanged.

```python
result = solver.solve(
    t_end=1.0, dt=0.001, time_scheme="ftm", step_solver="monolithic",
)
print(result.step_frequencies, result.ftm_fallback_reasons)
```

`ftm_kwargs={"frequency": 0.0}` selects the zero-frequency, exact-precursor
control. This deliberately differs from BE's rational precursor update.
For verification, a prescribed nonzero frequency is also accepted.

## Comparison protocol

The homogeneous cases use the real FV/FGMRES pipeline on a reflective 4 by 4
by 1 mesh, one energy group and one precursor family. Their physics reduces
exactly to independent point kinetics. The parameters are `v=2.2e5 cm/s`,
`Sigma_a=0.03/cm`, `nuSigma_f=0.035/cm`, `D=1.3 cm`, `beta=0.0065`,
`lambda=0.08/s`. Reactivity rises to 0.002, either as a step or over a 0.2 s
ramp, and the calculation ends at 1 s.

Coupled cases use the actual split conduction/Doppler driver. The initial
temperature is 600 K, sink temperature 590 K, sink coefficient
1 W/(cm3 K), initial heating 10 W/cm3, and Doppler coefficient
0.03 K^-1/2. Heat capacities are 10 and 0.5 J/(cm3 K), labelled slow and
fast. Thermal exchange widths are 20 and 5 ms. These are manufactured
mechanism/stress cases, not HP-MR material data or full-core throughput tests.

Independent SciPy Radau ODE references include the physical feedback and
energy integral. The reference is checked against tighter ODE tolerances.
Power is compared on 4,001 common output points with the **same linear output
interpolation for every method**, and endpoint-only errors are also saved.
Temperature is compared at actual thermal exchange endpoints. The heat-energy
budget applies to coupled cases; uncoupled source-integral errors are reported
as diagnostics. Spatial shape changes are outside these uniform-case tests.

Two error screens are recorded:

| Screen | Maximum power error | Coupled energy error | Temperature error |
|---|---:|---:|---:|
| Exploratory coupled/uncoupled matrix | 0.5% | 0.5% | 0.1 K |
| Tighter uncoupled matrix | 0.1% | diagnostic only | not applicable |

Fixed-width ladders, adaptive BE, and automatic BDF1--5 tolerance ladders are
tested. Comparisons select the fastest **tested** configuration passing the
same declared budgets. They do not establish a continuous optimum over all
possible widths/tolerances, and equal `rtol` is not treated as equal accuracy.

CPU measurements use FP64, one BLAS/OMP thread, three interleaved timed
repetitions, and reused initial eigenpairs/hot equilibria. March time includes
step setup, source integration, feedback and rejected work; initial reference
and steady-state setup is reported separately. Final timing runs were
serialized, with no concurrent regression jobs. These tiny cases emphasize
algorithm/software overhead, and cannot predict GPU or production mesh speed.
Measurements use the modified working tree based on `b496309`; recorded source
SHA256 hashes identify the tested implementations rather than claiming that
the new method exists in that base commit.

## CPU results

<!-- RESULTS:START -->
Best tested CPU march times in seconds, three-run medians. The tighter
screen requires maximum power error <= 0.1%.

| Uncoupled case | Fixed BE | Adaptive BE | Adaptive BDF1--5 | Fixed FTM |
|---|---:|---:|---:|---:|
| uniform-step | 1.935 | 0.578 | 0.076 | 4.478 |
| uniform-ramp | 0.564 | 0.233 | 0.149 | 0.656 |

For the step, accepted/rejected counts are backward-euler: 2000/0, bdf1: 482/5, bdf5: 45/2, ftm: 4000/0.

The exploratory coupled screen uses power/energy errors <= 0.5% and
temperature error <= 0.1 K:

| Coupled case / thermal width | Fixed BE | Adaptive BE | Adaptive BDF1--5 | Fixed FTM |
|---|---:|---:|---:|---:|
| coupled-slow / 20 ms | 0.172 | 0.188 | 0.186 | 0.192 |
| coupled-slow / 5 ms | 0.405 | 0.471 | 0.472 | 0.420 |
| coupled-fast / 20 ms | no passing row | no passing row | no passing row | no passing row |
| coupled-fast / 5 ms | 0.390 | 0.437 | 0.457 | 0.397 |
<!-- RESULTS:END -->

The prompt-step result is the important limitation: a lagged frequency does
not resolve the initial mixture of prompt and delayed modes accurately enough
to use large widths. Adaptive BDF exploits temporal order and scheduling;
FTM's first-order transformed derivative still needs substantial refinement.

At identical accepted BE grids, FTM removes no accepted spatial steps and
adds frequency/source/setup work. At the tighter screen, replaying the BDF
grids does not generally satisfy the FTM error budget. Thus scheduling alone
does not explain the difference. Replays are diagnostic: their own march time
and the extra cost of generating the guide grid are both recorded, and they
are excluded from the standalone FTM speedup summary.

## Thermal efficiency

The exponential heat integral reduces endpoint-rule bias. At 5 ms neutron
and thermal widths in the slow case, maximum exchange-temperature error is
about 0.00016 K with FTM, versus 0.00113 K with BE. The explicit endpoint-heat
FTM control separates this quadrature effect from the neutron time transform.
This is a source-integration accuracy benefit, not evidence that more accurate
heat quadrature requires an FTM neutron solver.

In the fast-feedback case, every tested method fails the 0.5% power screen
with 20 ms thermal exchanges. Reducing neutron width as far as 1 ms does not
remove that error. At 5 ms thermal exchanges, qualifying configurations exist
for every method. Thermal cadence therefore imposes both an accuracy floor
and a maximum neutron width: here at least 200 neutron endpoints are required
over 1 s. Fixed BE is competitive with FTM at that limit; adaptive methods
carry additional startup/rejection work.

The conduction equation and lagged cross-section exchange remain first order.
An exponential source integral does not justify increasing `dt_thermal`
without a feedback-cadence convergence check. More generally, fewer neutron
steps can increase nonlinear feedback work or leave thermal splitting as the
dominant error. Report power peaks and energy as well as final temperatures.

## Heterogeneous, fully implicit LRA stress check

This additional CPU study uses the existing two-group/two-precursor LRA
problem, 15 cm FV cells (`refine=1`), cell-local adiabatic heating and fully
implicit feedback. It assesses temporal behavior on the **same coarse spatial
model**, not agreement with the mesh-converged ANL reference.

<!-- LRA:START -->
| Method | Accepted/rejected | FGMRES applies | March seconds | First peak, W/cm3 | Peak error vs tighter BDF |
|---|---:|---:|---:|---:|---:|
| FTM, h=0.01 s | 300/0 | 21,815 | 18.67 | 3183.71 | -46.7046% |
| BE, h=0.01 s | 300/0 | 21,864 | 18.69 | 2797.36 | -53.1721% |
| Adaptive BDF, rtol=1e-5 | 469/25 | 14,773 | 12.80 | 5972.86 | -0.0140% |
| Adaptive BDF, rtol=1e-6 | 692/32 | 19,248 | 16.74 | 5973.70 | +0.0000% |

The BDF peak changes by 0.0140% between the two reference tolerances.
FTM performs 598 feedback solves, versus 516 for BDF at 1e-5. Fewer accepted steps therefore do not translate into less coupled work.
<!-- LRA:END -->

The fixed 10 ms cases are deliberately a stress diagnostic and fail a useful
peak-accuracy comparison. Their runtimes must not be advertised as a
matched-error speedup. Tightening adaptive BDF's tolerance checks reference
peak stability; raw histories retain the peak-time and full-curve behavior.
The LRA timings are single runs and are indicative, while the homogeneous
tables above use repeated medians.

## Validation

Scalar and solver checks cover positive/negative exponential modes, stable
precursor limits, exact stationary equilibrium, actual-BE fallback, accepted
population history, nonuniform widths and event resets, material-dependent
velocities/beta, distinct delayed spectra, and agreement between groupwise
and monolithic solutions. Geometry checks include Cartesian, cylindrical,
triangular and prismatic operators. Thermal checks verify the integrated heat
across fixed and unequal neutron substeps and reject a schedule that crosses
an exchange. FTM transport/adaptive-BDF requests are explicitly refused.

The exact one-cell exponential diffusion mode agrees to approximately
3e-13 relative. Geometry-mode verification uses tightly converged initial
eigenpairs: otherwise a small critical-adjustment error can masquerade as a
transient integration error.

The final verification rerun reports **566 passed, 9 skipped, 4 deselected**;
the JUnit record is [verification-cpu.xml](verification-cpu.xml). The earlier
full CPU run also exercised the validation suite. Its three verification
failures (example documentation, an updated error-message expectation, and
the example subprocess's missing editable installation) were corrected and
are covered by the clean final verification rerun. Optional CUDA/external
checks remain skipped. No GPU performance result is claimed.

## Remaining work and decision

1. Improve frequency prediction through prompt jumps, shape changes and
   feedback; evaluate the cost of any frequency-correction solves explicitly.
2. Design an independent adaptive FTM controller and validate its physical
   error estimate, including precursors and temperature. A small transformed
   shape defect alone is insufficient. Higher-order transformed integration
   may be needed to compete with BDF.
3. Separate time-diagonal updates from geometric face construction. The
   current prototype can rebuild operator sets every step even with unchanged
   materials. This cannot repair a temporal accuracy/order deficit by itself.
4. Only after a matched-error advantage survives representative CPU cases,
   test 11-group/3-D GPU performance and stronger local/group frequencies.
   Keep IQS outside the primary comparison until those results justify it.

Keep the prototype for verification and further method experiments. Current
evidence favors the existing BE/adaptive BDF choices for practical solves.
The separate nondimensionalization/scaling implementation has not been pursued.

## Files and reproduction

- [Compact measurements and work counters](summary.json).
- [Exploratory raw histories](cpu-comparison.json.gz) and
  [tighter raw histories](tight-cpu-comparison.json.gz).
- [LRA fixed FTM/BE](lra-ftm-be-h001.json),
  [adaptive BDF at 1e-5](lra-adaptive-bdf5-rtol1e-5.json), and
  [adaptive BDF at 1e-6](lra-adaptive-bdf5-rtol1e-6.json).
- Harnesses: [homogeneous/coupled matrix](../../examples/exponential_transient_benchmark.py)
  and [LRA CLI](../../examples/lra2d_bdf_benchmark.py).

Install the checkout and test dependencies in an environment with NumPy/SciPy:

```bash
python3 -m pip install -e '.[test]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest tests/verification -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 examples/exponential_transient_benchmark.py \
  --repeats 3 --replay-adaptive-grids --output cpu-comparison.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 examples/exponential_transient_benchmark.py \
  --case uniform-step uniform-ramp --widths 0.005 0.002 0.001 0.0005 0.00025 \
  --rtols 0.0001 0.00001 --power-budget 0.001 --energy-budget 0.001 \
  --repeats 3 --replay-adaptive-grids --output tight-cpu-comparison.json
```

For LRA use `--time-scheme ftm --dt 0.01 --compare-backward-euler` or
`--order 5 --adaptive-rtol 1e-5 --automatic-order --dt 1e-6`, together with
`--refine 1 --implicit-feedback --thermal-zones cell --tol-step 2e-8
--include-history`. The tighter `1e-6` reference also needs
`--adaptive-min-dt 1e-8` to resolve a restarted interval. Use `--output` to
retain JSON. Full raw CPU matrices are gzip-compressed only after measurement;
their source hashes, controls, samples and histories remain available.
