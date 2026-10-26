# Exponential transformation for diffusion transients: assessment and plan

Assessment date: 2026-10-25. Source reviewed: `b496309290d15033234b9b737ad351d4e415d306`.

**Prototype follow-up:** the shared-frequency diffusion implementation is now
available as `time_scheme="ftm"`. Per the requested priority, the first
comparators are fixed/adaptive BE and adaptive BDF, including thermal coupling;
IQS comparisons are deferred. See the
[implementation, CPU measurements and remaining work](../benchmark-results/exponential-prototype/README.md).
Adaptive FTM itself and diagonal-only operator updates are still future work.
The [proposal follow-up](exponential_proposal_review.md) identifies true
second-order exponential theta, source-group precursor weights and optional
frequency correction as separate experiments, and adds a CN/work comparison.
The [second-order development checkpoint](../benchmark-results/exponential-second-order/README.md)
now implements shared-frequency `ftm2` with linear transformed precursor
production. Further work proceeds in small chunks. The heat-transfer solve
is outside the exponential-transformation scope.
The [second checkpoint](../benchmark-results/exponential-second-order/CHUNK2.md)
adds BE start-up and optional frequency refinement. A resolved rapid-growth
case now demonstrates a temporal benefit from the transform at equal work;
adaptive BDF still uses fewer tested solves. Stronger stiff damping is the
next temporal development before expanding the spatial frequency model.

**Recommendation: develop an opt-in frequency transformation method (FTM),
starting with one frequency shared by all cells and groups.** It can reuse
GARNET's matrix-free diffusion engines. Its main possible benefit is fewer
spatial solves at a given temporal error. A speedup over the best existing
adaptive BE/BDF or guarded IQS configuration remains unproven. Published
SIMULATE results do not establish performance in this implementation.

The [scaling assessment](numerical_scaling_plan.md) is a separate workstream.
Establish its error-norm diagnostics before adaptive comparisons; a change in
the error metric must not be mistaken for an FTM improvement.

## Existing capabilities and relevant history

| Implementation | Consequence |
|---|---|
| [`timescheme.py`](../garnet/timescheme.py): BE, theta methods, BDF1–6, nonuniform coefficients, prediction and order/width control | Much infrastructure is reusable. No exponential/frequency-transformed transient scheme was found. |
| [`TransientSolver`](../garnet/transient.py): precursor elimination, groupwise PCG, optional monolithic FGMRES, per-material kinetics and per-family delayed spectra | Preserve these capabilities. Current precursor formulas solve the **discretized BDF equations**, rather than exactly integrating the continuous precursor ODE for an exponential source. |
| [`MultigroupStepOperator`](../garnet/multigroup.py) and its energy-sweep preconditioner | A shared-frequency prototype can retain the coupled linear-system structure. |
| Adaptive integration requires a `BDF` object and `step_solver="monolithic"` | Adding an FTM class alone will not enable adaptivity: type checks, prediction, history and acceptance need an explicit extension. |
| [`quasistatic.py`](../garnet/quasistatic.py): guarded amplitude/shape factorization and full-diffusion fallback | FTM overlaps with some IQS benefits. Compare independently before composing them. |

The [raw-flux chronological predictor](../benchmark-results/transient-acceleration/step-01-chronological-initial-guess.md)
increased work on the stiff 11-group case. A transformed time discretization
changes the approximation, so that result does not reject FTM; it does warn
against assuming that amplitude prediction improves nested multigroup solves.

The [matched HP-MR study](../benchmark-results/transient-acceleration/step-07j-hpmr-speedup-and-coupling.md)
found adaptive BE and automatic BDF essentially tied on mild motion. The
[LRA stored summary](../benchmark-results/lra2d-adaptive-bdf-report/benchmark-summary.json)
instead records a 5.10x CPU reduction for BDF over BE at comparable first-peak
temporal error (201 versus 3180 accepted steps). FTM must compete with these
existing methods, rather than only a deliberately fine fixed-BE reference.

## Formulation and implementation fit

This is an integrating-factor change of unknown within each step, not a
Fourier transform or a large matrix-exponential solve. Published
[SIMULATE5-K-VVER work](https://doi.org/10.1016/j.anucene.2023.110146)
supports FTM alongside BE. The authors' [M&C 2023 paper, sections 2.1–2.2](https://www.researchgate.net/publication/374111958_Development_and_Benchmarking_of_Transient_Nodal_Code_SIMULATE5-K_Neutron_Kinetics_Solver_for_VVERs_and_Hexagonal_Geometries)
describes local group frequencies, transformed flux integration and analytic
precursor elimination. GARNET can adopt the temporal idea without adopting
SIMULATE's nodal spatial model. SIMULATE-3K-style linear transformed-shape
precursor interpolation is also described by
[Kreher, Smith and Forget](https://doi.org/10.1080/00295639.2021.1980363).

Let `h = t[n+1] - t[n]`, `M_g = 1/v_g`, and write

```text
phi_g(t) = exp(omega * (t - t[n])) psi_g(t).
```

For a frozen frequency, backward differencing of `psi` gives, in physical
endpoint flux variables,

```text
[L_g + M_g (1/h + omega)] phi_g[n+1]
  = M_g exp(omega h) phi_g[n]/h + scattering + prompt + delayed.
```

`L_g` is GARNET's loss operator before the time shift. This preserves face
couplings. Even a later cell-dependent frequency can be used in this endpoint
equation without explicitly differentiating `exp(omega(x) t)` in the leakage
term. Geometry-dependent RHS weights must still match the existing solver.

For the simplest exponential end-of-step source approximation, define

```text
H_i = h phi_1(-(lambda_i + omega) h),
phi_1(z) = (exp(z) - 1)/z,  phi_1(0) = 1,
C_i[n+1] = exp(-lambda_i h) C_i[n] + beta_i H_i S[n+1].
```

Here `S = sum_g nuSigma_f,g phi_g / k0`, retaining the critical adjustment.
In GARNET's total-spectrum convention the emission weight and fixed delayed
source become

```text
w_g = chi_total,g - sum_i chi_delayed,ig beta_i (1 - lambda_i H_i),
delayed_history_g = sum_i chi_delayed,ig lambda_i exp(-lambda_i h) C_i[n].
```

Treating `fields.chi` as a prompt spectrum would silently change the physics.
These expressions also allow mapped material beta/velocity fields and
distinct delayed spectra. Evaluate `H_i` using `expm1` and a series near zero:
`omega = -lambda_i` is a finite limit.

With a shared frequency, one scalar fission source remains sufficient. With
a separate frequency for each **source group**, precursor production instead
contains `sum_h beta_i H_i,h nuSigma_f,h phi_h/k0`. It generally cannot be
represented by changing only destination-group `w_fis[g]`. Group-dependent
FTM needs extra source contractions or generalized low-rank fission coupling;
defer it until the shared-frequency study identifies a useful regime.

The initial approximation above is first order. A later variant can
interpolate `psi` linearly and integrate precursors with stable `phi_1`/`phi_2`
coefficients; that alone does not make flux integration second order.
**Zero frequency restores BE's flux derivative but not BE's rational
precursor update.** Benchmark a zero-frequency/exact-precursor control
separately, and make fallback invoke the actual existing BE method.

## Potential gain and practical limits

For `y' = alpha y`, transformed BE has amplification
`exp(omega h)/(1 - (alpha - omega)h)`, which is exact at `omega = alpha`.
For a close, fixed frequency its leading growth-rate error depends on
`(alpha - omega)^2 h`, rather than `alpha^2 h`. This explains the opportunity
on near-exponential segments; it gives no general speedup factor for a
superposition of spatial and precursor modes.

Independent [analytic probes](../benchmark-results/transient-numerics-assessment/analytic-probes.json)
use one-family point kinetics with `beta=0.0065`, `lambda=0.08/s`,
`Lambda=1e-4 s`, and reactivity step `rho=0.002`. Frequency is lagged from two
populations. Errors are maximum relative population errors at each run's
step endpoints:

| Window | h | BE error | Lagged FTM error |
|---|---:|---:|---:|
| Entire step response, 0–10 s | 0.01 s | 2.422% | 2.544% |
| Entire step response, 0–10 s | 0.001 s | 0.2832% | 0.3514% |
| Exponential tail, 0.5–10 s | 0.1 s | 0.05990% | 2.88e-9% |

The tail starts from the **exact** state and one exact previous population
sample, removing inherited startup error. These are independent ODE probes,
not GARNET runs, BDF comparisons or measured speedups. They demonstrate both
the opportunity and the weakness of naive frequency prediction at a prompt jump.

If spatial neutronics is fraction `f` of baseline march time, `r` is the
reduction in total attempted spatial steps, and `c` is their average cost
multiplier, an illustrative total speedup is

```text
speedup = 1 / ((1 - f) + f c/r).
```

With `f=0.97` (consistent with existing coupled profiles) and **assumed**,
unmeasured `c=1.2`, halving the attempt count gives 1.63x; quartering it gives
3.12x. With unchanged count it gives 0.84x. These are scenarios, not forecasts.
A step-doubling controller can require three spatial solves, so its overhead
cannot be assumed to be only 20%.

Reasons a benefit may disappear:

- Abrupt insertion, changing shape, opposing regional responses and strong
  feedback need resolution even when global power looks exponential.
- Larger steps weaken the positive BE time diagonal; negative frequency can
  weaken it further. Inner/outer iterations or failures may increase.
  Implicit treatment alone does not establish unconditional stability or
  positivity for a nonlinear frequency update.
- Changing frequency currently requires operator/preconditioner rebuilds
  unless time diagonals are separated from geometric stencil construction.
- Control frames, thermal exchanges and required output endpoints limit step
  widths. Dense output would need its own error check.
- Guarded IQS already removes most fine-step spatial work on slow maneuvers.
  Its [GH200 acceptance](../benchmark-results/hpmr-iqs-gh200/README.md)
  establishes the production comparator, not a measured FTM headroom.

## Separate development plan

| Phase | Concrete work | Gate and deliverable |
|---|---|---|
| E0: baseline | Record fixed/adaptive BE, automatic BDF and guarded IQS on homogeneous, TWIGL, LRA and HP-MR cases. Add per-field defect diagnostics from the scaling plan without changing decisions. | Reproducible accuracy/work/time baselines, including rejected attempts and rebuilds. |
| E1: fixed-width prototype | Add opt-in `time_scheme="ftm"` in `timescheme.py`/`transient.py`. Supply time diagonal, physical history source, precursor history and source coefficient explicitly. Reuse fixed-point and monolithic engines. Start with one frozen global frequency per step. | Exact equilibrium and exponential eigenmode; convergence through prompt jumps; agreement between converged spatial solvers. No adaptive/default change. |
| E2: frequency and caching | Estimate frequency from accepted neutron population `sum_cells V sum_g phi_g/v_g`, rather than cross-section-dependent fission power alone. Optionally reuse an existing adjoint; do not add an eigen solve solely for frequency. Use the actual previous accepted width, guarded log ratios, finite bounds, event resets and BE fallback. Separate diagonal updates from geometry, preserving workspace/graph contracts. | Frequency/guard/fallback telemetry; no stale operators/history; lower setup overhead; robust positive and negative maneuvers. |
| E3: adaptive FTM | Extract shared scheduling/commit/rollback machinery from the BDF path. Add method-neutral adaptive controls while retaining `adaptive_bdf` compatibility. Validate error estimates with full-step/two-half-step physical flux and precursor comparisons first; seek a cheaper estimator only after validation. | Acceptance includes frequency error and all physical state blocks. Rejections restore frequency, flux, precursors and provisional thermal state. Events/exchanges remain exact endpoints. |
| E4: matched-error CPU/GPU study | Use the matrix and criteria below. Count frequency correctors, failed attempts, inner iterations, contractions, rebuilds, transfers and setup. | A useful end-to-end advantage over the best applicable existing full-diffusion method. |
| E5: selective extensions | Only after E4: test regional/cell frequencies, then source-group frequencies or linear transformed-source precursor interpolation. Finally evaluate FTM in IQS correctors/fallbacks. | Each extension earns its own accuracy/cost result. Speedups over IQS are measured, not multiplied from independent studies. |

E1 is a moderate change, likely several focused development days. E2–E4
are the substantial work: roughly another one to several weeks depending on
feedback regressions and access to production GPUs. Local group-frequency
support is a separate, more intrusive extension. These are planning estimates.

Implementation requirements:

- `TimeScheme.a0()` is one scalar used for time shift and history weighting.
  Avoid encoding everything through it: zero `1 + omega*h` would make an
  artificial carried-state division fail, and group/cell frequencies do not
  fit its contract.
- Retry with smaller `h` or actual BE when estimates overflow, the group
  operator is no longer admissible for PCG, or the coupled solve fails.
  Check the physical residual and block solve; a positive diagonal alone
  does not prove SPD or full-system stability.
- Freeze frequency during the initial linear solve. If endpoint frequency
  correction is added, count every extra solve and converge the
  frequency/feedback iteration before committing state.
- Small transformed-shape defect does not bound physical amplitude error
  when the estimated frequency is wrong.

## Benchmark and acceptance contract

1. **Analytic correctness:** stationary critical equilibrium; prompt-only
   diffusion modes; one- and six-family point kinetics; prescribed
   exponential-source precursors; `omega + lambda_i -> 0`; small/large `h`;
   falling flux; temporal refinement. Check neutron balance with the new
   time derivative and precursor formula.
2. **Coverage:** per-material velocities/beta, delayed spectra, masks/blends,
   Cartesian and triangular/prismatic geometry, cylindrical RHS weighting,
   feedback retries and variable-step rollback. Start with diffusion;
   transport moments require a separate derivation.
3. **Applications:** TWIGL step/ramp and shape changes; LRA prompt peak and
   rod stop; 11-group HP-MR uniform insertion; slow/fast/asymmetric drums;
   strong-feedback coupling; then production 3-D GPU cases.
4. **Fair comparison:** time-refined and demonstrably converged reference;
   match power, peak time, energy, local/shape power and temperature error.
   Equal `rtol`, width or output count is insufficient. Use the same spatial
   model/control path and common output grid. Start exploration at declared
   0.1% maximum power and volume-weighted shape-error budgets and 0.1 K
   temperature budget, tightening where an existing benchmark requires it.
5. **Promotion:** seek at least 1.2x median march improvement on two target
   cases against the fastest acceptable full-diffusion configuration, with
   no unexplained regression at sharp events. This is a proposed retention
   gate, not a predicted gain. Report initial eigen/adjoint setup separately
   and include total application time. Warm/synchronize GPUs; retain at
   least three interleaved samples and raw work counters.

Stop broadening the implementation if gains occur only in ideal tails already
handled more cheaply by IQS, or disappear at matched physical error. An
accurate fixed-width FTM can still remain an optional reference method.

## Evidence and reproduction limits

The source/history and recorded reports were reviewed; no previous FTM
implementation was identified in the accessible checkout. Reproduce the
independent probes with:

```bash
python3 tools/assess_transient_numerics.py \
  --output benchmark-results/transient-numerics-assessment/analytic-probes.json
```

At the initial assessment, the environment had Python 3.14.4 but no NumPy,
SciPy or CuPy. Only the independent probes ran at that stage, and the timings
in this original assessment came from existing reports. NumPy, SciPy and
pytest were subsequently installed; new solver validation and CPU timings
belong to the linked prototype study. GPU performance remains unmeasured.
