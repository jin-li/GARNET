# Second checkpoint: prompt events and the SIMULATE connection

2026-10-25. Neutrons and delayed precursors only. Heat-transfer integration is
outside this workstream. This follows the [first checkpoint](README.md).

## What to emulate from published SIMULATE work

Bahadir's [SIMULATE5-K method paper, section 2.2](https://www.researchgate.net/publication/360764337_Development_and_Benchmarking_of_Transient_Nodal_Code_SIMULATE5-K_Neutron_Kinetics_Solver)
uses implicit differencing after the frequency transform, analytic precursor
integration, and frequencies initially predicted from history and refined
during iteration. Node/group frequencies also support the pseudo-static nodal
leakage calculation. The transform has both temporal and architectural uses.

The [SPERT validation, section 2.3](https://www.frontiersin.org/journals/nuclear-engineering/articles/10.3389/fnuen.2026.1771859/full)
describes adaptive steps based on temporal-derivative heuristics and backup
steps. Its RIA calculations limit steps to 1 ms, or 0.5 ms at hot full power.
It does not justify large uniform steps through every prompt event.

Shared-frequency trapezoidal FTM2 remains a distinct second-order experiment.
The tests below emulate resolved initial intervals and recursive frequency
fitting with explicit implementation choices. They do not reconstruct the
commercial code or its unpublished update rules.

## Implemented additions

- FTM2's `startup_be_steps=2` selects actual BE for two accepted steps after
  initialization and each explicit history restart. They report order one
  and reason `startup-backward-euler`. The default is zero. This option does
  not alter the time grid; initial steps still need suitable widths.
- For FTM/FTM2, `frequency_iterations=N` permits up to N extra monolithic
  endpoint solves. Each fit uses the provisional endpoint population and the
  last accepted population. Accepted histories and external callbacks advance
  once, after the endpoint is accepted.
- `frequency_tolerance` bounds `h*abs(omega_observed-omega_used)` (default
  1e-6); `frequency_relaxation` lies in (0,1] (default 1). The final remaining
  defects are in `step_frequency_log_defects` and extra solve counts in
  `step_frequency_refinements`. Hitting a cap does not imply convergence.
  Unsafe fits trigger a fresh actual BE solve. A missing defect makes no
  frequency convergence claim, including BE fallback/start-up.

These controls live in `ftm_kwargs`. Refinement cannot use a prescribed
frequency. It requires the monolithic solver and currently refuses implicit
feedback. Every repetition enters the endpoint-solve and Krylov counters.
This research corrector repeats a converged global solve; updating frequencies
inside an existing spatial iteration can have a different cost.

Neither BE start-up nor a converged frequency is an error controller. The
event script uses a prescribed graded mesh: initial width 1e-5 s, growth
factor 1.1 and a chosen maximum width. The transform and zero-frequency
control use identical meshes. No reference values select their steps. This
is not adaptive FTM. Interior-event tests use the old material at the event
endpoint and the new material in the following interval, preserving state
continuity.

## A measured justification for the transform

The rapid-rise case has rho=0.01, beta=0.0065, lambda=0.08/s, generation time
1/6600 s and duration 0.5 s. Power rises from 1 to 315,130.88 without thermal
feedback. Its reflective 4 by 4 by 1 FV model reduces exactly to two-mode
point kinetics, giving an independent closed-form reference.

On the **same mesh and same two BE start-up steps**, maximum width 0.01 s:

| Scheme | Maximum power error | Maximum endpoint error | Neutron solves | PCG |
|---|---:|---:|---:|---:|
| Zero-frequency FTM2 | 5.2462% | 4.6015% | 113 | 1,110 |
| Lagged-frequency FTM2 | 0.8022% | 0.1090% | 113 | 1,181 |
| Refined-frequency FTM2 | 0.7970% | 0.1035% | 235 | 2,374 |

The transform reduces maximum error about 6.5 times at the same solve count;
endpoint error improves about 42 times. This isolates the exponential factor
from second order and from the mesh. The zero-frequency control has
trapezoidal neutrons with analytic linear precursor integration; it differs
from the existing CN precursor discretization.

Refinement converges in these cases but supplies little extra accuracy for
its additional complete solves. This does not reject updates inside an
existing spatial iteration or group/local frequency corrections.

Fewest **tested** solves meeting the same 1% maximum power-error budget:

| Scheme/configuration | Maximum error | Neutron solves | PCG |
|---|---:|---:|---:|
| Fixed BE, h=5e-5 s | 0.6784% | 10,000 | 51,867 |
| Adaptive BE, rtol=1e-6 | 0.4520% | 15,105 | 70,105 |
| Zero-frequency FTM2, graded cap=0.002 s | 0.2300% | 296 | 2,846 |
| Lagged FTM2, graded cap=0.01 s | 0.8022% | 113 | 1,181 |
| Adaptive BDF1--5, rtol=1e-4 | 0.8217% | 90 | 889 |

At a 0.1% budget, FTM2 needs 296 tested solves (0.0574% error), the
zero-frequency control 539 (0.0583%), and BDF 251 (0.0966%). The sampled BE
controls do not qualify at this tighter budget; their required work is not
extrapolated.

BDF remains the strongest comparator here. Output interpolation matters:
its endpoint error at rtol=1e-7 is 0.0000947%, while common linear output
interpolation raises its maximum to 0.0966%. Both metrics are retained for
every scheme. Dense BDF output would require a separate comparison. These
single-run work counts do not establish CPU/GPU speedups.

## Start-up, shutdown and stability

For rho=0.002 over 1 s, uniform FTM2 h=0.01 s has 5.09% maximum error.
Two BE steps on that coarse mesh reduce it to 1.65%. The graded mesh reduces
it to 0.0125% with 163 solves; the zero-frequency control is almost as
accurate. Resolved start-up supplies most of the benefit in that case.

For rho=-0.05 over 0.2 s, uniform FTM2 h=0.01 s produces two unphysical
power reversals and 237% maximum relative error. Two coarse BE steps alone
are insufficient. The graded mesh removes the reversals and gives 0.297%
error with 83 solves; the zero-frequency control gives 0.211%. This mesh
misses the 0.1% budget, while BDF meets it (0.0449%, 200 solves). Reducing
only the later maximum width cannot remove an initial error.

The published transformed implicit step has BE's stiff damping; transformed
trapezoidal integration can retain oscillatory fast modes. A second-order
transform needs deliberate damping and time control in addition to frequency
estimation.

## Verification and reproduction

[chunk2-events.json](chunk2-events.json) records 117 completed FV configurations
across insertion, rapid rise and shutdown: full solves, Krylov counts,
fallbacks, final frequency defects, monotonicity checks and source hashes.
Errors are the maximum of each run's own endpoints and a common 8,001-point
linearly interpolated output grid. Rejections and frequency repeats are charged.
Adaptive runs start at 1e-5 s with maximum width 0.02 s and minimum width
1e-11 s, allowing the tightest BDF control to resolve the initial insertion.

[chunk2-verification.xml](chunk2-verification.xml) records 168 passing targeted
tests. A separate [interior-event check](chunk2-interior-event-verification.xml)
adds continuity and second-order refinement at an aligned material jump:
169 distinct cases pass. These include bounded/capped correction, mixed-mode
precursor references and the exact linear scalar growth-rate identity.
Repository-wide and GPU checks were not repeated for this checkpoint.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tools/assess_exponential_events.py
python3 -m pytest tests/verification/test_exponential_theta.py -q
```

## Next small chunk

Prototype a second-order transformed step with stiff damping on independent
scalar kinetics before adding it to the FV engine. Compare its stage-solve
cost with FTM2 and adaptive BDF. Per-group frequencies and a localized
multigroup benchmark such as TWIGL are the next spatial extension.
Refinement inside existing iterations and adaptive FTM remain separate
experiments. Heat-transfer integration stays unchanged.
