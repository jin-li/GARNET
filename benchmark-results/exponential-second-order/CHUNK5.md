# Linear-output-aware adaptive FTM: fifth development checkpoint

2026-10-26. This follows the physical-state controller in
[checkpoint four](CHUNK4.md). Optional `adaptive_ftm['output_rtol']` now
restricts macro widths for linear interpolation, using the midpoint already
computed by step doubling. The integration formulas, adaptive BDF controls
and heat-transfer solver keep their existing behavior.

## Implemented restriction

For each physical flux group and precursor family, compare the first
half-step state `y_mid` with `(y_start + y_end)/2`. Normalize the defect
pointwise by `output_atol + output_rtol*abs(y_mid)` and take its
active-volume weighted RMS. A separate scalar test compares midpoint power
with the endpoint power chord. This is necessary when changing fission
coefficients curve the recorded power while the physical flux stays constant.

The maximum state/full-fine defect and output/chord defect controls
acceptance. Each check proposes a next width; take the smaller one. The
output proposal uses square-root scaling because a chord defect is O(h²),
including when the endpoint integration is second order. Rejections restore
the same physical, production, recurrence and frequency histories as before.
No spatial solve is added to an attempt; three solves plus any frequency
refinements remain charged. Keeping the midpoint and forming field chords
does add storage and reductions; this checkpoint makes no timing claim.

`output_rtol` defaults to `None`. `output_atol` defaults to zero, requires
an enabled output tolerance, and is a floor in each field's units and P/P0
for scalar power. With the option enabled, `local_errors`/`rejected_errors`
contain the combined maximum. Full/fine defects remain in
`local_error_blocks`/`rejected_error_blocks`; the new
`output_error_blocks`/`rejected_output_error_blocks` contain flux groups,
precursor families, then scalar power. The output lists remain empty when
disabled. Reported times, callbacks and `power_at` remain macro endpoints
and their linear interpolant.

These are local resolution checks. A single midpoint does not certify all
interior extrema or accumulated state error. A discontinuous power observable
requires two-sided output at its event, which this continuous chord does not
provide. Thermal integration and coupled-state adaptation remain separate.

## Reproduced interpolation limit and refinement

The rapid-rise case uses an independent physical two-mode kinetics reference,
with `rho=0.01`, duration 0.5 s, scalar initial width 0.01 s and maximum width
0.02 s. At fixed state tolerance `rtol=1e-7`:

| FTM2 output control | Maximum endpoint power error | Maximum common-grid power error | Power-integral error | Endpoint solves |
|---|---:|---:|---:|---:|
| State only | 0.000463% | 2.20658% | 0.910182% | 1,326 |
| `output_rtol=1e-4` | 0.000338% | 0.006738% | 0.004572% | 2,181 |

The maximum accepted macro width drops from 0.017995 s to 0.0009733 s.
The [focused verification](chunk5-output-verification.xml) also checks
refinement from `output_rtol=1e-3` to `1e-4` for both FTM2 and transformed
BDF2, with over fivefold reductions in interpolated power and integral error.
The test uses an 8,001-point reference trapezoid for the integral;
the benchmark report instead uses independent adaptive quadrature. The small
difference between those integral estimates is retained in their records.

## Matched physical error and work

[chunk5-output.json](chunk5-output.json) records 160 executions: state-only
and output-aware versions of the fourth-checkpoint sweep, four cases, five
state tolerances and four method families. There are 140 distinct
configurations; the 20 unchanged adaptive BDF comparators repeat in both
halves. Output-aware FTM uses `output_rtol=1e-4`. References and the
8,001-point common output grid are independent of the adaptive meshes.

The following are the fewest tested solves meeting **0.1% maximum power,
power-integral and final precursor error**, with nonnegative power and no
wrong-direction macro increments. Parentheses give maximum power error.

| Case | Output-aware FTM2 | Output-aware transformed BDF2 | Output-aware zero-frequency BDF2 | Adaptive BDF1–5 |
|---|---:|---:|---:|---:|
| Prompt rise | 339 (0.01358%) | 285 (0.01287%) | 285 (0.01301%) | 84 (0.09451%) |
| Rapid rise | 1,683 (0.01027%) | 1,608 (0.02218%) | 1,608 (0.05795%) | 261 (0.09661%) |
| Shutdown | 591 (0.01645%) | 528 (0.02363%) | 528 (0.02364%) | 209 (0.04520%) |
| Interior shutdown | 603 (0.01886%) | 543 (0.01776%) | 543 (0.01771%) | 213 (0.04502%) |

Output-aware FTM2 can now meet the rapid-rise budget that no state-only
FTM2 configuration in the sweep met. Extra output resolution can increase
work on already adequate runs; the chosen local output tolerance is more
conservative than the global budget. Adaptive BDF remains cheapest among
the tested configurations. Zero-frequency controls follow different adaptive
grids, so these counts alone do not isolate a benefit from the transform.
The earlier fixed/shared-mesh studies retain that role.

## Verification and remaining work

The [314-case regression](chunk5-regression.xml) passes across exponential
coefficients/schemes, BDF/theta, kinetics, material tables, public transient
APIs, multigroup solves, coupling and examples. The updated
[58-case focused batch](chunk5-output-verification.xml) also passes; it
overlaps the regression batch and adds four guarded physical-reference
configurations. This checkpoint does not repeat the full repository suite,
slow studies or GPU execution.

New checks cover:

- Rapid exponential growth, endpoint accuracy, interpolation and power
  integrals at fixed state tolerance.
- Spectrum redistribution with constant total power, against an independent
  physical two-group matrix exponential.
- Localized two-group/two-family spatial histories against a separately
  assembled 16-state generator and matrix-exponential actions, plus
  non-equilibrium six-family precursors against Radau.
- Output-driven rejection, exact event/alignment endpoints, callbacks and
  accepted-half-step replay on Cartesian, cylindrical and masked triangular
  meshes, including BE startup and event restarts.
- Curved power with constant physical flux, initially zero precursor fields,
  invalid tolerances, minimum-width failure and zero-duration output.

The source archive is rebuilt and checked from an isolated installation;
[the smoke record](chunk5-archive-smoke.json) exercises guarded cylindrical
event/rejection/callback behavior. The
[QA record](../../docs/qa/garnet-exponential-output-2026-10-26.json) retains
environment, source fingerprints and validation counts.

The next bounded task is to calibrate a cheaper physical-state estimator
against this three-solve oracle, with the output check retained independently.
A resolved spatial application, setup reuse, local/group frequencies and
coupled thermal adaptation remain separate gates. This checkpoint resolves
an output representation limitation; it does not establish a speedup.

Follow-up: the [sixth checkpoint](CHUNK6.md) implements and calibrates the
two-solve history candidate while retaining this step-doubling controller
and its separate output check as the reference.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m tools.assess_exponential_output
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest tests/verification/test_exponential_output.py tests/verification/test_exponential_adaptive.py -q
python3 tools/build_src_zip.py
```
