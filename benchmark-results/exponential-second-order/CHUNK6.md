# Physical-history state estimator: sixth development checkpoint

2026-10-26. This implements the cheaper-estimator portion of E3b after the
[output restriction](CHUNK5.md). Set
`adaptive_ftm={"rtol": 1e-4, "output_rtol": 1e-4, "estimator": "history"}`
to use quadratic physical-history prediction on eligible second-order
attempts. The existing two-half-step trajectory remains the committed state;
the full comparison solve can be skipped. Step doubling remains the default
and the reference controller. Thermal integration is outside this work.

## Prediction and safeguards

Let the macro start be time zero, the previous accepted midpoint be at `-d`,
the current midpoint at `k=h/2`, and the target endpoint at `2k`. For each
physical flux group and precursor family, use quadratic extrapolation:

```text
r = k/d
P(2k) = [2*r^2/(1+r)] y(-d) - (1+2*r) y(0)
        + [2*(1+2*r)/(1+r)] y(k).
```

The implementation evaluates this in difference form to preserve constant
fields. It uses actual accepted widths and physical state values, with no
history from an old exponential frame. The endpoint correction `y(2k)-P(2k)`
is normalized with the existing per-field active-volume RMS and
`atol + rtol*max(abs(start),abs(endpoint))`. There is no Richardson divisor
or presumed universal relationship to the full/fine defect. On a smooth
second-order trajectory the indicator is O(h³); it is not a local-error bound.

An eligible attempt uses **two spatial endpoint solves**, plus any frequency
refinements. Step doubling is used instead when history is unavailable after
initialization or an event, the previous trajectory had a first-order
safeguard, the current/previous half-width ratio lies outside `[0.25,2]`, or
a full-width scheme probe would invoke BE. The width bounds condition the
predictor; they do not prove stability of the full transient.

If a provisional half unexpectedly invokes BE, restore the accepted state
and replay that macro attempt with the full/fine oracle. All discarded
half solves and corrections remain charged. Such attempts can cost four or
five solves before corrections. Order diagnostics describe the active
replayed trajectory; trial/work diagnostics retain the discarded work.

The output midpoint/chord check is independent and unchanged. It still uses
the actual solved midpoint. History states update only on acceptance and
reset at explicit events; rejection restores physical state, production,
frequency and recurrence histories without advancing callbacks.

`estimator="history-audit"` records the candidate defect on the same trials
as step doubling while retaining reference decisions and work. Accepted and
rejected `ftm_estimator_diagnostics` record `method`, `reason` and
`history_error_blocks`; the latter is `None` when history is not eligible.
`local_error_blocks` contains the selected estimator's physical defects.
Default-mode estimator lists remain empty.

## Same-trial calibration

[chunk6-estimator.json](chunk6-estimator.json) records 240 executions:
three estimator modes, four kinetics cases, five state tolerances and four
method families. There are 200 configurations counting audit as a separate
mode; the 20 unchanged adaptive BDF configurations repeat in all three
sweeps. All runs retain `output_rtol=1e-4` for FTM. Independent physical
kinetics matrices provide endpoint and 8,001-point common-grid power
references, independent quadrature provides the power integral, and final
precursors, positivity and wrong-direction increments are also checked.

The 60 audit configurations provide **20,410 paired eligible trials**.
There are no cases where the history state check passes and the full/fine
state check fails. There are **3,361** disagreements in the other direction.
These are state-check decisions; output checks can also reject a trial.
Audit power/error/work records match the step-doubling records exactly.

To avoid relying only on output-limited steps,
[chunk6-state-audit.json](chunk6-state-audit.json) disables output control.
Its 24 frozen-coefficient configurations supply 5,380 paired trials, including
four failed reference state checks. Archived checkpoint-four solve counts
and endpoint errors match exactly. Two additional smooth-insertion cases
supply 102 paired trials and nine failed reference checks. The candidate
flags all **13** of these reference failures. Actual history-driven retries
also occur in both smooth cases; independent Radau errors and accepted-half
replay are verified separately.

These empirical checks do not establish equivalence of the estimators. The
defect ratio varies substantially with frequency fitting, modes and widths.
On tight nearly exponential tails, the physical polynomial still sees
curvature even when the transform removes most integration error. For
example, rapid-rise FTM2 at state rtol=1e-7 takes 3,351 solves with history
versus 2,181 with step doubling, while delivering a smaller power error.
The history option remains experimental; it is not a uniformly cheaper
controller at every local tolerance.

## Matched physical error and work

Fewest tested endpoint solves satisfying **0.1% maximum power,
power-integral and final precursor error**, nonnegative power and no
wrong-direction macro increments:

| Case | FTM2: full/fine → history | Transformed BDF2: full/fine → history | Zero-frequency BDF2: full/fine → history | Adaptive BDF1–5 |
|---|---:|---:|---:|---:|
| Prompt rise | 339 → 232 | 285 → 194 | 285 → 194 | 84 |
| Rapid rise | 1,683 → 1,129 | 1,608 → 1,076 | 1,608 → 1,076 | 261 |
| Shutdown | 591 → 402 | 528 → 358 | 528 → 358 | 209 |
| Interior shutdown | 603 → 412 | 543 → 370 | 543 → 370 | 213 |

These selected history runs retain the reference power/integral/precursor
errors and reduce solves by about **31.5–33.1%**. On rapid rise at state
rtol=1e-4, verification also establishes identical accepted grids and power
histories: FTM2 falls from 1,698 to 1,140 solves and transformed BDF2 from
1,614 to 1,081. All startup, rejected, corrected and discarded work is
included. Fewer endpoint solves do not establish a runtime speedup: extra
history storage, setup copies and reductions are not separately timed here.
Adaptive BDF remains the cheapest tested comparator at this physical budget.

## Verification and next gate

**363 distinct CPU checks pass** across the
[361-case regression](chunk6-regression.xml) and
[two smooth-insertion checks](chunk6-ramp-verification.xml). The
[101-case focused batch](chunk6-estimator-verification.xml) overlaps the
regression record. Coverage includes unequal-width polynomial exactness and
cubic scaling, audit trajectory equality, actual two-solve work, six-family
Radau references, localized multigroup matrix-exponential histories, event
and rejection replay on Cartesian/cylindrical/masked-triangular meshes,
tiny alignment widths, an injected provisional BE fallback and all discarded
work. Default BE/BDF/theta, coupling, examples and public transient APIs
remain covered by the regression batch. The full repository suite, slow
studies and GPU execution were not repeated.

The rebuilt archive passes an isolated installed cylindrical event test
with history-based rejection, fixed-half-step replay, callbacks and exact
audit/reference trajectory equality. The
[archive smoke record](chunk6-archive-smoke.json) and
[QA record](../../docs/qa/garnet-exponential-estimator-2026-10-26.json)
retain environment, source hashes and validation counts.

The next bounded gate is a resolved spatial application such as TWIGL,
comparing power, spatial shape, precursor inventories and integrated power
at matched physical error against adaptive BDF. Retain default step doubling
and audit controls, then measure setup reuse before CPU/GPU timing claims.
Local/group frequencies, two-sided power jumps, coupled thermal adaptation
and the independent Krylov-action track remain separate work.

Follow-up: the [seventh checkpoint](CHUNK7.md) assesses the TWIGL gate,
including independent references, output-width comparators and a separate
spatial-resolution diagnosis. Its spatial-shape and performance-promotion
conditions are not met at the tested resolutions.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m tools.assess_exponential_estimator
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m tools.assess_exponential_state_audit
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest tests/verification/test_exponential_estimator.py tests/verification/test_exponential_estimator_ramp.py tests/verification/test_exponential_adaptive.py tests/verification/test_exponential_output.py -q
python3 tools/build_src_zip.py
```
