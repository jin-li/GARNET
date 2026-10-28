# Fourth checkpoint: physical-state adaptive FTM

26 October 2026. This follows the [damping checkpoint](CHUNK3.md).
The scope remains neutron and delayed-neutron integration. Heat-transfer
integration and GPU performance are separate workstreams.

An opt-in controller now supports both second-order transforms:

```python
run = core.transient(
    t_end=0.1, dt=0.005, time_scheme="ftm-bdf2", device="cpu",
    adaptive_ftm={"rtol": 1e-4, "min_dt": 1e-10, "max_dt": 0.01},
)
print(run.local_errors, run.local_error_blocks, run.rejected_steps)
```

The [transient API guide](../../docs/transient_api.md) explains material
callbacks, event conventions, diagnostics and controller controls.

## Acceptance and rollback

Each macro attempt starts from the last accepted physical state and performs
one full step and two half steps. The full step is discarded. The fine
trajectory recomputes frequency from its own population history and retains
the first half's physical flux/precursor history for the second half.
Successful attempts commit the fine state once at the macro endpoint, with
one external callback. Failed attempts restore flux, precursors, production,
frequency samples, physical derivatives, startup counters and BDF history.
Spatial setup is invalidated on rollback; work counters survive it.

For each flux group and precursor family b, the physical defect is

```text
d_b,j = atol + rtol * max(abs(u_start,b,j), abs(u_fine,b,j))
E_b   = sqrt(sum_active V_j * ((u_fine,b,j-u_full,b,j)/d_b,j)^2
             / sum_active V_j)
E     = max_b E_b.
```

Zero scale contributes zero only for zero defect. The norm includes masks
and cylindrical volume factors. Regular Cartesian and triangular cells have
constant volumes that cancel. Large precursor concentrations cannot dilute
flux-group defects. Reductions stay on the selected backend until one small
per-field result transfer. Existing adaptive BDF's concatenated
predictor/corrector norm remains unchanged; equal rtol is not a matched-error
comparison between these controllers.

The raw full/fine difference is used without a Richardson divisor or state
extrapolation. Startup, frequency changes and nonuniform BDF histories do not
justify assuming a universal factor of three. The width proposal uses the
minimum trial order, including BE safeguards. Growth defaults to 1.2, so a
new full BDF trial is at most 2.4 times the preceding accepted half width,
within the existing `1+sqrt(2)` guard. Larger user-selected growth still goes
through that guard. Macro width limits do not prohibit smaller internal
half steps or a shorter event-aligned final interval.

Both halves preserve the normal frequency, historical exponent and BE
safeguards. Nonpositive/nonfinite fine population rejects the macro attempt;
an invalid first half can stop that attempt early. Linear solver exceptions
still surface to the caller. Implicit feedback, external error-state callbacks
and the coupled thermal driver explicitly refuse this controller until their
provisional states and heat quadrature have their own rollback treatment.

Declared events are exact macro endpoints. The event state belongs to the
preceding smooth segment; history is restarted before leaving it. FTM2's
physical derivative is reseeded at that state. The separate alignment seam
hits an endpoint without restarting history.

## Verification

The [new tests](../../tests/verification/test_exponential_adaptive.py) cover:

- Tolerance refinement against independent two-mode kinetics for prompt rise,
  rapid rise, shutdown and an interior shutdown.
- Exact replay of accepted half steps after rejected/coarse trials, for
  Cartesian, cylindrical and triangular geometries, including final flux,
  precursors, event continuity and callback counts.
- Independent Radau references with six non-equilibrium precursor families.
- A localized two-group/two-family perturbation on four reflecting cells,
  with distinct delayed spectra and an independently assembled 16-state
  diffusion/kinetics matrix exponential. This checks changing spatial shape;
  it is not a TWIGL application or spatial convergence study.
- Separate field scaling, inactive cells, volume weights, initially dark
  fields, frequency-corrector work, alignment, minimum-width failure and
  unsupported control combinations.

The masked triangular replay exposed an existing FTM2 defect: small inactive
flux components could acquire meaningless derivatives from the inactive
operator's identity rows, then grow while contributing no fission power.
FTM now initializes inactive flux/precursors to zero, excludes inactive cells
from neutron population and masks seeded physical derivatives. This is
needed for physical frequency estimates and finite physical-state histories.

The [focused verification record](chunk4-verification.xml) has 259 passes.
The [default CPU record](chunk4-cpu.xml) has 801 passes, 12 skips and one
example subprocess import failure from an uninstalled checkout. The
[four-check example retest](chunk4-examples-verification.xml) passes with the
absolute checkout path in `PYTHONPATH`, giving 802 distinct passing CPU
checks across both records. Ten slow cases were excluded. The
[QA record](../../docs/qa/garnet-exponential-adaptive-2026-10-26.json) retains
environment, source fingerprints and the isolated archive smoke test.
GPU execution and excluded slow studies have not been repeated.

## Matched physical error and work

[chunk4-adaptive.json](chunk4-adaptive.json) records 80 configurations:
four cases, five tolerances and four method families. Every configuration
uses the actual one-cell diffusion API, scalar initial width 0.01 s,
maximum width 0.02 s and minimum width 1e-11 s. There is no prescribed graded
mesh or reference-selected time grid. Event restarts are shared by all
methods. Adaptive BDF1--5 retains its existing norm and automatic order
selection; its allowed consecutive rejection count is 40.

The reference is the independent physical two-mode kinetics matrix used in
the damping checkpoint. Errors take the maximum of each run's macro
endpoints and an 8,001-point common grid using linear interpolation. Final
precursor error, nonnegative power and no wrong-direction increments are
additional gates. All endpoint solves are charged, including discarded full
steps, both half steps and rejected attempts. No runtime speedup is claimed.

Fewest tested solves meeting **0.1% maximum power and final precursor error**:

| Case | Adaptive FTM2 | Adaptive transformed BDF2 | Zero-frequency transformed BDF2 | Adaptive BDF1--5 |
|---|---:|---:|---:|---:|
| Prompt rise | 225 (0.08502%) | 180 (0.07215%) | 180 (0.08288%) | 84 (0.09451%) |
| Rapid rise | No tested configuration | 3,030 (0.08228%) | 2,118 (0.03114%) | 261 (0.09661%) |
| Shutdown | 693 (0.06029%) | 396 (0.05000%) | 246 (0.07955%) | 209 (0.04520%) |
| Interior shutdown | 702 (0.07472%) | 405 (0.05053%) | 258 (0.07850%) | 213 (0.04502%) |

Parentheses give maximum power error. These adaptive transforms can now meet
the shutdown budget that the preceding prescribed meshes missed. The extra
step-doubling solves and this conservative estimator still cost more than
the best tested adaptive BDF at matched output error.

The rapid-rise limit needs particular care. At rtol=1e-7, lagged FTM2 has
only **0.000463%** maximum endpoint power error and **0.000455%** final
precursor error, but its linearly interpolated output reaches **2.20658%**
error. An accurately integrated exponential tail can still permit long
macro steps. Tightening the state defect does not directly constrain the
curvature of the line segment used by `power_at`. This is an output
representation limit, not evidence of a 2.2% endpoint state error.

The zero-frequency controls use their own adaptive grids. Their lower work
at some matched output budgets does not isolate the transform's endpoint
benefit; different accepted widths and interpolation errors also contribute.
The earlier fixed/shared-mesh studies remain the controlled evidence for
that benefit. Adaptive BDF remains the strongest tested comparator here.

## Next bounded development

Follow-up: the [fifth checkpoint](CHUNK5.md) implements and verifies the
output-aware restriction below; the original measurements remain unchanged.

Validate an output-aware step restriction or physical dense interpolation
before inferring accuracy from sparsely reported exponential trajectories.
Then test a cheaper physical-state estimator against this three-solve
reference controller and expand the localized spatial check to a resolved
multigroup application such as TWIGL. Diagonal-only operator updates,
group/local frequencies, thermal adaptation and GPU timing remain separate
gates. This checkpoint establishes explicit physical error control and
rollback; it does not earn a production performance claim.

The subsequent [five-paper literature review](../../docs/exponential_literature_review_2026-10-26.md)
adds PARCS physical-source precursor controls and a separate bounded
KIOPS/matrix-action gate to the [development plan](../../docs/exponential_transformation_plan.md).

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m tools.assess_exponential_adaptive
python3 -m pytest tests/verification/test_exponential_adaptive.py -q
```
