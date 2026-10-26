# Third checkpoint: stiff damping and transformed BDF2

26 October 2026. This follows the [prompt-event checkpoint](CHUNK2.md).
Neutron and delayed-neutron integration only; the heat equation is unchanged.

An experimental damped variant is now available through the diffusion API:

```python
run = core.transient(
    t_end=0.1, dt=0.001, time_scheme="ftm-bdf2", device="cpu",
    step_solver="monolithic",
)
print(run.time_orders, run.step_frequencies, run.ftm_fallback_reasons)
```

`core` needs materials and kinetics, as in the
[transient guide](../../docs/transient_api.md). Fixed and explicit nonuniform
widths work on Cartesian, cylindrical and triangular grids. Both spatial
solvers work; optional frequency refinement requires monolithic stepping.
The method remains experimental. It has not displaced adaptive BDF as the
preferred comparator or earned a production performance claim.

## Method and safeguards

This applies BDF2 to **both neutrons and precursors** after removing one
shared exponential during each step. Every old physical state is converted
to the current step's frame, rather than reusing an old transformed history.
For current width h, previous width k and ratio r=h/k:

```text
a0 = (1 + 2*r)/(1 + r)
a1 = -(1 + r)
a2 = r*r/(1 + r)
H  = -a1*exp(omega*h)*y_n - a2*exp(omega*(h+k))*y_(n-1)

M*(a0/h + omega)*y_new - R(y_new, t_new) = M*H/h
```

M contains inverse neutron speeds and unit precursor masses. A precursor
family therefore satisfies

```text
C_new = (H_C + h*beta*S_new)/(a0 + (lambda + omega)*h).
```

The fission-source elimination retains GARNET's total-spectrum convention and
family-dependent delayed spectra. This precursor update is a rational BDF
recurrence. It differs from FTM2's analytic integral of linearly interpolated
transformed production. At zero frequency the complete method reproduces
ordinary BDF2, including its precursors.

One endpoint solve is needed per accepted step before any requested frequency
refinement. Startup and declared history restarts use BDF1. Frequency is
estimated from accepted neutron populations and the actual previous width;
`ftm_kwargs={"frequency": 0.0}` supplies a control. Existing frequency
refinement options and work counters also apply.

Unsafe steps use **actual backward Euler**, with order one and a reason in
`ftm_fallback_reasons`:

| Reason | Trigger |
|---|---|
| `frequency-bound` | `abs(omega*h)` exceeds `max_exponent` (default 0.5) |
| `history-frequency-bound` | The older history exponent exceeds twice that bound |
| `step-ratio-bound` | Width grows by more than `1 + sqrt(2)` |

The history bound prevents large exponential weights after abrupt width
shrinking. The ratio guard avoids an amplifying parasitic-history coefficient.
These safeguards do not establish unconditional stability or positivity for
arbitrary frequency updates, materials or time grids. The existing solver
still rejects nonpositive global neutron populations; no clipping is added.

At fixed frequency and width, BDF2 damps stiff modes that transformed trapezoidal integration
can retain. Damping does not supply temporal resolution: an underresolved
shutdown can still ring and have large early errors. `adaptive_bdf` is not an
FTM error controller and is explicitly refused for this method.

Thermal coupling receives the ordinary endpoint heat weight h. No heat
transform, new thermal quadrature or coupled second-order accuracy claim is
introduced.

## Why the two-stage candidate remains a research driver

Before extending the FV engine, the independent kinetics driver tested the
two-stage second-order SDIRK method of
[Alexander (1977)](https://doi.org/10.1137/0714068), with
`gamma = 1 - 1/sqrt(2)`, in a frozen exponential frame. This L-stable method
damps stiff decay, but can produce negative values at coarse widths; a
scalar counterexample is retained as a test.

Two implicit stages would mean two spatial solves per macro step. For the
rapid-rise case at a 1% error budget, lagged transformed SDIRK2 needs 200
tested stage solves, compared with 113 for FTM2/transformed BDF2 and 90 for
adaptive BDF. At 0.1%, the corresponding counts are 500, 296 and 251.
On the resolved shutdown, its two stages also fail to earn their extra work.
SDIRK2 therefore remains a dense research prototype; there is no SDIRK2
dispatch in `TransientSolver`. Its stage counts are prospective spatial work,
not measured FV timings or Krylov counts.

## Recorded comparison

[chunk3-damping.json](chunk3-damping.json) records **196 configurations**:
49 each for prompt insertion, rapid rise, shutdown and an interior shutdown.
The first three use the same physics as the previous checkpoint. The
interior case stays critical until 0.05 s and then applies rho=-0.05, ending
at 0.25 s. The independent reference is the physical two-mode kinetics matrix.
Each FV control uses the public model API on a reflective one-cell grid,
which reduces exactly to that kinetics system. No spatial-convergence result
is claimed by this comparison.

On the same graded rapid-rise mesh (first width 1e-5 s, growth 1.1,
maximum width 0.01 s):

| Scheme | Maximum power error | Endpoint power error | Final precursor error | Endpoint solves |
|---|---:|---:|---:|---:|
| Zero-frequency transformed BDF2 | 17.4450% | 16.9474% | 16.9335% | 113 |
| Lagged transformed BDF2 | 0.8304% | 0.1230% | 0.0450% | 113 |
| Lagged FTM2, two BE startup steps | 0.8022% | 0.1090% | 0.1088% | 113 |

The transform reduces the BDF2 control's maximum power error about 21-fold at
the same endpoint-solve count. This is evidence for the exponential factor
on that trajectory; it is not an advantage over FTM2 or adaptive BDF.

Fewest tested solves meeting **0.1% power and final precursor error**, with
nonnegative power and no wrong-direction increments:

| Case | Lagged FTM2 | Lagged transformed BDF2 | Adaptive BDF1--5 |
|---|---:|---:|---:|
| Prompt rise | 163 (0.0125%) | 163 (0.0495%) | 74 (0.0933%) |
| Rapid rise | 296 (0.0574%) | 296 (0.0898%) | 251 (0.0966%) |
| Shutdown | No tested configuration qualifies | No tested configuration qualifies | 200 (0.0449%) |
| Interior shutdown | No tested configuration qualifies | No tested configuration qualifies | 218 (0.0453%) |

Parentheses give maximum power errors. At the looser 1% budget, the graded
shutdown qualifies with 83 solves for both FTM2 and transformed BDF2. Their
errors are 0.2966% and 0.4642%, respectively. Reducing only the later width cap
cannot fix errors inherited from the prescribed initial mesh.

Errors take the maximum of each run's endpoints and an 8,001-point common
grid using linear interpolation. Final precursor error is an additional
gate; it does not bound the whole precursor history. Accepted and rejected
solves, Krylov iterations, operator rebuilds and fallbacks are retained.
All adaptive controls allow 40 consecutive rejections: the tight interior
shutdown needs more than the default 20 halvings immediately after restart.
This changes the allowed retry count, not the accuracy criterion. Graded
meshes are prescribed, not adaptive FTM. No CPU/GPU runtime speedup is claimed.

## Verification and next work

The [new verification module](../../tests/verification/test_exponential_damping.py)
checks smooth second-order refinement, complete eigenmodes, non-equilibrium
six-family precursors against Radau, and distinct delayed spectra against a
matrix exponential. It also checks spatial-solver agreement with mapped
material kinetics, current-frame histories on all three grid types, BE
fallback, frequency corrector accounting, event continuity, a stiff shutdown
mode, and existing thermal exchanges/progress callbacks. See the
[V&V report](../../docs/verification_validation.md#exponential-damping-check)
for the full CPU run and source fingerprints.

The next useful step is a physical-state error controller with explicit
rollback and event handling, followed by a localized multigroup/shape case
such as TWIGL. A better damping property alone did not beat adaptive BDF in
these comparisons. Local/group frequencies, inner-iteration fitting and GPU
performance remain separate gates. Heat-transfer integration remains outside
this exponential-transform workstream.

Follow-up: the [fourth checkpoint](CHUNK4.md) implements the physical-state
controller, checks a localized multigroup shape change and records the
remaining output-interpolation and work limitations.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/assess_exponential_damping.py
python -m pytest tests/verification/test_exponential_damping.py -q
```
