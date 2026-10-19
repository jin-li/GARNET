# Integration verification

This integration retains numerical verification separately from reactor-model
validation. No new Serpent production simulations are needed to verify the
software changes. Superseded research branches remain available and a local
Git bundle preserves their history; no branch deletion is part of integration.

## Baselines and repairs

The unchanged original main (`d8f4433`) passed 587 regular CPU tests, with
12 skips and 10 slow tests deselected. Clean fork dev (`a8859c5`) passed 593
and exposed an existing example private-import violation. The profiling
example now exercises the public coupled-transient API instead.

The first combined suite found four additional regressions (1515 passed,
22 skipped, 10 deselected). Separate focused checks found the initial HPMR
adjoint and many-group fixed-source problems. Repairs retain the requested
physical gates and output assertions:

- Physical sources inherit the existing scalar roundoff bound; accelerated
  proposals still require strict nonnegativity, without clipping.
- Coupled fixed-source acceleration has a bounded restart allocation estimate
  and group-sweep budget, with ordinary sweeps when memory is unavailable.
- A bounded Anderson history window permits resolving clustered modes without
  discarding useful history after each nonmonotone residual step.
- Owned forward/coarse-adjoint transient initialization defaults account for
  dtype precision; explicit accuracy requests remain strict.
- The stationary-feedback test uses a more accurate supplied eigenpair, so its
  stationary-reference premise holds; its output tolerance is unchanged.
- Locally refined HPMR explicitly retains its historical TPFA geometry; new
  normal-distance interfaces remain available. Neither mode supplies a general
  nonorthogonal correction.

## Device checks

The synthetic source snapshot was exercised on an AMD Radeon 8060S and an
NVIDIA RTX 5070 Ti using CuPy 14.2. Each host passed native SP3/SDP1 GPU tests,
standard/common-mesh CPU/GPU comparisons, actual many-group acceleration,
work-budget and roundoff guards, and public event profiling. Device arrays
were checked directly rather than inferred from `device="auto"`.
The machine-readable synthetic result summary is {download}`GPU validation <gpu_validation.json>`.
These checks establish numerical backend agreement for the recorded small
problems; they are not warmed performance or peak-memory benchmarks.

## Manuals and packaging

Strict HTML builds validate references; both manual PDFs are generated from
the same maintained sources. Generated artifacts remain outside version
control. Portable synthetic examples run without Serpent, ACE, CuPy or
reactor data, including from a clean wheel installation. The tracked Colab upload archive is rebuilt and checked byte-for-byte against
current Python sources and vendored NPZ data. GitHub Actions adds
regular CPU checks on Python 3.10/3.13, packaging/example smoke checks and strict
HTML/PDF builds. GPU validation remains a separately recorded hardware check.

Final source hashes and final-suite outcomes are recorded with the completed
integration evidence; the initial failing runs above remain diagnostic history.

## Final local gate

The final complete CPU run passed **1524 tests**, with 22 optional skips and
10 slow tests deselected (501.83s, single-thread BLAS). The new byte-level
Colab archive check was added after that collection and passed separately;
the five example/API/archive checks all pass. The combined regular suite
therefore contains 1525 passing tests plus the same 22 skips before platform-
specific differences. GitHub CI runs the complete collection again.

Both final GPU hosts passed two native tests, ten CPU/GPU comparisons,
profiling/budget/roundoff checks and two FP32 monolithic coarse-correction
initialization paths. GPU states remained CuPy arrays. Neither hardware run
used a precision fallback. Final peak device memory/performance is not claimed.

Clean-wheel examples were copied outside the source checkout and run with
Python isolated mode; package imports were confirmed to come from the installed
wheel. The source distribution includes the manual sources and excludes local
scratch directories and generated manuals.

Detailed source hashes and local gate results: {download}`integration validation <integration_validation.json>`.

## Static branch-library development

The subsequent experimental branch-library implementation has **1582 distinct
passing software tests**, with 16 optional skips and 10 slow tests deselected,
combining the regression run and final focused checks. The first regression
run found a source archive that became stale during final API hardening; the
archive was rebuilt and its byte-level check passed in the final 56-test run.
The new branch-specific public checks comprise 51 tests.

The wheel and source distribution build successfully. Four examples ran from
an isolated wheel installation outside the checkout, and the final wheel's
module bytes match the source. Strict HTML/PDF builds and visual review of the
new manual chapters passed. Numerical fixtures do not establish reactor
interpolation accuracy: that qualification awaits independent transport states.

Source hashes and the exact verification scope:
{download}`branch-library validation <branch_xs_validation.json>`.
