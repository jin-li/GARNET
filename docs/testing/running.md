# Running and recording checks

Run commands from the repository root in the intended Python environment. The
runtime requires NumPy and SciPy; the test extra adds pytest. CPU testing does not
require CuPy, Serpent, ACE libraries or preCICE.

```sh
python -m pip install -e '.[test]'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q -ra --junitxml=results/cpu.xml
```

Single-thread settings reduce oversubscription and improve comparability for
small numerical checks. They do not guarantee identical floating-point results
across BLAS libraries or devices. On NixOS use the repository development shell;
a missing shared library is an environment failure to investigate before
interpreting numerical results.

## Selection rules

`pyproject.toml` sets `testpaths = ["tests"]` and
`addopts = "-m 'not slow'"`. `tests/conftest.py` separately controls legacy
collection with `--run-legacy`. Directory names do not determine cost: expensive
verification checks also exist, and individual `slow` marks select costlier
cases. An unmarked case may still take appreciable time.

| Intent | Exact command | Selection |
| --- | --- | --- |
| Default regression | `python -m pytest -q -ra` | Current suite, excludes `slow` and legacy collection |
| One affected feature | `python -m pytest -q -ra tests/test_branch_workflow.py` | That module, still excludes any marked slow cases |
| Only expensive current checks | `python -m pytest -q -ra -m slow tests` | CLI marker replaces the configured default expression |
| All current checks | `python -m pytest -q -ra -o addopts= tests` | Removes default slow exclusion; legacy remains opt-in |
| Legacy, ordinary cost | `python -m pytest -q -ra tests/legacy --run-legacy` | Legacy collection enabled, slow cases still excluded |
| Legacy, including slow | `python -m pytest -q -ra -o addopts= tests/legacy --run-legacy` | All archived cases |
| Everything | `python -m pytest -q -ra -o addopts= tests --run-legacy` | Current and archived cases, including slow |
| Inspect selected inventory | `python -m pytest --collect-only -q tests` | Default collection/selection without solving |

Use a node ID such as `path/to/test.py::test_name` to reproduce a specific
failure. Combine `-x` with `-vv` for the first failing assertion and
`--durations=20` to identify expensive checks. Do not change default tolerances
or physical gates merely to make a selected test pass.

## Optional dependencies and skipped evidence

Always retain `-ra` output and report skip reasons alongside totals. A skipped
test provides no evidence for the omitted path. Optional checks include:

- FEMFFUSION reader and VVER mesh checks that require a separate local reference
  checkout; inspect the module's path guard and obtain the documented public
  revision before running them.
- `tests/validation/test_coupled_precice.py`, which uses
  `pytest.importorskip("precice")`; install the compatible preCICE library and
  Python bindings as described in the [coupling guide](../coupling.md).
- Symbolic SDP derivation (`sympy`) and selected transport preconditioning
  (`pyamg`) checks. These are separate optional tools, not core dependencies.
- Local Serpent result probes in `test_serpent_adf_reader.py` and
  `verification/test_serpent_xs.py`. Synthetic parser tests run without them.
  The ADF probes use `GARNET_SERPENT_ADF_CORE_PROBE` and
  `GARNET_SERPENT_ADF_INVALID_CORE_PROBE`; read the expected qualification
  contract in those tests before selecting local files.
- Actual-device tests requiring an importable CuPy distribution and an available
  compatible runtime/device. Install only the appropriate CUDA or ROCm extra.

For example, run the explicit coupled-moment hardware comparison with:

```sh
python -m pytest -q -ra tests/test_coupled_moment_gpu.py::test_actual_gpu_against_cpu_and_independent_dense
```

This test compares SP3/SDP1 results with CPU and an independent dense eigenproblem,
checks full residuals, and exercises a limited inner budget and corrupted
assembly. Other tests in that module use doubles to check API and transfer
contracts. A passing module with hardware cases skipped is not GPU verification.
Once device availability is established, numerical or API failures should fail,
not become dependency skips. Also distinguish the hybrid coupled-moment adapter
from a fully device-resident solver; see the methodology capability table.

## Record a reproducible result

Record the source revision and local changes, exact command, Python/package
versions, OS and BLAS/thread settings, selected and deselected counts, skips,
failures and elapsed time. For device runs include the requested and observed
backend, GPU model, driver/runtime and CuPy version. Record input hashes, source
versions, solver controls, residuals and refinement choices for numerical studies.
Keep JUnit XML and logs as artifacts; a test count alone loses this context.

The CPU CI matrix and documentation workflow are defined in
`.github/workflows/verify.yml`. CI runs the default selection on its configured
Python versions, builds distributions, exercises a clean wheel and builds all
three manuals. It does not run the full slow/legacy inventory, private-data
probes or provide actual-GPU evidence.
