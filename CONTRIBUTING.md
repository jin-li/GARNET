# Contributing

Changes are integrated in the `jin-li/NDgpu` fork through feature PRs to `dev`,
then a verified `dev` to `main` PR. Target the fork explicitly. Work intended
for the original repository needs a separate upstream review.

Install `.[test]` and run `python -m pytest -q` from the repository root. The
regular suite excludes tests marked `slow`; run selected expensive tests with
`python -m pytest -m slow <test-path>` when affected by a change. Set
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1` for comparable
small verification workloads. On NixOS use the documented development shell.

A numerical change needs an independent analytical limit, operator/pencil
comparison, or physical conservation check appropriate to its scope. Test
forward and adjoint orientation and all affected moment blocks. Do not loosen
physical gates or test tolerances to hide a regression. Device mocks establish
API control flow, not GPU execution; record real hardware checks separately.

After changing package code, rebuild the tracked Colab upload archive with
`python tools/build_src_zip.py`. The example checks reject stale archive code;
normal wheel and source-distribution builds remain separate.

Update the User and Software Methodology manuals with each feature. Explain
units, array orientation, assumptions, supported solver/backend combinations,
convergence settings and remaining limits. Experimental interfaces must be
identified as such. Add small synthetic examples that require no private
reactor data. Installation must remain usable without CuPy, Serpent or preCICE.

Install `.[docs]` and build as described in [documentation](docs/index.md).
The strict HTML build and both PDF builds must pass; inspect math, tables and
code listings visually. The tested Python documentation dependency snapshot is
`docs/requirements-build.txt`; core runtime dependencies remain separate.

Do not commit local research worktrees, Serpent/ACE libraries, raw simulations
or generated build artifacts. Preserve the exact source revisions used for
published calculations. The [integration inventory](docs/development/integration_inventory.md)
records retained and superseded feature branches; an archive entry is not a
request to delete provenance.
