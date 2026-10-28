# Extending, diagnosing and shipping checks

## Add a meaningful test

Choose the layer by the assertion: a focused feature contract belongs at the
test root, a mathematical or API invariant in `verification/`, and an end-to-end
named benchmark comparison in `validation/`. Add expensive cases with
`@pytest.mark.slow`; keep a small independent limit in the ordinary selection
when feasible. Legacy tests preserve earlier entry points and are not the place
for new feature coverage.

Start from the physical or algebraic contract. Use an analytic limit,
independently assembled dense operator/pencil, conservation identity or published
reference with a stated uncertainty. Include forward and adjoint orientation,
unequal volumes, off-diagonal group/moment coupling and boundaries when those
features matter. A second call to the implementation under test is not an
independent oracle. Explain tolerance choices in terms of discretization,
iteration error, floating-point conditioning or reference uncertainty.

Test invalid and underspecified inputs where they can change the physical
meaning: orientation, group order, units, geometry, fission representation,
closure identity, estimator qualification and provenance. Assert that failures
are explicit. Avoid silently renormalizing away data mismatches. Keep optional
hardware checks separate from mock-only API checks so unavailable hardware is
visible in reports.

## Fixtures and provenance

Use small synthetic materials and meshes by default. Serpent reader tests write
literal synthetic result blocks to `tmp_path`; ACE fixtures likewise construct
records without licensed nuclear libraries. This keeps ordinary checks portable
and allows deliberately malformed cases. No private reactor datasets, ACE
libraries, raw Serpent runs or machine-specific absolute paths belong in the
repository or its test artifacts.

For a public reference record its title/source, version or revision, units,
transcription/transformation, normalization, uncertainties and applicable license.
Keep benchmark builders and shared reference constants together under
`garnet/benchmarks/`. If a local external file is optional, use a narrow explicit
guard and state its absence as a skip. Keep hashes and declared identities in
derived snapshots; a hash identifies bytes and does not qualify their physics.
Do not make private file locations or contents visible in published CI logs.

## Failure triage

1. Reproduce the failing node with the original command and environment. Separate
   import/shared-library/optional-dependency problems from assertion failures;
   collect the skip summary and deselection rules before comparing totals.
2. Check units, group/moment orientation, material/mesh identities, boundary
   conventions and source normalization. For changed reference inputs inspect
   hashes, versions and transcription before replacing a baseline.
3. Read convergence diagnostics. Compare eigenvalue change, fission-source
   change, full balance/residual, positivity and inner-solver status. Inspect
   outer and inner budgets, restart semantics and failure reasons. A nearly
   constant eigenvalue can coexist with an unconverged spatial source.
4. If an iteration cap is reached, perform a controlled budget/tolerance study
   and compare to an independent solution or residual. Increase a budget only
   with evidence that the same physical gate is attained. Preserve tests of
   deliberately exhausted budgets and explicit failure states. Do not convert
   a failure into a skip or relax the acceptance gate to hide it.
5. For transient failures inspect accepted/rejected time steps, event alignment,
   rollback, error norms and physical-state/output errors. Refine space and time
   separately; agreement caused by a common spatial error is not temporal
   verification. For GPU failures record the actual runtime and device and check
   a CPU/dense oracle before attributing the difference to roundoff.

`test_nem_requested_convergence.py` independently gates absolute eigenvalue
change, relative source change and full eigen residual when requested.
`test_eigen_source_convergence.py` exercises source/balance gates and exhausted
budgets across eigenvalue pathways. `test_coupled_moment_gpu.py` checks complete
restart budgets and true inner iteration counts. These controls have different
contracts; do not assume a single `maxiter` has identical meaning everywhere.

## Package, archive and manual checks

After a package change regenerate the tracked Colab upload archive and check
its contents against the checkout:

```sh
python tools/build_src_zip.py
python -m pytest -q tests/verification/test_examples_current.py
python -m pip install build
python -m build --outdir build/dist
```

The archive test compares README, project metadata, package Python files and
packaged NPZ data with `dist/garnet-src.zip`. It does not replace wheel/sdist
validation. Install the wheel in a clean environment and run examples outside
the checkout so local imports cannot mask a packaging problem:

```sh
python -m venv /tmp/garnet-wheel
/tmp/garnet-wheel/bin/python -m pip install build/dist/*.whl
mkdir -p /tmp/garnet-wheel-run/examples
cp -R examples/isonees_features /tmp/garnet-wheel-run/examples/
cp -R examples/branch_xs /tmp/garnet-wheel-run/examples/
cd /tmp/garnet-wheel-run
/tmp/garnet-wheel/bin/python -I -c "import garnet; print(garnet.__file__)"
/tmp/garnet-wheel/bin/python -I examples/isonees_features/common_mesh_fv.py
/tmp/garnet-wheel/bin/python -I examples/isonees_features/nem_local_reference.py
/tmp/garnet-wheel/bin/python -I examples/isonees_features/static_kinetics.py
/tmp/garnet-wheel/bin/python -I examples/isonees_features/branch_xs.py
/tmp/garnet-wheel/bin/python -I examples/branch_xs/thermal_star.py
```

Use fresh temporary directory names if these already contain an older run.
Check that `garnet.__file__` belongs to the clean environment. Return to the
repository root for the following documentation commands:

```sh
python -m pip install -r docs/requirements-build.txt -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m sphinx -M latexpdf docs docs/_build -W --keep-going
```

The PDF build requires pdfLaTeX, latexmk and make; `nix develop .#docs` supplies
the pinned tools on NixOS. The outputs under `docs/_build/latex/` are
`garnet-user-manual.pdf`, `garnet-methodology-manual.pdf` and
`garnet-testing-manual.pdf`. Strict Sphinx checks catch broken references and
source warnings, but visual inspection must still check equations, long tables,
code wrapping, page headers and links. LaTeX layout warnings also deserve review.
Generated HTML/PDFs are CI artifacts; building them does not publish a site.

Update all affected manuals with a feature: user procedure, numerical contract,
and the verification procedure/evidence. Record remaining skips and limitations
in the change report. A clean package build, default CPU suite and manual build
are separate gates, each with a distinct claim.
