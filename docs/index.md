# GARNET documentation

Start with the User Manual for running calculations. The Software Methodology
Manual explains the equations, numerical methods and limits of the implementation.
The Testing and Verification Manual explains executable evidence, test selection
and release checks. All three manuals are maintained as Markdown and built as searchable HTML and PDF.

```{toctree}
:maxdepth: 2
:caption: Manuals

user/index
methodology/index
testing/index
```

```{toctree}
:maxdepth: 1
:caption: Detailed guides and development evidence

user_guide
model_api
transient_api
coupling
serpent_xs
amd_rocm
kinetics_parameters
api_audit
verification_validation
development/integration_inventory
development/validation
```

## Experimental fitted-mesh and native methods

Use the [method selection guide](development/native-methods.md) to distinguish
corrected diffusion, native full-matrix moments and angular transport pilots,
then follow the relevant API contract. These opt-in paths retain separate
validation and device scopes.

```{toctree}
:maxdepth: 1
:caption: Experimental mesh and native methods

development/native-methods
development/subpin-mesh-geometry
development/polygon-mesh
development/corrected-fv
development/corrected-mesh-moments
development/corrected-native-moments
development/moment-homogenization
development/native-angular-source
development/native-mesh-streaming
development/native-mesh-transport
development/native-angular-p0
development/native-nem-p0
```

## Building locally

Install the optional documentation dependencies with `pip install -e '.[docs]'`.
From the repository root, run:

```sh
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m sphinx -M latexpdf docs docs/_build -W --keep-going
```

On NixOS, `nix develop .#docs` supplies the pinned TeX and make tools.
Install `docs/requirements-build.txt` to reproduce the tested Python documentation
environment.

PDF builds additionally require pdfLaTeX, latexmk and make from TeX Live. The generated
files are `docs/_build/latex/garnet-user-manual.pdf` and
`docs/_build/latex/garnet-methodology-manual.pdf`, plus
`docs/_build/latex/garnet-testing-manual.pdf`. Generated HTML and PDFs are build
artifacts; hosting and package publication are separate actions.

```{toctree}
:maxdepth: 1

development/garnet-migration
```

## Research and verification records

These records preserve scoped numerical experiments, input uncertainties and
implementation audits. They supplement the manuals; their presence does not
qualify a reactor model or promote an experimental method to production use.
Development plans remain excluded from the manual sources and are linked as
downloads where needed for provenance.

```{toctree}
:maxdepth: 1
:caption: Research and verification records

adf_slab_reference
bernal_adf_benchmark
c5g7_colorset_adf
numerical_implementation_notes
dallosso_1d_reference
qa/adf_audit_2026-10-27
qa/code_audit_2026-10-28
exponential_literature_review_2026-10-26
exponential_proposal_review
moc_2d1d_feasibility_2026-10-26
```
