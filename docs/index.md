# GARNET documentation

Start with the User Manual for running calculations. The Software Methodology
Manual explains the equations, numerical methods and limits of the implementation.
Both manuals are maintained as Markdown and built as searchable HTML and PDF.

```{toctree}
:maxdepth: 2
:caption: Manuals

user/index
methodology/index
```

```{toctree}
:maxdepth: 1
:caption: Detailed guides and development evidence

user_guide
model_api
coupling
serpent_xs
amd_rocm
kinetics_parameters
api_audit
verification_validation
development/integration_inventory
development/validation
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
`docs/_build/latex/garnet-methodology-manual.pdf`. Generated HTML and PDFs are build
artifacts; hosting and package publication are separate actions.

```{toctree}
:maxdepth: 1

development/garnet-migration
```
