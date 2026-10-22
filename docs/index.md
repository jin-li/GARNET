# NDgpu documentation

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
regional_xs
development/subpin-mesh-geometry
development/corrected-fv
development/corrected-mesh-moments
amd_rocm
kinetics_parameters
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
files are `docs/_build/latex/ndgpu-user-manual.pdf` and
`docs/_build/latex/ndgpu-methodology-manual.pdf`. Generated HTML and PDFs are build
artifacts; hosting and package publication are separate actions.
