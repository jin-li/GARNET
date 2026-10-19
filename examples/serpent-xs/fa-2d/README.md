# Reflective 2-D assembly from regional Serpent constants

The local Serpent input `i2_cro_e71_2d_4g` supplies the rectangular bounds and
its `_res.m` file supplies one four-group record for each universe 1001–1289.
`run.py` reads these as data, preserving the supplied files unchanged. Neither
raw file is included in the feature commits.

## Geometry and method

The domain is a 21.5313 × 21.5313 cm square. Interior bands are 1.2618 cm wide;
the outermost bands are 1.30215 cm wide, including the peripheral water in
their homogenization. Each edge is therefore extended by 0.04035 cm relative
to a uniform 17-pitch grid. Universe `1001 + 17*row + column` increases with x
first, then y, starting at the lower left. Bounds are read directly from the
input and checked against that ordering rather than inferred from the pitch.
There are 264 fissile and 25 non-fissile regions.

NDgpu's unstructured finite-volume solver uses axis-aligned rectangular cells,
the actual areas and centroid-to-face distances, and reflective boundaries
(`alpha_boundary=0`). Interface conductance is
`face_length / (distance_left/D_left + distance_right/D_right)`, including
unequal-width material interfaces. This is an orthogonal mesh. A unit axial
depth is a convention for the 2-D solve, not a physical assembly height.

Refinement splits each homogenized region into equal rectangles without
changing its XS. Factors 1/2/4/8 correspond to 17/34/68/136 cells per axis.
Results are integrated back onto the original 289 regions before comparison.
No SPH or discontinuity factors are applied.

The default is explicitly `INF_DIFFCOEF`, an out-scattering diffusion
approximation. The supplied CMM coefficients are unsuitable for assuming a
local transport model: the first-to-last records' first-group values range
from 1.73898 to 472.971 cm. Serpent's CMM method is intended for homogenization
covering the full infinite-lattice geometry, as described in the
[group-constant guidance](https://serpent.vtt.fi/docs/user_guide/gc_generation.html).
`--diffusion cmm` is available only as an explicitly labelled diagnostic study.
The small errors obtained with INF do not validate it for general core models.

## References and comparisons

The common assembly reference is `IMP_KEFF = 1.18309`, with relative standard
error `4.1e-5` (absolute standard error about `4.85e-5`). Regional `INF_KINF`
is not a reference eigenvalue for the coupled assembly.

Serpent `INF_FLX` represents spatial/group integrals with a common source
normalization. Sum the regional integrals first, then normalize the assembly
spectrum. Do not multiply those reference integrals by area again or normalize
each region separately. NDgpu flux is a cell-average field and does require
area integration. The [Serpent output documentation](https://serpent.vtt.fi/docs/user_guide/standard_output.html)
defines the flux and fission fields.

Reference regional fission energy is reconstructed as
`sum_g(INF_FLX_g * INF_FISS_g * INF_KAPPA_g)` and normalized to sum one.
This uses the recorded heterogeneous-region reaction balance; it is not an
independent detector measurement. The supplied input's power detector is
commented out. Reconstructed total flux, neutron production, fission rate and
fission power close against the global Serpent tallies within 1.5e-6 relative,
supporting the normalization convention. `INF_NSF` is not used as a power proxy.

The runner reports signed k difference, reactivity difference in pcm, spectrum
maximum absolute/relative L2 errors, and fuel-region power RMS/maximum relative
errors. Zero-power guide/instrument regions are excluded from relative power
errors. CSVs contain all 289 normalized powers and group flux contributions.
Raw regional flux uncertainties are preserved; no covariance-based uncertainty
propagation or pin-power significance claim is made.

Numerical acceptance requires convergence, an independent sparse balance
residual ≤1e-8, and CPU/AMD k error ≤1e-8 and globally normalized flux maximum
error ≤1e-7. The 289-cell solve is additionally compared against a separately
assembled SciPy sparse eigenproblem. `passed` indicates these numerical gates;
Serpent physics differences are reported without choosing thresholds after
seeing the results.

## Reproduce

Use the existing repository Nix/uv convention; see the root README for CPU
and ROCm environment installation. From the repository root:

```sh
nix develop path:. --command .venv/bin/python examples/serpent-xs/fa-2d/run.py --refinements 1 2 4 8
nix develop path:.#rocm --command .venv/bin/python examples/serpent-xs/fa-2d/run.py --device gpu --require-rocm --refinements 1 2 4 8 --output-dir results/serpent-fa-2d-halo
nix develop path:. --command .venv/bin/python -m pytest -q tests/verification/test_serpent_fa_2d.py tests/verification/test_serpent_xs.py tests/verification/test_unstructured_mesh.py

# Optional figures: isolated uv environment, no project dependency changes
nix develop path:. --command uv run --no-project --with matplotlib --python .venv/bin/python examples/serpent-xs/fa-2d/plot.py results/serpent-fa-2d-halo
```

GPU runs require actual CuPy arrays; `--require-rocm` additionally requires
HIP. Fusion toggles exercise shared Krylov kernels; the irregular mesh gather
and group sweeps retain their existing array implementation. The runs do not
claim fused structured stencil or graph-capture coverage. Solve timings are
synchronized diagnostics, not warmed CPU/GPU performance benchmarks.

Local reports go under ignored `results/`. Selected halo evidence and figures
are in [benchmark-results/serpent-fa-2d-halo](../../../benchmark-results/serpent-fa-2d-halo).
These are fixed-state pin-homogenized diffusion comparisons. Before extending
to leaking full-core models, assess regional transport corrections, interface
equivalence and boundary conditions with separate references.
