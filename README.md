# GARNET♦️: A GPU-native Advanced Reactor NEutronics Toolkit

**GARNET**️ is a Python-based reactor physics solver for multigroup neutron diffusion and simplified transport (SPN/SDPN). It is a GPU-focused code; the employed methods are meant to simplify GPU implementation and optimise performance. The code is written with CuPy (CUDA Python) for GPU with an automatic numpy fallback for testing on CPU. The main application is homogenized full-core transient analysis.

GARNET was formerly called **NDgpu**; that name may still appear in saved
results and older reports. The Python package is now `garnet`.

The core solvers support:
- Cartesian and cylindrical r-z grids, body-fitted triangular meshes, and
  extruded triangular prisms; unstructured meshes for diffusion.
- Criticality and adjoint calculations with diffusion, SP1/3/5/7, and SDP1/2/3.
- Neutron kinetics, coupled thermal transients, and quasi-static acceleration.
- SPH correction and cross-section readers for Griffin and FEMFFUSION data.

## Built for GPU

- **Matrix-free stencils.** On structured grids, operators work directly with
  each cell and its neighbours, avoiding the cost of assembling and storing
  large sparse matrices.
- **Device-resident fields.** Fluxes, material data, and temperatures stay on
  the GPU during iterations, reducing transfers to and from the CPU.
- **CUDA kernel fusion.** Custom kernels combine small array operations into
  fewer GPU launches, reducing launch overhead and temporary arrays.
- **Neumann polynomial preconditioning.** The optional preconditioner uses a
  few cheap parallel passes to improve each solver correction. This can reduce
  iterations and global synchronizations: the pauses needed to combine results
  across the GPU.
- **Warm starts and adaptive tolerances.** Solves reuse previous solutions.
  Early iterations use looser accuracy targets, which tighten as the calculation
  converges. Working memory is reused too.

See the [GPU profiling and measurements](docs/gpu_kernel_optimization_plan.md)
for timings and hardware-dependent trade-offs. Users without access to GPU
hardware can test drive the [Google Colab notebooks](https://github.com/poncetma/GARNET/tree/main/notebooks).


## Fork extensions and manuals

This fork is maintained at [jin-li/GARNET](https://github.com/jin-li/GARNET)
and follows [poncetma/GARNET](https://github.com/poncetma/GARNET).
The Python distribution and import are both `garnet`. Existing environments
should reinstall the editable project after updating. Historical research
commits and saved results retain their original NDgpu names.

- [User Manual](docs/user/index.md) and [Software Methodology Manual](docs/methodology/index.md).
- [Testing and Verification Manual](docs/testing/index.md): commands, coverage,
  provenance, failure diagnosis and release checks.
- [HTML/PDF build instructions](docs/index.md) and [feature inventory](docs/development/integration_inventory.md).
- [Serpent XS interface](docs/serpent_xs.md), [ADFs](docs/user/serpent-mesh.md),
  [kinetics parameters](docs/kinetics_parameters.md), and [NEM](docs/user/nem-kinetics.md).

The fork retains CPU/AMD support and experimental whole-face NEM, reflector
ADFs, static adjoint kinetics and native-moment paths. Their documented
limitations remain in force. Experimental features are integrated through
feature branches into `dev`, then `main`. Frozen study revisions remain
available in Git history; private study repositories are excluded from the
software distribution.

## Install

Python 3.10 or later is required. From a checkout:

```bash
python -m pip install -e .              # CPU: NumPy and SciPy
python -m pip install -e '.[cuda12]'    # GPU: also install CuPy for CUDA 12
```

Set `device="cpu"`, `"gpu"`, or `"auto"` on a solve. `auto` selects a GPU
when available and otherwise uses the CPU; `gpu` requires a working CUDA
installation.

## Quickstart

A homogeneous two-group box using the supplied demonstration material:

```python
from garnet import Model, PWR_TWO_GROUP

core = (Model(size=(90, 90, 90), cells=(24, 24, 24))  # lengths in cm
        .fill(PWR_TWO_GROUP)
        .set_boundary("vacuum"))

result = core.run(device="auto")
print(result)             # eigenvalue, convergence, and neutron balance
print(result.k_eff)
flux = result.flux        # NumPy array: (groups, nx, ny, nz)
```

Use `Model` for Cartesian geometry, `CylindricalModel` for r-z,
`HexLattice.build()` → `TriReactor` for hexagonal or prismatic cores, and
`MeshModel` for an existing unstructured mesh. The
[user guide](docs/user_guide.md) walks through materials, geometry,
SPH data, moving control drums, and coupled calculations.

## Documentation

- [User guide](docs/user_guide.md), [model API reference](docs/model_api.md), and [API audit](docs/api_audit.md).
- [Transient and r-z workflows](docs/transient_api.md): timed controls, adaptive BDF, progress callbacks, and reusable results.
- [Thermal coupling](docs/coupling.md): conduction, feedback, energy balance,
  and preCICE integration.
- [Finite-volume theory report](docs/report.html): discretization and solver
  derivations, with the original July 2026 benchmark snapshot.
- [Examples](examples/) and a [GPU walkthrough](notebooks/colab_coupled_transient_gpu_latest.ipynb).

## Verification and validation
**GARNET** code is AI generated. For that reason, there is greater emphasis on qualification via verification and validation. Results of major V&V tests are examined with human eyes.

More than 700 automated test cases cover analytic solutions, spatial and
temporal convergence, conservation, and published reactor benchmarks. The
[V&V report](docs/verification_validation.md) maps each core feature to its
tests, states the acceptance criteria, and records the current CPU results.

| Evidence | Report or study |
|---|---|
| Diffusion, SPN/SDPN, mesh convergence, adjoints, and SPH | [Core numerical verification](docs/verification_validation.md#core-numerical-verification) |
| C5G7, BIBLIS, IAEA-3D, VVER-440, TWIGL, and ANL kinetics | [Reactor benchmark comparisons](docs/verification_validation.md#reactor-benchmark-comparisons) |
| Conduction, feedback, energy balance, and coupling | [Thermal and coupling checks](docs/verification_validation.md#thermal-and-coupling-checks) |
| Adaptive BDF, LRA-2D accuracy, and matched-error cost | [Rendered study (PDF)](benchmark-results/lra2d-adaptive-bdf-report/adaptive-bdf-lra-benchmark.pdf) · [data and reproduction](benchmark-results/lra2d-adaptive-bdf-report/) |
| 3-D HP-MR guarded quasi-static transients on GH200 | [Accuracy study and recorded results](benchmark-results/hpmr-iqs-gh200/) |
| T4 GPU accuracy and performance | [3-D transient study](benchmark-results/transient-acceleration/step-08a-adaptive-bdf-gpu-readiness.md) · [optimization studies](benchmark-results/transient-acceleration/) |

These checks cover the numerical methods and the stated benchmark models.
HP-MR demonstration cross sections and homogenization choices still need
assessment for a new design; the reports distinguish these from solver error.
GPU results are tied to the hardware and workloads used in each study.

To run the tests:

```bash
python -m pip install -e '.[test]'
python -m pytest tests                 # regular suite; excludes slow cases
python -m pytest tests -o addopts=     # include the expensive full-core cases
python -m pytest tests/legacy --run-legacy  # preserved pre-API tests
```

Optional tests require CUDA, preCICE, or external reference files. See the
[V&V report](docs/verification_validation.md#reproducing-the-checks) for setup
and the [test layout](tests/README.md) for the verification/validation split.

Licensed under the [Apache v2.0](LICENSE).

## References

Key sources for the formulations and numerical methods:

- **SDPN:** A. Carreño, A. Vidal-Ferràndiz, D. Ginestar and G. Verdú (2024),
  [Development of a diffusive simplified double PN approximation to the neutron transport equation](https://doi.org/10.1016/j.anucene.2024.110675).
  *Annals of Nuclear Energy* **207**, 110675. SDPN equations, coefficient tables,
  and boundary conditions.
- **SP3:** P. S. Brantley and E. W. Larsen (2000),
  [The Simplified P3 Approximation](https://doi.org/10.13182/NSE134-01).
  *Nuclear Science and Engineering* **134**, 1–21. Two-moment formulation and
  Marshak boundary conditions.
- **GPU diffusion and preconditioning:** Y. E et al. (2017),
  [Development of a GPU-accelerated 3D neutron dynamics code for PB-FHR](https://doi.org/10.1016/j.nucengdes.2017.04.001).
  *Nuclear Engineering and Design* **320**, 88–102. Fine-mesh finite volumes
  and Neumann-polynomial preconditioned CG on GPU.
- **SPH equivalence:** V. Labouré et al. (2019),
  [Hybrid super homogenization and discontinuity factor method for continuous finite element diffusion](https://doi.org/10.1016/j.anucene.2019.01.003).
  *Annals of Nuclear Energy* **128**, 443–454. Reaction-rate/leakage equivalence
  and SPH normalization, adapted here to finite volumes.
- **Diffusion transients:** A. Cherezov, A. Vasiliev and H. Ferroukhi (2025),
  [Application of Backward Differential Formula and Anderson’s method for multigroup diffusion transient equation](https://doi.org/10.1016/j.anucene.2024.110837).
  *Annals of Nuclear Energy* **210**, 110837. BDF integration, multigroup
  iteration, and transient benchmark comparisons.
- **Quasi-static kinetics:** J. Devooght and E. Mund (1980),
  [Generalized Quasi-Static Method for Nuclear Reactor Space-Time Kinetics](https://doi.org/10.13182/NSE80-A19288).
  *Nuclear Science and Engineering* **76**, 10–17. Background for flux
  amplitude/shape factorization; GARNET uses a single amplitude.
- **Anderson acceleration:** H. F. Walker and P. Ni (2011),
  [Anderson Acceleration for Fixed-Point Iterations](https://doi.org/10.1137/10078356X).
  *SIAM Journal on Numerical Analysis* **49**, 1715–1735. The least-squares
  mixing form used for SPH and coupled fixed-point solves.
- **Flexible Krylov solves:** Y. Saad (1993),
  [A Flexible Inner-Outer Preconditioned GMRES Algorithm](https://doi.org/10.1137/0914028).
  *SIAM Journal on Scientific Computing* **14**, 461–469. FGMRES with variable
  inner preconditioning for coupled energy-group systems.
## Fork GPU environments

On NixOS (x86_64 Linux), use the pinned development shell and a project-local
`uv` environment. Nix supplies Python and the native runtime libraries needed
by NumPy/SciPy wheels; `uv` installs the editable project and locks its Python
dependencies. Python downloads are disabled in the shell.

```bash
nix develop path:.
uv sync --locked --extra test
uv run --locked --extra test python examples/bare_reactor.py 16 cpu
uv run --locked --extra test python examples/reflected_slab.py 90 cpu
uv run --locked --extra test python examples/hex_lattice.py 2 cpu
uv run --locked --extra test python examples/transient_rod_drop.py cpu
uv run --locked --extra test pytest -q tests/verification/test_diffusion_analytic.py
```

For GPU execution, the `cuda12-ctk` extra installs CuPy and NVIDIA's CUDA
component wheels, including runtime libraries and compilation headers, without
a system-wide CUDA toolkit:

```bash
nix develop path:.
uv sync --locked --extra test --extra cuda12-ctk
uv run --locked --extra test --extra cuda12-ctk python examples/bare_reactor.py 32 gpu
uv run --locked --extra test --extra cuda12-ctk python examples/hpmr_3d.py 1 10 polar gpu
uv run --locked --extra test --extra cuda12-ctk python examples/speed_benchmark.py 32 64 96
```

Keep the GPU extra in subsequent `uv run` commands so environment syncing
retains it. The shell exposes the host driver through `/run/opengl-driver/lib`
and keeps CuPy's compiled kernel cache under `.cache/cupy/`. GPU processes
need access to `/dev/nvidia*`; a sandbox that hides those devices can make
`nvidia-smi` report a driver communication failure even with a healthy host
driver. Check `nvidia-smi` outside that sandbox before diagnosing the driver.

Keep `flake.lock` and `uv.lock` with the project. `.venv/` and `.cache/` are
local, ignored artifacts.

### AMD GPUs on NixOS (experimental ROCm)

Use the separate ROCm shell and AMD extra on x86_64 Linux:

```bash
nix develop path:.#rocm
uv sync --locked --extra test --extra rocm
uv run --locked --extra test --extra rocm python -c \
  'import cupy; cupy.show_config(); assert cupy.cuda.runtime.is_hip'
GARNET_FUSED=0 uv run --locked --extra test --extra rocm python examples/bare_reactor.py 16 gpu
uv run --locked --extra test --extra rocm python examples/bare_reactor.py 16 gpu
uv run --locked --extra test --extra rocm python examples/validate_rocm.py --sizes 8 12
```

`device="gpu"` requires a GPU and never accepts a CPU fallback. `device="rocm"`
(or `"hip"`) additionally requires an AMD build of CuPy; `"cuda"` requires a
CUDA build. CuPy's ROCm build intentionally uses the `cupy.cuda` namespace.
Device labels identify `rocm (cupy)` or `cuda (cupy)`.

The shell supplies ROCm 7.2.3 from the existing pinned Nixpkgs revision and a
merged `ROCM_HOME` for CuPy header discovery. The `rocm` extra pins the Python
3.13-compatible `cupy-rocm-7-0` wheel. ROCm and CUDA extras conflict because
they install the same `cupy` module; keep your selected extra in every `uv run`
command. The default CPU/NVIDIA shell remains available. ROCm provisioning can
download a large toolkit closure on first use.

The host must expose `/dev/kfd` and `/dev/dri/renderD*` with read/write access.
Check `rocminfo` in the same execution environment as Python. A sandbox hiding
these devices can report missing devices even when host ROCm works. No system
configuration changes are required by the project shell. Do not set
`HSA_OVERRIDE_GFX_VERSION` to impersonate a different GPU.

See [AMD validation and limitations](docs/amd_rocm.md) for halo's measured
results, explicit tolerances, graph fallback, sparse limitations and benchmark
methodology. Raw local reports default to ignored `results/`; selected reports
for review belong under `benchmark-results/`. The validation command exits
unsuccessfully unless actual AMD arrays and the required numerical checks pass.
