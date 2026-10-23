# AMD ROCm on NixOS

GARNET uses CuPy for both NVIDIA CUDA and AMD HIP. AMD support is experimental:
the matrix-free physics code is shared, but library and runtime capabilities
must be checked on the actual GPU. `device="gpu"` raises if CuPy cannot access
a GPU. `device="auto"` may fall back to NumPy and is unsuitable for validation.

## Reproduce the environment

From the repository root on x86_64 Linux:

```sh
nix develop path:.#rocm
uv sync --locked --extra test --extra rocm
rocminfo
uv run --locked --extra test --extra rocm python -c \
  'import cupy; cupy.show_config(); assert cupy.cuda.runtime.is_hip; print(cupy.cuda.runtime.getDeviceProperties(0))'
GARNET_FUSED=0 uv run --locked --extra test --extra rocm python examples/bare_reactor.py 16 gpu
GARNET_FUSED=1 uv run --locked --extra test --extra rocm python examples/bare_reactor.py 16 gpu
uv run --locked --extra test --extra rocm python examples/validate_rocm.py \
  --sizes 8 12 --cube-sizes 32 64 --repeats 3 --output results/rocm-validation.json
```

Keep the `rocm` extra on subsequent commands. `uv sync` replaces the selected
GPU distribution when switching extras; ROCm and CUDA distributions cannot
coexist because each owns `cupy`. The default shell and existing `cuda12-ctk`
extra remain available for NVIDIA systems. `device="rocm"`/`"hip"` requires
an AMD CuPy build; `device="cuda"` requires an NVIDIA build.

The existing `flake.lock` pins Nixpkgs to
`b6c8664de9b6cc07fe5666a29f91884ba81197c4`. The separate `rocm` shell supplies
ROCm 7.2.3 with HIP/HIPRTC, BLAS, sparse, random, solver and FFT libraries,
plus compiler headers and device bitcode. A Nix store symlink tree provides
`ROCM_HOME`, and `LD_LIBRARY_PATH` exposes native libraries to the unpatched
manylinux wheel. Python remains Nix Python 3.13; `uv.lock` pins
`cupy-rocm-7-0==14.2.0`. No system-wide Python installation or NixOS rebuild is
part of this setup. The first shell realization downloads a large native
toolkit closure, including BLAS kernels for multiple architectures.

`.venv/`, `UV_CACHE_DIR=.cache/uv` and `CUPY_CACHE_DIR=.cache/cupy` stay local
and ignored. Generated JSON defaults to ignored `results/`; reviewed evidence
belongs in `benchmark-results/`. Nix owns the native dependency closure in its
store. Avoid copying native libraries into `.venv` or setting architecture
overrides such as `HSA_OVERRIDE_GFX_VERSION`.

## Host and version boundaries

Halo has a Ryzen AI Max+ 395 and Radeon 8060S (`gfx1151`, RDNA 3.5). Host
`rocminfo` enumerated the GPU with kernel dispatch and 40 compute units. This
is an APU: the GPU uses shared system memory. GPU-reported allocation pools or
`totalGlobalMem` must not be described as dedicated VRAM or the full amount of
available system RAM.

The Python process needs read/write access to `/dev/kfd` and
`/dev/dri/renderD*`. Sandboxed processes can see neither even when host ROCm
works; inspect `rocminfo` outside a device-hiding sandbox before changing
drivers or group membership. The project does not modify device permissions,
system configuration, or services.

AMD lists this GPU in the [ROCm 7.2 Ryzen Linux support matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html).
That is not certification of NixOS. The [CuPy 14.2 installation guide](https://docs.cupy.dev/en/v14.2.0/install.html#using-cupy-on-amd-gpu-experimental)
requires ROCm 7.x and describes experimental functionality. Its wheel section
still contains older packaging guidance; the [actual PyPI release files](https://pypi.org/project/cupy-rocm-7-0/14.2.0/#files)
provide Linux x86_64 CPython 3.13 wheels built for ROCm 7.0. Runtime execution,
not the package name or successful GPU enumeration, establishes compatibility
with the pinned 7.2.3 libraries.

## Validation and timing methodology

`examples/validate_rocm.py` requires `get_backend("gpu")`, HIP runtime identity,
and actual CuPy output arrays. It checks float64 CPU and AMD diffusion, SP3,
a three-step delayed-neutron transient with a fission perturbation, and internal
HP-MR thermal coupling. Each AMD case runs first with GARNET fusion disabled,
then enabled. CPU and AMD must converge and satisfy these tolerances:

| Quantity | Maximum difference |
|---|---:|
| k-effective | absolute `2e-7` |
| Flux after L2 normalization | relative L2 `2e-5` |
| Relative transient power | absolute `2e-5` |
| Coupled temperature | absolute `2e-3 K` |
| PCG graph-request solve | relative residual `<1e-7` |

The eigenpair stopping tolerances are `tol_k=1e-9` and `tol_source=1e-8`.
These are backend equivalence checks, not a mesh-convergence or reactor
validation study. HP-MR uses the built-in two-group placeholder materials and
a coarse polar mesh. The transient is a short functional test.

Wall time includes fresh solver construction and solve, with device
synchronization before and after each measurement. Identical CPU/GPU inputs,
initial guesses and tolerances are used. An initial run is reported separately
from at least three warmed runs, whose median is compared. Initial times
include any remaining compilation and runtime initialization; they are not a
measurement of compiler time alone. Disk CuPy caches can also warm an initial
run in a new process. Use a fresh project-local `CUPY_CACHE_DIR` for a deliberately
uncached experiment. Small grids can be much faster on CPU; no speedup is a
correctness requirement.

## Graphs and sparse transport

PCG graph requests retain a direct device iteration fallback. The workspace
records `graph_error`, captures and replays, allowing unsupported capture to
be distinguished from a graph actually executing. Graph capture requires fused
reductions, an `out=` operator and an allocation-free preconditioner. Disabled
fusion deliberately exercises the direct fallback. Stream creation failures
are handled along with capture failures.

TriSN levels sweeps also retain a device loop fallback and record
`graphs_active` and `_graph_error`. The validation script compares a small
vacuum levels problem with graphs off/on against CPU and checks the internal
sweep returns a CuPy array. Its public eigenpair flux is host-resident by API.
The LU engine and periodic levels boundaries have existing restrictions.

CuPy documents sparse matrices among its ROCm limitations. The script records
a CSR matvec probe and the real SN/TriSN sparse DSA factories with host and
device residuals separately. Successful diffusion/SP3 does not certify DSA,
CMFD multigrid, or arbitrary sparse functionality. CPU sparse LU remains
available; host LU CMFD bridges device arrays and is not a device sparse solve.
SN row sweeps and TriSN LU are CPU paths. External preCICE, large full-core
transport, multi-GPU execution and long coupled transients require separate
validation.

## Measured on halo, 2026-10-05

The [reviewed report](https://github.com/jin-li/GARNET/blob/main/benchmark-results/amd-rocm-halo/README.md) records
actual HIP execution on `gfx1151` with CuPy 14.2.0, ROCm 7.2.3, Python
3.13.15, NumPy 2.5.3, SciPy 1.18.1 and Linux 7.2.8. All nine required cases
passed with fusion off and on: the two square resolutions for diffusion,
SP3 and transient, internal thermal coupling, and two 3D diffusion resolutions.

Across initial and warmed runs, the largest CPU/AMD k difference was
`1.28e-14`, the normalized flux L2 difference `6.23e-12`, transient power
difference `1.65e-12`, and coupled temperature difference `1.14e-12 K`.
PCG captured once and replayed four times; deliberately unavailable stream
creation used direct device iterations with relative residual `3.06e-12`.
TriSN levels graphs also executed successfully. CSR matvec and both SN/TriSN
DSA factories passed host/device residual checks. These small sparse probes
establish support for the tested operations despite the broad limitation in
CuPy's documentation; they do not establish full sparse API coverage. Coupled
event timing and the NVTX wrapper worked in the tested region.

| Workload | CPU warmed median | AMD unfused | AMD fused |
|---|---:|---:|---:|
| 32³ heterogeneous diffusion | 0.0822 s | 0.0912 s | 0.0579 s |
| 64³ heterogeneous diffusion | 1.3975 s | 0.7371 s | 0.6334 s |
| 8×8 SP3 | 0.00418 s | 0.0411 s | 0.0238 s |
| Internal HP-MR thermal coupling | 0.0366 s | 0.2509 s | 0.1846 s |

The 64³ fused AMD solve was 2.21× faster than CPU in this run. Tiny workloads
remain launch-bound and slower on GPU. With a fresh CuPy disk cache, the first
8×8 unfused diffusion run took 3.56 s versus 0.0290 s warmed; its fused first
run took 0.556 s versus 0.0212 s warmed. Later cases share compiled kernels,
so their initial times are not independent cold compilation measurements.
Both CPU thread environment variables were set to one. These are single-host
measurements, not universal speedups.

CPU regression checks passed 81 tests with two GPU tests skipped in the
default shell, plus 90 transient/SP3/conduction/feedback/coupling tests. The
AMD shell passed 82 focused tests including both device DSA tests; one
CPU-without-CuPy-specific test was appropriately skipped. Existing NVIDIA
extras still resolve in a dry run, but NVIDIA hardware execution was not
repeated on halo. No new unsupported core path was found in this scope;
the untested and existing CPU-only paths listed above remain limits.
