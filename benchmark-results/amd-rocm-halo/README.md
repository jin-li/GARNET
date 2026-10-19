# Halo AMD validation — 2026-10-05

Actual Radeon 8060S (`gfx1151`) execution through CuPy HIP, with
`device="gpu"`; CPU fallback was not accepted. All required comparisons and
the tested graph, sparse DSA and event-profiling paths passed.

Environment: Nixpkgs `b6c8664de9b6cc07fe5666a29f91884ba81197c4`, ROCm 7.2.3,
CuPy ROCm 14.2.0, Python 3.13.15, NumPy 2.5.3, SciPy 1.18.1, Linux 7.2.8.
The Ryzen AI Max+ 395 is an APU with shared system memory. GPU allocation
pool sizes do not represent dedicated VRAM.

From the repository root:

```sh
nix develop path:.#rocm
uv sync --locked --extra test --extra rocm
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Use an unused project-local cache for an initial compilation measurement.
export CUPY_CACHE_DIR="$PWD/.cache/cupy-rocm-halo-cold-20261005"
uv run --locked --extra test --extra rocm python examples/validate_rocm.py \
  --sizes 8 12 --cube-sizes 32 64 --repeats 3 \
  --output results/rocm-halo/validation-cold.json
```

The reviewed [validation.json](validation.json) was copied from that report.
It contains inputs, explicit tolerances, CPU/AMD numerical differences for all
initial/warm runs, hardware properties, software versions, synchronized wall
times, and graph/sparse/profiling probes. Initial runs are separate from three
fresh warmed runs. Later workloads reuse compiled kernels; only the first use
of each specialization includes compilation. First-use time includes other
runtime overhead and is not compiler-only time.

Validator SHA-256:
`bc29352aac07e07465db4e6e1cc31d7bcb1a6123a4d7df18920fb2940f4a290c`.

Supporting evidence: [CuPy configuration](cupy-config.txt),
[GPU enumeration](rocminfo.txt), and [kernel](kernel.txt).
CuPy's CUDA-named runtime APIs report HIP build version `70051831` and
runtime/driver `70253211`; the configuration confirms `gfx1151`.

| Workload | CPU warm median (s) | AMD unfused (s) | AMD fused (s) |
|---|---:|---:|---:|
| 8×8 diffusion | 0.00274 | 0.0290 | 0.0212 |
| 8×8 SP3 | 0.00418 | 0.0411 | 0.0238 |
| 8×8 transient, three steps | 0.00872 | 0.0839 | 0.0528 |
| Internal HP-MR thermal | 0.0366 | 0.2509 | 0.1846 |
| 12×12 diffusion | 0.00411 | 0.0405 | 0.0267 |
| 12×12 SP3 | 0.00621 | 0.0577 | 0.0302 |
| 12×12 transient, three steps | 0.0135 | 0.1248 | 0.0731 |
| 32³ diffusion | 0.0822 | 0.0912 | 0.0579 |
| 64³ diffusion | 1.3975 | 0.7371 | 0.6334 |

The 64³ fused AMD run was 2.21× faster than CPU. Small workloads were slower
on AMD. Initial 8×8 diffusion times were 3.56 s unfused and 0.556 s fused,
demonstrating why compiler/runtime startup must be kept separate.

Maximum errors across all initial and warmed AMD runs:

| Quantity | Observed maximum | Gate |
|---|---:|---:|
| Absolute k difference | 1.28e-14 | 2e-7 |
| Normalized flux L2 difference | 6.23e-12 | 2e-5 |
| Relative power absolute difference | 1.65e-12 | 2e-5 |
| Temperature absolute difference (K) | 1.14e-12 | 2e-3 |

PCG captured one graph and replayed four blocks. Unfused requests and an
injected stream-construction failure ran direct device iterations, with
relative residual 3.06e-12. TriSN levels sweeps passed with graphs off/on and
graphs were active when requested. The public TriSN flux is host-resident by
API; the internal sweep was confirmed to return CuPy arrays. CSR matvec, both
SN/TriSN DSA factory host/device residual probes and coupled event timing
passed.

These are numerical backend-equivalence checks on synthetic diffusion/SP3
problems, a short transient, and coarse two-group placeholder HP-MR coupling.
They do not validate full reactor physics, arbitrary sparse APIs, CMFD
multigrid, long transients, multi-GPU workloads or external preCICE. SN row
sweeps and TriSN LU remain CPU paths; periodic TriSN levels are unsupported.
NVIDIA hardware was not retested on halo. See the [setup and limits](../../docs/amd_rocm.md).
