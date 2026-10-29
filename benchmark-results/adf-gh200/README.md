# Cartesian ADF on GH200: verification and performance campaign

Status: **prepared, not yet run.** No GPU numbers below have been measured.
This covers stage 3 of the [ADF plan](../../docs/assembly_discontinuity_factors_plan.md)
(device-resident solve, measured) and the GPU part of the
[C5G7 colorset study](../../docs/c5g7_colorset_adf.md).

## Submitting

On a Merlin login node:

```bash
bash ~/GARNET/slurm/stage_and_submit_adf_gh.sh
```

This rsyncs the working tree to `/data/scratch/shared/poncet_m/garnet-adf-stage/repo`.
It writes `PROVENANCE.txt` there (HEAD plus uncommitted diff), then submits four
`gh-interactive` jobs. `verify` runs first; the others start only after it
succeeds (`afterok`). Logs go to
`/data/scratch/shared/poncet_m/garnet-adf/garnet-adf-<job>.out`, and results to
`/data/scratch/shared/poncet_m/garnet-adf/results/<job>-<stage>/`.

Useful overrides:

```bash
GARNET_ADF_STAGES="verify" bash slurm/stage_and_submit_adf_gh.sh          # tests only
GARNET_ADF_PARTITION=gh-hourly bash slurm/stage_and_submit_adf_gh.sh      # other partition
GARNET_ADF_TIME_BERNAL=01:30:00 bash slurm/stage_and_submit_adf_gh.sh     # longer limit
GARNET_ADF_NSYS=1 bash slurm/stage_and_submit_adf_gh.sh                   # + nsys transfer profile
SBATCH=echo bash slurm/stage_and_submit_adf_gh.sh                         # dry run
```

The jobs use the GH200 environment
`/data/scratch/shared/poncet_m/garnet-gh-py313-v1`: standalone CPython 3.13.7
(aarch64) with NumPy 2.3.2, SciPy 1.16.3, CuPy 13.6.0, pytest 8.4.2 and the
CUDA 12.6 runtime from `nvidia-*-cu12` wheels. It needs no modules; build it on
a login node with `slurm/setup_gh_environment.sh` and check it on a GPU with
`sbatch slurm/probe_gh_environment.sh`. The source defaults to `~/GARNET`
(override with `GARNET_SRC_ROOT`). Every
stage first compiles the directed-weight kernel and checks it against the
generic stencil, so a broken CUDA stack fails within a minute.

## Stages and acceptance gates

| stage | default limit | runs | gate |
|---|---|---|---|
| `verify` | 30 min | pytest: `test_cartesian_adf` (including 12 parametrized fused-kernel cases and GPU vs CPU solves), `test_colorset_adf`, slab and Dall'Osso references, `test_fused_kernels`, `test_linalg_krylov`, `test_quasistatic`, `test_adjoint`, with `-o addopts=` | all pass; the GPU tests must **run**, not skip (check the `-ra` summary) |
| `bernal` | 45 min | `tools/assess_adf_gpu.py bernal --refinements 1 2 4 6 8 10 --devices gpu cpu --cpu-max-refinement 4 --repeats 3` | unity ADF = ADF off to 2e-8; GPU = CPU k to 2e-8 at r ≤ 4; every configuration converged, with residual < 1e-7 |
| `contrast` | 30 min | `tools/assess_adf_gpu.py contrast --refinement 6 --device gpu --repeats 3`: factor exponents 0, 0.5, 1, 2, 4 and two random fields (ratio ≤ 4) × {GMRES, BiCGStab} × Neumann degree 0–3 | record failures; identical k across solver/preconditioner for each factor set (|Δk| ≤ 2e-8) |
| `colorset` | 45 min | `examples/c5g7_colorset_adf.py --cells-per-pin 8 16 --nodes 1 2 17 --device gpu` | reference XS + DF at N = 17 reproduces k_ref to < 0.05 pcm in every layout; library-ADF conclusions compared with the CPU s = 2, 4 results |

## Results to fill in

Bernal (provisional map), best of 3, tol_k = 1e-9, tol_source = 1e-8. Δpcm is
against VALKIN 1.040374, a nodal reference, not truth.

| config | r | cells | device | k_eff | Δpcm | outers | inners | setup [s] | t [s] | GPU speedup |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| _pending job_ | | | | | | | | | | |

Factor contrast (r = 6, GPU):

| factors | solver | degree | converged | outers | inners | t [s] |
|---|---|---:|---|---:|---:|---:|
| _pending job_ | | | | | | |

Factor contrast, CPU reference (28 October 2026, Merlin7 `daily`, `cn002`,
x86_64, NumPy 2.3.5, one thread; `contrast-r6-cpu.json`). r = 6 is 261,360
cells, 252,720 active. Table^p raises the Table 1 factors to the power p; the
largest Table 1 face ratio is 1.78, so table^4 reaches about 10. Inners are
for Neumann degree 0 / 3. Best of 3.

| factors | max \|Δk\| over solver × degree | GMRES inners | BiCGStab inners | outers | t [s] |
|---|---:|---:|---:|---:|---:|
| table^0 | 2.4e-9 | 776 / 287 | 989 / 318 | 34–39 | 3.5–5.0 |
| table^0.5 | 5.2e-9 | 725 / 301 | 918 / 323 | 35–37 | 3.5–4.7 |
| table^1 | 3.4e-9 | 756 / 292 | 975 / 323 | 35–39 | 3.8–5.1 |
| table^2 | 2.5e-9 | 729 / 315 | 896 / 337 | 35–38 | 3.7–5.1 |
| table^4 | 2.9e-9 | 700 / 303 | 876 / 334 | 34–37 | 3.5–5.0 |
| random ≤ 4, seed 1 | 2.9e-9 | 1206 / 475 | 1524 / 528 | 53–57 | 5.6–8.0 |
| random ≤ 4, seed 2 | 3.4e-9 | 1264 / 488 | 1534 / 541 | 54–59 | 6.1–8.5 |

All 56 configurations converged with relative eigen-residual ≤ 8.3e-9, and
the k gate (|Δk| ≤ 2e-8) passes for every factor set. Normalized assembly
fission production agrees across configurations to ≤ 1.2e-6. Inner iterations
do not grow with factor magnitude up to table^4. The spatially random fields
cost about 1.6× the inners and 1.5× the outers; the outer increase is in the
power iteration, not the linear solver. On the CPU, degree 3 cuts inners about
2.6× but not wall time. Degree 1 is fastest, and BiCGStab is 3–13% faster than
GMRES at each degree (summed over all factor sets: BiCGStab 29.7, 33.7, 39.9 and 33.9 s for
degrees 1, 0, 2 and 3; GMRES 31.5 to 41.3 s). Repeat the solver ranking on the
GPU, where per-iteration synchronization is more expensive.

Colorset (GPU): use the tables of `docs/c5g7_colorset_adf.md`, adding s = 8 and 16.
The CPU s = 8 and 16 results are already in that document's 28 October update;
the GPU stage now checks device agreement and timing.

Host transfers (optional `GARNET_ADF_NSYS=1`): from `nsys-stats.txt`, report
the H2D/D2H counts and bytes inside the NVTX solve range. Scalar D2H copies
are expected: GMRES synchronizes each Arnoldi dot product and norm. Any
array-sized transfer per iteration is a defect.

## Files

- `colorset-cpu-s2-s4.json`: the CPU colorset baseline at s = 2 and 4
  (`lclrs81`, x86_64, one thread), from `examples/c5g7_colorset_adf.py
  --cells-per-pin 2 4 --nodes 1 2 17 --device cpu`.
- `colorset-cpu-s8-s16.json`: the same at s = 8 and 16 (Merlin7 `cn018`, one
  thread), `--cells-per-pin 8 16 --nodes 1 2 17 --device cpu`.
- `contrast-r6-cpu.json`: the CPU factor-contrast study (Merlin7 `cn002`,
  one thread), `tools/assess_adf_gpu.py contrast --refinement 6 --device cpu
  --repeats 3`.
