# Cartesian ADF on GH200: verification and performance campaign

Status: **prepared, not yet run.** No GPU numbers below have been measured.
This covers stage 3 of the [ADF plan](../../docs/assembly_discontinuity_factors_plan.md)
(device-resident solve, measured) and the GPU part of the
[C5G7 colorset study](../../docs/c5g7_colorset_adf.md).

## Submitting

On a Merlin login node:

```bash
bash /afs/psi.ch/project/stars/workspace/RND/SB-RND-ACT-011-24/WP2/PM41/GARNET/slurm/stage_and_submit_adf_gh.sh
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

The jobs use the existing GH200 environment
`/data/scratch/shared/poncet_m/garnet-gh-py313-v1` (NumPy 2.3.2, SciPy 1.16.3,
CuPy 13.6.0, pytest 8.4.2; built by `slurm/setup_gh_environment.sh`). Every
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

Colorset (GPU): use the tables of `docs/c5g7_colorset_adf.md`, adding s = 8 and 16.

Host transfers (optional `GARNET_ADF_NSYS=1`): from `nsys-stats.txt`, report
the H2D/D2H counts and bytes inside the NVTX solve range. Scalar D2H copies
are expected: GMRES synchronizes each Arnoldi dot product and norm. Any
array-sized transfer per iteration is a defect.

## Files

- `colorset-cpu-s2-s4.json`: the CPU colorset baseline at s = 2 and 4
  (`lclrs81`, x86_64, one thread), from `examples/c5g7_colorset_adf.py
  --cells-per-pin 2 4 --nodes 1 2 17 --device cpu`.
