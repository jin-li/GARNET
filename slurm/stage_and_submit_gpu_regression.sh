#!/usr/bin/env bash
# Stage an independent GARNET snapshot in shared scratch and submit the complete
# GPU regression suite. The separate stage avoids mutating queued benchmarks.

set -euo pipefail

src_root="/afs/psi.ch/project/stars/workspace/RND/SB-RND-ACT-011-24/WP2/PM41/GARNET"
stage_root="${GARNET_REGRESSION_STAGE:-/data/scratch/shared/poncet_m/garnet-regression-stage}"
logdir="${GARNET_REGRESSION_LOG_DIR:-/data/scratch/shared/poncet_m/garnet-regression-logs}"

mkdir -p "$stage_root" "$logdir"
rsync -a --delete \
    --exclude '.git' \
    --exclude '.agents' \
    --exclude '.codex' \
    --exclude 'results' \
    "$src_root/" "$stage_root/repo/"

export GARNET_REPO="$stage_root/repo"
export GARNET_PYTHON_BIN="${GARNET_PYTHON_BIN:-/opt/psi/conda-envs/x86_64/ra-standard_py312/bin/python}"
export GARNET_TEST_DEPS="${GARNET_TEST_DEPS:-/data/scratch/shared/poncet_m/garnet-test-deps/pytest8}"
export GARNET_REGRESSION_LOG_DIR="$logdir"
sbatch "$stage_root/repo/slurm/run_garnet_gpu_regression.sh"
