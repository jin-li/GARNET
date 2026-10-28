#!/usr/bin/env bash
# Stage an independent GARNET snapshot and submit the full speed benchmark to
# an x86_64 A100 node. Command-line sbatch options override the GH defaults in
# the shared GPU runner without mutating queued job snapshots.

set -euo pipefail

src_root="${GARNET_SRC_ROOT:-$HOME/GARNET}"
stage_root="${GARNET_A100_STAGE:-/data/scratch/shared/poncet_m/garnet-a100-stage}"
logdir="${GARNET_A100_LOG_DIR:-/data/scratch/shared/poncet_m/garnet-a100-logs}"

mkdir -p "$stage_root" "$logdir"
rsync -a --delete \
    --exclude '.git' \
    --exclude '.agents' \
    --exclude '.codex' \
    --exclude 'results' \
    "$src_root/" "$stage_root/repo/"

export GARNET_REPO="$stage_root/repo"
export GARNET_PYTHON_BIN="${GARNET_PYTHON_BIN:-/opt/psi/conda-envs/x86_64/ra-standard_py312/bin/python}"
export GARNET_DEBUG_DIR="$logdir"

sbatch \
    --cluster=gmerlin7 \
    --partition=a100-hourly \
    --job-name=garnet-a100 \
    --chdir="$logdir" \
    --output="$logdir/garnet-a100-%j.out" \
    --error="$logdir/garnet-a100-%j.err" \
    "$stage_root/repo/slurm/run_garnet_gh.sh"
