#!/usr/bin/env bash
#
# Stage the GARNET checkout into shared scratch and submit the GH benchmark
# from there. This avoids running batch jobs directly from AFS.

set -euo pipefail

src_root="/afs/psi.ch/project/stars/workspace/RND/SB-RND-ACT-011-24/WP2/PM41/GARNET"
stage_root="/data/scratch/shared/poncet_m/garnet-stage"

mkdir -p "$stage_root"
rsync -a --delete \
    --exclude '.git' \
    --exclude '.agents' \
    --exclude '.codex' \
    --exclude 'slurm/garnet-*' \
    "$src_root/" "$stage_root/repo/"

export GARNET_REPO="$stage_root/repo"
export GARNET_PYTHON_BIN="${GARNET_PYTHON_BIN:-/data/scratch/shared/poncet_m/garnet-gh-py313-v1/bin/python}"
sbatch_args=()
if [[ -n "${GARNET_SLURM_DEPENDENCY:-}" ]]; then
    sbatch_args+=("--dependency=$GARNET_SLURM_DEPENDENCY")
fi
sbatch "${sbatch_args[@]}" "$stage_root/repo/slurm/run_garnet_gh.sh"
