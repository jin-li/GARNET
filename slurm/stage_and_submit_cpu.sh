#!/usr/bin/env bash
#
# Stage the GARNET checkout into shared scratch and submit the CPU benchmark
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
export GARNET_PYTHON_BIN="${GARNET_PYTHON_BIN:-/opt/psi/conda-envs/x86_64/ra-standard_py312/bin/python}"
sbatch "$stage_root/repo/slurm/run_garnet_cpu.sh"
