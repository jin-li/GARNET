#!/usr/bin/env bash
#
# Stage a GARNET snapshot for the ADF GPU campaign and submit it to GH200.
# Run on a Merlin login node:
#
#   bash slurm/stage_and_submit_adf_gh.sh
#
# The verify job runs first; bernal, contrast and colorset start only if it
# succeeds (afterok). The dedicated stage directory never touches snapshots
# used by other queued GARNET jobs.
#
# Overrides:
#   GARNET_ADF_STAGES     subset/order of stages (default: verify bernal contrast colorset)
#   GARNET_ADF_PARTITION  partition for every stage (default: gh-interactive)
#   GARNET_ADF_TIME_<STAGE>  time limit, e.g. GARNET_ADF_TIME_BERNAL=01:30:00
#   GARNET_ADF_NSYS=1     also profile host/device transfers in the bernal stage
#   GARNET_SLURM_DEPENDENCY  extra dependency for the first job
#   SBATCH                command used to submit (e.g. "echo" for a dry run)
#   GARNET_ADF_STAGE_ROOT, GARNET_ADF_WORK_ROOT  staging and results roots

set -euo pipefail

src_root="${GARNET_SRC_ROOT:-$HOME/GARNET}"
stage_root="${GARNET_ADF_STAGE_ROOT:-/data/scratch/shared/poncet_m/garnet-adf-stage}"
work_root="${GARNET_ADF_WORK_ROOT:-/data/scratch/shared/poncet_m/garnet-adf}"
partition="${GARNET_ADF_PARTITION:-gh-interactive}"
read -r -a stages <<<"${GARNET_ADF_STAGES:-verify bernal contrast colorset}"
sbatch_cmd="${SBATCH:-sbatch}"

declare -A default_time=([verify]=00:30:00 [bernal]=00:45:00 [contrast]=00:30:00 [colorset]=00:45:00)

mkdir -p "$stage_root" "$work_root/results"
rsync -a --delete \
    --exclude '.git' \
    --exclude '.agents' \
    --exclude '.codex' \
    --exclude '__pycache__' \
    --exclude '.pytest_cache' \
    --exclude 'results' \
    "$src_root/" "$stage_root/repo/"

# .git is not staged, so record exactly which source state the jobs ran.
{
    printf 'staged: %s by %s on %s\n' "$(date -Is)" "$(id -un)" "$(hostname)"
    printf 'source: %s\n' "$src_root"
    git -C "$src_root" log -1 --format='commit: %H %ad %s' --date=iso
    printf '\n# git status --short\n'
    git -C "$src_root" status --short
    printf '\n# git diff --stat HEAD\n'
    git -C "$src_root" diff --stat HEAD
} >"$stage_root/repo/PROVENANCE.txt" 2>&1 || true

export GARNET_REPO="$stage_root/repo"
export GARNET_PYTHON_BIN="${GARNET_PYTHON_BIN:-/data/scratch/shared/poncet_m/garnet-gh-py313-v1/bin/python}"
export GARNET_ADF_NSYS="${GARNET_ADF_NSYS:-0}"
export GARNET_ADF_RESULTS="$work_root/results"

first=""
for stage in "${stages[@]}"; do
    var="GARNET_ADF_TIME_${stage^^}"
    limit="${!var:-${default_time[$stage]:-00:45:00}}"
    args=(--parsable --partition="$partition" --time="$limit" --job-name="garnet-adf-$stage"
          --export="ALL,GARNET_ADF_STAGE=$stage")
    if [[ -z "$first" && -n "${GARNET_SLURM_DEPENDENCY:-}" ]]; then
        args+=(--dependency="$GARNET_SLURM_DEPENDENCY")
    elif [[ -n "$first" ]]; then
        args+=(--dependency="afterok:$first")
    fi
    id="$($sbatch_cmd "${args[@]}" "$GARNET_REPO/slurm/run_garnet_adf_gh.sh")"
    id="${id%%;*}"
    printf '%-9s job %s  (%s, %s)\n' "$stage" "$id" "$partition" "$limit"
    if [[ -z "$first" ]]; then first="$id"; fi
done
printf 'logs:    %s/garnet-adf-<job>.out\nresults: %s/results/<job>-<stage>/\n' "$work_root" "$work_root"
