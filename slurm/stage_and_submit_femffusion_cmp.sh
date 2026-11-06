#!/usr/bin/env bash
#
# Stage a GARNET snapshot and submit plan files of the GARNET vs FEMFFUSION
# performance report (one job per plan, run by slurm/run_perf_plan.sh). Run
# on a Merlin login node:
#
#   bash slurm/stage_and_submit_femffusion_cmp.sh cpu-hourly PLAN.plan [PLAN.plan ...]
#   bash slurm/stage_and_submit_femffusion_cmp.sh gpu  PLAN.plan [...]
#
# Targets:
#   cpu-hourly, cpu-daily, cpu
#        merlin7/hourly (1 h), daily (1 d) or general (7 d), one exclusive
#        AMD EPYC 7742 node per plan (chains pinned one per NUMA domain), for
#        FEMFFUSION and single-core GARNET runs. general has the lowest job
#        priority (factor 1 vs 500 daily and 1000 hourly) and its nodes are
#        shared with both, so it can stall for days: split plans into
#        sub-hour jobs for cpu-hourly and use cpu-daily for the long runs.
#        An exclusive job waits for a whole empty node: give it a short
#        GARNET_PERF_TIME so the backfill scheduler can start it in a gap.
#   cpu-daily-shared
#        merlin7/daily on part of a node shared with other jobs
#        (GARNET_PERF_CPUS cores, one per chain, default 4 -- the runner asks
#        for --hint=nomultithread; GARNET_PERF_MEM, default 64G),
#        unpinned. Only for runs whose accuracy is used and not their time
#        (ffref- tags).
#   gpu  gmerlin7/gh-daily, one GH200 per plan, one chain.
#   gpu-hourly, a100: as gpu on gh-hourly (1 h) or a100-daily.
#
# Every submission gets its own snapshot (<stage_root>/<timestamp>/repo), so a
# later staging never changes the code under a running job. PROVENANCE.txt
# records the GARNET commit, git status and diff stat, and the FEMFFUSION
# commit and build. Overrides: GARNET_SRC_ROOT, FEMFFUSION_DIR, GARNET_PERF_TIME,
# GARNET_PERF_STAGE_ROOT, GARNET_PERF_WORK_ROOT, SBATCH (e.g. "echo").

set -euo pipefail

target="${1:?usage: $0 cpu-hourly|cpu-daily|cpu-daily-shared|cpu|gpu|gpu-hourly|a100 PLAN...}"
shift
[[ $# -gt 0 ]] || { echo "no plan files given" >&2; exit 1; }
src_root="${GARNET_SRC_ROOT:-$HOME/GARNET}"
ff_root="${FEMFFUSION_DIR:-$HOME/FEMFFUSION}"
stage_root="${GARNET_PERF_STAGE_ROOT:-/data/scratch/shared/poncet_m/garnet-femffusion-stage}"
work_root="${GARNET_PERF_WORK_ROOT:-/data/scratch/shared/poncet_m/garnet-femffusion-perf}"
sbatch_cmd="${SBATCH:-sbatch}"

case "$target" in
    cpu) opts=(--clusters=merlin7 --partition=general --exclusive --mem=0
               --constraint=AMD_EPYC_7742 "--time=${GARNET_PERF_TIME:-3-00:00:00}") ;;
    cpu-daily) opts=(--clusters=merlin7 --partition=daily --exclusive --mem=0
                     --constraint=AMD_EPYC_7742 "--time=${GARNET_PERF_TIME:-1-00:00:00}") ;;
    cpu-daily-shared) opts=(--clusters=merlin7 --partition=daily --ntasks=1
                            "--cpus-per-task=${GARNET_PERF_CPUS:-4}" "--mem=${GARNET_PERF_MEM:-64G}"
                            --constraint=AMD_EPYC_7742 "--time=${GARNET_PERF_TIME:-1-00:00:00}")
                      export GARNET_PERF_PIN=0 ;;
    cpu-hourly) opts=(--clusters=merlin7 --partition=hourly --exclusive --mem=0
                      --constraint=AMD_EPYC_7742 "--time=${GARNET_PERF_TIME:-00:59:00}") ;;
    gpu) opts=(--clusters=gmerlin7 --partition=gh-daily --gres=gpu:1 --cpus-per-task=1
               --mem=46G "--time=${GARNET_PERF_TIME:-23:00:00}") ;;
    gpu-hourly) opts=(--clusters=gmerlin7 --partition=gh-hourly --gres=gpu:1
                      --cpus-per-task=1 --mem=46G "--time=${GARNET_PERF_TIME:-00:59:00}") ;;
    a100) opts=(--clusters=gmerlin7 --partition=a100-daily --gres=gpu:1 --cpus-per-task=1
                --mem=46G "--time=${GARNET_PERF_TIME:-23:00:00}") ;;
    *) echo "unknown target $target" >&2; exit 1 ;;
esac

stamp="$(date +%Y%m%dT%H%M%S)"
stage="$stage_root/$stamp/repo"
mkdir -p "$stage" "$work_root/results" "$work_root/plans"
rsync -a --exclude '.git' --exclude '.agents' --exclude '.codex' \
    --exclude '__pycache__' --exclude '.pytest_cache' --exclude 'results' \
    --exclude 'benchmark-results' "$src_root/" "$stage/"
commit="$(git -C "$src_root" rev-parse HEAD)"
{
    printf 'staged: %s by %s on %s\n' "$(date -Is)" "$(id -un)" "$(hostname)"
    printf 'source: %s\n' "$src_root"
    git -C "$src_root" log -1 --format='commit: %H %ad %s' --date=iso
    printf 'branch: %s\n' "$(git -C "$src_root" rev-parse --abbrev-ref HEAD)"
    printf '\n# git status --short\n'
    git -C "$src_root" status --short
    printf '\n# git diff --stat\n'
    git -C "$src_root" diff --stat
    printf '\n# FEMFFUSION\n'
    git -C "$ff_root" log -1 --format='commit: %H %ad %s' --date=iso
    git -C "$ff_root" status --short | grep -v '^??' || true
    cat "$ff_root/build-release/build-info.json" 2>/dev/null || echo "build-info: none"
} >"$stage/PROVENANCE.txt" 2>&1 || true

for plan in "$@"; do
    name="$(basename "$plan" .plan)"
    cp "$plan" "$work_root/plans/$stamp-$name.plan"
    GARNET_REPO="$stage" GARNET_COMMIT="$commit" FEMFFUSION_DIR="$ff_root" \
    GARNET_PERF_PLAN="$work_root/plans/$stamp-$name.plan" \
        $sbatch_cmd "${opts[@]}" --job-name="gvf-$name" --export=ALL \
        "$stage/slurm/run_perf_plan.sh"
done
