#!/usr/bin/env bash
#
# Stage a GARNET snapshot and submit the GPU/CPU performance comparison of the
# Carreno et al. (2024) V&V report: SP1, SP3, SDP1 and SDP3 with the Marshak
# vacuum, on Brantley-Larsen (n x n) and pin-resolved C5G7 (s cells per pin
# side), on one GH200 and on one core of an exclusive merlin7 CPU node, on
# identical meshes. Run on a Merlin login node:
#
#   bash slurm/stage_and_submit_carreno_perf.sh
#
# GPU: one gh-daily job for Brantley-Larsen and one per method for C5G7, each
# run sequentially (best of 3 solves). CPU: one exclusive node per case; the
# four methods run side by side as single-threaded processes (a 128-core node
# leaves memory bandwidth to spare), best of 3 where a solve takes minutes and
# a single solve where it takes hours; the "after" chain re-times one case
# alone as the concurrency control.
#
# The "3d" plans extrude both problems in z over reflective layers (tools/
# assess_carreno2024.py --nz) to reach the cell counts of a 3-D core while the
# solution, and k, stay those of the 2-D problem: Brantley-Larsen n = 640 with
# 16, 64 and (GPU, CG methods) 256 layers; C5G7 s = 8 with 8, 64 and (GPU)
# 128 layers, and (GPU) s = 16 with 64 layers. The largest GPU cases stop at
# the GH200's 96 GiB. On the CPU,
# SDP3 stops at 32 layers, in a chain of its own, to fit the 5-day limit;
# gpu-3d-z32 times the same two cases on the GH200.
#
# Overrides: GARNET_PERF_WHICH (subset of "gpu-bl gpu-c5g7 cpu-bl cpu-c5g7
# gpu-bl3d gpu-c5g7-3d gpu-3d-z32 cpu-bl3d cpu-c5g7-3d"; default the four 2-D
# plans),
# GARNET_PERF_STAGE_ROOT, GARNET_PERF_WORK_ROOT, SBATCH (e.g. "echo").

set -euo pipefail

src_root="${GARNET_SRC_ROOT:-$HOME/GARNET}"
stage_root="${GARNET_PERF_STAGE_ROOT:-/data/scratch/shared/poncet_m/garnet-carreno-perf-stage}"
work_root="${GARNET_PERF_WORK_ROOT:-/data/scratch/shared/poncet_m/garnet-carreno-perf}"
read -r -a which <<<"${GARNET_PERF_WHICH:-gpu-bl gpu-c5g7 cpu-bl cpu-c5g7}"
sbatch_cmd="${SBATCH:-sbatch}"
methods="SP1 SP3 SDP1 SDP3"
c5="--tol-k 1e-9 --tol-source 1e-8 --no-lines"

mkdir -p "$stage_root" "$work_root/results" "$work_root/plans"
rsync -a --delete --exclude '.git' --exclude '.agents' --exclude '.codex' \
    --exclude '__pycache__' --exclude '.pytest_cache' --exclude 'results' \
    --exclude 'benchmark-results' "$src_root/" "$stage_root/repo/"
{
    printf 'staged: %s by %s on %s\n' "$(date -Is)" "$(id -un)" "$(hostname)"
    printf 'source: %s\n' "$src_root"
    git -C "$src_root" log -1 --format='commit: %H %ad %s' --date=iso
    printf '\n# git status --short\n'
    git -C "$src_root" status --short
} >"$stage_root/repo/PROVENANCE.txt" 2>&1 || true

plans="$work_root/plans"
# GPU, Brantley-Larsen: every method and mesh in one sequential chain.
{
    for m in $methods; do
        for n in 80 160 320 640 1280 2560; do
            echo "gpu bl-$m-n$n bl --methods $m --vacuum marshak --meshes $n --repeats 3 --device gpu"
        done
    done
} >"$plans/gpu-bl.plan"
# GPU, C5G7: one plan per method.
for m in $methods; do
    meshes="4 8 16 24"
    [[ $m == SP1 || $m == SP3 ]] && meshes="4 8 16 24 32 48"
    for s in $meshes; do
        echo "gpu c5g7-$m-s$s c5g7 --methods $m --vacuum marshak --cells-per-pin $s $c5 --repeats 3 --device gpu"
    done >"$plans/gpu-c5g7-$m.plan"
done
# CPU, Brantley-Larsen: one chain per method, side by side.
{
    for m in $methods; do
        for n in 80 160 320; do
            echo "$m bl-$m-n$n bl --methods $m --vacuum marshak --meshes $n --repeats 3"
        done
        if [[ $m == SP1 || $m == SP3 ]]; then
            echo "$m bl-$m-n640 bl --methods $m --vacuum marshak --meshes 640 --repeats 3"
            echo "$m bl-$m-n1280 bl --methods $m --vacuum marshak --meshes 1280"
            echo "$m bl-$m-n2560 bl --methods $m --vacuum marshak --meshes 2560"
        else
            echo "$m bl-$m-n640 bl --methods $m --vacuum marshak --meshes 640"
            echo "$m bl-$m-n1280 bl --methods $m --vacuum marshak --meshes 1280"
        fi
    done
    echo "after bl-SP3-n640-alone bl --methods SP3 --vacuum marshak --meshes 640 --repeats 3"
} >"$plans/cpu-bl.plan"
# CPU, C5G7: one chain per method, side by side.
{
    for m in $methods; do
        echo "$m c5g7-$m-s4 c5g7 --methods $m --vacuum marshak --cells-per-pin 4 $c5 --repeats 3"
        if [[ $m == SP1 || $m == SP3 ]]; then
            echo "$m c5g7-$m-s8 c5g7 --methods $m --vacuum marshak --cells-per-pin 8 $c5 --repeats 3"
            echo "$m c5g7-$m-s16 c5g7 --methods $m --vacuum marshak --cells-per-pin 16 $c5"
            echo "$m c5g7-$m-s24 c5g7 --methods $m --vacuum marshak --cells-per-pin 24 $c5"
        else
            echo "$m c5g7-$m-s8 c5g7 --methods $m --vacuum marshak --cells-per-pin 8 $c5"
            echo "$m c5g7-$m-s16 c5g7 --methods $m --vacuum marshak --cells-per-pin 16 $c5"
        fi
    done
    echo "after c5g7-SP3-s8-alone c5g7 --methods SP3 --vacuum marshak --cells-per-pin 8 $c5 --repeats 3"
} >"$plans/cpu-c5g7.plan"
# Extruded ("3-D") plans.
{
    for m in $methods; do
        echo "gpu bl-$m-n640-z16 bl --methods $m --vacuum marshak --meshes 640 --nz 16 --repeats 2 --device gpu"
        echo "gpu bl-$m-n640-z64 bl --methods $m --vacuum marshak --meshes 640 --nz 64 --device gpu"
    done
    for m in SP1 SP3; do
        echo "gpu bl-$m-n640-z256 bl --methods $m --vacuum marshak --meshes 640 --nz 256 --device gpu"
    done
} >"$plans/gpu-bl3d.plan"
for m in $methods; do
    {
        echo "gpu c5g7-$m-s8-z8 c5g7 --methods $m --vacuum marshak --cells-per-pin 8 --nz 8 $c5 --repeats 2 --device gpu"
        echo "gpu c5g7-$m-s8-z64 c5g7 --methods $m --vacuum marshak --cells-per-pin 8 --nz 64 $c5 --device gpu"
        echo "gpu c5g7-$m-s8-z128 c5g7 --methods $m --vacuum marshak --cells-per-pin 8 --nz 128 $c5 --device gpu"
        [[ $m != SDP3 ]] && echo "gpu c5g7-$m-s16-z64 c5g7 --methods $m --vacuum marshak --cells-per-pin 16 --nz 64 $c5 --device gpu"
    } >"$plans/gpu-c5g7-3d-$m.plan"
done
{
    echo "gpu bl-SDP3-n640-z32 bl --methods SDP3 --vacuum marshak --meshes 640 --nz 32 --device gpu"
    echo "gpu c5g7-SDP3-s8-z32 c5g7 --methods SDP3 --vacuum marshak --cells-per-pin 8 --nz 32 $c5 --device gpu"
} >"$plans/gpu-3d-z32.plan"
# CPU: SDP3 at 64 layers would take about 110-140 h, so it stops at 32 layers,
# run as its own chain beside the others.
{
    for m in $methods; do
        echo "$m bl-$m-n640-z16 bl --methods $m --vacuum marshak --meshes 640 --nz 16"
        [[ $m != SDP3 ]] && echo "$m bl-$m-n640-z64 bl --methods $m --vacuum marshak --meshes 640 --nz 64"
    done
    echo "SDP3-z32 bl-SDP3-n640-z32 bl --methods SDP3 --vacuum marshak --meshes 640 --nz 32"
    echo "after bl-SP1-n640-z16-alone bl --methods SP1 --vacuum marshak --meshes 640 --nz 16"
} >"$plans/cpu-bl3d.plan"
{
    for m in $methods; do
        echo "$m c5g7-$m-s8-z8 c5g7 --methods $m --vacuum marshak --cells-per-pin 8 --nz 8 $c5"
        [[ $m != SDP3 ]] && echo "$m c5g7-$m-s8-z64 c5g7 --methods $m --vacuum marshak --cells-per-pin 8 --nz 64 $c5"
    done
    echo "SDP3-z32 c5g7-SDP3-s8-z32 c5g7 --methods SDP3 --vacuum marshak --cells-per-pin 8 --nz 32 $c5"
    echo "after c5g7-SP1-s8-z8-alone c5g7 --methods SP1 --vacuum marshak --cells-per-pin 8 --nz 8 $c5"
} >"$plans/cpu-c5g7-3d.plan"

export GARNET_REPO="$stage_root/repo"
export GARNET_PERF_RESULTS="$work_root/results"
job="$GARNET_REPO/slurm/run_garnet_carreno_perf.sh"
gpu=(--parsable --clusters=gmerlin7 --partition=gh-daily --gres=gpu:1 --cpus-per-task=1
     --mem=46G --time=23:00:00)
cpu=(--parsable --clusters=merlin7 --partition=general --exclusive --mem=0
     --time=5-00:00:00)
submit() {
    local plan=$1; shift
    local id
    id="$($sbatch_cmd "$@" --job-name="garnet-perf-$(basename "$plan" .plan)" \
        --export="ALL,GARNET_PERF_PLAN=$plan" "$job")"
    printf '%-16s job %s\n' "$(basename "$plan" .plan)" "${id%%;*}"
}
for w in "${which[@]}"; do
    case $w in
        gpu-bl) submit "$plans/gpu-bl.plan" "${gpu[@]}" ;;
        gpu-c5g7) for m in $methods; do submit "$plans/gpu-c5g7-$m.plan" "${gpu[@]}"; done ;;
        cpu-bl) submit "$plans/cpu-bl.plan" "${cpu[@]}" ;;
        cpu-c5g7) submit "$plans/cpu-c5g7.plan" "${cpu[@]}" ;;
        gpu-bl3d) submit "$plans/gpu-bl3d.plan" "${gpu[@]}" ;;
        gpu-c5g7-3d) for m in $methods; do submit "$plans/gpu-c5g7-3d-$m.plan" "${gpu[@]}"; done ;;
        gpu-3d-z32) submit "$plans/gpu-3d-z32.plan" "${gpu[@]}" ;;
        cpu-bl3d) submit "$plans/cpu-bl3d.plan" "${cpu[@]}" ;;
        cpu-c5g7-3d) submit "$plans/cpu-c5g7-3d.plan" "${cpu[@]}" ;;
        *) echo "unknown: $w" >&2; exit 2 ;;
    esac
done
printf 'logs:    %s/garnet-carreno-perf-<job>.out\nresults: %s/results/<job>-<plan>/\n' "$work_root" "$work_root"
