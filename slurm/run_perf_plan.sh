#!/usr/bin/env bash
#
# One job of the GARNET vs FEMFFUSION performance report: run a plan file of
# tool invocations. Normally submitted by slurm/stage_and_submit_femffusion_cmp.sh,
# which sets the cluster, partition, resources and GARNET_PERF_PLAN.
#
# GARNET_PERF_PLAN is a file with one invocation per line:
#     <chain> <tag> <tool> <tool arguments>
# where <tool> is a script in tools/ (run with this job's Python) and the
# arguments may use @OUT@ (= <results>/<tag>.json) and @WORK@ (= a scratch
# directory <results>/work/<tag>). Lines of one chain run in order; different
# chains run at the same time, each a single-threaded process. Lines of the
# chain "after" run alone once every other chain has finished (the concurrency
# control). Each line writes <results>/<tag>.log; progress.log gets a DONE or
# FAIL line per run, and a failed run does not stop its chain.
#
# On an exclusive CPU node, chain i is pinned with taskset to core i*STRIDE
# (STRIDE = 16: one chain per NUMA domain of the AMD EPYC 7742 in NPS4 mode,
# so concurrent chains do not share a memory controller); "after" runs on
# core 0. FEMFFUSION, started by tools/run_femffusion.py, inherits the pin.
# Optional overrides: GARNET_REPO, GARNET_PYTHON_BIN, GARNET_PERF_RESULTS,
# GARNET_PERF_PIN (1/0; default 1 when the job has >= 16 CPUs), GARNET_PERF_PIN_STRIDE.
#SBATCH --job-name=garnet-femffusion-perf
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --chdir=/data/scratch/shared/poncet_m/garnet-femffusion-perf
#SBATCH --output=/data/scratch/shared/poncet_m/garnet-femffusion-perf/%x-%j.out
#SBATCH --error=/data/scratch/shared/poncet_m/garnet-femffusion-perf/%x-%j.out

set -uo pipefail

plan="${GARNET_PERF_PLAN:?set GARNET_PERF_PLAN to a plan file}"
name="$(basename "$plan" .plan)"
if [[ "$(uname -m)" == "aarch64" ]]; then
    python_bin="${GARNET_PYTHON_BIN:-/data/scratch/shared/poncet_m/garnet-gh-py313-v1/bin/python}"
    # The CUDA 12.6 runtime libraries come from the environment's nvidia-* wheels.
    for lib in "$(dirname "$(dirname "$python_bin")")"/lib/python3.13/site-packages/nvidia/*/lib; do
        LD_LIBRARY_PATH="$lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    done
    export LD_LIBRARY_PATH
else
    python_bin="${GARNET_PYTHON_BIN:-/opt/psi/conda-envs/x86_64/ra-standard_py312/bin/python}"
fi
if [[ ! -x "$python_bin" ]]; then
    echo "error: Python not found: $python_bin" >&2
    exit 1
fi
repo="${GARNET_REPO:-}"
if [[ -z "$repo" ]]; then
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo="$(cd "$here/.." && pwd)"
fi
results="${GARNET_PERF_RESULTS:-/data/scratch/shared/poncet_m/garnet-femffusion-perf/results}/${SLURM_JOB_ID:-manual}-$name"
mkdir -p "$results/work"
cp "$plan" "$results/"
[[ -f "$repo/PROVENANCE.txt" ]] && cp "$repo/PROVENANCE.txt" "$results/"
cd "$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

ncpu="${SLURM_CPUS_ON_NODE:-$(nproc)}"
pin="${GARNET_PERF_PIN:-$(( ncpu >= 16 ? 1 : 0 ))}"
stride="${GARNET_PERF_PIN_STRIDE:-16}"

printf '=== GARNET vs FEMFFUSION plan %s started at %s on %s ===\n' "$name" "$(date -Is)" "$(hostname)"
printf 'job=%s repo=%s python=%s results=%s pin=%s stride=%s cpus=%s\n' \
    "${SLURM_JOB_ID:-manual}" "$repo" "$python_bin" "$results" "$pin" "$stride" "$ncpu"
grep -m1 "model name" /proc/cpuinfo || true
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | grep -v "has failed" || true

run_chain() {
    local chain=$1 core=$2
    awk -v c="$chain" '$1 == c' "$plan" | while read -r _ tag tool args; do
        args="${args//@OUT@/$results/$tag.json}"
        args="${args//@WORK@/$results/work/$tag}"
        local cmd=("$python_bin" "tools/$tool")
        local prefix=()
        [[ "$pin" == 1 ]] && prefix=(taskset -c "$core")
        # shellcheck disable=SC2086  # args are whitespace-separated by design
        if "${prefix[@]}" "${cmd[@]}" $args > "$results/$tag.log" 2>&1; then
            echo "DONE $tag $(grep -h '^RESULT' "$results/$tag.log" | tail -1)" >> "$results/progress.log"
        else
            echo "FAIL $tag (see $tag.log)" >> "$results/progress.log"
        fi
    done
}

chains=$(awk '$1 != "after" && NF && $1 !~ /^#/ {print $1}' "$plan" | awk '!seen[$0]++')
i=0
for chain in $chains; do
    run_chain "$chain" $(( i * stride )) &
    i=$(( i + 1 ))
done
wait
run_chain after 0
printf '=== finished at %s: %s done, %s failed ===\n' "$(date -Is)" \
    "$(grep -c '^DONE' "$results/progress.log" 2>/dev/null || true)" \
    "$(grep -c '^FAIL' "$results/progress.log" 2>/dev/null || true)"
