#!/usr/bin/env bash
#
# One job of the GPU/CPU performance comparison of the Carreno et al. (2024)
# V&V report. Normally submitted by slurm/stage_and_submit_carreno_perf.sh,
# which sets the cluster, partition, resources and GARNET_PERF_PLAN.
#
# GARNET_PERF_PLAN is a file with one tools/assess_carreno2024.py invocation
# per line: "<chain> <tag> <tool arguments>". Lines of one chain run in order;
# different chains run at the same time, each as one single-threaded process.
# Lines of the chain "after" run alone once every other chain has finished
# (the concurrency control). Each line writes <results>/<tag>.json and
# <results>/<tag>.log; progress.log gets a DONE or FAIL line per run, and a
# failed run does not stop its chain.
# Optional overrides: GARNET_REPO, GARNET_PYTHON_BIN, GARNET_PERF_RESULTS.

#SBATCH --job-name=garnet-carreno-perf
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --chdir=/data/scratch/shared/poncet_m/garnet-carreno-perf
#SBATCH --output=/data/scratch/shared/poncet_m/garnet-carreno-perf/garnet-carreno-perf-%j.out
#SBATCH --error=/data/scratch/shared/poncet_m/garnet-carreno-perf/garnet-carreno-perf-%j.out

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
results="${GARNET_PERF_RESULTS:-/data/scratch/shared/poncet_m/garnet-carreno-perf/results}/${SLURM_JOB_ID:-manual}-$name"
mkdir -p "$results"
cp "$plan" "$results/"
cd "$repo"

export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

printf '=== GARNET Carreno performance plan %s started at %s on %s ===\n' "$name" "$(date -Is)" "$(hostname)"
printf 'job=%s repo=%s python=%s results=%s\n' "${SLURM_JOB_ID:-manual}" "$repo" "$python_bin" "$results"
grep -m1 "model name" /proc/cpuinfo || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

run_chain() {
    local chain=$1
    awk -v c="$chain" '$1 == c' "$plan" | while read -r _ tag args; do
        if "$python_bin" tools/assess_carreno2024.py $args --verbose \
                --output "$results/$tag.json" > "$results/$tag.log" 2>&1; then
            echo "DONE $tag $(grep -h '^RESULT' "$results/$tag.log" | tail -1)" >> "$results/progress.log"
        else
            echo "FAIL $tag (see $tag.log)" >> "$results/progress.log"
        fi
    done
}

chains=$(awk '$1 != "after" && NF {print $1}' "$plan" | sort -u)
for chain in $chains; do
    run_chain "$chain" &
done
wait
run_chain after
printf '=== finished at %s: %s done, %s failed ===\n' "$(date -Is)" \
    "$(grep -c '^DONE' "$results/progress.log" 2>/dev/null || true)" \
    "$(grep -c '^FAIL' "$results/progress.log" 2>/dev/null || true)"
