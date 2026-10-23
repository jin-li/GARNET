#!/usr/bin/env bash
#
# Submit the full GARNET speed benchmark to a CPU partition.
#
# Usage:
#   sbatch slurm/run_garnet_cpu.sh
#
# Optional overrides are passed as environment variables:
#   GARNET_EXAMPLE   Python entrypoint to run
#   GARNET_ARGS      Arguments for the example
#
# Default workload:
#   examples/speed_benchmark.py 32 64 96 128

#SBATCH --job-name=garnet-cpu
#SBATCH --cluster=merlin7
#SBATCH --partition=hourly
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --mem=46G
#SBATCH --chdir=/data/scratch/shared/poncet_m/garnet-slurm-debug
#SBATCH --output=/data/scratch/shared/poncet_m/garnet-slurm-debug/garnet-cpu-%j.out
#SBATCH --error=/data/scratch/shared/poncet_m/garnet-slurm-debug/garnet-cpu-%j.err

set -euo pipefail

default_python="/opt/psi/conda-envs/x86_64/ra-standard_py312/bin/python"
python_bin="${GARNET_PYTHON_BIN:-$default_python}"
if [[ ! -x "$python_bin" ]]; then
    if [[ -n "${MODULESHOME:-}" ]]; then
        module load "${GARNET_PYTHON_MODULE:-Python/3.11.11}"
    fi
    python_bin="${PYTHON_BIN:-}"
fi
if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
    python_bin="$(command -v python3.11 || command -v python3.12 || command -v python3 || true)"
fi

repo="${GARNET_REPO:-}"
if [[ -z "$repo" ]]; then
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo="$(cd "$here/.." && pwd)"
fi

logdir="${GARNET_DEBUG_DIR:-/data/scratch/shared/poncet_m/garnet-slurm-debug}"
mkdir -p "$logdir"
logfile="$logdir/garnet-cpu-${SLURM_JOB_ID:-manual}.log"
touch "$logfile"
exec >>"$logfile" 2>&1
cd "$logdir"
printf '=== GARNET CPU speed benchmark started at %s on %s ===\n' "$(date -Is)" "$(hostname)"

export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

example="${GARNET_EXAMPLE:-$repo/examples/speed_benchmark.py}"
args="${GARNET_ARGS:-32 64 96 128}"

printf 'job=%s host=%s cwd=%s\n' "${SLURM_JOB_ID:-manual}" "$(hostname)" "$PWD"
printf 'example=%s\nargs=%s\n' "$example" "$args"

if [[ -z "$python_bin" ]]; then
    echo "error: no usable python interpreter found on PATH" >&2
    exit 127
fi

"$python_bin" -u "$example" $args
