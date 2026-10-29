#!/usr/bin/env bash
#
# Submit a single-node GARNET run to a Merlin7 GPU partition.
#
# Usage:
#   sbatch slurm/run_garnet_gh.sh
#
# Optional overrides are passed as environment variables:
#   GARNET_EXAMPLE   Python entrypoint to run
#   GARNET_ARGS      Arguments for the example
#   GARNET_DEVICE    Device string passed to GARNET (default: gpu)
#
# Default workload:
#   examples/speed_benchmark.py 32 64 96 128

#SBATCH --job-name=garnet-gh
#SBATCH --cluster=gmerlin7
#SBATCH --partition=gh-hourly
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:45:00
#SBATCH --mem=46G
#SBATCH --chdir=/data/scratch/shared/poncet_m/garnet-slurm-debug
#SBATCH --output=/data/scratch/shared/poncet_m/garnet-slurm-debug/garnet-gh-%j.out
#SBATCH --error=/data/scratch/shared/poncet_m/garnet-slurm-debug/garnet-gh-%j.err

set -euo pipefail

if [[ "$(uname -m)" == "aarch64" ]]; then
    default_python="/data/scratch/shared/poncet_m/garnet-gh-py313-v1/bin/python"
    # The CUDA 12.6 runtime libraries come from the environment's nvidia-* wheels.
    for lib in "$(dirname "$(dirname "${GARNET_PYTHON_BIN:-$default_python}")")"/lib/python3.13/site-packages/nvidia/*/lib; do
        LD_LIBRARY_PATH="$lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    done
    export LD_LIBRARY_PATH
else
    default_python="/opt/psi/conda-envs/x86_64/ra-standard_py312/bin/python"
fi
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
logfile="$logdir/garnet-gh-${SLURM_JOB_ID:-manual}.log"
touch "$logfile"
exec >>"$logfile" 2>&1
cd "$logdir"
printf '=== GARNET GPU speed benchmark started at %s on %s ===\n' "$(date -Is)" "$(hostname)"

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

if ! "$python_bin" - <<'PY'
import cupy
import numpy
import scipy
from cupy.cuda import runtime

from garnet import TransientSolver
from garnet.multigroup import MultigroupStepOperator

device_count = runtime.getDeviceCount()
if device_count < 1:
    raise RuntimeError("CuPy cannot see an allocated CUDA device")

props = runtime.getDeviceProperties(0)
name = props["name"]
if isinstance(name, bytes):
    name = name.decode()
print(f"python preflight: numpy={numpy.__version__} scipy={scipy.__version__} "
      f"cupy={cupy.__version__} cuda_devices={device_count} device0={name}")
print(f"transient preflight: {TransientSolver.__name__}, "
      f"{MultigroupStepOperator.__name__}")
PY
then
    cat >&2 <<EOF
error: the selected Python failed the NumPy/SciPy/CuPy/CUDA/GARNET preflight
selected interpreter: $python_bin
Set GARNET_PYTHON_BIN to a CuPy-enabled environment, for example one with
cuda-aware cupy installed alongside numpy and scipy.
EOF
    exit 1
fi

"$python_bin" -u "$example" $args
