#!/usr/bin/env bash
# Validate GARNET's aarch64 Python/CUDA environment on a GH200 node. The
# environment is built on a login node by slurm/setup_gh_environment.sh.

#SBATCH --job-name=garnet-gh-probe
#SBATCH --cluster=gmerlin7
#SBATCH --partition=gh-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --chdir=/data/scratch/shared/poncet_m/garnet-slurm-debug
#SBATCH --output=/data/scratch/shared/poncet_m/garnet-slurm-debug/garnet-gh-probe-%j.log
#SBATCH --error=/data/scratch/shared/poncet_m/garnet-slurm-debug/garnet-gh-probe-%j.log

set -euo pipefail

python_bin="${GARNET_PYTHON_BIN:-/data/scratch/shared/poncet_m/garnet-gh-py313-v1/bin/python}"

printf '=== GH200 environment probe: %s ===\n' "$(date -Is)"
printf 'host=%s arch=%s kernel=%s\n' "$(hostname)" "$(uname -m)" "$(uname -r)"
printf 'slurm_job=%s cuda_visible_devices=%s python=%s\n' \
    "${SLURM_JOB_ID:-unknown}" "${CUDA_VISIBLE_DEVICES:-unset}" "$python_bin"
nvidia-smi || true

# The CUDA 12.6 runtime libraries come from the environment's nvidia-* wheels.
for lib in "$(dirname "$(dirname "$python_bin")")"/lib/python3.13/site-packages/nvidia/*/lib; do
    LD_LIBRARY_PATH="$lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
export LD_LIBRARY_PATH

"$python_bin" - <<'PY'
import platform

import cupy
import cupyx.scipy.sparse as cusparse
import numpy
import pytest
import scipy
from cupy.cuda import nvrtc, runtime

x = cupy.arange(1_000_000, dtype=cupy.float64)
total = float(cupy.sum(x).get())
expected = 999_999 * 1_000_000 / 2
if total != expected:
    raise RuntimeError(f"CuPy reduction mismatch: {total} != {expected}")

# NVRTC compilation with CUDA headers, as GARNET's fused kernels need.
square = cupy.RawKernel(r'''
#include <cuda_fp16.h>
extern "C" __global__ void square(const double* x, double* y, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) y[i] = x[i] * x[i];
}''', "square")
y = cupy.empty_like(x)
square(((x.size + 255) // 256,), (256,), (x, y, cupy.int32(x.size)))
assert float(y[1000]) == 1000.0 ** 2

# cuSPARSE SpMV.
a = cusparse.identity(1000, format="csr", dtype=cupy.float64) * 2.0
assert float((a @ cupy.ones(1000)).sum()) == 2000.0

props = runtime.getDeviceProperties(0)
name = props["name"]
if isinstance(name, bytes):
    name = name.decode()
print(f"arch={platform.machine()} device={name} cuda_devices={runtime.getDeviceCount()}")
print(f"driver={runtime.driverGetVersion()} runtime={runtime.runtimeGetVersion()} "
      f"nvrtc={nvrtc.getVersion()}")
print(f"numpy={numpy.__version__} scipy={scipy.__version__} "
      f"cupy={cupy.__version__} pytest={pytest.__version__}")
print("probe: OK")
PY

printf '=== GH200 environment probe finished: %s ===\n' "$(date -Is)"
