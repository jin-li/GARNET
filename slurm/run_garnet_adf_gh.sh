#!/usr/bin/env bash
#
# One stage of the Cartesian ADF GPU campaign on a Merlin7 GH200 node.
# Normally submitted by slurm/stage_and_submit_adf_gh.sh, which sets the stage,
# time limit and partition on the sbatch command line.
#
# GARNET_ADF_STAGE selects the work:
#   verify    ADF, kernel, Krylov, adjoint and quasi-static regression tests
#   bernal    tools/assess_adf_gpu.py bernal (ADF off/unity/on, CPU vs GPU)
#   contrast  tools/assess_adf_gpu.py contrast (factor contrast x preconditioner)
#   colorset  examples/c5g7_colorset_adf.py on the GPU
# Optional overrides: GARNET_REPO, GARNET_PYTHON_BIN, GARNET_ADF_RESULTS,
# GARNET_ADF_ARGS (replaces the stage's default arguments), GARNET_ADF_NSYS=1
# (bernal only: also profile one configuration with Nsight Systems).

#SBATCH --job-name=garnet-adf
#SBATCH --cluster=gmerlin7
#SBATCH --partition=gh-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:45:00
#SBATCH --mem=46G
#SBATCH --chdir=/data/scratch/shared/poncet_m/garnet-adf
#SBATCH --output=/data/scratch/shared/poncet_m/garnet-adf/garnet-adf-%j.out
#SBATCH --error=/data/scratch/shared/poncet_m/garnet-adf/garnet-adf-%j.out

set -euo pipefail

stage="${GARNET_ADF_STAGE:-verify}"
case "$stage" in
    verify|bernal|contrast|colorset) ;;
    *) echo "error: unknown GARNET_ADF_STAGE=$stage" >&2; exit 2 ;;
esac

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "error: this job expects a GH200 (aarch64) node, got $(uname -m)" >&2
    exit 1
fi
unset PMODULES_ENV
module purge
module use Spack unstable
module load gcc/12.3 cuda/12.6.0-wak5 python/3.13.5-xbg5
python_bin="${GARNET_PYTHON_BIN:-/data/scratch/shared/poncet_m/garnet-gh-py313-v1/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    echo "error: GH200 Python environment not found: $python_bin" >&2
    echo "build it with slurm/setup_gh_environment.sh" >&2
    exit 1
fi

repo="${GARNET_REPO:-}"
if [[ -z "$repo" ]]; then
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo="$(cd "$here/.." && pwd)"
fi
results="${GARNET_ADF_RESULTS:-/data/scratch/shared/poncet_m/garnet-adf/results}/${SLURM_JOB_ID:-manual}-$stage"
mkdir -p "$results"
cd "$repo"

export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

printf '=== GARNET ADF stage %s started at %s on %s ===\n' "$stage" "$(date -Is)" "$(hostname)"
printf 'job=%s partition=%s repo=%s python=%s results=%s\n' "${SLURM_JOB_ID:-manual}" \
    "${SLURM_JOB_PARTITION:-unknown}" "$repo" "$python_bin" "$results"
if [[ -f "$repo/PROVENANCE.txt" ]]; then cp "$repo/PROVENANCE.txt" "$results/"; fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true

# Fail fast: the CUDA stack works and the directed-weight kernel compiles and
# agrees with the generic stencil before any long work starts.
"$python_bin" - <<'PY'
import cupy, numpy as np, scipy
from cupy.cuda import runtime
from garnet import Grid, kernels
from garnet.stencil import GroupOperator
name = runtime.getDeviceProperties(0)["name"]
name = name.decode() if isinstance(name, bytes) else name
print(f"preflight: numpy={np.__version__} scipy={scipy.__version__} cupy={cupy.__version__} "
      f"devices={runtime.getDeviceCount()} device0={name}")
rng = np.random.default_rng(0)
shape = (9, 7, 5)
adf = []
for axis in range(3):
    fshape = list(shape); fshape[axis] -= 1
    adf.append((0.5 + rng.random(fshape), 0.5 + rng.random(fshape)))
op = GroupOperator(cupy, Grid(shape, (9., 7., 5.)), cupy.asarray(0.5 + rng.random(shape)),
                   cupy.asarray(0.1 + rng.random(shape)), adf=tuple(adf))
phi = cupy.asarray(rng.random(shape))
assert kernels.use_fused(cupy, "stencil"), "fused stencil kernels are disabled"
fused = cupy.asnumpy(op.apply(phi))
kernels.set_fused(False)
plain = cupy.asnumpy(op.apply(phi))
kernels.set_fused(True)
err = float(np.abs(fused - plain).max() / np.abs(plain).max())
print(f"preflight: fused ADF stencil relative difference {err:.2e}")
assert err < 1e-13
PY

status=0
case "$stage" in
    verify)
        read -r -a extra <<<"${GARNET_ADF_ARGS:-}"
        tests=(
            tests/verification/test_cartesian_adf.py
            tests/verification/test_colorset_adf.py
            tests/verification/test_adf_slab_reference.py
            tests/verification/test_dallosso_adf_reference.py
            tests/verification/test_fused_kernels.py
            tests/verification/test_linalg_krylov.py
            tests/verification/test_quasistatic.py
            tests/verification/test_adjoint.py
        )
        "$python_bin" -u -m pytest -ra -q -o addopts= -p no:cacheprovider --durations=20 \
            --junitxml="$results/verify.xml" "${extra[@]}" "${tests[@]}" || status=$?
        ;;
    bernal)
        args="${GARNET_ADF_ARGS:---refinements 1 2 4 6 8 10 --devices gpu cpu --cpu-max-refinement 4 --repeats 3}"
        # shellcheck disable=SC2086
        "$python_bin" -u tools/assess_adf_gpu.py bernal $args --output "$results/bernal.json" || status=$?
        if [[ "${GARNET_ADF_NSYS:-0}" == 1 && $status -eq 0 ]]; then
            if command -v nsys >/dev/null; then
                nsys profile -t cuda,nvtx --stats=true -o "$results/bernal-r8-on-gmres" \
                    "$python_bin" -u tools/assess_adf_gpu.py bernal --refinements 8 --devices gpu \
                    --configs on-gmres --repeats 1 >"$results/nsys-stats.txt" 2>&1 || status=$?
            else
                echo "warning: nsys not found; skipping the transfer profile"
            fi
        fi
        ;;
    contrast)
        args="${GARNET_ADF_ARGS:---refinement 6 --device gpu --repeats 3}"
        # shellcheck disable=SC2086
        "$python_bin" -u tools/assess_adf_gpu.py contrast $args --output "$results/contrast.json" || status=$?
        ;;
    colorset)
        args="${GARNET_ADF_ARGS:---cells-per-pin 8 16 --nodes 1 2 17 --device gpu}"
        # shellcheck disable=SC2086
        "$python_bin" -u examples/c5g7_colorset_adf.py $args --output "$results/colorset.json" || status=$?
        ;;
esac

printf '=== GARNET ADF stage %s finished at %s with status %d ===\n' "$stage" "$(date -Is)" "$status"
exit "$status"
