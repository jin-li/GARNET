#!/bin/bash -l
#
# Build FEMFFUSION and its libraries (PETSc, SLEPc, deal.II) on a merlin7
# compute node, for the GARNET vs FEMFFUSION performance report.
#
# The versions are the ones FEMFFUSION's README documents (Ubuntu 24.04:
# PETSc 3.19, SLEPc 3.19, deal.II 9.5), built from source with -O3 and
# -march=znver2 so the comparison runs an optimized build on the AMD EPYC 7742
# nodes. The source tarballs must already be in $FEMFFUSION_DEPS/src (compute
# nodes are not assumed to reach the internet):
#   petsc-3.19.6.tar.gz, slepc-3.19.2.tar.gz, dealii-9.5.2.tar.gz
# Every stage is skipped when its install already exists, so a failed build
# can be resubmitted. FEMFFUSION is built out of source in
# $FEMFFUSION_SRC/build-release and checked with its own test suite (-t).
#
#   sbatch slurm/build_femffusion.sh
#
# Optional overrides: FEMFFUSION_DEPS, FEMFFUSION_SRC, BUILD_JOBS.
#SBATCH --job-name=femffusion-build
#SBATCH --clusters=merlin7
#SBATCH --partition=daily
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=/data/scratch/shared/poncet_m/femffusion-deps-v1/build-%j.out
#SBATCH --error=/data/scratch/shared/poncet_m/femffusion-deps-v1/build-%j.out

set -eo pipefail   # -u is enabled after the module loads (Pmodules is not -u safe)

deps="${FEMFFUSION_DEPS:-/data/scratch/shared/poncet_m/femffusion-deps-v1}"
src="${FEMFFUSION_SRC:-$HOME/FEMFFUSION}"
jobs="${BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
arch_flags="-O3 -march=znver2"
python_bin=/opt/psi/conda-envs/x86_64/ra-standard_py312/bin/python3

# Pmodules loads one module at a time reliably; a single "module load a b c"
# silently loaded nothing on merlin7. The login shell (#!/bin/bash -l)
# provides the module function.
module load gcc/12.3.0
module load openmpi/5.0.8
module load OpenBLAS/0.3.30
module list 2>&1
set -u

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$deps/build"
printf '=== FEMFFUSION build started at %s on %s (%s jobs) ===\n' "$(date -Is)" "$(hostname)" "$jobs"
grep -m1 "model name" /proc/cpuinfo || true

unpack() {  # unpack <tarball> -> prints the source directory
    local tarball="$deps/src/$1" dir="$deps/build/$2"
    [[ -d "$dir" ]] || tar -xzf "$tarball" -C "$deps/build"
    echo "$dir"
}

# --- PETSc: real scalars, 32-bit indices (FEMFFUSION passes int and double*
# --- to the SLEPc/PETSc API), no Fortran, optimized.
if [[ ! -f "$deps/petsc/lib/libpetsc.so" ]]; then
    cd "$(unpack petsc-3.19.6.tar.gz petsc-3.19.6)"
    "$python_bin" ./configure --prefix="$deps/petsc" \
        --with-cc=mpicc --with-cxx=mpicxx --with-fc=0 \
        --with-debugging=0 --with-shared-libraries=1 --with-x=0 \
        --with-scalar-type=real --with-precision=double --with-64-bit-indices=0 \
        --with-blaslapack-lib="-L$OPENBLAS_LIBRARY_DIR -lopenblas" \
        COPTFLAGS="$arch_flags" CXXOPTFLAGS="$arch_flags"
    make PETSC_DIR="$PWD" PETSC_ARCH=arch-linux-c-opt -j"$jobs" all
    make PETSC_DIR="$PWD" PETSC_ARCH=arch-linux-c-opt install
fi
export PETSC_DIR="$deps/petsc"
unset PETSC_ARCH

# --- SLEPc
if [[ ! -f "$deps/slepc/lib/libslepc.so" ]]; then
    cd "$(unpack slepc-3.19.2.tar.gz slepc-3.19.2)"
    export SLEPC_DIR="$PWD"
    "$python_bin" ./configure --prefix="$deps/slepc"
    make SLEPC_DIR="$PWD" PETSC_DIR="$PETSC_DIR" -j"$jobs"
    make SLEPC_DIR="$PWD" PETSC_DIR="$PETSC_DIR" install
fi
export SLEPC_DIR="$deps/slepc"

# --- deal.II: Release only, MPI + PETSc + SLEPc + LAPACK, bundled boost
# --- (FEMFFUSION's XML input uses boost::property_tree).
if [[ ! -f "$deps/dealii/lib/cmake/deal.II/deal.IIConfig.cmake" ]]; then
    dsrc="$(unpack dealii-9.5.2.tar.gz dealii-9.5.2)"
    rm -rf "$deps/build/dealii-build" && mkdir -p "$deps/build/dealii-build"
    cd "$deps/build/dealii-build"
    cmake "$dsrc" -DCMAKE_INSTALL_PREFIX="$deps/dealii" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
        -DDEAL_II_CXX_FLAGS="-march=znver2" \
        -DDEAL_II_COMPONENT_EXAMPLES=OFF -DDEAL_II_COMPONENT_DOCUMENTATION=OFF \
        -DDEAL_II_WITH_MPI=ON \
        -DDEAL_II_WITH_PETSC=ON -DPETSC_DIR="$PETSC_DIR" \
        -DDEAL_II_WITH_SLEPC=ON -DSLEPC_DIR="$SLEPC_DIR" \
        -DDEAL_II_WITH_LAPACK=ON \
        -DLAPACK_LIBRARIES="$OPENBLAS_LIBRARY_DIR/libopenblas.so" \
        -DBLAS_LIBRARIES="$OPENBLAS_LIBRARY_DIR/libopenblas.so" \
        -DDEAL_II_ALLOW_BUNDLED=ON -DDEAL_II_FORCE_BUNDLED_BOOST=ON \
        -DDEAL_II_WITH_TRILINOS=OFF -DDEAL_II_WITH_P4EST=OFF \
        -DDEAL_II_WITH_HDF5=OFF -DDEAL_II_WITH_GSL=OFF -DDEAL_II_WITH_ARPACK=OFF \
        -DDEAL_II_WITH_GMSH=OFF -DDEAL_II_WITH_SUNDIALS=OFF -DDEAL_II_WITH_CGAL=OFF \
        -DDEAL_II_WITH_ASSIMP=OFF -DDEAL_II_WITH_VTK=OFF -DDEAL_II_WITH_OPENCASCADE=OFF \
        -DDEAL_II_WITH_GINKGO=OFF -DDEAL_II_WITH_ADOLC=OFF -DDEAL_II_WITH_SYMENGINE=OFF \
        -DDEAL_II_WITH_SCALAPACK=OFF -DDEAL_II_WITH_METIS=OFF -DDEAL_II_WITH_CUDA=OFF
    make -j"$jobs" install
fi
export DEAL_II_DIR="$deps/dealii"

# --- FEMFFUSION (out of source; the checkout itself is not modified).
bdir="$src/build-release"
mkdir -p "$bdir" && cd "$bdir"
cmake "$src" -DDEAL_II_DIR="$DEAL_II_DIR" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j"$jobs"

commit="$(git -C "$src" rev-parse HEAD)"
"$python_bin" - "$deps" "$bdir" "$commit" <<'PY'
import json, os, platform, subprocess, sys
deps, bdir, commit = sys.argv[1:]
def first_line(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()[0]
    except Exception as exc:  # noqa: BLE001 - provenance only
        return f"unavailable: {exc}"
info = {
    "femffusion_commit": commit,
    "binary": os.path.join(bdir, "femffusion.exe"),
    "petsc": "3.19.6", "slepc": "3.19.2", "dealii": "9.5.2",
    "cxx": first_line(["mpicxx", "--version"]),
    "mpi": first_line(["mpirun", "--version"]),
    "blas": os.environ.get("OPENBLAS_LIBRARY_DIR", ""),
    "flags": "PETSc/SLEPc -O3 -march=znver2; deal.II Release + -march=znver2; FEMFFUSION Release",
    "build_host": platform.node(),
    "modules": os.environ.get("LOADEDMODULES", ""),
    # runtime library path of the build toolchain; tools/run_femffusion.py
    # prepends it so runs never pick up the system libstdc++/OpenBLAS
    "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
}
with open(os.path.join(bdir, "build-info.json"), "w") as fh:
    json.dump(info, fh, indent=2)
print(json.dumps(info, indent=2))
PY

# FEMFFUSION's tests use paths relative to the checkout and exit at once in a
# Debug build; run them single-threaded from the checkout.
cd "$src"
"$bdir/femffusion.exe" -t
printf '=== FEMFFUSION build finished at %s ===\n' "$(date -Is)"
