# NixOS and NVIDIA validation on 550w

Validation date: 2026-10-05. These selected artifacts preserve the earlier
NixOS/CPU/NVIDIA checks performed before integrating the AMD feature.
They are not a full-suite or reactor-physics validation.

Hardware: NVIDIA GeForce RTX 5070 Ti, 16 GB. Environment: Nix Python
3.13.15, NumPy 2.5.3, SciPy 1.18.1, CuPy 14.2.0, CUDA 12.9 component
wheels, NVIDIA driver 595.71.05.

- [CPU examples](cpu-examples.json): eleven commands exited successfully.
- [Selected CPU tests](cpu-tests.txt): exact command and 103 passing tests.
- [GPU command summary](gpu-examples.json): initialization probe plus seven
  successful example/benchmark commands. Paths in this historical JSON
  refer to the ignored local results directory; individual example logs
  are not included in this selected artifact set.
- [Driver diagnosis and comparison result](diagnosis.txt): sandbox device
  visibility, environment versions and CPU/GPU comparison tolerances.
- [CPU/GPU comparison script](compare_cpu_gpu.py): convergence and numerical
  agreement checks used in that environment.
- [Warmed benchmark output](speed-benchmark.txt): 32³, 64³ and 96³ bare-box
  diffusion, with speedups 1.9×, 22.6× and 55.2×.
- [Post integration checks](post-amd-checks.txt): commands and selected
  observed output from 52 regression tests and the NVIDIA smoke run.

The benchmark used a shared GPU and one timing sample. It is not comparable
directly to halo's repeated heterogeneous diffusion measurements. The
comparison script is historical evidence, not a general AMD validator.
Full original logs remain under ignored local `results/nixos-validation/`
and `results/nixos-gpu-validation/`.

After AMD synchronization, 52 backend/fused-kernel tests and an actual
NVIDIA 32³ bare-reactor smoke run passed. That smoke run was a compatibility
check, not a repetition of the historical benchmark. See the
[development assessments](../../docs/development_assessments.md) for current
findings and the [AMD report](../amd-rocm-halo/README.md) for halo evidence.
