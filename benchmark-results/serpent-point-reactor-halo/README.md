# Serpent point-reactor validation on halo

Validated on 2026-10-07 using halo's AMD Radeon 8060S (gfx1151), the pinned
repository ROCm Nix shell and project-local environment. Python 3.13.15,
NumPy 2.5.3, SciPy 1.18.1 and HIP CuPy 14.2.0 are recorded in `summary.json`.
The ROCm wheel is `cupy-rocm-7-0`; hardware/toolkit limits are described in
[the AMD setup documentation](../../docs/amd_rocm.md).

```sh
nix develop path:.#rocm --command .venv/bin/python examples/serpent-xs/point-reactor/run.py --device gpu --require-rocm --output-dir results/serpent-point-reactor
```

All five cases pass on CPU and actual HIP device arrays with fusion off/on.
The following k values are from the independent dense balance; spectrum
errors are the CPU maximum absolute difference from normalized `INF_FLX`.

| Groups | Dense k | Serpent IMP_KEFF | Maximum spectrum error |
| --- | --- | --- | --- |
| 2 | 1.1831958633 | 1.18316 | 3.31e-5 |
| 4 | 1.1831949524 | 1.18316 | 4.06e-5 |
| 7 | 1.1832017736 | 1.18316 | 2.26e-5 |
| 14 | 1.1831753298 | 1.18313 | 2.13e-5 |
| 23 | 1.1830793801 | 1.18309 | 4.86e-6 |

Maximum solver error against dense k is 8.49e-10; maximum relative balance
residual is 4.33e-10. Every Serpent k and normalized-spectrum check passes the
[documented tolerances](../../docs/serpent_xs.md). Some informational
`INF_KINF` comparisons exceed three reference-only standard errors; this
quantity is not used to fit or accept the imported balance.

`summary.json` contains all case results, source hashes, convergence, solver
settings, uncertainties, closure diagnostics and individual acceptance checks.
CSV files contain group boundaries and all five normalized spectra: Serpent,
dense, CPU, and the two GPU variants. Raw inputs remain local and unchanged.
The solve timings include initialization effects and are not performance
benchmarks. This evidence covers fixed-state zero-leakage diffusion eigenpairs.
