# Data provenance and verification

## Serpent fixed-state import

The literal parser never evaluates MATLAB. Select a unique universe/record explicitly, preserve the source SHA256 and version, and retain energy boundaries in MeV without implicit rebinning. Native relative errors are not percentages; printed record count is not an independent-sample count. Missing covariance prevents a general propagated uncertainty for normalized spectra or products.

The scalar diffusion adapter uses `INF_RABSXS`, source-major `INF_SP0`, `INF_NSF` and `INF_CHIT`. Reduced absorption and production scattering account for neutron multiplication channels and must not be silently replaced by physical collision absorption and collision scattering. `CMM_DIFFCOEF` is the default requested diffusion field; `diffusion="inf"` explicitly chooses `INF_DIFFCOEF`. Whole-lattice suitability of a coefficient does not establish its suitability in a pin or reflector subregion. Consult the [Serpent group-constant definitions](https://serpent.vtt.fi/docs/user_guide/gc_generation.html).

The scalar adapter, native moment reader, ADF reader and kinetics reader are separate interfaces. Importing one category does not qualify the others. Global IFP/analog/implicit estimates remain distinct; repeated whole-system fields must agree rather than being averaged. Exact estimator definitions depend on Serpent version. The [current time-constant reference](https://serpent.vtt.fi/docs/user_guide/standard_output.html#reactor-time-constants) must not be assumed identical to every older implementation.

## A hierarchy of evidence

| Check | What it establishes | What it leaves open |
| --- | --- | --- |
| Parser and schema rejection | Dimensions, finite values, declared conventions and provenance | Whether tallies represent the intended physics |
| Algebraic residual | Accuracy for the implemented discrete equations | Mesh, angular, time and homogenization error |
| Conservation | Pairwise leakage and global balance | Accuracy of cross sections or closure |
| Analytic limit | Correct reduction under controlled assumptions | Heterogeneous full-core prediction |
| Independent pencil/transpose test | Consistency of condensed response and adjoint | Transport estimator equivalence |
| Mesh/order/time refinement | Convergence for the selected model | Bias of that model |
| Independent benchmark | Agreement with a stated reference and conditions | General qualification outside those conditions |
| CPU/GPU comparison | Backend agreement for tested inputs | GPU availability or performance elsewhere |

For a reflected homogeneous scalar problem an independent dense calculation uses

```{math}
M=\operatorname{diag}(\Sigma_a+S\mathbf1)-S^T,
\quad q=M^{-1}\chi,\quad k=\nu\Sigma_f^Tq.
```

Its normalized spectrum can be compared with the finite-volume result without a spatial discretization error. Finite-leakage comparisons need compatible geometry and boundary treatment; a zero-leakage test does not qualify them.

## Reproducible records

Retain software revision, Python/NumPy/SciPy/CuPy versions, device and precision, exact source hashes, selected record identities, region/energy mappings, material/production kernels, boundaries, factor reconstruction settings and all refusal/diagnostic flags. Record tolerances, iteration limits, residuals, positivity minima, mesh and order convergence, and uncertainty assumptions separately from timing.

Benchmark results should identify their reference and normalization. Do not infer a standard error from arbitrary signed native metadata, combine correlated errors as independent, or present deterministic discrepancies as a validated uncertainty. A historical source hash preserves provenance of that historical computation; it does not qualify a later changed kernel.

The manuals and synthetic examples distribute method descriptions and reproducible artificial problems. They do not distribute proprietary source contents or reactor study results. Private data qualification belongs in a separately controlled evidence package.
