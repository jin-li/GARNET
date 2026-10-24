# GARNET Software Methodology Manual

This manual describes the equations, discretizations, data contracts and numerical evidence behind GARNET. It accompanies the separate [User Manual](../user/index.md). The equations use scalar neutron flux $\phi$, importance $\phi^*$, and eigenvalue $k$. A converged numerical solution is evidence for the specified discrete problem; it does not establish that its homogenization, boundary data or nuclear constants represent another reactor state.

```{toctree}
:maxdepth: 2

foundations
spatial-methods
transport
time-coupling
nem-adjoint
data-verification
references
```

## Capability and maturity

The categories below distinguish established package pathways from specialized research interfaces. They are scope statements, not a certification or a claim of validation for every input library.

| Pathway | Geometry and execution | Evidence and limits |
| --- | --- | --- |
| Standard diffusion | Cartesian, cylindrical r-z, triangles/prisms; NumPy/CuPy | Analytic limits, conservation and named benchmark tests; coefficients and boundary assumptions remain user responsibilities |
| SP1/3/5/7 and SDP1/2/3 | Structured and triangular moment blocks; mesh interfaces where provided | Implemented steady and selected transient pathways; not interchangeable with a native anisotropic-scattering importer |
| SN and hybrid SN/diffusion | Cartesian and triangular transport regions | Angular/spatial convergence and interface tests required; specialized controls differ from diffusion |
| Thermal and coupled transients | Supported structured/triangular operators; CPU/GPU | Conduction, prescribed sink, reduced feedback; no fluid dynamics or moving mesh |
| IQS | Diffusion shape/amplitude coupling | Guarded approximation with correction/fallback telemetry; compare against full spatial transients |
| Noise | Linear diffusion/SPN frequency response | Small perturbations about a static reference; restricted kinetics contracts |
| Native Serpent moment pathway | Explicit coupled full-matrix operators; CPU and opt-in hybrid GPU solve | Research interface; estimator/transport consistency is not inferred; GPU runs only CSR/Arnoldi work on device |
| Cuboid NEM and higher-order NEM | NumPy/SciPy CPU only | Quartic through degree 16, whole-face ADF, matching static adjoint; research qualification required |
| Static NEM kinetics | Frozen NEM discrete forward/adjoint pair | Physical time and raw delayed-production projection; does not implement a transient NEM solver |
| Moment hierarchy condensation | Homogeneous $B\to0$, CPU | Opt-in solution-dependent projection; no finite-buckling or heterogeneous equivalence guarantee |

