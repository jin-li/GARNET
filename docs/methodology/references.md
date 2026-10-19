# References and implementation map

The following are primary method or vendor references. They explain the underlying methods; the implementation modules define NDgpu's supported subset and conventions.

- P. S. Brantley and E. W. Larsen, “The Simplified P3 Approximation,” *Nuclear Science and Engineering* 134 (2000), 1–21. [Publisher and DOI](https://doi.org/10.13182/NSE134-01). SP3 variational derivation, interfaces and Marshak-like boundaries.
- A. Carreño, A. Vidal-Ferràndiz, D. Ginestar and G. Verdú, “Development of a diffusive simplified double PN approximation to the neutron transport equation,” *Annals of Nuclear Energy* 207 (2024), 110675. [DOI](https://doi.org/10.1016/j.anucene.2024.110675). Source of the implemented SDPN coefficient matrices.
- M. Nazari, A. Zolfaghari and M. Abbasi, “Simplified-DPN treatment of the neutron transport equation” (2023 revision). [Author preprint](https://arxiv.org/abs/2212.02140). Half-range SDPN closure; consult the paper for its precise derivation rather than inferring it from matched SPN block sizes.
- B. Christensen, *Three-Dimensional Static and Dynamic Reactor Calculations by the Nodal Expansion Method*, Risø-R-496 (1985). [Institutional record](https://orbit.dtu.dk/en/publications/three-dimensional-static-and-dynamic-reactor-calculations-by-the-/). Background on polynomial nodal expansion and transverse leakage. NDgpu's static implementation has its own narrower scope.
- Serpent developers, [group-constant generation](https://serpent.vtt.fi/docs/user_guide/gc_generation.html), [standard output](https://serpent.vtt.fi/docs/user_guide/standard_output.html), and [calculation options](https://serpent.vtt.fi/docs/user_guide/calculation_options.html). These are living references: retain the actual source-code/data version and estimator qualification.
- preCICE developers, [coupling documentation](https://precice.org/docs.html). Mapping, participants, acceleration and time windows.
- CuPy developers, [reviewed 14.2 GMRES implementation](https://github.com/cupy/cupy/blob/v14.2.0/cupyx/scipy/sparse/linalg/_iterative.py). The native-moment GPU wrapper qualifies this release series rather than assuming arbitrary version compatibility.

## Implementation map

- **Standard eigenvalue diffusion**: `solver.py`, `stencil.py`, `tri.py`, `mesh.py`. Evidence: `tests/verification`, `tests/validation`.

- **SPN/SDPN**: `sp3.py`, `spn.py`, `solver.py`. Evidence: SPN/SDPN and Brantley–Larsen tests/examples.

- **SN/hybrid**: `sn.py`, `tri_sn.py`, `hybrid_sn.py`, `hybrid_tri_sn.py`. Evidence: C5G7, SN acceleration and hybrid examples.

- **Time and IQS**: `transient.py`, `timescheme.py`, `quasistatic.py`. Evidence: TWIGL/LRA, point-kinetics and quasi-static tests.

- **Heat/feedback/coupling**: `thermal.py`, `feedback.py`, `power.py`, `coupling.py`. Evidence: conduction/feedback/power/coupling verification.

- **Frequency response**: `noise.py`. Evidence: noise point-kinetics and FEMFFUSION tests.

- **NEM/reference**: `nem.py`, `nem_higher_order.py`, `nem_geometry.py`, `nem_reference.py`. Evidence: `test_nem*`, reflector-interface tests.

- **Matching static importance**: `nem_kinetics.py`, `diffusion_kinetics.py`. Evidence: `test_nem_kinetics.py` and diffusion kinetics tests.

- **Serpent/native moments**: `serpent_xs.py`, `moment_materials.py`, `coupled_moments.py`. Evidence: Serpent moment and coupled-operator tests.

- **Condensation**: `moment_condensation.py`. Evidence: `test_moment_condensation.py`.

Module paths above are relative to `ndgpu/` unless they name a test directory. The repository's existing detailed notes provide additional API context: [model API](../model_api.md), [coupling](../coupling.md), [Serpent scalar import](../serpent_xs.md), and [static kinetics](../kinetics_parameters.md).
