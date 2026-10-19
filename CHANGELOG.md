# Changes

## Unreleased: static branch cross sections

- Add immutable Serpent branch records, explicit bounded absolute interpolation,
  full fission-matrix and whole-face ADF transfer, and a fresh CPU NEM factory.
- Add reactivity/temperature-coefficient/rod-worth validation helpers that
  distinguish interpolation error from model bias and unresolved precision.
- Add a data-free example and manual chapters. These experimental interfaces
  require withheld transport states for reactor interpolation qualification;
  existing feedback and transient APIs retain their behavior.

## Unreleased — ISONEES feature consolidation

- Preserve CPU/CUDA support and add the verified optional AMD/ROCm environment.
- Import fixed-state Serpent constants with explicit energy, scattering,
  production, uncertainty and provenance conventions.
- Add common-mesh diffusion/SP3/SDP1, masked rectilinear geometry, conservative
  scalar interface factors and diffusion face-current export.
- Settle multigroup fixed sources before eigenvalue updates; preserve physical
  roundoff bounds and bound many-group Krylov acceleration memory and work.
- Restore the historical locally refined HPMR distance convention through an
  explicit two-dimensional assembler option; keep face-normal metrics available.
- Make transient-owned initialization defaults precision-aware and preserve
  strict explicit accuracy requests.
- Add experimental CPU cuboid NEM, higher-order responses and matching fuel and
  reflector homogeneous-reference/factor interfaces.
- Add native kinetics/ACE/MDEP data readers, explicit raw regional production
  contracts and static diffusion/NEM forward-adjoint parameter projections.
- Retain native higher-moment CPU/hybrid-GPU operators and homogeneous
  moment-hierarchy condensation as explicit experimental interfaces.
- Add synthetic workflows, two HTML/PDF manuals, packaging checks and CI.

Existing APIs and defaults are retained. CPU-only NEM is separate from the
existing GPU solvers. Whole-face factors and subface extensions have different
assumptions. Static adjoint projections are not a transient NEM implementation;
branch XS and temperature/rod-motion studies remain future work. See the
[feature inventory](docs/development/integration_inventory.md) and
[manuals](docs/index.md) for support and qualification limits.
