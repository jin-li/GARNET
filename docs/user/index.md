# NDgpu User Manual

Use this manual to select a supported workflow, build a model and interpret its outputs. The companion [Software Methodology Manual](../methodology/index.md) explains equations, discretizations and qualification limits. Examples here use synthetic data or existing public benchmark entry points. No private study data are needed for the introductory runs.

```{toctree}
:maxdepth: 2

getting-started
models
serpent-mesh
branch-xs
nem-kinetics
transients
validation
```

## Choose a workflow

| Goal | Start with | Execution and scope |
| --- | --- | --- |
| Rectangular steady model | `Model` or `DiffusionEigenSolver` | CPU first; supported standard paths also use CuPy |
| Hexagonal/prismatic model | `HexLattice.build()` → `TriReactor` | Steady, selected higher moments, transient and thermal interfaces |
| General finite-volume mesh | `MeshModel`; low-level `MeshGrid` solvers for advanced moments | Builder is diffusion-only; low-level capabilities have separate contracts |
| Fixed Serpent scalar constants | `read_serpent_xs` | Explicit record/field choice; no automatic state interpolation |
| Static branch XS | `BranchLibrary`, `build_nem_solver` | Experimental bounded absolute interpolation; CPU NEM, independent transport validation required |
| Native anisotropic matrices | `read_serpent_moment_xs` and coupled moment operators | Opt-in research path, explicit orientation; distinct from standard SPN |
| Cuboid NEM/ADF | `CuboidNodalGrid`, NEM solver classes | CPU NumPy/SciPy; local reference qualification is explicit |
| Static physical kinetics | Matching forward/adjoint projection | Raw regional data required for delayed families; no transient NEM solver |
| Spatial transients and feedback | `Model.transient`, `TriReactor` coupling methods | Critical-adjusted initial state and documented kinetics assumptions |
| Frequency response | `NoiseSolver` | Small-signal complex calculation with its own global kinetics contract |

A reported GPU device confirms execution; `auto` by itself does not. A reported convergence flag confirms an iterative stop; residuals and refinement studies establish whether that stop is useful for your application.
