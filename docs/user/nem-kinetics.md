# Cuboid NEM, references and static kinetics

## A complete artificial forward/adjoint calculation

This example is CPU-only and uses no Monte Carlo data. Unequal reflected nodes share homogeneous one-group constants, giving $k=0.035/0.03$ and physical $\Lambda=1/(220000\times0.035)$ s.

```python
import numpy as np
from ndgpu.nem_geometry import CuboidNodalGrid
from ndgpu.nem_higher_order import NEMHigherOrderDiffusionSolver
from ndgpu.nem_kinetics import solve_nem_adjoint, project_nem_kinetics

grid = CuboidNodalGrid([
    [[0.0, 20.0], [0.0, 10.0], [0.0, 10.0]],
    [[20.0, 50.0], [0.0, 10.0], [0.0, 10.0]],
], "reflective", leakage_boundary="primary_rule")
solver = NEMHigherOrderDiffusionSolver(
    grid, [1.2], [0.03], [[0.0]], [0.035], [1.0], degree=8)
forward = solver.solve_eigenvalue(rtol=1e-11)
adjoint = solve_nem_adjoint(solver, forward)
beta = np.array([0.002, 0.004])  # raw artificial family fractions
parameters = project_nem_kinetics(
    solver, forward, adjoint,
    inverse_velocities=np.full((2, 1), 1.0/220000.0),
    delayed_production=np.broadcast_to(
        (0.035*beta)[None, :, None, None], (2, 2, 1, 1)),
    decay=[0.08, 0.3])
print(parameters.k_eff, parameters.generation_time)
print(parameters.beta, parameters.neutron_lifetime)
assert np.isclose(parameters.generation_time, 1/(220000*0.035))
```

`NEMDiffusionSolver` provides the base quartic receiver. The higher-order class permits explicit degree 4 through 16 in even steps. Its default is not a substitute for order convergence. These solvers have no `device="gpu"` pathway. NEM geometry requires exact conforming whole cuboid faces; reuse shared coordinates. Hanging/partial faces are rejected.

## Factors and local reconstruction

Factors use `(node,side,group)` and side order x−, y−, x+, y+, z−, z+. Exterior factors must be one. Importing a tally is distinct from constructing its matching homogeneous reference. The reference must match production kernel, geometry, polynomial degree and transverse leakage policy. Use `reconstruct_nem_reference` and the explicit reflector-inclusive qualification API for the supported reconstruction workflow. Keep negative profile diagnostics and reference-side current disagreements visible.

The runnable [local reference example](https://github.com/jin-li/NDgpu/blob/main/examples/isonees_features/nem_local_reference.py) supplies an artificial reference. A successful local reconstruction does not validate the full receiving core. An auxiliary reference closure source must never be inserted into core fission or population.

## Regional delayed-production input

For real qualified input, use the versioned supplemental contract:

```python
from ndgpu.kinetics_data import read_regional_kinetics
# These variables must describe the same frozen solver state:
# node_region_ids: one stable region ID per node
# energy_bounds: exact descending boundaries of this library
data = read_regional_kinetics("regional_kinetics.json")
inputs = data.for_nodes(node_region_ids, energy_bounds, solver.fission)
parameters = project_nem_kinetics(solver, forward, adjoint, **inputs)
```

This is a data-dependent template, continuing a matching forward/adjoint solver. Supply raw total and family production matrices, inverse velocities, positive decay constants, hashes and declared approximations. Total uses `(region,target,source)`; delayed uses `(region,family,target,source)`. The prompt remainder must be nonnegative. A global Serpent effective beta cannot fill these arrays.

If only inverse speeds are available, use `project_nem_time` for a time-only projection. Preserve independent isotope/native-family channels if they have different decay constants. Family summaries do not turn multiple poles into one exact transient family. See the [detailed input contract](../kinetics_parameters.md).

## Interpret the result

`generation_time` is physical static $\Lambda$, `neutron_lifetime` is $k\Lambda$, and `beta` holds projected effective channel fractions. `lambda_low_frequency` is an explicitly labeled one-family low-frequency reduction. These outputs do not start a NEM transient or establish equivalence with an unqualified Monte Carlo estimator.

Any material/geometry change requires a new solver and matching forward/adjoint states. The cached response rejects replacement. Reconstruct factors again when the reference kernel changes. Use [static kinetics](https://github.com/jin-li/NDgpu/blob/main/examples/isonees_features/static_kinetics.py) and [the standalone NEM example](https://github.com/jin-li/NDgpu/blob/main/examples/kinetics/static_nem.py) for reproducible synthetic runs.
