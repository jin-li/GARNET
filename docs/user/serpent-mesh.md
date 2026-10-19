# Serpent data and common finite-volume meshes

## Select a fixed-state scalar record

This template requires your own existing Serpent result file. It does not start Serpent:

```python
from ndgpu import DiffusionEigenSolver, Grid
from ndgpu.serpent_xs import read_serpent_xs
record = read_serpent_xs("assembly_res.m", universe="0")
material = record.to_diffusion_material(diffusion="cmm")
solver = DiffusionEigenSolver(Grid((1, 1, 1), (1.0, 1.0, 1.0)),
                             material, bc="reflective", device="cpu")
result = solver.solve(tol_k=1e-10, tol_source=1e-9)
print(record.source_sha256, result.k_eff)
```

Use `read_serpent_results` to inventory multiple records, then select a unique universe or zero-based record index. Preserve the exact energy boundaries. `diffusion="inf"` is an explicit alternative field choice; missing requested data should remain an error. The reflective single cell is a zero-leakage balance test, not a finite assembly model or a delayed point-kinetics calculation.

The scalar adapter uses reduced absorption and production scattering. Retained collision fields have different meanings. Read the [scalar import contract](../serpent_xs.md) before comparing balances or interpreting uncertainties. Fixed-state reading does not interpolate temperature, density, burnup or B1 leakage.

## Inspect native ADF tallies

For a selected record with ADF output, the reader keeps native face order and returns surface arrays as `(face,group)`:

```python
surface_flux = record.adf_estimate("DF_HET_SURF_FLUX")
net_inward = record.adf_estimate("DF_SURF_NET_CURR")
print(record.n_surfaces, surface_flux.mean, net_inward.mean)
```

This block continues the data-dependent scalar-record template. Surface flux is a surface average; native partial/net currents are integrated tallies, with net inward equal to incoming minus outgoing. The reader does not divide by area or reorder faces. Establish the native-to-receiver side map explicitly, and convert integrated current to current density only at the interface that requires it. Printed `DF_SURF_DF` is not automatically qualified for a different receiver or production kernel. Reference reconstruction and whole-face factor qualification remain separate operations.

## Native moments are opt-in

The separate reader requires orders and orientation explicitly:

```python
from ndgpu.serpent_xs import read_serpent_moment_xs
native = read_serpent_moment_xs("assembly_res.m", universe="0",
                                required_orders=(0, 1, 2, 3))
moments = native.to_moment_material(orientation="from-major")
print(moments.loss[1])
```

This template needs a file containing the required native fields. Choose `from-major` only when that flattening contract has been independently established. Signed higher moments and native reported error metadata remain unchanged. This object feeds the specialized coupled-moment path; standard `Material` SPN/SN constructors do not automatically consume it. See the [methodology](../methodology/transport.md) for full-matrix, hybrid GPU and $B\to0$ condensation limits.

## Build a nonuniform masked mesh

The following example needs no external data:

```python
import numpy as np
from ndgpu import Material, MeshModel
from ndgpu.nodal_mesh import build_rectilinear_mesh_3d
active = np.ones((1, 2, 2), dtype=bool)  # nz, ny, nx
active[0, 1, 1] = False
mesh, topology = build_rectilinear_mesh_3d(
    [0.0, 10.0, 25.0], [0.0, 12.0, 20.0], [0.0, 5.0],
    active_mask=active, tags=np.zeros(active.shape, dtype=int))
fuel = Material(name="synthetic", diffusion=[1.2], sigma_a=[0.025],
                nu_sigma_f=[0.030], sigma_s=[[0.0]], chi=[1.0])
model = MeshModel(mesh).fill(fuel).set_boundary("reflective")
print(model.run(device="cpu"))
print(topology["grid_indices"])
```

The topology records exposed masked faces separately from outer-box faces; this builder example deliberately makes both reflective. If your model requires different exposure conditions, assign them through the appropriate low-level mesh boundary interface. IDs follow active `(z,y,x)` C order, while centroid and width components are `(x,y,z)`. Tags are labels, not automatically material indices.

The vertex-free result supports finite-volume connectivity, not vertex export. `MeshModel` is a diffusion builder; low-level `MeshGrid` moment solvers have their own data and factor contracts. For general Gmsh input see [unstructured mesh](https://github.com/jin-li/NDgpu/blob/main/examples/unstructured_mesh.py). The synthetic [common mesh example](https://github.com/jin-li/NDgpu/blob/main/examples/isonees_features/common_mesh_fv.py) compares the integrated common-mesh pathway.
