# Geometry, materials and solver selection

## Materials and regions

`Material` stores macroscopic multigroup constants in fast-to-thermal order. The following is a complete two-group declaration:

```python
from garnet import Material, Model
fuel = Material(name="fuel", diffusion=[1.26, 0.35],
                sigma_a=[0.012, 0.121], nu_sigma_f=[0.0085, 0.185],
                sigma_s=[[0.0, 0.026], [0.0, 0.0]], chi=[1.0, 0.0])
reflector = Material(name="reflector", diffusion=[1.15, 0.90],
                     sigma_a=[0.0002, 0.005], nu_sigma_f=[0.0, 0.0],
                     sigma_s=[[0.0, 0.045], [0.0, 0.0]])
model = (Model(size=(120.0, 120.0), cells=(32, 32))
         .fill(reflector)
         .add_box(fuel, x=(30.0, 90.0), y=(30.0, 90.0))
         .set_boundary("vacuum"))
print(model.run(method="diffusion", device="cpu"))
```

The scattering entry `[0,1]` transfers neutrons from group 0 to group 1. Later boxes overwrite earlier boxes. Materials are registered by object identity: independently controlled regions need distinct objects even if their constants initially match. Inspect `model.materials` and `model.material_map` before writing any callback that returns a material list.

Set all boundaries with one name, or give per-axis/per-face specifications, for example `set_boundary(x=("reflective", "vacuum"), y="vacuum")`. Numeric albedo uses the package's current/flux Robin convention, rather than a generic reflected-particle probability.

## Hexagonal and prismatic models

This block continues the two materials above:

```python
from garnet import HexLattice
core = (HexLattice(pitch=20.0, refine=2)
        .set_disk(2, reflector)
        .set_disk(1, fuel)
        .set_boundary("vacuum")
        .extrude(height=60.0, nz=3, boundary="vacuum")
        .build(name="synthetic prismatic core"))
solution = core.steady(method="diffusion", device="cpu")
print(solution)
```

`refine` controls triangular subdivisions within a hex, not energy resolution. Axial regions can replace selected materials before `build`. Control drums mix an absorber arc into triangle cells by sampled area fraction; study both spatial refinement and arc sampling. Cached built frames are useful for moving-control transients.

See [custom triangular reactor](https://github.com/jin-li/GARNET/blob/main/examples/custom_tri_reactor.py), [hexagonal lattice](https://github.com/jin-li/GARNET/blob/main/examples/hex_lattice.py), and the detailed [builder reference](../model_api.md). Existing HP-MR examples exercise the bundled benchmark model, not a universal input template.

## Higher moments and transport

Supported steady method names include `diffusion`, `sp1`, `sp3`, `sp5`, `sp7`, `sdp1`, `sdp2` and `sdp3` in the applicable builders. Higher-moment calculations require appropriate total/transport data; do not interpret adding a method name to scalar Serpent diffusion constants as qualification for transport. Consult the actual solver constructor for geometry and `marshak_vacuum` controls.

Low-level classes expose options beyond the builders. `SNTransportSolver` is a 2-D Cartesian interface requiring `nz == 1`; triangular/prismatic SN uses its separate class. Hybrid solvers require an explicit transport-region mask. Use [C5G7](https://github.com/jin-li/GARNET/blob/main/examples/c5g7_2d.py), [SN acceleration](https://github.com/jin-li/GARNET/blob/main/examples/sn_dsa_benchmark.py), [Cartesian hybrid](https://github.com/jin-li/GARNET/blob/main/examples/hybrid_sn_hpmr.py), and [triangular hybrid](https://github.com/jin-li/GARNET/blob/main/examples/hybrid_tri_sn_hpmr.py) as runnable patterns. Compare quadratures, spatial meshes and hybrid interface locations.

An adjoint run returns importance. Its eigenvalue should agree with the forward value at the requested accuracy, but its shape is not a neutron population or a power shape.

## Experimental Cartesian face factors

The low-level steady `DiffusionEigenSolver` accepts Cartesian ADFs; the `Model`
builder does not expose them. This path is separate from the fork's mesh face
factor and cuboid NEM prescriptions. Its arrays and side order are different,
so do not pass a mesh or NEM factor array unchanged.

The following synthetic CPU example illustrates the contract, not a qualified
assembly library. It uses the `fuel` material declared above:

```python
import numpy as np
from garnet import Grid, DiffusionEigenSolver, assembly_face_factors

grid = Grid((4, 1, 1), (40.0, 1.0, 1.0))
regions = np.zeros(grid.shape, dtype=int)
regions[2:, :, :] = 1  # region identities need not equal material identities
sides = np.ones((2, 6, *grid.shape))  # group, side, nx, ny, nz
sides[:, 1, 1, :, :] = 0.95  # left region's x+ side
sides[:, 0, 2, :, :] = 1.05  # right region's x- side
faces = assembly_face_factors(regions, sides)
solver = DiffusionEigenSolver(grid, fuel, bc="reflective", device="cpu",
                              adf=faces, linear_solver="gmres")
result = solver.solve(tol_k=1e-9, tol_source=1e-8)
print(result.k_eff)
```

`sides` has shape `(G,6,nx,ny,nz)` in x−, x+, y−, y+, z−, z+ order.
The helper sets factors to unity within each region and uses the adjacent
oriented factors at region boundaries. The solver receives three pairs of
low/high-coordinate arrays, in x/y/z order; x arrays have shape
`(G,nx-1,ny,nz)`, with the analogous reduced dimension on y/z. Collapsed-axis
arrays are empty. All interior faces must be supplied. Factors must be real,
finite and strictly positive. Exterior and active/inactive faces retain their
homogeneous-flux boundary law.

Use GMRES or BiCGStab; CG is rejected whenever `adf` is supplied because the
directed leakage matrix is generally nonsymmetric. This interface does not
extend to transients, SPN, cylindrical, triangular/hexagonal or unstructured
solvers. A GPU kernel exists, but the upstream audit did not execute it on
hardware. See the [Cartesian reference](../bernal_adf_benchmark.md) for the full
contract and the [synthetic slab verification](../adf_slab_reference.md) for
independently constructed diffusion equivalence factors. The provisional
Bernal geometry remains unverified, and the Dall'Osso reconstruction retains
an unresolved reference discrepancy. None of this evidence qualifies PRATIC
or establishes transferability of Serpent factors to a different receiver.

For a library indexed by region, `expand_region_side_factors(regions, factors)`
expands `(G,R,6)` factors into the cell-side array consumed by
`assembly_face_factors`. `region_side_traces(operators, flux, regions)` collects
area-averaged traces and integrated outward currents on active interior region
interfaces. Its side order is x−, x+, y−, y+, z−, z+. Unscored sides have NaN
traces and zero area/current; they are not usable ADF denominators. A region's
same-orientation faces are pooled, so a single box-face interpretation requires
box-shaped regions.

The [C5G7 colorset example](../c5g7_colorset_adf.md) compares isolated-assembly
libraries on several arrangements. Read the reference-refinement results before
interpreting an ADF improvement; these are diffusion comparisons, not a new
qualification of transport-derived factors. The GH200 staging scripts prepare
separate optional GPU campaigns and do not launch from the normal test suite.
