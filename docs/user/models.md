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
