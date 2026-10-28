# Cartesian ADF development and provisional Bernal benchmark

This is the first implementation of Cartesian, steady-state diffusion with
assembly discontinuity factors (ADFs). It uses a two-point face-current
approximation, not Bernal et al.'s polynomial FVM. The provisional model below
uses their Table 1 group constants, not a verified reconstruction of their
reactor. The published comparisons are nodal diffusion calculations, not
heterogeneous transport truth.

## Using ADFs

`DiffusionEigenSolver(..., adf=faces, linear_solver="gmres")` accepts three
pairs of arrays `(low_side, high_side)`, in x, y, z order. Each array has a
leading energy-group dimension followed by the grid shape with one fewer cell
along the face axis. For example, x faces have shape `(G, nx-1, ny, nz)`.
Low/high refers to the coordinate of the cell, not the cell's face normal.
The factors are heterogeneous divided by homogeneous surface flux. They must be
real, finite and strictly positive; complex input arrays are rejected before
conversion. Arrays on axes of length one are empty. All
interior faces must be specified, including unity on faces where no ADF is
applied. Factors are ignored at exterior and active/inactive mask faces, where
the existing boundary laws apply to homogeneous flux.

`garnet.assembly_face_factors(regions, side_factors)` helps construct these
arrays. `regions` is a 3-D integer array identifying homogenized regions, not
materials. `side_factors` has shape `(G, 6, nx, ny, nz)`, with sides ordered
x-, x+, y-, y+, z-, z+. The helper uses the adjacent cells' factors on a region
boundary and unity within a region. Thus two assemblies can have the same
material but still have distinct regions and oriented factors; artificial
refinement faces never receive an ADF. The helper runs on the CPU and the
solver transfers the arrays to its backend.

For a face normal from L to R, the current is
`J = (f_L phi_L - f_R phi_R) / (f_L d_L / D_L + f_R d_R / D_R)`.
The same current enters the two cell balances with opposite signs. The
resulting matrix is generally nonsymmetric; the solver rejects CG whenever
`adf` is supplied, including equal-factor cases, and transposes the nonsymmetric stencil for adjoint solves.
The projected adjoint-residual diagnostic uses the same operator selection.
Each face-factor pair is normalized before resistance evaluation, preserving
the current under common scaling and avoiding factor-scale overflow. Factor
ratios and face coefficients that cannot be represented as finite positive
values in the operator dtype raise an error.
GMRES and BiCGStab are available. Cartesian ADFs are not wired into the model
API, transient, SPN, triangular/hexagonal, unstructured, or cylindrical solvers. The existing no-ADF stencil and fused kernel remain in use
when `adf` is omitted. A separate fused GPU kernel is provided for directed
face weights. It has not yet run on a GPU; the GH200 verification and
benchmark jobs are prepared in
[benchmark-results/adf-gh200](../benchmark-results/adf-gh200/README.md).

## Reconstructing the example

[Paper](https://doi.org/10.1016/j.anucene.2016.06.023), [open author manuscript](https://riunet.upv.es/bitstreams/2b913a56-bea8-4c47-8d0d-4bd409d79458/download).
The example uses the six materials in Table 1, 20 cm base cells, 10 axial
levels, 1,170 active nodes, zero-flux exterior faces, and the two-group
production/scattering equations of Eq. 3. The default lateral ADFs are the
Table 1 values and axial ADFs are unity. Each original node is a region, so
subdivisions do not introduce spurious factors.

The default material map is **a hypothesis, not the paper's verified input**.
The figures suggest an 11-by-11 radial footprint with its four corner cells
removed. Using symmetry, let `a=|x-5|`, `b=|y-5|`, with 0-based indices.
The interior eight levels contain material 6 where `max(a,b)=5`, material 2
where `max(a,b)=4`, and material 3 elsewhere, except that material 1 is placed
at the center on axial levels 3–9, at `(a,b)=(2,2)` on levels 4–9,
and at `(a,b)=(3,0),(0,3)` on levels 6–9. The six-level interior
column is a revised visual interpretation of Fig. 6: it appears taller than
the two four-level columns. The previous hypothesis made all three columns
four levels tall. The bottom and top levels use materials 4 and 5. The figure leaves
hidden cells and rod heights difficult to determine. Zero-flux on stepped mask
faces and the treatment of axial factors are also assumptions. An online search
located the author manuscript, but no original input deck or independent
material map. The paper states its ADFs are the same on lateral faces; it does
not establish the axial convention used here.

Run, for example:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python3 examples/bernal_adf_benchmark.py \
  --refinements 1 2 3 4 --output bernal-results.json
```

`--axial-factors` repeats the tabulated factors on axial faces as a sensitivity
study. `--device gpu` requests CuPy. `--map path.npy` supplies a corrected
integer map of shape `(11,11,10)` using 0 for inactive and 1–6 for materials.
The result includes eigenvalues, physical algebraic residuals, iterations,
timings, and normalized assembly **fission neutron production**. The paper's
power definition uses Σ_f, whereas its table provides νΣ_f, so this production
is not a substitute for physical assembly power or the paper's power errors.

## Initial CPU results

On the provisional map, with lateral ADFs and unit axial factors, relative pcm
is `1e5 * (k - 1.040374) / 1.040374`, using the paper's VALKIN eigenvalue:

| subdivisions per axis | k, ADF off | k, ADF on | ADF-on pcm vs VALKIN |
|---:|---:|---:|---:|
| 1 | 1.04893699 | 1.04889264 | +818.8 |
| 2 | 1.04117984 | 1.04015728 | −20.8 |
| 3 | 1.04038881 | 1.03894315 | −137.5 |
| 4 | 1.04049314 | 1.03881018 | −150.3 |

All eight relative algebraic residuals were below 8e-9. With the alternative
axial-factor assumption, ADF-on k was 1.04902609, 1.04016349, 1.03887937,
and 1.03870288 respectively. The prior four-level interpretation of the
interior column gave k = 1.04361364 for the refined ADF-on case. This
sensitivity rules out using agreement with the published k to confirm a map. These are useful sensitivity and code-exercise
results only: the material layout and boundary conventions are unverified and
the paper's polynomial method has a different discretization error. Their
change with mesh size does not establish a benefit against the actual reactor.

Next steps: obtain or independently reconstruct the exact map and axial ADF
convention; compare normalized assembly powers using compatible Σ_f; run GPU
fused/unfused equivalence and end-to-end timing, including host transfers; and
validate against a heterogeneous transport reference. A separate, fully
specified [heterogeneous slab verification](adf_slab_reference.md) exercises
reference factors and interface currents. Supported model APIs remain future
work.

`GroupOperator.face_currents(phi)` returns oriented interior current densities
for Cartesian grids, with zero on masked faces. Multiply by face area for
integrated current. Physical and mask-boundary currents are not included.

`GroupOperator.face_traces(phi)` returns, for each interior face, the two
homogeneous surface-flux reconstructions `phi_L - J h/(2 D_L)` and
`phi_R + J h/(2 D_R)`; with factors, `f_L * low == f_R * high`.
`garnet.region_side_traces(ops, flux, regions)` averages traces and integrates
net currents per region side. `garnet.expand_region_side_factors(regions,
factors)` turns a (G, R, 6) per-region library into the cell array taken by
`assembly_face_factors`.

`make_problem` and `run` in the example accept `linear_solver`,
`precond_degree`, `factor_exponent` (0 gives explicit unity factors) and
`node_factors`. `tools/assess_adf_gpu.py` uses these for its solver-separated
and factor-contrast GPU studies.
