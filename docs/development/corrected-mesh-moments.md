# Corrected order-1 mesh moments

`garnet.corrected_mesh_moments.CorrectedMeshMomentOperator` implements a fixed
linear nonorthogonal finite-volume closure for the two pseudo-moments of SP3
and SDP1. The constructor follows `SDPNGroupOperator`; `coeffs` selects the
family and `boundary_g` selects its coupled Marshak matrix. State storage is
`(2, n_cells, 1, 1)`, with flattened index `moment*n_cells + cell`.

This is a scalar material coefficient closure using diffusion, total cross
section, and removal. It does not establish consistency with native angular
transport or angularly resolved interface data.

## Reconstruction and currents

Each cell reconstructs both gradients simultaneously. For interior neighbor
`j`, the equation for moment `m` is

```
[r_ij + (D_mi/D_mj - 1) d_j n_ij] . grad(U_mi) = U_mj - U_mi.
```

The neighbor-owner equation reverses the normal and uses `d_i`. This relation
encodes continuous piecewise linear trace and normal current across a scalar
material interface. Normal distances come from the signed centroid-to-face
projections; tangent offsets come from the physical face centers.

Boundary rows encode the actual homogeneous boundary law. Reflective faces
supply `n . grad(U_m) = 0`; zero-flux faces supply
`r . grad(U_m) = -U_m`. Coupled Marshak vacuum supplies

```
D_m n . grad(U_m) + sum_k G_mk r . grad(U_k) = -sum_k G_mk U_k.
```

Vacuum constraints are coupled during reconstruction. Every equation is
normalized by its coefficient-vector norm before the local least-squares
solve. A singular-value rank test rejects deficient patches; there is no
fallback to an uncorrected stencil. The pseudoinverse defines fixed sparse
linear gradient coefficients at construction time.

Interior integrated current is

```
S [(U_mi + grad(U_mi).t_i) - (U_mj + grad(U_mj).t_j)]
  / (d_i/D_mi + d_j/D_mj).
```

The same current enters the two cell balances with opposite signs. On a
Marshak boundary, with `A = diag(D_m/d)`, the integrated outward current is
`K (U + grad(U).t)`, where `K = S G solve(A + G, A)`. Matrix multiplication
order matters for SDP1. Reflective currents vanish; zero-flux currents use
`K = S A`. Scalar Robin conditions are also homogeneous and apply separately
to each moment when a coupled vacuum matrix is absent.

The reaction uses the full SPN/SDPN coefficient matrices, with cell volume
multiplying every block entry. Diffusion coefficients are `D1` and
`1/(7*sigma_t)`. Source, scalar-flux, and time weights retain the existing
pseudo-moment conventions. Sparse rows are assembled once and evaluated with
the shared backend gather. `face_currents(state)` returns interior and
boundary arrays of shapes `(2, n_faces)` and `(2, n_boundary_faces)`.

## Supported scope and validation

The operator requires finite 2D/3D face centers, unit normals, positive signed
normal distances, and full-rank reconstruction patches. It supports order 1,
homogeneous boundary conditions, and ordinary material coefficients. It rejects
active/hybrid masks, discontinuity factors, and similarity symmetrization.
Nonorthogonal reconstruction generally yields a nonsymmetric operator; use
GMRES. No general positivity, coercivity, or convergence guarantee is asserted
for arbitrary distorted meshes or coefficient jumps.

`tests/test_corrected_mesh_moments.py` compares both families against literal
independent orthogonal dense matrices and the established operator. A skew
manufactured affine field checks recovered coupled gradients, interior
currents, and the Marshak boundary law. Reflective constants, conservative
balances, dense column orientation, and GMRES against a dense solve provide
additional checks. CPU validation does not establish GPU performance or
transport-reference accuracy.


The public common-mesh eigen drivers select this closure with
`mesh_discretization="corrected"` and use GMRES with fixed TPFA incomplete-LU
preconditioning. Actual AMD validation covers the synthetic two-group sheared
mesh with coupled vacuum boundaries, with fusion enabled and disabled; CPU
and GPU eigenvalues and scalar fluxes agree to double-precision roundoff.
That evidence does not qualify a large fitted reactor mesh or its XS.

## Nonconstant radial coupled-boundary manufactured solutions

`tests/test_corrected_mesh_radial_moment_mms.py` extends the scalar polar
manufactured problem to both SP3 and SDP1, using NumPy-generated triangles and
an independent SciPy sparse direct solve. Radii are 0.4107, 0.4186 and 0.6309;
the material diffusion coefficients for moment zero are 0.7, 1800 and 0.3.
The second moment has `D_2 = 1/(7 sigma_t) = 3 D_0/7`, with
`sigma_t = 1/(3 D_0)` and removal 0.13.

The independently stated mathematical PDE is
`-div(D_m grad U_m) + sum_k (0.13 C0_mk + sigma_t Ct_mk) U_k = q_m`.
The oracle uses literal rational matrices:

| Family | C0 | Ct | G |
| --- | --- | --- | --- |
| SP3 | `[[1,-2/3],[-2/3,4/9]]` | `[[0,0],[0,5/9]]` | `[[1/2,-1/8],[-1/8,7/24]]` |
| SDP1 | `[[1,-2/3],[-6/7,4/7]]` | `[[0,0],[0,5/7]]` | `[[1/2,-1/8],[-2/7,23/42]]` |

With `f = [1,-0.35]`, the circular exact solution is
`U_m(r) = b_m + f_m integral_r^R s/(2 D_m(s)) ds`, where
`b = solve(G, R f/2)`. Thus both boundary traces are nonzero and the coupled
homogeneous law is `-D_m normal.grad(U_m) = sum_k G_mk U_k`.
Both off-diagonal entries contribute; neither boundary is replaced by a scalar
Robin approximation. The operator receives the native family coefficient and
boundary tables, while the exact field and source use the literal matrices
above. This tests the numerical implementation of those specified mathematical
matrices, without independently establishing their angular transport derivation.

Each moment's quadratic field is averaged analytically over each triangle.
The integrated source is exactly `f_m area + sum_k R_mk mean(U_k) area` within
each material extension, where `R = 0.13 C0 + sigma_t Ct`. Source construction
does not call `op.apply`, read an operator reaction matrix, or use the discrete
solution. Refining both angles and all radial subdivisions gives 1,440, 5,952
and 24,192 unknowns. Single-thread CPU results for the combined area-weighted
L2 norm across both moments are:

| Family | L2, 1,440 unknowns | L2, 5,952 unknowns | L2, 24,192 unknowns | Orders |
| --- | --- | --- | --- | --- |
| SP3 | 2.211181499e-3 | 5.534289871e-4 | 1.384177784e-4 | 1.99835, 1.99937 |
| SDP1 | 2.259099966e-3 | 5.659952681e-4 | 1.416545891e-4 | 1.99689, 1.99841 |

The individual moment L2 errors are respectively
`[2.163360967e-3,4.573761561e-4]`,
`[5.415391426e-4,1.141008359e-4]`,
`[1.354448663e-4,2.853365670e-5]` for SP3, and
`[2.253309444e-3,1.616453018e-4]`,
`[5.645640581e-4,4.022521359e-5]`,
`[1.412890374e-4,1.017007988e-5]` for SDP1.
The tests require refinement error ratios above 3 for each moment and the
combined norm. They check direct-solve residuals, conservative per-cell balances
with the independently specified reaction, and each moment's global outward
current against integrated source minus reaction.

As in the scalar radial test, this is joint polygon geometry and discretization
convergence. Piecewise quadratic extensions have O(angular spacing squared)
trace jumps on interface chords. The circular boundary field also has a
nonzero Robin defect on outer chords; at chord midpoints this is
`G (b + f R^2 sin(a)^2/(4 D_outer)) - f R cos(a)/2`, with `a = pi/n_angles`.
The test prints that defect, which decreases by approximately four per level.
The radial current is analytically continuous, but the circular reference is
not an exact solution of the finite polygon problem. These results provide
nonconstant curved-interface evidence for the coupled numerical closure;
reactor eigenvalues, SerpentGd/TRC transport agreement, and physical Marshak
matrix derivation remain separate qualification questions.
