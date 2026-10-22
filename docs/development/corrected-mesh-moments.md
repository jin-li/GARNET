# Corrected order-1 mesh moments

`ndgpu.corrected_mesh_moments.CorrectedMeshMomentOperator` implements a fixed
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
