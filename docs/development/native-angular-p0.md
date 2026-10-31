# Fixed native P0 Galerkin angular preconditioner

`garnet.native_angular_p0.P0GalerkinTransport` is an explicit alternative to the
default `NativeMeshTransport` diagonal preconditioner. The default solver keeps
the same diagonal arithmetic and provenance. The subclass changes only the
fixed linear left preconditioner supplied to GMRES; the angular loss `A`, native
fission `F`, cold starts, normalization, positivity checks, iteration limits,
and true convergence gates remain inherited and unchanged.

```python
from garnet.native_angular_p0 import P0GalerkinTransport

solver = P0GalerkinTransport(streaming, boundary_conditions)
# Constructor assembles and factors one scalar-moment coarse matrix.
result = solver.solve()
```

This software stage follows a saved failed pilot whose weighted residual
squared was predominantly P0 (81.77%), with P2 contributing 11.49%. That evidence
motivates a P0 correction, without establishing that it will converge within
the existing process limits. The first failed pilot remains a failed result;
this implementation does not imply an actual pin retry, cap increase, physical
qualification, or continuous-energy/SPN accuracy claim.

Restriction and isotropic prolongation use the bound full-sphere quadrature:

```text
(R psi)[cell,group] = sum_n w[n]*psi[cell,n,group]
(P phi)[cell,n,group] = phi[cell,group]/(4*pi)
R P = I
Ac = R A P
```

`A` includes native scattering and specular reflection, with no fission term.
The coarse unknown order is cell-major, group-minor. For canonical native
production matrices `[to,from]`, each cell's reaction block is exactly
`V*(diag(Sigma_t)-SP0)`. Higher signed angular production moments are retained
unchanged in the full operator; their isotropic projection vanishes to
quadrature precision.

For each interior face, define the **discrete** half-range coefficient

```text
T_face = face_measure/(4*pi) * sum_n w[n]*max(Omega_n dot normal,0)
```

Add `+T_face` to both neighboring same-group diagonals and `-T_face` to their
same-group off-diagonal couplings. A vacuum face adds `T_face` to its owner's
diagonal; a specular face adds zero. The sum over negative normal velocities
must match the positive half range within floating-point tolerance scaled by
the face measure and quadrature's in-plane speed integral. This includes
oblique interior/vacuum faces and numerical tangencies. No planar direction
renormalization, continuum `face_measure/4` substitution, diffusion coefficient,
Marshak boundary term, SPD assumption, or material removal reconstruction is
used. Reflection retains the base solver's Cartesian-axis restriction.

The constructor calls `scipy.sparse.linalg.spilu` once with defaults
`drop_tol=1e-6`, `fill_factor=20`, `drop_rule='basic,area'`, `permc_spec='COLAMD'`,
`diag_pivot_thresh=1`, and `Equil=False`. Drop tolerance and fill factor are
explicit constructor parameters for controlled numerical testing, including
weak ILU fixtures. They remain fixed throughout an instance's lifetime. There
is no adaptive rebuild, shift, regularization, clipping, direct-solve fallback,
or hidden second factorization.

Each preconditioner application is the multiplicative linear correction

```text
z = P * ILU(Ac)^-1 * R * r
M^-1 r = z + D_stream^-1 * (r - A*z)
```

`D_stream` is the unchanged diagonal of streaming plus total collision. This
requires one extra full angular action per application. It is supplied through
the base solver's `precondition()` hook as a SciPy GMRES left preconditioner.
The factor and maps do not depend on the current residual, iteration number,
eigenvalue, or prior calls. Repeated application to the same array is
repeatable, and linear combinations are preserved within arithmetic precision.
No dense restriction/prolongation, full angular matrix, all-ordinate sparse
assembly, or fission contribution enters this preconditioner.

Construction rejects closed conservative singular systems, zero/nonfinite
streaming diagonals, inconsistent half ranges, nonfinite coarse matrices or
factors, exact zero/nonfinite pivots, and factorization errors. There is no
small-pivot rejection threshold or physical-unit floor. A mathematically
singular assembly can appear nonsingular after floating-point face additions;
successful ILU is not a condition-number or nonsingularity certificate.
The exact singular fixture avoids that perturbation. No pivot or matrix entry
is modified.
Application rejects nonfinite factor solutions/corrections. Failures terminate
construction or the inherited inner solve, with no fallback or automatic retry.

Before coarse triplets or factor storage are allocated, the budget includes
the base restarted-Krylov/terminal/history payload, coarse CSR/CSC/COO and Python
triplet construction, coarse vectors and permutations, estimated retained ILU
storage, and extra angular residual/action workspace. The numeric factor
estimate uses `32*ceil(fill_factor*entry_bound)` bytes; it is explicitly **not**
a hard guarantee on SuperLU memory. Actual visible retained L/U nnz and array
bytes, including row/column permutation arrays, are measured after factorization
and checked against the payload budget again. Opaque SuperLU/runtime overhead
is not claimed as measured. External 8 GiB address-space limits and the admitted
CPU, thread, and wall-time caps remain authoritative. A larger payload budget
must be chosen before an attempt; no retry increases it.

`preconditioner_provenance()` records the complete ILU configuration, one-time
factor count, coarse CSR fingerprint/nnz/bytes, measured L/U nnz/bytes,
preflight/retained memory estimates, pivot rule, NumPy/SciPy versions, and hashes
of the preconditioner, transport, streaming, angular-source, and material
implementations. Native source ledgers remain the caller's responsibility.
The exposed `coarse_matrix` is a read-only CSR copy so caller replacement of
its buffers cannot change the retained preconditioner matrix.

Tiny verification uses independent dense Legendre-kernel angular equations and
explicit R/P matrices to check `Ac=RAP`, plus direct matrix-free `R A P` action
checks. It covers unequal-volume heterogeneous materials, mixed boundaries,
oblique faces, physical discrete half ranges, orientation and signed L7
preservation, fixed-factor linearity/repeatability, unchanged A/F bytes, and
isolated physical eigenpair invariance under both default and weak ILU settings.
Failure and allocation guards are exercised without actual pin data. Passing
these tests does not qualify upwind spatial convergence, angular convergence,
physical positivity of a signed native P7 kernel, or pin accuracy.
