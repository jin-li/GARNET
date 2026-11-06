# Native fitted-mesh streaming foundation

`garnet.native_mesh_streaming.NativeMeshStreaming` evaluates a conservative,
first-order upwind streaming and native total-collision operator on a general
fitted 2D or 3D `Mesh`. It is an algebraic foundation, with no transport iteration,
eigenvalue solver, sparse solve, acceleration, reflection interpolation, or
physical qualification. No new transport or pin physics run is implied by its
verification tests.

```python
from garnet.native_mesh_streaming import NativeMeshStreaming

operator = NativeMeshStreaming(mesh, native_angular_source)
loss = operator.action(psi)             # [cell, angle, group]
b = operator.boundary_rhs(incoming)     # incoming [boundary_face, angle, group]
r = operator.residual(psi, incoming)    # loss - b
r_scatter = operator.residual(psi, incoming, include_scattering=True)
one_matrix = operator.assembled(angle=0, group=0)  # one cell-by-cell CSR
interior_current, boundary_current = operator.face_currents(psi, incoming)
```

Binding to one `NativeAngularSource` fixes the quadrature, cell-material IDs,
energy boundaries, and group count without a second interpretation of the
material layout. Geometry arrays consumed by this operator and native total
cross sections are copied into bytes-backed immutable storage at construction;
subsequent edits to the original mesh cannot change the operator. Totals are
retained once per material, exactly as supplied. They must be finite and
nonnegative; zero is allowed for evaluation, without any invertibility promise.
No total reconstruction, scattering cancellation, clipping, Marshak condition,
regularization, or fission treatment is applied.

The mesh must explicitly provide consistently 2D or 3D cell centroids, positive cell measures,
positive face measures and distances, face centers, and unit normals. Interior
normals point from the first owner (left) to the second owner (right), with the
face center between their centroid projections. Boundary normals point outward
from the owner. Owners must be distinct valid cell indices. For each cell the
sum of outward `face_measure*normal` vectors must vanish to a relative tolerance
of `2e-12` times its total face measure. Missing, reversed, nonfinite, or
inconsistent consumed geometry is rejected. These checks validate the consumed
finite-volume geometry; they do not establish arbitrary polygon topology or
mesh convergence. The read-only `dimension` attribute records the spatial
dimension. In 3D, `Mesh.area` stores cell volumes and face measures are areas.
Use `assemble_mesh_3d` or the existing Gmsh loader for conforming tetrahedra,
triangular prisms, or hexahedra; the polygon builder remains a 2D utility.

A 3D source requires `ProductQuadrature` with at least two polar nodes and
four azimuthal nodes. The P0 defaults have no axial transport and are rejected
for a 3D mesh; choose, for example,
`ProductQuadrature(0, n_polar=2, n_azimuth=4)` explicitly. Established 2D
defaults are unchanged. The default P7 quadrature already satisfies this guard.
These minimum counts establish directional coverage, not angular convergence.

For an ordinate `Omega`, physical three-dimensional direction components
are used without renormalization. A face's coefficient is
`s=A*sum_d(Omega_d*n_d)` over the mesh dimension, including z in 3D.
On an interior face, the upstream value is
the left cell when `s>=0` and the right cell when `s<0`. That face contributes
`+s*psi_upwind` to the left equation and `-s*psi_upwind` to the right equation.
The cell-integrated loss and boundary forcing are

```text
L psi[c,n,g] = V[c]*Sigma_t[material[c],g]*psi[c,n,g]
               + interior outward upwind fluxes
               + sum_boundary A*max(Omega dot n_out,0)*psi[c,n,g]
b[c,n,g] = sum_boundary -A*min(Omega dot n_out,0)*psi_in[face,n,g]
residual = L psi - b
residual_with_scattering = L psi - b - V*q_native(psi)
```

There is no quadrature weight in these per-ordinate equations, and the native
angular source is multiplied by cell area (2D) or volume (3D) exactly once. `incoming=None` means
vacuum (zero incoming flux). Outgoing and exactly tangential entries in a
supplied incoming array are ignored, even when those unused entries are NaN.
Consumed incoming entries and all consumed angular fluxes must be finite.
Signed flux values are allowed for mathematical verification. Tangential
ordinates require no inversion or special regularization. Floating-point
trigonometric near-tangencies retain their computed signed coefficient.

`face_currents` returns groupwise physical angular integrals, including the
face measure: `A*sum_n w[n]*(Omega_n dot normal)*psi_upwind[n,g]`. Interior
current is positive left-to-right. Boundary current is positive outward and
uses the owner trace for outgoing directions and the supplied incoming trace
for incoming directions. These currents use full-sphere weights summing to
`4*pi`; interior contributions cancel exactly between neighboring equations
up to floating-point summation. No surface-average division is applied.

`assembled(angle,group)` builds only one sparse cell-by-cell matrix. Its scalar
row construction is independent of the vectorized face accumulation used by
`action`, so their comparison can detect implementation discrepancies. Sparse
matrices have nonnegative diagonals and nonpositive off-diagonals for admitted
totals; no claim of strict diagonal dominance, nonsingularity, or transport
solution accuracy follows. No dense assembly or factorization is performed.

With `C` cells, `F` interior faces, `B` boundary faces, `N` angles, `G` groups,
and `K` materials, persistent storage is `O(C+F+B+K*G)` beyond the bound source.
Action and residual workspace is bounded by `O(C*N*G)` plus the bound angular
source's projection workspace. Physical currents add `(F+B)*G` output values.
One CSR assembly requires `O(C+F+B)` temporary storage. The default
`max_bytes=256 MiB` checks conservative allocation estimates before geometry
copies or operation workspace. These guards exclude caller-owned arrays,
existing source storage, and Python/library overhead; retain external process
resource limits. Outputs are checked for nonfinite arithmetic results.

The tiny verification suite includes one- and two-cell analytic operators,
unequal cell areas, oblique normals, both flow directions and tangencies,
constant fields with matched incoming flux, physical current balance,
independent CSR and boundary-forcing checks, matrix signs, native-source volume
scaling, frozen geometry, malformed inputs, zero totals, and allocation guards.
Default quadrature counts establish harmonic exactness, not angular streaming
convergence. No physical benchmark, mesh/angle refinement claim, reflection
condition, or transport solve is included.
