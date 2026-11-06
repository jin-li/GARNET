# Bounded native angular transport iteration

`garnet.native_mesh_transport.NativeMeshTransport` adds a bounded matrix-free
fixed-source and cold fission-power iteration to `NativeMeshStreaming` and
`NativeAngularSource`. It is a mathematical comparator pilot. Its convergence
flag qualifies the numerical equations described here, not the native data,
physical kernel, spatial/angular convergence, continuous-energy accuracy, or SPN
accuracy. Signed P7 kernels remain mathematical operators; they do not gain
physical admission by passing an iteration.

```python
from garnet.native_mesh_transport import NativeMeshTransport

# Explicitly choose vacuum or axis-aligned specular reflection for every face.
bc = tuple('vacuum' for _ in streaming.boundary_owners)
transport = NativeMeshTransport(streaming, bc)
fixed = transport.fixed_source(cell_integrated_external_source)
result = transport.solve()  # cold start only; no input eigenstate or warm cache
```

A boundary-condition sequence must match the boundary face order exactly. Each
entry is `vacuum` or `reflective`. Reflection is supported only for Cartesian
x/y aligned normals in 2D, and x/y/z aligned normals in 3D.
The quadrature supplies an exact index permutation that
is checked for involution, equal physical weights, preserved tangential
components (including z), and reversed normal components, within floating-point
coordinate tolerance. Oblique reflection is rejected; no angular interpolation
is introduced. An incoming reflective ordinate uses the corresponding outgoing
ordinate of its boundary owner. Vacuum incoming flux is zero. Arbitrary
reflective angular fields have zero integrated boundary net current up to
floating-point summation. Trigonometric near-tangencies retain their computed
coefficient, as in the streaming foundation.

The same native-moment source and energy data are used in both dimensions.
For an axially constant extrusion of height `h`, reflecting axial faces recover
`A3*psi = h*A2*psi` only when the field is even under z reflection. Arbitrary
angular fields require matched prescribed cap inflow for that cancellation;
z-odd fields generally differ. The physical fission action scales by `h`.
This physical-operator identity does not imply equality of preconditioners:
their streaming diagonal and block solves retain axial cap outflow.

The linear loss operator and fission operator are

```text
A psi = L_stream psi - V*q_native(psi) - R_reflect psi
phi[c,g] = sum_n w[n]*psi[c,n,g]
P(psi) = sum_c V[c] sum_g nuSigmaF[material[c],g]*phi[c,g]
F psi[c,n,g] = V[c]*chi[material[c],g]
               * sum_gp nuSigmaF[material[c],gp]*phi[c,gp]/(4*pi)
```

Native chi, nuSigmaF, and kappa-fission vectors must be finite and nonnegative;
they are copied exactly, without chi renormalization. For a
one-group homogeneous reflective cell, the expected eigenvalue is therefore
`chi*nuSigmaF/(Sigma_t-SP0)`, including when chi is deliberately unequal to one.
Volume appears once in both scattering and fission terms. Angular weights occur
in angular integrals, not per-ordinate streaming equations. The operators
support finite signed mathematical fields without clipping.

`fixed_source(rhs)` starts from exactly zero and uses SciPy GMRES with only the
diagonal of streaming plus total collision as its preconditioner. No LU,
factorization cache, warm start, DSA, or other acceleration is present. A zero
preconditioner diagonal fails clearly; no floor or regularization is applied.
`callback_type='legacy'` with an explicit callback makes `max_inner` bound actual
inner Arnoldi steps rather than restart cycles. `restart` is bounded by the
vector size and `max_inner`. Defaults are `max_inner=200`, `restart=30`, and
`max_outer=100`. Acceptance requires GMRES status zero and the independently
recomputed unpreconditioned norm `||A*y-rhs||/||rhs|| <= inner_tolerance`.
Zero or nonfinite RHS norm fails closed, including an exactly zero RHS. The
fixed-source result reports convergence of this linear algebra alone.

Closed, all-reflective systems whose native total equals the sum of outgoing
SP0 production in every group have a weighted particle-balance left nullspace.
They are rejected before GMRES, including conservative and void fixtures. The
comparison admits only roundoff differences (`64*machine_epsilon` times the
native magnitudes). This sufficient singularity guard changes no operator or
native data; other singular or ill-conditioned cases must pass GMRES status and
the true residual, or return a failure. It is not a general condition-number
certificate.

`solve()` starts with a uniform angular field, normalizes it to `P=1`, and takes
cold linear steps

```text
A*y = F*psi_old
k_new = P(y)
psi_new = y/k_new
```

There is no division by an estimated chi sum and no separate multiplication by
the previous eigenvalue. Every linear step starts from zero. Nonpositive
production/eigenvalue, exact-zero residual denominators, nonfinite arithmetic,
inner failure, and outer exhaustion return `converged=False` with a reason.
There are no zero or negative floors. Numerical acceptance requires all of

```text
GMRES status == 0
true inner residual <= 1e-7
abs(k_new-k_old)/abs(k_new) <= 1e-6
||F*psi_new-F*psi_old||/||F*psi_new|| <= 1e-5
||A*psi_new-F*psi_new/k_new||/||F*psi_new/k_new|| <= 1e-5
```

Tolerances can be tightened through constructor parameters; loosening these
gates is rejected. Norms are unweighted Euclidean norms of the cell-integrated
angular equations. The returned history stores scalar values only, including
true inner residual, inner status/step count, k change, source change, and full
residual. A stable k alone never qualifies convergence.

Normalized scalar flux and cell-integrated heat power must be at least
`-1e-12*scalar_flux_scale` and `-1e-12*power_scale`, respectively, at every cell
(and every scalar-flux group). Heat power is computed as
`V[c]*sum_g kappa_fission[material[c],g]*phi[c,g]`; it is distinct from the
neutron-production functional `P` used for eigenstate normalization. Diagnostics
report both quantities under separate names. The scales are explicit positive constructor
parameters, defaulting to one in the caller's physical units; callers must
choose scales appropriate to their normalization. Angular negativity is
reported separately through the minimum angular flux and the area/weight
integral `sum_c,n,g V[c]*w[n]*max(-psi[c,n,g],0)`. That diagnostic does not alter
the field or establish physical positivity of a truncated kernel.

Terminal angular/scalar arrays are bytes-backed immutable copies. History and
provenance mappings are immutable and record cold initial state, zero linear
initial guesses, convergence gates, declared positivity scales, resource
settings, energy boundaries, and `physical_admission=False`. An arithmetic
failure that prevents finite terminal diagnostics returns finite zero terminal
arrays with an explicit terminal-arithmetic failure reason, never convergence.

Before copying material fields or allocating iteration state, the constructor
checks the complete estimated own numeric payload: the restarted Krylov basis,
Hessenberg workspace, simultaneous operator/scattering/fission/residual arrays,
old/new/terminal copies, reflection buffers, scalar diagnostics, and bounded
history. The default is `max_bytes=256 MiB`; the estimate is available as
`memory_estimate_bytes`. Own numeric workspace is approximately
`8*((restart+1)*D+48*D+4*(restart+1)^2+...)` bytes for `D=C*N*G`, plus 8192 bytes
per allowed outer iteration for scalar history. Existing bound geometry/source
storage, caller-owned arrays, and library/runtime overhead are outside this
payload estimate. Source and streaming allocation guards also remain active.
Any larger budget must be selected before an attempt; external process CPU,
wall-time, thread, and RSS limits remain required. No automatic larger retry,
all-angle factorization, or dense production solve exists.

Tiny tests compare homogeneous reflective k, deliberately unnormalized chi,
heterogeneous reflective net boundary currents on both axes, and independent
P0/P1 generalized dense angular eigenproblems. The dense oracle constructs its
own harmonic kernel through the addition theorem and scalar face equations;
it never uses operator columns or the streaming assembly. Tests also exercise
inner failure, outer exhaustion, singular/void cases, zero denominators,
positivity scales, malformed boundaries/controls, and memory limits. They make
no pin, mesh-refinement, upwind spatial-convergence, or physical accuracy claim.
