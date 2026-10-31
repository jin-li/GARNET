# Native angular-source foundation

`garnet.native_angular_source` supplies an independent mathematical angular
scattering operator for a future heterogeneous finite-volume comparator. It
contains no streaming discretization, boundary transport solve, fission source,
eigenvalue iteration, or physical qualification of native moments.

`ProductQuadrature(L)` uses Gauss–Legendre polar cosine and uniform azimuth on
the full sphere. Physical weights sum to `4*pi`. The default counts are
`Npolar=L+1`, `Nazi=2L+2`; user counts must meet these lower bounds, and azimuth
must be even. This integrates harmonic products through degree `2L` for `L<=7`.
The real basis is ordered `(l,m)` with `l=0..L`, `m=-l..l`, using
Condon–Shortley associated Legendre functions, sine for negative `m`, cosine for
positive `m`, and normalization
`sqrt((2*l+1)*(l-|m|)!/(l+|m|)!)` (times `sqrt(2)` for nonzero `m`). Thus

```text
H00 = 1
integral Ha(Omega) Hb(Omega) dOmega = 4*pi*delta_ab
sum_m Hlm(Omega) Hlm(Omega') = (2*l+1)*Pl(Omega dot Omega')
```

Directions use polar-major, azimuth-minor ordering. `reflection('x'|'y'|'z')`
returns an exact index permutation for the axis reflection; coordinates agree
to floating-point precision. Frozen quadrature and source metadata use
bytes-backed read-only arrays.

```python
from garnet.native_angular_source import ProductQuadrature, NativeAngularSource

quadrature = ProductQuadrature(7)  # 128 directions, 64 harmonics
source = NativeAngularSource(quadrature, tuple(materials), cell_material_ids)
phi = source.moments(psi)  # psi[cell, angle, group], phi[cell, harmonic, group]
q = source.scatter(psi)   # q[cell, angle, group]
```

The default quadrature counts integrate the requested harmonic products; they
do not establish angular convergence for a streaming problem. In particular,
the L0 default has only two directions. A transport solver must choose and
verify its angular resolution independently.

Each material must be `MomentMaterial` with explicit provenance
`orientation_contract` equal to `from-major` or `to-major`, finite monotonic
`energy_bounds_mev`, and every production order `0..L`. All materials must have
identical group counts and energy boundaries. Production matrices are converted
once to `[to_group, from_group]`. Collision and loss matrices, total-removal
inferences, and fission data do not enter this calculation:

```text
phi[c,a,g] = sum_n w[n] H[n,a] psi[c,n,g]
q[c,n,to] = (1/(4*pi)) sum_a H[n,a]
             * sum_from SP_l[a][to,from] phi[c,a,from]
```

The addition theorem places the `2*l+1` factor in the harmonic sum; callers must
supply SP Legendre moments without multiplying that factor a second time.
Signed higher-order moments and signed reconstructed kernels remain unchanged.
Kernel positivity is not an admission requirement for this mathematical
operator. Native estimator convention, cross-section collapse, and truncated
kernel physical suitability still need independent qualification before using
this as a physical reference. Local inspection of estimator source supports the
selected Legendre convention, but does not establish binary/source identity or
physical admission.

For `C` cells, `N` directions, `A=(L+1)^2` harmonics, `G` groups, and `K`
materials, persistent production storage is `K*(L+1)*G^2` float64 values, rather
than a cell-expanded matrix. Projection stores `C*A*G` values; the source output
stores `C*N*G`. Each cell's transformed moments use `A*G` values. No `N*N`
angular matrix is built. All public operations have a default 256 MiB allocation
estimate limit (`max_bytes`); estimates include conservative construction copies
and projection workspace. These are allocation guards rather than process RSS
limits, and exclude caller-owned arrays and Python/library overhead. Quadrature
construction also accounts for the dense `Npolar^2` Legendre setup. Keep the
external process resource limits when running verification.

Verification covers L0/1/3/7 Gram matrices and addition theorems, isotropic
projection, an analytic two-group P0/P1 oracle, orientation equivalence, signed
higher moments, heterogeneous material selection, particle production balance,
reflection, immutability, malformed data, and allocation guards. These checks
establish the angular algebra only; they do not validate a transport solution.
