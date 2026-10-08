# Fixed-state Serpent cross sections

NDgpu reads Serpent `*_res.m` group constants as literal data, without
executing MATLAB or requiring an additional parser package. The initial
interface supports a fixed-state, homogenized infinite-spectrum diffusion
eigenvalue problem. It has been validated with the supplied Serpent 2.2.0
assembly datasets containing 2, 4, 7, 14 and 23 groups.
The [2-D assembly example](../examples/serpent-xs/fa-2d/README.md) also
reconstructs 289 regional four-group materials with exact nonuniform cell
dimensions and compares CPU/AMD diffusion results against Serpent.

```python
from ndgpu import DiffusionEigenSolver, Grid
from ndgpu.serpent_xs import read_serpent_results, read_serpent_xs

record = read_serpent_xs("assembly_res.m", universe="0")
material = record.to_diffusion_material(diffusion="cmm")
solver = DiffusionEigenSolver(
    Grid((1, 1, 1), (1., 1., 1.)), material,
    bc="reflective", device="cpu",
)
result = solver.solve(tol_k=1e-11, tol_source=1e-10, max_outer=2000)
spectrum = result.flux_numpy.reshape(-1)
spectrum /= spectrum.sum()
```

The reflective one-cell model has zero leakage. Its arbitrary cell dimensions
and diffusion coefficients do not affect the eigenpair. Here “point reactor”
means this spatially uniform multigroup balance; it does not introduce a
point-kinetics solver or delayed-neutron dynamics. Comparing it to a Serpent
assembly requires a compatible zero-leakage reference. A finite assembly
with vacuum boundaries would require a spatial model and leakage treatment.

## Records, conventions and validation

`read_serpent_results(path)` returns all records. `read_serpent_xs(path,
universe=None, record_index=None)` selects exactly one; ambiguous selections
raise an error. Record indices are zero-based. Each record retains its source
path, SHA256, universe, Serpent version, group count, raw literal fields and
descending energy boundaries in MeV. The supplied upper `1e37` sentinel and
lower zero boundary are preserved. Different group structures need not be
nested; the reader does not rebin them.

`record.estimate(name)` returns separate `mean`, `relative_error` and
`absolute_error` arrays for recognized statistical fields. Relative errors
are Serpent's relative standard errors, not percentages. Original interleaved
values remain in `record.fields`. No covariance or combined uncertainty is
inferred. The parser checks required fields, declared vector lengths,
dimensions, finite values, nonnegative means/errors, positive diffusion
coefficients and descending boundaries. Unsupported expressions in required
or recognized physics fields fail; unrelated expressions are ignored.

The diffusion adapter uses these conventions:

| NDgpu material quantity | Serpent field | Convention |
| --- | --- | --- |
| `sigma_a` | `INF_RABSXS` | Reduced absorption, including neutron multiplication corrections |
| `sigma_s` | `INF_SP0` | Scattering production, reshaped as source group × destination group |
| `nu_sigma_f` | `INF_NSF` | Total fission neutron production |
| `chi` | `INF_CHIT` | Total fission spectrum |
| `sigma_t` | `INF_TOT` | Total collision cross section |
| `diffusion` | `CMM_DIFFCOEF` | Default CMM diffusion coefficient |
| `kappa_fission` | `INF_KAPPA × INF_FISS` | Optional fission-energy cross section in MeV/cm |

Macroscopic cross sections use cm⁻¹ and diffusion coefficients use cm.
`diffusion="inf"` explicitly selects `INF_DIFFCOEF`; missing requested
coefficients raise an error. Chi is normalized only after its sum passes a
`1e-5` tolerance, without modifying the record.
Non-fissile regions may have absent or zero chi; their material receives a
benign first-group spectrum with zero fission source. `INF_FISS` and
`INF_KAPPA` must both be present or both absent. Their product supports
normalized power shapes; its uncertainty is not propagated without covariance.

The CMM default is appropriate for the whole-assembly infinite-lattice use
case. It must not be assumed valid for pin subregions. The 2-D example
explicitly selects `diffusion="inf"` as an approximation study and records
that limitation; it does not silently replace missing regional coefficients.

Physical absorption (`INF_ABS`) and collision scattering (`INF_S0`) remain
available separately. Replacing reduced absorption and production scattering
with them loses the `(n,xn)` neutron contribution: in these datasets that
changes k by about 175–182 × 10⁻⁵. Consequently, the adapted `sigma_a` must
not be interpreted as a physical absorption tally. `balance_diagnostics()`
reports collision and production closure discrepancies from printed constants;
it does not alter or fit the data.

See the [Serpent group-constant documentation](https://serpent.vtt.fi/docs/user_guide/gc_generation.html)
and [standard-output documentation](https://serpent.vtt.fi/docs/user_guide/standard_output.html)
for definitions and generation options.

## Point-reactor validation

The [example runner](../examples/serpent-xs/point-reactor/README.md) compares
each reflective NDgpu solve with an independent dense balance:

```text
M = diag(RABSXS + row_sum(SP0)) - transpose(SP0)
q = solve(M, CHIT)
k = NSF · q
flux = q / sum(q)
```

It then compares k with `IMP_KEFF` and the sum-normalized group-integrated
spectrum with `INF_FLX`. This normalization is not flux per unit energy or
lethargy. The runner requires zero `TOT_LEAKRATE` and records B1 metadata.
`ANA_KEFF` (total component) and `INF_KINF` are informational references:
they need not equal the eigenvalue of the rounded imported production balance.

Acceptance requires convergence, dense k and spectrum errors ≤ `1e-8`, a
relative balance residual ≤ `1e-8`, Serpent k error ≤
`max(1e-5, 3 * IMP_KEFF absolute_error)`, spectrum maximum absolute error ≤
`1e-4`, and spectrum relative L2 error ≤ `5e-4`. Reported sigma units use
reference-only uncertainty. Normalized flux errors are deterministic checks;
the raw group uncertainties are retained without pretending to know their
correlations.

CPU solves precede GPU validation. `--device gpu --require-rocm` requires
actual HIP CuPy arrays and checks fusion disabled, then enabled. Unavailable
GPUs and failed checks produce reports and a nonzero exit status. JSON contains
source hashes, software versions, solver settings and individual checks; CSV
contains energy boundaries and normalized spectra. Solve timings are diagnostic,
not warmed performance benchmarks.

## Scope limits

This release does not interpolate temperature/density branches, process
depletion history, apply B1 leakage corrections, import discontinuity factors,
or provide kinetics parameters. It does not validate these constants for
SPN/SN, transient or thermal-feedback calculations. Higher scattering moments
may be retained as raw data, but have no transport adapter. Nonzero-leakage
Serpent references are rejected by the point-reactor runner. Extending the
interface to those cases requires independent reference calculations and
explicit conventions for each additional quantity.
