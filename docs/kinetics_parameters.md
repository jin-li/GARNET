# Static kinetics parameters

The static kinetics interfaces distinguish three operations:

* importing whole-system Monte Carlo estimates;
* supplying **raw regional** production and inverse-speed data;
* calculating importance-weighted parameters using a matching forward/adjoint
  diffusion discretization.

A global effective delayed fraction is an output, never a regional input.
These interfaces do not start Serpent, infer missing yields, fit parameters,
or establish a three-dimensional transient NEM model.

## Existing Serpent output

```python
from ndgpu.serpent_kinetics import read_serpent_kinetics
mc = read_serpent_kinetics("input_res.m")
beta = mc.estimates["ADJ_IFP_IMP_BETA_EFF"]
print(beta.labels, beta.mean, beta.relative_error, mc.flags)
```

The reader accepts literal generated assignments only; it never executes
MATLAB. It preserves total/family beta and lambda entries and
total/prompt/delayed generation-time and lifetime entries, including native
relative errors. Analog, implicit and forward estimates remain distinct.
Whole-system fields repeated in GCU blocks must agree exactly and are not
averaged. Record counts are not independent sample counts. Write timestamps
are per-record metadata, not physics estimates. Missing fields, zero printed
errors and unresolved definitions remain explicit flags. `native_definitions`
records the qualified estimator definitions for exact version 2.2.2; other
versions retain qualification flags.

The official [calculation-options reference](https://serpent.vtt.fi/docs/user_guide/calculation_options.html#adjoint-weighted-point-kinetics-parameters)
and [output reference](https://serpent.vtt.fi/docs/user_guide/standard_output.html#reactor-time-constants)
describe the field ordering and default 15-generation IFP chain. Their current
editions postdate installed Serpent 2.2.2. A separate read-only audit of the
retained 2.2.2 implementation (`scoretimeconstants.c`, `scorefission.c`,
`score.c` and `collectresults.c`) established its native definitions:

* IFP implicit total lambda is the delayed-score-weighted arithmetic family
  decay mean. Analog IFP lambda uses its matching analog delayed-event/count
  weights. These are not the low-frequency reduction described below.
* Forward analog total lambda uses its delayed-emission statistical-score
  weights. Do not infer those weights from separately printed beta-zero
  family shares without further estimator qualification.
* Each total/prompt/delayed IFP generation-time entry is formed from its
  corresponding lifetime divided by the **analog cycle** eigenvalue.

These definitions apply at the native score/batch level. Ratios or products
of independently printed means need not reproduce the mean of a scored
ratio. In particular, do not replace the cycle eigenvalue by a printed
implicit eigenvalue and call the resulting identity exact. Definition
qualification also does not prove transport/diffusion estimator equivalence.
No proprietary source contents are distributed by this interface.

## Regional data exchange

`ndgpu.kinetics_data.read_regional_kinetics(path)` reads a supplemental JSON
contract. It does **not** extract unavailable family production from global
IFP results. Version 1 requires:

| Field | Meaning/shape |
|---|---|
| `schema_version` | integer 1 |
| `quantity_kind` | `raw_delayed_production` |
| `units` | energy `MeV`, inverse_velocity `s/cm`, production `1/cm`, decay `1/s` |
| `region_ids` | unique, stable regional string IDs |
| `family_ids` | unique, stable family string IDs |
| `energy_bounds` | exact descending nonnegative boundaries, length G+1 |
| `inverse_velocities` | positive (region,G), constant within each node/group |
| `decay` | positive (family,), common family system |
| `total_production` | (region,G_target,G_source) |
| `delayed_production` | (region,family,G_target,G_source) |
| `provenance` | named SHA256 hashes of sources |
| `approximations` | explicit list; empty only when none are used |

Every array must be finite. Production is nonnegative. The prompt remainder
`total_production - sum(delayed_production, axis=1)` must be nonnegative within
roundoff; no prompt or delayed spectrum is substituted or renormalized.
Family production matrices support incident-energy dependence and different
regional/family emission spectra. A scalar raw beta per region is a further
approximation, not the general data contract.

`data.for_nodes(region_ids, energy_bounds, total_production)` expands a declared
physical-node mapping, including repeated IDs for symmetry. It rejects unknown
regions, changed boundaries and a different frozen total-production library.
The returned dictionary supplies the projection arguments. Source hashes
provide provenance; they do not themselves establish physical qualification.

Serpent `INF_INVV` and prompt/delayed spectra are useful inventory fields.
`INF_CHID` is averaged over precursor families. Neither a global IFP beta nor
a regional score called `BETA_EFF` establishes raw regional family production.
The documented [microscopic group constants](https://serpent.vtt.fi/docs/user_guide/gc_generation.html#homogenised-microscopic-cross-sections)
can supply aggregate total/delayed nu-fission channels (MT452/455), but aggregate
delayed production alone does not identify family production matrices. Missing
data remain unavailable rather than being fabricated. Regional or isotope
dependent decay families need a separately qualified precursor model; this
interface requires a compatible common family system.

### Distinct isotope decay channels

A common channel system need not mean six shared decay constants. Retain each
nuclide/native-family pair as a distinct input channel, with its own positive
decay constant and regional production matrix. A channel absent from a region
has zero production there. The existing projection accepts any such channel
count; it does not require collapsing nuclides to one six-family decay vector.
Use unique IDs such as `U235/1` and `U238/1` in the supplemental input's
`family_ids`, retaining the nuclear-table identity in provenance.

For reporting under the library's native family labels, use an explicit map:

```python
from ndgpu.kinetics_families import summarize_delayed_channels
summary = summarize_delayed_channels(
    parameters.beta, parameters.decay,
    channel_ids=data.family_ids,
    reported_families=native_family_labels)
```

For a reported family i, `family_beta[i]` sums its channel beta values and
`family_decay[i]` is their beta-weighted arithmetic decay mean. No family
mapping is inferred by sorting lambda. A family with no delayed production
has decay `None`. The result retains every channel and marks whether each
reported family has exactly one physical decay pole. Disparate isotope
decays cannot be replaced by their arithmetic mean in a transient model
without an approximation. `lambda_low_frequency` consequently uses the
original channel values, preserving their sum(beta/lambda), rather than
first collapsing the channels. These are static reporting reductions;
they do not replace a missing raw production score.

## Matching NEM adjoint and physical projection

```python
from ndgpu.nem_kinetics import solve_nem_adjoint, project_nem_kinetics

forward = solver.solve_eigenvalue(rtol=1e-11)
adjoint = solve_nem_adjoint(solver, forward)
inputs = data.for_nodes(node_region_ids, energy_bounds, solver.fission)
parameters = project_nem_kinetics(solver, forward, adjoint, **inputs)
print(parameters.generation_time, parameters.beta, parameters.decay)
```

If only regional inverse velocities are qualified, `project_nem_time` computes
the physical time ratios without requiring or fabricating delayed-family data.
This is a time-only result, not a complete kinetics calculation.

This CPU NumPy/SciPy path supports the base quartic and higher-order NEM solver.
It retains the solver's exact geometry, whole-face factors, Robin boundaries,
transverse-leakage prescription, local responses and every retained source
moment. Cached operators reject material/geometry replacement. Rebuild the
solver for changed state data. An auxiliary reflector-reference source is
not included in core production or neutron population.

### Discrete derivation

Let `x` contain the uncondensed means, Legendre coefficients and face currents.
The verified physical eigenpencil is `A x = Q x / k`. Let `P` extract means and
all retained spatial moments, and let `E` inject their physical source terms
into the mean and weak moment equations. With frozen node-local production C,

```
y = P x
H = P A^-1 E
Q = E C P
T = H C
T y = k y
```

The complete source response is implemented without the large augmented LU.
Writing the condensed mean/current operator as Ac, the recovered moment maps
as MJ and Mq, the source-dependent face map as Fq, and face constraints as W,

```
rhs = [q_mean; -W Fq q_moment]
[mean; current] = Ac^-1 rhs
moment = MJ current + Mq q_moment
```

For a dual state w, reverse every path:

```
s = Ac^-T [w_mean; MJ^T w_moment]
H^T w = [s_mean; Mq^T w_moment - Fq^T W^T s_current]
```

The equilibrated matrix is `diag(row_scale) Ac diag(column_scale)`.
Its transpose solve is therefore
`row_scale * lu.solve(column_scale * rhs, trans="T")`.
Transposing Ac alone would miss the direct local source/recovery paths.

Solve `T^T w = k w`, and recover `z = H^T w`. The associated full physical
left vector is `ell = A^-T P^T w`; it satisfies
`A^T ell = Q^T ell / k`, with `E^T ell = z`.
Thus z is a **source-coordinate importance**, not an adjoint flux sampled at
node centers. Physical equation scaling and volumes are encoded in this dual;
do not multiply its contractions by volume a second time.

For node/group-constant inverse speed, mass is `B = E V^-1 P`. The two highest
Legendre coefficients are orthogonal to the retained weak tests and hence do
not contribute separately to this mass. For delayed family i,
`Q_i = E C_i P`. The physical static parameters are

```
Lambda = (z^T V^-1 y) / (z^T C y)
beta_eff_i = (z^T C_i y) / (z^T C y)
lifetime = k * Lambda
```

These equal the augmented-pencil bilinear forms exactly for this frozen
discretization. They do not prove equivalence to an unqualified Monte Carlo
estimator or a general transient closure. Arbitrary forward/adjoint
normalization cancels; scale-safe validation checks even extreme finite
normalizations. Positive fission/time importance and nonnegative
projected fractions are checked rather than repaired.

Keep the complete decay vector. `lambda_low_frequency` is explicitly the
one-family reduction `sum(beta_i)/sum(beta_i/lambda_i)` and is `None` if there
is no delayed production. It is not claimed to be Serpent's scalar lambda.

## Standard diffusion baseline

`ndgpu.diffusion_kinetics` provides a separately labeled physical static
projection using the existing standard diffusion forward/adjoint operators.
It is not a matching NEM/ADF calculation. The existing quasi-static
`project_effective_kinetics` retains its transient critical adjustment:
its `generation_time = k0*T/F`. The physical static Lambda is `T/F`.
The new static interfaces do not change that existing transient API.

The baseline requires the standard symmetric Cartesian or cylindrical FV
model, verifies complete forward/adjoint eigen-equations and k agreement,
and retains the existing `Kinetics` assumptions: material/global raw beta
independent of incident energy, common decay families and common/per-family
delayed spectra. Blended cells retain the existing flat-flux production
weighting. An inconsistent negative prompt-production remainder is rejected.
It is not a general importer for regional family-dependent production matrices.

## Verification and limits

`tests/test_nem_kinetics.py` checks signed H/H^T dot products, direct response
columns, all source moments, unequal nodes/ADFs, physical independent
uncondensed pencils and quadrature mass/delayed matrices at degrees 4/8/16,
both leakage policies, eigenvalue/residual agreement, normalization and
equation scaling, off-critical one-group limits, and an independent alpha
perturbation derivative `k'(0)=-k^2 Lambda`. Regional-data tests cover explicit
mapping, energy/XS compatibility, zero-production reflectors and rejection of
effective-input substitutes. The global reader has separate malformed-input,
repetition, metadata and missing-data tests.

Small-problem verification is not PRATIC validation. State tables require
qualified regional data, compatible family definitions, convergence evidence
and exact input/software provenance. A missing-data inventory remains useful
but cannot be presented as an NDgpu calculation of beta.

## Isotope production and an explicit total kernel

`serpent_mdep.read_serpent_mdep` imports literal native micro-depletion tables
without executing MATLAB. Supply exact selectors, group boundaries and energy
order. `production_xs` requires explicit isotope coverage and paired MT452/455
rows; smeared density times microscopic production gives central macroscopic
production. Native density/XS errors remain separate: their covariance is not
available. `build_delayed_channels` expands qualified constant isotope branches
into independent isotope/family matrices. It does not use global effective beta.

`ace_precursors` and `ace_delayed_spectra` read explicitly selected legacy ASCII
ACE tables, retaining source hashes, native family decay rates and probability
metadata. Qualify incident-energy independence before using constant branches.
Only supported LAW4 histogram/linear PDFs are integrated; out-of-domain spectra
are rejected. Explicit probability normalization is recorded by the caller;
PDF integration need not reproduce the native stored-CDF sampler bit for bit.
These readers alone cannot supply heterogeneous-core isotope production weights.

`kinetics_production.assemble_kinetics_production` constructs a new positive
prompt-plus-delayed total production kernel from explicit inputs:

```
Cp[target,source] = chip[target] * (nuSigmaF[source] - sum_i,target Cd_i)
C = Cp + sum_i Cd_i
```

It rejects negative remainders and unnormalized prompt spectra. Pass `C` as
`fission_matrix=` to either NEM solver; emitted/source column sums must match
the supplied nuSigmaF. Fission acts on all retained spatial moments, and the
matching adjoint/projection uses this same matrix. The default remains the
original `chi_total outer nuSigmaF` kernel.

This is a distinct production model, not a repair of a frozen sampled total
spectrum. Sampled total chi may print zero where library delayed spectra have
small nonzero tails, so subtracting these channels from that frozen kernel can
fail positivity correctly. Do not clip those tails, weaken the positivity gate,
or claim identical keff/ADF qualification. A full-core calculation with the new
kernel still needs a matching homogeneous reference and explicitly qualified
factor reconstruction. Prior study results and their source hashes are historical
results under their original kernel/software version.

`nem_reference.reconstruct_nem_reference` supplies the matching local reference
using the receiver's exact `R-C/k`, grid, degree and transverse-leakage policy.
Supply positive native means and either shared-face global-axis current density
or native region-side integrated net-inward currents. The latter preserves all
side tallies and reports shared-face disagreements without averaging. Exterior
reference currents remain prescribed; vacuum Robin conditions belong to the
receiving core. Negative reference profiles remain visible in the returned
diagnostics. The reported auxiliary mean source closes prescribed reference
balance only and must never enter core production or kinetics projection.

`qualify_nem_reference_factors` applies the existing reflector-inclusive local
prescription with three increasing reference orders (default12/14/16) ending
at the receiver's order. Both successive face changes must pass the default
0.005 native-mean-normalized limit. Final fuel profiles must be positive;
reflector internal traces require conservative earlier-order positivity and
factor-change bounds. Exterior factors stay unity. Rebuild a new receiving
solver with the returned factors and identical production matrix, then solve
and validate its forward/adjoint states. Passing local reconstruction does not
qualify the receiving core, transport data or independent prediction.

`kinetics_yield_transfer.transfer_delayed_yields` is an explicit modeling step
for homologous regions scored by different estimators, such as a physical D4
orbit and its representative virtual GCU. It preserves source raw delayed
fractions/spectra while multiplying by receiver incident-group total yields.
Supply both domain identities and a qualification statement; independently
verify region/energy compatibility before using it. Source/receiver total-yield
differences are descriptive metadata, not uncertainties or an acceptance gate.
This does not extract exact receiver-domain yields or infer global beta.
Zero source production cannot supply a positive receiver-production channel.

For an executable synthetic analytical example, run
`python examples/kinetics/static_nem.py` from the repository root. It needs no
Monte Carlo data and prints both physical Lambda and the discrete k*Lambda
lifetime for unequal reflected homogeneous nodes.
