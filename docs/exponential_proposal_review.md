# Review of the proposed ExponentialTheta method

Date: 2026-10-25. Follow-up to the
{download}`first prototype study <../benchmark-results/exponential-prototype/README.md>`.

Subsequent implementation checkpoints are linked from the
{download}`current development plan <exponential_transformation_plan.md>`. The
[26 October literature review](exponential_literature_review_2026-10-26.md)
compares the supplied PARCS/SIMULATE PDFs and distinguishes scalar frequency
transforms from the separate Krylov matrix-action feasibility track.

The temporal transformation is appropriate for FVM. Its performance benefit
remains a hypothesis. Most of the displayed algorithm is already the
implemented exponential backward-Euler method. Separate group frequencies,
within-step frequency correction and a genuine theta parameter are distinct
extensions. The earlier negative result applies to the shared-frequency,
first-order prototype, not all frequency-transformed methods.

## What matches the existing prototype

| Feature | Current GARNET | Extension in the proposal |
|---|---|---|
| BE, theta and BDF | Already available | Retain their baselines and public names. |
| Exponential flux transform | Physical endpoint formulation | The displayed equation is theta=1. |
| Frequency | One shared neutron-population rate from accepted history | One global rate per energy group. |
| Correction | No within-step frequency iteration | Optional nonlinear iteration; count every extra solve. |
| Precursors | Exact convolution for constant transformed endpoint production | Generally second-order integration needs linear transformed production. |
| Thermal coupling | Exponential source quadrature with BE conduction/split exchange; also implicit LRA | Group-dependent heat integration and compatible feedback accuracy. |

For frozen omega over width h, the quoted transformed BE derivative is
equivalent to the physical endpoint expression already implemented:

```text
(1/v_g) [(1 + omega_g h) phi_g_new - exp(omega_g h) phi_g_old] / h.
```

The exponential must be handled on both sides of the transformed equation.
With different group rates, transformed scattering/fission has factors
`exp((omega_source - omega_destination)*tau)`. Physical endpoint variables
avoid rewriting the leakage operator, but precursor-eliminated energy
coupling still changes.

Avoiding cellwise log ratios initially is sensible. Define each amplitude with
a positive global functional `A_g=<w_g,phi_g>` and normalize shape with that
same functional. State the weights: changed fission cross sections should not
create apparent growth just through the amplitude definition. Groupwise
amplitudes can capture changing spectra; they differ from IQS's single
amplitude and full multigroup shape.

## A genuine theta method

With `phi(t)=exp(Omega*tau) psi(t)` and frozen Omega, apply

```text
(psi_new-psi_old)/h = (1-theta) F_transformed(old)
                     + theta F_transformed(new),
F_transformed = exp(-Omega*tau) M^-1 R(phi,C,T) - Omega*psi,
M = diag(1/v_g).
```

Both the physical RHS and the `-Omega*psi` term need that weighting. The
displayed backward difference/new-time omega term is theta=1; the name
ExponentialTheta does not make it second order.

At theta=1/2, precursor production, changed cross sections and feedback need
compatible accuracy. A constant transformed endpoint precursor source can
retain a first-order error in general. Interpolate transformed **production**
when beta or fission cross sections vary, not just the flux.

For frozen coefficients and linear transformed flux, stable precursor weights
are available for each family j and source group g:

```text
z_jg = -(lambda_j + omega_g) h,
H_jg = h phi_1(z_jg), B_jg = h phi_2(z_jg),
A_jg = exp(omega_g h) (H_jg - B_jg),
phi_1(z)=(exp(z)-1)/z, phi_2(z)=(exp(z)-1-z)/z^2,
phi_1(0)=1, phi_2(0)=1/2.
```

Production then enters through `A_jg*phi_g_old + B_jg*phi_g_new`. Use the
appropriate production products when their coefficients change. Fitting an
exponential can reduce the error constant without increasing formal order.
CN also needs prompt-event and oscillation checks; its consistency does not
provide BE's stiff damping.

## Per-group rates require more than a diagonal change

For the current constant transformed-endpoint approximation,

```text
C_j_new = exp(-lambda_j h) C_j_old
          + beta_j sum_g H_jg (nuSigma_f,g/k0) phi_g_new.
```

With shared omega, H_jg is independent of g and the existing
`w_fis[destination] * S` factorization applies. With distinct rates, the
multiplier depends on the source group too. Changing only a destination-group
weight would implement the wrong Schur complement.

Keep existing leakage kernels and extend fission using its low-rank structure:
prompt production plus one weighted contraction per delayed family, rather
than a dense G by G materialization. Measure extra contractions and retain
device-resident buffers. The time-scheme interface also needs group-dependent
diagonal/history coefficients.

There is a useful cheaper special case. If prompt and all delayed spectra
coincide, the implicit emission still has rank one:

```text
b_g = 1 - sum_j beta_j + sum_j lambda_j beta_j H_jg,
S_effective = sum_g b_g (nuSigma_f,g/k0) phi_g_new,
emission_into_group_k = chi_k S_effective.
```

Keep these auxiliary source weights separate from the physical fission cross
sections used for power and precursor inventories. Family-specific precursor
updates still need their source-group weights, but not necessarily at every
Krylov application. This makes common-spectrum per-group FTM a practical
first extension; general spectra need the broader low-rank treatment above.

Per-group rates invalidate the single heat-source weight as well. Integrate
group contributions consistently with the driver's power normalization,
checking total energy and spatial shape. Constant-coefficient heat weights
contain `h*phi_1(-omega_g*h)` for each source group. Thermal losses and lagged
feedback retain their own error; neutron theta=1/2 alone does not make the
coupled march second order or permit arbitrary exchange widths.

## Error versus spatial work: additional GARNET comparison

The new test uses the existing FV/FGMRES solver, its CN scheme, and the same
smooth homogeneous ramp/independent Radau reference as the prototype study.
Smooth start-up avoids confusing temporal order with discontinuous-insertion
initial derivatives. Errors use the same common-grid linear interpolation.
These are single executions for **work counts**, not repeated timings.

Representative configurations within the same 0.1% power-error budget:

| Scheme | Width/control | Maximum power error | Block solves | PCG iterations |
|---|---|---:|---:|---:|
| BE | h=0.002 s | 0.057645% | 500 | 4,972 |
| CN | h=0.01 s | 0.059831% | 100 | 1,096 |
| Existing exponential BE | h=0.002 s | 0.067026% | 500 | 4,933 |
| Adaptive BDF1--5 | rtol=1e-5 | 0.031374% | 96 | 845 |

The [CSV](../benchmark-results/exponential-prototype/proposal-error-work.csv)
provides all curve samples. Switching the horizontal axis to solves does not
expose an advantage for the current transform here. CN's lower work motivates
a genuinely second-order exponential experiment.

GARNET is matrix-free: distinguish complete neutron block solves, FGMRES
iterations and group-PCG iterations. Include rejected attempts, frequency
correctors and thermal-feedback repetitions. In the original tighter prompt
step, BE used 2,000 block solves, existing FTM 4,000, and adaptive BDF 47.
The earlier finding was therefore not solely Python/GPU launch overhead.

## Frequency correction is not an accuracy guarantee

The optional loop is a legitimate experiment. Published
[SIMULATE5-K work](https://www.researchgate.net/publication/360764337_Development_and_Benchmarking_of_Transient_Nodal_Code_SIMULATE5-K_Neutron_Kinetics_Solver)
describes recursively improving frequencies during solution. Use iteration
limits/damping and a dimensionless convergence test such as
`h*max(abs(delta_omega)) < epsilon_log_amplitude`. Iterative convergence and
temporal error control are separate tests; charge all spatial/feedback solves.

In a homogeneous one-group problem, imposing
`omega=log(phi_new/phi_old)/h` makes the transformed BE difference vanish at
the fixed point. The neutron equation enforces the endpoint rate
`omega*phi_new=R_new/M`. This is still a first-order log-amplitude
discretization; it need not resolve mixed prompt/delayed modes better.

A separate one-family point-kinetics probe tests this literal loop with the
production scalar coefficients and a closed-form reference. At h=0.01 s
over 1 s:

| Method | Maximum endpoint population error | Linear subsolves |
|---|---:|---:|
| BE | 1.686930% | 100 |
| Lagged FTM | 1.868616% | 100 |
| Frequency-iterated FTM | 2.080243% | 247 |

This is an isolated ODE counterexample, **not a corrected spatial GARNET
implementation**; its endpoint metric differs from the common-grid metric
above. It rejects a guarantee that this correction improves accuracy, not all
possible frequency estimators. A cheap low-order frequency predictor could
avoid repeated spatial solves, but needs separate shape/feedback checks.

## Next experiment

Retain BE, theta/CN and adaptive BDF. Test true exponential theta=1/2 with
linear transformed-production precursor integration, initially sharing a rate
to isolate temporal order. One group already exhibits the present deficit,
so per-group frequencies alone cannot fix it. Then add global per-group rates
through the low-rank fission/heat coupling and test correction separately.
Keep exponential BE as a control.
Start groupwise source coupling with the common-spectrum case where possible,
then verify distinct prompt/delayed and family spectra separately.

Compare physical power/shape/energy/temperature errors against block-solve and
Krylov work as well as time. Include exponential tails, ramps, prompt events
and fully implicit LRA feedback. Recheck thermal cadence independently.
Production integrators were not changed for this review.

Reproduce the analysis with GARNET installed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tools/assess_exponential_proposal.py
```

The [JSON](../benchmark-results/exponential-prototype/proposal-review.json)
records source hashes, all CN/ramp results, correction controls and probe
parameters. Original prototype data/timings are preserved.
