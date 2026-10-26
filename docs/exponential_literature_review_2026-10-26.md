# Exponential integration literature review: PARCS, SIMULATE5-K and Krylov actions

26 October 2026. Reviewed the five PDFs in `~/Downloads/for_GARNET` against
the current working-tree implementation, including the
[physical-state adaptive checkpoint](../benchmark-results/exponential-second-order/CHUNK4.md).
The [inventory](qa/exponential-literature-inventory-2026-10-26.json) records
filenames, page counts, hashes and the sections used. Original PDFs remain
outside the repository. Page references below are PDF page numbers.

**Recommendation:** retain shared-frequency FTM2/transformed BDF2 as the
current experimental path. First address output accuracy and spatial
benchmarks, then test a PARCS-style physical-source precursor control before
expanding frequencies. Add a separate, bounded matrix-exponential-action
study: KIOPS is a credible candidate for that study, but it is neither a
replacement for the scalar coefficient functions nor an established GARNET
speedup. Rational Krylov deserves a later gate if polynomial Krylov struggles
with diffusion stiffness.

## 1. Relevance of the five documents

| Document | Relevance and action |
|---|---|
| **PARCS v2.6 Theory Manual**, Downar, Lee, Xu, Kozlowski and Staudenmier, draft November 2004; saved as `Nuclear_reactor_code_for_core_nerutronic.pdf` (107 pages) | Highest direct relevance to second-order FTM. Section 2.2, pp. 7–10, derives transformed theta integration and quadratic physical fission-source precursor integration. Section 3.2, pp. 12–18, concerns a Krylov **linear solver**. Section 8.2, pp. 99–104, explains transient updates, nodewise inverse periods and feedback coordination. Treat this as the supplied historical draft, not evidence of current PARCS internals. |
| **Development and Benchmarking of Transient Nodal Code SIMULATE5-K Neutron Kinetics Solver**, Bahadir, PHYSOR 2022 (10 pages), [DOI](https://doi.org/10.13182/PHYSOR22-37639) | Highest direct relevance to GARNET's original `ftm`. Distinguishes dynamic frequencies for nodal spatial coupling from frequencies for temporal integration. Provides groupwise precursor convolution and recursively refined flux frequencies. Its transformed neutron derivative is backward Euler. |
| **KIOPS: A fast adaptive Krylov subspace solver for exponential integrators**, Gaudreault, Rainwater and Tokman, JCP 372 (2018), 236–255 (23-page supplied PDF including update), [paper](https://doi.org/10.1016/j.jcp.2018.06.026), [author preprint](https://arxiv.org/abs/1804.05126) | Most relevant matrix-function implementation reference. Computes combinations of matrix phi-function actions using an augmented operator, incomplete orthogonalization and adaptive internal work. The supplied PDF includes the [2021 corrigendum](https://doi.org/10.1016/j.jcp.2021.110443): corrected Eqs. (18)/(19) and Algorithm 1 line 13 must be used. |
| **On the use of exponential time integration methods in atmospheric models**, Clancy and Pudykiewicz, Tellus A 65 (2013), 20898 (16 pages), [article](https://www.tandfonline.com/doi/full/10.3402/tellusa.v65i0.20898) | Strong methodological and cost reference, with EPI2/EPI3, Jacobian actions and `phipm`. Particularly useful because longer accurate steps did **not** beat its best semi-implicit comparator in runtime. Reactor-specific performance must be measured separately. |
| **Exploring the Exponential Integrators with Krylov Subspace Algorithms for Nonlinear Circuit Simulation**, Wang, Zhuang and Cheng, ICCAD 2017, 163–168 (6 pages), [author PDF](https://cseweb.ucsd.edu/~hazhuang/papers/iccad17_zhuang.pdf) | Relevant to rational/shift-and-invert Krylov, algebraically equivalent phi-function formulations, singular mass matrices and nonlinear correction. Its LU-based circuit implementation is an architectural comparison, not a directly reusable GPU diffusion solver. |

## 2. PARCS versus GARNET

Write `M*dphi/dt = R(phi,C,t)`, with `M_g=1/v_g`, step width `h`, and one
frozen frequency `omega` during that step. PARCS Eq. (2.22), p. 8, gives
the physical endpoint equation

```text
[M/(theta*h) + M*omega] phi_new - R_new
 = exp(omega*h) * ([M/(theta*h) - ((1-theta)/theta)*M*omega] phi_old
                  + ((1-theta)/theta)*R_old).
```

At `theta=1/2`, this is the neutron formula implemented by
[`ExponentialTheta`](../garnet/timescheme.py):

```text
[M*(2/h+omega)] phi_new - R_new
 = exp(omega*h) * [M*(2/h-omega)*phi_old + R_old].
```

This is direct support for GARNET's transformed theta approach. The old
physical derivative must be interpreted in the current step's frozen frame;
GARNET carries a physical derivative rather than carrying a derivative in an
obsolete exponential frame. The printed PARCS equation was checked visually:
the frequency multiplying the old flux is the current interval's frequency,
despite the adjacent old-flux time index.

The precursor formulas differ substantially. PARCS Eqs. (2.23)–(2.27), p. 9,
analytically integrate a **quadratic physical fission source** interpolated
through `S_(n-2), S_(n-1), S_n`, including unequal widths. GARNET `ftm2`
instead integrates **linear transformed production** through two endpoints:

```text
C_new = exp(-lambda*h)*C_old + A*(beta*S)_old + B*(beta*S)_new,
```

where `A` and `B` depend on `lambda+omega`. It retains old/new production
products separately, including mapped beta. Thus `ftm2` is a shared-frequency
relative of PARCS's neutron discretization, not a reproduction of the full
PARCS temporal algorithm. `ftm-bdf2` is a further GARNET experiment: it applies
a transformed BDF recurrence to neutrons and precursors together. That
recurrence is not the scheme described by this manual.

PARCS uses node/group inverse periods; GARNET currently uses one rate from
the accepted total neutron population. PARCS Eq. (2.30) then closes the
balance equations with CMFD/nodal current coefficients. GARNET directly
discretizes diffusion with finite volumes. Its mapped material kinetics,
arbitrary group/family counts and distinct delayed spectra also exceed the
simplifications used in the manual's two-group derivation. A published
coarse nodal time step is consequently not an accuracy target for a different
FV spatial model.

**Useful controlled experiment:** add physical quadratic-source precursor
integration as an optional research comparator, initially at shared frequency
on identical meshes. This isolates production interpolation from the
frequency model and neutron derivative. Stable moment formulas, startup,
variable beta, events and negative interpolation weights need explicit tests;
a quadratic interpolant does not guarantee positivity.

There is also a useful architectural consequence. PARCS-style physical-source
interpolation retains one scalar endpoint fission source even with
node/group-dependent **neutron** frequencies. This offers a simpler frequency
experiment than SIMULATE-style source-group exponential precursor weights.
It is not evidence that it will retain the same exponential-tail accuracy.

## 3. SIMULATE5-K versus GARNET

The paper describes two distinct uses of frequencies:

1. **Spatial nodal construction**, Section 2.1, pp. 2–3: neutron and precursor
   dynamic frequencies produce pseudo-static cross sections/spectra. The
   analytic nodal method then computes leakage coupling coefficients.
2. **Temporal integration**, Section 2.2, pp. 3–5: transformed neutron fluxes
   use an implicit first difference, while precursors are analytically
   integrated with exponential source-group weights. Flux frequencies are
   initially lagged and recursively improved during the iterative solve.

The temporal neutron formula is closest to GARNET's original
[`FrequencyTransform`](../garnet/timescheme.py), `time_scheme="ftm"`, rather
than to either second-order extension. The paper does not claim transformed
trapezoidal or BDF2 neutrons. Its Eq. (13) has the precursor source term

```text
sum_g beta_i * [1-exp(-(lambda_i+omega_g)*h)]/(lambda_i+omega_g)
             * nuSigma_f,g * phi_g,new.
```

GARNET's shared-frequency `ftm` is the corresponding common-rate special
case, with its own critical-adjustment and total-spectrum conventions.
In particular, the paper's `chi` is a prompt spectrum, whereas GARNET stores
the total spectrum. Preserve `chi_total-sum_i beta_i*chi_delayed_i` when
translating the prompt term; `(1-beta)*chi_total` is generally different.
With distinct source-group frequencies, this expression generally cannot be
implemented by merely changing destination-group emission weights. The
family/source-group contractions must be retained. GARNET's existing
low-rank fission treatment is a useful starting point, not already that
generalized coupling.

The paper's Eq. (12), as printed, uses a fixed `Delta t` where a running-time
exponential would be expected. The following convolution Eq. (13) supplies
the intended exponential-source dependence. An implementation should derive
and independently verify the coefficients rather than transcribe that
intermediate expression literally.

GARNET's optional frequency corrector repeats a converged monolithic
endpoint solve. The paper describes refinement during iteration but does
not specify enough update rules to equate its cost, convergence or safeguards
with GARNET's loop. Moving fitting inside existing iterations is worth a
separate measured experiment; frequency convergence alone is not an error
estimate.

The pseudo-static precursor frequencies used to construct nodal leakage are
**not** a prescription to transform GARNET's heat equation or replace its
full transient delayed-source elimination with a dynamic equilibrium spectrum.
GARNET has no equivalent analytic nodal leakage construction in this FV
path. The portable ideas are temporal coefficients, frequency estimates,
conditional reuse and benchmark choices.

The paper's TWIGL/LMW cases are neutronic tests; its LRA cases include simple
adiabatic heat-up and Doppler feedback (pp. 8–9). Calling the whole paper
feedback-free would be incorrect. It does not establish performance for
GARNET's conduction-coupled controller or a general coupled second-order
method.

## 4. Three different meanings of Krylov in this discussion

| Use | Mathematical task | GARNET status / implication |
|---|---|---|
| PARCS CMFD linear solution | Solve `A_step*x=b` using BiCGSTAB plus BILU3D | GARNET already uses Krylov solvers: group PCG and optional monolithic FGMRES with energy-sweep preconditioning. This paper does not introduce matrix-exponential time integration. |
| Polynomial exponential action, including KIOPS | Approximate `exp(h*J)*v` or a sum of `phi_k(h*J)*b_k` | A genuinely different integration path. Reuses stencil products and vector kernels, but needs a full physical neutron/precursor generator and an action solver. |
| Rational exponential action, circuit paper | Build a basis using shifted inverse actions such as `(I-gamma*J)^-1*v`, then evaluate a small projected exponential | Potentially useful for very stiff diffusion, at the cost of shifted coupled solves. Existing implicit machinery could help implement these actions; ordinary preconditioner applications are not exact shifted solves. |

GARNET's current [`MultigroupStepOperator`](../garnet/multigroup.py) already
contains the time shift and analytically eliminated precursors. It is **not**
the generator `J` to exponentiate. Likewise, replacing its scalar
`phi1`/`phi2` coefficient evaluations with KIOPS would solve the wrong problem.

PARCS's BiCGSTAB choice also does not by itself justify replacing monolithic
FGMRES. GARNET's inner PCG stopping and iteration counts make its sweep
preconditioner variable; [`fgmres`](../garnet/linalg.py) explicitly supports
that behavior. A standard BiCGSTAB comparison needs a suitable fixed linear
preconditioner or a separately justified flexible variant.

## 5. Where matrix-exponential actions could help

For frozen linear physics, retain the physical state
`y=[phi_1,...,phi_G,C_1,...,C_I]` and construct

```text
M*y' = A*y,       J = M^-1*A,
y(t+h) = exp(h*J)*y(t).
```

`M` contains inverse neutron speeds and unit precursor masses. `A` contains
physical leakage/loss, scattering, prompt fission and all delayed
production/decay. In GARNET's convention the prompt emission vector is
`chi_total - sum_i beta_i*chi_delayed_i`, with fission divided by `k0`.
Mapped velocities/beta, cylindrical equation weights and masks must be
handled before forming the physical generator action.

This can propagate several spatial, spectral and precursor modes together,
where one fitted population frequency captures only a common exponential
factor. For an autonomous linear problem it has no temporal discretization
error beyond the approximation of the exponential action. This is the
strongest reason to run a small feasibility study. It is an inference about
GARNET's opportunity, not a measured result from these papers.

For nonlinear autonomous physics, an EPI/exponential Rosenbrock method may
use `y_new=y+h*phi1(h*J_n)*F(y)` plus appropriate remainder terms. For
explicit time dependence, simply freezing the start-time operator does not
establish second order: use a correctly derived nonautonomous method or a
midpoint exponential for smooth prescribed `A(t)`. Material jumps still need
aligned endpoints, and feedback variation still needs physical error tests.

The continuous matrix phi-functions remain finite at a zero eigenvalue.
Critical reactor generators are singular, so expressions such as
`J^-1*(exp(h*J)-I)` are unsuitable as numerical definitions. Augmented-action
methods avoid requiring this inverse. GARNET's positive diagonal mass matrix
on active cells is also a different issue from the circuit paper's singular
capacitance matrix.

### Why start with KIOPS

KIOPS combines phi-actions in an augmented operator without assembling the
large augmented matrix. Its length-two incomplete orthogonalization reduces
the orthogonalization work from quadratic to linear in basis size, while
adapting internal subintervals and basis dimension. This fits matrix-free
neutron stencils and could reduce reduction/synchronization costs. The
author [implementation repository](https://gitlab.com/stephane.gaudreault/kiops)
is a reference to inspect, not an existing GARNET/CuPy backend.

Its tolerance controls the **matrix-function approximation**, not the
reactor's outer time-discretization, frequency error or feedback coupling.
The paper's posterior estimate is itself an approximation (Section 3.4),
and a theorem for negative semidefinite operators does not establish the
same guarantee for a nonsymmetric, growing reactor generator. Compare
full-orthogonalization and incomplete-orthogonalization actions against dense
references, in physical block scales, before trusting either estimate.

Incomplete orthogonalization still stores the basis for reconstruction. A
30-vector FP64 basis with one million cells, 11 neutron groups and six
precursor families occupies about **4.08 GB**, before the extra basis vector,
state, operators and scratch. This is a derived memory estimate, not a
measured GARNET allocation. Matrix-free operation does not eliminate this
cost. Current FGMRES GPU vector kernels can inform implementation, but its
flexible-preconditioned recurrence cannot simply become an exponential-action
routine.

### Why rational Krylov remains conditional

Diffusion can produce a broad, stiff negative spectrum as the mesh is refined.
Polynomial Krylov may then need large spaces or many internal substeps.
Shift-and-invert can concentrate approximation effort on the slower modes,
and the shifted systems resemble GARNET's implicit steps. That makes the
circuit paper useful even though its sparse LU factorization is a poor
production fit for GARNET's matrix-free GPU design.

Each rational basis vector can require a coupled spatial solve. Inexact
shifted solves need controlled accuracy, setup reuse and work accounting.
Nested FGMRES storage adds to the exponential basis storage. Arbitrarily
left-preconditioning `J` changes `exp(h*J)`; a preconditioner cannot be
inserted as if an exponential action were an ordinary linear solve.

The circuit paper's preference for a phi0 representation on its test cases
is not a general assertion that phi1/phi2 formulations are inferior, nor does
the phi-function index determine the temporal order. Those comparisons
should guide experiments rather than select GARNET's method in advance.

### Evidence against assuming a speedup

Clancy/Pudykiewicz's Table 2, p. 13, reports the semi-implicit method faster
than their EPI methods despite much longer EPI steps. Table 3 shows why:
for EPI2, increasing the physical step from one to two hours increased the
average basis size from 34.6 to 59.5 and `phipm` time per step from 6.3 to
11.8 seconds. These are historical shallow-water/MATLAB measurements, not
reactor predictions. KIOPS addresses part of that action cost, but its
benchmarks against `phipm` do not prove an advantage over adaptive BDF.

## 6. Development decisions

The [revised development plan](exponential_transformation_plan.md) now uses
the following order:

1. **Output accuracy first:** diagnose/control interpolation between long
   exponential steps; retain the physical step-doubling controller as an
   independent reference for cheaper error estimators.
2. **Resolved spatial comparison and reusable setup:** TWIGL/localized
   multigroup controls, fixed/shared meshes as well as adaptive matched-error
   runs, and diagonal-only operator updates with explicit invalidation.
3. **PARCS precursor control:** quadratic physical production versus existing
   linear transformed production and BDF recurrence. Only then test limited
   regional/group frequency extensions, keeping the PARCS and SIMULATE
   precursor contracts distinct.
4. **Separate small Krylov-action gate now:** validate a complete physical
   generator and corrected augmented phi-actions on dense/small FV problems.
   Compare conventional Arnoldi, KIOPS-style incomplete orthogonalization and
   an independent exponential-action reference. A full-scale integrator is
   contingent on this gate.
5. **Conditional rational/GPU expansion:** proceed if polynomial work or
   memory limits identify a concrete need and shifted solves are competitive.

The action study must separate temporal, matrix-function, spatial and
interpolation errors and include output, power/shape, energy and precursor
history diagnostics. Use [`scipy.linalg.expm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html)
on tiny assembled systems and
[`scipy.sparse.linalg.expm_multiply`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html)
as independent references where appropriate. The installed SciPy 1.16.3
`expm_multiply` uses a Taylor-action algorithm; it must not be labeled a KIOPS
or Krylov benchmark.

Promotion still requires matched physical error against the fastest
acceptable existing full-diffusion method, including adaptive BDF. Report
all matvecs, shifted solves, orthogonalization/reductions, rejected/internal
steps, rebuilds, basis memory and synchronized runtime. No timing from the
five papers is converted into a GARNET speedup estimate. The literature
assessment adds no new production time integrator or thermal transform.
