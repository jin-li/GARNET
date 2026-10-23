# Branch cross sections and interpolation

The branch-library interfaces are experimental static data tools. Numerical
tests establish interpolation and operator construction; reactor accuracy
requires independent withheld transport states. A single full-core branch
temperature represents a uniform perturbation, not arbitrary spatial thermal
feedback. No transient NEM or moving-mesh method is supplied here.

## Native, qualified and evaluated states

Keep unmodified native Serpent estimates, their printed errors and source
hashes separate from converted material data. Qualified table states additionally
identify the energy grid, ordered region identities, reaction convention,
diffusion prescription, face ordering, ADF definition and numerical reference
orders. Evaluated states carry the contributing state IDs and weights. These
identities must agree before any interpolation is meaningful.

`BranchLibrary` selects exact table states before interpolating. It uses
explicit segments or triangles; it neither triangulates across categorical
configurations automatically nor extrapolates outside supplied elements.
Fuel temperature can use the coordinate $u=\sqrt{T_f}$ and moderator density
can parameterize one declared fixed-pressure path. Density and moderator
temperature are not independent axes for that path. Inactive conditions must
match, and requests for another rod configuration or temperature family need
their own elements.

For barycentric weights $w_i\geq0$, $\sum_iw_i=1$, interpolate absolute data:

$$
q(\mathbf{x})=\sum_iw_iq_i.
$$

Convex interpolation preserves nonnegative reaction coefficients and positive
diffusion coefficients. It can create a reaction channel that is zero at one
endpoint. Removal is recomputed from reduced absorption and off-diagonal
outgoing production scattering. This preserves the adapter's production
balance convention; reduced absorption is not relabelled physical absorption.

Interpolate the fission matrix, with emitted group first:

$$
F_{gg'}=\chi_g\nu\Sigma_{f,g'},\qquad
F(\mathbf{x})=\sum_iw_iF_i,\qquad
\nu\Sigma_{f,g'}=\sum_gF_{gg'}.
$$

The interpolated matrix need not have rank one. Replacing it with the product
of independently interpolated spectra and production rates would define a
different operator. The NEM factory passes `fission_matrix` explicitly.
Power-production coefficients are interpolated as products, not as separately
interpolated energy-per-fission and fission XS.

## Face factors and fresh operators

For positive, qualified internal whole-face factors use

$$
f(\mathbf{x})=\exp\left(\sum_iw_i\log f_i\right).
$$

Exterior factors remain exactly one. Interpolation is a factor-transfer
approximation; positive values do not prove equivalence or transport accuracy.
The original matching reference prescription and its order/positivity checks
must first pass at each training state. Frozen table-factor qualification does
not qualify an interpolated state automatically.

The NEM factory constructs a new solver and sparse factorization for each
evaluated state. It never mutates a cached solver's XS. Geometry, physical
region-to-node mapping, boundary policy and polynomial degree remain explicit.
The branch API does not modify the older `TabulatedFeedback` hook, whose
clamping and multiplicative-update semantics are different.

## Reactivity, coefficients and rod worth

Use $\rho=1-1/k$. Signed changes and finite-interval coefficients are

$$
\Delta\rho_{A\to B}[\mathrm{pcm}]
=10^5\left(\frac1{k_A}-\frac1{k_B}\right),\qquad
\alpha_T=10^5\frac{\rho(T_+)-\rho(T_-)}{T_+-T_-}.
$$

For independent endpoint calculations with absolute printed standard errors
$s_{k,+}$ and $s_{k,-}$,

$$
s_\alpha=\frac{10^5}{T_+-T_-}
\sqrt{\left(\frac{s_{k,+}}{k_+^2}\right)^2+
      \left(\frac{s_{k,-}}{k_-^2}\right)^2}.
$$

A common seed does not establish covariance. Full-step and half-step
coefficients must be compared with statistical resolution visible. An
inconclusive comparison is not a pass, and finite-interval coefficients do
not by themselves establish a local derivative.

Rod libraries initially interpolate directly tallied, fixed-zone effective
XS as a function of the declared bank position. They do not volume-average
rodded/unrodded material data. The physical absorber tip and displaced water
must be correct at each training position. Solving the eigenproblem with
linearly interpolated XS does not make reactivity linear. Report interval
secants separately from pointwise derivatives; new samples are needed near
unresolved gradients or tip/interface crossings. Central withheld positions
do not qualify the outer intervals or the global derivative maximum.

## Independent validation

`garnet.branch_validation` distinguishes interpolation versus exact-state
GARNET, exact-state GARNET versus continuous-energy transport, and the full
interpolated prediction versus transport. A 20 pcm interpolation limit applies
to the first comparison, not to total model error. Power comparisons use
identical physical bins and explicit normalization/weights.

Withheld-state XS and factors are used only after prediction to construct the
exact-state comparison. A prediction that consumed those tallies is not
independent. Passing numerical fixture tests does not qualify a reactor
interpolation domain or replace the required external transport calculations.

## Thermal-star assembly and returned-data provenance

The high-level thermal builder creates four explicit triangles between the
nominal point and adjacent axis endpoints. The endpoints must bracket the
nominal actual temperatures and preserve the other thermal coordinate.
Each moderator density must reproduce the declared bounded benchmark path.
This guards against treating a temperature/density pair as independent axes
or admitting the uncomputed corners of a rectangular domain.

Training-role and evidence-policy checks precede construction. The evidence
policy belongs to the application: it must establish the sufficiency of local
ADF reconstruction, receiving-solver checks and compatible nuclear data.
The library stores policy references, while every interpolated state remains
unvalidated. Explicit diagnostic construction changes qualification labels,
not compatibility rules or XS values. Withheld data cannot enter the training
star through that mode.

Returned-data import records an ordered bijection of physical region IDs to
native universes, native completion/version/energy information, and hashes of
all caller-declared files before and after conversion. This detects ordinary
source changes during processing but provides neither a filesystem lock nor
an original-execution attestation. Unknown executable and ACE-content identities
remain explicit. Native printed errors and immutable arrays are retained;
input filenames and library labels do not prove nuclear-data equivalence.

The explicit NEM convergence controls distinguish absolute eigenvalue change,
normalized full spatial-source change and the physical residual. Linear and
auxiliary reconstruction tolerances are separate. A small change in eigenvalue
alone is insufficient to accept a branch prediction.

The receiving-solution assessor reconstructs the groupwise mean equation
$R\bar\phi+\nabla\cdot J-F\bar\phi/k=0$ using the full emitted/source
production matrix, including non-rank-one interpolated matrices. Interface
checks compare both corrected traces $f\phi_s$ and the stored shared trace;
exposed faces satisfy the declared zero-drive Robin condition. Independent
mean/interface/boundary calculations accompany fresh degree-specific solver
diagnostics. Nonnegative numerical profiles and positive integrated fission
production are necessary acceptance checks. This does not provide rigorous
interval bounds or establish Monte Carlo source convergence.
