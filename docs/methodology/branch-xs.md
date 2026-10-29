# Branch cross sections and interpolation

The branch-library interfaces are experimental static data tools. Numerical
tests establish interpolation and operator construction; reactor accuracy
requires independent withheld transport states. A single full-core branch
temperature represents a uniform perturbation, not arbitrary spatial thermal
feedback. No transient NEM or moving-mesh method is supplied here. Native NEM branch states use three
Cartesian axes and six whole-node faces in the order `x-, y-, x+, y+, z-, z+`.
Generic table interpolation does not add support for arbitrary spatial topology.

## Native, qualified and evaluated states

Keep unmodified native Serpent estimates, their printed errors and source
hashes separate from converted material data. Qualified table states additionally
identify the energy grid, ordered region identities, reaction convention,
diffusion prescription, face ordering, ADF definition and numerical reference
orders. Evaluated states carry the contributing state IDs and weights. These
identities must agree before any interpolation is meaningful.

Serpent version compatibility is strict unless the application explicitly
supplies `compatible_serpent_versions`, a tuple/list containing at least two
distinct nonblank exact version strings. Mixed source versions must all be
declared members. This persisted policy expresses an application assumption,
not verified version equivalence or permission to mix nuclear-data routes.
It does not waive geometry, scoring, nuclear-data or closure identity checks.
A version label difference alone does not mandate another Monte Carlo run;
input/data consistency and available evidence govern the application decision.
Original native and state versions remain intact. Every evaluated state
records contributor versions and the declared policy, and a mixed-version
prediction has no single `serpent_version` value. Older library files without
a policy retain strict comparison.

`BranchLibrary` selects exact table states before interpolating. It uses
explicit segments or triangles; it neither triangulates across categorical
configurations automatically nor extrapolates outside supplied elements.
Fuel temperature can use the coordinate $u=\sqrt{T_f}$ and moderator density
can parameterize one declared temperature/density path. Density and moderator
temperature are not independent axes for that path. Inactive conditions must
match, and requests for another rod configuration or temperature family need
their own elements.

`TabulatedModeratorPath` accepts application-owned positive finite knots:
temperature in K must increase, and density in g/cm3 must be strictly monotone
in either direction. Bounded PCHIP preserves that shape without extrapolating.
A caller-defined `path_id` identifies the tabulation and must match source
metadata for thermal-star construction. Optional positive `pressure_bar` is
provenance, not an equation-of-state parameter. The historical
`BenchmarkModeratorPath` wrapper retains the original three PRATIC knots and
155-bar restriction; generic thermal construction has no such pressure or
energy-group-count requirement.

Legacy compatibility identity names and their original presence rules are
centralized in `garnet.branch_xs`: closure declarations distinguish omitted
from null, while other legacy identities retain value comparison using a
missing default of null. The `compatibility_identities` metadata namespace is
an extensible caller-owned mapping compared in full. Extra keys, omitted keys,
and explicit null values therefore affect compatibility, including inside
nested mappings. Absence of the namespace differs from an empty mapping.
Serialization preserves both these declarations and the separate exact-string
Serpent-version policy; permitting versions never waives any identity check.

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

An ADF is tied to its spatial-reference closure: the transverse-leakage
prescription, reflector mask, leakage boundary treatment and reference-current
convention must accompany it and agree with the receiving calculation.
Record these as `transverse_leakage_identity`, `reflector_mask_sha256`,
`leakage_boundary` and `reference_current_convention`. `BranchLibrary` rejects
elements with unequal declarations or a declaration missing at another
vertex, including explicit null versus omission. If every vertex omits a key,
legacy numerical behavior remains available; agreement by omission supplies
no qualification evidence. These are metadata compatibility checks, not a
verification of the actual mask, reference reconstruction or receiver closure.

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

Coefficient accuracy compares the predicted secant with the continuous-energy
(CE) or other reference secant over identical physical endpoints, with the
same fixed conditions and density path. Its relative tolerance is caller
policy against that stated reference. Pointwise eigenvalue/power interpolation
against an exact reconstructed solver and full-step/half-step convergence are
separate checks; neither substitutes for same-interval coefficient accuracy.

Nominal relative agreement does not establish that reference precision resolves
the requested tolerance. Treating the prediction as deterministic while
propagating native Monte Carlo endpoint errors gives only a conditional
uncertainty screen: it excludes training-XS and ADF uncertainty. Retain an
inconclusive precision assessment even when the nominal error meets the
caller target; this screen does not establish application qualification.

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
interpolated prediction versus transport. The historical compatibility defaults
apply 20 pcm to the first comparison and 0.01 fractional RMS to its power
comparison, not to total model error. These are application policies, not
universal reactor accuracy criteria. `compare_reconstruction_chain` exposes
`target_pcm` and `power_target` independently. Setting either to `None` leaves
that metric descriptive; any requested unassessed metric prevents an overall
pass, while a failure in an assessed metric still fails. Setting both to `None`
is the metric-only route. Transport comparisons carry no inherited gate.
Power comparisons use identical physical bins and explicit normalization/weights.
`normalized_power_rms` also accepts `target=None`; coefficient convergence and
rod-worth assessment accept `relative_target=None`. The latter still reports
missing intervals and peak resolution. No absent threshold accepts a result,
and all physical and uncertainty-input validation still applies.

Withheld-state XS and factors are used only after prediction to construct the
exact-state comparison. A prediction that consumed those tallies is not
independent. Passing numerical fixture tests does not qualify a reactor
interpolation domain or replace the required external transport calculations.

Cases inspected while selecting a spatial closure, reflector mask or factor
method belong to method development. Blind validation requires those choices
and the interpolation domain to be frozen before inspecting the held-out
transport results. Exact-state versus continuous-energy error measures the
diffusion model, closure and factor-transfer accuracy at a supplied state;
interpolation versus exact-state error measures the branch approximation
within that frozen method. Small interpolation error can coexist with large
exact-state model error. Report both and the total prediction error, retaining
the same closure and mask with the factors throughout each comparison.

## Thermal-star assembly and returned-data provenance

The high-level thermal builder creates four explicit triangles between the
nominal point and adjacent axis endpoints. The endpoints must bracket the
nominal actual temperatures and preserve the other thermal coordinate.
Each moderator density must reproduce the declared bounded tabulated path.
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

The `build_thermal_star` constructor also accepts `compatible_serpent_versions`
and forwards that policy to its `BranchLibrary`. This permits an explicitly
accepted executable-version pair without waiving training roles, nuclear-data
identity or the caller-supplied qualification policy.

## Experimental node-local rod occupancy weights

The optional `garnet.branch_rod.evaluate_rod_occupancy` evaluator applies only
to an explicit one-dimensional segment in an identity-transformed rod
coordinate. It first uses `BranchLibrary.evaluate` to enforce the existing
domain and fixed-condition contract and recover the contributing source IDs.
Exact anchors retain the ordinary exact-state result without invoking geometry.
For an interpolated segment with global right weight $w$, a caller-supplied
geometry function provides the absorber fractions $o_{L,n}$, $o_{R,n}$ and
$o_{t,n}$ for each physical region $n$. Its local right weight is

$$
w_n = \begin{cases}
\dfrac{o_{t,n}-o_{L,n}}{o_{R,n}-o_{L,n}}, & o_{R,n}\ne o_{L,n},\\
w, & o_{R,n}=o_{L,n}.
\end{cases}
$$

Every occupancy must be finite and within $[0,1]$; target occupancy must also
lie within the closed interval spanned by its endpoint occupancies. This
includes equality when endpoints coincide. Reversed segment orientation and
decreasing occupancy are supported by the same expression. The method rejects
nonbracketed geometry instead of modifying its values. The geometry and its
coordinate units are application responsibilities; the module contains no
reactor-specific dimensions or region identities.

For each region, the same $w_n$ is applied to $D$, absorption, scattering, the
full emitted/source fission matrix, the independent spectrum and optional
kappa-fission/total channels:

$$ X_{t,n}=(1-w_n)X_{L,n}+w_n X_{R,n}. $$

Face factors use

$$ f_{t,n,s,g}=\exp\big((1-w_n)\log f_{L,n,s,g}
                         +w_n\log f_{R,n,s,g}\big). $$

The spectrum is normalized by the same rule as the ordinary library evaluator;
exterior face factors are set to exact unity. Nonnegative XS and positive
diffusion/factors follow from convex interpolation of compatible sources.
Regions with unchanged occupancy use the original global interpolation,
including core-induced changes in their homogenized properties. If every
region has unchanged occupancy, this reduces to the ordinary segment algebra.
This is an interpolation prescription, not a proof of conserved transport
reaction rates, preservation of a heterogeneous flux, or qualified rod worth.

Returned global contributor weights remain lineage, while immutable metadata
records the separate local weights, occupancies, source IDs/roles and explicit
geometry identity. The experimental result has no native transport record and
receives no source qualification. Original states and their roles are retained
unchanged; interpolated metadata is explicitly labelled `role="prediction"`. The geometry callback is deliberately excluded from library
serialization: archive its implementation and geometry provenance separately
with the source library and prediction snapshot. Existing development cases
must remain identified as such; introducing this evaluator does not turn a
retrospective comparison into blind validation. No PRATIC, full-rod or
differential-worth qualification is implied by this software interface.


## Native micro-group processing identity

The optional native `MICRO_NG`/`MICRO_E` pair is validated independently of the
macro-group structure: one positive integral count and exactly count-plus-one
finite, nonnegative, strictly descending boundaries in MeV. The outer micro-
and macro-group boundaries need not be identical. The identity includes the current `inf` diffusion formulation, so it cannot
be reused silently by a future formulation. No tolerance, resampling or
normalization is applied to the printed numeric micro-grid.

Each `BranchState` derives its reserved `microgroup_identity` from every
retained native record. Partial pairs, differing universe grids and mixtures
of known and unknown universe grids are rejected. This also recovers identity
from older JSON files that retained the raw fields without derived metadata.
Conflicting caller declarations fail; replacing other metadata cannot hide a
native grid. A library requires identical effective identities, independently
of its Serpent version policy. Exact selection, interpolation and JSON
save/load carry the identity. A wholly unknown collection remains supported
for backward compatibility and synthetic calculations, visibly marked unknown.
Native-free manually authored states may declare the reserved identity explicitly;
that declaration is validated but is caller evidence, not native provenance.
This check changes no cross sections, factors or interpolation formulas.

The [Serpent manual](https://serpent.vtt.fi/docs/user_guide/gc_generation.html#infinite-and-critical-spectrum-calculation)
describes collapsing intermediate micro-group constants to the final macro
groups, including for infinite-spectrum constants. It states that micro-grid
choice has virtually no effect on infinite-spectrum results apart from limited
floating-point precision. Comparing processing identities therefore records a
compatibility condition, rather than proving a physical explanation for any
observed difference or establishing nuclear-data equivalence.
