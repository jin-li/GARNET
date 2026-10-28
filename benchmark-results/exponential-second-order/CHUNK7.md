# Resolved TWIGL comparison: seventh development checkpoint

2026-10-26. This carries the [history estimator](CHUNK6.md) into the E4
spatial gate: the repository's two-group TWIGL seed/blanket step and ramp
on 20×20 and 40×40 Cartesian meshes. An 80×80 reference measures separate
spatial sensitivity. This checkpoint adds research tools, verification and
error/work evidence. The production integration algorithms are unchanged.

## Physical problem and reference

The existing [TWIGL builder](../../garnet/benchmarks/twigl.py) supplies an
80×80 cm quarter core with a central perturbed seed, an unperturbed L-shaped
seed and a blanket. Two inner faces reflect; the outer x/y faces impose
zero surface flux. One delayed family has beta=0.0075 and lambda=0.08/s;
group velocities are 1e7 and 2e5 cm/s. Thermal absorption in the central seed
drops by 2.3333% for the step, or ramps down through 0.2 s. All marches end
at 0.5 s. The ramp corner is an explicit event for every method.

The reference independently assembles the full physical neutron/precursor
generator, retaining harmonic face diffusion, half-cell zero-flux leakage,
scattering, prompt and delayed production, velocities and the same initial
critical adjustment k0. It does not use GARNET's time-shifted or
precursor-eliminated step operator. Sparse Radau integrates this system,
splitting at the ramp corner. Both the initial eigenpair and its normalized
physical state are shared with the FV calculation, isolating temporal error
on each mesh. This is an independent temporal reference for the same
semidiscretization, not an independent determination of its initial k0.

Reference tolerances 3e-9 and 3e-11 are compared on each study mesh, with
per-field absolute floors derived from the initial RMS state. The 80×80
spatial reference uses 3e-9 and 3e-10. Random physical generator actions
match the matrix-free kernels. A small independent augmented dense matrix
exponential verifies both endpoint states and the reference power integral.
Three-point Gauss integration of each cubic Radau dense segment supplies
the reference integral.

| Case / mesh | k0 | P(0.1 s) | P(0.5 s) |
|---|---:|---:|---:|
| Step 20×20 | 0.913000921 | 2.0617151 | 2.1316342 |
| Ramp 20×20 | 0.913000921 | 1.3082486 | 2.1101498 |
| Step 40×40 | 0.913100486 | 2.0608846 | 2.1307117 |
| Ramp 40×40 | 0.913100486 | 1.3083129 | 2.1092603 |
| Step 80×80 | 0.913176133 | 2.0613587 | 2.1312300 |
| Ramp 80×80 | 0.913176133 | 1.3084919 | 2.1097647 |

The largest temporal-reference field refinement defect on the study meshes is 3.27e-09 in relative RMS, well below the 0.001 budget.

The rounded power and eigenvalue values in the repository remain contextual
benchmark checks. The finer-mesh and tolerance-refinement results below
separate spatial discretization from transient integration error.

## Error contract and work controls

Each configuration is judged on the union of its own accepted endpoints
and a common grid containing 2,001 uniform times plus 401 logarithmically
spaced early times, with the control corner included. Flux and power are
linearly interpolated between recorded macro endpoints. Measurements are:

- Maximum relative power error.
- Maximum per-group physical flux RMS error, relative to the reference
  field norm; equal cell volumes cancel in this ratio.
- Maximum per-group shape RMS error after dividing flux by relative power.
- Maximum relative fission power error separately in all three regions.
- Integrated relative power error using the reported endpoint trapezoids.
- Final precursor field RMS and total inventory errors.

All seven error categories must satisfy the same declared budget; recorded
flux and final precursors must be nonnegative. The precursor gate covers
final fields and inventories, not unsaved precursor trajectories. Power
uses the existing TWIGL relative fission-source convention; its time integral
is not a heat-transfer energy calculation. Sampled maxima are not certified
bounds at every interior time or cell.

The main sweep tests FTM2 and transformed BDF2 with both history estimation
and full/fine step doubling, adaptive BE, and automatic adaptive BDF1–5.
It uses state rtols 1e-3, 1e-4 and 1e-5, scalar initial width 0.005 s,
minimum width 1e-11 s and maximum width 0.02 s. FTM also uses its independent
output_rtol=1e-4. Controller norms and defaults remain unchanged; equal
rtol is not treated as equal physical accuracy.

A bounded BE/BDF supplement checks BDF maximum widths 0.01 and 0.005 s
at rtol=1e-3/1e-4, plus BE at 0.005 s and rtol=1e-3. This tests output
resolution independently of the legacy state estimator. Shared prescribed
graded meshes compare lagged and zero frequencies for both FTM schemes:
initial width 1e-7 s, growth factor 1.15, maximum 0.005 s, regraded at the
ramp event. These common meshes isolate the frequency transform from
different adaptive schedules. Separate spatial audit runs compare state
estimators on identical full/fine trials.

There are 116 recorded FV configurations: 72 in the main adaptive sweep,
16 common-mesh frequency controls, eight estimator audits and 20 width
controls. Work includes every constituent solve, rejected attempt,
frequency correction and fallback, with group-PCG iterations, FGMRES
iterations and operator rebuilds recorded. Shared initial eigenpairs and
reference construction are outside the marching counts. Single-execution
CPU seconds are diagnostic; overlapping runs and missing setup reuse make
them unsuitable for a runtime speedup claim.

## Matched physical error and work

The following are the fewest tested constituent solves among adaptive configurations meeting **0.1% in
all seven error categories**, including the output-width supplement.

| Case / mesh | FTM2 history | FTM2 doubling | FTM-BDF2 history | FTM-BDF2 doubling | Adaptive BE | Adaptive BDF |
|---|---:|---:|---:|---:|---:|---:|
| Step 20×20 | 257 | 375 | 243 | 354 | 758 | 109 |
| Ramp 20×20 | 180 | 264 | 189 | 273 | 224 | 92 |
| Step 40×40 | 257 | 375 | 243 | 354 | 757 | 138 |
| Ramp 40×40 | 180 | 264 | 178 | 258 | 224 | 92 |

| Case / mesh | Fewest FTM solves, including fixed controls | Fewest BDF solves | Fewest FTM group-PCG iterations | Fewest BDF group-PCG iterations |
|---|---:|---:|---:|---:|
| Step 20×20 | 171 | 109 | 54,704 | 37,221 |
| Ramp 20×20 | 180 | 92 | 65,936 | 41,310 |
| Step 40×40 | 171 | 138 | 111,578 | 85,431 |
| Ramp 40×40 | 178 | 92 | 134,614 | 84,233 |

The first table covers adaptive configurations. The second also includes prescribed-grid FTM controls and minimizes each work column independently; its solve and PCG minima can select different FTM configurations. Full qualifying configurations are retained in the combined JSON. BDF wins both work measures in all four cases at this budget.

The ramp demonstrates why output controls belong in the comparator search.
A coarse BDF output interval can miss the power budget despite accurate
endpoints. A smaller width cap can require less total work than tightening
the state tolerance while retaining the original cap.

| Ramp mesh | BDF solves with 0.02 s cap | Selected cap / state rtol | Selected solves | Maximum power error |
|---|---:|---|---:|---:|
| Ramp 20×20 | 416 | 0.01 s / 0.001 | 92 | 0.05949% |
| Ramp 40×40 | 413 | 0.01 s / 0.001 | 92 | 0.05948% |

The eight spatial audits supply 807 eligible paired trials. They record 0 failed reference state checks and no decision disagreements. Since these spatial trials contain no failed eligible reference state checks, they do not establish detection of spatial rejection failures; the independent physical output comparisons and earlier state-driven rejection tests retain that role.

## Common meshes and spatial sensitivity

The shared-mesh controls below show maximum sampled power error in percent.
Each lagged/zero pair has the same endpoints, event treatment and solve
count. Changing frequency is the only scheme setting changed within a pair.

| Case / mesh | FTM2 lagged / zero error | FTM-BDF2 lagged / zero error | Shared solves |
|---|---:|---:|---:|
| Step 20×20 | 0.05289% / 0.04669% | 0.19794% / 0.15305% | 171 |
| Ramp 20×20 | 0.01508% / 0.01589% | 0.01063% / 0.01926% | 242 |
| Step 40×40 | 0.05287% / 0.04667% | 0.19780% / 0.15306% | 171 |
| Ramp 40×40 | 0.01504% / 0.01584% | 0.01059% / 0.01919% | 242 |

These controls do not establish a uniform benefit from the shared frequency
on TWIGL. Adaptive history estimation reduces FTM work relative to its
full/fine controller, but that saving does not by itself earn promotion over
the strongest tested BE/BDF configuration.

For spatial comparison, fine fields are conservatively area-averaged onto
the coarser grid. Multiplication by cell count restores a common density
normalization before comparing meshes whose engine power convention sums
cell sources to one. Flux shape is then normalized by each mesh's power.

| Case / coarse → fine mesh | Maximum power difference | Fast shape RMS | Thermal shape RMS | Final precursor field RMS |
|---|---:|---:|---:|---:|
| Step 20×20 → 80×80 | 0.07547% | 0.45307% | 3.50345% | 1.85005% |
| Step 40×40 → 80×80 | 0.04327% | 0.10692% | 1.10289% | 0.64093% |
| Ramp 20×20 → 80×80 | 0.01910% | 0.44873% | 3.50345% | 1.85041% |
| Ramp 40×40 → 80×80 | 0.02708% | 0.10617% | 1.10289% | 0.64098% |

The 0.1% spatial-shape condition is **not met**. Thermal-group differences are already present at time zero. The follow-up [initial-state mesh ladder](chunk7-initial-resolution.json) compares stationary fields to a conservatively restricted 320×320 solve:

| Initial mesh → 320×320 | Fast flux RMS | Thermal flux RMS | Equilibrium precursor RMS |
|---|---:|---:|---:|
| 20×20 | 0.47576% | 3.87822% | 2.04686% |
| 40×40 | 0.14056% | 1.51159% | 0.87867% |
| 80×80 | 0.03603% | 0.42096% | 0.25508% |
| 160×160 | 0.00736% | 0.08799% | 0.05398% |

160×160 is the first level in this initial-state ladder whose differences to 320×320 are below 0.1% in all three fields. This identifies a starting resolution for further investigation; it does not establish transient or continuum convergence at 160×160.

These differences are spatial sensitivity, not error assigned to a time
integrator. The work comparison uses a separate converged temporal reference
on each mesh; it does not claim continuum accuracy from a local state
tolerance or from agreement at two rounded literature output times.

## Verification and next gate

The [118-case targeted regression](chunk7-regression.xml) passes across the
new spatial checks, adaptive FTM, output/history estimation, rejection
replay and existing TWIGL/LMW validation. A later
[12-case focused check](chunk7-twigl-verification.xml) also verifies the
independent augmented-exponential integral test; these checks overlap the
regression batch. No full repository, slow-suite or GPU rerun is claimed.
The [QA record](../../docs/qa/garnet-exponential-twigl-2026-10-26.json) retains
environment, source hashes, physical gates and the combined work selections.

**Gate outcome:** the temporal error/work comparison is established on both study meshes; spatial-accuracy and performance-promotion conditions are unmet. Tuned BDF costs less in the tested work measures, and the mesh error exceeds the declared shape budget. Hold frequency-model expansion and speedup claims. The next accuracy task is a finer transient mesh/reference study guided by the initial-state ladder. Shared operator/setup reuse remains a separate bounded optimization before any repeated CPU/GPU timing promotion.

Reproduce from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m tools.assess_exponential_twigl
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m tools.assess_exponential_twigl_widths
python3 -m tools.summarize_exponential_twigl
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m tools.assess_exponential_twigl_resolution
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest tests/verification/test_exponential_twigl.py -q
```

Evidence: [main study](chunk7-twigl.json), [output-width controls](chunk7-twigl-widths.json),
and [combined physical-budget selections](chunk7-summary.json).
