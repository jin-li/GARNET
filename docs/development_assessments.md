# NDgpu development findings and priorities

Historical naming: this assessment records NDgpu before its rename to GARNET.
Source links below follow the current `garnet/` package.

Assessment date: 2026-10-05. Code baseline: upstream `d8f4433`, the NixOS
feature `e081c17`, and the AMD feature through `13a59cc`, integrated into
`dev` at `b48f672`.

This document records the project exploration, environment validation and
development assessments for the fork. NDgpu runs on both the tested NVIDIA
and AMD systems. The main development gaps are general branch cross-section
libraries, PWR/BWR coolant thermal hydraulics, broad discontinuity-factor
support, and performance of the complete coupled calculation.

Observed results below come from saved runs or inspected code. Proposed
designs and effort estimates are engineering judgments, not implemented
features or measured runtimes. Backend agreement does not establish reactor
physics accuracy. The scope starts with PWR/BWR and can later extend to
pebble-bed and heat-pipe reactors.

## Current solver capabilities

The solver uses NumPy on CPU and CuPy on GPU, with matrix-free stencils and
optional fused kernels. It supports multigroup diffusion, SPN/SDPN methods,
adjoints, delayed-neutron transients, and advanced transport/noise interfaces.
Cartesian, cylindrical, triangular/extruded-prism and unstructured paths have
different capabilities; they are not interchangeable entry points. The
high-level `Model` API is not a general CAD or transport interface. See the
[user guide](user_guide.md), [model API](model_api.md) and
[backend implementation](../garnet/backend.py).

| Area | Current finding | Development implication |
|---|---|---|
| CPU and NVIDIA | Representative examples and selected tests passed on 550w | Preserve these paths while developing AMD and multiphysics |
| AMD | Real HIP execution validated on halo | Support is experimental and bounded by the tested operations |
| Cross sections | Static `Material` objects plus limited temperature feedback | Add a general state library and explicit interpolation contract |
| Solid thermal model | Conduction, sink, surface losses and heat capacity | Useful foundation, but no complete PWR/BWR coolant model |
| External coupling | preCICE example exchanges power and temperature between two NDgpu participants | An independent TH solver still needs an adapter and validation |
| Discontinuity factors | Triangular diffusion has an optional implementation | Cartesian assembly-face DFs and GPU preconditioning remain work |

## NixOS environment and NVIDIA validation

The first feature added a pinned Nix development shell, `flake.lock`,
`uv.lock`, project-local caches, and the `cuda12-ctk` extra. Nix supplies
Python and native runtime libraries; uv manages the editable project and
Python packages. Python downloads are disabled. Run Python inside the Nix
shell: invoking `.venv/bin/python` outside it can fail to load
`libstdc++.so.6`. Environments and generated results remain ignored.

```sh
nix develop path:.
uv sync --locked --extra test --extra cuda12-ctk
uv run --locked --extra test --extra cuda12-ctk python examples/bare_reactor.py 32 gpu
```

Retain the selected GPU extra in subsequent `uv run` commands. CUDA and ROCm
CuPy distributions own the same module and cannot share an environment;
the AMD feature added explicit uv conflicts to enforce this.

On 550w, the initial `nvidia-smi` failure was caused by the execution sandbox
hiding `/dev/nvidia*`. Host checks succeeded with matching kernel/userspace
driver 595.71.05. No driver repair or system configuration change was needed.
The tested GPU was an RTX 5070 Ti, 16 GB, using CuPy 14.2.0 and CUDA 12.9
component wheels with Nix Python 3.13.15. The CUDA API advertised by the driver
is not the same thing as the toolkit/runtime selected by CuPy.

Earlier validation ran 11 CPU example commands, 103 selected CPU tests, a
direct CPU/GPU eigenvalue and full-flux comparison, and seven NVIDIA
example/benchmark commands. The direct comparison required convergence and
absolute k tolerance `1e-8`, with flux `rtol=1e-5`, `atol=1e-9`.
The CPU examples covered bare and reflected problems, hex/triangular and
unstructured meshes, transients, HP-MR, thermal coupling and SDPN.
These were selected checks, not the complete suite. Reviewed evidence is in
[the 550w report](../benchmark-results/nixos-nvidia-550w/README.md).

The original warmed bare-box benchmark recorded:

| Mesh | CPU seconds | NVIDIA seconds | Reported speedup |
|---|---:|---:|---:|
| 32³ | 0.11 | 0.06 | 1.9× |
| 64³ | 1.71 | 0.08 | 22.6× |
| 96³ | 9.69 | 0.18 | 55.2× |

These rounded values came from a single timing sample on a shared GPU.
They are evidence of acceleration for those workloads, not a general
speedup guarantee. After integrating AMD support, 52 backend/fused-kernel
tests passed and a NVIDIA 32³ bare-reactor smoke run converged with
`device: cuda (cupy): NVIDIA GeForce RTX 5070 Ti` and `k_eff=1.212294`.
The full earlier validation was not repeated during documentation work.
The [post integration record](../benchmark-results/nixos-nvidia-550w/post-amd-checks.txt)
preserves the new commands and selected observed output.

## AMD support and measured limits

The initial assessment found ROCm could enumerate halo's Radeon 8060S
(`gfx1151`) but CuPy was absent. That assessment has been superseded by
implemented and tested support. The fork now provides a separate `.#rocm`
Nix shell, ROCm 7.2.3 libraries and a merged `ROCM_HOME`, with
`cupy-rocm-7-0==14.2.0`. Backend selection recognizes `rocm`/`hip` and `cuda`,
rejects the wrong CuPy family, and reports the correct device label.
Explicit `device="gpu"` requires a device; `auto` can fall back to CPU and
must not be used to demonstrate GPU execution.

[AMD setup and limitations](amd_rocm.md) and the
[halo validation report](../benchmark-results/amd-rocm-halo/README.md)
are the detailed sources of record. Halo uses a Ryzen AI Max+ 395 APU;
its GPU shares system memory. Allocation pools must not be added together
or reported as dedicated VRAM. NixOS execution is demonstrated on this host,
not certified across all ROCm versions and GPUs.

Nine required cases passed with fusion off and on: diffusion, SP3, short
delayed-neutron transients, internal thermal coupling and 3D diffusion.
Maximum CPU/AMD differences were `1.28e-14` in k, `6.23e-12` in normalized
flux L2, `1.65e-12` in relative power, and `1.14e-12 K` in temperature.
Graph capture/replay, deliberate graph fallback, a CSR matvec, tested
SN/TriSN DSA factories and event timing also passed their recorded probes.
The latter transport/profiling probes are reported separately by the
validator; passing its required gate alone does not certify every probe.

| Halo workload | CPU warm median | AMD fused warm median | Interpretation |
|---|---:|---:|---|
| 32³ heterogeneous diffusion | 0.0822 s | 0.0579 s | GPU faster for this case |
| 64³ heterogeneous diffusion | 1.3975 s | 0.6334 s | 2.21× GPU speedup |
| 8×8 SP3 | 0.00418 s | 0.0238 s | Small workload slower on GPU |
| Internal HP-MR thermal coupling | 0.0366 s | 0.1846 s | Small coupled workload slower on GPU |

First-use compilation/runtime overhead was substantial. The first 8×8
unfused diffusion run took 3.56 s versus 0.0290 s warmed; fused took
0.556 s versus 0.0212 s warmed. Later cases reused compiled kernels.
NVIDIA and AMD tables use different hosts, workloads and methodologies;
they are not a controlled NVIDIA-versus-AMD comparison.

CuPy's broad documented ROCm sparse limitation did not prevent the specific
tested sparse operations. Full sparse API coverage, CMFD multigrid, large
transport problems, multi-GPU execution, external preCICE and long coupled
transients remain unvalidated in this work. SN row sweeps and TriSN LU
remain CPU paths; periodic TriSN levels have existing restrictions.

## Cross-section data and branch coverage

The NPZ suffix describes storage, not whether data is fixed-state or
branched. Inspection of
[`hpmr_core_xs_g11.npz`](../garnet/benchmarks/data/hpmr_core_xs_g11.npz)
found five fuel temperature branches at 600, 700, 800, 1000 and 1200 K.
They follow the diagonal `Tfuel = Tmod`, not an independent two-dimensional
temperature grid. Other materials use reference-state arrays. Burnup,
moderator density/void, boron and history are not general branch coordinates.

The extract contains material arrays `D`, `sa`, `nsf`, `ss`, `chi`, `kf`,
metadata `G` and `ids`, velocities and delayed-neutron data. Fuel branch
vectors have shape `(5, 11)` and scattering `(5, 11, 11)`, including branch
`total` and `kf`. Velocities and kinetics are single-state arrays.
The [Griffin reader](../garnet/griffin_xs.py) selects XML tables and constructs
[Material objects](../garnet/materials.py); NPZ is not a required solver input
format or an established general interchange schema.

`Material` expects macroscopic cross sections in `cm^-1`, diffusion in `cm`,
and scattering indexed `[group_from, group_to]`. It stores absorption,
neutron production, optional total and fission-energy data, and normalized
chi. Without total, `1/(3D)` is used; that fallback must not be confused with
the physical total cross section for arbitrary transport/SPN input.

[TabulatedFeedback](../garnet/feedback.py) interpolates one temperature
coordinate linearly and clamps beyond the table range. It applies ratios
to reference values for absorption, production, diffusion, total and
existing off-diagonal scattering, then rebuilds removal. Important limits:

- `chi` is passed by the HP-MR helper but left unchanged by the update code,
  despite broader wording in its docstring. `kappa_fission` and kinetics are
  not updated by this hook.
- A zero reference value produces a unit ratio, so a reaction cannot appear
  from zero. A scattering channel absent from the original fields is skipped.
- The selected material is rejected when volume-mixed in affected cells.
- One solid temperature cannot represent separate PWR/BWR fuel and coolant
  temperatures, density/void or depletion history.

Thus the existing data has limited branches, but the general library and
feedback capabilities needed for PWR/BWR remain development work.

## Serpent and CASMO5 library interfaces

The intended inputs are Serpent MATLAB `.m` results and CASMO5/CMSLINK
libraries. A converter should first build a common, versioned library model
with group boundaries, units, material/assembly identifiers, explicit state
coordinates, provenance, uncertainty where supplied, and field conventions.
It should then create NDgpu material fields for a requested state. Merely
renaming arrays into the current HP-MR NPZ layout would preserve its limits.

For Serpent, parse the generated assignment format as data; do not execute
arbitrary MATLAB code. Preserve universe, burnup and branch identity and
separate mean values from statistical errors. Map the chosen homogenization
and leakage convention deliberately: infinite-spectrum and B1 quantities
are different. Resolve scattering orientation, transport correction,
fission-spectrum representation, energy production units and kinetics.
Use [Serpent group-constant documentation](https://serpent.vtt.fi/docs/user_guide/gc_generation.html)
and [output parameter definitions](https://serpent.vtt.fi/mediawiki/index.php/Output_parameters)
alongside representative version-specific files.

For CASMO5, the user's target is CMSLINK output rather than an assumed
generic text export. Studsvik describes CASMO5 as a generator of
[cross-section data for SIMULATE](https://www.studsvik.com/key-offerings/nuclear-simulation-software/software-products/casmo5/),
but the public product page does not establish a CMSLINK record specification.
We have not inspected a sample library or confirmed its encoding/version.
Prefer a vendor-supported reader/export where available. Direct decoding
depends on the actual record layout, branch matrix, units, correction
conventions and permitted documentation. A CASMO listing parser is not
automatically a CMSLINK reader.

Proposed interpolation should preserve positivity, chi normalization and
cross-section balance, and define out-of-range behavior explicitly. Regular
grids and sparse case matrices need different interpolation handling.
Burnup history, rod state and material type cannot always be treated as
continuous coordinates. Depletion tables provide state dependence; importing
them does not implement a depletion solver.

| Proposed scope | Rough effort for one engineer | Assumptions |
|---|---|---|
| Serpent fixed-state converter | 1–2 weeks | Representative results and known conventions |
| Common branch model and runtime interpolation | 3–6 weeks | Bounded coordinates and reference tests; reusable by both interfaces |
| CASMO5 through documented export/reader | 2–4 weeks | Export and branch metadata available |
| Direct CMSLINK decoding | 4–8+ weeks, uncertain | Documented records and multiple sample libraries; otherwise not reliably estimable |
| Broad validation and production hardening | Additional work | Multiple assemblies, states and independent core references |

These are planning ranges, not additive project commitments. A sample
Serpent case, CMSLINK version/specification or export, group structure,
branch matrix and expected validation quantities are prerequisites for a
firmer estimate. No converters were implemented during those assessments.
The subsequent [fixed-state Serpent interface](serpent_xs.md) now implements
literal results-file import and validates zero-leakage point-reactor eigenpairs
for the supplied 2, 4, 7, 14 and 23 group datasets on CPU and AMD. Branch
interpolation and the broader interfaces discussed here remain future work.

## Thermal hydraulics capability and coupling schemes

NDgpu already supports coupled neutronics and solid thermal analysis.
The [thermal solver](../garnet/thermal.py) solves conduction, a volumetric
sink, Robin surface losses and transient heat capacity. The
[coupling driver](../garnet/coupling.py) provides steady Picard/Anderson
iterations and coupled transient stepping. These components can run through
the CPU/GPU array backends.

It does not solve general coolant mass, momentum and enthalpy transport,
pressure drop, boiling, void fraction or two-phase flow. The HP-MR heat-pipe
example uses a calibrated conductance to a prescribed sink temperature;
it is not an explicit heat-pipe vapor/liquid or operating-limit model.
See [existing coupling documentation](coupling.md).

The [preCICE example](../examples/precice/README.md) exchanges `Power`
(`W/cm³`) and `Temperature` (K) between two processes using NDgpu's own
physics on identical meshes. It demonstrates coupling machinery, not an
implemented OpenFOAM, commercial CFD or reactor-system adapter. preCICE was
not installed or exercised in the NixOS validation reported here.

For PWR/BWR, the proposed exchanged state includes spatial fission heating,
effective fuel temperature, moderator temperature and density/void, with
pressure/enthalpy where needed. Fuel temperature requires a defined radial
conduction or effective-temperature model. Preserve integrated power under
mesh transfer and distinguish volumetric heating from wall heat flux.
Do not carry the example's identical-mesh mapping rules unchanged into
unequal coolant and neutronics meshes; intensive fields still require
volume-aware conservation of the derived energy source.

Three coupling schemes are practical:

1. **Steady partitioned implicit:** neutronics supplies power, TH updates
   fuel/coolant state, the branch library updates XS, and the loop converges
   with relaxation or acceleration. Check power, temperatures and density
   as well as k. This is the preferred first PWR implementation.
2. **Transient staggered or iterated coupling:** advance each subsystem over
   a coupling window. Staggering is simpler but introduces feedback lag;
   strong feedback can require iteration, rollback and smaller windows.
   Subcycling and different subsystem timesteps need an accuracy study.
3. **Monolithic or tightly integrated coupling:** solve a joint nonlinear
   residual. This offers stronger consistency but requires residual/Jacobian
   interfaces and preconditioning across systems; it is a larger redesign.

| TH option | Proposed use and integration | Performance and scope limits |
|---|---|---|
| Internal simple TH | Assembly/channel energy balance, pressure-drop model and radial fuel/gap/clad conduction; retain backend arrays | Lowest integration burden; single-phase PWR first. BWR needs separate two-phase closures and stability work |
| MOOSE THM | Reduced channels and heat structures through a custom bridge | CPU reduced models may still match the required turnaround; no ready NDgpu bridge. [Module overview](https://mooseframework.inl.gov/modules/thermal_hydraulics/index.html) |
| OpenFOAM + preCICE | Detailed coolant/CHT through its maintained thermal adapter and a new NDgpu participant | Detailed CFD can dominate runtime; choose supported OpenFOAM/adapter versions. [Adapter](https://precice.org/adapter-openfoam-overview) |
| Code_Saturne + SYRTHES | Alternative detailed fluid/solid thermal route | Existing thermal coupling is not a general NDgpu adapter; availability of a suitable external interface still needs confirmation. [Coupling docs](https://code-saturne.org/doc/code_saturne-9.2/advanced_coupling.html) |
| NekRS / Cardinal | GPU CFD candidate for selected detailed flow/CHT studies | Cardinal wraps NekRS in MOOSE and supports CUDA/HIP builds; integrating NDgpu is new work. GPU execution does not remove CFD resolution cost. [Cardinal architecture](https://cardinal.cels.anl.gov/), [GPU builds](https://cardinal.cels.anl.gov/hpc_build_run_tips.html) |
| Fluent + System Coupling | Commercial CFD through an external participant | Participant API, installed release, GPU physics support and licenses must be checked; no NDgpu integration exists. [Participant library](https://devdocs.ansys.com/system-coupling/participantlib/) |
| STAR-CCM+ | Commercial CFD/co-simulation candidate | Its 2602 release documents AMD GPU support; compatible physics, hardware and coupling APIs still need verification. [Release description](https://blogs.sw.siemens.com/simcenter/simcenter-star-ccm-2602-released/) |
| TRACE / RELAP family | Plant/system transients if code access and an external interface are available | Access/distribution conditions and version-specific coupling require investigation; not ordinary pip dependencies. [NRC access information](https://www.nrc.gov/education-regulatory-research/research/obtaining-the-codes) |

SAM, Sockeye and Pronghorn are later candidates for advanced reactor systems,
heat pipes and porous-medium pebble-bed cooling. They are distributed through
[NCRC access arrangements](https://mooseframework.inl.gov/bison/help/inl/applications.html);
they should not be presented as unrestricted public THM installations.
Pebble-bed work needs solid/coolant nonequilibrium and porous resistance;
heat-pipe work needs a justified reduced operating model or explicit
two-phase pipe physics. None of these models is established by the current
fixed-sink demonstration.

Planning judgment: a bounded steady single-phase channel prototype is
several weeks of work; usable PWR transients and verified feedback are a
multi-month effort. BWR two-phase capability is a separate larger project.
External adapter demonstrations may take weeks, while validated coupled
reactor applications can take months. Code access, mesh mapping and reference
problems determine the estimate more than the nominal connector API.

## Performance of the complete coupled calculation

Fast neutronics alone cannot guarantee fast multiphysics. For a serial
partitioned steady solve, a useful planning model is:

```text
T_total ≈ T_setup + N_outer × (T_neutronics + T_TH + T_mapping + T_IO)
```

This is a model, not a measured result. If TH takes 1800 s per exchange and
GPU acceleration reduces neutronics from 60 s to 5 s, one exchange changes
from 1860 s to 1805 s: only about 1.03× faster, ignoring other overhead.
The time compared must be the TH update for the same state/window, not
unrelated total runtimes from differently sized simulations.

A reduced CPU channel solver can therefore be a better partner than a
GPU CFD solver with millions of cells. Full CFD is appropriate when its
flow detail is needed, but it should not be the default online model for
rapid core studies. GPU-native commercial CFD or NekRS remains subject to
resolution, physics, precision and hardware constraints. Sharing one GPU
also introduces memory/bandwidth contention; parallel participants do not
guarantee useful overlap or zero-copy exchange.

Before choosing a TH partner, measure complete coupled wall time with
identical physics inputs and tolerances, synchronized GPU timing, initial
and warmed runs, repeated measurements, thread counts and software versions.
Record phase times, outer/inner iterations, rejected steps, transfers,
compilation, memory and energy-balance errors. Solve each update to the
accuracy the outer coupling needs; excessive inner accuracy can waste time.
The current steady driver's docstring records an example where requesting
a coupled residual below the inner-solver noise floor increased iterations
without materially changing the answer; it is not a benchmark from this
session.

Potential improvements are warm starts with reliable convergence checks,
infrequent expensive mesh setup, subsystem subcycling, adaptive coupling
windows, and validated reduced/surrogate TH models. Freezing TH for many
neutronics updates or using one-way coupling changes the approximation and
must be justified, rather than counted as equivalent acceleration.

### Existing host transfers in steady coupling

The core GPU stencil is not the entire coupled path. In the current steady
driver, [power_density](../garnet/power.py) calls `asnumpy(flux)` and edits
power on the CPU. `thermal_step` returns host temperature, and Picard/
Anderson updates use NumPy. Thus each steady exchange can move full fields
between host and device. The coupled transient path already has more
device-resident power, thermal and feedback operations plus profiling.
This is a code finding; its contribution to runtime has not been isolated
here. Optimize and measure the actual selected path before claiming a
fully device-resident coupled solve.

## Discontinuity factors and their GPU performance issue

DFs allow homogenized flux to jump while reconstructed heterogeneous
surface flux and interface current remain continuous. With the convention
used by the triangular implementation:

```text
f_left × phi_homogeneous_surface_left
    = f_right × phi_homogeneous_surface_right
```

[TriGroupOperator](../garnet/tri.py) implements per-group cell factors or
per-face pairs for its three in-plane edge families. The helper
`face_df_from_pairs` maps ordered region pairs to face factors.
`TriDiffusionEigenSolver` accepts `df` and boundary coefficients `bcf`.
Unit factors recover ordinary coupling. Axial couplings still use the
ordinary harmonic coefficient. No comparable DF input was found in the
standard Cartesian, regular hexagonal or unstructured diffusion operators,
or the general material/feedback API. Broad SPN DF treatment is not supplied
by the triangular diffusion implementation.

The [SPH plus DF fitting routine](../garnet/sph.py) is explicitly marked
"SHELVED" and is not the supported correction path. Its code records
partial-current matching problems near strongly absorbing regions and
different behavior for absolute k and control worth. Those historical
comments do not establish DF usefulness or accuracy for PWR/BWR.
The existing DF fused-stencil arithmetic regression passed during the
assessment and is included in the 52-test check after synchronization;
this does not constitute an assembly/core physics validation.

The important GPU issue is in `TriGroupOperator.ilu_preconditioner`:

1. It transfers operator coefficients to the host and assembles SciPy CSR.
2. It computes SciPy `spilu` on the CPU.
3. On **each preconditioner application**, it transfers the residual from
   GPU to CPU, performs a CPU sparse solve, then transfers the answer back
   to GPU with `xp.asarray`.

`TriDiffusionEigenSolver` installs this preconditioner whenever `df` is not
`None`, including explicitly supplied unit-factor arrays. Unequal side
factors generally make the operator nonsymmetric; use GMRES/BiCGSTAB rather
than ordinary CG. The code warns that its default diagonal/Neumann
preconditioning can lose effectiveness for these coefficients. A statement
that loss of row diagonal dominance necessarily destroys the M-matrix
property would be too strong; preconditioner choice should be based on the
actual operator and convergence evidence.

**The device-resident DF stencil therefore does not imply a device-resident
DF solve.** Repeated synchronization, full residual transfers and CPU
triangular solves can become a major bottleneck. This follows directly from
the implementation; the magnitude has not been benchmarked. Halo's passed
SN/TriSN DSA probes exercise different factories and do not resolve this
DF-ILU issue. Host transfers also obstruct capturing a fully device-only
iteration in a GPU graph.

An implementation plan should separately address assembly-face coefficients,
their library conventions, and the linear solver. Evaluate device-native
preconditioners or a justified variable transformation; do not simply
delete ILU without checking convergence. Per-face factors need not admit
a single global symmetrizing transformation. Benchmark CPU and GPU
end-to-end with nontrivial DFs, record transfer time and Krylov iterations,
and preserve current conservation and the unit-factor limit.

For Serpent imports, match the receiving coarse model's DF definition.
Serpent documents that nonzero-current ADFs depend on the homogeneous
solution method; a surface/volume ratio from a zero-current lattice is not
a universal coarse-solver correction.
[Serpent ADF definitions](https://serpent.vtt.fi/docs/user_guide/gc_generation.html#discontinuity-factors)
explain this requirement. CASMO/CMSLINK face ordering, assembly orientation
and normalization need the same explicit treatment.

Planning estimate from the DF assessment: 1–2 engineer-weeks for a bounded
Cartesian diffusion prototype, or roughly 3–6 weeks total for a validated
GPU implementation with one documented library format and available
references. CMSLINK decoding, independent branches, axial DFs and SPN
equivalence add scope. No DF extension or preconditioner fix was implemented
as part of documenting this issue.

## Fork workflow and next development steps

`origin` is [jin-li/NDgpu](https://github.com/jin-li/NDgpu); `upstream` is
[poncetma/NDgpu](https://github.com/poncetma/NDgpu). Keep `main` unchanged
during this development phase. Features start from `dev`, receive focused
commits and validation, are pushed to the fork, then merge into `dev` when
complete. No upstream pull requests are part of the current workflow.
Halo's checkout is `~/Documents/GitHub/NDgpu`; its local environment and
caches are not transferred by Git.

```sh
git switch dev
git pull --ff-only origin dev
git switch -c feature/<scope>
# Implement and validate, then commit specific files.
git push -u origin feature/<scope>
# After review and acceptance:
git switch dev
git merge --no-ff feature/<scope>
git push origin dev
```

Recommended next steps, with the AMD feature now integrated:

1. Obtain representative Serpent and CASMO5/CMSLINK files and independent
   state/reference results. Define the common branch-library contract.
2. Build fixed-state import checks, then independent temperature/density
   branches with GPU-compatible interpolation and explicit error handling.
3. Add Cartesian assembly-face DFs and address GPU preconditioning together;
   compare assembly powers and leakage as well as k.
4. Add a bounded internal single-phase PWR channel/fuel model and benchmark
   the complete coupled calculation before selecting a heavier TH partner.
5. Extend to PWR transients, then BWR two-phase feedback as distinct scopes.
   Investigate detailed external CFD and later pebble-bed/heat-pipe models
   where the application requires their additional physics.

These are proposed priorities from the assessment. The fixed-state Serpent
import and point-reactor checks are now implemented; the remaining branch,
TH and expanded DF work has not been implemented by that interface.
