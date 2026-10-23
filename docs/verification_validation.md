# Verification and validation

GARNET's QA combines exact mathematical checks, reactor benchmark comparisons,
and studies of coupled transients and GPU execution. This report links the
evidence to the core features exposed in the [README](../README.md). Test
assertions are the executable acceptance criteria; benchmark reports retain
the mesh, time step, physics data, and hardware behind their results.

## Diffusion API check

The diffusion API audit on 23 October 2026 added 44 workflow cases. The full
default CPU suite passed: **639 passed, 0 failed, 7 skipped, 13 slow cases
excluded**, in 712.10 s. A final focused run passed all **75 cases**, covering
the original model/TWIGL checks, new workflows, example imports, and the last
mesh-boundary input guard. The skips are the same optional dependencies,
external mesh inputs and CUDA cases listed in the earlier CPU record below.

The new checks verify named material replacement, constructor-option forwarding,
power integration on all three model types, snapshot consistency, Cartesian
thermal feedback and coupled equilibrium, invalid-input errors, and a one-cell
Gmsh mesh. Acceptance criteria and remaining API gaps are in the
[audit report](api_audit.md); the
[workflow tests](../tests/verification/test_diffusion_workflow_api.py) are the
executable specification.

The refreshed `dist/garnet-src.zip` was installed in isolation. It reproduced
the README eigenvalue `1.1352040511640966`, the `(2, 24, 24, 24)` flux shape,
and a power field integrating to 2 MW, and ran the new Cartesian coupled
example successfully. Syntax was checked in 177 Python files. GPU execution
and the excluded slow studies were not repeated.

Evidence: [full pytest record](qa/garnet-api-cpu-2026-10-23.xml) and
[environment, source fingerprints, focused checks and archive validation](qa/garnet-api-2026-10-23.json).
The record distinguishes the full-suite source snapshot from the two files
with the subsequent boundary guard and its final focused checks.

## GARNET rename check

The renamed checkout was checked on 23 October 2026 in a local staging
directory. **595 passed, 0 failed, 7 skipped, and 13 slow cases were
excluded** in 774.36 s. The earlier example-launch timeout did not
reproduce in this run; its original record is retained below. The staged
files are the ones applied to the renamed checkout.

The checks also covered all 17 Colab notebooks (111 code cells parsed), syntax
in 180 Python files and 11 shell scripts, and installation from
`dist/garnet-src.zip`. The installed package loaded its bundled 11-group
cross sections, honored `GARNET_FUSED=0`, and ran the README example with the
same eigenvalue and flux shape as before. The library's 56 source files
contain only the expected name substitutions.

The LRA figures and PDF were rebuilt from the preserved numerical records.
Historical JSON, CSV, cross-section data, and earlier raw QA records were
checked for byte-for-byte preservation. Hardware GPU studies were not rerun.
See the [rename validation record](qa/garnet-rebrand-2026-10-23.json) for the
source fingerprint, environment, archive checks, and per-module counts, and
the [pytest record](qa/garnet-cpu-2026-10-23.xml) for individual cases.

## Earlier CPU check

Checked on 23 October 2026 against revision `9241d54`. Pytest collected 614
cases, selecting 601 across 57 test modules: 511 verification cases and 90
validation cases. The configured default excluded 13 expensive cases.

**Result: 594 passed, 1 failed, 7 skipped, 13 deselected in 838.82 s.**

The failure is
[`test_cross_example_import_works_from_documented_entry_point`](../tests/verification/test_examples_current.py):
its subprocess running `hpmr_adaptive_coupled_benchmark.py --help` exceeded the
10 s timeout. A focused recheck reproduced the timeout. Direct invocation of
the same command without that limit completed successfully. The numerical
checks reported no failures; the example-launch timeout remains an outstanding
QA failure in this recorded run.

The seven skips comprise unavailable preCICE bindings, two unavailable
external VVER mesh cases, two CUDA-device kernel cases, an optional `pyamg`
case, and a CUDA-less error-path test that skips when CuPy is installed.
The preCICE skip occurs at module import, so pytest reports it in addition to
the 601 selected test cases. Slow cases and GPU studies were not rerun here.

The environment was Linux x86_64, Python 3.12.12, NumPy 2.3.5, SciPy 1.16.3,
and pytest 8.4.2, with single-threaded OpenMP/OpenBLAS and pytest plugin
autoload disabled. The five FEMFFUSION cross-section reader tests used local
reference inputs through an `examples/` symlink to the checkout's `Examples/`
directory. The separate fixed VVER mesh path was absent.

Artifacts: [environment and per-module results (JSON)](qa/cpu-2026-10-23.json),
[full pytest record (JUnit XML)](qa/cpu-2026-10-23.xml), and
[focused timeout recheck (JUnit XML)](qa/cpu-example-recheck-2026-10-23.xml).
The JSON records the source fingerprint and command. The README quickstart
also converged on CPU and returned the documented `(2, 24, 24, 24)` flux shape.

## Core numerical verification

| Feature | Reference and acceptance criteria | Automated checks |
|---|---|---|
| Cartesian multigroup diffusion | Exact bare-box eigenvalues and sine-mode flux; mesh doubling reduces eigenvalue error by a factor between 3 and 5. Reflective boundaries recover the infinite-medium eigenvalue within `1e-6`. | [Analytic diffusion](../tests/verification/test_diffusion_analytic.py), [boundary conditions](../tests/verification/test_boundary_conditions.py), [neutron balance](../tests/verification/test_neutron_balance.py) |
| Cylindrical r-z diffusion | Bare-cylinder Bessel-mode eigenvalue: finest-mesh absolute error below `5e-6`; successive errors decrease by about four. Changing the boundary specification at the zero-area axis leaves the answer unchanged within `1e-12`. | [Cylindrical tests](../tests/verification/test_cylindrical.py) |
| Triangular and prismatic geometry | Reflective tri-prisms recover infinite-medium `k` within `1e-9`; axial slab errors decrease by `4 ± 0.3` under refinement. Reflective extrusion of VVER-440 preserves its 2-D eigenvalue within `1e-7`. | [Tri-prisms](../tests/verification/test_tri_prisms.py), [triangular SP3](../tests/verification/test_tri_sp3.py), [hexagonal geometry](../tests/verification/test_hex_lattice.py) |
| Unstructured diffusion | Compare structured and unstructured finite volumes on matching meshes; check reflective limits and boundary treatments. The full-core 3-D HP-MR comparison is a separate slow test. | [Mesh verification](../tests/verification/test_unstructured_mesh.py), [3-D HP-MR comparison](../tests/validation/test_hpmr_3d_mesh.py) |
| SPN and SDPN | SP3 analytic eigenvalues and second-order refinement; independent angular-coefficient derivation; symmetry and transformed-operator equivalence. Coupled Marshak SP3/SDP1 eigenvalues extrapolate to the Brantley–Larsen literature values within `1e-4`. | [SP3 analytics](../tests/verification/test_sp3_analytic.py), [SDPN derivation](../tests/verification/test_sdpn_derivation.py), [operator symmetry](../tests/verification/test_sdpn_symmetric.py), [Brantley–Larsen](../tests/validation/test_brantley_larsen_2d.py) |
| Adjoint and perturbation theory | Forward and adjoint eigenvalues agree within `1e-7`. The adjoint-weighted absorption perturbation predicts a separately re-solved reactivity change within 2%. This also checks the importance flux. | [Adjoint](../tests/verification/test_adjoint.py), [perturbations](../tests/verification/test_perturbation.py) |
| SPH and homogenization | Reaction-rate preservation, material consistency, improved leaky-colorset eigenvalues, and preservation of fission-energy coefficients. Slow HP-MR checks require corrected diffusion to recover each selected transport-family reference within 15 pcm. | [SPH verification](../tests/verification/test_sph.py), [HP-MR SPH](../tests/validation/test_hpmr_sph.py), [reference families](../tests/validation/test_hpmr_sph_reference_families.py) |
| Cross-section input | Griffin material balances and conventions; FEMFFUSION files compared with independent benchmark transcriptions, including scattering direction and upscatter. | [Griffin readers](../tests/verification/test_griffin_xs.py), [FEMFFUSION readers](../tests/verification/test_femffusion_io.py) |
| Delayed-neutron transients | Uniform perturbations agree with an independent point-kinetics solution to relative error below `5e-4` in the specified diffusion and SDPN cases. Equilibrium, precursor hand-off, and restart invariants are checked separately. | [Point kinetics](../tests/verification/test_transient_point_kinetics.py), [material-dependent kinetics](../tests/verification/test_transient_per_material.py), [kinetics theory](../tests/verification/test_transient_kinetics_theory.py) |
| BDF time integration | Orders 1–6 differentiate polynomials exactly to floating-point tolerance; nonuniform history, precursor elimination, convergence order, event alignment, rejected-step rollback, and automatic order selection have explicit checks. | [Time schemes](../tests/verification/test_timescheme.py), [transient integration](../tests/verification/test_transient_point_kinetics.py) |
| Quasi-static kinetics | Exact homogeneous projections, normalization invariance, stationary equilibrium, and comparisons with full spatial transients. Reduced HP-MR ramps check power and temperature errors; guards check early shape corrections and full-diffusion fallback. | [Quasi-static tests](../tests/verification/test_quasistatic.py) |

The per-case tolerances above belong to those meshes and inputs. They are not
general accuracy guarantees for arbitrary geometries or cross-section sets.

## Reactor benchmark comparisons

Problem builders, cross sections, and reference constants live in
[`garnet/benchmarks/`](../garnet/benchmarks/). These cases test several levels
of agreement: published solutions, independent code results, and expected
physical behavior. They should be read with their modeling assumptions.

| Benchmark | What is compared | Evidence and scope |
|---|---|---|
| OECD/NEA C5G7, 2-D | Seven-group eigenvalue, geometry, homogenization, and SPN/SDPN hierarchy against the transport reference `k = 1.18655`. | [Tests](../tests/validation/test_c5g7.py). Pin-cell-homogenized calculations: the coarse diffusion test allows 2% relative error; the angular-family comparison allows 0.5%. This is not a pin-resolved transport validation. |
| BIBLIS and IAEA-3D | Eigenvalues, active masks, material symmetry, and Robin/albedo boundaries. | [Tests](../tests/validation/test_biblis_iaea.py). BIBLIS uses a coarse-mesh envelope; IAEA-3D checks absolute eigenvalue error below `1e-3` against `1.02903`. |
| VVER-440 | Body-fitted triangular geometry, eigenvalue, and a bulk transient; optional unstructured solve on the FEMFFUSION mesh. | [Tests](../tests/validation/test_vver440.py). The coarsest triangular test allows `6e-3` absolute eigenvalue error; the external-mesh comparison allows `1e-3`. |
| TWIGL step and ramp | Static eigenvalue, transient power samples, and the public Model API. | [Model tests](../tests/validation/test_model_twigl.py), [solver tests](../tests/validation/test_twigl_langenbuch.py), [worked example](model_api.md#worked-example-the-twigl-benchmark-validated-against-literature). Fine static `k` is within `1e-4` of `0.91321`; API transient samples are within `0.02` in normalized power. |
| Langenbuch/LMW | Three-dimensional moving rod-bank maps and transient peak/decline. | [Tests](../tests/validation/test_twigl_langenbuch.py). A coarse regression with a peak-power envelope, rather than a fine convergence study. |
| ANL-7416 8-A1 | Cylindrical static and transient results, including alternative Krylov solvers and divergence-form operators. | [Tests](../tests/validation/test_anl7416_8a1.py). Early power samples allow 3%; later published traces use a wider envelope, with an independent 4 s comparison. |
| LRA-2D with thermal feedback | Original ANL specification, reflector-data audit, static mesh convergence, power/temperature histories, adaptive BDF, and matched-error work. | [Tests](../tests/validation/test_lra2d.py), [study and data](../benchmark-results/lra2d-adaptive-bdf-report/), [PDF](../benchmark-results/lra2d-adaptive-bdf-report/adaptive-bdf-lra-benchmark.pdf). The archived 3.75 cm run reports first-peak power 3.48% above the original reference; finer-mesh and feedback-zone effects are documented. |
| HP-MR | Assembly counts, symmetry, absorber volume mixing, drum-worth ordering, mesh stability, 3-D extrusion, and SPH reference recovery. | [Geometry and worth tests](../tests/validation/test_hpmr.py), [SPH study tests](../tests/validation/test_hpmr_sph_reference_families.py). The default two-group data are placeholders. Checks against GARNET transport references establish numerical consistency, not independent validation of a physical microreactor. |

## Thermal and coupling checks

The conduction equation is checked against an exact hyperbolic-cosine slab
solution with both imposed-temperature and Robin boundaries. Measured spatial
orders must lie between 1.9 and 2.1. Cartesian, cylindrical, and triangular-prism
tests also check sink limits and energy conservation; the masked HP-MR steady
case requires a relative balance residual below `1e-10`.
See [conduction tests](../tests/verification/test_conduction.py).

[Power normalization](../tests/verification/test_power_density.py),
[feedback](../tests/verification/test_feedback.py), and
[coupled-solver tests](../tests/verification/test_coupling.py) check rated-power
normalization, feedback sign, zero-power and zero-feedback limits, stationary
coupled transients, thermal subcycling, and agreement between accelerators.
The coupled steady energy balance must close below `1e-10`; accelerators must
reach eigenvalues within `2e-8` of each other.

External coupling has a separate [preCICE comparison](../tests/validation/test_coupled_precice.py):
identity data mapping, iteration-by-iteration agreement under identical
relaxation, and agreement at the fixed point with different accelerators.
The [coupling report](coupling.md#6-verification) records a maximum eigenvalue
difference of `5e-13` over 33 lockstep iterations and a fixed-point difference
below `2e-8`. These are archived results; the tests require preCICE and its
Python bindings to reproduce.

## GPU and transient studies

CPU verification exercises the shared numerical implementation. CUDA kernel
execution, memory use, and performance need hardware measurements as well.

| Study | Recorded result | Scope |
|---|---|---|
| [Guarded IQS on GH200](../benchmark-results/hpmr-iqs-gh200/) | Maximum sampled power error of 0.173%, versus 5.708% for fixed-cadence IQS. The guarded 200 s run made 131 shape updates and two full-diffusion fallbacks. | 105,600 active cells, 11 groups, 20 axial layers. Accuracy was compared every 10 s through 110 s, where the full-diffusion reference ends. [Machine-readable record](../benchmark-results/hpmr-iqs-gh200/adaptive-fallback-r4-nz20.json). |
| [LRA adaptive-BDF study](../benchmark-results/lra2d-adaptive-bdf-report/) | Temporal/spatial convergence, controller histories, and a 5.18× CPU speedup over adaptive backward Euler at matched error. | An LRA-specific result. The report also records a coupled 11-group HP-MR case where BDF did not outperform matched adaptive backward Euler. |
| [T4 3-D adaptive/monolithic study](../benchmark-results/transient-acceleration/step-08a-adaptive-bdf-gpu-readiness.md) | Accuracy, energy-group batching, persistent Krylov memory, state-update microbenchmarks, and repeated timing comparisons. | Quick case: 13,200 active cells and 145,200 flux unknowns. Larger production and reference-convergence checks remain pending. |
| [Energy-sweep acceleration study](../benchmark-results/transient-acceleration/step-10a-energy-sweep-anderson.md) | CPU and T4 configurations were evaluated separately; the T4 fixed-polynomial configuration gave a 1.101× median gain over the plain configuration. | Three interleaved T4 samples. Tolerance-based energy Anderson was rejected on T4 despite its CPU benefit. |

The [optimization index](../benchmark-results/transient-acceleration/) retains
accepted and rejected experiments. Performance claims belong to the stated
hardware, precision, mesh, and accuracy target.

## Additional solver checks

The lower-level noise solver has exact zero-power transfer-function tests:
complex amplitude and phase errors below `1e-6`, with spatial flatness checked
separately. A stored FEMFFUSION slab reference checks both diffusion and SP3
noise, requiring relative field L2 error below 1% on the 60-cell mesh.
See [point-kinetics noise](../tests/validation/test_noise_point_kinetics.py)
and [FEMFFUSION noise](../tests/validation/test_noise_femffusion.py).

Discrete-ordinates and hybrid solvers have their own
[Cartesian](../tests/verification/test_sn.py),
[triangular](../tests/verification/test_tri_sn.py), and
[hybrid-interface](../tests/verification/test_hybrid_tri_sn.py) tests. Their
coverage is separate from the core diffusion/coupled workflow described above.

## Reproducing the checks

From an installed checkout:

```bash
python -m pip install -e '.[test]'
python -m pytest tests -ra
python -m pytest tests -m slow -ra     # expensive cases only
python -m pytest tests -o addopts= -ra # all cases, including slow
```

The default marker selection is defined in [pyproject.toml](../pyproject.toml).
`FEMFFUSION_DIR` must point to reference inputs with an `examples/` directory
for the file-reader tests. Some mesh tests use a separate fixed external path;
inspect the skip reasons with `-ra`. The FEMFFUSION noise comparison embeds
its reference and does not require a live FEMFFUSION solver.

Install the CUDA extra and use actual GPU hardware for CUDA-only kernel tests.
The [GPU regression job](../slurm/run_garnet_gpu_regression.sh) checks that CuPy
can see a device and clears the default slow-test exclusion; many physics
tests still explicitly select the CPU. For CPU/GPU solve comparisons use the
[benchmark notebook](../notebooks/colab_cpu_gpu_benchmarks.ipynb); for coupled
transients use the [GPU walkthrough](../notebooks/colab_coupled_transient_gpu_latest.ipynb)
and [3-D adaptive-BDF notebook](../notebooks/colab_hpmr3d_adaptive_bdf_gpu.ipynb).
preCICE installation and its test command are in the
[coupling guide](coupling.md#running-it).

The [July theory report](report.html) retains the original discretization
derivations and benchmark snapshot. Use this report and the
[user guide](user_guide.md) for the current coverage and supported workflows.
