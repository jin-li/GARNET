# Test taxonomy and coverage map

The repository has three active layers: focused feature modules directly in
`tests/`, mathematical/API checks in `tests/verification/`, and benchmark
comparisons in `tests/validation/`. `tests/legacy/` is a separate opt-in comparison
with earlier low-level entry points. Its retained cases include fixture
adaptations made during integration: in
`legacy/verification/test_transient_point_kinetics.py`, the monolithic-feedback
case requests tighter eigenvalue/source convergence and asserts balance, and the
schedule-validation case expects the current BDF/FTM error wording. The archive
is not a byte-for-byte snapshot of every original module. Consult source history when reproducing an
earlier revision.

## Root feature modules

These are active default-suite tests, not miscellaneous files outside the test
policy. Paths below are relative to `tests/`; related modules extend each topic.

### Serpent moments

`test_serpent_moments.py`, `test_serpent_adf_reader.py`

Synthetic native records, signed pairs, explicit group/moment orientation, face-major ADF layout, uncertainty/geometry validation and malformed input rejection. Parsing a synthetic record does not qualify a native estimator or a full library.

### Mesh and interface factors

`test_mesh_df.py`, `test_mesh_moment_df.py`, `test_adf_fixed_source_integration.py`, `test_reflector_interface_nem_integration.py`, `test_face_conventions.py`

Directed interface factors, forward/transpose behavior, dense multigroup fixed-source pencil and reflector interface integration. Mesh, Cartesian and NEM factor contracts are distinct; do not transfer qualification between them.

Asymmetric face values and round trips check the shared ordering utility;
degree-policy checks retain the supported orders and three-level reference
qualification. Array reordering does not convert current normalization or
homogeneous reference physics.

### NEM reference operators

`test_nem.py`, `test_nem_reference.py`, `test_nodal_polynomial_reference.py`, `test_nem_polynomial_axis.py`

Independent augmented systems, reference/axis polynomial identities and NEM assembly. CPU cuboid local-method evidence, not a general mesh or transient NEM solver.

### Higher-order NEM and convergence

`test_nem_higher_order.py`, `test_nem_boundary_leakage.py`, `test_nem_requested_convergence.py`

Higher-order reconstruction, boundary leakage, separate eigenvalue/source/balance gates and compatibility defaults. Convergence must be checked against the requested controls and full residual.

### Coupled moments

`test_coupled_moments.py`, `test_coupled_moment_solver.py`, `test_coupled_moment_gpu.py`, `test_coupled_mesh_adapter.py`

Noncommuting full moment blocks, independent operator/dense references, solver budgets, adapters and optional real devices. Native coupled moments are a research pathway; scalar or diagonalized examples cannot establish full-block correctness.

### Branch cross sections

`test_branch_xs.py`, `test_branch_io.py`, `test_branch_closure_identity.py`, `test_branch_generic.py`

Immutable absolute channels/full fission matrices, convex interpolation, safe serialization, file hashes, explicit identities and version policy. Metadata compatibility and interpolation algebra do not validate reactor predictions.

Non-benchmark, three-group fixtures check custom moderator paths and nested
compatibility identities. Metric-only reports cannot turn unassessed errors
into acceptance; historical thresholds remain an explicit compatibility path.

### Thermal and rod branch workflows

`test_branch_thermal.py`, `test_branch_rod.py`, `test_branch_workflow.py`

Thermal-star domains, withheld roles, rod geometry/lineage and fresh NEM end-to-end limits. Synthetic homogeneous checks cannot supply missing external transport validation.

### Branch validation

`test_branch_validation.py`, `test_branch_nem_validation.py`

Reactivity units, uncertainty/spacing, ordered power bins, interpolation-only gates and independent returned-field/balance validation. Passing interpolation comparisons does not waive closure or geometry qualification.

### Kinetics production

`test_kinetics_production.py`, `test_kinetics_families.py`, `test_kinetics_yield_transfer.py`, `test_kinetics_workflow.py`, `test_nem_kinetics.py`

Raw delayed-production channels, native/reduced families, extreme representable tails, physical-time projection and matching forward/adjoint workflows. Static kinetics projection does not establish transient NEM dynamics.

### Source convergence and examples

`test_eigen_source_convergence.py`, `test_isonees_examples.py`

Independent source/balance gates, failure states and runnable synthetic workflow invariants. A stable eigenvalue alone is insufficient convergence evidence.

## Mathematical verification and API checks

Representative modules in `tests/verification/` include:

- `test_diffusion_analytic.py`, `test_cylindrical.py`, `test_sp3_analytic.py` and
  `test_tri_prisms.py`: analytic eigenvalues, refinement order and geometric
  limits. `test_neutron_balance.py` and `test_conduction.py` check conservation
  and independent thermal references.
- `test_adjoint.py`, `test_perturbation.py`, `test_linalg_krylov.py` and
  `test_multigroup_step.py`: forward/adjoint orientation, perturbation identities,
  nonsymmetric linear solves and multigroup convergence behavior.
- `test_cartesian_adf.py`, `test_dallosso_adf_reference.py` and
  `test_serpent_fa_2d.py`: scoped interface-factor and reference problems. Read
  each fixture's convention and normalization before interpreting agreement.
- `test_transient_point_kinetics.py`, `test_transient_model_api.py`,
  `test_exponential_damping.py`, `test_exponential_adaptive.py`,
  `test_exponential_output.py` and `test_exponential_twigl.py`: ODE/matrix
  exponential references, physical-state errors, events, rejected-step rollback,
  output integrals and spatial generator checks. Passing these does not promote
  experimental frequency-transform methods over tuned adaptive BDF.
- `test_ace_precursors.py`, `test_ace_delayed_spectra.py`,
  `test_serpent_kinetics.py` and `test_kinetics_data.py`: synthetic nuclear-data
  layouts, units, provenance and kinetics contracts.
- `test_examples_current.py`: example compilation/imports, a documented entry
  point, public metadata and the Colab source archive's packaged-file contents.
  Compilation is not execution of every example.

Some verification tests are marked `slow`, including symbolic derivation and
full-core homogenization checks. Use the selection commands in [Running and
recording checks](running.md), rather than assuming this directory is fast.

## Benchmark validation

`garnet/benchmarks/` contains importable problem builders and reference constants.
The tests document which model and observable are compared; a benchmark name
alone does not define an acceptance criterion.

- `test_c5g7.py`, `test_biblis_iaea.py`, `test_vver440.py`: Named core eigenvalue/reference-code comparisons; distinguish a diffusion approximation from the published transport result and inspect reference tolerances.

- `test_twigl_langenbuch.py`, `test_anl7416_8a1.py`, `test_lra2d.py`, `test_c5g7_td.py`: Published transient problems and power histories with their stated meshes, time steps and kinetics assumptions.

- `test_hpmr.py`, `test_hpmr_3d_mesh.py`: HP-MR behavior, symmetry and mesh checks; placeholder-data behavior is not reactor-data validation.

- `test_hpmr_sph.py`, `test_hpmr_sph_reference_families.py`: Expensive SPH/reference-family consistency and comparisons; selected cases are marked slow.

- `test_coupled_precice.py`: Optional external-coupling agreement with the internal solver and mapping/iteration contracts; requires the external library.

`benchmark-results/` and development reports preserve additional scoped study
evidence. Timing harnesses such as `examples/speed_benchmark.py` measure execution
cost; speed alone does not establish physics accuracy.
