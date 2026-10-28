# Assembly discontinuity factors for GARNET finite volumes

Assessment and development plan, 27 October 2026. Updated after reading the
published Bernal et al. paper supplied by the user.

## Recommendation

Implement a supported, input-driven assembly discontinuity factor (ADF) path
for the steady multigroup diffusion solver, starting on Cartesian meshes and
reusing the existing triangular face formulation. This is feasible with a
matrix-free GPU stencil. It is worth pursuing when compatible homogenized
cross sections and face factors are available and assembly homogenization is a
material source of error. The benefit and computational cost must be measured;
ADFs are not a substitute for transport, an accurate spatial discretization,
or state-dependent group constants.

GARNET already contains a triangular implementation, but it is a prototype,
not a complete GPU-ready ADF workflow. The first deliverable should be a
verified, device-resident steady-state implementation and an honest benchmark,
not a general rewrite of all approximations and transient solvers.

## Literature

- Bernal, Román, Miró and Verdú, [Assembly discontinuity factors for the
  neutron diffusion equation discretized with the finite volume method:
  Application to BWR](https://doi.org/10.1016/j.anucene.2016.06.023),
  *Annals of Nuclear Energy* 97 (2016), 76–85. This is direct finite-volume
  precedent. The full paper confirms the factor convention and conservative
  interface conditions below, but its polynomial reconstruction and
  cell-and-face-current unknowns differ from GARNET's two-point stencil. Its
  simple test provides cross sections, factors, geometry figures and comparison
  results for an additional benchmark; some details need resolution before
  claiming exact reproduction. See the paper review below.
- Vidal-Ferràndiz et al., [Use of discontinuity factors in high-order finite
  element methods](https://doi.org/10.1016/j.anucene.2015.06.021), *Annals of
  Nuclear Energy* 87 (2016), 728–738. ADFs and reference DFs are imposed in an
  interior-penalty discontinuous Galerkin method; the authors report nodal
  method agreement and discuss the extra difficulty of reference factors in
  two dimensions.
- Nichita, Zabienski and Gravel, [A three-dimensional two-energy-group
  finite-difference neutron-diffusion code with discontinuity factors](https://proceedings.cns-snc.ca/index.php/pcns/article/view/5899)
  (2007), provides a related cell-centered finite-difference implementation.
- [Serpent's ADF output documentation](https://serpent.vtt.fi/mediawiki/index.php/Output_parameters)
  describes surface flux/current tallies and different treatments for
  reflective and nonzero-current cases. Imported data must be checked against
  the exact conventions and diffusion model used to generate them.
- [Asymptotic, multigroup flux reconstruction and consistent discontinuity
  factors](https://doi.org/10.1080/00223131.2015.1041568) explains why
  factors from an isolated lattice are approximations to whole-core reference
  factors. Exact reference equivalence is a verification device, not evidence
  that isolated-assembly ADFs will be exact for every environment.

## What the full Bernal paper changes

The recommendation remains to implement input-driven ADFs in the existing
two-point method first. The paper strengthens the case for doing so, but it
also makes a **mesh-resolution study and an independent heterogeneous reference
essential**. In particular:

- Equations (8)–(10) use a common net current and define the factor as the
  heterogeneous-to-homogeneous surface-flux ratio, exactly the convention used
  here. Their quantities are face and volume averages (Eqs. 12–14), so
  comparisons should use integrated currents and surface averages.
- This is a higher-order finite-volume method, not the proposed two-point
  method. The paper reconstructs each cell's flux from N_faces + 1 polynomial
  terms. For hexahedra it uses 1, x, y, z, x², y², z². It retains a cell flux
  and one unknown current per interior face per group; current continuity and
  boundary conditions are built into that system, while corrected surface-flux
  continuity supplies face equations (Eqs. 15–29). GARNET instead eliminates
  the face current using independent linear half-cell resistances. Thus the
  same interface physics does not imply the same coarse-mesh solution or the
  same number of unknowns. Porting Bernal's method would be a separate
  higher-order project, not a prerequisite for ADF support.
- The authors solve a two-group steady eigenproblem using Krylov–Schur with
  GMRES and additive Schwarz for inner systems. Their reported matrix
  conditioning and timing belong to their particular reduced formulation and
  implementation; they do not establish GPU performance or preconditioner
  quality for GARNET.
- In the simple six-material case (Section 3.1; Figs. 4–6 and Table 1), the
  base model has 1,170 nodes of 20 cm per side and 10 axial levels, with
  zero-flux boundaries. The paper supplies two-group XS and ADFs, reports one
  common ADF for the four lateral faces, and subdivides each base node 2³, 3³,
  and 4³ for structured refinement. The quoted VALKIN and PARCS fundamental
  eigenvalues are 1.040374 and 1.041186, respectively. Table 2 reports the
  polynomial FVM's relative eigenvalue errors against VALKIN, from +78.26 pcm
  on mesh 1 to +11.19 pcm on mesh 4, and maximum assembly-power errors of 1.98%
  and 0.30%, respectively. These are comparisons against a nodal diffusion
  calculation, not error against an exact or heterogeneous transport solution.
  The paper defines its pcm error as 10⁵(k − k_ref)/k_ref, so use that
  definition when comparing its tables.
- The realistic BWR comparisons likewise use VALKIN and PARCS as references,
  not heterogeneous transport; the paper does not provide an ADF-off
  comparison. For the rodded case, its Table 13 reports maximum assembly-power
  differences from VALKIN of 10.83% on mesh 1 and 4.86% on mesh 4. The authors
  observe the largest differences near rods and suggest large ADF differences
  as a possible cause, not a demonstrated diagnosis. Their results support
  refinement and strong-contrast tests, not a general claim that ADFs improve
  accuracy or that GARNET will achieve these errors.

For the simple case, the paper says the inputs permit reproduction, but the
material layout is depicted graphically, the stated factor uniformity covers
the four lateral faces, and Table 1 lists νΣ_f while its power definition uses
Σ_f. Confirm the full spatial map, axial-factor treatment and power conversion
before treating it as an exact benchmark. Keep any assumptions explicit and
compare eigenvalue and normalized fission production independently of an
unverified power conversion. Do not apply an assembly factor at artificial
interior faces created by mesh refinement.

## Interface formulation

For one energy group, take an interior face with normal from cell L to cell R,
positive diffusion coefficients D, center-to-face distances d, and factors
f defined as heterogeneous surface flux divided by homogeneous diffusion
surface flux. Let J be current density from L to R. Reconstruct homogeneous
face values using the existing two-point approximation:

    phi_s,L = phi_L - J d_L / D_L
    phi_s,R = phi_R + J d_R / D_R
    f_L phi_s,L = f_R phi_s,R
    J = (f_L phi_L - f_R phi_R) /
        (f_L d_L / D_L + f_R d_R / D_R).

The same integrated current A_face J enters the L balance and leaves the R
balance, preserving conservation. With f_L = f_R = 1, this is the usual
two-point harmonic-resistance flux. A cell-centered unknown is not itself an
interface flux: apply the factors to reconstructed face values, only at
homogenization-region interfaces. A single face-averaged factor distributed
across its subfaces is an approximation, especially with strong transverse
gradients. ADFs correct homogenization error; they do not remove the spatial
error of a coarse two-point stencil. Refine the mesh independently.

The left and right matrix rows use different neighbor coefficients. Arbitrary
face-specific factors therefore generally produce a nonsymmetric operator; the
existing CG default is not valid for that operator. Positive face coefficients
alone do not guarantee that Jacobi or a polynomial preconditioner will converge
quickly. Equal factors on both sides cancel, including equal non-unit values.
The face formulation is invariant under a common rescaling of its two factors,
but each side's actual factor is still needed for reconstructed surface flux
and partial-current reporting.

For an external Robin boundary, do not silently reuse an interior face factor.
First define whether the boundary law acts on homogeneous or reconstructed
physical surface flux. If it is J = alpha f phi_s, the effective Robin
coefficient is alpha f; this is a modeling convention, not an automatic
consequence of every supplied ADF. Reflective boundaries remain zero-current.
For the initial release support interior factors only, with existing boundary
conditions explicitly documented; add external factors only with compatible
reference data and dedicated tests.

## Current repository state

- [`TriGroupOperator`](../garnet/tri.py) already implements this two-sided
  current for its three in-plane face families, using per-cell factors or a
  six-array per-face tuple. [`face_df_from_pairs`](../garnet/tri.py) builds
  factors from ordered region pairs, and [`tri_partial_currents`](../garnet/tri.py)
  reconstructs in-plane currents. A region-pair key alone cannot distinguish
  multiple geometrically distinct interfaces between the same two regions.
- The triangular fused stencil in [`kernels.py`](../garnet/kernels.py) already
  reads two directed weights per in-plane face, and
  [`test_fused_kernels.py`](../tests/verification/test_fused_kernels.py) checks
  CPU emulation against the unfused operator with non-unit per-cell factors.
  That is useful implementation evidence, not validation of ADF physics or
  GPU end-to-end performance.
- `TriDiffusionEigenSolver` accepts `df`, but defaults are inherited from the
  general solver: the code does not currently reject CG for non-unit factors.
  It unconditionally builds a SciPy ILU when `df` is supplied; its
  preconditioner copies vectors to the host and back on GPU. This can dominate
  a GPU solve. The extruded triangular axial coupling remains symmetric and
  ignores `df`. The tuple representation needs a clear per-group public
  contract and validation.
- [`sph_df_correct`](../garnet/sph.py) is an explicitly shelved, coupled
  SPH/DF/boundary-coefficient fitting experiment. Its comments report poor
  HP-MR results for the tested strong-absorber case, and difficulty matching
  transport partial currents using diffusion's P1 reconstruction. These are
  case-specific observations, not a general ADF validation or a reason to
  discard conventional externally generated ADFs. Its fitting procedure and
  the ordered-region-pair factors are not an assembly-library import workflow.
- Cartesian [`GroupOperator`](../garnet/stencil.py) and the unstructured
  [`_MeshGroupOperator`](../garnet/mesh.py) currently build symmetric face
  weights. [`model.py`](../garnet/model.py) does not provide a documented
  assembly-face ADF model interface. Transient and other approximation paths
  must not be assumed to propagate or interpret ADFs.

## Development stages and acceptance gates

1. **Specify data and scope.** Introduce a separate assembly/region map, not
   a material-ID boundary test: distinct assemblies can share a material and
   the same material can appear in different states or orientations. Define
   group order, factor convention, face orientation, rotations/reflections,
   axial segmentation, subface mapping, and units. Store per-region-side,
   per-group, per-physical-face factors independently of materials. Require
   positive finite values, compatible groups and complete interface coverage;
   unspecified interfaces must either be explicitly unity or fail clearly.
   Reject unsupported geometries, boundary corrections, and solver paths
   rather than silently dropping data. Record provenance and state for XS and
   factors. First delivery: static Cartesian, orthogonal mesh aligned to
   interfaces, externally supplied factors and steady diffusion.

2. **Implement and verify the local operator.** Add directed face weights and
   correct diagonal contributions in [`stencil.py`](../garnet/stencil.py),
   using the current's integrated/volume convention consistently. Retain the
   existing fast symmetric path for absent or unity factors. Preserve masks
   and physical boundary treatments. Reject CG when the active operator is
   nonsymmetric, or select an explicitly documented nonsymmetric solver;
   provide an exact safe rule for when CG is allowed. Refactor/share the
   triangular arithmetic where useful, without silently changing old results.
   Supply face-current diagnostics that reconstruct both homogeneous traces,
   corrected trace, and oriented net current.

   Gate: asymmetric two-cell analytic cases with unequal D, distances and
   factors; equality of reconstructed corrected traces; equal-and-opposite
   integrated currents and global neutron balance; unity and equal-factor
   regression; factor-side reversal; different factors on different faces;
   masks and physical boundaries. Compare matrix-free application and solves
   with an independently assembled small matrix, including heterogeneous
   removal and multiple groups. Add explicit invalid input and CG rejection
   tests. Check floating-point tolerances rather than assuming bitwise identity.

3. **Make the solve device-resident.** Extend the Cartesian fused kernel with
   directed weights, retaining the no-ADF fast path. Start with existing GPU
   BiCGStab/GMRES and Jacobi, measure robustness on representative factor
   contrasts, and compare CPU ILU only as a reference. If needed, evaluate a
   GPU-capable preconditioner or a nonsymmetric Krylov solve preconditioned by
   the symmetric no-ADF operator. A diagonal similarity transform can recover
   symmetry only when compatible per-cell scalings exist around every cycle;
   do not assume arbitrary face-specific ADFs satisfy that constraint.
   Measure actual GPU host transfers, setup time, iteration counts, wall time,
   peak memory and physical residual; keep factors and iterative work on
   device. Benchmark ADF off/on on the same mesh and compare CPU/GPU k,
   normalized flux, power, currents and residuals. Treat performance thresholds
   as benchmark decisions, not claims made before measurement.

4. **Demonstrate physics value.** First reconstruct Bernal's simple 3-D case
   from the paper where its inputs are unambiguous. Document material maps,
   axial-factor and power assumptions; independently check them before
   asserting reproduction. Apply lateral factors at original assembly faces,
   use unity across artificial subdivision faces, and refine each base node
   1³, 2³, 3³ and 4³. Compare against the paper's published VALKIN and PARCS
   k and assembly power metrics as method-to-method comparisons, not exact
   targets for a two-point coarse mesh. Also report the GARNET ADF-on/off
   difference and convergence of interface currents. Include the largest
   factor contrasts and strong absorbers in the solver performance study.

   Separately build a reproducible 1-D two-region
   manufactured/reference-DF case and a heterogeneous 2-D colorset with
   compatible group constants and independently generated ADFs. Compare
   heterogeneous reference, homogenized no-ADF, and homogenized ADF results at
   multiple mesh resolutions: k in pcm, assembly powers and reaction rates,
   interface currents and global balance. Align eigenvector normalization and
   report pointwise as well as integrated quantities. Include uncertainty and
   generation conditions for Monte Carlo data. Reference factors can verify
   equivalence in their reference configuration when generated consistently
   with the discrete low-order operator and its mesh; isolated-lattice ADFs
   must be assessed separately in varied neighboring configurations. Report
   regressions as well as improvements. Adopt the feature for a target use
   only if it improves relevant observables at acceptable solver cost.

5. **Harden the triangular path and expand only on evidence.** Support
   geometry-specific face identity, rotation and axial factors; extend the
   fused and unfused paths consistently, with explicit 2-D/3-D tests. Replace
   the host-ILU dependency for GPU use and fix solver selection. Extend to
   unstructured meshes only after face areas, both center-to-face distances,
   orientation, and mesh orthogonality are represented and validated; the
   current mesh operator uses a single center distance and a harmonic mean.
   Consider Cartesian cylindrical metric weighting separately. Expose the
   feature through model APIs and user documentation after the low-level path
   passes the gates. Update adjoint/transpose, transient initialization and
   time-dependent factor handling, and SPN/SP3 only in separately verified
   follow-on work; reject these combinations until implemented. A time-varying
   library requires consistent state interpolation and operator/preconditioner
   rebuilds, including at events.

## Open inputs and decisions

- Resolve the simple Bernal benchmark's figure-derived material layout,
  axial-factor convention, and fission-power coefficients, ideally against
  author-supplied input or an independent calculation. The realistic BWR's full
  cross-section set is not provided in the paper, so its results cannot be
  independently reproduced from that paper alone.
- Identify the target reactor, geometry, available assembly XS/ADF library,
  state variables, group ordering and intended boundary conventions. The
  initial Cartesian-first recommendation is conditional on that use case.
- Decide whether to support external-surface factors and which solver and
  memory cost are acceptable after the steady Cartesian benchmark. Avoid
  combining SPH and ADF corrections without a consistent derivation and
  jointly generated data.

## Development started

The initial Cartesian steady-state face operator, nonsymmetric solve guard,
region-to-face preprocessing and fused GPU implementation are in place. See
[the implementation and provisional Bernal results](bernal_adf_benchmark.md)
for the API, test coverage, exact assumptions and remaining validation. The
GPU path still requires execution on CUDA hardware; the Bernal case is not yet
a verified reproduction.

The Cartesian adjoint now applies the algebraic transpose of the ADF leakage
operator, with an adjoint-specific polynomial preconditioner. A one- and
two-group analytical reference-DF study and a preliminary published two-group
slab comparison are described in [adf_slab_reference.md](adf_slab_reference.md)
and [dallosso_1d_reference.md](dallosso_1d_reference.md). These do not resolve
the Bernal input ambiguity or constitute transport validation.

### Update, 28 October 2026: GPU readiness and colorset study

An audit of the 27 October code confirmed the operator algebra: directed
weights, diagonals, transpose, CSR assembly and the fused-kernel indexing all
agree with the generic stencil. It also found the following, now addressed:

- The ADF CUDA kernel was derived from the symmetric kernel source by
  unchecked string replacement. Each edit must now match exactly once, or
  kernel construction fails; a test covers this.
- The only GPU test was a forward apply in float64 on a 4×3×2 grid. It is
  replaced by a parametrized fused-vs-unfused-vs-CPU test (three shapes
  including an empty face axis; float32 and float64; with and without masks;
  forward and transpose; with and without `out=`), plus GPU vs CPU forward and
  adjoint eigensolves and a Bernal base-case comparison.
- Stage 2's trace diagnostic was missing. `GroupOperator.face_traces` now
  returns both homogeneous surface-flux reconstructions. `region_side_traces`
  and `expand_region_side_factors` (in `garnet/adf.py`) take per-region,
  per-side traces and factors to solver input.
- `examples/adf_slab_reference.py` and `examples/bernal_adf_benchmark.py`
  violated `test_examples_current.py` (`PYTHONPATH=.` in their usage strings);
  this is fixed.
- The Bernal example compared CG without ADFs against GMRES with ADFs.
  `tools/assess_adf_gpu.py` separates the two: ADF off with CG and with GMRES,
  explicit unity factors, and Table 1 factors with GMRES and BiCGStab. It also
  has a factor-contrast × preconditioner study.

Stage 4's first independent-factor case is the
[C5G7 colorset study](c5g7_colorset_adf.md). Two-point nodes of assembly size
cannot hold admissible reference DFs for C5G7, and coarse-mesh error swamps the
ADF effect there. At pin-pitch nodes, isolated-assembly ADFs improve assembly
powers and currents in two of three layouts, but bias k by 5–20 pcm. Reference
XS plus DF reproduce the reference exactly.

The GH200 runs are prepared but not yet executed; see
[the campaign README](../benchmark-results/adf-gh200/README.md). Stage 3
remains open until those runs pass.
