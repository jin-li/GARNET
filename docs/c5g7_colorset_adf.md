# C5G7 UO2/MOX colorsets with isolated-assembly ADFs

This study tests the full ADF workflow in a case where the factors are **not**
generated from the answer. Assembly cross sections and ADFs come from isolated
assemblies and are then used in colorsets with different neighbours. It is a
diffusion-to-diffusion equivalence study on pin-cell-homogenized C5G7 data. It
says nothing about transport accuracy, and the fine reference is itself a
discretized diffusion solution.

Run it with `python examples/c5g7_colorset_adf.py` (see `--help`). The fast
regression tests are in `tests/verification/test_colorset_adf.py`.

## Definition

- **Data:** C5G7 pin-cell-homogenized materials from
  `garnet.benchmarks.c5g7.build_c5g7_2d`: UO2, the three MOX enrichments, guide
  tube and fission chamber, with 7 groups including upscatter. Assembly power
  is the Σf reaction rate (C5G7 supplies Σf), normalized to mean 1.
- **Geometry:** 17×17 assemblies of 21.42 cm, 2-D (one 1 cm reflective axial
  cell), reflective on every outer face, `s` fine cells per pin pitch.
- **Layouts** (`layout[row][col]`, x = column):
  - `checker2`: UM / MU, an infinite checkerboard;
  - `checker3`: MUM / UMU / MUM. The centre MOX sees four UO2 neighbours, each
    corner MOX only two;
  - `island3`: UUU / UMU / UUU, one MOX assembly in UO2.
- **Reference:** the fine-mesh heterogeneous solve of the colorset.
- **Library:** each assembly type is solved alone, reflected, on the same fine
  mesh. Cross sections are flux-volume weighted
  (`garnet.sph.flux_weighted_homogenize`). The homogeneous reflected solution
  is flat, so each side's ADF is the side-averaged boundary-cell flux over the
  assembly average. At a zero-current boundary that is the discrete two-point
  surface flux.
- **Coarse solves:** N×N homogenized nodes per assembly, N ∈ {1, 2, 17}, all
  with GMRES. Library ADFs are applied only on assembly faces
  (`expand_region_side_factors` → `assembly_face_factors`), never on the
  internal faces of a refined assembly.
- **Reference XS + DF:** a node-level equivalence check. Every coarse node gets
  cross sections homogenized from the colorset reference. Every node face gets
  the discrete-consistent factor `f = φ_het,surface / (φ̄ − J_out·h/(2D))`,
  using the reference node-average flux and outgoing current density
  (`region_side_traces` on the fine solution). With these the reference node
  fluxes satisfy the coarse equations exactly. That checks trace
  reconstruction, factor expansion, the directed operator and the solver
  together.

## Results (CPU, 4 cells per pin)

Δpcm = 10⁵(k − k_ref). Pmax/Prms are assembly-power errors in %. Jmax is the
largest assembly-interface net-current error (group sum, both solutions
normalized to unit fission production), relative to the largest reference
interface current. Iterations are outers / inners. Times are single runs on
`lclrs81` (x86_64, NumPy 2.3.5, one thread). The GPU runs are pending (see
[the GH200 campaign](https://github.com/jin-li/GARNET/tree/main/benchmark-results/adf-gh200)).

Isolated-assembly library at s = 4: k∞ is 1.3406619 for UO2 and 1.1743673 for
MOX. The thermal-group (g7) ADF is 0.879 for UO2 and 1.126 for MOX, and g6
is 0.950 and 1.074; groups 1–5 are within 2% of unity. Each assembly's four lateral factors
agree to round-off.

**checker2**: k_ref = 1.2606556 (18,496 fine cells)

| case | N | k | Δpcm | Pmax | Prms | Jmax | it | t [s] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| library XS, no ADF | 1 | 1.2694422 | +878.7 | 13.66 | 11.86 | 0.108 | 38 / 273 | 0.06 |
| library XS + ADF | 1 | 1.2693396 | +868.4 | 13.62 | 11.83 | 0.114 | 38 / 272 | 0.06 |
| library XS, no ADF | 2 | 1.2644461 | +379.1 | 6.37 | 5.53 | 0.107 | 39 / 345 | 0.07 |
| library XS + ADF | 2 | 1.2643741 | +371.8 | 6.44 | 5.60 | 0.117 | 33 / 297 | 0.06 |
| library XS, no ADF | 17 | 1.2613331 | +67.7 | 0.34 | 0.30 | 0.019 | 34 / 1632 | 0.32 |
| library XS + ADF | 17 | 1.2614292 | +77.4 | 0.39 | 0.34 | 0.012 | 35 / 1642 | 0.31 |
| reference XS, no DF | 17 | 1.2589489 | −170.7 | 0.29 | 0.26 | 0.006 | 40 / 1906 | 0.36 |
| reference XS + DF | 17 | 1.2606556 | −0.0 | 0.00 | 0.00 | 0.000 | 37 / 1766 | 0.34 |

**checker3**: k_ref = 1.2471264 (41,616 fine cells)

| case | N | k | Δpcm | Pmax | Prms | Jmax | it | t [s] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| library XS, no ADF | 1 | 1.2571052 | +997.9 | 14.62 | 13.94 | 0.202 | 62 / 581 | 0.10 |
| library XS + ADF | 1 | 1.2570103 | +988.4 | 14.60 | 13.94 | 0.208 | 44 / 420 | 0.08 |
| library XS, no ADF | 2 | 1.2515868 | +446.0 | 7.24 | 6.75 | 0.151 | 33 / 326 | 0.06 |
| library XS + ADF | 2 | 1.2515453 | +441.9 | 7.54 | 6.90 | 0.163 | 45 / 437 | 0.08 |
| library XS, no ADF | 17 | 1.2479383 | +81.2 | 0.81 | 0.44 | 0.020 | 34 / 1646 | 0.46 |
| library XS + ADF | 17 | 1.2481406 | +101.4 | 0.60 | 0.52 | 0.019 | 33 / 1580 | 0.43 |
| reference XS, no DF | 17 | 1.2456209 | −150.6 | 0.61 | 0.42 | 0.010 | 33 / 1604 | 0.45 |
| reference XS + DF | 17 | 1.2471264 | +0.0 | 0.00 | 0.00 | 0.000 | 36 / 1662 | 0.46 |

**island3**: k_ref = 1.3226297 (41,616 fine cells)

| case | N | k | Δpcm | Pmax | Prms | Jmax | it | t [s] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| library XS, no ADF | 1 | 1.3258610 | +323.1 | 20.56 | 7.09 | 0.045 | 49 / 388 | 0.08 |
| library XS + ADF | 1 | 1.3258305 | +320.1 | 20.55 | 7.09 | 0.044 | 58 / 481 | 0.10 |
| library XS, no ADF | 2 | 1.3240643 | +143.5 | 9.94 | 3.47 | 0.047 | 43 / 420 | 0.08 |
| library XS + ADF | 2 | 1.3240433 | +141.4 | 10.12 | 3.54 | 0.058 | 39 / 368 | 0.07 |
| library XS, no ADF | 17 | 1.3228717 | +24.2 | 0.93 | 0.32 | 0.024 | 39 / 1729 | 0.47 |
| library XS + ADF | 17 | 1.3229200 | +29.0 | 0.43 | 0.19 | 0.005 | 44 / 1885 | 0.52 |
| reference XS, no DF | 17 | 1.3203376 | −229.2 | 0.55 | 0.21 | 0.004 | 45 / 2048 | 0.56 |
| reference XS + DF | 17 | 1.3226297 | −0.0 | 0.00 | 0.00 | 0.000 | 41 / 1906 | 0.51 |

The reference-DF rows for N = 1 and 2 are omitted above because no admissible
factor exists there; see the findings below. The s = 2 run gives the same
pattern. Its fine k_ref is 33–52 pcm lower and its N = 17 library-ADF power
improvements are larger: Pmax drops from 0.42 to 0.28% in `checker2`, from
0.93 to 0.43% in `checker3`, and from 1.06 to 0.25% in `island3`. That run's
JSON is `benchmark-results/adf-gh200/colorset-cpu-s2-s4.json`.

## Findings

1. **The equivalence pipeline is exact where it can be.** With reference
   cross sections and reference DFs, the N = 17 coarse solve reproduces the
   fine k to < 0.05 pcm and the assembly powers and interface currents to
   round-off in all three layouts. A caveat: with pin-cell-homogenized fine
   data, N = 17 nodes coincide with the uniform pin cells. The node cross
   sections are then the exact pin materials, so these factors correct
   coarse-mesh error, not homogenization error. That is why "reference XS, no
   DF" at N = 17 is independent of s.
2. **Assembly-sized two-point nodes cannot host reference DFs.** At N = 1 and
   2, the homogeneous trace φ̄ − J·h/(2D) is negative for some thermal-group
   faces. With thermal D ≈ 0.2–0.3 cm and h/2 ≈ 5–11 cm, the drop exceeds the
   average flux. No positive factor exists, so the study reports these cases
   as inadmissible instead of solving them. This is a property of the two-point
   discretization, not of the factors. Nodal methods avoid it with
   higher-order intra-node shapes, and Bernal et al. use a polynomial
   reconstruction.
3. **At N = 1 and 2, coarse-mesh error swamps homogenization error.** The
   coarse solves sit 140–1000 pcm and 6–20% in power from the reference.
   Library ADFs move k by only 2–10 pcm, and power slightly in either
   direction. On this stencil, ADFs are no substitute for spatial refinement.
4. **At pin-pitch resolution (N = 17), library ADFs mostly improve power and
   interface currents, but not k.** (At s ≤ 4. With a converged s = 16
   reference the benefit largely disappears; see the
   [28 October update](#update-28-october-2026-converged-reference-cpu-s--8-and-16).) Pmax falls from 0.93 to 0.43% in
   `island3` and from 0.81 to 0.60% in `checker3`, and island3's current
   error from 0.024 to 0.005. In `checker2` at s = 4, Pmax rises slightly,
   0.34 → 0.39%. k moves 5–20 pcm *away* from the reference in every layout.
   Isolated-assembly factors are approximations to whole-core factors (see the
   plan's literature notes). That approximation shows up as a k bias of the
   same order as the improvement in power. Treat these as mixed results, not
   evidence of a general benefit.

## Update, 28 October 2026: converged reference (CPU, s = 8 and 16)

The s = 8 and 16 runs were made on the CPU while the GH200 nodes were
unavailable (Merlin7 `daily`, `cn018`, x86_64, NumPy 2.3.5, one thread; fine
solve 426 s for island3 at s = 16). JSON:
`benchmark-results/adf-gh200/colorset-cpu-s8-s16.json`. From s = 8 to 16 the
fine k_ref changes by 2–4 pcm, and every N = 17 error below by at most 0.5 pcm
and 0.01 percentage points, so s = 16 is converged for these purposes.
The N = 1 and 2 rows change by less than 1 pcm from s = 4, and those
reference DFs remain inadmissible.

N = 17, library cross sections, without → with isolated-assembly ADFs:

| layout | s | Δpcm | Pmax [%] | Prms [%] | Jmax |
|---|---:|---|---|---|---|
| checker2 | 2 | +63.3 → +72.5 | 0.42 → 0.28 | 0.37 → 0.24 | 0.020 → 0.009 |
| checker2 | 16 | +69.2 → +78.9 | 0.32 → 0.43 | 0.28 → 0.37 | 0.018 → 0.013 |
| checker3 | 2 | +75.2 → +95.1 | 0.93 → 0.43 | 0.53 → 0.40 | 0.023 → 0.015 |
| checker3 | 16 | +83.1 → +103.5 | 0.78 → 0.65 | 0.41 → 0.56 | 0.019 → 0.020 |
| island3 | 2 | +22.4 → +27.1 | 1.06 → 0.25 | 0.37 → 0.13 | 0.025 → 0.005 |
| island3 | 16 | +24.8 → +29.7 | 0.88 → 0.49 | 0.31 → 0.21 | 0.023 → 0.006 |

Reference XS + DF at N = 17 reproduces k_ref to < 0.001 pcm in every layout
at s = 8 and 16, which meets the `colorset` acceptance gate on the CPU.

As the reference converges, the no-ADF errors fall slightly and the ADF-on
errors grow. Most of the apparent library-ADF benefit at s = 2 was therefore
reference discretization error. At s = 16:

- `island3` still improves clearly: Pmax 0.88 → 0.49%, Prms 0.31 → 0.21%,
  Jmax 0.023 → 0.006.
- `checker3` is mixed: Pmax improves (0.78 → 0.65%) but Prms worsens
  (0.41 → 0.56%), and Jmax is unchanged.
- `checker2` is worse in power (Pmax 0.32 → 0.43%, Prms 0.28 → 0.37%) but
  better in interface current (0.018 → 0.013).
- k moves away from the reference in every layout, by +9.7, +20.4 and
  +4.9 pcm respectively.

Finding 4 should therefore be read as: at pin-pitch nodes, isolated-assembly
ADFs improve RMS assembly power in one of three layouts, improve interface
currents in two, and bias k by 5–20 pcm. This is no evidence of a general
benefit for these layouts. The comparison remains between diffusion
solutions with a pin-cell-homogenized reference (see below).

## Limits and next steps

- Fine meshes of s = 8 and 16 confirm a converged reference on the CPU (see
  the update above). The GPU `colorset` stage now checks only device
  agreement and timing.
- The fine reference is pin-cell homogenized. A pin-resolved fine mesh
  (`build_c5g7_2d(pin_resolved=True)` style, with blended cells) would make
  N = 17 a genuine homogenization test. This is implemented as
  `--pin-resolved` and smoke-tested on the CPU; the GPU runs and gates are in
  {download}`c5g7_pin_resolved_plan.md <c5g7_pin_resolved_plan.md>`.
- A water reflector needs reflector factors from a fuel/reflector colorset.
  It is not included.
- Transport (S_N) references and SPH combinations are out of scope.
