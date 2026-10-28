# Pin-resolved C5G7 colorset reference: plan

Status, 28 October 2026: **implemented and smoke-tested on the CPU; the
production runs wait for the GH200 nodes.** This is the second next step of the
[C5G7 colorset study](c5g7_colorset_adf.md) and part of stage 4 of the
[ADF plan](assembly_discontinuity_factors_plan.md).

## Why

The colorset reference so far is pin-cell homogenized. At N = 17 the coarse
nodes coincide with those uniform pin cells, so the node cross sections are
the exact pin materials and the study measures coarse-mesh error, not
homogenization error. With a converged s = 16 reference, library ADFs improved
RMS assembly power in only one of three layouts and moved k away from the
reference in all three (see the
[28 October update](c5g7_colorset_adf.md#update-28-october-2026-converged-reference-cpu-s--8-and-16)).
A fine reference that resolves the fuel cylinder makes assembly and pin-cell
homogenization a real approximation. Only such a test can show whether
isolated-assembly ADFs help where homogenization error dominates.

This remains diffusion against diffusion. Diffusion is a poor model inside a
0.54 cm fuel pin, so a positive result here is evidence about the equivalence
workflow, not about C5G7 physics or transport accuracy.

## What is implemented

`examples/c5g7_colorset_adf.py --pin-resolved` rasterizes every lattice
position (fuel, guide tubes and fission chamber) as an r = 0.54 cm cylinder of
the pin's own material in water. It uses the same rule as
`build_c5g7_2d(pin_resolved=True)`: cells inside the cylinder are pure pin,
cells outside are water, and boundary cells blend pin into water by covered
area. The solver blends reaction cross sections linearly and D harmonically.
The colorset reference, the isolated-assembly library and the reference node
cross sections all use this geometry.

Details that matter for correctness:

- **Homogenization of blended cells.** `flux_weighted_homogenize` splits each
  blended cell into two volume fractions. The blended fraction must go to the
  cell's own region (`mix_region_map=regions`), not to the helper's default,
  which numbers regions after `mix_material`. A test covers this through the
  reference-DF exactness check.
- **Fission rates** for assembly powers use the blended per-cell Σ_f
  (`cell_sigma_f`), for both the reference and the homogenized rates.
- **Fuel area.** `pin_coverage(s, sub)` samples each cell on a sub × sub
  grid. The builder's default sub = 16 misstates the fuel area by −0.66% at
  s = 4, +0.06% at s = 8 and −0.02% at s = 16. At about 25 (UO2) to 45 (MOX)
  pcm per 0.1% of fuel area, that would add non-monotone noise of tens of pcm
  to k_ref. The example uses sub = 256 (`COVERAGE_SAMPLES`), which keeps the
  area error below 5e-5 at every s. `build_c5g7_2d` keeps sub = 16 so other
  published results do not move. `_pin_coverage` became the public
  `pin_coverage` so the example can import it.
- The default (pin-cell homogenized) path is unchanged: s = 2 reproduces the
  stored baseline to 1e-13.

## CPU smoke tests (28 October 2026)

Login node `cn000`, NumPy 2.3.5, one thread. Three new tests in
`tests/verification/test_colorset_adf.py`:

| check | result |
|---|---|
| s = 1 pin-resolved vs pin-cell homogenization at the same sampled fuel fraction (UO2, MOX) | k agrees to < 1e-9 pcm: a one-cell blend is pin-cell homogenization |
| isolated-assembly homogenization preserves k∞ with blends (s = 4) | ≤ 5e-11 |
| reference XS + DF at N = 17 reproduces the pin-resolved reference (checker2, s = 4) | Δk < 1e-3 pcm, power < 1e-4%, current < 1e-6 |

The full study at s = 2 and 4 (33 s) gave the following:

- **Reference XS + DF at N = 17** reproduces k_ref to < 3e-5 pcm in every
  layout. The smallest homogeneous-trace ratio is 0.62, well inside the
  admissible range.
- **Reference XS without DF** is now off by 117–214 pcm at N = 17.
- **N = 1 and 2** remain inadmissible.
- **At s = 2** every cell is a blend, so the results are close to
  pin-cell homogenization.
- **At s = 4**, library ADFs worsen N = 17 power in all three layouts: Prms
  0.01 → 0.69% (checker2), 0.19 → 0.88% (checker3) and 0.14 → 0.39%
  (island3). This is a 4 × 4 raster of a 0.54 cm cylinder, so it is not yet a
  result. It does show that the answer may differ from the pin-homogenized
  study, and that convergence in s has to be shown before drawing any
  conclusion.

All colorset, mixing, fused-kernel, example-API and Cartesian ADF tests pass
on the CPU (79 passed; the 21 skips are GPU tests or a missing FEMFFUSION
checkout).

## GPU production runs

Fine meshes of s = 8, 16 and 32 per pin side, N = 1, 2 and 17, all three
layouts. s = 32 is 1632² cells (2.7 M) for the 3 × 3 layouts. Fine-solve inner
iterations roughly double per mesh doubling, as in the pin-homogenized runs,
so s = 32 costs about 8× s = 16. On one CPU thread, s = 16 took 7 min per
3 × 3 layout, which puts s = 32 near an hour per layout on the CPU. The GPU
time is unmeasured. From a login node, with the code in `~/GARNET`:

```bash
GARNET_ADF_STAGES="verify colorset" GARNET_ADF_TIME_COLORSET=02:00:00 \
GARNET_ADF_ARGS="--pin-resolved --cells-per-pin 8 16 32 --nodes 1 2 17 --device gpu" \
    bash ~/GARNET/slurm/stage_and_submit_adf_gh.sh
```

`GARNET_ADF_ARGS` replaces the colorset stage's defaults, and the stage runs
only if `verify` passes. If s = 32 does not fit the limit, rerun it alone with
`--cells-per-pin 32`. Results go to
`/data/scratch/shared/poncet_m/garnet-adf/results/<job>-colorset/colorset.json`.
Copy them into `benchmark-results/adf-gh200/` as
`colorset-pin-resolved-gpu-s8-s32.json`. Also run s = 8 with `--device cpu`,
either as a CPU batch job or as a 2-D check, to compare devices on the
pin-resolved geometry.

## Acceptance gates

1. **Equivalence:** reference XS + DF at N = 17 reproduces k_ref to
   < 0.05 pcm, and power and interface current to round-off, in every layout
   at every s.
2. **Converged reference:** k_ref changes by ≤ 5 pcm from s = 16 to 32, and
   the N = 17 library rows by ≤ 0.5 pcm and ≤ 0.02 percentage points in
   power. Otherwise add s = 48 or use Richardson extrapolation before
   interpreting.
3. **Device agreement:** GPU and CPU k_ref agree to 2e-8 at s = 8.

## Analysis and decision

Report, per layout at the converged s and N = 17, without → with library
ADFs: Δpcm, Pmax, Prms and Jmax. Also report the "reference XS, no DF" row,
which now measures pure homogenization error and gives the scale ADFs have to
recover. Repeat the comparison at N = 1 and 2 against the pin-homogenized
numbers. Compare the isolated-assembly ADFs with the pin-homogenized ones; the
thermal factors moved from 0.877 (UO2) and 1.113 (MOX) at s = 2 to 0.910 and
1.159 at s = 4.

Proposed reading, to confirm before the runs: library ADFs show value on this
colorset if they reduce Prms and Jmax in at least two of three layouts without
a k bias larger than the no-ADF error. A mixed or negative result is still
publishable as such; the plan requires reporting regressions.

## Out of scope / follow-on

- **Exact circle-cell intersection areas** instead of sampling. They would
  remove the sub parameter. Use them in `build_c5g7_2d` only together with a
  documented change to published results.
- **Water reflector colorsets**, which need reflector factors.
- **Transport (S_N) references and SPH combinations.**
