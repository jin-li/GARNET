# Dall'Osso 2014 two-group slab: preliminary comparison

Dall'Osso, [Spatial rehomogenization of cross sections and discontinuity
factors for nodal calculations](https://www.researchgate.net/publication/264475099_Spatial_rehomogenization_of_cross_sections_and_discontinuity_factors_for_nodal_calculations),
PHYSOR 2014, section 4.1, describes a two-group, one-dimensional
heterogeneous diffusion problem. The user supplied the full paper, allowing
Fig. 2 to be inspected directly. Its Table 1 gives materials and its Table 2
reports a heterogeneous fine-difference reference k of **1.04125**.

## Geometry interpretation and remaining uncertainty

The following interpretation of Fig. 2 is implemented in
[the example](../examples/dallosso_1d_reference.py). The material names and
cross sections are the paper's; lengths are in cm.

| Region left to right | Length | Material |
|---|---:|---|
| UO₂, including the 10.5 cm half assembly | 31.5 | U |
| outer part of MOX | 3.15 | M1 |
| inner part of MOX | 14.7 | M2 |
| outer part of MOX | 3.15 | M1 |
| gap | 0.525 | W |
| baffle | 3.15 | B |
| reflector | 17.325 | W |

The left boundary is reflective and the right boundary is zero-flux. The
code assumes all fission neutrons enter group 1, no upscatter, and zero
transverse leakage. The paper defines its scattering value as downscatter and
states that the transverse leakage is uniform in the reference and nodal
calculations; it does **not** specify a transverse leakage magnitude in this
case. We have not established whether these assumptions completely reproduce
the paper's calculation. The paper describes 20 mesh points per assembly and
20 in the baffle-reflector; the grids below instead align with every material
interface. The comparison thus includes both mesh and potentially input/model
differences. The paper defines pcm as `1e5*(k-k_ref)`, without dividing by k.

| dx (cm) | GARNET heterogeneous k | Difference from reported k (pcm) |
|---:|---:|---:|
| 0.525 | 1.03259425 | −865.6 |
| 0.2625 | 1.03262368 | −862.6 |
| 0.13125 | 1.03263123 | −861.9 |
| 0.065625 | 1.03263313 | −861.7 |

The discrepancy persists under refinement. It cannot be attributed to ADFs,
which are absent in these heterogeneous calculations, and the cause has not
been identified. An independent piecewise-exact two-group multi-shooting
calculation with the same assumptions gives k = 1.03263376024, consistent
with the FV refinement. In particular, the unspecified transverse leakage is a
possible input difference, not an established explanation. The paper's power
uses fission cross sections, but only νΣf is tabulated; the example reports
normalized fission-neutron production and does not claim its distribution is
identical to the paper's power.

## Reference-factor exercise

The example can use its own *same-grid heterogeneous solution* as a reference.
It flux-weights the reaction constants in each of the five 10.5 cm fuel nodes,
chooses flux-weighted D, and imposes the reference net currents in independent
local two-group homogeneous FV solves. Their reconstructed traces provide
mesh-specific reference factors. The reflector remains heterogeneous on the
same grid, so its physical boundary treatment is unchanged. This setup does
not reduce the number of mesh unknowns; it tests spatial homogenization and
the ADF interface formulation. Both factors and cross sections use the
reference answer, so it is an equivalence test, not evidence for transferability
of assembly-library ADFs.

| dx (cm) | heterogeneous k | homogenized k, no factors | homogenized k, reference factors | max group/node flux error, no factors | max group/node flux error, reference factors |
|---:|---:|---:|---:|---:|---:|
| 0.525 | 1.03259425 | 1.03158093 | 1.03259425 | 3.51% | <1e-8 |
| 0.2625 | 1.03262368 | 1.03160429 | 1.03262368 | 3.51% | <1e-8 |

The corrected interface currents agree within 4e-10 in the solver's chosen
flux normalization. To reproduce the table:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python3 examples/dallosso_1d_reference.py \
  --refinements 1 2 --reference-factors
```

The automatic checks are in
[test_dallosso_adf_reference.py](../tests/verification/test_dallosso_adf_reference.py).
Before using this as a comparison against the paper's numbers, resolve the
missing modeling details, including the fission spectrum and transverse
leakage, and independently verify the geometric interpretation.
