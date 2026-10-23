# Opt-in regional isotropic cross sections

`ndgpu.regional_xs` is an experimental, explicit, standalone adapter. Its curve
evidence is supplied by the caller; this is not a verified correction table.
It does not change the default Serpent reader or qualify a transport correction for physical use.
Region identities are supplied as `RegionMapping(region_id, material, pin,
ring, axial, component="fuel")`; pin and component are strings, and ring/axial
are nonnegative integer indices. Fuel rings, gap, clad and coolant may share a
pin with distinct components. Use distinct component labels such as
`inner-coolant` and `outer-coolant` for separate coolant regions in the same pin.
Universe names are recorded for audit and never parsed into geometry.

```python
from ndgpu.regional_xs import (
    CurveProvenance, RegionMapping, read_regional_dataset, RegionDataset,
)

evidence = CurveProvenance(
    source="my-correction-table.json",
    source_sha256=correction_table_sha256,
    temperature_k=900.,
    temperature_evidence="Source section 2 identifies this curve at 900 K.",
)
dataset = read_regional_dataset(
    "regional_res.m",
    [(RegionMapping("fuel-r0", "fuel", "pin-0", 0, 0), 0)],
    mode="transport-corrected", curve_provenance=evidence,
    balance_sigma=1., balance_atol=1.e-12,
)
dataset.write_manifest("regional-means.json")
restored = RegionDataset.read_manifest("regional-means.json")
```

The selection integer is the zero-based record index in source-file order.
Duplicate region IDs, pin/component/ring/axial identities, or source digest/record pairs
are ambiguous and rejected. All regions must have exactly the same descending
energy boundaries. Different curve evidence per region can be supplied using
`regional_material_from_record` individually and then `RegionDataset`.

For transport-corrected mode, with physical total `T`, native from-group-major
production `P = INF_SP0`, and `D = TRC_DIFFCOEF`, the adapter computes

```text
T_eff = 1 / (3 D)
P_eff = P - diag(T - T_eff)
```

Off-diagonal production, reduced absorption `INF_RABSXS`, `INF_NSF`, and native
chi remain unchanged; kappa-fission is `INF_FISS * INF_KAPPA` when both exist.
There is no chi renormalization or substitute spectrum. The effective loss
matrix equals the physical-total/native-production loss matrix. Reduced
absorption is not a physical absorption tally. Missing TRC, invalid dimensions,
nonpositive diffusion, negative effective production (including the diagonal),
and negative reduced absorption are errors. No clipping, fitting, coefficient
averaging, or fallback is performed.

The independent diagnostic mode `physical-isotropic` retains physical total
and native production and uses `D = 1 / (3 T)`. It needs no curve evidence.
Neither mode establishes physical applicability; supplied correction provenance
is retained but not automatically verified or qualified.

Every construction checks the per-group scalar residual
`r = T - INF_RABSXS - sum_out(P)` against the declared tolerance

```text
abs(r) <= balance_atol + balance_sigma * (
    abs(T) * err_T + abs(INF_RABSXS) * err_RABSXS
    + sum_out(abs(P) * err_P)
)
```

`balance_sigma` is required explicitly. This absolute-error envelope uses
printed relative errors; it is not an inferred covariance model, confidence
interval, or independence assumption. Printed zero errors remain zero. The
adapter records the tolerance, residual, native means and errors, energy
boundaries, source SHA-256, record index, version and curve evidence. Values and
nested provenance are immutable snapshots; the input record is not modified.
Physical collision scattering and absorption remain available in `raw_means`,
separate from effective production and reduced absorption.

A balance check is a consistency screen, not a precision qualification. Large
native errors can make its tolerance wide enough to accept a large residual.
Inspect required scattering rows, energy-group flux support, individual transfer
errors and source convergence separately. Do not infer exactness from a printed
zero error, or a particle-event count from a printed 100% relative error. In the
Serpent 2.2.2 group-constant processing path, a batch with absent group flux does
not contribute that group's XS ratio; fewer than two contributing statistical
samples can produce a zero printed XS error. Increasing particles per processing
batch and increasing the number of batches address different sampling concerns.
Their adequacy requires evidence from the calculation, beyond adapter admission.

The transformed diagonal must also satisfy
`1/(3 D[g]) >= T[g] - P[g,g]`. Raw balance acceptance does not imply this
condition. If it fails, the adapter refuses the constants rather than adjusting
the row. The caller remains responsible for qualifying the native estimates,
correction curve and physical closure for the intended calculation.

`region.as_material()` copies effective arrays into the solver's `Material`
interface. As in the original scalar adapter, this conversion normalizes a
nonzero chi and gives an all-zero nonfissile spectrum a benign first-group
spectrum. Fissile all-zero chi is rejected. Native chi remains unchanged in the
regional snapshot.

The versioned JSON manifest contains means, native printed relative errors and
provenance. Loading reconstructs adapter admission from those native means and
errors, rechecks curve evidence and scalar balance, and requires exact agreement
of derived arrays and recorded policy. A manifest cannot assert verified physical
applicability or covariance. It does not copy result files, curves or binary
data, or verify referenced source files. Consistently rewriting both native data
and provenance cannot be detected without the original source artifacts;
preserve them to reproduce and independently audit a dataset.

`aggregate_power_bins(flux, kappa_fission, volumes, bins)` computes a bin's
volume-average power density, `sum_regions(V * sum_groups(phi*kappaSf))/sum(V)`.
Flux and kappa-fission have shape `(regions, groups)`; volume and bin labels have
one entry per region. Flux normalization and power units are the caller's
contract. This is a tally aggregation only; it supplies no coarse diffusion or
cross-section homogenization rule.
