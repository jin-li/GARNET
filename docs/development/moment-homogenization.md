# Opt-in spatial moment homogenization

`garnet.moment_homogenization.homogenize_moment_materials(materials,
volume_integrated_flux)` pools an explicitly selected collection of native
`MomentMaterial` fuel inputs. It does not select rings, pool across pins, alter
bindings, or change solver defaults. Callers own membership and exact binding of
each flux row to its source record and energy grid. No region volume argument
is accepted: Serpent `INF_FLX` is already volume integrated.

For input material `i`, incoming group `g`, and supplied integrated scalar flux
`w[i,g]`, define `W[g] = sum_i w[i,g]`. The derived total, reduced absorption
(`INF_RABSXS`), fission (`INF_FISS`), neutron production (`INF_NSF`), and each
incoming row of every collision `S_l` and production `SP_l` matrix use
`Xbar[g,...] = sum_i w[i,g] X[i,g,...] / W[g]`. Outgoing groups are retained.
Native matrices must be from-major; derived losses are
`R_l = diag(totalbar) - SPbar_l.T`. Thus, up to floating point roundoff,
`Rbar_l @ W = sum_i R_i,l @ w_i` for every retained order, without requiring
commuting matrices. Signed higher moments remain signed.

Power uses `powerbar[g] = sum_i w[i,g] kappa_fission_i[g] / W[g]` directly.
It never multiplies separately averaged fission and kappa coefficients. A raw
derived `INF_KAPPA = powerbar / FISSbar` is included only if every pooled
fission group is positive; if any group has zero fission and zero power, the
entire raw kappa vector is omitted because its ratio is undefined. Nonzero
power with zero fission is refused. Consumers should use `kappa_fission` for
power, including when a subsequent finite-volume flux shape differs from the
reference weights; reusing original ring power coefficients changes the case.

For spectra, `Q_i = sum_g w[i,g] NSF_i[g]` and
`chibar[g'] = sum_i Q_i chi_i[g'] / sum_i Q_i`. This preserves the pooled
emitted-group fission source at the reference weights, without renormalizing
the reported spectra. Rates are preserved for the pooled collection, not
individually for its member rings after rebinding.

Inputs must have finite nonnegative weights, positive finite `W` in every
group, and a positive finite pooled fission source. Nonfissile collections are
unsupported. All materials must share the exact energy grid, group count,
declared native orders including zero, and from-major orientation. Matrix and
vector shapes, raw physical vectors, and production/loss consistency are
checked. Missing metadata or unsupported data cause `ValueError`. All inputs
are validated even for singleton collections, which return their original
object unchanged for exact identity controls.

Reduced absorption may be signed, for example with neutron-multiplying
scattering, and is pooled without clipping. Zero-order collision and production
entries must be nonnegative; signed higher orders remain supported. Negative
noisy zero-order estimates require separate source qualification before using
this helper and are refused rather than repaired.

The result is immutable, with derived raw coefficient means and no aggregate
reported errors. It retains the ordered source provenance, supplied weights,
support and source strengths. Aggregate uncertainty is explicitly unknown;
donor errors are not copied, covariance is not inferred, and physical/source
admission remains false. Singleton identity retains the original provenance
and original errors because no derived estimate is created.

The reduced-absorption balance residual is retained algebraically rather than
forced to zero. Higher moment spatial pooling uses scalar flux; it does not
preserve angular-moment reaction rates or qualify native estimator consistency
for deterministic transport. Pooling printed means is not equivalent to a new
Serpent pooled-GCU calculation and may retain batch-estimator bias. This helper
alone establishes no transport, spatial convergence, or uncertainty claim.

The method context is Serpent's [group constant generation equations
2.26–2.33](https://serpent.vtt.fi/docs/user_guide/gc_generation.html) and
[standard output](https://serpent.vtt.fi/docs/user_guide/standard_output.html).
The motivating dataset reports Serpent 2.2.2; exact local record/flux provenance
and declared field repairs remain the caller's responsibility.

Focused synthetic validation is in `tests/test_moment_homogenization.py`:
unequal volumes and flux weights, incoming-row and all-order loss algebra,
signed moments, emitted-group sources, direct power and changed FV shape,
identity, immutability, unknown uncertainty, and invalid/missing support.
