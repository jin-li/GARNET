"""Opt-in spatial pooling of native moments at a supplied reference flux.

The weights are volume INTEGRATED scalar flux, never flux density. Higher
moments use those same scalar weights; angular-moment rates are not preserved.
This module does not bind regions, infer volumes, or change solver defaults.
"""
import numpy as np

from .moment_materials import MomentMaterial


def _finite(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"Invalid finite {name}: expected shape {shape}")
    return array


def _validate(material, groups, grid, orders):
    if not isinstance(material, MomentMaterial):
        raise ValueError("Explicit MomentMaterial inputs required")
    p = material.provenance
    if p.get('orientation_contract') != 'from-major':
        raise ValueError("Explicit from-major orientation required")
    candidate_grid = _finite(p.get('energy_bounds_mev'), (groups+1,), 'energy grid')
    if not np.array_equal(candidate_grid, grid):
        raise ValueError("Matching energy grids required")
    if tuple(p.get('orders', ())) != orders or any(
            set(mapping) != set(orders) for mapping in
            (material.collision, material.production, material.loss)):
        raise ValueError("Matching declared collision/production/loss orders required")
    for name in ('total', 'chi', 'nu_sigma_f', 'kappa_fission'):
        a = _finite(getattr(material, name), (groups,), name)
        if np.any(a < 0) or (name == 'total' and np.any(a == 0)):
            raise ValueError(f"Unsupported negative or zero {name}")
    for field in ('INF_TOT', 'INF_NSF', 'INF_RABSXS', 'INF_FISS'):
        if field not in material.raw:
            raise ValueError(f"Missing native {field}")
        a = _finite(material.raw[field], (groups,), field)
        # Reduced absorption can be signed when scattering multiplies neutrons.
        if field != 'INF_RABSXS' and np.any(a < 0):
            raise ValueError(f"Unsupported negative {field}")
    for field, name in (('INF_TOT', 'total'), ('INF_NSF', 'nu_sigma_f'), ('INF_CHIT', 'chi')):
        if field in material.raw and not np.array_equal(
                _finite(material.raw[field], (groups,), field), getattr(material, name)):
            raise ValueError(f"Native {field} differs from material {name}")
    if 'INF_KAPPA' in material.raw:
        kappa = _finite(material.raw['INF_KAPPA'], (groups,), 'INF_KAPPA')
        if np.any(kappa < 0) or not np.allclose(
                material.raw['INF_FISS']*kappa, material.kappa_fission, rtol=1e-13, atol=0):
            raise ValueError("Native FISS*KAPPA differs from power coefficient")
    if np.any((material.raw['INF_FISS'] == 0) & (material.kappa_fission != 0)):
        raise ValueError("Nonzero power with zero fission unsupported")
    for order in orders:
        for kind, mapping in (('S', material.collision), ('SP', material.production)):
            a = _finite(mapping[order], (groups, groups), f'{kind}{order}')
            if order == 0 and np.any(a < 0):
                raise ValueError("Nonnegative zero-order collision/production required")
            field = f'INF_{kind}{order}'
            if field in material.raw and not np.array_equal(material.raw[field], a):
                raise ValueError(f"Native {field} differs from material matrix")
        loss = _finite(material.loss[order], (groups, groups), f'loss {order}')
        if not np.array_equal(loss, np.diag(material.total)-material.production[order].T):
            raise ValueError("Native production/loss orientation identity differs")


def homogenize_moment_materials(materials, volume_integrated_flux):
    """Pool a nonempty collection of fuel materials, retaining the energy grid.

    ``volume_integrated_flux[i,g]`` is the already integrated scalar flux of
    input i in incoming group g. It must be nonnegative, with positive summed
    support in every group. Chi uses each material's total fission-neutron
    source, sum_g flux[i,g]*nu_sigma_f[i,g], whose pooled sum must be positive.
    No coefficient or spectrum is clipped or renormalized. Singleton inputs
    return the original object after the same validation/support checks.

    Only from-major native matrices are accepted. All declared orders must
    match and include order zero. Nonfissile collections are unsupported.
    Derived statistical errors/covariance are unknown; donor errors are never
    assigned to a pooled estimate. Original provenance is retained in order.
    """
    materials = tuple(materials)
    if not materials or not isinstance(materials[0], MomentMaterial):
        raise ValueError("Nonempty MomentMaterial collection required")
    first = materials[0]
    groups = first.n_groups
    if groups < 1:
        raise ValueError("Nonempty energy grid required")
    grid = _finite(first.provenance.get('energy_bounds_mev'), (groups+1,), 'energy grid')
    if np.any(grid < 0) or not (np.all(np.diff(grid) > 0) or np.all(np.diff(grid) < 0)):
        raise ValueError("Strictly monotonic nonnegative energy grid required")
    orders = tuple(first.provenance.get('orders', ()))
    if not orders or 0 not in orders or len(set(orders)) != len(orders) or any(
            isinstance(o, (bool, np.bool_)) or not isinstance(o, (int, np.integer)) or o < 0
            for o in orders):
        raise ValueError("Explicit nonnegative native orders including zero required")
    for material in materials:
        _validate(material, groups, grid, orders)
    flux = _finite(volume_integrated_flux, (len(materials), groups), 'volume-integrated flux')
    if np.any(flux < 0):
        raise ValueError("Nonnegative volume-integrated flux required")
    with np.errstate(over='ignore', invalid='ignore'):
        weights = flux.sum(axis=0)
        source = np.sum(flux*np.stack([m.nu_sigma_f for m in materials]), axis=1)
        source_sum = source.sum()
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Positive finite flux support required in every group")
    if not np.all(np.isfinite(source)) or not np.isfinite(source_sum) or source_sum <= 0:
        raise ValueError("Positive finite pooled fission source required")
    if len(materials) == 1:
        return first

    def pool(values):
        a = np.stack(values)
        w = flux if a.ndim == 2 else flux[:, :, None]
        denominator = weights if a.ndim == 2 else weights[:, None]
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            result = np.sum(w*a, axis=0)/denominator
        if not np.all(np.isfinite(result)):
            raise ValueError("Nonfinite derived coefficient")
        return result

    total = pool([m.total for m in materials])
    nsf = pool([m.nu_sigma_f for m in materials])
    power = pool([m.kappa_fission for m in materials])
    with np.errstate(over='ignore', invalid='ignore'):
        chi = np.sum(source[:, None]*np.stack([m.chi for m in materials]), axis=0)/source_sum
    if not np.all(np.isfinite(chi)):
        raise ValueError("Nonfinite derived chi")
    collision = {o: pool([m.collision[o] for m in materials]) for o in orders}
    production = {o: pool([m.production[o] for m in materials]) for o in orders}
    loss = {o: np.diag(total)-production[o].T for o in orders}
    raw = {field: pool([m.raw[field] for m in materials])
           for field in ('INF_RABSXS', 'INF_FISS')}
    raw.update(INF_TOT=total, INF_NSF=nsf, INF_CHIT=chi)
    # A raw KAPPA is defined only where division by pooled FISS is supported.
    # Omit the entire raw vector if any group has undefined 0/0; power remains
    # available directly without an arbitrary effective energy per fission.
    kappa_defined = bool(np.all(raw['INF_FISS'] > 0))
    if kappa_defined:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            raw['INF_KAPPA'] = power/raw['INF_FISS']
        if not np.all(np.isfinite(raw['INF_KAPPA'])):
            raise ValueError("Nonfinite derived INF_KAPPA")
    for order in orders:
        raw[f'INF_S{order}'] = collision[order]
        raw[f'INF_SP{order}'] = production[order]
    residual = total-raw['INF_RABSXS']-production[0].sum(axis=1)
    if any(not np.all(np.isfinite(a)) for a in (*loss.values(), residual)):
        raise ValueError("Nonfinite derived loss or balance residual")
    provenance = dict(
        method='spatial_scalar_flux_homogenization', orientation_contract='from-major',
        energy_bounds_mev=grid, orders=orders,
        source_provenance=tuple(dict(m.provenance) for m in materials),
        volume_integrated_scalar_flux=flux, group_flux_support=weights,
        material_fission_source=source, pooled_fission_source=float(source_sum),
        weights_are_volume_integrated=True, volume_applied_again=False,
        chi_renormalized=False, signed_higher_moments_retained=True,
        raw_kappa_defined=kappa_defined, power_pooled_directly=True,
        derived_coefficient_means=True, raw_fields_are_native_estimates=False,
        aggregate_uncertainty='unknown', covariance_inferred=False,
        donor_errors_copied=False, uncertainty_qualification=False,
        physical_admission=False, production_admission=False,
        higher_estimator_transport_consistency_validated=False,
        angular_moment_rate_preservation=False, source_qualified=False)
    return MomentMaterial(total, chi, nsf, power, collision, production, loss,
                          raw, {}, provenance,
                          {'total_minus_reduced_absorption_production_out': residual})
