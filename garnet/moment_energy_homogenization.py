"""Opt-in energy aggregation of native coefficient means at a scalar flux.

Arrays are explicitly from-major. This scalar-flux algebra for Serpent's
reported moment estimators does not qualify angular-moment rates or their
consistency with transport theory. No spatial regions are combined here.
"""
from collections.abc import Mapping
import re

import numpy as np

from .moment_materials import MomentMaterial


def _finite(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"Invalid finite {name}: expected shape {shape}")
    return array


def collapse_native_energy_coefficients(fields, volume_integrated_flux,
                                        energy_bounds_mev, partition, *,
                                        source_provenance=None):
    """Aggregate mean-only native fields into contiguous energy groups.

    ``partition`` lists nonempty sequences of original zero-based group indices,
    in original order, covering every group exactly once. Flux is already volume
    integrated; each aggregate must have positive finite scalar-flux support.
    S/SP matrices have incoming groups on rows, outgoing groups on columns.
    Chi is summed over outgoing groups without renormalization; power pools
    FISS*KAPPA directly. A zero-flux/zero-total input group is allowed only when
    its aggregate has support. There is no donor filling or regularization.

    Derived means have unknown errors and covariance. Scalar-flux weighted
    higher moments do not establish angular-moment-rate preservation or
    physical/estimator qualification.
    """
    flux = np.asarray(volume_integrated_flux, dtype=float)
    if flux.ndim != 1 or flux.size < 1:
        raise ValueError("Nonempty one-dimensional volume-integrated flux required")
    groups = flux.size
    flux = _finite(flux, (groups,), 'volume-integrated flux')
    if np.any(flux < 0):
        raise ValueError("Nonnegative volume-integrated flux required")
    grid = _finite(energy_bounds_mev, (groups+1,), 'energy grid')
    if np.any(grid < 0) or not (np.all(np.diff(grid) > 0) or np.all(np.diff(grid) < 0)):
        raise ValueError("Strictly monotonic nonnegative energy grid required")
    try:
        blocks = tuple(tuple(block) for block in partition)
    except TypeError as exc:
        raise ValueError("Contiguous full-cover partition required") from exc
    indices = tuple(index for block in blocks for index in block)
    if (not blocks or any(not block for block in blocks)
            or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
                   for i in indices) or indices != tuple(range(groups))):
        raise ValueError("Contiguous full-cover partition in original order required")
    if not isinstance(fields, Mapping):
        raise ValueError("Mapping of native coefficient means required")
    vectors = {'INF_TOT', 'INF_NSF', 'INF_FISS', 'INF_KAPPA', 'INF_RABSXS', 'INF_CHIT'}
    orders = set()
    for key in fields:
        if key in vectors or key == 'INF_ABS':
            continue
        match = re.fullmatch(r'INF_(S|SP)([0-7])', key) if isinstance(key, str) else None
        if match is None:
            raise ValueError(f"Unsupported native mean field {key}")
        orders.add(int(match[2]))
    if not orders or orders != set(range(max(orders)+1)):
        raise ValueError("Consecutive native orders 0..L, L <= 7 required")
    orders = tuple(sorted(orders))
    required = vectors | {f'INF_{kind}{order}' for kind in ('S', 'SP') for order in orders}
    if not required.issubset(fields):
        raise ValueError(f"Missing native means: {sorted(required-set(fields))}")
    native = {}
    for key, value in fields.items():
        matrix = key not in vectors and key != 'INF_ABS'
        a = _finite(value, (groups, groups) if matrix else (groups,), key)
        signed = key == 'INF_RABSXS' or (matrix and int(key[-1]) > 0)
        if not signed and np.any(a < 0):
            raise ValueError(f"Nonnegative native {key} required")
        native[key] = a
    if np.any((flux > 0) & (native['INF_TOT'] <= 0)):
        raise ValueError("Positive total required wherever input flux is positive")
    with np.errstate(over='ignore', invalid='ignore'):
        support = np.array([flux[list(block)].sum() for block in blocks])
    if not np.all(np.isfinite(support)) or np.any(support <= 0):
        raise ValueError("Positive finite aggregate flux support required")

    def pool_vector(a):
        return np.array([np.sum(flux[list(block)]*a[list(block)])/weight
                         for block, weight in zip(blocks, support)])

    def pool_matrix(a):
        return np.array([[np.sum(flux[list(incoming), None]*a[np.ix_(incoming, outgoing)])/weight
                          for outgoing in blocks]
                         for incoming, weight in zip(blocks, support)])

    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        raw = {key: pool_vector(a) for key, a in native.items()
               if key in vectors | {'INF_ABS'} and key not in ('INF_CHIT', 'INF_KAPPA')}
        raw['INF_CHIT'] = np.array([native['INF_CHIT'][list(block)].sum() for block in blocks])
        power = pool_vector(native['INF_FISS']*native['INF_KAPPA'])
        kappa_defined = bool(np.all(raw['INF_FISS'] > 0))
        if kappa_defined:
            raw['INF_KAPPA'] = power/raw['INF_FISS']
        collision = {o: pool_matrix(native[f'INF_S{o}']) for o in orders}
        production = {o: pool_matrix(native[f'INF_SP{o}']) for o in orders}
        total = raw['INF_TOT']
        loss = {o: np.diag(total)-production[o].T for o in orders}
        residual = total-raw['INF_RABSXS']-production[0].sum(axis=1)
    for order in orders:
        raw[f'INF_S{order}'] = collision[order]
        raw[f'INF_SP{order}'] = production[order]
    if (any(not np.all(np.isfinite(a)) for a in (*raw.values(), power, *loss.values(), residual))
            or np.any(total <= 0)):
        raise ValueError("Finite derived coefficients and positive aggregate total required")
    collapsed_grid = grid[[blocks[0][0], *(block[-1]+1 for block in blocks)]]
    provenance = dict(
        method='native_energy_scalar_flux_aggregation', orientation_contract='from-major',
        orders=orders, partition=blocks, energy_bounds_mev=collapsed_grid,
        original_energy_bounds_mev=grid, volume_integrated_scalar_flux=flux,
        group_flux_support=support, original_zero_support_indices=tuple(np.flatnonzero(flux == 0)),
        source_provenance={} if source_provenance is None else dict(source_provenance),
        weights_are_volume_integrated=True, volume_applied_again=False,
        chi_renormalized=False, signed_higher_moments_retained=True,
        raw_kappa_defined=kappa_defined, power_pooled_directly=True,
        derived_coefficient_means=True, raw_fields_are_native_estimates=False,
        aggregate_uncertainty='unknown', covariance_inferred=False,
        donor_errors_copied=False, donor_filling=False, uncertainty_qualification=False,
        physical_admission=False, production_admission=False,
        higher_estimator_transport_consistency_validated=False,
        angular_moment_rate_preservation=False, source_qualified=False)
    return MomentMaterial(total, raw['INF_CHIT'], raw['INF_NSF'], power,
                          collision, production, loss, raw, {}, provenance,
                          {'total_minus_reduced_absorption_production_out': residual})
