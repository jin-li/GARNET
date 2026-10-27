"""Experimental geometry-aware interpolation on an explicit rod segment.

The occupancy provider is application code, not serialized library data. This
module changes interpolation weights only; it does not qualify rod worth or
conserve transport reaction rates.
"""
from collections.abc import Mapping
from dataclasses import replace
import numbers

import numpy as np

from .branch_xs import BranchLibrary, _plain


def _occupancies(provider, conditions, region_ids):
    # Providers receive immutable conditions and ordered immutable region IDs.
    values = provider(conditions, region_ids)
    if not isinstance(values, Mapping) or set(values) != set(region_ids):
        raise ValueError('occupancy must return exactly the library region IDs')
    result = []
    for region in region_ids:
        value = values[region]
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
                or not np.isfinite(value) or not 0 <= value <= 1):
            raise ValueError('occupancy fractions must be finite real scalars in [0,1]')
        result.append(float(value))
    return np.asarray(result)


def _check_channel(value, left, right, name):
    if (not np.all(np.isfinite(value)) or np.any(value < 0)
            or (name in ('D', 'factors') and np.any(value <= 0))):
        raise ValueError(f'{name}: invalid interpolated coefficients')
    tolerance = 5e-14 * np.maximum(1., np.maximum(np.abs(left), np.abs(right)))
    if (np.any(value < np.minimum(left, right) - tolerance)
            or np.any(value > np.maximum(left, right) + tolerance)):
        raise ValueError(f'{name}: interpolated coefficients exceed convex endpoint bounds')


def evaluate_rod_occupancy(library, conditions, *, occupancy, geometry_id,
                           axis='rod_insertion_cm'):
    """Evaluate an experimental rod interpolation with node-local weights.

    ``occupancy(conditions, region_ids)`` must return one scalar absorber
    fraction in [0,1] per region. Geometry, coordinate units and region mapping
    are the caller's responsibility. Exact states bypass the provider and
    retain native records and qualification. Interpolated results remain
    experimental and unvalidated, regardless of the source qualifications.
    """
    if not isinstance(library, BranchLibrary):
        raise TypeError('library must be BranchLibrary')
    if library.axes != (axis,) or library.transforms[axis] != 'identity':
        raise ValueError('rod occupancy requires one matching identity-transform axis')
    if not isinstance(geometry_id, str) or not geometry_id.strip():
        raise ValueError('geometry_id must be explicit nonblank text')
    if not callable(occupancy):
        raise TypeError('occupancy must be callable')

    # Public guards and exact selection precede every geometry evaluation.
    evaluated = library.evaluate(conditions)
    if not evaluated.interpolated:
        return evaluated
    left, right = (library.states[key] for key in evaluated.contributing_ids)
    regions = evaluated.region_ids
    ol = _occupancies(occupancy, left.conditions, regions)
    or_ = _occupancies(occupancy, right.conditions, regions)
    ot = _occupancies(occupancy, evaluated.conditions, regions)
    if np.any(ot < np.minimum(ol, or_)) or np.any(ot > np.maximum(ol, or_)):
        raise ValueError('target occupancy is outside endpoint occupancy bracket')
    changed = or_ != ol
    weights = np.full(len(regions), evaluated.weights[1])
    weights[changed] = (ot[changed] - ol[changed]) / (or_[changed] - ol[changed])
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or np.any(weights > 1):
        raise ValueError('local occupancy weights must be finite and convex')
    data = {}
    for key in ('D', 'sigma_a', 'sigma_s', 'fission_matrix', 'chi',
                'kappa_fission', 'total'):
        a, b = getattr(left, key), getattr(right, key)
        if a is not None:
            w = weights.reshape((-1,) + (1,) * (a.ndim - 1))
            data[key] = (1 - w) * a + w * b
            _check_channel(data[key], a, b, key)
    w = weights[:, None, None]
    data['factors'] = np.exp((1 - w) * np.log(left.factors) + w * np.log(right.factors))
    data['factors'][left.exterior_mask] = 1
    _check_channel(data['factors'], left.factors, right.factors, 'factors')
    chi = data['chi']
    norm = chi.sum(axis=1)
    chi[norm > 0] /= norm[norm > 0, None]
    for key, value in data.items():
        baseline = getattr(evaluated, key)
        tolerance = 5e-14 * np.maximum(1., np.abs(baseline[~changed]))
        if np.any(np.abs(value[~changed] - baseline[~changed]) > tolerance):
            raise ValueError(f'{key}: unchanged occupancy must preserve global interpolation')
    metadata = _plain(evaluated.metadata)
    metadata['role'] = 'prediction'
    metadata['rod_occupancy'] = {
        'method': 'experimental-node-local-occupancy-linear-xs-log-adf',
        'geometry_id': geometry_id, 'axis': axis,
        'region_ids': list(regions), 'local_right_weights': weights,
        'left_occupancy': ol, 'right_occupancy': or_, 'target_occupancy': ot,
        'source_ids': list(evaluated.contributing_ids),
        'source_roles': {s.state_id: _plain(s.metadata.get('role')) for s in (left, right)},
        'transport_reaction_rate_conservation': False,
    }
    return replace(evaluated, **data, native=None,
                   qualification='experimental-rod-occupancy-unvalidated', metadata=metadata)
