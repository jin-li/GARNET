"""Exact identity of Serpent's optional printed micro-group structure."""
from collections.abc import Mapping
import numpy as np

MICROGROUP_IDENTITY_KEY = 'microgroup_identity'


def microgroup_bounds(fields):
    """Return immutable boundaries, or None when both optional fields are absent.

    Lists/tuples support saved JSON and frozen native records as well as parser
    arrays. The micro-grid need not share the macro-grid's outer boundaries.
    """
    present = ('MICRO_NG' in fields, 'MICRO_E' in fields)
    if not any(present):
        return None
    if not all(present):
        raise ValueError('MICRO_NG/MICRO_E must both be present or both absent')
    ng = np.asarray(fields['MICRO_NG'])
    if (ng.dtype.kind not in 'iuf' or ng.shape != (1,)
            or not np.isfinite(ng[0]) or ng[0] < 1 or ng[0] != int(ng[0])):
        raise ValueError('MICRO_NG must contain one finite positive integer')
    bounds = np.asarray(fields['MICRO_E'])
    if (bounds.dtype.kind not in 'iuf' or bounds.shape != (int(ng[0]) + 1,)
            or not np.all(np.isfinite(bounds)) or np.any(bounds < 0)
            or not np.all(np.diff(bounds.astype(float)) < 0)):
        raise ValueError('MICRO_E must contain G+1 finite nonnegative strictly descending boundaries in MeV')
    return tuple(float(value) for value in bounds)


def identity_from_bounds(bounds):
    return ({'status': 'unknown', 'diffusion': 'inf'} if bounds is None else
            {'status': 'known', 'diffusion': 'inf', 'n_groups': len(bounds) - 1,
             'energy_bounds_mev': bounds})


def validate_identity(identity):
    """Canonicalize the reserved serialized identity, rejecting hidden data."""
    if not isinstance(identity, Mapping):
        raise ValueError('microgroup_identity must be a mapping')
    if dict(identity) == {'status': 'unknown', 'diffusion': 'inf'}:
        return identity_from_bounds(None)
    if (identity.get('status') != 'known'
            or identity.get('diffusion') != 'inf'
            or set(identity) != {'status', 'diffusion', 'n_groups', 'energy_bounds_mev'}):
        raise ValueError('invalid microgroup_identity declaration')
    return identity_from_bounds(microgroup_bounds(
        {'MICRO_NG': [identity['n_groups']], 'MICRO_E': identity['energy_bounds_mev']}))


def native_identity(records):
    grids = []
    for record in records.values():
        if not isinstance(record, Mapping):
            raise ValueError('native microgroup identity requires mapping records')
        fields = record.get('fields', record)
        if not isinstance(fields, Mapping):
            raise ValueError('native microgroup fields must be a mapping')
        grids.append(microgroup_bounds(fields))
    if grids and any(grid != grids[0] for grid in grids[1:]):
        raise ValueError('inconsistent native microgroup structures (including known/unknown records)')
    return identity_from_bounds(grids[0] if grids else None)
