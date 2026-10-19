"""Versioned raw regional kinetics exchange contract.

This is a supplemental-data importer, not an inference of missing regional
delayed yields from global Serpent effective parameters. All arrays retain
explicit region, family and incident/emitted-energy identities.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
import numpy as np


def _array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf':
        raise ValueError(f'{name} must be a real numeric array')
    a = np.array(raw, dtype=float, copy=True)
    if not np.all(np.isfinite(a)):
        raise ValueError(f'{name} must be finite')
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class RegionalKineticsData:
    """Validated raw production data, using cm, seconds and descending MeV."""
    region_ids: tuple
    family_ids: tuple
    energy_bounds: np.ndarray
    inverse_velocities: np.ndarray
    decay: np.ndarray
    total_production: np.ndarray
    delayed_production: np.ndarray
    provenance: object
    approximations: tuple
    source: Path
    source_sha256: str

    def for_nodes(self, region_ids, energy_bounds, total_production):
        """Expand an explicit physical-node mapping, verifying frozen total XS.

        Repeated region IDs permit declared symmetry expansion. No interpolation
        or positional matching is performed. Returned arrays are read-only.
        """
        energy = _array(energy_bounds, 'energy_bounds')
        if not np.array_equal(energy, self.energy_bounds):
            raise ValueError('energy boundaries differ from regional kinetics data')
        lookup = {name: i for i, name in enumerate(self.region_ids)}
        try:
            indices = [lookup[name] for name in region_ids]
        except (KeyError, TypeError) as exc:
            raise ValueError('unknown physical-node region ID') from exc
        if not indices:
            raise ValueError('physical-node mapping must be nonempty')
        total = _array(total_production, 'total_production')
        expected = self.total_production[indices]
        if total.shape != expected.shape or not np.allclose(total, expected, rtol=1e-12, atol=0):
            raise ValueError('total production differs from the frozen XS library')
        return {'inverse_velocities': _array(self.inverse_velocities[indices], 'inverse_velocities'),
                'delayed_production': _array(self.delayed_production[indices], 'delayed_production'),
                'decay': self.decay}


def read_regional_kinetics(path):
    """Read version-1 JSON containing qualified raw regional kinetics data.

    Schema requires explicit units, family IDs, region IDs, exact boundaries,
    total and delayed production matrices, provenance hashes and declared
    approximations. Global beta_eff is intentionally not an accepted input.
    Missing family data cannot be completed using a default spectrum/yield.
    """
    path = Path(path)
    raw = path.read_bytes()
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f'duplicate JSON field: {key}')
            out[key] = value
        return out
    data = json.loads(raw, object_pairs_hook=unique)
    if (not isinstance(data, dict) or type(data.get('schema_version')) is not int
            or data['schema_version'] != 1):
        raise ValueError('regional kinetics schema_version must be 1')
    if data.get('quantity_kind') != 'raw_delayed_production':
        raise ValueError('only raw_delayed_production is accepted; effective beta is not an input')
    if data.get('units') != {'energy': 'MeV', 'inverse_velocity': 's/cm',
                              'production': '1/cm', 'decay': '1/s'}:
        raise ValueError('explicit MeV, s/cm, 1/cm, 1/s units required')
    def ids(name):
        value = data.get(name)
        if (not isinstance(value, list) or not value or any(not isinstance(x, str) or not x for x in value)
                or len(set(value)) != len(value)):
            raise ValueError(f'{name} must contain unique nonempty string IDs')
        return tuple(value)
    regions, families = ids('region_ids'), ids('family_ids')
    energy = _array(data.get('energy_bounds'), 'energy_bounds')
    if energy.ndim != 1 or len(energy) < 2 or np.any(energy < 0) or np.any(np.diff(energy) >= 0):
        raise ValueError('energy_bounds must be strictly descending nonnegative MeV')
    N, I, G = len(regions), len(families), len(energy)-1
    inv = _array(data.get('inverse_velocities'), 'inverse_velocities')
    decay = _array(data.get('decay'), 'decay')
    total = _array(data.get('total_production'), 'total_production')
    delayed = _array(data.get('delayed_production'), 'delayed_production')
    if inv.shape != (N,G) or np.any(inv <= 0):
        raise ValueError('inverse_velocities must be positive (region,group)')
    if decay.shape != (I,) or np.any(decay <= 0):
        raise ValueError('decay must be positive (family,) with a common family system')
    if total.shape != (N,G,G) or np.any(total < 0):
        raise ValueError('total_production must be nonnegative (region,target,source)')
    if delayed.shape != (N,I,G,G) or np.any(delayed < 0):
        raise ValueError('delayed_production must be nonnegative (region,family,target,source)')
    if np.any(delayed.sum(axis=1)-total > 1e-13*max(float(total.max()), np.finfo(float).tiny)):
        raise ValueError('negative prompt production remainder')
    provenance = data.get('provenance')
    if (not isinstance(provenance, dict) or not provenance
            or any(not isinstance(k, str) or not k or not isinstance(v, str)
                   or re.fullmatch(r'[0-9a-f]{64}', v) is None for k,v in provenance.items())):
        raise ValueError('provenance requires named SHA256 source hashes')
    approx = data.get('approximations')
    if not isinstance(approx, list) or any(not isinstance(x, str) or not x for x in approx):
        raise ValueError('approximations must be an explicit list of descriptions')
    return RegionalKineticsData(regions, families, energy, inv, decay, total, delayed,
        MappingProxyType(dict(provenance)), tuple(approx), path, hashlib.sha256(raw).hexdigest())
