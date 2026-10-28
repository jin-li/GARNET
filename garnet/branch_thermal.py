"""Explicit, bounded thermal stars; construction never validates predictions."""
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from .branch_xs import (BranchLibrary, BranchState, TabulatedModeratorPath,
                        THERMAL_STAR_REQUIRED_IDENTITIES, _plain)


@dataclass(frozen=True)
class ThermalStarEvidence:
    """Caller policy acceptance with traceable evidence, not a physics label.

    A qualification policy must inspect each original state and return this
    record. The policy owns the sufficiency of the cited evidence (including
    reconstruction and nuclear-data equivalence). Source qualification strings
    are never interpreted as acceptance by this module.
    """
    policy_id: str
    evidence_refs: tuple

    def __post_init__(self):
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ValueError('nonempty qualification policy_id required')
        refs = tuple(self.evidence_refs) if not isinstance(self.evidence_refs, str) else ()
        if not refs or any(not isinstance(r, str) or not r.strip() for r in refs):
            raise ValueError('nonempty qualification evidence_refs required')
        object.__setattr__(self, 'evidence_refs', refs)


def build_thermal_star(nominal, fuel_low, fuel_high, moderator_low, moderator_high,
                       *, path, roles=None, qualification_policy=None,
                       diagnostic=False, compatible_serpent_versions=()):
    """Build four triangles covering a five-point transformed diamond.

    ``low/high`` refer to actual fuel or moderator *temperature*. Moderator
    density follows the caller-defined monotone path.
    All inputs must be original native-backed BranchState objects, with actual
    ``moderator_temperature`` in conditions or metadata, and metadata
    ``moderator_path_id`` matching ``path``. The dependent moderator temperature
    is retained as provenance rather than a third interpolation coordinate.
    Each source must also declare nonempty geometry_family,
    nuclear_data_family, scoring_identity, adf_prescription and solver_family
    identity strings, including in diagnostic mode.

    Each state's ``role`` metadata must be ``nominal`` for the center or
    ``training`` for endpoints. An explicit state-ID-to-role mapping may supply
    missing roles, but cannot override conflicting metadata or admit withheld
    states. ``branch_role`` is also recognized for existing callers.

    By default a caller-supplied policy must return ThermalStarEvidence for
    every state. ``diagnostic=True`` explicitly permits unqualified construction
    and labels every table entry unvalidated. Neither mode waives compatibility
    checks or makes interpolation qualified. Absolute XS are never adjusted.
    ``compatible_serpent_versions`` forwards the explicit application policy to
    BranchLibrary; source/native versions and all other identity checks remain.
    """
    if not isinstance(path, TabulatedModeratorPath):
        raise TypeError('path must be TabulatedModeratorPath')
    if type(diagnostic) is not bool:
        raise TypeError('diagnostic must be boolean')
    if qualification_policy is not None and not callable(qualification_policy):
        raise TypeError('qualification_policy must be callable')
    if diagnostic and qualification_policy is not None:
        raise ValueError('choose diagnostic construction or a qualification policy')
    if not diagnostic and qualification_policy is None:
        raise ValueError('explicit qualification policy required; choose diagnostic=True for unvalidated construction')
    states = (nominal, fuel_low, fuel_high, moderator_low, moderator_high)
    if any(type(s) is not BranchState or s.native is None for s in states):
        raise ValueError('original native-backed BranchState inputs required; evaluated states are forbidden')
    ids = tuple(s.state_id for s in states)
    if len(set(ids)) != 5:
        raise ValueError('five distinct state IDs required')
    if roles is not None and (not isinstance(roles, Mapping) or set(roles) != set(ids)):
        raise ValueError('roles must map exactly the five state IDs')

    converted = []
    temperatures = []
    for index, s in enumerate(states):
        for key in THERMAL_STAR_REQUIRED_IDENTITIES:
            identity = s.metadata.get(key)
            if not isinstance(identity, str) or not identity.strip():
                raise ValueError(f'{s.state_id}: nonempty explicit {key} identity required')
        declared = [s.metadata[k] for k in ('role', 'branch_role') if k in s.metadata]
        if roles is not None:
            declared.append(roles[s.state_id])
        expected = 'nominal' if index == 0 else 'training'
        if not declared or any(r != expected for r in declared):
            raise ValueError(f'{s.state_id}: explicit {expected} role required; withheld/conflicting roles forbidden')
        if s.metadata.get('moderator_path_id') != path.path_id:
            raise ValueError(f'{s.state_id}: moderator path identity mismatch or missing')
        candidates = [container['moderator_temperature'] for container in
                      (s.conditions, s.metadata) if 'moderator_temperature' in container]
        if not candidates:
            raise ValueError(f'{s.state_id}: actual moderator_temperature required')
        temperature = candidates[0]
        density = path.evaluate(temperature)
        if any(not np.isclose(t, temperature, rtol=0, atol=1e-12) for t in candidates):
            raise ValueError(f'{s.state_id}: conflicting moderator_temperature provenance')
        actual_density = s.conditions.get('moderator_density')
        if isinstance(actual_density, (bool, np.bool_)) or not isinstance(actual_density, (int, float, np.number)) or not np.isclose(actual_density, density, rtol=0, atol=1e-12):
            raise ValueError(f'{s.state_id}: moderator density does not match temperature path')
        temperatures.append(float(temperature))
        conditions = dict(s.conditions)
        conditions.pop('moderator_temperature', None)
        metadata = {**_plain(s.metadata), 'role': expected,
                    'moderator_temperature': float(temperature),
                    'thermal_star': {'schema': 'garnet.thermal-star.v1',
                                     'construction': 'diagnostic-unvalidated' if diagnostic else 'evidence-policy',
                                     'nominal_state_id': nominal.state_id}}
        if diagnostic:
            metadata['thermal_star']['source_qualification'] = s.qualification
        else:
            evidence = qualification_policy(s)
            if not isinstance(evidence, ThermalStarEvidence):
                raise ValueError(f'{s.state_id}: qualification policy must return ThermalStarEvidence')
            metadata['thermal_star']['evidence'] = {
                'policy_id': evidence.policy_id, 'evidence_refs': evidence.evidence_refs}
        converted.append(replace(s, conditions=conditions, metadata=metadata,
                                 qualification='diagnostic-unvalidated' if diagnostic else s.qualification))

    # Adjacent endpoints: lower fuel, lower moderator T, higher fuel,
    # higher moderator T. Either monotone density direction gives a diamond;
    # the center is an explicit vertex in every cell.
    c, fl, fh, ml, mh = converted
    simplices = ((c.state_id, fl.state_id, ml.state_id),
                 (c.state_id, ml.state_id, fh.state_id),
                 (c.state_id, fh.state_id, mh.state_id),
                 (c.state_id, mh.state_id, fl.state_id))
    library = BranchLibrary(converted, axes=('fuel_temperature', 'moderator_density'),
                            transforms={'fuel_temperature': 'sqrt', 'moderator_density': 'identity'},
                            simplices=simplices,
                            compatible_serpent_versions=compatible_serpent_versions)
    tf = [s.conditions['fuel_temperature'] for s in converted]
    rho = [s.conditions['moderator_density'] for s in converted]
    if not tf[1] < tf[0] < tf[2] or not temperatures[3] < temperatures[0] < temperatures[4]:
        raise ValueError('low/high endpoints must strictly bracket nominal temperatures')
    if not (tf[3] == tf[0] == tf[4] and rho[1] == rho[0] == rho[2]
            and temperatures[1] == temperatures[0] == temperatures[2]):
        raise ValueError('axis endpoints must preserve nominal fixed thermal coordinate')
    return library
