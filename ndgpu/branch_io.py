"""Retrospective snapshots of returned Serpent results for static CPU branches.

This records what is readable now, not what an original execution consumed.
Declared auxiliary files are hashed without interpreting or following includes.
"""
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .branch_xs import import_serpent_state
from .serpent_xs import read_serpent_results


def _paths(paths):
    if isinstance(paths, (str, Path)):
        paths = (paths,)
    return tuple(Path(p).absolute() for p in paths)


def _snapshot(path):
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f'declared source is not a regular file: {path}')
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
            size += len(block)
    return {'path': str(path), 'resolved_path': str(resolved),
            'sha256': digest.hexdigest(), 'size_bytes': size}


def import_returned_serpent_state(result_files, *, state_id, conditions,
                                  region_universes, source_files=(), metadata=None,
                                  factors=None, exterior_mask=None,
                                  qualification='unvalidated'):
    """Return ``(BranchState, JSON-safe evidence dict)`` from returned output.

    ``region_universes`` is an ordered mapping or sequence of (region, universe)
    pairs. It must cover every native universe exactly once across result files;
    duplicate records/universes are ambiguous and rejected. All records require
    native SIMULATION_COMPLETED=1 and LOST_PARTICLES=0. ``source_files`` are
    additional caller-declared files to snapshot, not verified execution inputs.

    Hashes are checked before and after parsing/conversion (no filesystem lock
    or original-run attestation is implied). Executable and ACE content identity
    remain unknown even when auxiliary files or caller metadata name libraries.
    Caller metadata, qualification, and factors retain their existing importer
    meaning; no ADF qualification or nuclear-data equivalence is inferred.
    """
    results, auxiliary = _paths(result_files), _paths(source_files)
    if not results:
        raise ValueError('at least one result file is required')
    paths = results + auxiliary
    before = [_snapshot(path) for path in paths]
    if len({entry['resolved_path'] for entry in before}) != len(paths):
        raise ValueError('duplicate declared source files')
    pairs = list(region_universes.items() if isinstance(region_universes, Mapping)
                 else region_universes)
    if not pairs or any(not isinstance(p, (tuple, list)) or len(p) != 2
                        or any(not isinstance(v, str) or not v for v in p)
                        for p in pairs):
        raise ValueError('ordered nonempty (region, universe) string pairs required')
    if len({p[0] for p in pairs}) != len(pairs) or len({p[1] for p in pairs}) != len(pairs):
        raise ValueError('duplicate region or universe mapping')
    metadata = dict(metadata or {})
    reserved = {'provenance', 'retrospective_import', 'serpent_version',
                'adf_origin', 'original_runner_receipt', 'execution_receipt',
                'executable_identity', 'executable_sha256', 'ace_identity', 'ace_sha256',
                'ace_content_identity', 'nuclear_data_equivalence'}
    if reserved.intersection(metadata):
        raise ValueError('metadata cannot overwrite reserved provenance fields')
    records = []
    for path, snapshot in zip(results, before):
        parsed = read_serpent_results(path)
        if any(r.source_sha256 != snapshot['sha256'] for r in parsed):
            raise ValueError(f'declared source changed during read: {path}')
        records.extend(parsed)
    first = records[0]
    by_universe = {}
    for record in records:
        for name, expected in (('SIMULATION_COMPLETED', 1), ('LOST_PARTICLES', 0)):
            value = record.fields.get(name)
            if (not isinstance(value, np.ndarray) or value.shape != (1,)
                    or value[0] != expected):
                raise ValueError(f'{record.source}: {name} must equal {expected}')
        if (record.version != first.version or record.n_groups != first.n_groups
                or not np.array_equal(record.energy_bounds_mev, first.energy_bounds_mev)):
            raise ValueError('inconsistent native version or energy groups')
        if record.universe in by_universe:
            raise ValueError(f'duplicate native universe: {record.universe}')
        by_universe[record.universe] = record
    if {p[1] for p in pairs} != set(by_universe):
        raise ValueError('region mapping must cover exactly the native universes')
    evidence = {
        'schema': 'ndgpu.retrospective-serpent-import.v1',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'kind': 'retrospective-import', 'original_runner_receipt': False,
        'executable_identity': None, 'ace_content_identity': None,
        'nuclear_data_equivalence': 'not-established',
        'source_files': [{**entry, 'role': 'returned-results' if i < len(results)
                          else 'caller-declared-auxiliary'} for i, entry in enumerate(before)],
        'serpent_version': first.version,
        'n_groups': first.n_groups, 'energy_bounds_mev': first.energy_bounds_mev.tolist(),
        'region_universes': [{'region_id': region, 'universe': universe,
                              'record_index': by_universe[universe].record_index,
                              'source': str(by_universe[universe].source)}
                             for region, universe in pairs],
        'simulation_completed': 1, 'lost_particles': 0,
    }
    state = import_serpent_state(
        state_id, conditions, {region: by_universe[universe] for region, universe in pairs},
        factors=factors, exterior_mask=exterior_mask, qualification=qualification,
        metadata={**metadata, 'serpent_version': first.version,
                  'retrospective_import': evidence})
    # Capture validated caller conditions without NumPy scalars or mutable aliases.
    evidence['state_id'] = state.state_id
    evidence['conditions'] = {k: v.item() if isinstance(v, np.generic) else v
                              for k, v in state.conditions.items()}
    # Freeze a separate evidence copy including the validated labels.
    state = replace(state, metadata={**dict(state.metadata), 'retrospective_import': evidence})
    after = [_snapshot(path) for path in paths]
    if before != after:
        raise ValueError('declared source changed during import')
    return state, json.loads(json.dumps(evidence, allow_nan=False))
