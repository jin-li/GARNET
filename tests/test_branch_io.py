"""Synthetic returned-result snapshots; no external reactor data or execution."""
import hashlib
import json

import numpy as np
import pytest

from garnet import branch_io


def write_result(path, universe='u1', overrides=None):
    fields = {'VERSION': "'Serpent 2.2.2'", 'GC_UNIVERSE_NAME': repr(universe),
              'MACRO_NG': '2', 'MACRO_E': '[20 1 0]',
              'SIMULATION_COMPLETED': '1', 'LOST_PARTICLES': '0'}
    for name, values in {'TOT': [1, 2], 'ABS': [.4, .8], 'RABSXS': [.4, .6],
                         'NSF': [.2, .1], 'CHIT': [1, 0], 'FLX': [2, 1],
                         'DIFFCOEF': [1, .5], 'S0': [.2, .3, .1, 1.1],
                         'SP0': [.2, .4, .1, 1.3]}.items():
        fields['INF_' + name] = '[' + ' '.join(f'{v} 0' for v in values) + ']'
    fields.update(overrides or {})
    lines = []
    for key, value in fields.items():
        if value is None:
            continue
        count = len(value[1:-1]) if value.startswith("'") else len(value.strip('[]').split())
        lines.append(f'{key} (idx,[1:{count}]) = {value};')
    path.write_text('\n'.join(lines))
    return path


def load(paths, **kwargs):
    options = dict(state_id='cold', conditions={'fuel_temperature': 300., 'family': 'synthetic'},
                   region_universes=[('fuel', 'u1')])
    options.update(kwargs)
    return branch_io.import_returned_serpent_state(paths, **options)


def test_snapshot_is_honest_json_safe_and_ordered(tmp_path):
    a = write_result(tmp_path / 'a_res.m')
    b = write_result(tmp_path / 'b_res.m', universe='u2')
    source = tmp_path / 'input'; source.write_text('synthetic declared input')
    factors = np.full((2, 6, 2), 1.1)
    state, evidence = load([a, b], source_files=[source],
                           region_universes=[('reflector', 'u2'), ('fuel', 'u1')],
                           metadata={'geometry_family': 'synthetic', 'nuclear_data_family': 'caller-label'},
                           factors=factors, qualification='synthetic-only')
    assert state.region_ids == ('reflector', 'fuel')
    assert state.qualification == 'synthetic-only'
    assert state.metadata['adf_origin'] == 'caller-supplied'
    np.testing.assert_array_equal(state.factors, factors)
    assert evidence['state_id'] == 'cold'
    assert evidence['conditions'] == dict(state.conditions)
    assert evidence['kind'] == 'retrospective-import'
    assert evidence['original_runner_receipt'] is False
    assert evidence['executable_identity'] is None
    assert evidence['ace_content_identity'] is None
    assert evidence['nuclear_data_equivalence'] == 'not-established'
    assert json.loads(json.dumps(evidence)) == evidence
    assert evidence['source_files'][2]['role'] == 'caller-declared-auxiliary'
    assert evidence['source_files'][0]['sha256'] == hashlib.sha256(a.read_bytes()).hexdigest()
    assert state.native.records['reflector']['source_sha256'] == hashlib.sha256(b.read_bytes()).hexdigest()
    evidence['source_files'][0]['sha256'] = 'changed'
    assert state.metadata['retrospective_import']['source_files'][0]['sha256'] != 'changed'


@pytest.mark.parametrize('overrides', [
    {'SIMULATION_COMPLETED': None}, {'SIMULATION_COMPLETED': '0'},
    {'SIMULATION_COMPLETED': '[1 1]'}, {'SIMULATION_COMPLETED': "'1'"},
    {'LOST_PARTICLES': None}, {'LOST_PARTICLES': '1'}, {'LOST_PARTICLES': 'nan'},
])
def test_rejects_incomplete_or_lost_native_output(tmp_path, overrides):
    with pytest.raises(ValueError, match='SIMULATION_COMPLETED|LOST_PARTICLES'):
        load(write_result(tmp_path / 'result.m', overrides=overrides))


@pytest.mark.parametrize('mapping', [[], [('fuel', 'absent')], [('fuel', 'u1'), ('fuel', 'u2')],
                                     [('fuel', 'u1'), ('other', 'u1')], [('fuel', 1)]])
def test_mapping_must_be_explicit_unique_and_complete(tmp_path, mapping):
    with pytest.raises(ValueError, match='mapping|pairs'):
        load(write_result(tmp_path / 'result.m'), region_universes=mapping)


def test_all_native_universes_require_mapping_and_unique_records(tmp_path):
    a = write_result(tmp_path / 'a.m')
    b = write_result(tmp_path / 'b.m', universe='u2')
    with pytest.raises(ValueError, match='cover exactly'):
        load([a, b])
    write_result(b)
    with pytest.raises(ValueError, match='duplicate native universe'):
        load([a, b])
    with pytest.raises(ValueError, match='duplicate declared'):
        load(a, source_files=[a])


@pytest.mark.parametrize('overrides', [{'VERSION': "'Serpent 2.2.1'"}, {'MACRO_E': '[21 1 0]'}])
def test_native_version_and_groups_must_match(tmp_path, overrides):
    a = write_result(tmp_path / 'a.m')
    b = write_result(tmp_path / 'b.m', universe='u2', overrides=overrides)
    with pytest.raises(ValueError, match='inconsistent native'):
        load([a, b], region_universes=[('fuel', 'u1'), ('other', 'u2')])


@pytest.mark.parametrize('key', ['provenance', 'retrospective_import', 'serpent_version',
                               'adf_origin', 'executable_sha256', 'ace_sha256', 'original_runner_receipt'])
def test_caller_cannot_overwrite_provenance(tmp_path, key):
    with pytest.raises(ValueError, match='reserved provenance'):
        load(write_result(tmp_path / 'result.m'), metadata={key: 'misleading'})


@pytest.mark.parametrize('which', ['results', 'auxiliary', 'parser'])
def test_changes_during_read_fail_closed(tmp_path, monkeypatch, which):
    path = write_result(tmp_path / 'result.m')
    auxiliary = tmp_path / 'input'; auxiliary.write_text('synthetic')
    original_reader = branch_io.read_serpent_results

    def changed_reader(p):
        if which == 'parser':
            # Reader digests must agree even if the original bytes reappear.
            initial = path.read_bytes()
            path.write_bytes(initial + b'\n% transient change')
            records = original_reader(p)
            path.write_bytes(initial)
        else:
            records = original_reader(p)
            target = path if which == 'results' else auxiliary
            target.write_bytes(target.read_bytes() + b'\n% changed')
        return records

    monkeypatch.setattr(branch_io, 'read_serpent_results', changed_reader)
    with pytest.raises(ValueError, match='changed'):
        load(path, source_files=[auxiliary])


def test_unity_defaults_do_not_qualify_adfs(tmp_path):
    state, evidence = load(write_result(tmp_path / 'result.m'))
    assert state.metadata['adf_origin'] == 'unity'
    assert state.qualification == 'unvalidated'
    assert 'adf_prescription' not in state.metadata
    assert evidence['serpent_version'] == state.metadata['serpent_version']
    np.testing.assert_array_equal(state.factors, 1.)
