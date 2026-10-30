"""Native micro-grid compatibility survives legacy JSON and interpolation."""
from dataclasses import replace
import json
import numpy as np
import pytest

from garnet.branch_xs import BranchLibrary, NativeBranchState, import_serpent_state
from garnet.serpent_xs import read_serpent_xs
from test_branch_xs import state, line
from test_serpent_adf_reader import fields, write


def native(bounds, *, second=None, container=np.array):
    def record(grid):
        return {'fields': {} if grid is None else
                {'MICRO_NG': container([len(grid)-1]), 'MICRO_E': container(grid)}}
    records = {'u': record(bounds)}
    if second is not None:
        records['v'] = record(second)
    return NativeBranchState(records)


def known(name='a', bounds=(20., 1., 1e-11), **kwargs):
    return state(name, native=native(bounds), **kwargs)


@pytest.mark.parametrize('container', [np.array, list, tuple])
def test_native_identity_is_exact_and_immutable(container):
    a = state('a', native=native((20., 1., 1e-11), container=container))
    assert a.microgroup_identity == {'status': 'known', 'diffusion': 'inf', 'n_groups': 2,
                                     'energy_bounds_mev': (20., 1., 1e-11)}
    with pytest.raises(TypeError):
        a.microgroup_identity['status'] = 'unknown'
    # The micro-grid's endpoints intentionally differ from the macro-grid.
    assert tuple(a.energy_bounds_mev) == (10., .1, 0.)


@pytest.mark.parametrize('other', [(20., .5, 1e-11), None])
def test_library_rejects_same_count_different_edges_or_known_unknown(other):
    a = known()
    b = state('b', t=900., native=native(other))
    with pytest.raises(ValueError, match='incompatible microgroup'):
        line(a, b)


def test_versions_cannot_waive_70_versus_168_native_grid():
    a = known(bounds=tuple(np.geomspace(20., 1e-11, 71)), metadata={'serpent_version': '2.2.0'})
    b = known('b', tuple(np.geomspace(20., 1e-11, 169)), t=900., metadata={'serpent_version': '2.2.2'})
    with pytest.raises(ValueError, match='incompatible microgroup'):
        BranchLibrary([a, b], axes=('fuel_temperature',), simplices=[('a', 'b')],
                      compatible_serpent_versions=('2.2.0', '2.2.2'))


@pytest.mark.parametrize('bad', [
    {'MICRO_NG': [2]}, {'MICRO_E': [20., 1., 0.]},
    {'MICRO_NG': [2.5], 'MICRO_E': [20., 1., 0.]},
    {'MICRO_NG': [0], 'MICRO_E': [20.]},
    {'MICRO_NG': [True], 'MICRO_E': [20., 0.]},
    {'MICRO_NG': [float('inf')], 'MICRO_E': [20., 0.]},
    {'MICRO_NG': [2], 'MICRO_E': [20., 0.]},
    {'MICRO_NG': [2], 'MICRO_E': [20., 1., -1.]},
    {'MICRO_NG': [2], 'MICRO_E': [20., 1., 1.]},
    {'MICRO_NG': [2], 'MICRO_E': [0., 1., 20.]},
    {'MICRO_NG': [2], 'MICRO_E': [20., float('nan'), 0.]},
    {'MICRO_NG': [[2]], 'MICRO_E': [20., 1., 0.]},
])
def test_branch_creation_validates_preserved_fields(bad):
    with pytest.raises(ValueError, match='MICRO_|finite'):
        state('bad', native=NativeBranchState({'u': {'fields': bad}}))


def test_per_universe_consistency_and_missing_metadata():
    with pytest.raises(ValueError, match='inconsistent native microgroup'):
        state('bad', native=native((20., 1., 0.), second=(20., .5, 0.)))
    with pytest.raises(ValueError, match='known/unknown'):
        state('bad', native=NativeBranchState({'u': {'fields': {'MICRO_NG': [1], 'MICRO_E': [20., 0.]}}, 'v': {'fields': {}}}))


@pytest.mark.parametrize('declaration', [None, {}, {'status': 'unknown', 'diffusion': 'inf'},
    {'status': 'known', 'diffusion': 'inf', 'n_groups': 2, 'energy_bounds_mev': (20., .5, 1e-11)}])
def test_caller_cannot_hide_or_conflict_with_native(declaration):
    with pytest.raises(ValueError, match='microgroup_identity'):
        known(metadata={'microgroup_identity': declaration})
    # Replacing all metadata in expanded physical states still derives identity.
    assert replace(known(), metadata={'geometry_family': 'expanded'}).microgroup_identity == known().microgroup_identity


def test_legacy_reload_roundtrip_and_evaluation_identity(tmp_path):
    a, b = known(t=400.), known('b', t=900.)
    path = tmp_path / 'library.json'
    line(a, b).save(path)
    payload = json.loads(path.read_text())
    # Previous saves already retained native MICRO fields but no derived key.
    for item in payload['states']:
        del item['metadata']['microgroup_identity']
    path.write_text(json.dumps(payload))
    restored = BranchLibrary.load(path)
    assert restored.states['a'].microgroup_identity == a.microgroup_identity
    exact = restored.select('a')
    out = restored.evaluate({**dict(a.conditions), 'fuel_temperature': 625.})
    assert out.native is None
    assert out.microgroup_identity == exact.microgroup_identity == a.microgroup_identity
    restored.save(path)
    assert BranchLibrary.load(path).states['a'].microgroup_identity == a.microgroup_identity
    # Editing the saved identity cannot conceal the native grid.
    payload = json.loads(path.read_text())
    payload['states'][0]['metadata']['microgroup_identity'] = {'status': 'unknown', 'diffusion': 'inf'}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='conflicts with retained native'):
        BranchLibrary.load(path)


def test_unknown_synthetic_library_keeps_numerics(tmp_path):
    a, b = state('a', t=400.), state('b', t=900., strength=2.)
    lib = line(a, b)
    out = lib.evaluate({**dict(a.conditions), 'fuel_temperature': 625.})
    assert out.microgroup_identity == {'status': 'unknown', 'diffusion': 'inf'}
    np.testing.assert_allclose(out.sigma_a, (a.sigma_a+b.sigma_a)/2)
    path = tmp_path / 'unknown.json'
    lib.save(path)
    assert BranchLibrary.load(path).states['a'].microgroup_identity == {'status': 'unknown', 'diffusion': 'inf'}


def test_parser_optional_properties_import_and_unsupported_expression(tmp_path):
    raw = fields()
    raw['INF_DIFFCOEF'] = raw['CMM_DIFFCOEF'].copy()
    assert read_serpent_xs(write(tmp_path, raw)).n_micro_groups is None
    assert read_serpent_xs(write(tmp_path, raw)).micro_energy_bounds_mev is None
    raw.update(MICRO_NG=np.array([2]), MICRO_E=np.array([20., 1., 1e-11]))
    rec = read_serpent_xs(write(tmp_path, raw))
    assert rec.n_micro_groups == 2
    np.testing.assert_array_equal(rec.micro_energy_bounds_mev, raw['MICRO_E'])
    rec.micro_energy_bounds_mev[:] = 0  # Returned arrays do not mutate raw data.
    a = import_serpent_state('import', {'fuel_temperature': 600., 'moderator_density': .7}, {'u': rec})
    assert a.microgroup_identity['energy_bounds_mev'] == (20., 1., 1e-11)
    path = write(tmp_path, raw)
    path.write_text(path.read_text().replace('MICRO_NG (idx, [1:1]) = [ 2 ]', 'MICRO_NG (idx, [1:1]) = compute()'))
    with pytest.raises(ValueError, match='MICRO_NG'):
        read_serpent_xs(path)


@pytest.mark.parametrize('replacement', [
    {'MICRO_NG': np.array([2])}, {'MICRO_E': np.array([20., 1., 0.])},
    {'MICRO_NG': np.array([2.5]), 'MICRO_E': np.array([20., 1., 0.])},
    {'MICRO_NG': np.array([2]), 'MICRO_E': np.array([20., 0.])},
    {'MICRO_NG': np.array([2]), 'MICRO_E': np.array([20., 1., 1.])},
])
def test_parser_rejects_malformed_micro_fields(tmp_path, replacement):
    raw = fields(); raw.update(replacement)
    with pytest.raises(ValueError, match='MICRO_'):
        read_serpent_xs(write(tmp_path, raw))


def test_manual_declaration_is_validated_but_native_remains_authoritative():
    identity = known().microgroup_identity
    manual = state('manual', metadata={'microgroup_identity': identity})
    assert manual.microgroup_identity == identity
    with pytest.raises(ValueError, match='microgroup_identity'):
        state('bad', metadata={'microgroup_identity': {**dict(identity), 'diffusion': 'cmm'}})
    with pytest.raises(ValueError, match='conflicts with retained native'):
        replace(manual, native=native(None))


def test_import_rejects_per_universe_microgrid_mismatch(tmp_path):
    raw = fields()
    raw.update(INF_DIFFCOEF=raw['CMM_DIFFCOEF'].copy(),
               MICRO_NG=np.array([2]), MICRO_E=np.array([20., 1., 1e-11]))
    a = read_serpent_xs(write(tmp_path, raw))
    raw['MICRO_E'] = np.array([20., .5, 1e-11])
    b = read_serpent_xs(write(tmp_path, raw))
    with pytest.raises(ValueError, match='inconsistent native microgroup'):
        import_serpent_state('mixed', {'fuel_temperature': 600., 'moderator_density': .7}, {'u': a, 'v': b})


def test_legacy_reload_rejects_different_native_grids(tmp_path):
    path = tmp_path / 'legacy-mixed.json'
    line(known(t=400.), known('b', t=900.)).save(path)
    payload = json.loads(path.read_text())
    for item in payload['states']:
        del item['metadata']['microgroup_identity']
    payload['states'][1]['native']['records']['u']['fields']['MICRO_E'][1] = .5
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='incompatible microgroup'):
        BranchLibrary.load(path)


@pytest.mark.parametrize('other', [(20., .5, 1e-11), None])
def test_unrelated_archive_exact_selection_allowed_but_mixed_vertices_rejected(other):
    a = known(t=400., metadata={'serpent_version': '2.2.0'})
    b = state('b', t=900., native=native(other), metadata={'serpent_version': '2.2.2'})
    library = BranchLibrary([a, b], axes=('fuel_temperature',),
                            compatible_serpent_versions=('2.2.0', '2.2.2'))
    assert library.select('a').microgroup_identity == a.microgroup_identity
    assert library.evaluate(b.conditions).microgroup_identity == b.microgroup_identity
    with pytest.raises(ValueError, match='incompatible microgroup'):
        library._compatible([a, b])
    # Recheck the selected vertices even if a caller changes the domain later.
    library.simplices = (('a', 'b'),)
    with pytest.raises(ValueError, match='incompatible microgroup'):
        library.evaluate({**dict(a.conditions), 'fuel_temperature': 625.})
