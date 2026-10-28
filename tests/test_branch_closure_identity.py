"""Synthetic closure-identity guards, independent of transport qualification."""
from dataclasses import replace
import json

import numpy as np
import pytest

from garnet.branch_xs import BranchLibrary, BranchState, NativeBranchState


CLOSURE = {
    'transverse_leakage_identity': {'method': 'synthetic-spatial', 'order': 2},
    'reflector_mask_sha256': 'a' * 64,
    'leakage_boundary': 'synthetic-zero-drive',
    'reference_current_convention': 'outward-face-average',
}
DIFFERENT = {
    'transverse_leakage_identity': {'method': 'synthetic-alternative', 'order': 2},
    'reflector_mask_sha256': 'b' * 64,
    'leakage_boundary': 'synthetic-alternative-boundary',
    'reference_current_convention': 'positive-coordinate-face-average',
}


def _state(name, temperature, metadata, *, upper=False):
    exterior = np.ones((2, 6), dtype=bool)
    exterior[0, 2] = exterior[1, 0] = False
    factors = np.ones((2, 6, 2))
    factors[~exterior] = 1.8 if upper else .8
    # Different emitted spectra produce a non-rank-one interpolated fuel F.
    fuel_fission = [[0., 0.], [.06, .12]] if upper else [[.02, .05], [0., 0.]]
    return BranchState(
        state_id=name,
        conditions={'fuel_temperature': temperature, 'configuration': 'synthetic',
                    'family': 'thermal'},
        region_ids=('fuel', 'reflector'), energy_bounds_mev=[10., .1, 0.],
        D=np.ones((2, 2)), sigma_a=np.full((2, 2), .02),
        sigma_s=np.zeros((2, 2, 2)),
        fission_matrix=[fuel_fission, np.zeros((2, 2))],
        chi=[[0., 1.] if upper else [1., 0.], [1., 0.]],
        factors=factors, exterior_mask=exterior,
        metadata={'adf_prescription': 'synthetic-only', **metadata},
    )


def _library(left_metadata, right_metadata, **kwargs):
    left = _state('low', 400., left_metadata)
    right = _state('high', 900., right_metadata, upper=True)
    return BranchLibrary([left, right], axes=('fuel_temperature',),
                         simplices=[('low', 'high')], **kwargs)


@pytest.mark.parametrize('metadata', [CLOSURE, {}], ids=['declared', 'legacy-absent'])
def test_matching_closures_preserve_full_fission_adfs_and_unvalidated_status(metadata):
    library = _library(metadata, metadata)
    left, right = library.states['low'], library.states['high']
    for source in (left, right):
        exact = library.evaluate(source.conditions)
        assert exact.contributing_ids == (source.state_id,)
        assert not exact.interpolated
        assert exact.qualification == 'unvalidated'
        for key in source.metadata:
            assert exact.metadata[key] == source.metadata[key]
        assert exact.metadata['source_serpent_versions'] == {source.state_id: None}
        assert exact.metadata['compatible_serpent_versions'] == ()
        np.testing.assert_array_equal(exact.fission_matrix, source.fission_matrix)
        np.testing.assert_array_equal(exact.factors, source.factors)

    result = library.evaluate({**dict(left.conditions), 'fuel_temperature': 625.})
    assert result.weights == (.5, .5)
    assert result.qualification == 'interpolated-unvalidated'
    expected = (left.fission_matrix + right.fission_matrix) / 2
    np.testing.assert_allclose(result.fission_matrix, expected)
    np.testing.assert_allclose(result.nu_sigma_f, expected.sum(axis=1))
    assert np.linalg.matrix_rank(result.fission_matrix[0]) == 2
    assert not np.allclose(result.fission_matrix,
                           result.chi[:, :, None] * result.nu_sigma_f[:, None, :])
    np.testing.assert_allclose(result.factors[~left.exterior_mask], 1.2)
    np.testing.assert_array_equal(result.factors[left.exterior_mask], 1.)
    for key in metadata:
        assert result.metadata[key] == left.metadata[key]
    if not metadata:
        assert not set(CLOSURE).intersection(result.metadata)


@pytest.mark.parametrize('key', CLOSURE)
def test_mismatched_closure_rejected_when_constructing_interpolation_element(key):
    with pytest.raises(ValueError, match='incompatible'):
        _library(CLOSURE, {**CLOSURE, key: DIFFERENT[key]})


@pytest.mark.parametrize('key', CLOSURE)
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('explicit_null', [False, True])
def test_closure_declaration_cannot_match_omission(key, reverse, explicit_null):
    declared = {**CLOSURE, key: None} if explicit_null else dict(CLOSURE)
    omitted = {k: v for k, v in CLOSURE.items() if k != key}
    left, right = (omitted, declared) if reverse else (declared, omitted)
    with pytest.raises(ValueError, match='incompatible spatial-reference closure'):
        _library(left, right)


VERSIONS = ('2.2.0', '2.2.2')


def _versioned_library(policy=(), *, right_version='2.2.2', right_changes=None):
    states = []
    for name, temperature, version, upper in (
            ('low', 400., '2.2.0', False), ('high', 900., right_version, True)):
        metadata = {**CLOSURE, 'serpent_version': version,
                    'nuclear_data_family': 'synthetic-library'}
        if upper:
            metadata.update(right_changes or {})
        state = _state(name, temperature, metadata, upper=upper)
        states.append(replace(state, native=NativeBranchState(
            {'fuel': {'VERSION': version}}, metadata={'serpent_version': version})))
    return BranchLibrary(states, axes=('fuel_temperature',), simplices=[('low', 'high')],
                         compatible_serpent_versions=policy)


def test_mixed_versions_require_explicit_policy():
    with pytest.raises(ValueError, match='incompatible Serpent versions'):
        _versioned_library()
    library = _versioned_library(VERSIONS)
    result = library.evaluate({**dict(library.states['low'].conditions), 'fuel_temperature': 625.})
    assert result.metadata['source_serpent_versions'] == dict(zip(('low', 'high'), VERSIONS))
    assert result.metadata['compatible_serpent_versions'] == VERSIONS
    assert 'serpent_version' not in result.metadata
    assert result.qualification == 'interpolated-unvalidated'
    np.testing.assert_allclose(result.fission_matrix,
                               (library.states['low'].fission_matrix + library.states['high'].fission_matrix) / 2)
    np.testing.assert_allclose(result.factors[~result.exterior_mask], 1.2)
    exact = library.evaluate(library.states['high'].conditions)
    assert exact.metadata['serpent_version'] == '2.2.2'
    assert exact.metadata['source_serpent_versions'] == {'high': '2.2.2'}
    assert exact.metadata['compatible_serpent_versions'] == VERSIONS


def test_version_policy_roundtrip_preserves_native_versions_and_predictions(tmp_path):
    library = _versioned_library(list(VERSIONS))
    path = tmp_path / 'versions.json'
    library.save(path)
    restored = BranchLibrary.load(path)
    assert restored.compatible_serpent_versions == VERSIONS
    for name, version in zip(('low', 'high'), VERSIONS):
        source = restored.states[name]
        assert source.metadata['serpent_version'] == version
        assert source.native.records['fuel']['VERSION'] == version
        assert source.native.metadata['serpent_version'] == version
        assert source.metadata == library.states[name].metadata
    result = restored.evaluate({**dict(restored.states['low'].conditions), 'fuel_temperature': 625.})
    # Provenance survives conversion to a JSON-compatible prediction record.
    prediction_metadata = json.loads(json.dumps({
        'source_serpent_versions': dict(result.metadata['source_serpent_versions']),
        'compatible_serpent_versions': result.metadata['compatible_serpent_versions'],
    }))
    assert prediction_metadata == {'source_serpent_versions': {'low': '2.2.0', 'high': '2.2.2'},
                                   'compatible_serpent_versions': list(VERSIONS)}
    assert 'serpent_version' not in result.metadata
    payload = json.loads(path.read_text())
    del payload['compatible_serpent_versions']
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='incompatible Serpent versions'):
        BranchLibrary.load(path)


def test_legacy_library_without_policy_loads_with_strict_default(tmp_path):
    library = _versioned_library(right_version='2.2.0')
    path = tmp_path / 'legacy.json'
    library.save(path)
    payload = json.loads(path.read_text())
    del payload['compatible_serpent_versions']
    path.write_text(json.dumps(payload))
    restored = BranchLibrary.load(path)
    assert restored.compatible_serpent_versions == ()
    result = restored.evaluate({**dict(restored.states['low'].conditions), 'fuel_temperature': 625.})
    assert result.metadata['serpent_version'] == '2.2.0'


@pytest.mark.parametrize('version', ['2.2.1', None, 'Serpent 2.2.2'])
def test_policy_rejects_unlisted_or_missing_exact_version(version):
    with pytest.raises(ValueError, match='incompatible Serpent versions'):
        _versioned_library(VERSIONS, right_version=version)


@pytest.mark.parametrize('change', [
    {'nuclear_data_family': 'other-library'},
    {'geometry_family': 'other-geometry'},
    {'scoring_identity': 'other-scoring'},
    *({key: DIFFERENT[key]} for key in CLOSURE),
])
def test_version_policy_never_waives_other_identities(change):
    with pytest.raises(ValueError, match='incompatible'):
        _versioned_library(VERSIONS, right_changes=change)


@pytest.mark.parametrize('policy', [
    '2.2.0', None, {'2.2.0', '2.2.2'}, ('2.2.0',), ['2.2.0', '2.2.0'],
    ('2.2.0', ''), ('2.2.0', '  '), ('2.2.0', 222), ('2.2.0', []),
])
def test_malformed_version_policy_rejected(policy):
    with pytest.raises(ValueError, match='compatible_serpent_versions'):
        _versioned_library(policy)
