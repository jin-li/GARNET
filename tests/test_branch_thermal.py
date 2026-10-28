"""Independent transformed-coordinate geometry and fail-closed star tests."""
from dataclasses import replace

import numpy as np
import pytest

from garnet.branch_thermal import build_thermal_star, ThermalStarEvidence
from garnet.branch_xs import BranchState, BranchLibrary, BenchmarkModeratorPath, NativeBranchState


def inputs():
    path = BenchmarkModeratorPath(temperatures=(500., 550., 600.),
                                 densities=(.9, .8, .7), path_id='test-path')
    points = [('center', 625., 550.), ('fl', 400., 550.), ('fh', 900., 550.),
              ('ml', 625., 500.), ('mh', 625., 600.)]
    states = []
    for i, (name, tf, tm) in enumerate(points):
        value = 1 + i
        states.append(BranchState(
            name, {'fuel_temperature': tf, 'moderator_temperature': tm,
                   'moderator_density': path.evaluate(tm), 'configuration': 'ARO',
                   'family': 'thermal', 'boron': 100.}, ('fuel',), [10., 0.],
            [[value]], [[.01 * value]], [[[.02 * value]]], [[[.03 * value]]], [[1.]],
            factors=np.full((1, 6, 1), np.exp(i / 10)),
            native=NativeBranchState({'fuel': {'source_sha256': name}}),
            metadata={'role': 'nominal' if i == 0 else 'training',
                      'geometry_family': 'manufactured-pin',
                      'nuclear_data_family': 'manufactured-data',
                      'scoring_identity': 'manufactured-regions',
                      'adf_prescription': 'manufactured-whole-face',
                      'solver_family': 'manufactured-solver',
                      'moderator_path_id': path.path_id, 'library_identity': 'same-library'},
            qualification='qualified'))
    return states, path


def build(states, path, **kwargs):
    return build_thermal_star(*states, path=path, **kwargs)


def test_four_triangles_analytic_weights_absolute_channels_and_diamond():
    states, path = inputs()
    library = build(states, path, diagnostic=True)
    assert library.simplices == (('center', 'fl', 'ml'), ('center', 'ml', 'fh'),
                                 ('center', 'fh', 'mh'), ('center', 'mh', 'fl'))
    by_id = {s.state_id: s for s in states}
    for ids in library.simplices:
        vertices = [by_id[i] for i in ids]
        # Independent weighted coordinates in the prescribed transformed space.
        expected_weights = np.array([.5, .2, .3])
        tf = sum(w * np.sqrt(s.conditions['fuel_temperature'])
                 for w, s in zip(expected_weights, vertices)) ** 2
        rho = sum(w * s.conditions['moderator_density']
                  for w, s in zip(expected_weights, vertices))
        conditions = {**dict(library.states['center'].conditions),
                      'fuel_temperature': tf, 'moderator_density': rho}
        out = library.evaluate(conditions)
        assert out.contributing_ids == ids
        np.testing.assert_allclose(out.weights, expected_weights, atol=1e-14)
        for channel in ('D', 'sigma_a', 'sigma_s', 'fission_matrix'):
            expected = sum(w * getattr(s, channel) for w, s in zip(expected_weights, vertices))
            np.testing.assert_allclose(getattr(out, channel), expected)
        expected_factors = np.exp(sum(w * np.log(s.factors) for w, s in zip(expected_weights, vertices)))
        np.testing.assert_allclose(out.factors, expected_factors)
        assert out.qualification == 'interpolated-unvalidated'
    for tf in (400., 900.):
        for rho in (.7, .9):
            with pytest.raises(ValueError, match='outside'):
                library.evaluate({**conditions, 'fuel_temperature': tf, 'moderator_density': rho})
    assert library.select('center').conditions['fuel_temperature'] == 625.
    assert library.select('center').qualification == 'diagnostic-unvalidated'
    assert states[0].qualification == 'qualified'


def test_evidence_policy_required_even_for_qualified_string_and_preserved(tmp_path):
    states, path = inputs()
    with pytest.raises(ValueError, match='qualification policy required'):
        build(states, path)
    with pytest.raises(ValueError, match='ThermalStarEvidence'):
        build(states, path, qualification_policy=lambda s: True)
    seen = []
    def policy(s):
        seen.append(s)
        return ThermalStarEvidence('external-review-v1', (f'audit:{s.state_id}',))
    library = build(states, path, qualification_policy=policy)
    assert seen == states
    target = tmp_path / 'star.json'
    library.save(target)
    restored = BranchLibrary.load(target)
    assert restored.simplices == library.simplices
    for state in restored.states.values():
        assert state.metadata['thermal_star']['evidence']['policy_id'] == 'external-review-v1'
        assert state.native is not None
    out = restored.evaluate_temperatures(26. ** 2, 560., path,
                                         conditions={'boron': 100.}, configuration='ARO', family='thermal')
    assert out.interpolated and out.qualification == 'interpolated-unvalidated'
    assert out.metadata['evaluated_moderator_temperature'] == 560.
    assert out.conditions['moderator_density'] == pytest.approx(.78)


@pytest.mark.parametrize('index,change,match', [
    (1, {'metadata': {'role': 'withheld'}}, 'role'),
    (0, {'metadata': {'role': 'training'}}, 'role'),
    (2, {'conditions': {'boron': 200.}}, 'fixed conditions'),
    (2, {'conditions': {'configuration': 'ARI'}}, 'fixed conditions'),
    (2, {'conditions': {'family': 'rod'}}, 'fixed conditions'),
    (2, {'metadata': {'library_identity': 'different'}}, 'incompatible'),
    (3, {'metadata': {'moderator_path_id': 'other-path'}}, 'path identity'),
    (3, {'conditions': {'moderator_density': .89}}, 'does not match'),
    (3, {'conditions': {'fuel_temperature': 630.}}, 'fixed thermal coordinate'),
    (1, {'conditions': {'moderator_temperature': 551., 'moderator_density': .798}}, 'fixed thermal coordinate'),
])
def test_reject_roles_fixed_conditions_identities_and_off_axis(index, change, match):
    states, path = inputs()
    changes = {key: {**dict(getattr(states[index], key)), **value} for key, value in change.items()}
    states[index] = replace(states[index], **changes)
    with pytest.raises(ValueError, match=match):
        build(states, path, diagnostic=True)


def test_roles_mapping_cannot_override_withheld_and_requires_exact_membership():
    states, path = inputs()
    roles = {s.state_id: 'nominal' if i == 0 else 'training' for i, s in enumerate(states)}
    plain = [replace(s, metadata={k: v for k, v in s.metadata.items() if k != 'role'}) for s in states]
    assert build(plain, path, roles=roles, diagnostic=True)
    states[1] = replace(states[1], metadata={**dict(states[1].metadata), 'role': 'withheld'})
    with pytest.raises(ValueError, match='role'):
        build(states, path, roles=roles, diagnostic=True)
    with pytest.raises(ValueError, match='exactly'):
        build(plain, path, roles={}, diagnostic=True)


@pytest.mark.parametrize('identity', ['geometry_family', 'nuclear_data_family',
                                    'scoring_identity', 'adf_prescription', 'solver_family'])
@pytest.mark.parametrize('value', [None, '', '   ', False])
@pytest.mark.parametrize('diagnostic', [True, False])
def test_missing_or_empty_identity_rejected_even_when_all_sources_agree(identity, value, diagnostic):
    states, path = inputs()
    modified = []
    for s in states:
        metadata = dict(s.metadata)
        if value is None:
            metadata.pop(identity)
        else:
            metadata[identity] = value
        modified.append(replace(s, metadata=metadata))
    options = {'diagnostic': True} if diagnostic else {
        'qualification_policy': lambda s: ThermalStarEvidence('review', ('audit',))}
    with pytest.raises(ValueError, match=identity):
        build(modified, path, **options)


@pytest.mark.parametrize('identity', ['geometry_family', 'nuclear_data_family',
                                    'scoring_identity', 'adf_prescription', 'solver_family'])
def test_explicit_but_incompatible_identity_rejected(identity):
    states, path = inputs()
    states[2] = replace(states[2], metadata={**dict(states[2].metadata), identity: 'different'})
    with pytest.raises(ValueError, match='incompatible'):
        build(states, path, diagnostic=True)


def test_reversed_degenerate_nonnative_and_evaluated_inputs():
    states, path = inputs()
    reversed_states = [states[0], states[2], states[1], states[3], states[4]]
    with pytest.raises(ValueError, match='bracket'):
        build(reversed_states, path, diagnostic=True)
    degenerate = list(states)
    degenerate[1] = replace(states[1], conditions=states[0].conditions)
    with pytest.raises(ValueError, match='degenerate'):
        build(degenerate, path, diagnostic=True)
    for replacement in (replace(states[0], native=None),
                        BranchLibrary([states[0]]).select('center')):
        with pytest.raises(ValueError, match='native-backed'):
            build([replacement, *states[1:]], path, diagnostic=True)


def test_diagnostic_persistence_and_boundary_selection(tmp_path):
    states, path = inputs()
    library = build(states, path, diagnostic=True)
    target = tmp_path / 'diagnostic.json'
    library.save(target)
    restored = BranchLibrary.load(target)
    assert all(s.qualification == 'diagnostic-unvalidated' for s in restored.states.values())
    assert all(s.metadata['thermal_star']['construction'] == 'diagnostic-unvalidated'
               for s in restored.states.values())
    for original in states:
        exact = restored.evaluate_temperatures(original.conditions['fuel_temperature'],
                  original.conditions['moderator_temperature'], path,
                  conditions={'boron': 100.}, configuration='ARO', family='thermal')
        assert exact.contributing_ids == (original.state_id,)
        np.testing.assert_array_equal(exact.sigma_a, original.sigma_a)


@pytest.mark.parametrize('diagnostic', [True, False])
def test_explicit_version_policy_reaches_thermal_star_and_survives_roundtrip(tmp_path, diagnostic):
    states, path = inputs()
    policy = ('Serpent 2.2.0', 'Serpent 2.2.2')
    states = [replace(s, metadata={**dict(s.metadata), 'serpent_version': policy[bool(i)]},
                      native=NativeBranchState(dict(s.native.records),
                                               metadata={'serpent_version': policy[bool(i)]}))
              for i, s in enumerate(states)]
    options = {'diagnostic': True} if diagnostic else {
        'qualification_policy': lambda s: ThermalStarEvidence('manufactured-review', ('audit',))}
    with pytest.raises(ValueError, match='incompatible Serpent versions'):
        build(states, path, **options)
    library = build(states, path, compatible_serpent_versions=policy, **options)
    target = tmp_path / 'mixed-version-star.json'
    library.save(target)
    restored = BranchLibrary.load(target)
    assert restored.compatible_serpent_versions == policy
    result = restored.evaluate_temperatures(625., 560., path,
                                            conditions={'boron': 100.}, configuration='ARO', family='thermal')
    assert set(result.metadata['source_serpent_versions'].values()) == set(policy)
    assert 'serpent_version' not in result.metadata
    assert result.qualification == 'interpolated-unvalidated'
    assert restored.states['center'].native.metadata['serpent_version'] == policy[0]
    assert restored.states['mh'].native.metadata['serpent_version'] == policy[1]
    # The application policy cannot authorize a change of nuclear-data identity.
    states[2] = replace(states[2], metadata={**dict(states[2].metadata),
                                           'nuclear_data_family': 'different-data'})
    with pytest.raises(ValueError, match='incompatible'):
        build(states, path, compatible_serpent_versions=policy, **options)
