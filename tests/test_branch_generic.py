"""Application-owned paths, compatibility contracts and metric-only reports."""
from dataclasses import replace

import numpy as np
import pytest

from garnet.branch_thermal import build_thermal_star
from garnet.branch_validation import (
    assess_rod_worth, coefficient_convergence, compare_reconstruction_chain,
    normalized_power_rms,
)
from garnet.branch_xs import (
    BenchmarkModeratorPath, BranchLibrary, BranchState, NativeBranchState,
    TabulatedModeratorPath,
)


def state(name, temperature, metadata):
    # A three-group, full-rank fission source, unrelated to PRATIC inputs.
    return BranchState(
        name, {'fuel_temperature': temperature}, ('salt',), [20., 1., .01, 0.],
        [[1., 2., 3.]], [[.1, .2, .3]], np.zeros((1, 3, 3)),
        [np.diag([.3, .2, .1])], [[.5, .3, .2]], metadata=metadata)


def library(left, right, **kwargs):
    return BranchLibrary([state('a', 400., left), state('b', 900., right)],
                         axes=('fuel_temperature',), simplices=[('a', 'b')], **kwargs)


def test_custom_identity_complete_comparison_and_policy_roundtrip(tmp_path):
    identities = {'dataset': 'salt-v4', 'processing': {'route': 'local', 'option': None},
                  'group_order': ['fast', 'epithermal', 'thermal']}
    versions = ('producer-1', 'producer-2')
    left = {'compatibility_identities': identities, 'serpent_version': versions[0]}
    right = {**left, 'serpent_version': versions[1]}
    source = library(left, right, compatible_serpent_versions=versions)
    target = tmp_path / 'custom.json'
    source.save(target)
    restored = BranchLibrary.load(target)
    assert restored.compatible_serpent_versions == versions
    assert restored.states['a'].metadata == source.states['a'].metadata
    result = restored.evaluate({'fuel_temperature': 625.})
    assert result.metadata['compatibility_identities']['processing']['option'] is None
    assert result.metadata['source_serpent_versions'] == dict(zip(('a', 'b'), versions))
    np.testing.assert_array_equal(result.fission_matrix, [np.diag([.3, .2, .1])])
    assert np.linalg.matrix_rank(result.fission_matrix[0]) == 3
    with pytest.raises(ValueError, match='caller compatibility'):
        library(left, {**right, 'compatibility_identities': {**identities, 'extra': None}},
                compatible_serpent_versions=versions)


@pytest.mark.parametrize('left,right', [
    ({}, {'compatibility_identities': {}}),
    ({'compatibility_identities': {}}, {'compatibility_identities': {'option': None}}),
    ({'compatibility_identities': {'process': {}}},
     {'compatibility_identities': {'process': {'option': None}}}),
])
def test_missing_custom_identity_differs_from_null_in_both_directions(left, right):
    for a, b in ((left, right), (right, left)):
        with pytest.raises(ValueError, match='caller compatibility'):
            library(a, b)


@pytest.mark.parametrize('identities', [None, [], 'route', {1: 'route'}, {'': 'route'},
                                      {'nested': {1: 'route'}}])
def test_namespace_requires_named_identity_mapping(identities):
    with pytest.raises(ValueError, match='compatibility_identities'):
        state('a', 400., {'compatibility_identities': identities})


def test_legacy_nonclosure_null_policy_preserved():
    assert library({}, {'geometry_family': None})
    with pytest.raises(ValueError, match='closure identity declarations'):
        library({}, {'transverse_leakage_identity': None})


@pytest.mark.parametrize('densities', [(1.1, 1., .8), (.8, 1., 1.1)])
def test_nonbenchmark_path_monotone_bounded_and_exact(densities):
    path = TabulatedModeratorPath((300., 400., 700.), densities, 'my-fluid')
    assert path.pressure_bar is None
    for temperature, density in zip(path.temperatures, densities):
        assert path.evaluate(temperature) == pytest.approx(density)
    values = [path.evaluate(t) for t in np.linspace(300., 700., 201)]
    assert min(values) >= min(densities) and max(values) <= max(densities)
    assert np.all(np.diff(values) < 0) if densities[0] > densities[-1] else np.all(np.diff(values) > 0)
    for bad in (299., 701., np.nan, True, '400', 400j):
        with pytest.raises(ValueError, match='outside tabulated path'):
            path.evaluate(bad)


@pytest.mark.parametrize('changes', [
    {'temperatures': (300., 300., 500.)}, {'temperatures': (0., 400., 500.)},
    {'densities': (.9, .9, .7)}, {'densities': (.9, .7, .8)},
    {'densities': (.9, 0., -.1)}, {'densities': (.9, np.nan, .7)},
    {'densities': (.9, .7)}, {'path_id': ' '},
    {'pressure_bar': 0.}, {'pressure_bar': True},
])
def test_path_validation(changes):
    arguments = dict(temperatures=(300., 400., 500.), densities=(.9, .8, .7),
                     path_id='local-fluid', pressure_bar=None)
    with pytest.raises(ValueError):
        TabulatedModeratorPath(**{**arguments, **changes})


def test_legacy_path_positional_api_and_defaults():
    path = BenchmarkModeratorPath()
    assert path.pressure_bar == 155.
    assert path.evaluate(573.15) == .72683
    custom = BenchmarkModeratorPath((300., 500.), (1., .8), 155., 'legacy-custom')
    assert custom.path_id == 'legacy-custom'
    with pytest.raises(ValueError, match='155 bar'):
        BenchmarkModeratorPath(pressure_bar=20.)


@pytest.mark.parametrize('pressure', [None, 20.])
@pytest.mark.parametrize('densities', [(1., .9, .8), (.8, .9, 1.)])
def test_nonbenchmark_three_group_thermal_star(pressure, densities, tmp_path):
    path = TabulatedModeratorPath((300., 400., 500.), densities, 'custom-fluid', pressure)
    points = [('center', 625., 400.), ('fl', 400., 400.), ('fh', 900., 400.),
              ('ml', 625., 300.), ('mh', 625., 500.)]
    states = []
    for name, tf, tm in points:
        metadata = {key: 'local-contract' for key in (
            'geometry_family', 'nuclear_data_family', 'scoring_identity',
            'adf_prescription', 'solver_family')}
        metadata.update(moderator_path_id=path.path_id,
                        role='nominal' if name == 'center' else 'training',
                        compatibility_identities={'application': 'salt-test'})
        base = state(name, tf, metadata)
        states.append(replace(base, conditions={**dict(base.conditions),
                              'moderator_temperature': tm, 'moderator_density': path.evaluate(tm)},
                              native=NativeBranchState({'salt': {'source': name}})))
    source = build_thermal_star(*states, path=path, diagnostic=True)
    target = tmp_path / 'star.json'
    source.save(target)
    result = BranchLibrary.load(target).evaluate_temperatures(625., 420., path)
    assert result.interpolated
    assert result.metadata['moderator_pressure_bar'] == pressure
    assert result.fission_matrix.shape == (1, 3, 3)
    assert result.qualification == 'interpolated-unvalidated'


def test_metric_only_thresholds_never_accept_even_perfect_agreement():
    assert normalized_power_rms([1, 2], [1, 2], target=None)['status'] == 'descriptive'
    coefficient = {'value_pcm_per_K': -2., 'standard_error_pcm_per_K': 0.}
    assert coefficient_convergence(coefficient, coefficient, relative_target=None)['status'] == 'descriptive'
    worth = [{'interval_cm': [0., 1.], 'value_pcm_per_cm': -2., 'standard_error_pcm_per_cm': 0.}]
    assert assess_rod_worth(worth, worth, required_intervals=[[0, 1]],
                            reference_peak=2., peak_is_resolved=True,
                            relative_target=None)['status'] == 'descriptive'
    report = compare_reconstruction_chain(1., 1., 1., target_pcm=None, power_target=None,
                                         power_interpolated=[1, 2], power_exact=[1, 2], power_ce=[1, 2])
    assert report['status'] == 'descriptive'
    assert report['comparisons']['interpolated_vs_exact']['target'] is None
    assert report['comparisons']['interpolated_vs_exact']['power']['target_fraction'] is None


def test_chain_forwards_explicit_power_target_and_cannot_hide_unassessed_metric():
    powers = dict(power_interpolated=[1, 2], power_exact=[2, 1], power_ce=[2, 1])
    assert compare_reconstruction_chain(1, 1, 1, power_target=1., **powers)['status'] == 'pass'
    assert compare_reconstruction_chain(1, 1, 1, power_target=.1, **powers)['status'] == 'fail'
    assert compare_reconstruction_chain(1, 1, 1, power_target=None, **powers)['status'] == 'descriptive'
    assert compare_reconstruction_chain(1, 1, 1, target_pcm=None, power_target=1., **powers)['status'] == 'descriptive'
    with pytest.raises(ValueError):
        normalized_power_rms([1, -1], [1, 2], target=None)
    with pytest.raises(ValueError):
        compare_reconstruction_chain(0, 1, 1, target_pcm=None)
