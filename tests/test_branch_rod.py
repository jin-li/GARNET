"""Manufactured geometry oracles for experimental rod interpolation."""
from dataclasses import replace

import numpy as np
import pytest

from garnet.branch_rod import evaluate_rod_occupancy
from garnet.branch_xs import BranchLibrary, BranchState, NativeBranchState


AXIS = 'rod_insertion_cm'


def state(name, position, *, high=False, metadata=None):
    x = 2. if high else 1.
    mask = np.zeros((3, 6), dtype=bool)
    mask[:, 5] = True
    factors = np.full((3, 6, 2), 9. if high else 1.)
    factors[mask] = 1
    fission = np.tile(np.array([[.04, .02], [.01, .03]]) * x, (3, 1, 1))
    if high:
        fission[:, 0, 1] = .19  # Not a separable chi * nu-fission oracle.
    return BranchState(
        name, {AXIS: position, 'configuration': 'bank-a', 'family': 'rod'},
        ('lower', 'upper', 'unchanged'), [20., .1, 0.],
        D=np.full((3, 2), x), sigma_a=np.full((3, 2), .1*x),
        sigma_s=np.tile(np.array([[.2, .03], [.02, .1]])*x, (3, 1, 1)),
        fission_matrix=fission, chi=np.tile([.2, .8] if high else [.8, .2], (3, 1)),
        kappa_fission=np.full((3, 2), .3*x), total=np.full((3, 2), .4*x),
        factors=factors, exterior_mask=mask,
        metadata=metadata or {'role': 'training', 'geometry_family': 'synthetic'},
        qualification='manufactured-only',
        native=NativeBranchState({'raw': {'native_error': [.01]}}))


def library(*, reverse=False, **kwargs):
    a, b = state('out', 0.), state('in', 10., high=True)
    return BranchLibrary([a, b], axes=(AXIS,),
                         simplices=[('in', 'out') if reverse else ('out', 'in')], **kwargs)


def request(position=5.):
    return {AXIS: position, 'configuration': 'bank-a', 'family': 'rod'}


def geometry(conditions, regions):
    x = conditions[AXIS] / 10
    return dict(zip(regions, (x*x, x**.5, .2)))


def evaluate(lib=None, **kwargs):
    return evaluate_rod_occupancy(lib or library(), request(), occupancy=geometry,
                                  geometry_id='manufactured-geometry-v1', **kwargs)


def test_nonlinear_geometry_full_channels_and_lineage():
    lib = library()
    out = evaluate(lib)
    a, b = lib.states.values()
    weights = np.array([.25, np.sqrt(.5), .5])
    for key in ('D', 'sigma_a', 'sigma_s', 'fission_matrix', 'chi', 'kappa_fission', 'total'):
        av, bv = getattr(a, key), getattr(b, key)
        w = weights.reshape((-1,) + (1,)*(av.ndim-1))
        expected = (1-w)*av + w*bv
        if key == 'chi':
            expected /= expected.sum(axis=1)[:, None]
        np.testing.assert_allclose(getattr(out, key), expected, rtol=1e-15)
        if key != 'chi':
            assert np.all(getattr(out, key) >= np.minimum(av, bv))
            assert np.all(getattr(out, key) <= np.maximum(av, bv))
    expected_adf = np.exp(weights[:, None, None] * np.log(b.factors))
    expected_adf[a.exterior_mask] = 1
    np.testing.assert_allclose(out.factors, expected_adf)
    assert np.all(out.factors[a.exterior_mask] == 1)
    assert out.weights == (.5, .5)
    assert out.contributing_ids == ('out', 'in')
    assert out.native is None
    assert out.metadata['role'] == 'prediction'
    assert out.qualification == 'experimental-rod-occupancy-unvalidated'
    assert out.metadata['rod_occupancy']['geometry_id'] == 'manufactured-geometry-v1'
    assert dict(out.metadata['rod_occupancy']['source_roles']) == {'out': 'training', 'in': 'training'}
    assert not out.metadata['rod_occupancy']['transport_reaction_rate_conservation']
    assert not np.allclose(out.fission_matrix, out.chi[:, :, None]*out.nu_sigma_f[:, None, :])
    global_out = lib.evaluate(request())
    np.testing.assert_array_equal(out.D[2], global_out.D[2])
    for arr in (out.D, out.factors, out.metadata['rod_occupancy']['local_right_weights']):
        with pytest.raises(ValueError):
            arr.setflags(write=True)
    with pytest.raises(TypeError):
        out.metadata['rod_occupancy']['geometry_id'] = 'changed'
    assert a.native is not None and a.metadata['role'] == 'training'
    assert 'rod_occupancy' not in a.metadata


def test_reverse_segment_has_same_physics_and_reversed_lineage():
    forward, reverse = evaluate(), evaluate(library(reverse=True))
    for key in ('D', 'sigma_a', 'sigma_s', 'fission_matrix', 'chi', 'factors'):
        np.testing.assert_allclose(getattr(forward, key), getattr(reverse, key), rtol=1e-15)
    assert reverse.contributing_ids == ('in', 'out')
    np.testing.assert_allclose(reverse.metadata['rod_occupancy']['local_right_weights'],
                               1-np.asarray(forward.metadata['rod_occupancy']['local_right_weights']))


@pytest.mark.parametrize('position', [0., 10.])
def test_exact_anchor_bypasses_provider_preserves_native_and_qualification(position):
    lib = library()
    def forbidden(*_):
        raise AssertionError('provider must not run for an exact state')
    out = evaluate_rod_occupancy(lib, request(position), occupancy=forbidden, geometry_id='exact')
    expected = lib.evaluate(request(position))
    assert out.native is expected.native
    assert out.qualification == expected.qualification
    assert out.metadata == expected.metadata
    assert not out.interpolated
    np.testing.assert_array_equal(out.factors, expected.factors)


def test_unchanged_geometry_equals_global_algebra_and_optional_channels():
    lib = library()
    states = [replace(s, total=None, kappa_fission=None) for s in lib.states.values()]
    lib = BranchLibrary(states, axes=(AXIS,), simplices=[('out', 'in')])
    out = evaluate_rod_occupancy(lib, request(3.), occupancy=lambda c, rs: dict.fromkeys(rs, .2),
                                 geometry_id='constant')
    expected = lib.evaluate(request(3.))
    for key in ('D', 'sigma_a', 'sigma_s', 'fission_matrix', 'chi', 'factors'):
        np.testing.assert_array_equal(getattr(out, key), getattr(expected, key))
    assert out.total is None and out.kappa_fission is None


@pytest.mark.parametrize('bad', [True, np.bool_(False), np.nan, np.inf, -1., 1.01,
                                 '0.5', [.5], np.array(.5), np.array([.5]), 1j])
def test_malformed_fraction_rejected(bad):
    with pytest.raises(ValueError, match='fractions'):
        evaluate_rod_occupancy(library(), request(), occupancy=lambda c, rs: dict.fromkeys(rs, bad),
                               geometry_id='bad')


@pytest.mark.parametrize('result', [None, [], {'lower': .2},
    {'lower': .2, 'upper': .2, 'unchanged': .2, 'extra': .1}])
def test_wrong_mapping_rejected(result):
    with pytest.raises(ValueError, match='exactly'):
        evaluate_rod_occupancy(library(), request(), occupancy=lambda *_: result, geometry_id='bad')


@pytest.mark.parametrize('constant', [False, True])
def test_target_occupancy_must_be_bracketed_even_for_unchanged_endpoints(constant):
    def nonbracket(c, rs):
        value = .8 if c[AXIS] == 5 else (.2 if constant else c[AXIS]/20)
        return dict.fromkeys(rs, value)
    with pytest.raises(ValueError, match='bracket'):
        evaluate_rod_occupancy(library(), request(), occupancy=nonbracket, geometry_id='bad')


@pytest.mark.parametrize('bad_conditions', [request(-.01), request(10.01),
    {**request(), 'configuration': 'another'}, {**request(), 'family': 'thermal'},
    {**request(), AXIS: True}])
def test_public_domain_and_fixed_condition_guards_run_before_geometry(bad_conditions):
    def forbidden(*_):
        raise AssertionError('provider must not be called before public guards')
    with pytest.raises(ValueError):
        evaluate_rod_occupancy(library(), bad_conditions, occupancy=forbidden, geometry_id='bad')


def test_closure_and_version_guards_stay_in_force():
    a = state('out', 0., metadata={'serpent_version': '2.2.0', 'transverse_leakage_identity': 'a'})
    b = state('in', 10., high=True, metadata={'serpent_version': '2.2.2', 'transverse_leakage_identity': 'b'})
    with pytest.raises(ValueError, match='incompatible'):
        BranchLibrary([a,b], axes=(AXIS,), simplices=[('out','in')],
                      compatible_serpent_versions=('2.2.0','2.2.2'))
    b = replace(b, metadata={**dict(b.metadata), 'transverse_leakage_identity': 'a'})
    with pytest.raises(ValueError, match='versions'):
        BranchLibrary([a,b], axes=(AXIS,), simplices=[('out','in')])
    lib = BranchLibrary([a,b], axes=(AXIS,), simplices=[('out','in')],
                        compatible_serpent_versions=('2.2.0','2.2.2'))
    out = evaluate(lib)
    assert dict(out.metadata['source_serpent_versions']) == {'out':'2.2.0', 'in':'2.2.2'}


@pytest.mark.parametrize('identity', ['', ' ', None, 1])
def test_explicit_geometry_identity_required(identity):
    with pytest.raises(ValueError, match='geometry_id'):
        evaluate_rod_occupancy(library(), request(), occupancy=geometry, geometry_id=identity)


def test_callback_axis_type_and_serialization_contract(tmp_path):
    lib = library()
    with pytest.raises(TypeError, match='callable'):
        evaluate_rod_occupancy(lib, request(), occupancy=None, geometry_id='x')
    with pytest.raises(TypeError, match='BranchLibrary'):
        evaluate_rod_occupancy(None, request(), occupancy=geometry, geometry_id='x')
    with pytest.raises(ValueError, match='identity-transform'):
        evaluate(lib, axis='other')
    transformed = library(transforms={AXIS:'sqrt'})
    with pytest.raises(ValueError, match='identity-transform'):
        evaluate(transformed)
    path = tmp_path/'library.json'
    lib.save(path)
    before = path.read_bytes()
    evaluate(lib)
    lib.save(path)
    assert path.read_bytes() == before
    assert b'rod_occupancy' not in before
    loaded = BranchLibrary.load(path)
    np.testing.assert_array_equal(evaluate(loaded).D, evaluate(lib).D)
