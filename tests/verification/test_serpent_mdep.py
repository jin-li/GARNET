"""Synthetic invented nuclides/data only; no nuclear-library fixture content."""
import hashlib

import numpy as np
import pytest

from ndgpu.serpent_mdep import read_serpent_mdep, production_xs, build_delayed_channels


TEXT = '''% Synthetic native-shaped two-group output
FLUX_fuel_01 = [10 .01 20 .02];
XS_fuel_01 = [
10010 452 0 0.02 .003 2 .04 4 .05
10010 455 0 0.02 .003 .02 .06 .08 .07
20020 452 0 0.01 .002 3 .03 5 .04
20020 455 0 0.01 .002 .06 .05 .10 .06
];
% Static tables are not kinetics input
nrea = [10010 452 0 0 0];
'''


def read(tmp_path, text=TEXT, **kwargs):
    path = tmp_path / 'input_mdx0.m'
    path.write_text(text)
    args = dict(energy_bounds_mev=[20, 1, 0], energy_order='descending', selectors=['fuel_01'])
    args.update(kwargs)
    return read_serpent_mdep(path, **args)


def test_native_rows_units_order_errors_and_immutable_provenance(tmp_path):
    result = read(tmp_path)
    assert result.source_sha256 == hashlib.sha256(TEXT.encode()).hexdigest()
    assert result.energy_bounds_mev == (20, 1, 0)
    region = result.regions['fuel_01']
    assert region.flux == (10, 20)
    assert region.flux_relative_error == (.01, .02)
    row = region.rows[10010, 455, 0]
    assert row.density == .02
    assert row.density_relative_error == .003
    assert row.xs == (.02, .08)
    assert row.xs_relative_error == (.06, .07)
    assert row.macroscopic_xs == pytest.approx((.0004, .0016))
    assert 'product_covariance_unavailable' in result.flags
    with pytest.raises(TypeError):
        region.rows[10010, 455, 0] = None
    with pytest.raises(TypeError):
        result.regions['other'] = region


def test_paired_production_and_explicit_coverage(tmp_path):
    region = read(tmp_path).regions['fuel_01']
    xs = production_xs(region, expected_isotopes=[10010, 20020])
    assert xs[10010].total == pytest.approx((.04, .08))
    assert xs[10010].delayed == pytest.approx((.0004, .0016))
    assert xs[10010].prompt == pytest.approx((.0396, .0784))
    for expected in ([10010], [10010, 20020, 30030], [10010, 10010], [10010.0, 20020]):
        with pytest.raises(ValueError):
            production_xs(region, expected_isotopes=expected)


@pytest.mark.parametrize('old,new', [
    ('10010 455 0 0.02 .003', '10010 455 0 0.03 .003'),
    ('10010 455 0 0.02 .003', '10010 455 0 0.02 .004'),
    ('10010 455 0', '10010 455 1'),
    ('.02 .06 .08 .07', '3 .06 5 .07'),
])
def test_inconsistent_production_pairs_fail(tmp_path, old, new):
    region = read(tmp_path, TEXT.replace(old, new)).regions['fuel_01']
    with pytest.raises(ValueError):
        production_xs(region, expected_isotopes=[10010, 20020])


@pytest.mark.parametrize('addition', [
    'FLUX_fuel_01 = [10 .01 20 .02];',
    'FLUX_bad = system("touch /tmp/never-execute-mdep");',
    'XS_bad = [1 2 3',
    'XS_bad(1,:) = [1 2 3];',
])
def test_duplicate_or_executable_or_unterminated_relevant_assignment_rejected(tmp_path, addition):
    with pytest.raises(ValueError):
        read(tmp_path, TEXT + addition)


@pytest.mark.parametrize('old,new', [
    ('10010 452', '10010.5 452'),
    ('10010 452 0', '10010 452 -1'),
    ('2 .04 4 .05', '2 -.04 4 .05'),
    ('2 .04 4 .05', '2 .04 nan .05'),
    ('2 .04 4 .05', '2 .04 4+1 .05'),
    ('2 .04 4 .05', '2 .04'),
    ('10010 455', '10010 452'),
    ('FLUX_fuel_01 = [10 .01 20 .02]', 'FLUX_fuel_01 = [10 .01 -20 .02]'),
])
def test_invalid_row_shapes_literals_keys_errors_and_flux_rejected(tmp_path, old, new):
    with pytest.raises(ValueError):
        read(tmp_path, TEXT.replace(old, new))


@pytest.mark.parametrize('kwargs', [
    dict(selectors=['fuel']), dict(selectors=['fuel_01', 'fuel_01']),
    dict(selectors=[1]), dict(energy_order='unknown'),
    dict(energy_order='ascending'), dict(energy_bounds_mev=[20, 20, 0]),
    dict(energy_bounds_mev=[20, 1, -1]), dict(energy_bounds_mev=[20, 0]),
])
def test_explicit_selection_and_energy_contract_required(tmp_path, kwargs):
    with pytest.raises(ValueError):
        read(tmp_path, **kwargs)


def channel_inputs():
    return dict(expected_isotopes=[10010, 20020],
                branching={10010: [.25, .75], 20020: [.6, .4]},
                decay={10010: [.01, .1], 20020: [.02, .2]},
                chi_delayed={10010: [[.2, .8], [.7, .3]], 20020: [[.4, .6], [.9, .1]]},
                qualification={10010: 'Invented energy-independent synthetic branches',
                               20020: 'Invented energy-independent synthetic branches'})


def test_channels_preserve_isotope_family_target_source_and_raw_sum(tmp_path):
    region = read(tmp_path).regions['fuel_01']
    result = build_delayed_channels(region, **channel_inputs())
    assert result.labels == ((10010, 1), (10010, 2), (20020, 1), (20020, 2))
    assert result.decay == (.01, .1, .02, .2)
    c = np.array(result.production)
    assert c.shape == (4, 2, 2)
    assert c[0] == pytest.approx(np.outer([.2, .8], np.array([.0004, .0016]) * .25))
    assert c.sum(axis=(0, 1)) == pytest.approx([.001, .0026])
    assert not np.allclose(c[1], c[1].T)
    with pytest.raises(TypeError):
        result.qualification[10010] = 'changed'


@pytest.mark.parametrize('key,value', [
    ('branching', {10010: [.3, .8], 20020: [.6, .4]}),
    ('branching', {10010: [-.1, 1.1], 20020: [.6, .4]}),
    ('decay', {10010: [0, .1], 20020: [.02, .2]}),
    ('decay', {10010: [np.inf, .1], 20020: [.02, .2]}),
    ('branching', {10010: np.array([.25 + .1j, .75]), 20020: [.6, .4]}),
    ('chi_delayed', {10010: [[.2, .7], [.7, .3]], 20020: [[.4, .6], [.9, .1]]}),
    ('qualification', {10010: '', 20020: 'synthetic'}),
    ('qualification', {10010: 'synthetic'}),
])
def test_channel_builder_refuses_unqualified_or_inconsistent_inputs(tmp_path, key, value):
    kwargs = channel_inputs()
    kwargs[key] = value
    with pytest.raises(ValueError):
        build_delayed_channels(read(tmp_path).regions['fuel_01'], **kwargs)


def test_newline_and_semicolon_rows_d_exponents_and_zero_error_flags(tmp_path):
    text = TEXT.replace('0.02 .003', '2D-2 0').replace('\n10010 455', ';10010 455')
    result = read(tmp_path, text)
    assert result.regions['fuel_01'].rows[10010, 452, 0].density == .02
    assert 'printed_zero_relative_error:fuel_01' in result.flags


def test_empty_isotope_coverage_cannot_return_fabricated_zero_production(tmp_path):
    with pytest.raises(ValueError, match='expected isotopes'):
        production_xs(read(tmp_path).regions['fuel_01'], expected_isotopes=[])
