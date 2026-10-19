"""Synthetic ACE records only; no licensed nuclear data fixtures."""
from hashlib import sha256

import numpy as np
import pytest

from ndgpu.ace_precursors import ACEPrecursorError, read_ace_precursors


def table(name='1001.00c', probabilities=((.3, .3), (.7, .7)), *,
          schemes=None, ranges=None, decay=(1e-10, 2e-10), outgoing=True):
    xss = [0.0]  # unrelated leading word tests one-based pointers
    bdd = 2
    for i, values in enumerate(probabilities):
        energies = ranges[i] if ranges else (1e-11, 20.)
        xss.append(decay[i])
        if schemes is None:
            xss.extend([0, len(energies)])
        else:
            xss.extend([1, len(energies), schemes[i], len(energies)])
        xss.extend(energies)
        xss.extend(values)
    dnedl = len(xss)+1 if outgoing else 0
    dned = dnedl + len(probabilities) if outgoing else 0
    if outgoing:
        xss.extend(1 + 4*i for i in range(len(probabilities)))
        for i in range(len(probabilities)):
            # Law payload is deliberately opaque to the metadata reader.
            xss.extend([0, 4, 4+4*i, 123.])
    nxs = [len(xss), 1001, 0, 0, 0, 0, 0, len(probabilities)] + [0]*8
    jxs = [0]*32
    jxs[24:27] = [bdd, dnedl, dned]
    lines = [f'{name} 1.0 5.1704e-8 01/01/26', 'synthetic test data']
    lines += [' '.join(['0 0.0']*4)]*4
    lines += [' '.join(map(str, nxs[i:i+8])) for i in range(0, 16, 8)]
    lines += [' '.join(map(str, jxs[i:i+8])) for i in range(0, 32, 8)]
    lines += [' '.join(map(str, xss[i:i+4])) for i in range(0, len(xss), 4)]
    return '\n'.join(lines)+'\n'


def read(tmp_path, text=None, name='1001.00c'):
    path = tmp_path/'synthetic.ace'
    path.write_text(table() if text is None else text)
    return read_ace_precursors(path, name)


def test_metadata_units_provenance_outgoing(tmp_path):
    text = table()
    result = read(tmp_path, text)
    assert result.source_sha256 == sha256(text.encode()).hexdigest()
    assert result.table_name == '1001.00c'
    assert result.temperature_mev == 5.1704e-8
    assert result.atomic_weight_ratio == 1.
    assert result.group_count == 2
    assert result.families[0].decay_inverse_shakes == 1e-10
    assert result.families[0].decay_s_inv == .01
    law = result.families[1].outgoing_laws[0]
    assert (law.relative_locator, law.law, law.data_locator) == (5, 4, 8)
    assert result.qualified_constant_branching() == (.3, .7)


def test_sum_printing_tolerance_does_not_erase_energy_dependence(tmp_path):
    result = read(tmp_path, table(probabilities=((.3,.3000001),(.7,.6999999))))
    with pytest.raises(ACEPrecursorError, match='varies'):
        result.qualified_constant_branching()
    assert result.qualified_constant_branching(constancy_atol=2e-7) == (.3,.7)


def test_union_grid_preserves_raw_and_normalization(tmp_path):
    result = read(tmp_path, table(probabilities=((.2, .3, .4), (.8, .6)),
                                 ranges=((1e-11, 1., 20.), (1e-11, 20.))))
    union = result.branching_on_union_grid()
    assert union.energy_mev == (1e-11, 1., 20.)
    assert union.raw_probabilities[0] == (.2, .3, .4)
    np.testing.assert_allclose(np.sum(union.normalized_probabilities, axis=0), 1.)
    assert union.probability_sums[1] != 1.
    with pytest.raises(ACEPrecursorError, match='varies'):
        result.qualified_constant_branching()


def test_constancy_covers_entire_range_not_endpoints_only(tmp_path):
    result = read(tmp_path, table(probabilities=((.3, .4, .3), (.7, .6, .7)),
                                 ranges=((0., 1., 20.), (0., 1., 20.))))
    with pytest.raises(ACEPrecursorError, match='varies'):
        result.qualified_constant_branching()


def test_explicit_linear_interpolation(tmp_path):
    result = read(tmp_path, table(schemes=(2, 2)))
    assert result.families[0].branching.breakpoints == (2,)
    assert result.families[0].branching.interpolation == (2,)
    assert result.qualified_constant_branching() == (.3, .7)


def test_unsupported_interpolation_is_preserved_not_evaluated(tmp_path):
    result = read(tmp_path, table(schemes=(5, 2)))
    assert result.families[0].branching.interpolation == (5,)
    with pytest.raises(ACEPrecursorError, match='Unqualified'):
        result.qualified_constant_branching()


def test_different_ranges_need_explicit_common_domain(tmp_path):
    result = read(tmp_path, table(ranges=((0., 20.), (1., 19.))))
    with pytest.raises(ACEPrecursorError, match='ranges differ'):
        result.qualified_constant_branching()
    assert result.qualified_constant_branching((1., 19.)) == (.3, .7)
    with pytest.raises(ACEPrecursorError, match='uncovered'):
        result.qualified_constant_branching((0., 20.))
    with pytest.raises(ACEPrecursorError, match='outside'):
        result.families[0].branching.evaluate([-1.])


def test_no_silent_constant_probability_renormalization(tmp_path):
    result = read(tmp_path, table(probabilities=((.2, .2), (.7, .7))))
    with pytest.raises(ACEPrecursorError, match='sum to unity'):
        result.qualified_constant_branching()
    assert result.branching_on_union_grid().probability_sums == pytest.approx((.9, .9))


def test_exact_name_and_duplicate_detection(tmp_path):
    result = read(tmp_path, table('1002.00c') + table(), '1001.00c')
    assert result.table_name == '1001.00c'
    with pytest.raises(ACEPrecursorError, match='found 0'):
        read(tmp_path, table(), '1001')
    with pytest.raises(ACEPrecursorError, match='found 2'):
        read(tmp_path, table()+table())
    with pytest.raises(ACEPrecursorError, match='exact table'):
        read(tmp_path, table(), '')


@pytest.mark.parametrize('content, message', [
    ('2.0.0 1001.00c\n', '2.x'),
    ('\0\0\0', 'Binary'),
    ('1001.00c 1.0 1e-8\n', 'Truncated'),
    (table().rsplit('\n', 2)[0]+'\n', 'truncated'),
    (table(name='1001.00t'), 'continuous-energy'),
    (table(decay=(0., 1e-10)), 'decay'),
    (table(probabilities=((-.1, .3), (.7, .7))), 'probabilities'),
    (table(ranges=((1., 1.), (0., 2.))), 'energies'),
])
def test_reject_malformed_and_unsupported(tmp_path, content, message):
    with pytest.raises(ACEPrecursorError, match=message):
        read(tmp_path, content, '1001.00t' if '1001.00t' in content else '1001.00c')


def test_binary_nonascii(tmp_path):
    path = tmp_path/'binary.ace'
    path.write_bytes(b'\xff\x00')
    with pytest.raises(ACEPrecursorError, match='Binary'):
        read_ace_precursors(path, '1001.00c')


def test_absent_outgoing_metadata_is_explicit(tmp_path):
    result = read(tmp_path, table(outgoing=False))
    assert all(f.outgoing_locator is None and f.outgoing_laws == () for f in result.families)


def test_endf_float_without_exponent_marker(tmp_path):
    result = read(tmp_path, table().replace('1e-10', '1.0-10'))
    assert result.families[0].decay_s_inv == .01


def test_pointer_bounds_and_law_cycle(tmp_path):
    text = table()
    lines = text.splitlines()
    words = lines[11].split()  # JXS 25..32
    words[0] = '999999'
    lines[11] = ' '.join(words)
    with pytest.raises(ACEPrecursorError, match='outside XSS'):
        read(tmp_path, '\n'.join(lines))
    # Locate DNED block and replace first next-law locator by self reference.
    jxs = [int(v) for line in text.splitlines()[8:12] for v in line.split()]
    header = text.splitlines()[:12]
    xss = [float(v) for line in text.splitlines()[12:] for v in line.split()]
    xss[jxs[26]-1] = 1
    cycle = '\n'.join(header + [' '.join(map(str, xss[i:i+4])) for i in range(0, len(xss), 4)])
    with pytest.raises(ACEPrecursorError, match='Cyclic'):
        read(tmp_path, cycle)


def rewrite_header(text, *, nxs_change=None, jxs_change=None):
    lines = text.splitlines()
    for begin, end, changes in ((6, 8, nxs_change), (8, 12, jxs_change)):
        if changes:
            words = [v for line in lines[begin:end] for v in line.split()]
            for index, value in changes.items():
                words[index] = str(value)
            lines[begin:end] = [' '.join(words[i:i+8]) for i in range(0, len(words), 8)]
    return '\n'.join(lines)+'\n'


def test_zero_families_is_explicit(tmp_path):
    text = rewrite_header(table(), nxs_change={7: 0}, jxs_change={24: 0, 25: 0, 26: 0})
    result = read(tmp_path, text)
    assert result.group_count == 0 and result.families == ()
    with pytest.raises(ACEPrecursorError, match='no delayed'):
        result.qualified_constant_branching()


@pytest.mark.parametrize('nxs,jxs,message', [
    ({7: 0}, None, 'zero family'),
    (None, {24: 0}, 'Missing BDD'),
    (None, {25: 0}, 'Incomplete'),
    (None, {26: 0}, 'Incomplete'),
    (None, {25: 3}, 'DNEDL locator'),
])
def test_inconsistent_header_pointers(tmp_path, nxs, jxs, message):
    with pytest.raises(ACEPrecursorError, match=message):
        read(tmp_path, rewrite_header(table(), nxs_change=nxs, jxs_change=jxs))


def test_malformed_interpolation_breakpoint(tmp_path):
    # One interpolation region must terminate at the last energy index.
    text = table(schemes=(2, 2)).splitlines()
    xss = [float(v) for line in text[12:] for v in line.split()]
    xss[3] = 3
    malformed = '\n'.join(text[:12] + [' '.join(map(str, xss[i:i+4])) for i in range(0, len(xss), 4)])
    with pytest.raises(ACEPrecursorError, match='breakpoints'):
        read(tmp_path, malformed)
