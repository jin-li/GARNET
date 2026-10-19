"""Analytic synthetic LAW4 fixtures, without distributed nuclear data."""
from hashlib import sha256

import numpy as np
import pytest

from ndgpu.ace_precursors import ACEPrecursorError
from ndgpu.ace_delayed_spectra import (
    ACEOutgoingPDF, ACEDelayedFamilySpectrum, read_ace_delayed_spectra,
)


def uniform(lo=0., hi=1.):
    return ACEOutgoingPDF((lo, hi), (1/(hi-lo), 1/(hi-lo)), (0., 1.), 1)


def family(distributions, interpolation=()):
    return ACEDelayedFamilySpectrum((1., 3.), (2,) if interpolation else (),
                                    interpolation, tuple(distributions))


def synthetic_ace(distributions=None, *, incident_interp=None, law=4, flag=None):
    distributions = distributions or (uniform(), uniform())
    xss = [1e-10, 0, 2, 1., 3., 1., 1.]  # BDD, one family
    dnedl = len(xss)+1
    xss.append(1)
    dned = len(xss)+1
    base = len(xss)
    xss.extend([0, law, 10, 0, 2, 1., 3., 1., 1.])  # law + applicability
    if incident_interp is None:
        xss.extend([0, 2, 1., 3., 0, 0])
    else:
        xss.extend([1, 2, incident_interp, 2, 1., 3., 0, 0])
    locator_indices = (len(xss)-2, len(xss)-1)
    for index, d in zip(locator_indices, distributions):
        xss[index] = len(xss)-base+1
        xss.extend([d.interpolation if flag is None else flag, len(d.energy_mev)])
        xss.extend(d.energy_mev)
        xss.extend(d.pdf_per_mev)
        xss.extend(d.stored_cdf)
    nxs = [len(xss), 1001, 0, 0, 0, 0, 0, 1]+[0]*8
    jxs = [0]*32
    jxs[24:27] = (1, dnedl, dned)
    header = ['1001.00c 1.0 5.1704e-8 01/01/26', 'synthetic delayed LAW4']
    header += ['0 0.0 '*4]*4
    header += [' '.join(map(str, nxs[i:i+8])) for i in range(0, 16, 8)]
    header += [' '.join(map(str, jxs[i:i+8])) for i in range(0, 32, 8)]
    return '\n'.join(header+[' '.join(map(str, xss[i:i+4])) for i in range(0, len(xss), 4)])+'\n'


def read(tmp_path, text=None):
    path = tmp_path/'synthetic.ace'
    path.write_text(synthetic_ace() if text is None else text)
    return read_ace_delayed_spectra(path, '1001.00c')


def test_linear_pdf_exact_irregular_group_integrals():
    d = ACEOutgoingPDF((0., .3, 1.), (0., .6, 2.), (0., .09, 1.), 2)
    f = family((d, d))
    bounds = np.array([0., .15, .4, .9, 1.])
    result = f.integrate_groups(2., bounds)
    np.testing.assert_allclose(result.probabilities, np.diff(bounds**2), atol=1e-15)
    assert result.below_group_range == result.above_group_range == 0
    assert sum(result.normalized_probabilities()) == pytest.approx(1.)
    assert d.maximum_cdf_discrepancy < 1e-15


def test_histogram_irregular_pdf_and_descending_groups():
    d = ACEOutgoingPDF((0., .2, .8, 2.), (1., .5, 5/12, 0.), (0., .2, .5, 1.), 1)
    f = family((d, d))
    result = f.integrate_groups(2., (2., 1.4, .5, .1, 0.))
    np.testing.assert_allclose(result.probabilities, (.25, .4, .25, .1))


def test_unit_base_interpolation_moves_support():
    f = family((uniform(0., 1.), uniform(2., 4.)))
    # Midpoint incident energy has support [1,2.5], not the original union [0,4].
    np.testing.assert_allclose(f.integrated_cdf(2., [0., 1., 1.75, 2.5, 5.]), [0, 0, .5, 1, 1])
    np.testing.assert_allclose(f.integrate_groups(2., (1., 1.75, 2.5)).probabilities, (.5, .5))


def test_unit_base_mixes_different_shapes():
    triangle = ACEOutgoingPDF((2., 4.), (0., 1.), (0., 1.), 2)
    f = family((uniform(0., 1.), triangle))
    # unit coordinate t=.25, incident fraction r=.5: .5*t + .5*t^2
    assert f.integrated_cdf(2., 1.+.25*1.5) == pytest.approx(.15625)


def test_histogram_incident_interpolation_and_exact_knots():
    f = family((uniform(0., 1.), uniform(2., 4.)), (1,))
    assert f.integrated_cdf(2., .5) == pytest.approx(.5)
    assert f.integrated_cdf(3., 3.) == pytest.approx(.5)
    assert f.integrated_cdf(1., .5) == pytest.approx(.5)


def test_incident_independence_is_exact_full_distribution_test():
    assert family((uniform(), uniform())).incident_independent
    assert not family((uniform(), uniform(0., 2.))).incident_independent


def test_uncovered_probability_reported_and_never_renormalized():
    f = family((uniform(), uniform()))
    with pytest.raises(ACEPrecursorError, match='Uncovered'):
        f.integrate_groups(2., (.1, .9))
    incomplete = f.integrate_groups(2., (.1, .9), require_complete=False)
    assert incomplete.below_group_range == pytest.approx(.1)
    assert incomplete.above_group_range == pytest.approx(.1)
    assert incomplete.probabilities == pytest.approx((.8,))
    with pytest.raises(ACEPrecursorError, match='Cannot normalize'):
        incomplete.normalized_probabilities()
    tiny = f.integrate_groups(2., (1e-12, 1.), require_complete=False)
    with pytest.raises(ACEPrecursorError, match='uncovered'):
        tiny.normalized_probabilities()


@pytest.mark.parametrize('energy,pdf,cdf,flag', [
    ((0., 1.), (-1., 1.), (0., 1.), 2),
    ((0., 0.), (1., 1.), (0., 1.), 1),
    ((0., 1.), (1., 1.), (1., 0.), 1),
    ((0., 1.), (1., 1.), (0., .9), 1),
    ((0., 1.), (.9, .9), (0., 1.), 1),
    ((0., 1.), (float('nan'), 1.), (0., 1.), 1),
    ((0., 1.), (1., 1.), (0., 1.), 12),
])
def test_reject_bad_pdf_cdf_or_discrete_lines(energy, pdf, cdf, flag):
    with pytest.raises(ACEPrecursorError):
        ACEOutgoingPDF(energy, pdf, cdf, flag)


@pytest.mark.parametrize('bounds', [(0., 1., .5), (-1., 1.), (0., 0.), (0., float('inf'))])
def test_reject_bad_group_bounds(bounds):
    with pytest.raises(ACEPrecursorError, match='Group bounds'):
        family((uniform(), uniform())).integrate_groups(2., bounds)


@pytest.mark.parametrize('energy', [0., 4., float('nan')])
def test_no_incident_extrapolation(energy):
    with pytest.raises(ACEPrecursorError, match='Incident energy'):
        family((uniform(), uniform())).integrate_groups(energy, (0., 1.))


def test_read_synthetic_law4_provenance(tmp_path):
    text = synthetic_ace()
    result = read(tmp_path, text)
    assert result.source_sha256 == sha256(text.encode()).hexdigest()
    assert result.table_name == '1001.00c'
    assert result.temperature_mev == 5.1704e-8
    assert len(result.families) == 1
    f = result.families[0]
    assert f.incident_independent
    np.testing.assert_allclose(f.integrate_groups(2., (0., .3, 1.)).probabilities, (.3, .7))


def test_read_varying_histogram_incident_law4(tmp_path):
    result = read(tmp_path, synthetic_ace((uniform(), uniform(2., 4.)), incident_interp=1))
    assert result.families[0].interpolation == (1,)
    assert not result.families[0].incident_independent
    assert result.families[0].integrated_cdf(2., .5) == pytest.approx(.5)


@pytest.mark.parametrize('changes,message', [
    ({'law': 7}, 'single outgoing LAW4'),
    ({'incident_interp': 5}, 'incident interpolation'),
    ({'flag': 12}, 'discrete lines'),
])
def test_parser_rejects_unqualified_formats(tmp_path, changes, message):
    with pytest.raises(ACEPrecursorError, match=message):
        read(tmp_path, synthetic_ace(**changes))


def test_print_precision_cdf_overshoot_is_preserved_and_explicitly_normalized():
    d = ACEOutgoingPDF((0., 1.), (1.+1e-8, 1.+1e-8), (0., 1.+1e-8), 1)
    f = family((d, d))
    result = f.integrate_groups(2., (0., .25, 1.))
    assert result.total_probability == pytest.approx(1.+1e-8, abs=1e-15)
    assert result.probabilities[0] != .25
    assert result.normalized_probabilities() == pytest.approx((.25, .75))
    assert d.stored_cdf[-1] == 1.+1e-8


def test_parser_checks_cdf_and_applicability(tmp_path):
    text = synthetic_ace().splitlines()
    xss = [float(v) for line in text[12:] for v in line.split()]
    xss[-1] = .8
    broken = '\n'.join(text[:12]+[' '.join(map(str, xss[i:i+4])) for i in range(0, len(xss), 4)])
    with pytest.raises(ACEPrecursorError, match='normalization'):
        read(tmp_path, broken)
    xss = [float(v) for line in text[12:] for v in line.split()]
    xss[15] = .9  # first law applicability probability
    broken = '\n'.join(text[:12]+[' '.join(map(str, xss[i:i+4])) for i in range(0, len(xss), 4)])
    with pytest.raises(ACEPrecursorError, match='applicability'):
        read(tmp_path, broken)


def test_parser_locator_cannot_point_outside_delayed_block(tmp_path):
    text = synthetic_ace().splitlines()
    xss = [float(v) for line in text[12:] for v in line.split()]
    xss[21] = 999999  # LAW4 first outgoing table locator
    broken = '\n'.join(text[:12]+[' '.join(map(str, xss[i:i+4])) for i in range(0, len(xss), 4)])
    with pytest.raises(ACEPrecursorError, match='out-of-range'):
        read(tmp_path, broken)


def test_very_large_group_upper_bound_does_not_extrapolate_or_overflow():
    f = family((uniform(0., 1e-10), uniform(0., 1e-10)))
    result = f.integrate_groups(2., (0., 1e308))
    assert result.probabilities == pytest.approx((1.,))
