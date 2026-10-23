"""Physics identities and deliberately inconclusive branch assessments."""
import json
import math

import numpy as np
import pytest

from garnet.branch_validation import (
    assess_rod_worth, centered_temperature_coefficient, coefficient_convergence,
    compare_reconstruction_chain, interval_rod_worth, normalized_power_rms,
    reactivity, reactivity_change,
)


def test_pcm_is_reactivity_not_delta_k():
    result = reactivity_change(.9, 1.1, k_from_standard_error=1e-5,
                               k_to_standard_error=2e-5)
    assert result['value_pcm'] == pytest.approx(1e5 * (1 / .9 - 1 / 1.1))
    assert result['value_pcm'] != pytest.approx(1e5 * .2)
    assert result['standard_error_pcm'] == pytest.approx(1e5 * math.hypot(1e-5 / .9**2, 2e-5 / 1.1**2))
    assert reactivity_change(1.1, .9)['value_pcm'] < 0


def test_actual_temperature_spacing_and_error():
    result = centered_temperature_coefficient(1., .99, 500., 700.,
        k_lower_standard_error=1e-5, k_upper_standard_error=2e-5)
    assert result['temperature_midpoint_K'] == 600.
    assert result['value_pcm_per_K'] == pytest.approx(1e5 * (1 - 1 / .99) / 200)
    assert result['standard_error_pcm_per_K'] == pytest.approx(1e5 * math.hypot(1e-5, 2e-5 / .99**2) / 200)


def coeff(value, error=0.):
    return {'value_pcm_per_K': value, 'standard_error_pcm_per_K': error}


def test_step_convergence_respects_precision_and_weak_signal():
    assert coefficient_convergence(coeff(-4), coeff(-4.1))['status'] == 'pass'
    assert coefficient_convergence(coeff(-3), coeff(-4))['status'] == 'fail'
    assert coefficient_convergence(coeff(-4, .2), coeff(-4, .2))['status'] == 'inconclusive'
    assert coefficient_convergence(coeff(0), coeff(0))['status'] == 'inconclusive'
    assert coefficient_convergence(coeff(.001), coeff(.001, .01))['status'] == 'inconclusive'


def test_power_uses_same_ordered_bins_and_weighted_mean():
    assert normalized_power_rms([1, 2], [2, 4])['rms_fraction'] == pytest.approx(0)
    result = normalized_power_rms([1, 2], [2, 1], weights=[1, 3])
    assert result['rms_fraction'] == pytest.approx(np.sqrt(np.average((np.array([1, 2]) / 1.75 - np.array([2, 1]) / 1.25)**2, weights=[1, 3])))
    assert result['status'] == 'fail'


def test_three_comparisons_gate_only_interpolation():
    result = compare_reconstruction_chain(1.10001, 1.1, 1., k_ce_standard_error=1e-5,
        power_interpolated=[1, 2], power_exact=[2, 4], power_ce=[1, 1])
    assert result['status'] == 'pass'
    assert result['comparisons']['exact_vs_ce']['value_pcm'] > 9000
    assert result['comparisons']['exact_vs_ce']['standard_error_pcm'] == pytest.approx(1)
    assert 'target' not in result['comparisons']['exact_vs_ce']
    assert result['comparisons']['exact_vs_ce']['power']['status'] == 'descriptive'
    json.dumps(result, allow_nan=False)
    assert compare_reconstruction_chain(1.11, 1.1, 1.1)['status'] == 'fail'


def test_interpolation_power_can_fail_even_if_k_passes():
    result = compare_reconstruction_chain(1, 1, 1, power_interpolated=[1, 2],
        power_exact=[2, 1], power_ce=[2, 1])
    assert result['status'] == 'fail'


def test_rod_worth_sign_spacing_and_error():
    result = interval_rod_worth(1., .99, 20., 25., k_start_standard_error=1e-5)
    assert result['value_pcm_per_cm'] == pytest.approx(reactivity(.99) * 1e5 / 5)
    assert result['value_pcm_per_cm'] < 0
    assert result['standard_error_pcm_per_cm'] == pytest.approx(.2)


def worth(a, b, value=-10., error=0.):
    return {'interval_cm': [a, b], 'value_pcm_per_cm': value,
            'standard_error_pcm_per_cm': error}


def test_rod_coverage_and_peak_not_inferred_from_sparse_checks():
    records = [worth(25, 50), worth(50, 75)]
    kwargs = dict(required_intervals=[[25, 50], [50, 75]], reference_peak=10.)
    assert assess_rod_worth(records, records, **kwargs)['status'] == 'inconclusive'
    result = assess_rod_worth(records, records, peak_is_resolved=True, **kwargs)
    assert result['status'] == 'pass'
    result = assess_rod_worth(records, records, required_intervals=[[0, 25], [25, 50]],
                              reference_peak=10., peak_is_resolved=True)
    assert result['status'] == 'inconclusive'
    assert result['missing_intervals_cm'] == [[0., 25.]]
    json.dumps(result, allow_nan=False)


def test_rod_uncertainty_and_failure():
    kwargs = dict(required_intervals=[[0, 10]], reference_peak=10., peak_is_resolved=True)
    assert assess_rod_worth([worth(0, 10, -9)], [worth(0, 10)], **kwargs)['status'] == 'fail'
    assert assess_rod_worth([worth(0, 10, error=1.)], [worth(0, 10, error=1.)], **kwargs)['status'] == 'inconclusive'
    assert assess_rod_worth([worth(0, 10)], [worth(0, 10)], peak_standard_error=10., **kwargs)['status'] == 'inconclusive'


@pytest.mark.parametrize('k', [0, -1, np.inf, np.nan])
def test_bad_eigenvalues(k):
    with pytest.raises(ValueError):
        reactivity(k)


@pytest.mark.parametrize('predicted,reference,weights', [
    ([0, 0], [1, 2], None), ([1, -1], [1, 2], None),
    ([1, 2], [1], None), ([1, 2], [1, 2], [0, 1]),
    ([1, 2], [1, 2], [1, np.nan]), ([], [], None),
    ([1, np.nan], [1, 2], None),
])
def test_bad_powers(predicted, reference, weights):
    with pytest.raises(ValueError):
        normalized_power_rms(predicted, reference, weights=weights)


def test_bad_inputs():
    with pytest.raises(ValueError):
        centered_temperature_coefficient(1, 1, 600, 500)
    with pytest.raises(ValueError):
        interval_rod_worth(1, 1, 5, 5)
    with pytest.raises(ValueError):
        reactivity_change(1, 1, k_from_standard_error=-1)
    with pytest.raises(ValueError):
        compare_reconstruction_chain(1, 1, 1, power_ce=[1])
    with pytest.raises(ValueError):
        coefficient_convergence(coeff(1), coeff(1), relative_target=-1)
    with pytest.raises(ValueError):
        assess_rod_worth([], [], required_intervals=[])
    with pytest.raises(ValueError):
        assess_rod_worth([worth(0, 1), worth(0, 1)], [], required_intervals=[[0, 1]])


def test_step_convergence_checks_supplied_temperature_geometry():
    full = centered_temperature_coefficient(1, .99, 500, 700)
    half = centered_temperature_coefficient(1, .99, 550, 650)
    assert coefficient_convergence(full, half)['status'] == 'fail'
    with pytest.raises(ValueError, match='midpoint'):
        coefficient_convergence(full, centered_temperature_coefficient(1, .99, 560, 660))
    with pytest.raises(ValueError, match='separation'):
        coefficient_convergence(full, centered_temperature_coefficient(1, .99, 575, 625))
    with pytest.raises(ValueError, match='both coefficients'):
        coefficient_convergence(full, coeff(1))
