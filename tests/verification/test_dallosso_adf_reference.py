"""Dall'Osso figure interpretation and same-grid reference-factor checks."""
import pytest

from examples.dallosso_1d_reference import analytical_k, reference_factor_comparison, run


def test_heterogeneous_dallosso_interpretation_converges():
    exact = analytical_k()
    assert exact == pytest.approx(1.03263376024, abs=1e-10)
    results = [run(n) for n in (1, 2, 4)]
    assert all(r['relative_eigen_residual'] < 1e-8 for r in results)
    differences = [results[i+1]['k'] - results[i]['k'] for i in (0, 1)]
    assert 3.6 < differences[0] / differences[1] < 4.3
    assert abs(results[-1]['k'] - exact) < abs(results[0]['k'] - exact) / 12
    # This is the interpreted no-transverse-leakage problem, not the paper's
    # reported eigenvalue. The discrepancy remains under investigation.
    assert results[-1]['k'] == pytest.approx(1.03263123, abs=3e-8)


@pytest.mark.parametrize('n', [1, 2])
def test_two_group_reference_factor_equivalence(n):
    result = reference_factor_comparison(n)
    baseline, corrected = result['runs']
    assert abs(baseline['delta_k_pcm']) > 100
    assert baseline['max_region_flux_relative_error'] > .03
    assert abs(corrected['delta_k_pcm']) < 1e-4
    assert corrected['max_region_flux_relative_error'] < 1e-8
    assert corrected['max_interface_current_absolute_error'] < 1e-8
