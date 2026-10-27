"""State-driven history rejections against an independent smooth reference."""

import pytest

from tools.assess_exponential_state_audit import smooth_ramp


@pytest.mark.parametrize('scheme', ['ftm2', 'ftm-bdf2'])
def test_smooth_insertion_exercises_audited_state_failures_and_history_rollback(scheme):
    record = smooth_ramp(scheme)
    assert record['oracle_failed_state_checks'] > 0
    assert record['history_pass_oracle_fail'] == 0
    assert record['cheap_history_rejections'] > 0
    assert record['max_endpoint_relative_power_error'] < 1e-4
    assert record['final_relative_precursor_error'] < 1e-4
    assert record['replay_max_relative_power_difference'] < 2e-9
    assert record['replay_max_relative_precursor_difference'] < 2e-9
    assert record['implicit_solves'] == record['recorded_trial_solves']
    assert record['callbacks'] == record['accepted_steps']
