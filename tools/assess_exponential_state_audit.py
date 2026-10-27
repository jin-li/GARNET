"""Audit physical-history defects without the output width restriction.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m tools.assess_exponential_state_audit

This complements assess_exponential_estimator's output-aware work comparison.
Reference decisions stay with step doubling, while candidate and reference
state defects are recorded on the same trials, including rejected attempts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.integrate import solve_ivp

from garnet import Kinetics, Material, Model
from tools.assess_exponential_adaptive import ROOT
from tools.assess_exponential_damping import BETA, DECAY, GENERATION, CASES, problem


def smooth_ramp(scheme):
    """Exercise state-driven history rejection on a resolved smooth insertion."""
    fuel = Material(name='fuel', diffusion=[1.3], sigma_a=[0.03], nu_sigma_f=[0.035])
    model = (Model((1, 1, 1), (1, 1, 1)).fill(fuel).set_boundary('reflective')
             .set_kinetics(Kinetics([2.2e5], [BETA], [DECAY])))
    rho = lambda t: 0.001 * (1 + np.tanh((t - 0.03) / 0.001))
    initial = np.array([1., BETA / (GENERATION * DECAY)])
    def rhs(t, y):
        return [(rho(t) - BETA) / GENERATION * y[0] + DECAY * y[1],
                BETA / GENERATION * y[0] - DECAY * y[1]]
    reference = solve_ivp(rhs, (0., 0.06), initial, method='Radau',
                          rtol=2e-12, atol=1e-13, dense_output=True)
    assert reference.success
    options = dict(device='cpu', time_scheme=scheme, step_solver='monolithic',
        materials_at=lambda t: {'fuel': fuel.replace(sigma_a=[0.03 * (1 - rho(t))])},
        tol_step=1e-11, multigroup_kwargs={'rtol': 1e-12})
    controls = dict(rtol=1e-5, min_dt=1e-11, max_dt=0.02)
    audit = model.transient(0.06, 0.001, adaptive_ftm=dict(
        controls, estimator='history-audit'), **options)
    observed = []
    cheap = model.transient(0.06, 0.001, adaptive_ftm=dict(
        controls, estimator='history'),
        on_step=lambda t, phi, power: observed.append(t), **options)
    replay = model.transient(0.06, np.repeat(cheap.step_widths / 2, 2), **options)
    paired = [(max(d['history_error_blocks']), max(blocks)) for d, blocks in zip(
        audit.ftm_estimator_diagnostics + audit.rejected_ftm_estimator_diagnostics,
        audit.local_error_blocks + audit.rejected_error_blocks) if d['history_error_blocks'] is not None]
    attempts = cheap.ftm_trial_diagnostics + cheap.rejected_ftm_trial_diagnostics
    return {'scheme': scheme, 'rtol': 1e-5, 'output_rtol': None,
        'reactivity': '0.001*(1+tanh((t-0.03)/0.001))',
        'paired_trials': len(paired), 'oracle_failed_state_checks': sum(oracle > 1 for history, oracle in paired),
        'history_pass_oracle_fail': sum(history <= 1 < oracle for history, oracle in paired),
        'cheap_history_rejections': sum(d['method'] == 'history' for d in cheap.rejected_ftm_estimator_diagnostics),
        'accepted_steps': cheap.n_steps, 'rejected_steps': cheap.rejected_steps,
        'max_endpoint_relative_power_error': float(np.max(np.abs(
            cheap.power / reference.sol(cheap.times)[0] - 1))),
        'final_relative_precursor_error': float(np.max(np.abs(
            cheap.precursors_numpy.ravel() / GENERATION / reference.sol(0.06)[1:] - 1))),
        'replay_max_relative_power_difference': float(np.max(np.abs(cheap.power / replay.power[::2] - 1))),
        'replay_max_relative_precursor_difference': float(np.max(np.abs(
            cheap.precursors_numpy / replay.precursors_numpy - 1))),
        'implicit_solves': sum(cheap.feedback_iterations) + sum(cheap.rejected_feedback_iterations),
        'recorded_trial_solves': sum(len(trials) for trials in attempts),
        'callbacks': len(observed)}


def main():
    sources = ['garnet/timescheme.py', 'garnet/transient.py',
               'tools/assess_exponential_state_audit.py', 'tools/assess_exponential_damping.py']
    report = {'date': '2026-10-26', 'scope': 'State-only same-trial audit; no output width restriction',
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        'reference_decisions': 'Step doubling; unchanged archived checkpoint-four counts checked at identical controls',
        'audit_filter': 'Only safe second-order history with actual half-width ratio in [0.25,2]; startup/events/safeguards excluded',
        'reference': 'Independent physical two-mode kinetics matrix', 'records': []}
    previous = json.loads((ROOT / 'benchmark-results/exponential-second-order/chunk4-adaptive.json').read_text())
    for name, (rho, duration, event) in CASES.items():
        _, _, reference = problem(name)
        fuel = Material(name='fuel', diffusion=[1.3], sigma_a=[0.03], nu_sigma_f=[0.035])
        changed = fuel.replace(sigma_a=[0.03 * (1 - rho)])
        model = (Model((1, 1, 1), (1, 1, 1)).fill(fuel).set_boundary('reflective')
                 .set_kinetics(Kinetics([2.2e5], [BETA], [DECAY])))
        steady = model.steady(device='cpu', tol_k=1e-12, tol_source=1e-11)
        for scheme in ('ftm2', 'ftm-bdf2'):
            for tolerance in (1e-3, 1e-5, 1e-7):
                run = model.transient(duration, 0.01, device='cpu', initial_steady=steady,
                    time_scheme=scheme, step_solver='monolithic',
                    adaptive_ftm=dict(estimator='history-audit', rtol=tolerance,
                        min_dt=1e-11, max_dt=0.02, max_rejections=40),
                    event_times=[event] if event else None,
                    materials_at=lambda t: {'fuel': changed} if t > event else {},
                    tol_step=1e-10, multigroup_kwargs={'rtol': 1e-11})
                diagnostics = run.ftm_estimator_diagnostics + run.rejected_ftm_estimator_diagnostics
                blocks = run.local_error_blocks + run.rejected_error_blocks
                paired = [(max(d['history_error_blocks']), max(b)) for d, b in zip(diagnostics, blocks)
                          if d['history_error_blocks'] is not None]
                ratios = [history / oracle for history, oracle in paired if oracle >= 1e-3]
                solves = sum(run.feedback_iterations) + sum(run.rejected_feedback_iterations)
                archived = next(row for row in previous['cases'][name]['records']
                    if row['configuration'] == {'method': scheme + '-lagged', 'rtol': tolerance})
                assert (run.n_steps, run.rejected_steps, solves) == (
                    archived['accepted_steps'], archived['rejected_steps'], archived['implicit_solves'])
                endpoint_error = float(np.max(np.abs(run.power / reference(run.times)[:, 0] - 1)))
                assert endpoint_error == archived['max_endpoint_relative_power_error']
                row = {'case': name, 'scheme': scheme, 'rtol': tolerance,
                    'accepted_steps': run.n_steps, 'rejected_steps': run.rejected_steps,
                    'implicit_solves': solves, 'paired_trials': len(paired),
                    'oracle_failed_state_checks': sum(oracle > 1 for history, oracle in paired),
                    'history_pass_oracle_fail': sum(history <= 1 < oracle for history, oracle in paired),
                    'oracle_pass_history_fail': sum(oracle <= 1 < history for history, oracle in paired),
                    'minimum_ratio_where_oracle_ge_1e-3': min(ratios, default=None),
                    'maximum_ratio_where_oracle_ge_1e-3': max(ratios, default=None),
                    'max_endpoint_relative_power_error': endpoint_error,
                    'archived_reference_counts_and_endpoint_error_match': True}
                report['records'].append(row)
                print(name, scheme, tolerance, 'pairs',len(paired),
                    'oracle failures',row['oracle_failed_state_checks'],
                    'history misses',row['history_pass_oracle_fail'],flush=True)
    report['smooth_ramp_verification'] = [smooth_ramp(scheme) for scheme in ('ftm2', 'ftm-bdf2')]
    output = ROOT / 'benchmark-results/exponential-second-order/chunk6-state-audit.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
