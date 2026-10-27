"""Compare physical-state adaptive FTM with existing adaptive BDF kinetics.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m tools.assess_exponential_adaptive

All methods use the public one-cell diffusion API. References are independent
physical kinetics matrices. Work includes full/half-step trials, rejections,
frequency refinements and Krylov iterations. No runtime speedup is inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.integrate import quad, trapezoid

from garnet import Kinetics, Material, Model
from tools.assess_exponential_damping import BETA, DECAY, GENERATION, CASES, problem


ROOT = Path(__file__).resolve().parents[1]


def case(name, tolerances, *, output_rtol=None, estimator="step-doubling"):
    rho, duration, event = CASES[name]
    _, _, reference = problem(name)
    fuel = Material(name='fuel', diffusion=[1.3], sigma_a=[0.03], nu_sigma_f=[0.035])
    changed = fuel.replace(sigma_a=[0.03 * (1 - rho)])
    model = (Model((1, 1, 1), (1, 1, 1)).fill(fuel).set_boundary('reflective')
             .set_kinetics(Kinetics([2.2e5], [BETA], [DECAY])))
    steady = model.steady(device='cpu', tol_k=1e-12, tol_source=1e-11)
    sample = np.linspace(0, duration, 8001)
    expected = reference(sample)[:, 0]
    expected_integral = quad(lambda t: float(reference(t)[0]), 0, duration,
                             points=[event] if event else None, epsabs=1e-11, epsrel=1e-12)[0]
    records = []
    for method in ('ftm2-lagged', 'ftm-bdf2-lagged', 'ftm-bdf2-zero', 'adaptive-bdf'):
        for tolerance in tolerances:
            controls = {'rtol': tolerance, 'min_dt': 1e-11, 'max_dt': 0.02,
                        'max_rejections': 40}
            if method == 'adaptive-bdf':
                options = {'time_scheme': 'bdf5', 'adaptive_bdf': dict(
                    controls, automatic_order=True)}
            else:
                if estimator != 'step-doubling':
                    controls['estimator'] = estimator
                if output_rtol is not None:
                    controls['output_rtol'] = output_rtol
                options = {'time_scheme': 'ftm2' if method.startswith('ftm2-') else 'ftm-bdf2',
                           'adaptive_ftm': controls}
                if method.endswith('-zero'):
                    options['ftm_kwargs'] = {'frequency': 0.0}
            result = model.transient(duration, 0.01, device='cpu', initial_steady=steady,
                event_times=[event] if event else None,
                materials_at=lambda t: {'fuel': changed} if t > event else {},
                step_solver='monolithic', tol_step=1e-10,
                multigroup_kwargs={'rtol': 1e-11}, **options)
            endpoint_error = float(np.max(np.abs(result.power / reference(result.times)[:, 0] - 1)))
            output_error = float(np.max(np.abs(result.power_at(sample) / expected - 1)))
            final_precursor = float(np.max(np.abs(result.precursors_numpy.ravel()
                                        / GENERATION / reference(duration)[1:] - 1)))
            differences = np.diff(result.power) * np.sign(rho)
            wrong_direction = int(np.count_nonzero(differences < -1e-12 * np.maximum(
                np.abs(result.power[:-1]), np.abs(result.power[1:]))))
            attempts = result.ftm_trial_diagnostics + result.rejected_ftm_trial_diagnostics
            row = {
                'configuration': {'method': method, 'rtol': tolerance},
                'accepted_steps': result.n_steps, 'rejected_steps': result.rejected_steps,
                'implicit_solves': sum(result.feedback_iterations) + sum(result.rejected_feedback_iterations),
                'krylov_iterations': result.total_inner_iterations,
                'fgmres_iterations': sum(result.step_iterations) + sum(result.rejected_step_iterations),
                'operator_rebuilds': result.operator_rebuilds,
                'fallback_trial_steps': sum(bool(trial['fallback_reason'])
                                          for attempt in attempts for trial in attempt),
                'maximum_accepted_local_error': max(result.local_errors, default=0),
                'maximum_accepted_error_by_block': (np.max(result.local_error_blocks, axis=0).tolist()
                                                   if result.local_error_blocks else None),
                'max_endpoint_relative_power_error': endpoint_error,
                'max_common_grid_relative_power_error': output_error,
                'max_relative_power_error': max(endpoint_error, output_error),
                'relative_power_integral_error': float(abs(
                    trapezoid(result.power, result.times) / expected_integral - 1)),
                'final_relative_precursor_error': final_precursor,
                'minimum_power': float(np.min(result.power)),
                'wrong_direction_increments': wrong_direction,
                'minimum_accepted_width_s': float(min(result.step_widths)),
                'maximum_accepted_width_s': float(max(result.step_widths)),
            }
            if output_rtol is not None and method != 'adaptive-bdf':
                row['configuration']['output_rtol'] = output_rtol
                row['maximum_accepted_output_error_by_block'] = (
                    np.max(result.output_error_blocks, axis=0).tolist())
                row['output_driven_rejections'] = sum(
                    max(state) <= 1 < max(output) for state, output in zip(
                        result.rejected_error_blocks, result.rejected_output_error_blocks))
            if estimator != 'step-doubling' and method != 'adaptive-bdf':
                row['configuration']['estimator'] = estimator
                diagnostics = result.ftm_estimator_diagnostics + result.rejected_ftm_estimator_diagnostics
                reference_blocks = result.local_error_blocks + result.rejected_error_blocks
                row['history_attempts'] = sum(d['method'] == 'history' for d in diagnostics)
                row['oracle_attempts'] = sum(d['method'] == 'step-doubling' for d in diagnostics)
                row['estimator_fallback_reasons'] = {reason: sum(d['reason'] == reason for d in diagnostics)
                    for reason in sorted({d['reason'] for d in diagnostics if d['reason']})}
                paired = [(max(d['history_error_blocks']), max(blocks))
                          for d, blocks in zip(diagnostics, reference_blocks)
                          if d['history_error_blocks'] is not None]
                if estimator == 'history-audit':
                    meaningful = [(history / oracle) for history, oracle in paired if oracle >= 1e-3]
                    row['audit'] = {'paired_attempts': len(paired),
                        'history_pass_oracle_fail': sum(history <= 1 < oracle for history, oracle in paired),
                        'oracle_pass_history_fail': sum(oracle <= 1 < history for history, oracle in paired),
                        'minimum_ratio_where_oracle_ge_1e-3': min(meaningful, default=None),
                        'maximum_ratio_where_oracle_ge_1e-3': max(meaningful, default=None)}
            records.append(row)
            print(name, method, tolerance,
                  f"error={100 * row['max_relative_power_error']:.5f}% solves={row['implicit_solves']}",
                  flush=True)
    budgets = {}
    for budget in (0.01, 0.001):
        choices = {}
        for method in ('ftm2-lagged', 'ftm-bdf2-lagged', 'ftm-bdf2-zero', 'adaptive-bdf'):
            qualifying = [row for row in records if row['configuration']['method'] == method
                          and row['max_relative_power_error'] <= budget
                          and row['final_relative_precursor_error'] <= budget
                          and row['minimum_power'] >= 0 and row['wrong_direction_increments'] == 0]
            choices[method] = min(qualifying, key=lambda row: row['implicit_solves']) if qualifying else None
        budgets[str(budget)] = choices
    return {'reactivity': rho, 'duration_s': duration, 'event_s': event,
            'records': records, 'fewest_tested_solves_at_budget': budgets}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', choices=list(CASES), default=list(CASES))
    parser.add_argument('--rtols', nargs='+', type=float, default=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    parser.add_argument('--output', type=Path, default=ROOT / (
        'benchmark-results/exponential-second-order/chunk4-adaptive.json'))
    args = parser.parse_args()
    sources = ['garnet/timescheme.py', 'garnet/transient.py', 'garnet/_model_support.py',
               'garnet/_exponential.py', 'tools/assess_exponential_adaptive.py',
               'tools/assess_exponential_damping.py']
    report = {
        'date': '2026-10-26',
        'scope': 'Physical-state adaptive neutron/precursor FTM; no thermal transform or GPU timing',
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'reference': 'Independent physical two-mode kinetics eigendecomposition',
        'initial_dt_s': 0.01, 'maximum_dt_s': 0.02, 'minimum_dt_s': 1e-11,
        'ftm_error_norm': 'Maximum per-field active-volume weighted RMS of physical full/fine defects, scaled by atol + rtol*max(abs(start),abs(fine)); no Richardson divisor',
        'bdf_error_norm': 'Unchanged legacy concatenated predictor/corrector ratio of norms; equal rtol does not imply matched physical accuracy',
        'error_sampling': 'Maximum of own endpoints and common 8001-point linear interpolation grid; final precursor error also gated',
        'budget_filter': 'Power and final precursor errors within budget; nonnegative power; no wrong-direction macro increments',
        'work_comparison': 'Every FV endpoint solve, including discarded full steps and rejected full/half attempts; no runtime speedup claim',
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        'cases': {},
    }
    for name in args.cases:
        report['cases'][name] = case(name, args.rtols)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
