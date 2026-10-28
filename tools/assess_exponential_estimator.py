"""Calibrate the two-solve physical-history estimator against step doubling.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m tools.assess_exponential_estimator

All methods use actual FV endpoint solves and independent physical kinetics
references. Audit mode preserves the step-doubling trajectory and records
both defects on each eligible trial. Matched output, integral and precursor
errors determine which work counts qualify; equal local tolerance does not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy

from tools.assess_exponential_adaptive import ROOT, case
from tools.assess_exponential_damping import CASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', choices=list(CASES), default=list(CASES))
    parser.add_argument('--rtols', nargs='+', type=float, default=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    parser.add_argument('--output-rtol', type=float, default=1e-4)
    parser.add_argument('--output', type=Path, default=ROOT / (
        'benchmark-results/exponential-second-order/chunk6-estimator.json'))
    args = parser.parse_args()
    if not np.isfinite(args.output_rtol) or args.output_rtol <= 0:
        parser.error('--output-rtol must be finite and positive')
    if any(not np.isfinite(tol) or tol <= 0 for tol in args.rtols):
        parser.error('--rtols must be finite and positive')
    sources = ['garnet/timescheme.py', 'garnet/transient.py', 'garnet/model.py',
               'garnet/_model_support.py', 'garnet/_exponential.py',
               'tools/assess_exponential_adaptive.py', 'tools/assess_exponential_estimator.py',
               'tools/assess_exponential_damping.py', 'tests/verification/test_exponential_estimator.py',
               'tests/verification/test_exponential_adaptive.py']
    report = {
        'date': '2026-10-26',
        'scope': 'Two-half-step physical-history estimator; CPU neutron/precursor integration',
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'reference': 'Independent physical two-mode kinetics eigendecomposition and power-integral quadrature',
        'initial_dt_s': 0.01, 'maximum_dt_s': 0.02, 'minimum_dt_s': 1e-11,
        'state_rtols': args.rtols, 'output_rtol': args.output_rtol,
        'history_prediction': 'Quadratic physical-state extrapolation from previous accepted midpoint, macro start and current midpoint; actual unequal widths; no Richardson divisor',
        'safeguards': 'Step doubling for bootstrap/events, width ratios outside [0.25,2], and BE safeguards; unexpected trial fallback rolls back and replays oracle, charging discarded work',
        'audit': 'Same-trial candidate history and reference full/fine defects; history-audit retains reference decisions and output grid',
        'error_sampling': 'Own endpoints and common 8001-point public power_at grid; endpoint-trapezoid integral versus independent quadrature; final precursors',
        'budget_filter': 'Power, power integral and final precursor errors within budget; nonnegative power and no wrong-direction increments',
        'work': 'Every full/half endpoint solve, discarded rescue, rejection, frequency correction and Krylov iteration; no runtime speedup claim',
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        'cases': {},
    }
    methods = ('ftm2-lagged', 'ftm-bdf2-lagged', 'ftm-bdf2-zero', 'adaptive-bdf')
    for name in args.cases:
        studies = {mode: case(name, args.rtols, output_rtol=args.output_rtol, estimator=mode)
                   for mode in ('step-doubling', 'history', 'history-audit')}
        budgets = {}
        for budget in (0.01, 0.001):
            choices = {}
            for mode in ('step-doubling', 'history'):
                choices[mode] = {}
                for method in methods:
                    passing = [row for row in studies[mode]['records']
                               if row['configuration']['method'] == method
                               and row['max_relative_power_error'] <= budget
                               and row['relative_power_integral_error'] <= budget
                               and row['final_relative_precursor_error'] <= budget
                               and row['minimum_power'] >= 0 and row['wrong_direction_increments'] == 0]
                    choices[mode][method] = min(passing, key=lambda row: row['implicit_solves']) if passing else None
            budgets[str(budget)] = choices
        # Audit mode must not silently change the reference accuracy or work.
        for reference, audit in zip(studies['step-doubling']['records'], studies['history-audit']['records']):
            assert all(audit[key] == value for key, value in reference.items() if key != 'configuration')
        report['cases'][name] = {'studies': studies,
            'fewest_tested_solves_at_power_integral_precursor_budget': budgets}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
