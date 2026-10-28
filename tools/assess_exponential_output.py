"""Compare linear-output-aware FTM with the state-only and adaptive BDF controls.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m tools.assess_exponential_output

The independent kinetics reference and all endpoint work use the fourth
checkpoint's harness. The output guard adds no spatial solves per attempt.
Power, its time integral and final precursors are measured independently;
local output tolerances are not claimed to bound accumulated physical error.
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
        'benchmark-results/exponential-second-order/chunk5-output.json'))
    args = parser.parse_args()
    if not np.isfinite(args.output_rtol) or args.output_rtol <= 0:
        parser.error('--output-rtol must be finite and positive')
    if any(not np.isfinite(tol) or tol <= 0 for tol in args.rtols):
        parser.error('--rtols must be finite and positive')
    sources = ['garnet/timescheme.py', 'garnet/transient.py', 'garnet/model.py',
               'garnet/_model_support.py', 'garnet/_exponential.py',
               'tools/assess_exponential_adaptive.py', 'tools/assess_exponential_output.py',
               'tools/assess_exponential_damping.py', 'tests/verification/test_exponential_output.py']
    report = {
        'date': '2026-10-26',
        'scope': 'Opt-in physical midpoint/chord width restriction; neutron/precursor CPU integration',
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'reference': 'Independent physical two-mode kinetics eigendecomposition; independent quadrature for power integral',
        'initial_dt_s': 0.01, 'maximum_dt_s': 0.02, 'minimum_dt_s': 1e-11,
        'state_rtols': args.rtols,
        'output_rtol': args.output_rtol,
        'output_norm': 'Midpoint minus macro-endpoint chord, per-field active-volume weighted RMS scaled by output_atol + output_rtol*abs(midpoint); scalar relative power appended',
        'width_control': 'Minimum of state-order and order-one controller proposals; chord defect is O(h**2)',
        'error_sampling': 'Own endpoints and common 8001-point grid via public power_at; endpoint-trapezoid power integral versus independent reference quadrature',
        'work': 'Every full/half endpoint solve, rejection, corrector and Krylov iteration; no runtime speedup claim',
        'bdf_control': 'Existing adaptive BDF1--5 and its norm unchanged; appears in both runs to make each comparison self-contained',
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        'cases': {},
    }
    for name in args.cases:
        baseline = case(name, args.rtols)
        guarded = case(name, args.rtols, output_rtol=args.output_rtol)
        budgets = {}
        for budget in (0.01, 0.001):
            choices = {}
            for family, study in (('state-only', baseline), ('output-aware', guarded)):
                choices[family] = {}
                for method in ('ftm2-lagged', 'ftm-bdf2-lagged', 'ftm-bdf2-zero', 'adaptive-bdf'):
                    passing = [row for row in study['records']
                               if row['configuration']['method'] == method
                               and row['max_relative_power_error'] <= budget
                               and row['relative_power_integral_error'] <= budget
                               and row['final_relative_precursor_error'] <= budget
                               and row['minimum_power'] >= 0 and row['wrong_direction_increments'] == 0]
                    choices[family][method] = min(passing, key=lambda row: row['implicit_solves']) if passing else None
            budgets[str(budget)] = choices
        report['cases'][name] = {'state_only': baseline, 'output_aware': guarded,
            'fewest_tested_solves_at_power_integral_precursor_budget': budgets}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
