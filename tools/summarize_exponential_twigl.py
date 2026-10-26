"""Combine TWIGL state-tolerance and output-width sweeps at physical budgets.

    python3 -m tools.summarize_exponential_twigl

Run assess_exponential_twigl and assess_exponential_twigl_widths first.
Both constituent and Krylov work rankings retain the complete qualifying
configuration and its input report. Single-run timings are not ranked.
"""

from __future__ import annotations

import hashlib
import json

from tools.assess_exponential_twigl import METHODS, ROOT, error_gate


def main():
    folder = ROOT / 'benchmark-results/exponential-second-order'
    primary_path, widths_path = folder/'chunk7-twigl.json',folder/'chunk7-twigl-widths.json'
    primary, widths = json.loads(primary_path.read_text()),json.loads(widths_path.read_text())
    if set(primary['cases'])!=set(widths['cases']) or set(primary.get('spatial_comparisons',{}))!={'step','ramp'}:
        raise RuntimeError('complete both sweeps and spatial references before summarizing')
    summary = dict(date='2026-10-26',scope='Combined matched physical error/work TWIGL gate',
        input_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (primary_path,widths_path)},
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ('tools/summarize_exponential_twigl.py','tools/assess_exponential_twigl.py')},
        primary_configurations=sum(len(c['records']) for c in primary['cases'].values()),
        shared_mesh_controls=sum(r['configuration']['fixed_widths'] for c in primary['cases'].values() for r in c['controls']),
        spatial_audit_controls=sum('audit' in r for c in primary['cases'].values() for r in c['controls']),
        width_configurations=sum(len(c['records']) for c in widths['cases'].values()),cases={})
    for name,case in primary['cases'].items():
        assert case['k0']==widths['cases'][name]['k0']
        records = []
        for label,batch in (('primary',case['records']),('width-control',widths['cases'][name]['records'])):
            for row in batch:
                copied = dict(row,source=label)
                copied['configuration'] = dict(row['configuration'])
                copied['configuration'].setdefault('max_dt',.02)
                records.append(copied)
        controls = [dict(row,source='shared-mesh' if row['configuration']['fixed_widths']
                         else 'estimator-audit') for row in case['controls']]
        choices = {}
        for budget in (.01,.001):
            choices[str(budget)] = {}
            for work in ('implicit_solves','krylov_iterations'):
                choices[str(budget)][work] = {method:min((r for r in records
                    if r['configuration']['method']==method and error_gate(r,budget)),
                    key=lambda r:r[work],default=None) for method in METHODS}
        overall = {str(budget):{
            work:min((r for r in records+controls if r['configuration']['method'].startswith('ftm')
                      and error_gate(r,budget)),key=lambda r:r[work],default=None)
            for work in ('implicit_solves','krylov_iterations')} for budget in (.01,.001)}
        summary['cases'][name] = dict(k0=case['k0'],fewest_tested_work_at_budget=choices,
                                      best_ftm_including_shared_mesh_at_budget=overall)
    audit = [r['audit'] for c in primary['cases'].values() for r in c['controls'] if 'audit' in r]
    summary['spatial_audits'] = {key:sum(r[key] for r in audit) for key in
        ('paired_trials','oracle_failed_state_checks','history_pass_oracle_fail','oracle_pass_history_fail')}
    summary['spatial_comparisons'] = primary['spatial_comparisons']
    (folder/'chunk7-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    for name,case in summary['cases'].items():
        print(name,{m:None if r is None else (r['implicit_solves'],r['configuration']['rtol'],r['configuration']['max_dt'])
            for m,r in case['fewest_tested_work_at_budget']['0.001']['implicit_solves'].items()})


if __name__=='__main__':
    main()
