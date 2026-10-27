"""Check output-width caps for the TWIGL adaptive BE/BDF comparators.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m tools.assess_exponential_twigl_widths

The primary spatial sweep caps widths at 0.02 s. This bounded supplement
tests BDF caps of 0.01/0.005 s at rtol=1e-3/1e-4 and BE at 0.005 s, rtol=1e-3.
It changes no controller error norm or interpolation formula.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.assess_exponential_twigl import ROOT, Reference, TwiglSystem, measure


def run_width(system,reference,kind,method,tolerance,maximum):
    snapshots = [system.initial[:2*system.count].reshape(2,system.count)]
    result = system.model.transient(.5,.005,device='cpu',initial_steady=system.steady,
        at=system.problem.problem_at,event_times=[.2] if kind=='ramp' else None,
        time_scheme='bdf5' if method=='adaptive-bdf' else 'bdf1',
        adaptive_bdf=dict(rtol=tolerance,min_dt=1e-11,max_dt=maximum,max_rejections=40,
                          automatic_order=method=='adaptive-bdf'),
        on_step=lambda t,phi,power:snapshots.append(np.asarray(phi).reshape(2,system.count).copy()),
        step_solver='monolithic',tol_step=1e-10,multigroup_kwargs=dict(rtol=1e-11))
    row = measure(system,reference,result.times,result.power,np.asarray(snapshots),result.precursors_numpy)
    row.update(configuration=dict(method=method,rtol=tolerance,max_dt=maximum),
        accepted_steps=result.n_steps,rejected_steps=result.rejected_steps,
        implicit_solves=sum(result.feedback_iterations)+sum(result.rejected_feedback_iterations),
        krylov_iterations=result.total_inner_iterations,
        fgmres_iterations=sum(result.step_iterations)+sum(result.rejected_step_iterations),
        operator_rebuilds=result.operator_rebuilds,diagnostic_solve_seconds=result.solve_seconds,
        final_power=float(result.power[-1]))
    print(kind,system.problem.grid.shape[0],method,tolerance,maximum,
        f"power={100*row['maximum_relative_power_error']:.5f}% solves={row['implicit_solves']}",flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refinements',nargs='+',type=int,default=[2,4])
    parser.add_argument('--kinds',nargs='+',choices=['step','ramp'],default=['step','ramp'])
    parser.add_argument('--output',type=Path,default=ROOT/'benchmark-results/exponential-second-order/chunk7-twigl-widths.json')
    args = parser.parse_args()
    if any(r<1 for r in args.refinements):
        parser.error('refinements must be positive')
    sources = ['garnet/transient.py','garnet/timescheme.py','garnet/benchmarks/twigl.py',
               'tools/assess_exponential_twigl.py','tools/assess_exponential_twigl_widths.py']
    report = dict(date='2026-10-26',scope='Stronger output-width BE/BDF controls for the spatial FTM gate',
        reference='Same independently assembled physical system and sparse Radau rtol=3e-11 as primary sweep',
        primary_default_max_dt_s=.02,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},cases={})
    for refinement in args.refinements:
        for kind in args.kinds:
            system = TwiglSystem.build(kind,refinement)
            reference = Reference.solve(system,kind,rtol=3e-11)
            records = [run_width(system,reference,kind,'adaptive-bdf',tolerance,maximum)
                       for maximum in (.01,.005) for tolerance in (1e-3,1e-4)]
            records.append(run_width(system,reference,kind,'adaptive-be',1e-3,.005))
            report['cases'][f'{kind}-{refinement}'] = dict(k0=system.steady.k_eff,records=records)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
