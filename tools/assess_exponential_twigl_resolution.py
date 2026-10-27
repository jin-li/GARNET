"""Locate TWIGL initial-shape resolution limits separately from integration.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m tools.assess_exponential_twigl_resolution

This inexpensive steady-state ladder does not claim transient convergence.
It follows up spatial discrepancies already present in the initial field.
"""

from __future__ import annotations

import hashlib
import json
from time import perf_counter

import numpy as np

from garnet import Model
from garnet.benchmarks import build_twigl
from tools.assess_exponential_twigl import ROOT, relative_blocks, restrict_field


def main():
    states, rows = {}, []
    for refinement in (2,4,8,16,32):
        problem = build_twigl('none',refinement)
        materials,mmap = problem.problem_at(0.)
        model = Model.from_grid(problem.grid,materials,mmap,bc=problem.bc,kinetics=problem.kinetics)
        start = perf_counter()
        steady = model.steady(device='cpu',tol_k=1e-12,tol_source=1e-11)
        if not steady.converged:
            raise RuntimeError('static spatial reference did not converge')
        n = problem.grid.shape[0]
        fission = np.array([m.nu_sigma_f for m in materials])[mmap.ravel()].T/steady.k_eff
        flux = steady.flux_numpy.reshape(2,n*n)
        flux = flux/np.sum(flux*fission)
        precursor = .0075/.08*np.sum(flux*fission,axis=0)
        states[n] = np.r_[flux.ravel(),precursor].reshape(3,n,n)*(n*n)
        rows.append(dict(shape=[n,n],k0=steady.k_eff,diagnostic_steady_seconds=perf_counter()-start))
        print('static resolution',n,steady.k_eff,flush=True)
    finest = max(states)
    for row in rows:
        n = row['shape'][0]
        row['relative_rms_difference_to_320_by_field'] = relative_blocks(
            states[n].reshape(3,-1),restrict_field(states[finest],n).reshape(3,-1)).tolist()
        if 2*n in states:
            row['relative_rms_difference_to_next_mesh_by_field'] = relative_blocks(
                states[n].reshape(3,-1),restrict_field(states[2*n],n).reshape(3,-1)).tolist()
    sources = ['garnet/solver.py','garnet/benchmarks/twigl.py','tools/assess_exponential_twigl.py',
               'tools/assess_exponential_twigl_resolution.py']
    report = dict(date='2026-10-26',scope='Initial stationary shape resolution only; no transient convergence claim',
        fields=['fast flux','thermal flux','equilibrium precursor'],
        comparison='Physical density normalization and conservative area restriction',
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},records=rows)
    output = ROOT/'benchmark-results/exponential-second-order/chunk7-initial-resolution.json'
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
