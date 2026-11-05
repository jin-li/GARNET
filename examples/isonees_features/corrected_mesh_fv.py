"""Synthetic fitted-grid diffusion/SP3/SDP1; no private reactor data needed."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import numpy as np
from garnet import Material, MeshDiffusionEigenSolver, MeshSP3EigenSolver, MeshSDP1EigenSolver
from garnet.mesh import assemble_mesh
from garnet.backend import asnumpy, device_name
from garnet import kernels


def run(device='cpu', fused=False):
    kernels.set_fused(fused)
    # A sheared fitted mesh makes the correction nonzero.
    points=[(x+.2*y,y) for y in (0.,.4,1.) for x in (0.,.3,1.)]
    cells=[(j*3+i,j*3+i+1,(j+1)*3+i+1,(j+1)*3+i) for j in range(2) for i in range(2)]
    mesh=assemble_mesh(points,cells,[0,1,0,1])
    mats=[Material([.7,.3],[.03,.08],[.02,.15],[[.2,.01],[.002,.4]],[1.,0.],total=[.24,.482]),
          Material([.4,.2],[.06,.12],[.04,.2],[[.1,.02],[.001,.3]],[1.,0.],total=[.18,.421])]
    results=[]
    for cls in (MeshDiffusionEigenSolver,MeshSP3EigenSolver,MeshSDP1EigenSolver):
        kw={} if cls is MeshDiffusionEigenSolver else dict(marshak_vacuum=True,symmetrize=False)
        solver=cls(mesh,mats,mesh.cell_tag,boundary_conditions='vacuum',
                   mesh_discretization='corrected',device=device,**kw)
        result=solver.solve(tol_k=1e-6,tol_source=1e-5,tol_balance=1e-5,
                            tol_fixed_source=1e-6,inner_rtol_floor=1e-7)
        if not result.converged:raise RuntimeError(f'{cls.__name__} did not converge')
        results.append(dict(solver=cls.__name__,device=device_name(solver.xp),fused=fused,
                            k_eff=result.k_eff,balance_error=result.balance_error,flux=asnumpy(result.flux).tolist(),
                            qualification='synthetic numerical validation; not reactor accuracy or timing'))
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cpu');p.add_argument('--fused',action='store_true');p.add_argument('--output',type=Path)
    args=p.parse_args();text=json.dumps(run(args.device,args.fused),indent=2)+'\n'
    if args.output:args.output.write_text(text)
    else:print(text)
