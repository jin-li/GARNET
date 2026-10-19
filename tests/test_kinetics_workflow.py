"""Compose opt-in yield transfer, general C, reference factors and projection."""
import numpy as np
from ndgpu.kinetics_yield_transfer import transfer_delayed_yields
from ndgpu.kinetics_production import assemble_kinetics_production
from ndgpu.nem_higher_order import NEMHigherOrderDiffusionSolver
from ndgpu.nem_reference import qualify_nem_reference_factors
from ndgpu.nem_kinetics import solve_nem_adjoint, project_nem_kinetics
from test_nem import tensor_grid, materials


def test_raw_production_to_matching_reference_receiver_and_kinetics():
    grid=tensor_grid(([0.,.7,1.9],[0.,1.3],[0.,.8]))
    mat=materials(grid,2)
    source_total=mat[3]*np.array([[1.01,.99],[.98,1.02]])
    delayed=np.empty((2,2,2,2))
    delayed[:,0]=np.array([.95,.05])[None,:,None]*(source_total*[.004,.005])[:,None,:]
    delayed[:,1]=np.array([.8,.2])[None,:,None]*(source_total*[.001,.002])[:,None,:]
    transfer=transfer_delayed_yields(source_total,delayed,mat[3],
        source_domain='synthetic orbit estimator',receiver_domain='synthetic representative estimator',
        qualification='Synthetic homologous regions; explicit coefficient perturbations, not transport data.')
    production=assemble_kinetics_production(mat[3],mat[4],transfer.delayed_production)
    baseline=NEMHigherOrderDiffusionSolver(grid,*mat,degree=8,fission_matrix=production.total)
    reference=baseline.solve_eigenvalue(rtol=1e-12,max_iterations=2000)
    factors=qualify_nem_reference_factors(baseline,k_eff=reference.k_eff,means=reference.means,
        currents=reference.currents,heterogeneous_faces=reference.side_face_means,
        fuel_mask=np.ones(2,dtype=bool),degrees=(4,6,8))
    np.testing.assert_allclose(factors.factors,1.,rtol=2e-10)
    # Build a new immutable receiver; do not mutate the reference solver.
    receiver=NEMHigherOrderDiffusionSolver(grid,*mat,factors.factors,degree=8,
                                          fission_matrix=production.total)
    inv=np.array([[1e-7,2e-5],[1.3e-7,2.6e-5]])
    results=[]
    for solver in (baseline,receiver):
        fwd=solver.solve_eigenvalue(rtol=1e-12,max_iterations=2000)
        adj=solve_nem_adjoint(solver,fwd,rtol=1e-9)
        results.append(project_nem_kinetics(solver,fwd,adj,inverse_velocities=inv,
                        delayed_production=production.delayed,decay=[.08,.3]))
    np.testing.assert_allclose(results[1].beta,results[0].beta,rtol=2e-9)
    np.testing.assert_allclose(results[1].generation_time,results[0].generation_time,rtol=2e-9)
    assert 0<results[1].beta_total<.02
    assert results[1].generation_time>0
