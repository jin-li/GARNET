"""Native-family summaries retain distinct isotope decay channels."""
import numpy as np
import pytest
from ndgpu.kinetics_families import summarize_delayed_channels


def test_two_isotope_native_families_and_reduction_objectives():
    b = [.001,.002,.0015,.001]
    lam = [.08,.3,.1,.35]
    result = summarize_delayed_channels(b,lam,
        channel_ids=['A/1','A/2','B/1','B/2'], reported_families=['1','2','1','2'])
    np.testing.assert_allclose(result.family_beta,[.0025,.003])
    np.testing.assert_allclose(result.family_decay,[(.001*.08+.0015*.1)/.0025,
                                                   (.002*.3+.001*.35)/.003])
    assert result.family_single_pole_exact == (False,False)
    assert result.channel_beta == tuple(b)
    assert result.lambda_low_frequency == pytest.approx(sum(b)/np.sum(np.array(b)/lam))
    assert result.lambda_score_weighted == pytest.approx(np.dot(b,lam)/sum(b))
    # Grouping disparate poles is only a report: the physical kernel is the
    # sum over original isotope channels, and is not silently replaced.
    for s in [.01,.2,2.]:
        exact = np.sum(np.array(b)/(s+np.array(lam)))
        collapsed = np.sum(np.array(result.family_beta)/(s+np.array(result.family_decay)))
        assert abs(exact-collapsed) > 1e-8


def test_zero_production_and_exact_common_decay():
    result = summarize_delayed_channels([.002,0.,.001,0.],[.1,.3,.1,.7],
        channel_ids=['A/1','A/2','B/1','B/2'], reported_families=['1','2','1','2'])
    assert result.family_decay == (.1,None)
    assert result.family_single_pole_exact == (True,True)
    assert result.lambda_low_frequency == pytest.approx(.1)
    zero = summarize_delayed_channels([0.],[.1],channel_ids=['zero'],reported_families=['1'])
    assert zero.lambda_score_weighted is zero.lambda_low_frequency is None
    inactive = summarize_delayed_channels([.002,0.],[.1,1e-310],
        channel_ids=['active','inactive'],reported_families=['1','1'])
    assert inactive.lambda_low_frequency == pytest.approx(.1)
    assert inactive.lambda_score_weighted == pytest.approx(.1)


def test_isotope_channels_end_to_end_unequal_volume_NEM():
    from ndgpu.nem_geometry import CuboidNodalGrid
    from ndgpu.nem_higher_order import NEMHigherOrderDiffusionSolver
    from ndgpu.nem_kinetics import solve_nem_adjoint,project_nem_kinetics
    grid = CuboidNodalGrid([[[0.,1.],[0.,2.],[0.,3.]],[[1.,3.],[0.,2.],[0.,3.]]],'reflective')
    solver = NEMHigherOrderDiffusionSolver(grid,[.5],[.07],[[0.]],[.1],[1.],degree=8)
    fwd = solver.solve_eigenvalue(rtol=1e-12)
    adj = solve_nem_adjoint(solver,fwd)
    C = np.zeros((2,2,1,1))
    C[0,0,0,0] = .1*.003
    C[1,1,0,0] = .1*.006
    projected = project_nem_kinetics(solver,fwd,adj,inverse_velocities=np.full((2,1),1e-5),
                                    delayed_production=C,decay=[.08,.12])
    np.testing.assert_allclose(projected.beta,[.001,.004],rtol=1e-10)
    result = summarize_delayed_channels(projected.beta,projected.decay,
        channel_ids=['A/1','B/1'],reported_families=['1','1'])
    assert result.family_decay[0] == pytest.approx((.001*.08+.004*.12)/.005)
    assert result.family_single_pole_exact == (False,)


@pytest.mark.parametrize('beta,decay,ids,families',[
    ([.01],[0.],['x'],['1']), ([.01],[float('nan')],['x'],['1']),
    ([-.01],[.1],['x'],['1']), ([1.],[.1],['x'],['1']),
    ([.01,.01],[.1,.2],['x','x'],['1','1']),
    ([.01],[.1],['x'],[]), ([.01],[.1],['x'],['']),
])
def test_invalid_channel_contract_rejected(beta,decay,ids,families):
    with pytest.raises(ValueError):
        summarize_delayed_channels(beta,decay,channel_ids=ids,reported_families=families)
