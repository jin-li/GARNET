"""Scalar-only native iteration observation must not change numerical results."""
from dataclasses import replace
import json

import numpy as np
import pytest

import garnet.coupled_moment_solver as driver
from garnet.corrected_native_moments import CorrectedNativeMomentOperator
from garnet.mesh import MeshGrid
from test_corrected_mesh import mesh_grid
from test_corrected_native_moments import op as sp3_operator, orthogonal
from test_coupled_moments import CoupledMomentOperator, material
from test_native_sp5 import operator as sp5_operator


def make_operator(family,corrected=True):
    if family == 'sp5':
        return sp5_operator(corrected)
    mesh = mesh_grid(2,0.)
    mats = [material(i%2) for i in range(mesh.n_cells)]
    if corrected:
        return sp3_operator(mesh,mats)
    return CoupledMomentOperator(orthogonal(mesh),mats)


def snapshot(event):
    assert all(isinstance(value,(str,int,float,bool)) for value in event.values())
    return json.loads(json.dumps(event,allow_nan=False))


@pytest.mark.parametrize('family', ['sp3','sp5'])
@pytest.mark.parametrize('corrected', [False,True])
@pytest.mark.parametrize('preconditioner', ['none','group_ilu'])
def test_default_and_logging_bitwise_states_history_actions_and_counts(family,corrected,preconditioner):
    a = make_operator(family,corrected)
    default = driver.CoupledMomentEigenSolver(a).solve(preconditioner=preconditioner)
    events = []
    def callback(event):
        events.append(snapshot(event))
        event.clear()  # Snapshot mutation must not reach a solver record.
        event.update(outer=-999,converged=True,true_action_relative_residual=0.)
    logged = driver.CoupledMomentEigenSolver(a).solve(preconditioner=preconditioner,iteration_callback=callback)
    assert default.converged and logged.converged
    assert logged.k_eff == default.k_eff and logged.k_eff.hex() == default.k_eff.hex()
    assert logged.history == default.history
    assert logged.reason == default.reason and logged.iterations == default.iterations
    assert logged.full_residual == default.full_residual
    for name in ('moments','scalar_flux','power'):
        actual,expected = getattr(logged,name),getattr(default,name)
        np.testing.assert_array_equal(actual,expected)
        assert actual.tobytes() == expected.tobytes()
    np.testing.assert_array_equal(a.action(logged.moments),a.action(default.moments))
    np.testing.assert_array_equal(a.fission_action(logged.moments),a.fission_action(default.moments))
    starts = [e for e in events if e['phase'] == 'inner_start']
    completes = [e for e in events if e['phase'] == 'inner_complete']
    outers = [e for e in events if e['phase'] == 'outer_complete']
    assert len(starts) == len(completes) == len(outers) == logged.iterations
    for row,start,complete,outer in zip(logged.history,starts,completes,outers):
        index = row['outer']
        progress = [e for e in events if e['phase'] == 'inner_iteration' and e['outer'] == index]
        assert start == dict(phase='inner_start',outer=index,max_inner=1000,restart=min(40,a.size),inner_tolerance=1e-11)
        assert complete['count'] == len(progress) == row['inner_iterations']
        assert [e['inner_iteration'] for e in progress] == list(range(1,len(progress)+1))
        assert all(e['residual_kind'] == 'scipy_preconditioned_relative_residual_monitor_only' and e['used_for_acceptance'] is False for e in progress)
        assert complete['gmres_info'] == row['gmres_info'] == 0 and complete['accepted_linear'] is True
        assert complete['true_action_relative_residual'] == row['linear_residual']
        for name in ('k_eff','k_change','source_change','full_residual'):
            assert outer[name] == row[name]
        assert outer['reconstructed_nonnegative'] is True
        assert outer['converged'] == (index == logged.iterations)
    assert outers[-1]['reason'] == 'converged'


def test_logging_preserves_exact_gmres_parameters_and_fresh_initial(monkeypatch):
    a = make_operator('sp5')
    original = driver.gmres
    calls = []
    def gmres(matrix,rhs,**kwargs):
        assert kwargs['x0'] is None and kwargs['atol'] == 0 and kwargs['M'] is None
        assert kwargs['restart'] == min(40,a.size) and kwargs['maxiter'] == 1000
        assert kwargs['rtol'] == 1e-11 and kwargs['callback_type'] == 'legacy'
        calls.append(True)
        return original(matrix,rhs,**kwargs)
    monkeypatch.setattr(driver,'gmres',gmres)
    result = driver.CoupledMomentEigenSolver(a).solve(iteration_callback=lambda event:None)
    assert result.converged and len(calls) == result.iterations


def test_preconditioned_progress_cannot_replace_true_action_acceptance():
    a = make_operator('sp3')
    a.assembled = a.assembled*1.1  # Solve matrix intentionally violates true action.
    events = []
    failed = driver.CoupledMomentEigenSolver(a).solve(preconditioner='group_ilu',iteration_callback=lambda e:events.append(snapshot(e)))
    assert not failed.converged and failed.reason == 'inner_failure_or_full_linear_residual'
    progress = [e for e in events if e['phase'] == 'inner_iteration']
    complete = [e for e in events if e['phase'] == 'inner_complete']
    assert progress and len(complete) == 1
    assert progress[-1]['scipy_preconditioned_residual_norm'] < 1e-11
    assert complete[0]['true_action_relative_residual'] > .01
    assert complete[0]['accepted_linear'] is False
    assert complete[0]['true_action_relative_residual'] == failed.history[0]['linear_residual']
    assert not any(e['phase'] == 'outer_complete' for e in events)


def test_failed_gmres_info_cannot_emit_accepted_progress(monkeypatch):
    a = make_operator('sp3')
    def gmres(matrix,rhs,**kwargs):
        kwargs['callback'](0.)
        return np.linalg.solve(matrix.toarray(),rhs),7
    monkeypatch.setattr(driver,'gmres',gmres)
    events = []
    failed = driver.CoupledMomentEigenSolver(a).solve(iteration_callback=lambda e:events.append(snapshot(e)))
    assert not failed.converged
    complete = events[-1]
    assert complete['phase'] == 'inner_complete' and complete['gmres_info'] == 7
    assert complete['true_action_relative_residual'] < 1e-11
    assert complete['accepted_linear'] is False
    assert not any(e.get('converged') is True for e in events)


@pytest.mark.parametrize('family', ['sp3','sp5'])
def test_negative_power_outer_event_is_not_converged_or_clipped(family):
    original = make_operator(family)
    mats = [replace(m,kappa_fission=-m.kappa_fission) for m in original.materials]
    a = CorrectedNativeMomentOperator(MeshGrid(original.grid.mesh,'vacuum'),mats,family=family)
    events = []
    failed = driver.CoupledMomentEigenSolver(a).solve(iteration_callback=lambda e:events.append(snapshot(e)))
    assert not failed.converged and failed.reason == 'negative_reconstructed_scalar_or_power'
    assert np.all(failed.power < 0)
    complete = [e for e in events if e['phase'] == 'inner_complete']
    assert len(complete) == 1 and complete[0]['accepted_linear'] is True
    assert events[-1]['phase'] == 'outer_complete'
    assert events[-1]['reconstructed_nonnegative'] is False and events[-1]['converged'] is False
    assert events[-1]['reason'] == failed.reason


def test_outer_budget_progress_distinguishes_nonconvergence():
    events = []
    result = driver.CoupledMomentEigenSolver(make_operator('sp5')).solve(max_outer=1,iteration_callback=lambda e:events.append(snapshot(e)))
    assert not result.converged and result.reason == 'outer_iteration_limit'
    assert events[-1]['phase'] == 'outer_complete' and events[-1]['converged'] is False
    assert events[-1]['reason'] == 'outer_iteration_limit'
    assert events[-1]['reconstructed_nonnegative'] is True


@pytest.mark.parametrize('phase', ['inner_start','inner_iteration','inner_complete','outer_complete'])
def test_callback_failure_propagates_without_returned_convergence_or_fallback(monkeypatch,phase):
    original = driver.gmres
    calls = []
    cause = OSError('injected observation failure')
    def gmres(*args,**kwargs):
        calls.append(True)
        return original(*args,**kwargs)
    def callback(event):
        if event['phase'] == phase:
            raise cause
    monkeypatch.setattr(driver,'gmres',gmres)
    with pytest.raises(OSError) as caught:
        driver.CoupledMomentEigenSolver(make_operator('sp3')).solve(iteration_callback=callback)
    assert caught.value is cause
    assert len(calls) == (0 if phase == 'inner_start' else 1)


@pytest.mark.parametrize('value', [False,True,1,'callback',[],{}])
def test_invalid_iteration_callback_rejected_before_gmres(monkeypatch,value):
    def forbidden(*args,**kwargs):
        raise AssertionError('Invalid callback must be rejected before iteration')
    monkeypatch.setattr(driver,'gmres',forbidden)
    with pytest.raises(ValueError,match='iteration_callback'):
        driver.CoupledMomentEigenSolver(make_operator('sp3')).solve(iteration_callback=value)
