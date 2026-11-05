"""Explicit group ILU fill budgets and metadata-only setup events."""
import json

import numpy as np
import pytest
from scipy.linalg import eig

import garnet.coupled_moment_solver as driver
from test_corrected_native_moments import orthogonal
from test_native_sp5 import dense_pencil, operator


class FactorWithoutMaterializedLU:
    def __init__(self, factor):
        self.factor = factor

    @property
    def nnz(self):
        return self.factor.nnz

    def solve(self, rhs):
        return self.factor.solve(rhs)

    @property
    def L(self):
        raise AssertionError('Do not materialize L for setup statistics')

    @property
    def U(self):
        raise AssertionError('Do not materialize U for setup statistics')


@pytest.mark.parametrize('requested,effective', [(None,20),(10,10),(20,20),(np.float32(10.5),10.5)])
def test_fill_forwarded_once_per_group_fixed_factors_and_copied_events(monkeypatch,requested,effective):
    a = operator(True)
    spilu, gmres = driver.spilu, driver.gmres
    calls, applications, events = [], [], []
    def factor(block, **kwargs):
        assert kwargs == dict(drop_tol=1e-6,fill_factor=effective)
        assert block.shape == (a.n_cells*3,)*2
        calls.append(kwargs)
        return FactorWithoutMaterializedLU(spilu(block,**kwargs))
    def linear(*args, **kwargs):
        applications.append(kwargs['M']@np.arange(1.,a.size+1))
        return gmres(*args,**kwargs)
    def event(payload):
        # JSON serialization rejects any accidental array/factor in the event.
        events.append(json.loads(json.dumps(payload,allow_nan=False)))
        # Callback mutations cannot change accumulated or current statistics.
        for previous in payload['completed_groups']:
            previous['superlu_nnz'] = -123
        if 'current_group_stats' in payload:
            payload['current_group_stats']['matrix_nnz'] = -123
    monkeypatch.setattr(driver,'spilu',factor)
    monkeypatch.setattr(driver,'gmres',linear)
    kwargs = {} if requested is None else {'group_ilu_fill_factor':requested}
    result = driver.CoupledMomentEigenSolver(a).solve(preconditioner='group_ilu',group_ilu_setup_callback=event,**kwargs)
    assert result.converged and result.iterations > 1
    assert len(calls) == a.n_groups
    for application in applications:
        np.testing.assert_array_equal(application,applications[0])
    metadata = result.provenance['preconditioner_metadata']
    assert metadata['fill_factor'] == metadata['group_ilu_fill_factor'] == effective
    assert result.provenance['group_ilu_fill_factor'] == effective
    assert [event['phase'] for event in events] == ['before_extraction','before_factorization','after_factorization']*a.n_groups
    for index, payload in enumerate(events):
        group = index//3
        assert payload['group'] == group and payload['fill_factor'] == effective
        completed = group+(payload['phase'] == 'after_factorization')
        assert len(payload['completed_groups']) == completed
        assert all(row['superlu_nnz'] > 0 for row in payload['completed_groups'])
        if payload['phase'] == 'after_factorization':
            stats = payload['current_group_stats']
            assert stats == payload['completed_groups'][-1]
            assert stats['dimension'] == a.n_cells*3 and stats['matrix_nnz'] > 0
            assert stats['superlu_nnz'] > 0 and stats['setup_seconds'] >= 0
    assert all(row['matrix_nnz'] > 0 and row['superlu_nnz'] > 0 for row in metadata['groups'])


@pytest.mark.parametrize('fill', [10,20])
def test_noncommuting_sp5_budgets_match_independent_dense_and_true_criteria(fill):
    a = operator(True)
    before = [getattr(a.assembled,key).copy() for key in ('data','indices','indptr')]
    A, F = dense_pencil(orthogonal(a.grid.mesh),a.materials)
    values, vectors = eig(F,A)
    index = np.argmax(values.real)
    expected = vectors[:,index].real
    source = sum(v*(m.nu_sigma_f@p) for v,m,p in zip(a.volumes,a.materials,a.scalar_flux(expected)))
    expected /= source
    result = driver.CoupledMomentEigenSolver(a).solve(preconditioner='group_ilu',group_ilu_fill_factor=fill)
    assert result.converged and result.reconstructed_nonnegative
    assert result.k_eff == pytest.approx(values[index].real,rel=2e-9)
    np.testing.assert_allclose(result.moments.ravel(),expected,rtol=2e-8,atol=1e-10)
    assert result.full_residual <= 1e-9
    assert all(row['linear_residual'] <= 1e-11 for row in result.history)
    rhs = a.fission_action(result.moments)/result.k_eff
    assert np.linalg.norm(a.action(result.moments)-rhs)/np.linalg.norm(rhs) <= 1e-9
    for key, original in zip(('data','indices','indptr'),before):
        np.testing.assert_array_equal(getattr(a.assembled,key),original)
    assert not result.provenance['fallback']


def test_ordinary_ilu_retains_fill20_and_does_not_emit_group_events(monkeypatch):
    a = operator(True)
    original = driver.spilu
    calls, events = [], []
    def factor(matrix, **kwargs):
        calls.append(kwargs)
        return FactorWithoutMaterializedLU(original(matrix,**kwargs))
    monkeypatch.setattr(driver,'spilu',factor)
    result = driver.CoupledMomentEigenSolver(a).solve(preconditioner='ilu',group_ilu_fill_factor=10,group_ilu_setup_callback=events.append)
    assert result.converged and calls == [dict(drop_tol=1e-6,fill_factor=20)]
    assert events == []
    assert result.provenance['preconditioner_metadata']['fill_factor'] == 20
    assert result.provenance['group_ilu_fill_factor'] is None


@pytest.mark.parametrize('value', [True,False,np.bool_(True),0,-1,.5,np.nan,np.inf,-np.inf,'10',None,1+0j,[10],np.array(10),10**400])
def test_invalid_fill_budget_before_factorization(monkeypatch,value):
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid fill must be rejected before setup')
    monkeypatch.setattr(driver,'spilu',forbidden)
    with pytest.raises(ValueError,match='group_ilu_fill_factor'):
        driver.CoupledMomentEigenSolver(operator(True)).solve(preconditioner='group_ilu',group_ilu_fill_factor=value)


@pytest.mark.parametrize('value', [False,1,'callback',[]])
def test_callback_must_be_callable(value):
    with pytest.raises(ValueError,match='group_ilu_setup_callback'):
        driver.CoupledMomentEigenSolver(operator(True)).solve(group_ilu_setup_callback=value)


@pytest.mark.parametrize('stage', ['extraction','factorization'])
@pytest.mark.parametrize('exception', [MemoryError,RuntimeError,ValueError])
def test_later_failure_preserves_completed_stats_events_cause_and_no_fallback(monkeypatch,stage,exception):
    a = operator(True)
    extract, spilu = driver._group_principal_block, driver.spilu
    extraction_calls, factor_calls, events = [], [], []
    cause = exception('injected third-group failure')
    def extraction(matrix,groups,group,**kwargs):
        if group == 2 and stage == 'extraction':
            raise cause
        extraction_calls.append(group)
        return extract(matrix,groups,group,**kwargs)
    def factor(matrix,**kwargs):
        assert kwargs == dict(drop_tol=1e-6,fill_factor=10)
        if len(factor_calls) == 2 and stage == 'factorization':
            raise cause
        factor_calls.append(True)
        return FactorWithoutMaterializedLU(spilu(matrix,**kwargs))
    def event(payload):
        events.append(json.loads(json.dumps(payload,allow_nan=False)))
    def forbidden(*args, **kwargs):
        raise AssertionError('No GMRES/fallback after group setup failure')
    monkeypatch.setattr(driver,'_group_principal_block',extraction)
    monkeypatch.setattr(driver,'spilu',factor)
    monkeypatch.setattr(driver,'gmres',forbidden)
    expected = MemoryError if exception is MemoryError else RuntimeError
    with pytest.raises(expected,match='group 2.*no solver fallback') as caught:
        driver.CoupledMomentEigenSolver(a).solve(preconditioner='group_ilu',group_ilu_fill_factor=10,group_ilu_setup_callback=event)
    assert caught.value.__cause__ is cause
    assert caught.value.group == 2 and caught.value.fill_factor == 10
    assert len(caught.value.completed_groups) == len(factor_calls) == 2
    failure = events[-1]
    assert failure['phase'] == 'failure' and failure['failed_phase'] == stage
    assert failure['error_type'] == exception.__name__ and failure['error'] == str(cause)
    assert failure['completed_groups'] == caught.value.completed_groups
    assert [row['group'] for row in failure['completed_groups']] == [0,1]
    assert all(row['superlu_nnz'] > 0 and row['dimension'] == a.n_cells*3 for row in failure['completed_groups'])
    assert 'fill_factor=10.0' in str(caught.value)
    if stage == 'factorization':
        assert failure['current_group_stats']['dimension'] == a.n_cells*3


def test_failed_failure_callback_does_not_replace_original_cause(monkeypatch):
    cause = MemoryError('factor allocation failed')
    def factor(*args, **kwargs):
        raise cause
    def callback(event):
        if event['phase'] == 'failure':
            raise OSError('event log unavailable')
    monkeypatch.setattr(driver,'spilu',factor)
    with pytest.raises(MemoryError) as caught:
        driver.CoupledMomentEigenSolver(operator(True)).solve(preconditioner='group_ilu',group_ilu_setup_callback=callback)
    assert caught.value.__cause__ is cause
    assert caught.value.callback_error == 'event log unavailable'
