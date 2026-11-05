"""Small solver hook checks; actual cache admission remains a separate probe."""
import hashlib
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

import garnet.coupled_moment_solver as driver
from garnet.corrected_native_moments import CorrectedNativeMomentOperator as Operator
from garnet.native_coarse import NativeCoarsePreconditioner as Coarse
from test_native_compact import fixture
from test_native_compact_solver import GATES

CACHE = dict(coarse_cache_prolongation_action=True, coarse_cache_max_bytes=8_000_000)


def operator(family='sp3', storage='deferred'):
    grid, mats = fixture(family)
    return Operator(grid, mats, family=family, storage=storage, gradient_chunk_cells=1, batch_size=3)


@pytest.mark.parametrize('family', ['sp3', 'sp5'])
@pytest.mark.parametrize('warm', [False, True])
def test_real_cached_native_solver_parity_and_native_residual_calls(family, warm, monkeypatch):
    import garnet.native_coarse as coarse_module
    ids = np.array([0, 0, 1, 1], dtype=np.int64); ids.setflags(write=False)
    monkeypatch.setattr(coarse_module, 'operator_aggregates', lambda op: ids)
    op = operator(family)
    native = driver.CoupledMomentEigenSolver(op).solve(preconditioner='group_ilu_coarse',
        inner_warm_start=warm, **GATES)
    action = op.action
    active = [False]
    external_actions, accepted, events = [], [], []
    def spy(values):
        if not active[0]:
            external_actions.append(np.array(values, copy=True))
        return action(values)
    real_gmres = driver.gmres
    def gmres(*args, **kwargs):
        active[0] = True
        try:
            return real_gmres(*args, **kwargs)
        finally:
            active[0] = False
    monkeypatch.setattr(op, 'action', spy)
    monkeypatch.setattr(driver, 'gmres', gmres)
    cached = driver.CoupledMomentEigenSolver(op).solve(preconditioner='group_ilu_coarse',
        inner_warm_start=warm, accepted_outer_state_callback=accepted.append,
        iteration_callback=events.append, **CACHE, **GATES)
    assert native.converged and cached.converged
    assert cached.reason == native.reason == 'converged'
    for name in ('k_eff', 'moments', 'scalar_flux', 'power', 'full_residual'):
        np.testing.assert_allclose(getattr(cached, name), getattr(native, name), rtol=2e-12, atol=5e-14)
    assert len(external_actions) == 2*len(accepted)+1
    for index, state in enumerate(accepted):
        np.testing.assert_array_equal(external_actions[2*index], state['linear_solution'])
        np.testing.assert_array_equal(external_actions[2*index+1], state['native_state'])
        residual = np.linalg.norm(action(state['linear_solution'])-state['rhs'])/np.linalg.norm(state['rhs'])
        assert state['linear_residual'] == residual
        assert state['coarse_cached_ap_matvecs'] > 0
    np.testing.assert_array_equal(external_actions[-1], cached.moments.ravel())
    meta = cached.provenance['preconditioner_metadata']['coarse']
    assert cached.provenance['coarse_cache_prolongation_action'] is True
    assert cached.provenance['coarse_cache_max_bytes'] == CACHE['coarse_cache_max_bytes']
    assert meta['extra_a_matvecs'] == 0
    assert meta['cached_ap_matvecs'] == meta['applications'] > 0
    assert cached.history[-1]['coarse_cached_ap_matvecs'] == meta['cached_ap_matvecs']
    assert all(row['linear_residual'] <= GATES['inner_tolerance'] for row in cached.history)
    assert cached.full_residual <= GATES['residual_tolerance']
    hashes = cached.provenance['implementation_sha256']
    path = Path(driver.__file__).with_name('native_coarse_action_cache.py')
    assert hashes[path.name] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.name not in native.provenance['implementation_sha256']
    assert all('coarse_cached_ap_matvecs' not in row for row in native.history)
    assert all(e['coarse_cached_ap_matvecs'] > 0 for e in events if e['phase'] == 'inner_complete')


@pytest.mark.parametrize('options', [
    dict(coarse_cache_prolongation_action=1), dict(coarse_cache_prolongation_action=np.bool_(True)),
    dict(coarse_cache_prolongation_action='yes'), dict(coarse_cache_max_bytes=8_000_000),
    dict(coarse_cache_max_bytes=0), dict(coarse_cache_prolongation_action=True),
    dict(coarse_cache_prolongation_action=True, coarse_cache_max_bytes=True),
    dict(coarse_cache_prolongation_action=True, coarse_cache_max_bytes=0),
    dict(coarse_cache_prolongation_action=True, coarse_cache_max_bytes=-1),
    dict(coarse_cache_prolongation_action=True, coarse_cache_max_bytes=1.5),
    dict(coarse_cache_prolongation_action=True, coarse_cache_max_bytes='8000000'),
])
def test_invalid_cache_kwargs_reject_before_actions_or_factors(options, monkeypatch):
    op = operator()
    def forbidden(*args, **kwargs):
        raise AssertionError('Action or setup occurred before cache option rejection')
    for name in ('action', 'fission_action', 'scalar_flux', 'linear_operator', 'group_principal_block'):
        monkeypatch.setattr(op, name, forbidden)
    monkeypatch.setattr(driver, 'spilu', forbidden)
    monkeypatch.setattr(Coarse, 'from_operator', forbidden)
    with pytest.raises(ValueError, match='coarse_cache|cache_max_bytes'):
        driver.CoupledMomentEigenSolver(op).solve(preconditioner='group_ilu_coarse', **options)


@pytest.mark.parametrize('storage,mode', [('assembled', 'group_ilu_coarse'),
    ('deferred', 'none'), ('deferred', 'group_ilu'), ('deferred', 'ilu')])
def test_unsupported_cache_modes_reject_before_actions_or_factors(storage, mode, monkeypatch):
    op = operator(storage=storage)
    def forbidden(*args, **kwargs):
        raise AssertionError('Unsupported mode performed numerical work')
    for name in ('action', 'fission_action', 'scalar_flux', 'linear_operator', 'group_principal_block'):
        monkeypatch.setattr(op, name, forbidden)
    monkeypatch.setattr(driver, 'spilu', forbidden)
    with pytest.raises(ValueError, match='requires deferred corrected-native group_ilu_coarse'):
        driver.CoupledMomentEigenSolver(op).solve(preconditioner=mode, **CACHE)


def test_default_factory_call_and_explicit_false_are_unchanged(monkeypatch):
    op = operator()
    factory = Coarse.from_operator
    calls = []
    def spy(operator, smoother, **kwargs):
        calls.append(kwargs)
        return factory(operator, smoother, **kwargs)
    monkeypatch.setattr(Coarse, 'from_operator', spy)
    a = driver.CoupledMomentEigenSolver(op).solve(preconditioner='group_ilu_coarse', **GATES)
    b = driver.CoupledMomentEigenSolver(op).solve(preconditioner='group_ilu_coarse',
        coarse_cache_prolongation_action=False, coarse_cache_max_bytes=None, **GATES)
    assert calls == [{}, {}]
    assert a.history == b.history and a.k_eff == b.k_eff
    for name in ('moments', 'scalar_flux', 'power'):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


@pytest.mark.parametrize('relative,accepted', [(.9e-7, True), (1.1e-7, False)])
def test_cached_path_preserves_independent_true_inner_gate(relative, accepted, monkeypatch):
    op = operator()
    matrix = operator(storage='assembled').assembled.toarray()
    def gmres(linear, rhs, **kwargs):
        # Apply cached M to ensure the enabled path is exercised, then claim
        # convergence for a controlled inaccurate native solution.
        kwargs['M']@rhs
        return (1+relative)*np.linalg.solve(matrix, rhs), 0
    monkeypatch.setattr(driver, 'gmres', gmres)
    result = driver.CoupledMomentEigenSolver(op).solve(preconditioner='group_ilu_coarse',
        max_outer=1, **CACHE, **GATES)
    first = result.history[0]
    np.testing.assert_allclose(first['linear_residual'], relative, rtol=1e-12, atol=1e-14)
    assert ('k_eff' in first) == accepted
    assert (result.reason == 'inner_failure_or_full_linear_residual') != accepted
    assert first['coarse_cached_ap_matvecs'] == 1


@pytest.mark.parametrize('phase', ['setup', 'action'])
def test_cache_errors_propagate_without_fallback(phase, monkeypatch):
    from garnet.native_coarse_action_cache import ProlongationActionCache
    class CacheFailure(Exception):
        pass
    def fail(*args, **kwargs):
        raise CacheFailure('original cache failure')
    if phase == 'setup':
        monkeypatch.setattr(Coarse, 'from_operator', fail)
    else:
        monkeypatch.setattr(ProlongationActionCache, 'action', fail)
    with pytest.raises(CacheFailure, match='original cache failure'):
        driver.CoupledMomentEigenSolver(operator()).solve(preconditioner='group_ilu_coarse', **CACHE, **GATES)


def test_cache_budget_rejection_is_not_silently_disabled():
    with pytest.raises(ValueError, match='estimated peak'):
        driver.CoupledMomentEigenSolver(operator()).solve(preconditioner='group_ilu_coarse',
            coarse_cache_prolongation_action=True, coarse_cache_max_bytes=1, **GATES)


def test_packed_cpu_cache_compatibility():
    from garnet.packed_superlu import _check_backend
    try:
        _check_backend()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    op = operator()
    native = driver.CoupledMomentEigenSolver(op).solve(preconditioner='group_ilu_coarse',
        group_ilu_storage='packed_cpu', **GATES)
    cached = driver.CoupledMomentEigenSolver(op).solve(preconditioner='group_ilu_coarse',
        group_ilu_storage='packed_cpu', **CACHE, **GATES)
    assert native.converged and cached.converged
    for name in ('k_eff', 'moments', 'scalar_flux', 'power'):
        np.testing.assert_allclose(getattr(cached, name), getattr(native, name), rtol=2e-12, atol=5e-14)
    assert 'packed_superlu.py' in cached.provenance['implementation_sha256']
    assert cached.provenance['preconditioner_metadata']['coarse']['extra_a_matvecs'] == 0


def test_original_476e758_default_coarse_solver_bitwise():
    baseline = subprocess.run(['git', 'show', '476e758:garnet/coupled_moment_solver.py'],
        cwd=Path(driver.__file__).parent.parent, capture_output=True, text=True)
    if baseline.returncode:
        pytest.skip('Original 476e758 object unavailable in this source archive')
    name = 'garnet._cache_hook_baseline_solver'
    module = types.ModuleType(name); module.__file__ = driver.__file__; sys.modules[name] = module
    try:
        exec(compile(baseline.stdout, driver.__file__, 'exec'), module.__dict__)
        op = operator()
        for warm in (False, True):
            kwargs = dict(preconditioner='group_ilu_coarse', inner_warm_start=warm, **GATES)
            old = module.CoupledMomentEigenSolver(op).solve(**kwargs)
            new = driver.CoupledMomentEigenSolver(op).solve(**kwargs)
            assert old.history == new.history and old.k_eff == new.k_eff and old.full_residual == new.full_residual
            for field in ('moments', 'scalar_flux', 'power'):
                np.testing.assert_array_equal(getattr(old, field), getattr(new, field))
    finally:
        del sys.modules[name]


def test_enabled_progress_counter_without_accepted_state_callback():
    events = []
    result = driver.CoupledMomentEigenSolver(operator()).solve(preconditioner='group_ilu_coarse',
        iteration_callback=events.append, max_outer=1, **CACHE, **GATES)
    complete = next(e for e in events if e['phase'] == 'inner_complete')
    assert complete['coarse_cached_ap_matvecs'] == result.history[0]['coarse_cached_ap_matvecs'] > 0
    assert complete['coarse_extra_a_matvecs'] == 0
