"""Small inner-initial-guess checks; no assembly or physical qualification."""
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

import garnet.coupled_moment_solver as driver
from garnet.corrected_native_moments import CorrectedNativeMomentOperator
from garnet.mesh import MeshGrid
from test_corrected_mesh import mesh_grid
from test_native_sp5 import operator


def make_operator(family):
    op = operator(True, n=2)
    return op if family == 'sp5' else CorrectedNativeMomentOperator(op.grid, op.materials, family=family)


def source_size(op, state):
    return float(sum(v*np.dot(m.nu_sigma_f, p)
        for v, m, p in zip(op.volumes, op.materials, op.scalar_flux(state))))


@pytest.mark.parametrize('family', ['sp3', 'sp5'])
def test_first_cold_then_scaled_normalized_state_fixed_rhs_and_no_alias(family, monkeypatch):
    op = make_operator(family)
    real_gmres = driver.gmres
    calls = []
    initial = np.zeros(op.shape); initial[:, 0, :] = 2.
    initial_before = initial.copy()
    matrix_before = [getattr(op.assembled, key).tobytes() for key in ('data', 'indices', 'indptr')]
    def solve(matrix, rhs, **kwargs):
        assert matrix is op.assembled
        assert kwargs['restart'] == min(40, op.size) and kwargs['maxiter'] == 1000
        assert kwargs['rtol'] == 1e-11 and kwargs['atol'] == 0
        assert kwargs['callback_type'] == 'legacy'
        if not calls:
            assert kwargs['x0'] is None
            np.testing.assert_array_equal(rhs, op.fission_action(initial.ravel()/source_size(op, initial)))
        else:
            previous = calls[-1]['solution']
            k = source_size(op, previous)
            normalized = previous/k
            assert source_size(op, normalized) == pytest.approx(1., abs=2e-15)
            np.testing.assert_array_equal(kwargs['x0'], k*normalized)
            np.testing.assert_array_equal(rhs, op.fission_action(normalized))
            np.testing.assert_allclose(kwargs['x0'], previous, rtol=2e-16, atol=2e-16)
            assert kwargs['M'] is calls[0]['M']
        solution, info = real_gmres(matrix, rhs, **kwargs)
        calls.append(dict(solution=solution.copy(), M=kwargs['M']))
        if kwargs['x0'] is not None:
            # A GMRES implementation can mutate x0. Its separate allocation
            # must not corrupt the source-normalized state or returned solution.
            assert not np.shares_memory(kwargs['x0'], solution)
            kwargs['x0'][:] = -1.e100
        return solution, info
    monkeypatch.setattr(driver, 'gmres', solve)
    events = []
    warm = driver.CoupledMomentEigenSolver(op).solve(initial=initial, max_outer=2,
        preconditioner='group_ilu_coarse', inner_warm_start=True, iteration_callback=events.append)
    assert len(calls) == len(warm.history) == 2
    assert all(row['gmres_info'] == 0 and row['linear_residual'] <= 1e-11 for row in warm.history)
    assert [row['inner_initial_guess_reused'] for row in warm.history] == [False, True]
    assert [row['inner_initial_guess'] for row in warm.history] == ['zero', 'k*u']
    assert all(row['restart'] == min(40, op.size) for row in warm.history)
    assert warm.provenance['inner_initial_guess_reuse_count'] == 1
    assert warm.provenance['initial_state_reused'] is False  # legacy field unchanged
    assert warm.provenance['initial_guess_source'] == 'previous source-normalized outer state scaled by k'
    for phase in ('inner_start', 'inner_complete'):
        records = [event for event in events if event['phase'] == phase]
        assert [event['inner_initial_guess_reused'] for event in records] == [False, True]
    assert all(all(isinstance(value, (str, int, float, bool)) for value in event.values()) for event in events)
    np.testing.assert_array_equal(initial, initial_before)
    assert matrix_before == [getattr(op.assembled, key).tobytes() for key in ('data', 'indices', 'indptr')]
    assert source_size(op, warm.moments) == pytest.approx(1., abs=2e-15)


@pytest.mark.parametrize('value', [None, 0, 1, 1., 'true', [], np.bool_(True)])
def test_explicit_python_bool_required(value, monkeypatch):
    monkeypatch.setattr(driver, 'gmres', lambda *args, **kwargs: pytest.fail('Invalid flag reached GMRES'))
    with pytest.raises(ValueError, match='inner_warm_start must be a bool'):
        driver.CoupledMomentEigenSolver(make_operator('sp3')).solve(inner_warm_start=value)


@pytest.fixture(scope='module')
def frozen_solver():
    baseline = subprocess.run(['git', 'show', '70c399c:garnet/coupled_moment_solver.py'],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    if baseline.returncode:
        pytest.skip('Frozen 70c399c source object unavailable in source archive')
    name = 'garnet._inner_warm_start_frozen_solver'
    module = types.ModuleType(name); module.__file__ = driver.__file__
    sys.modules[name] = module
    try:
        exec(compile(baseline.stdout, driver.__file__, 'exec'), module.__dict__)
        yield module.CoupledMomentEigenSolver
    finally:
        del sys.modules[name]


@pytest.mark.parametrize('family', ['sp3', 'sp5'])
@pytest.mark.parametrize('preconditioner', ['none', 'ilu', 'group_ilu', 'group_ilu_coarse'])
def test_default_and_explicit_false_bitwise_against_70c399c(family, preconditioner, frozen_solver):
    op = make_operator(family)
    expected = frozen_solver(op).solve(preconditioner=preconditioner)
    for options in ({}, dict(inner_warm_start=False)):
        events = []
        current = driver.CoupledMomentEigenSolver(op).solve(preconditioner=preconditioner,
            iteration_callback=events.append, accepted_outer_state_callback=None, **options)
        assert current.history == expected.history
        assert current.k_eff.hex() == expected.k_eff.hex()
        assert current.full_residual.hex() == expected.full_residual.hex()
        assert current.reason == expected.reason and current.converged == expected.converged
        for key in ('moments', 'scalar_flux', 'power'):
            assert getattr(current, key).tobytes() == getattr(expected, key).tobytes()
        assert current.provenance['inner_warm_start'] is False
        assert current.provenance['inner_initial_guess_reuse_count'] == 0
        assert 'initial_guess_source' not in current.provenance
        assert all('inner_initial_guess_reused' not in event for event in events)
        assert all('coarse_applications' not in event for event in events)


@pytest.mark.parametrize('failure', ['status', 'independent_residual'])
def test_failed_warm_inner_still_rejected(failure, monkeypatch):
    op = make_operator('sp5')
    real_gmres = driver.gmres; calls = []
    def solve(matrix, rhs, **kwargs):
        if not calls:
            assert kwargs['x0'] is None
            solution, info = real_gmres(matrix, rhs, **kwargs)
            calls.append(solution.copy())
            return solution, info
        assert kwargs['x0'] is not None
        solution = np.linalg.solve(matrix.toarray(), rhs)
        calls.append(solution.copy())
        return (solution, 1) if failure == 'status' else (.9*solution, 0)
    monkeypatch.setattr(driver, 'gmres', solve)
    accepted = []
    result = driver.CoupledMomentEigenSolver(op).solve(inner_warm_start=True, max_outer=2,
        preconditioner='group_ilu_coarse', accepted_outer_state_callback=accepted.append)
    assert len(result.history) == 2 and len(calls) == 2
    assert result.reason == 'inner_failure_or_full_linear_residual' and not result.converged
    assert result.provenance['inner_initial_guess_reuse_count'] == 1
    assert result.history[1]['inner_initial_guess_reused'] is True
    assert len(accepted) == 1 and accepted[0]['outer'] == 1
    # Rejecting the second solve leaves the accepted first outer state intact.
    np.testing.assert_array_equal(result.moments.ravel(), calls[0]/source_size(op, calls[0]))


@pytest.mark.parametrize('value', [False, 0, 'callback', []])
def test_state_callback_requires_callable(value):
    with pytest.raises(ValueError, match='accepted_outer_state_callback must be callable or None'):
        driver.CoupledMomentEigenSolver(make_operator('sp3')).solve(accepted_outer_state_callback=value)


@pytest.mark.parametrize('family', ['sp3', 'sp5'])
def test_candidate_callback_copies_ordering_count_and_mutation_isolation(family):
    op = make_operator(family)
    settings = dict(preconditioner='group_ilu_coarse', inner_warm_start=True)
    expected = driver.CoupledMomentEigenSolver(op).solve(**settings)
    accepted = []; order = []; progress_records = []
    def progress(event):
        progress_records.append(event)
        order.append((event['outer'], event['phase']))
    def checkpoint(payload):
        outer = payload['outer']
        assert order[-1] == (outer, 'inner_complete')
        order.append((outer, 'candidate'))
        y, rhs, state = (payload[key] for key in ('linear_solution', 'rhs', 'native_state'))
        assert y.shape == rhs.shape == state.shape == (op.size,)
        assert not np.shares_memory(y, state) and not np.shares_memory(y, rhs)
        np.testing.assert_array_equal(state, y/payload['k_eff'])
        assert source_size(op, state) == pytest.approx(1., abs=2e-15)
        assert np.linalg.norm(op.action(y)-rhs)/np.linalg.norm(rhs) == payload['linear_residual']
        assert payload['gmres_info'] == 0 and payload['linear_residual'] <= 1e-11
        assert np.min(op.scalar_flux(state)) >= 0
        accepted.append({key:value.copy() if isinstance(value,np.ndarray) else value for key,value in payload.items()})
        for key in ('linear_solution', 'rhs', 'native_state'):
            payload[key][:] = -1.e100
        payload.clear(); payload['outer'] = -999
    observed = driver.CoupledMomentEigenSolver(op).solve(**settings,
        iteration_callback=progress, accepted_outer_state_callback=checkpoint)
    assert len(accepted) == observed.iterations == len(observed.history)
    assert observed.history == expected.history and observed.k_eff == expected.k_eff
    for key in ('moments','scalar_flux','power'):
        assert getattr(observed,key).tobytes() == getattr(expected,key).tobytes()
    for record in accepted:
        outer = record['outer']
        index = order.index((outer,'candidate'))
        assert order[index+1] == (outer,'outer_complete')
        assert record['converged'] == (outer == observed.iterations)
        start = next(e for e in progress_records if e['phase']=='inner_start' and e['outer']==outer)
        complete = next(e for e in progress_records if e['phase']=='inner_complete' and e['outer']==outer)
        assert record['coarse_applications'] == complete['coarse_applications'] > start['coarse_applications']
        assert record['coarse_applications'] == record['coarse_solves'] == record['coarse_extra_a_matvecs']
        assert record['coarse_setup_coarse_solves'] == 1
        assert record['coarse_aggregate_count'] == op.n_cells
        assert record['coarse_dimension'] == op.size
        assert record['coarse_aggregate_sha256'] == complete['coarse_aggregate_sha256'] == start['coarse_aggregate_sha256']
        assert len(record['coarse_aggregate_sha256']) == 64
        if outer == 1:
            assert start['coarse_applications'] == start['coarse_solves'] == start['coarse_extra_a_matvecs'] == 0
        else:
            assert start['coarse_applications'] == accepted[outer-2]['coarse_applications']
    meta = observed.provenance['preconditioner_metadata']['coarse']
    assert accepted[-1]['coarse_applications'] == meta['applications']
    assert all(all(isinstance(v,(str,int,float,bool)) for v in e.values()) for e in progress_records)


def test_candidate_callback_exception_prevents_next_gmres_and_one_outer_reuse(monkeypatch):
    op = make_operator('sp5'); real_gmres = driver.gmres; calls = []
    def solve(*args, **kwargs):
        calls.append(True)
        return real_gmres(*args, **kwargs)
    monkeypatch.setattr(driver,'gmres',solve)
    class CheckpointStop(Exception):
        pass
    saved = []
    def checkpoint(payload):
        assert payload['outer'] == 1 and payload['inner_initial_guess_reused'] is False
        saved.append(payload)
        raise CheckpointStop('checkpoint saved; stop before another GMRES')
    with pytest.raises(CheckpointStop, match='checkpoint saved'):
        driver.CoupledMomentEigenSolver(op).solve(inner_warm_start=True,
            preconditioner='group_ilu_coarse', accepted_outer_state_callback=checkpoint)
    assert len(saved) == len(calls) == 1
    one_outer = driver.CoupledMomentEigenSolver(op).solve(inner_warm_start=True,
        preconditioner='group_ilu_coarse', max_outer=1)
    assert one_outer.provenance['inner_initial_guess_reuse_count'] == 0
    assert one_outer.history[0]['inner_initial_guess_reused'] is False


def test_no_candidate_callback_when_reconstruction_fails_positivity(monkeypatch):
    op = make_operator('sp5'); real_gmres = driver.gmres; real_scalar_flux = op.scalar_flux
    solved = []; accepted = []
    def solve(*args, **kwargs):
        result = real_gmres(*args, **kwargs)
        solved.append(True)
        return result
    def scalar_flux(state):
        flux = real_scalar_flux(state)
        if solved:
            flux = np.array(flux, copy=True)
            flux[0] = -np.abs(flux[0])
            flux[1:] = np.abs(flux[1:])+100*np.abs(flux[0])
        return flux
    monkeypatch.setattr(driver,'gmres',solve)
    monkeypatch.setattr(op,'scalar_flux',scalar_flux)
    result = driver.CoupledMomentEigenSolver(op).solve(inner_warm_start=True,
        preconditioner='group_ilu_coarse', accepted_outer_state_callback=accepted.append)
    assert result.reason == 'negative_reconstructed_scalar_or_power' and not result.converged
    assert len(solved) == 1 and not accepted


@pytest.fixture(scope='module')
def actual_materials():
    source = os.environ.get('NDGPU_NATIVE_COARSE_TEST_XS')
    if not source:
        pytest.skip('Set NDGPU_NATIVE_COARSE_TEST_XS to audited native derivative for local evidence')
    from garnet.serpent_xs import read_serpent_moment_results
    records = read_serpent_moment_results(Path(source), required_orders=tuple(range(6)))
    return [record.to_moment_material(orientation='from-major') for record in records]


@pytest.mark.parametrize('family', ['sp3', 'sp5'])
def test_actual_data_tiny_warm_cold_residual_balance_and_positivity(family, actual_materials):
    mesh = mesh_grid(2, .15)
    op = CorrectedNativeMomentOperator(MeshGrid(mesh, 'vacuum'),
        [actual_materials[i % len(actual_materials)] for i in range(mesh.n_cells)], family=family)
    settings = dict(preconditioner='group_ilu_coarse', group_ilu_fill_factor=10,
        inner_tolerance=1e-10, k_tolerance=1e-9, source_tolerance=1e-8, residual_tolerance=1e-8)
    cold = driver.CoupledMomentEigenSolver(op).solve(**settings)
    warm = driver.CoupledMomentEigenSolver(op).solve(inner_warm_start=True, **settings)
    assert cold.converged and warm.converged
    for result in (cold, warm):
        state = result.moments.ravel()
        rhs = op.fission_action(state)/result.k_eff
        independent = np.linalg.norm(op.action(state)-rhs)/np.linalg.norm(rhs)
        assert independent == result.full_residual and independent <= settings['residual_tolerance']
        assert source_size(op, state) == pytest.approx(1., abs=3e-15)
        assert result.reconstructed_nonnegative and np.min(result.scalar_flux) >= 0 and np.min(result.power) >= 0
        assert result.history[-1]['source_change'] <= settings['source_tolerance']
        assert result.history[-1]['k_change'] <= settings['k_tolerance']
    assert warm.history[0]['inner_initial_guess_reused'] is False
    assert warm.provenance['inner_initial_guess_reuse_count'] == warm.iterations-1
    np.testing.assert_allclose(warm.k_eff, cold.k_eff, rtol=2e-9)
    np.testing.assert_allclose(warm.moments, cold.moments, rtol=2e-7, atol=1e-10)
    np.testing.assert_allclose(warm.power, cold.power, rtol=2e-7, atol=1e-10)
