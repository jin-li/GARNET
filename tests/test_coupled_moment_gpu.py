"""CPU contract tests and optional real-device computational validation.

Test doubles exercise adapter control flow only; they are not GPU evidence.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from scipy.linalg import eig
import ndgpu.coupled_moment_gpu as adapter
from ndgpu.coupled_moments import CoupledMomentOperator
from ndgpu.coupled_moment_solver import CoupledMomentEigenSolver

spec = importlib.util.spec_from_file_location(
    'gpu_oracle_fixture', Path(__file__).with_name('test_coupled_moments.py'))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def operator(family='sp3'):
    return CoupledMomentOperator(fixtures.mesh(),
                                 [fixtures.material(), fixtures.material(1)], family=family)


def host_double(op, gmres):
    """Bypass GPU construction solely to test CPU controls/result contracts."""
    solver = adapter.CoupledMomentGPUEigenSolver.__new__(adapter.CoupledMomentGPUEigenSolver)
    solver.operator = op
    solver.xp = SimpleNamespace(asarray=np.asarray, asnumpy=np.asarray, __version__='14.2.0')
    solver.device = 'gpu'
    solver.device_name = 'test double (no GPU execution)'
    solver._matrix = op.assembled.copy()
    solver._gmres = gmres
    return solver


@pytest.mark.parametrize('size,budget,expected', [
    (8, 1, (1, 1)), (8, 7, (7, 7)), (8, 9, (8, 8)),
    (100, 39, (39, 39)), (100, 41, (40, 40)), (100, 1000, (40, 1000))])
def test_complete_restart_budget(size, budget, expected):
    assert adapter._inner_budget(size, budget) == expected
    restart, effective = expected
    assert effective <= budget and effective % restart == 0


def test_budget_forwarding_and_callback_count():
    calls = []
    def gmres(A, rhs, **kw):
        calls.append(kw)
        assert kw['callback_type'] == 'pr_norm' and kw['atol'] == 0
        assert kw['x0'] is None
        for _ in range(kw['maxiter'] // kw['restart']):
            kw['callback'](1.)
        return np.zeros_like(rhs), kw['maxiter']
    r = host_double(operator(), gmres).solve(max_inner=19)
    assert not r.converged and r.reason == 'inner_failure_or_full_linear_residual'
    assert calls[0]['restart'] == 8 and calls[0]['maxiter'] == 16
    assert r.history[0]['inner_iterations'] == 16
    assert r.history[0]['restart_cycles'] == 2
    assert r.provenance['unused_inner_budget'] == 3
    assert r.provenance['fallback'] is False
    assert 'CPU' in r.provenance['solver'] and 'least squares' in r.provenance['cpu_work']


def test_budget_contract_violation_and_api_error_propagate():
    def overshoot(A, rhs, **kw):
        for _ in range(3): kw['callback'](1.)
        return rhs, 24
    with pytest.raises(RuntimeError, match='budget contract'):
        host_double(operator(), overshoot).solve(max_inner=19)
    def incompatible(A, rhs, **kw):
        raise TypeError('unexpected keyword rtol')
    with pytest.raises(TypeError, match='rtol'):
        host_double(operator(), incompatible).solve()


@pytest.mark.parametrize('device', ['cpu', 'auto', '', 'bogus', None, 1])
def test_explicit_gpu_device_required(device):
    with pytest.raises(ValueError, match='Explicit device'):
        adapter.CoupledMomentGPUEigenSolver(operator(), device=device)


def test_invalid_operator_and_non_gpu_backend(monkeypatch):
    with pytest.raises(ValueError, match='CoupledMomentOperator'):
        adapter.CoupledMomentGPUEigenSolver(None)
    monkeypatch.setattr(adapter.backend, 'get_backend', lambda device: np)
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace())
    with pytest.raises(RuntimeError, match='Actual CuPy'):
        adapter.CoupledMomentGPUEigenSolver(operator())


def test_verified_api_required():
    def modern(A, b, *, rtol=1e-5, atol=0., restart=None, maxiter=None,
               callback=None, callback_type=None): pass
    def legacy(A, b, *, tol=1e-5): pass
    adapter._check_gmres_api(SimpleNamespace(__version__='14.2.0'), modern)
    with pytest.raises(RuntimeError, match='API'):
        adapter._check_gmres_api(SimpleNamespace(__version__='14.2.0'), legacy)
    with pytest.raises(RuntimeError, match='14.2'):
        adapter._check_gmres_api(SimpleNamespace(__version__='13.6.0'), modern)


def test_invalid_controls_and_initial_before_gmres():
    def forbidden(*args, **kw): raise AssertionError('GMRES must not be called')
    solver = host_double(operator(), forbidden)
    for kw in ({'max_inner': 0}, {'max_inner': True}, {'max_outer': 1.5},
               {'inner_tolerance': np.nan}, {'source_tolerance': -1},
               {'positivity_tolerance': 1}, {'initial': np.ones(3)},
               {'initial': np.full(solver.operator.shape, np.nan)},
               {'initial': -np.ones(solver.operator.shape)},
               {'initial': np.zeros(solver.operator.shape)}):
        with pytest.raises(ValueError): solver.solve(**kw)


def test_cpu_contract_raw_result_and_independent_action():
    def exact(A, rhs, **kw):
        kw['callback'](0.)
        return np.linalg.solve(A.toarray(), rhs), 0
    op = operator()
    solver = host_double(op, exact)
    initial = np.zeros(op.shape); initial[:, 0, :] = 2
    before = initial.copy()
    result = solver.solve(initial=initial)
    assert result.converged and result.full_residual < 1e-9
    np.testing.assert_array_equal(initial, before)
    assert sum(v * (m.nu_sigma_f @ p) for v, m, p in
               zip(op.mesh.volumes, op.materials, result.scalar_flux)) == pytest.approx(1.)
    with pytest.raises(ValueError): result.moments.setflags(write=True)
    with pytest.raises(TypeError): result.provenance['fallback'] = True
    solver._matrix = solver._matrix * 1.1
    failed = solver.solve()
    assert not failed.converged
    assert failed.reason == 'inner_failure_or_full_linear_residual'
    assert failed.history[0]['linear_residual'] > .01


def actual_gpu():
    # Only absence of CuPy or an available device may skip. API, construction,
    # and numerical failures after availability is established must fail tests.
    try:
        import cupy
    except ImportError:
        pytest.skip('CuPy unavailable; no GPU execution evidence')
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f'GPU runtime unavailable: {exc}')
    if count < 1: pytest.skip('No GPU device available')
    return cupy


@pytest.mark.parametrize('family', ['sp3', 'sdp1'])
def test_actual_gpu_against_cpu_and_independent_dense(family):
    xp = actual_gpu()
    op = operator(family)
    # Full unequal-volume, noncommuting losses; never scalarized or diagonalized.
    assert not np.allclose(op.materials[0].loss[1] @ op.materials[0].loss[2],
                           op.materials[0].loss[2] @ op.materials[0].loss[1])
    solver = adapter.CoupledMomentGPUEigenSolver(op,
                    device='rocm' if adapter.backend.is_rocm(xp) else 'cuda')
    result = solver.solve()
    cpu = CoupledMomentEigenSolver(op).solve()
    F = np.column_stack([op.fission_action(np.eye(op.size)[:, i]) for i in range(op.size)])
    values, vectors = eig(F, fixtures.oracle(op.mesh, op.materials, family))
    i = np.argmax(values.real); u = vectors[:, i].real
    u /= sum(v * (m.nu_sigma_f @ p) for v, m, p in
             zip(op.mesh.volumes, op.materials, op.scalar_flux(u)))
    assert result.converged and cpu.converged
    assert result.k_eff == pytest.approx(values[i].real, rel=2e-9)
    assert result.k_eff == pytest.approx(cpu.k_eff, rel=2e-9)
    np.testing.assert_allclose(result.moments.ravel(), u, rtol=2e-8, atol=1e-10)
    for name in ('moments', 'scalar_flux', 'power'):
        np.testing.assert_allclose(getattr(result, name), getattr(cpu, name), rtol=2e-8, atol=1e-10)
    rhs = op.fission_action(result.moments) / result.k_eff
    residual = np.linalg.norm(op.action(result.moments) - rhs) / np.linalg.norm(rhs)
    assert residual <= 1e-9 and residual == pytest.approx(result.full_residual)
    assert all(h['inner_iterations'] <= 1000 for h in result.history)
    limited = solver.solve(max_inner=1)
    assert not limited.converged
    assert limited.history[0]['inner_iterations'] == 1
    initial = np.zeros(op.shape); initial[:, 0, :] = 2; before = initial.copy()
    again = solver.solve(initial=initial)
    assert again.converged
    np.testing.assert_array_equal(initial, before)
    # A new adapter snapshots a deliberately corrupted assembly; the independent
    # CPU operator action must reject the otherwise converged GPU linear solve.
    op.assembled = op.assembled * 1.1
    bad = adapter.CoupledMomentGPUEigenSolver(op).solve()
    assert not bad.converged and bad.reason == 'inner_failure_or_full_linear_residual'
    assert bad.history[0]['linear_residual'] > .01


def test_constructor_transfers_once_and_requires_runtime_family(monkeypatch):
    op = operator(); transfers = []
    runtime = SimpleNamespace(getDeviceCount=lambda: 1, is_hip=True)
    xp = SimpleNamespace(__version__='14.2.0', cuda=SimpleNamespace(runtime=runtime))
    def modern(A, b, *, rtol=1e-5, atol=0., restart=None, maxiter=None,
               callback=None, callback_type=None):
        raise AssertionError('Constructor must not solve')
    def csr(A):
        transfers.append(A)
        return A.copy()
    monkeypatch.setitem(sys.modules, 'cupy', xp)
    monkeypatch.setitem(sys.modules, 'cupyx.scipy.sparse', SimpleNamespace(csr_matrix=csr))
    monkeypatch.setitem(sys.modules, 'cupyx.scipy.sparse.linalg', SimpleNamespace(gmres=modern))
    monkeypatch.setattr(adapter.backend, 'get_backend', lambda device: xp)
    monkeypatch.setattr(adapter.backend, 'device_name', lambda xp: 'rocm test double')
    solver = adapter.CoupledMomentGPUEigenSolver(op, device='hip')
    assert solver.device == 'hip' and len(transfers) == 1 and transfers[0] is op.assembled
    with pytest.raises(RuntimeError, match='family'):
        adapter.CoupledMomentGPUEigenSolver(op, device='cuda')
    assert len(transfers) == 1
    runtime.getDeviceCount = lambda: 0
    with pytest.raises(RuntimeError, match='Actual CuPy'):
        adapter.CoupledMomentGPUEigenSolver(op)
    assert len(transfers) == 1


def test_raw_negative_output_is_not_clipped_and_info_is_required():
    from ndgpu.moment_materials import MomentMaterial
    from ndgpu.coupled_moments import OrthogonalMesh
    m = fixtures.material()
    bad = MomentMaterial(m.total, np.array([1., -.4]), m.nu_sigma_f,
                         m.kappa_fission, {}, {}, dict(m.loss), {}, {}, dict(m.provenance), {})
    op = CoupledMomentOperator(OrthogonalMesh([1], [[0]]), [bad])
    def exact(A, rhs, **kw):
        kw['callback'](0.)
        return np.linalg.solve(A.toarray(), rhs), 0
    result = host_double(op, exact).solve()
    assert not result.converged and not result.reconstructed_nonnegative
    assert result.reason == 'negative_reconstructed_scalar_or_power'
    assert np.any(result.scalar_flux < 0) and np.any(result.power < 0)
    def nonzero_info(A, rhs, **kw):
        y, _ = exact(A, rhs, **kw)
        return y, kw['restart']
    result = host_double(operator(), nonzero_info).solve()
    assert result.history[0]['linear_residual'] < 1e-11
    assert not result.converged and result.reason == 'inner_failure_or_full_linear_residual'
