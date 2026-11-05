"""Opt-in packed native group ILU parity and fail-closed ownership checks."""

import json
import weakref

import numpy as np
import pytest
from scipy.linalg import eig

import garnet.coupled_moment_solver as driver
import garnet.packed_superlu as packed_module
from garnet.corrected_native_moments import CorrectedNativeMomentOperator
from garnet.packed_superlu import PackedSuperLU
from test_corrected_native_moments import orthogonal
from test_native_compact import fixture
from test_native_sp5 import dense_pencil, operator


@pytest.fixture
def packed_backend():
    """Packed numerical checks require the exact private ABI production admits."""
    from garnet.packed_superlu import _check_backend
    try:
        _check_backend()
    except RuntimeError as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize('mode', ['group_ilu', 'group_ilu_coarse'])
@pytest.mark.parametrize('deferred', [False, True])
@pytest.mark.usefixtures("packed_backend")
def test_native_eigen_residual_coarse_and_signed_application_parity(mode, deferred, monkeypatch):
    # Noncommuting three-group SP5 material, repeated faces, mixed boundaries.
    grid, materials = fixture('sp5')
    op = CorrectedNativeMomentOperator(grid, materials, family='sp5',
        storage='deferred' if deferred else 'assembled')
    applications = []
    original_gmres = driver.gmres
    rng = np.random.default_rng(552)
    vector = rng.normal(size=op.size)

    def linear(*args, **kwargs):
        precondition = kwargs['M']
        positive, negative = precondition @ vector, precondition @ (-vector)
        np.testing.assert_allclose(negative, -positive, rtol=2e-14, atol=2e-14)
        np.testing.assert_array_equal(precondition @ vector, positive)
        applications.append(positive)
        return original_gmres(*args, **kwargs)

    monkeypatch.setattr(driver, 'gmres', linear)
    solver = driver.CoupledMomentEigenSolver(op)
    reference = solver.solve(preconditioner=mode, group_ilu_fill_factor=10)
    reference_applications = applications[:]
    applications.clear()
    events = []
    result = solver.solve(preconditioner=mode, group_ilu_fill_factor=10,
        group_ilu_storage='packed_cpu', group_ilu_setup_callback=events.append)
    assert result.converged and reference.converged
    assert result.reason == reference.reason == 'converged'
    assert all(row['linear_residual'] <= 1e-11 for row in result.history)
    assert result.full_residual <= 1e-9
    np.testing.assert_allclose(result.k_eff, reference.k_eff, rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(result.moments, reference.moments, rtol=2e-11, atol=2e-13)
    for application in applications:
        np.testing.assert_allclose(application, reference_applications[0], rtol=2e-13, atol=2e-13)
    rhs = op.fission_action(result.moments) / result.k_eff
    assert np.linalg.norm(op.action(result.moments)-rhs) / np.linalg.norm(rhs) <= 1e-9
    metadata = result.provenance['preconditioner_metadata']
    assert metadata['group_ilu_storage'] == result.provenance['group_ilu_storage'] == 'packed_cpu'
    assert 'packed_superlu.py' in result.provenance['implementation_sha256']
    assert 'packed_superlu.py' not in reference.provenance['implementation_sha256']
    assert not result.provenance['fallback']
    assert [e['phase'] for e in events] == ['before_extraction', 'before_factorization', 'after_factorization'] * op.n_groups
    for row in metadata['groups']:
        assert row['superlu_nnz'] > 0 and row['packed_nnz'] > 0
        assert row['packed_payload_nbytes'] > 0 and row['packing_seconds'] >= 0
        assert row['setup_seconds'] >= row['packing_seconds']
    if mode == 'group_ilu_coarse':
        original = reference.provenance['preconditioner_metadata']['coarse']
        for key in ('aggregate_sha256', 'coarse_sha256', 'coarse_dimension', 'aggregate_count'):
            assert metadata['coarse'][key] == original[key]


@pytest.mark.usefixtures("packed_backend")
def test_small_native_sp5_matches_independent_dense_pencil():
    op = operator(True)
    A, F = dense_pencil(orthogonal(op.grid.mesh), op.materials)
    values, vectors = eig(F, A)
    index = np.argmax(values.real)
    expected = vectors[:, index].real
    expected /= sum(v*(m.nu_sigma_f @ p) for v, m, p in
                    zip(op.volumes, op.materials, op.scalar_flux(expected)))
    result = driver.CoupledMomentEigenSolver(op).solve(
        preconditioner='group_ilu', group_ilu_storage='packed_cpu')
    assert result.converged
    assert result.k_eff == pytest.approx(values[index].real, rel=2e-9)
    np.testing.assert_allclose(result.moments.ravel(), expected, rtol=2e-8, atol=1e-10)


def test_default_and_explicit_superlu_are_bitwise_identical_without_packing(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Default SuperLU must not check or construct packed factors')
    monkeypatch.setattr(packed_module, '_check_backend', forbidden)
    monkeypatch.setattr(PackedSuperLU, 'from_superlu', forbidden)
    op = operator(True)
    solver = driver.CoupledMomentEigenSolver(op)
    default = solver.solve(preconditioner='group_ilu')
    explicit = solver.solve(preconditioner='group_ilu', group_ilu_storage='superlu')
    assert default.converged and explicit.converged
    assert default.k_eff == explicit.k_eff and default.full_residual == explicit.full_residual
    assert default.history == explicit.history
    np.testing.assert_array_equal(default.moments, explicit.moments)
    assert default.provenance['group_ilu_storage'] == 'superlu'
    assert all('packed_nnz' not in s for s in default.provenance['preconditioner_metadata']['groups'])


@pytest.mark.usefixtures("packed_backend")
def test_original_factors_released_sequentially_and_metadata_keeps_counts(monkeypatch):
    op = operator(True)
    original_spilu, original_pack = driver.spilu, PackedSuperLU.from_superlu
    references, events, reports = [], [], []

    class OriginalFactor:
        def __init__(self, factor):
            self.factor = factor
            self.nnz = int(factor.nnz) + 10000  # Distinguish capacity from packed slots.

    def factor(*args, **kwargs):
        assert all(reference() is None for reference in references)
        assert kwargs == dict(drop_tol=1e-6, fill_factor=10)
        result = OriginalFactor(original_spilu(*args, **kwargs))
        references.append(weakref.ref(result))
        reports.append(result.nnz)
        return result

    def pack(cls, source):
        return original_pack(source.factor)

    def callback(payload):
        # Events contain statistics only and cannot keep factors alive.
        events.append(json.loads(json.dumps(payload, allow_nan=False)))
        if payload['phase'] == 'after_factorization':
            assert all(reference() is None for reference in references)
            payload['current_group_stats']['packed_nnz'] = -1
            payload['completed_groups'][-1]['packed_payload_nbytes'] = -1

    monkeypatch.setattr(driver, 'spilu', factor)
    monkeypatch.setattr(PackedSuperLU, 'from_superlu', classmethod(pack))
    result = driver.CoupledMomentEigenSolver(op).solve(preconditioner='group_ilu',
        group_ilu_storage='packed_cpu', group_ilu_fill_factor=10,
        group_ilu_setup_callback=callback)
    assert result.converged and len(references) == op.n_groups
    assert all(reference() is None for reference in references)
    stats = result.provenance['preconditioner_metadata']['groups']
    assert [row['superlu_nnz'] for row in stats] == reports
    assert all(row['packed_nnz'] < row['superlu_nnz'] and row['packed_payload_nbytes'] > 0 for row in stats)


@pytest.mark.parametrize('value', ['', 'packed', None, False, 1])
def test_invalid_storage_is_rejected_before_setup(value, monkeypatch):
    op = operator(True)
    monkeypatch.setattr(driver, 'spilu', lambda *a, **k: pytest.fail('unexpected setup'))
    with pytest.raises(ValueError, match='group_ilu_storage'):
        driver.CoupledMomentEigenSolver(op).solve(preconditioner='group_ilu', group_ilu_storage=value)


@pytest.mark.parametrize('mode', ['none', 'ilu'])
def test_packed_storage_requires_group_mode(mode):
    with pytest.raises(ValueError, match='requires group_ilu'):
        driver.CoupledMomentEigenSolver(operator(True)).solve(preconditioner=mode, group_ilu_storage='packed_cpu')


def test_unsupported_backend_fails_before_operator_actions_or_assembly(monkeypatch):
    op = operator(True)
    cause = RuntimeError('unsupported private ABI')
    def fail_backend():
        raise cause
    def forbidden(*args, **kwargs):
        pytest.fail('Unsupported backend must fail before operator setup')
    monkeypatch.setattr(packed_module, '_check_backend', fail_backend)
    monkeypatch.setattr(op, 'scalar_flux', forbidden)
    monkeypatch.setattr(op, 'fission_action', forbidden)
    monkeypatch.setattr(driver, 'spilu', forbidden)
    with pytest.raises(RuntimeError, match='backend unavailable; no solver fallback') as caught:
        driver.CoupledMomentEigenSolver(op).solve(preconditioner='group_ilu', group_ilu_storage='packed_cpu')
    assert caught.value.__cause__ is cause


@pytest.mark.parametrize('exception', [MemoryError, RuntimeError, ValueError, TypeError])
@pytest.mark.usefixtures("packed_backend")
def test_packing_failure_preserves_original_cause_stats_and_callback_phase(exception, monkeypatch):
    op = operator(True)
    original = PackedSuperLU.from_superlu
    cause = exception('injected third-group packing failure')
    calls, events = [], []
    def pack(cls, source):
        calls.append(True)
        if len(calls) == 3:
            raise cause
        return original(source)
    def callback(payload):
        events.append(json.loads(json.dumps(payload, allow_nan=False)))
        if payload['phase'] == 'failure':
            raise OSError('failure log unavailable')
    monkeypatch.setattr(PackedSuperLU, 'from_superlu', classmethod(pack))
    monkeypatch.setattr(driver, 'gmres', lambda *a, **k: pytest.fail('No GMRES after packing failure'))
    expected = MemoryError if exception is MemoryError else RuntimeError
    with pytest.raises(expected, match='packing failed for group 2.*no solver fallback') as caught:
        driver.CoupledMomentEigenSolver(op).solve(preconditioner='group_ilu',
            group_ilu_storage='packed_cpu', group_ilu_setup_callback=callback)
    assert caught.value.__cause__ is cause
    assert caught.value.callback_error == 'failure log unavailable'
    assert len(calls) == 3 and len(caught.value.completed_groups) == 2
    failure = events[-1]
    assert failure['phase'] == 'failure' and failure['failed_phase'] == 'packing'
    assert failure['error_type'] == exception.__name__ and failure['error'] == str(cause)
    assert failure['current_group_stats']['superlu_nnz'] > 0
    assert all(row['packed_nnz'] > 0 for row in failure['completed_groups'])
