"""Tiny opt-in angular factor storage checks; no physical-case solves."""
import inspect
import sys

import numpy as np
import pytest
from scipy.sparse import csc_matrix

import garnet.native_angular_p0 as p0_module
import garnet.native_angular_sweep as sweep_module
import garnet.packed_superlu as packed_module
from garnet.mesh import assemble_mesh
from garnet.moment_materials import MomentMaterial
from garnet.native_angular_source import NativeAngularSource, ProductQuadrature
from garnet.native_mesh_streaming import NativeMeshStreaming
from garnet.native_angular_p0 import P0GalerkinTransport
from garnet.native_angular_sweep import (
    StreamingBlockP0Transport, build_streaming_block, factor_streaming_block, same_csc,
)
from garnet.packed_superlu import PackedSuperLU


def test_storage_is_keyword_only_and_preserves_streaming_fill_positional_slot():
    for kind in (P0GalerkinTransport, StreamingBlockP0Transport):
        parameter = inspect.signature(kind).parameters['factor_storage']
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default == 'superlu'
    positional = [name for name, parameter in inspect.signature(
        StreamingBlockP0Transport).parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert positional[-3:] == ['ilu_drop_tolerance', 'ilu_fill_factor',
                               'streaming_fill_estimate']


def fixture(boundary='mixed'):
    mesh = assemble_mesh(np.array([[0., 0.], [1., 0.], [3., 0.],
                                   [0., 1.], [1., 1.], [3., 1.]]),
                         [(0, 1, 4, 3), (1, 2, 5, 4)], [0, 1])
    materials = []
    for cell in range(2):
        sp0 = np.array([[.25, .08, .01, .02], [.12, .35, .02, .01],
                        [.01, .03, .2, .04], [.04, .01, .02, .3]])
        # Signed higher angular moments and all-group coupling; different
        # material/group totals prevent accidental identical-block reuse.
        sp1 = .002*np.array([[2., -1., 3., -2.], [-3., 2., -1., 1.],
                             [1., -2., 2., -1.], [-1., 3., -2., 2.]])
        materials.append(MomentMaterial(
            np.array([1.3, 1.7, 1.2, 1.6])-.1*cell,
            np.array([.4, .3, .2, .1]), np.array([.12, .3, .08, .1]),
            np.array([2., 5., 3., 4.]), {}, {0: sp0, 1: sp1}, {}, {}, {},
            {'orientation_contract': 'to-major',
             'energy_bounds_mev': [20., 2., 1., .1, 0.]}, {}))
    source = NativeAngularSource(ProductQuadrature(1), tuple(materials), np.array([0, 1]))
    stream = NativeMeshStreaming(mesh, source)
    bc = tuple('reflective' if boundary == 'reflective' or
               (boundary == 'mixed' and abs(n[0]) < 1e-13) else 'vacuum'
               for n in stream.boundary_normals)
    return stream, bc


def assert_payload(factor, config):
    assert isinstance(factor, PackedSuperLU)
    assert config['original_superlu_retained'] is False
    assert config['factor_backend_abi'] == 'scipy-1.18.1:_superlu.gstrs'
    assert config['packed_payload_array_bytes'] == factor.payload_nbytes
    assert len(config['packed_payload_sha256']) == 64
    assert config['packed_action_probe_count'] >= 2
    assert config['packed_action_relative_error'] <= 1e-12
    assert config['external_hard_memory_cap_required']
    for array in factor.owned_arrays:
        assert array.base is None and array.flags.owndata and not array.flags.writeable


@pytest.mark.parametrize('kind', [P0GalerkinTransport, StreamingBlockP0Transport])
@pytest.mark.parametrize('boundary', ['vacuum', 'mixed', 'reflective'])
def test_signed_four_group_actions_and_cold_tiny_rhs(kind, boundary):
    stream, bc = fixture(boundary)
    original = kind(stream, bc, inner_tolerance=1e-12)
    packed = kind(stream, bc, factor_storage='packed', inner_tolerance=1e-12)
    rhs = np.random.default_rng(871).normal(size=stream.shape)
    np.testing.assert_array_equal(packed.action(rhs), original.action(rhs))
    np.testing.assert_array_equal(packed.fission(rhs), original.fission(rhs))
    for name in ('data', 'indices', 'indptr'):
        np.testing.assert_array_equal(getattr(packed.coarse_matrix, name),
                                      getattr(original.coarse_matrix, name))
    np.testing.assert_allclose(packed.precondition(rhs), original.precondition(rhs),
                               rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(packed.precondition(-rhs), -packed.precondition(rhs),
                               rtol=1e-12, atol=1e-13)
    old_result, result = original.fixed_source(rhs), packed.fixed_source(rhs)
    assert old_result.converged and result.converged
    assert old_result.true_residual <= 1e-12 and result.true_residual <= 1e-12
    np.testing.assert_allclose(result.angular_flux, old_result.angular_flux,
                               rtol=1e-11, atol=1e-12)
    assert_payload(packed._factor, packed._p0_provenance)
    if kind is StreamingBlockP0Transport:
        assert packed._streaming_block_map == original._streaming_block_map
        for block, old in zip(packed._streaming_blocks, original._streaming_blocks):
            assert same_csc(block.matrix, old.matrix)
            assert_payload(block._factor, block.provenance)
            assert block.provenance['packed_verification_relative_residual'] <= 1e-10
            assert block.provenance['retained_charge_bytes'] == (
                block.provenance['matrix_array_bytes']+block._factor.payload_nbytes)


def test_default_preserves_action_and_provenance_without_packing(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('default must not check or construct packed backend')
    monkeypatch.setattr(packed_module, '_check_backend', forbidden)
    monkeypatch.setattr(PackedSuperLU, 'from_superlu', forbidden)
    stream, bc = fixture()
    for kind in (P0GalerkinTransport, StreamingBlockP0Transport):
        default, explicit = kind(stream, bc), kind(stream, bc, factor_storage='superlu')
        rhs = np.ones(stream.shape)
        np.testing.assert_array_equal(default.precondition(rhs), explicit.precondition(rhs))
        assert default.preconditioner_provenance() == explicit.preconditioner_provenance()
        assert 'factor_storage' not in default._p0_provenance


def test_original_superlu_objects_are_released_and_csc_is_unchanged(monkeypatch):
    factors = []
    for module, name in ((p0_module, 'spilu'), (sweep_module, 'splu')):
        original = getattr(module, name)
        def capture(*args, _original=original, **kwargs):
            factor = _original(*args, **kwargs)
            factors.append(factor)
            return factor
        monkeypatch.setattr(module, name, capture)
    stream, bc = fixture()
    packed = StreamingBlockP0Transport(stream, bc, factor_storage='packed')
    # Only this list owns each source factor after construction. SuperLU
    # objects do not support weak references.
    assert all(sys.getrefcount(factors[index]) == 2 for index in range(len(factors)))
    matrix = build_streaming_block(stream, 0, 0)
    snapshot = matrix.copy()
    block = factor_streaming_block(matrix, factor_storage='packed')
    assert same_csc(matrix, snapshot) and same_csc(block.matrix, snapshot)
    assert packed._streaming_blocks


@pytest.mark.parametrize('storage', ['bad', None, 1])
def test_storage_rejected_before_factor_allocation(storage, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('factorization must not run')
    monkeypatch.setattr(p0_module, 'spilu', forbidden)
    monkeypatch.setattr(sweep_module, 'splu', forbidden)
    stream, bc = fixture()
    for kind in (P0GalerkinTransport, StreamingBlockP0Transport):
        with pytest.raises(ValueError, match='factor_storage'):
            kind(stream, bc, factor_storage=storage)
    with pytest.raises(ValueError, match='factor_storage'):
        factor_streaming_block(csc_matrix(np.eye(2)), factor_storage=storage)


def test_unsupported_abi_fails_before_factorization(monkeypatch):
    monkeypatch.setattr(packed_module.scipy, '__version__', 'unsupported')
    def forbidden(*args, **kwargs):
        raise AssertionError('factorization must not run')
    monkeypatch.setattr(p0_module, 'spilu', forbidden)
    monkeypatch.setattr(sweep_module, 'splu', forbidden)
    stream, bc = fixture()
    with pytest.raises(RuntimeError, match='No fallback'):
        StreamingBlockP0Transport(stream, bc, factor_storage='packed')
    with pytest.raises(RuntimeError, match='No fallback'):
        factor_streaming_block(csc_matrix(np.eye(2)), factor_storage='packed')


def test_packing_budget_rejected_before_conversion(monkeypatch):
    stream, bc = fixture()
    original = P0GalerkinTransport(stream, bc)
    matrix = build_streaming_block(stream, 0, 0)
    block = factor_streaming_block(matrix, factor_storage='packed')
    def forbidden(*args, **kwargs):
        raise AssertionError('packing must not run after failed budget')
    monkeypatch.setattr(PackedSuperLU, 'from_superlu', forbidden)
    with pytest.raises(ValueError, match='allocation estimate'):
        P0GalerkinTransport(stream, bc, factor_storage='packed',
                            max_bytes=original.memory_estimate_bytes+1)
    with pytest.raises(ValueError, match='allocation estimate'):
        factor_streaming_block(matrix, factor_storage='packed',
                               max_bytes=block.provenance['packing_peak_estimate_bytes']-1)


def test_corrupt_packed_action_fails_comparison(monkeypatch):
    stream, bc = fixture()
    original = PackedSuperLU.solve
    def corrupt(self, rhs, trans='N'):
        # Pass the positive probe and fail specifically on the signed probe.
        return original(self, rhs, trans=trans)*(1.001 if np.any(rhs < 0) else 1.)
    monkeypatch.setattr(PackedSuperLU, 'solve', corrupt)
    with pytest.raises(ValueError, match='action comparison'):
        P0GalerkinTransport(stream, bc, factor_storage='packed')
    with pytest.raises(ValueError, match='action comparison'):
        factor_streaming_block(build_streaming_block(stream, 0, 0), factor_storage='packed')
