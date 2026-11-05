"""Tiny independent streaming/preconditioner algebra; no physical-case solves."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.sparse import csc_matrix

import garnet.native_angular_sweep as module
from garnet.native_angular_source import _readonly
from garnet.native_angular_p0 import P0GalerkinTransport
from garnet.native_angular_sweep import (
    StreamingBlockP0Transport, build_streaming_block, factor_streaming_block, same_csc,
)
from test_native_angular_p0 import bound, dense_oracle


def dense_block(stream, angle, group):
    """Physical cell-face equations, independently assembled as a dense matrix."""
    cells = stream.shape[0]
    matrix = np.diag([stream.area[c]*stream.source.materials[mid].total[group]
                      for c, mid in enumerate(stream.source.cell_material_ids)])
    direction = stream.quadrature.directions[angle]
    for (left, right), area, normal in zip(stream.owners, stream.measures, stream.normals):
        transfer = area*sum(direction[k]*normal[k] for k in (0, 1))
        if transfer > 0:
            matrix[left, left] += transfer
            matrix[right, left] -= transfer
        elif transfer < 0:
            matrix[left, right] += transfer
            matrix[right, right] -= transfer
    for owner, area, normal in zip(stream.boundary_owners, stream.boundary_measures, stream.boundary_normals):
        transfer = area*sum(direction[k]*normal[k] for k in (0, 1))
        if transfer > 0:
            matrix[owner, owner] += transfer
    return matrix


@pytest.mark.parametrize('boundary', ['vacuum', 'reflective', 'mixed'])
def test_independent_L0_original_outflow_and_reflection_difference(boundary):
    stream, bc = bound(lmax=0, bc_mode=boundary)
    # Exact axis directions explicitly exercise tangential boundary faces.
    object.__setattr__(stream.quadrature, 'directions', _readonly([[1., 0., 0.], [-1., 0., 0.]]))
    op = StreamingBlockP0Transport(stream, bc)
    nc, na, ng = stream.shape
    a, _, _, _, _ = dense_oracle(stream, bc)
    vacuum_a, _, _, _, _ = dense_oracle(stream, ('vacuum',)*len(bc))
    reflection = np.zeros_like(a)
    for spec, owner, area, normal in zip(bc, stream.boundary_owners, stream.boundary_measures, stream.boundary_normals):
        for angle, direction in enumerate(stream.quadrature.directions):
            transfer = area*np.dot(direction[:2], normal)
            if spec == 'reflective' and transfer < 0:
                target = direction.copy()
                target[:2] -= 2*np.dot(direction[:2], normal)*normal
                incoming = np.flatnonzero(np.all(stream.quadrature.directions == target, axis=1))
                assert len(incoming) == 1
                for group in range(ng):
                    reflection[(owner*na+angle)*ng+group, (owner*na+incoming[0])*ng+group] += transfer
    np.testing.assert_allclose(a-vacuum_a, reflection, atol=2e-15, rtol=0)
    for angle in range(na):
        for group in range(ng):
            matrix = build_streaming_block(stream, angle, group)
            np.testing.assert_allclose(matrix.toarray(), dense_block(stream, angle, group), atol=0, rtol=0)
            np.testing.assert_allclose(matrix.diagonal(), op.diagonal[:, angle, group], atol=0, rtol=0)
            with pytest.raises(ValueError):
                matrix.data.setflags(write=True)
    assert any(np.dot(stream.quadrature.directions[0, :2], n) == 0 for n in stream.boundary_normals)


def test_single_factor_helper_dense_inverse_residual_and_immutable_snapshot():
    stream, _ = bound(shear=.4, bc_mode='vacuum')
    matrix = build_streaming_block(stream, 2, 1)
    original = matrix.toarray().copy()
    block = factor_streaming_block(matrix)
    rhs = np.array([2., -7.])
    np.testing.assert_allclose(block.solve(rhs), np.linalg.solve(dense_block(stream, 2, 1), rhs), atol=2e-15)
    assert block.provenance['verification_relative_residual'] <= 1e-10
    assert block.provenance['external_hard_memory_cap_required']
    assert not block.provenance['opaque_superlu_workspace_bounded']
    assert block.provenance['retained_charge_bytes'] == (
        block.provenance['matrix_array_bytes']+2*block.provenance['materialized_lu_array_bytes']
        + block.provenance['permutation_array_bytes'])
    with pytest.raises(TypeError):
        block.provenance['fallback'] = True
    exposed = block.matrix
    with pytest.raises(ValueError):
        exposed.data.setflags(write=True)
    exposed.data = np.zeros_like(exposed.data)
    matrix.data = np.zeros_like(matrix.data)
    np.testing.assert_array_equal(block.matrix.toarray(), original)
    np.testing.assert_allclose(block.solve(rhs), np.linalg.solve(original, rhs), atol=2e-15)


def test_native_angle_group_packing_and_complete_dense_preconditioner_formula():
    stream, bc = bound(bc_mode='reflective')
    op = StreamingBlockP0Transport(stream, bc, ilu_drop_tolerance=1e-12)
    a, _, restrict, prolong, _ = dense_oracle(stream, bc)
    rng = np.random.default_rng(729)
    rhs = rng.normal(size=stream.shape)
    coarse = restrict @ a @ prolong
    z = prolong @ np.linalg.solve(coarse, restrict @ rhs.ravel())
    defect = (rhs.ravel()-a @ z).reshape(stream.shape)
    expected = z.reshape(stream.shape).copy()
    inverse = np.empty_like(rhs)
    for angle in range(stream.shape[1]):
        for group in range(stream.shape[2]):
            matrix = dense_block(stream, angle, group)
            inverse[:, angle, group] = np.linalg.solve(matrix, rhs[:, angle, group])
            expected[:, angle, group] += np.linalg.solve(matrix, defect[:, angle, group])
    np.testing.assert_allclose(op.streaming_inverse(rhs), inverse, atol=2e-15)
    np.testing.assert_allclose(op.precondition(rhs), expected, atol=8e-15, rtol=2e-14)


def test_fixed_linearity_repeatability_and_unchanged_A_F_and_coarse_matrix():
    stream, bc = bound()
    old = P0GalerkinTransport(stream, bc)
    factor = module.splu
    with patch.object(module, 'splu', wraps=factor) as count:
        op = StreamingBlockP0Transport(stream, bc)
        rng = np.random.default_rng(91)
        x, y = rng.normal(size=stream.shape), rng.normal(size=stream.shape)
        first = op.precondition(x)
        other = op.precondition(y)
        np.testing.assert_array_equal(first, op.precondition(x))
        np.testing.assert_allclose(op.precondition(2*x-.3*y), 2*first-.3*other, atol=4e-15, rtol=2e-14)
        assert count.call_count == len(op._streaming_blocks)
    assert op.action(x).tobytes() == old.action(x).tobytes()
    assert op.fission(x).tobytes() == old.fission(x).tobytes()
    np.testing.assert_array_equal(op.coarse_matrix.toarray(), old.coarse_matrix.toarray())
    config = op.preconditioner_provenance()['streaming_blocks']
    assert config['required_block_count'] == stream.shape[1]*stream.shape[2]
    assert config['reflected_coupling_retained_in_A']
    assert not config['scattering_in_L0'] and not config['fission_in_L0']
    assert config['base_p0_memory_estimate_bytes'] == old.memory_estimate_bytes
    assert config['retained_unique_charge_bytes'] == sum(b.provenance['retained_charge_bytes'] for b in op._streaming_blocks)
    assert op.memory_estimate_bytes >= old.memory_estimate_bytes+config['retained_unique_charge_bytes']


def test_exact_CSC_cache_reuse_and_hash_collisions_never_share_nonidentical_blocks():
    stream, bc = bound()
    # A deliberately constant candidate hash exercises the complete equality
    # comparison even for different energy totals and different face transfers.
    with patch.object(module, '_matrix_sha256', return_value='synthetic_collision'):
        op = StreamingBlockP0Transport(stream, bc)
    matrices = [dense_block(stream, n, g) for n in range(stream.shape[1]) for g in range(stream.shape[2])]
    indices = [index for row in op._streaming_block_map for index in row]
    for left in range(len(matrices)):
        for right in range(len(matrices)):
            assert (indices[left] == indices[right]) == np.array_equal(matrices[left], matrices[right])
    assert len(set(indices)) < len(indices)  # opposite z ordinates have identical L0
    assert op._streaming_block_map[0][0] != op._streaming_block_map[0][1]
    a = csc_matrix([[2., 0.], [-1., 3.]])
    b = a.copy()
    assert same_csc(a, b)
    b.data[0] = np.nextafter(b.data[0], np.inf)
    assert not same_csc(a, b)
    assert not same_csc(a, csc_matrix([[2., -1.], [0., 3.]]))
    assert not same_csc(a, csc_matrix([[2., 0., 0.], [-1., 3., 0.], [0., 0., 1.]]))


def test_explicit_budget_guards_before_assembly_and_factor_and_after_measured_fill():
    stream, bc = bound()
    with patch.object(type(stream), 'assembled') as assembled:
        with pytest.raises(ValueError, match='allocation estimate'):
            build_streaming_block(stream, 0, 0, max_bytes=1)
        assembled.assert_not_called()
    matrix = build_streaming_block(stream, 0, 0)
    with patch.object(module, 'splu') as factor:
        with pytest.raises(ValueError, match='allocation estimate'):
            factor_streaming_block(matrix, max_bytes=1)
        with pytest.raises(ValueError, match='allocation estimate'):
            factor_streaming_block(matrix, max_bytes=100_000, reserved_bytes=100_000)
        factor.assert_not_called()
    original_bytes = module._sparse_bytes
    with patch.object(module, '_sparse_bytes', side_effect=lambda value: original_bytes(value) if value is matrix else 2**40):
        with pytest.raises(ValueError, match='allocation estimate'):
            factor_streaming_block(matrix)
    old = P0GalerkinTransport(stream, bc)
    with patch.object(module, 'build_streaming_block') as build:
        with pytest.raises(ValueError, match='allocation estimate'):
            StreamingBlockP0Transport(stream, bc, max_bytes=old.memory_estimate_bytes+1)
        build.assert_not_called()


def test_factor_failure_known_solve_failure_and_bad_application_fail_closed():
    stream, _ = bound()
    matrix = build_streaming_block(stream, 0, 0)
    with patch.object(module, 'splu', side_effect=RuntimeError('synthetic singular')) as factor:
        with pytest.raises(ValueError, match='without fallback'):
            factor_streaming_block(matrix)
        assert factor.call_count == 1
    with pytest.raises(ValueError, match='without fallback'):
        factor_streaming_block(csc_matrix([[1., 1.], [1., 1.]]))
    real = module.splu(matrix)
    bad = SimpleNamespace(L=real.L, U=real.U, perm_r=real.perm_r, perm_c=real.perm_c,
                          solve=lambda rhs: np.zeros_like(rhs))
    with patch.object(module, 'splu', return_value=bad):
        with pytest.raises(ValueError, match='verification residual'):
            factor_streaming_block(matrix)
    block = factor_streaming_block(matrix)
    object.__setattr__(block, '_factor', SimpleNamespace(solve=lambda rhs: np.full_like(rhs, np.nan)))
    with pytest.raises(ValueError, match='nonfinite'):
        block.solve(np.ones(matrix.shape[0]))


@pytest.mark.parametrize('matrix', [csc_matrix([[0., 0.], [0., 0.]]),
                                   csc_matrix([[np.nan, 0.], [0., 1.]]),
                                   csc_matrix(np.ones((2, 3)))])
def test_invalid_or_singular_blocks_rejected(matrix):
    with pytest.raises(ValueError):
        factor_streaming_block(matrix)
