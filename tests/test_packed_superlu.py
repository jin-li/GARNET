"""Small direct numerical/ownership checks for the experimental CPU adapter."""

import gc
import sys

import numpy as np
import pytest
from scipy.sparse import csc_matrix, diags, random as sparse_random
from scipy.sparse.linalg import spilu, splu

from garnet.packed_superlu import PackedSuperLU


def _matrix(dtype):
    values = np.array([[0., 2, 0, 1], [3, 0, 4, 0],
                       [1, 0, 0, 1], [0, 3, 1, 0]], dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        values *= 1 + .2j
        values[0, 3] += .1j
    return csc_matrix(values)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("factory", [splu, spilu])
def test_signed_multiple_repeated_rhs_match_source(dtype, factory):
    factor = factory(_matrix(dtype))
    n = factor.shape[0]
    assert np.any(factor.perm_r != np.arange(n))
    assert np.any(factor.perm_c != np.arange(n))
    source_arrays = [factor.L.data, factor.L.indices, factor.L.indptr,
                     factor.U.data, factor.U.indices, factor.U.indptr,
                     factor.perm_r, factor.perm_c]
    snapshots = [value.copy() for value in source_arrays]
    reference_count = sys.getrefcount(factor)
    packed = PackedSuperLU.from_superlu(factor)
    assert sys.getrefcount(factor) == reference_count
    assert packed.nnz == factor.L.nnz + factor.U.nnz
    assert packed.payload_nbytes == sum(a.nbytes for a in packed.owned_arrays)
    for value in packed.owned_arrays:
        assert value.flags.owndata and value.base is None
        assert not value.flags.writeable
        assert all(not np.shares_memory(value, original) for original in source_arrays)
    stored = [a.copy() for a in packed.owned_arrays]
    rhs = np.arange(1, n + 1, dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        rhs += np.array([1j, -2j, .5j, -1j], dtype=dtype)
    tolerance = 3e-6 if np.dtype(dtype).itemsize <= (8 if np.iscomplexobj(rhs) else 4) else 3e-14
    for candidate in (rhs, -rhs, 2.5 * rhs, np.zeros_like(rhs),
                      np.column_stack((rhs, -2 * rhs, rhs[::-1])),
                      np.column_stack((rhs, -rhs))[:, ::-1]):
        before = candidate.copy()
        expected = factor.solve(candidate)
        for _ in range(3):
            actual = packed.solve(candidate)
            assert actual.shape == candidate.shape
            assert actual.dtype == np.dtype(dtype)
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
            np.testing.assert_array_equal(candidate, before)
    for value, before in zip(source_arrays, snapshots):
        np.testing.assert_array_equal(value, before)
    for value, before in zip(packed.owned_arrays, stored):
        np.testing.assert_array_equal(value, before)


def test_random_sparse_factor_and_original_release():
    rng = np.random.default_rng(814)
    matrix = sparse_random(71, 71, density=.09, random_state=rng, format="csc")
    matrix = matrix + diags(np.full(71, 2.))
    factor = splu(matrix)
    rhs = rng.normal(size=(71, 5))
    expected = factor.solve(rhs)
    packed = PackedSuperLU.from_superlu(factor)
    # The result also survives mutation of every source numeric factor value.
    factor.L.data[:] = 9
    factor.U.data[:] = -4
    del factor
    gc.collect()
    actual = packed.solve(rhs)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(matrix @ actual, rhs, rtol=2e-14, atol=2e-14)


def test_repeated_calls_pass_the_same_sparse_arrays(monkeypatch):
    from garnet import packed_superlu as module
    packed = PackedSuperLU.from_superlu(splu(_matrix(np.float64)))
    original = module._superlu.gstrs
    seen = []

    def record(*args):
        assert all(a is b for a, b in zip(args[3:6], packed._lower))
        assert all(a is b for a, b in zip(args[8:11], packed._upper))
        seen.append(True)
        return original(*args)

    monkeypatch.setattr(module._superlu, "gstrs", record)
    for _ in range(4):
        packed.solve(np.ones(4))
    assert len(seen) == 4


def test_explicit_backend_and_version_guards(monkeypatch):
    from garnet import packed_superlu as module
    factor = splu(_matrix(np.float64))
    packed = PackedSuperLU.from_superlu(factor)
    with pytest.raises(TypeError, match="CPU SuperLU"):
        PackedSuperLU.from_superlu(object())
    monkeypatch.setattr(module.scipy, "__version__", "1.18.2")
    with pytest.raises(RuntimeError, match="No fallback"):
        PackedSuperLU.from_superlu(factor)
    with pytest.raises(RuntimeError, match="No fallback"):
        packed.solve(np.ones(4))


@pytest.mark.parametrize("which, value, message", [
    ("L", np.inf, "finite"), ("U", np.nan, "finite"),
    ("L", 2., "unit lower"), ("U", 0., "nonzero upper"),
    ("U", np.nextafter(0., 1.), "inverse upper"),
])
def test_invalid_factors_are_rejected(which, value, message):
    factor = splu(csc_matrix(np.eye(3)))
    getattr(factor, which).data[0] = value
    with pytest.raises(ValueError, match=message):
        PackedSuperLU.from_superlu(factor)


def test_rhs_guards_and_empty_multiple_rhs(monkeypatch):
    from garnet import packed_superlu as module
    packed = PackedSuperLU.from_superlu(splu(_matrix(np.float64)))
    for rhs in (np.ones(3), np.ones((4, 1, 1)), np.array(1.)):
        with pytest.raises(ValueError, match="shape"):
            packed.solve(rhs)
    with pytest.raises(TypeError, match="NumPy CPU backend"):
        packed.solve([1., 2., 3., 4.])
    with pytest.raises(TypeError, match="safely cast"):
        packed.solve(np.ones(4, dtype=complex))
    with pytest.raises(ValueError, match="finite"):
        packed.solve(np.full(4, np.inf))
    with pytest.raises(NotImplementedError, match="trans='N'"):
        packed.solve(np.ones(4), trans="T")
    assert packed.solve(np.empty((4, 0))).shape == (4, 0)
    monkeypatch.setattr(module._superlu, "gstrs", lambda *args: (args[-1], 1))
    with pytest.raises(np.linalg.LinAlgError, match="info=1"):
        packed.solve(np.ones(4))


@pytest.mark.parametrize("diagonal, off_diagonal, message", [
    (1e-300, 1e300, "finite"), (1e300, 1e-300, "underflowed"),
])
def test_unrepresentable_upper_encoding_is_rejected(diagonal, off_diagonal, message):
    factor = splu(csc_matrix([[1., 1.], [0., 1.]]), permc_spec="NATURAL")
    upper = factor.U
    upper.data[upper.indptr[1]:upper.indptr[2]] = [off_diagonal, diagonal]
    with pytest.raises(ValueError, match=message):
        PackedSuperLU.from_superlu(factor)
