"""Asymmetric data exposes swapped axes and signs in face indexing."""
import numpy as np
import pytest

from garnet.face_conventions import (
    CARTESIAN_AXIS_FACE_PAIRS, CARTESIAN_FACE_ORDER,
    NEM_AXIS_FACE_PAIRS, NEM_FACE_ORDER, NEM_NORMAL_SIGNS,
    NEM_OPPOSITE_FACES, reorder_faces,
)
from garnet.nem_degree_policy import (
    DEFAULT_REFERENCE_DEGREES, DEFAULT_REFERENCE_ORDER_TOLERANCE,
    SUPPORTED_NEM_DEGREES, validate_nem_degree, validate_reference_degrees,
)


def test_asymmetric_face_conversion_and_roundtrip():
    # Signed values deliberately survive: this utility changes indices only.
    cartesian = np.array([[-11, 23, -37, 41, -59, 67],
                          [-71, 83, -97, 101, -113, 127]])
    nem = reorder_faces(cartesian, source_order=CARTESIAN_FACE_ORDER,
                        target_order=NEM_FACE_ORDER)
    np.testing.assert_array_equal(nem, cartesian[:, [0, 2, 1, 3, 4, 5]])
    assert nem.dtype == cartesian.dtype
    assert not np.shares_memory(nem, cartesian)
    restored = reorder_faces(nem, source_order=NEM_FACE_ORDER,
                             target_order=CARTESIAN_FACE_ORDER)
    np.testing.assert_array_equal(restored, cartesian)


@pytest.mark.parametrize('axis', [0, 1, 2, -1, -2, -3])
def test_explicit_face_axis(axis):
    shape = [2, 3, 4]
    shape[axis] = 6
    values = np.arange(np.prod(shape)).reshape(shape)
    actual = reorder_faces(values, source_order=CARTESIAN_FACE_ORDER,
                           target_order=NEM_FACE_ORDER, axis=axis)
    np.testing.assert_array_equal(actual, np.take(values, [0, 2, 1, 3, 4, 5], axis=axis))
    identity = reorder_faces(values, source_order=NEM_FACE_ORDER,
                             target_order=NEM_FACE_ORDER, axis=axis)
    np.testing.assert_array_equal(identity, values)
    assert not np.shares_memory(identity, values)


@pytest.mark.parametrize('order', [None, 'cartesian', (), ('x-',)*6,
                                    ('x-', 'x+', 'y-', 'y+', 'z-', 'r+'),
                                    ([],)*6])
@pytest.mark.parametrize('argument', ['source_order', 'target_order'])
def test_invalid_order_is_rejected(order, argument):
    orders = dict(source_order=CARTESIAN_FACE_ORDER, target_order=NEM_FACE_ORDER)
    orders[argument] = order
    with pytest.raises(ValueError, match='face order'):
        reorder_faces(np.ones(6), **orders)


@pytest.mark.parametrize('values,axis', [(1, -1), (np.ones(5), -1),
                                        (np.ones((6, 2)), 1), (np.ones(6), 1),
                                        (np.ones(6), -2), (np.ones(6), .5)])
def test_invalid_face_axis_is_rejected(values, axis):
    with pytest.raises(ValueError, match='face axis'):
        reorder_faces(values, source_order=CARTESIAN_FACE_ORDER,
                      target_order=NEM_FACE_ORDER, axis=axis)


def test_axis_pairs_opposites_and_outward_normals_agree():
    for order, pairs in [(CARTESIAN_FACE_ORDER, CARTESIAN_AXIS_FACE_PAIRS),
                         (NEM_FACE_ORDER, NEM_AXIS_FACE_PAIRS)]:
        for coordinate, (minus, plus) in zip('xyz', pairs):
            assert (order[minus], order[plus]) == (coordinate+'-', coordinate+'+')
    for side, opposite in enumerate(NEM_OPPOSITE_FACES):
        assert NEM_OPPOSITE_FACES[opposite] == side
        assert NEM_FACE_ORDER[side][0] == NEM_FACE_ORDER[opposite][0]
        assert NEM_NORMAL_SIGNS[side] == -NEM_NORMAL_SIGNS[opposite]
        assert NEM_NORMAL_SIGNS[side] == (-1 if NEM_FACE_ORDER[side][1] == '-' else 1)


@pytest.mark.parametrize('degree', SUPPORTED_NEM_DEGREES)
def test_supported_degrees_preserve_integral_float_inputs(degree):
    assert validate_nem_degree(degree) == degree
    assert validate_nem_degree(float(degree)) == degree
    assert validate_nem_degree(np.int64(degree)) == degree


@pytest.mark.parametrize('degree', [True, np.bool_(False), None, '4', 2, 5, 18,
                                   np.inf, np.nan, 4.5, [4], np.array([4])])
def test_unsupported_degrees_rejected(degree):
    with pytest.raises(ValueError, match='degree must be one of'):
        validate_nem_degree(degree)


@pytest.mark.parametrize('degrees,receiver', [((12, 14), 14), ((10, 12, 14, 16), 16),
                                           ((14, 12, 16), 16), ((12, 12, 16), 16),
                                           ((12, 14, 16), 14), ((12, True, 16), 16),
                                           ((12, 15, 16), 16), (None, 16)])
def test_reference_requires_exactly_three_increasing_levels_ending_at_receiver(degrees, receiver):
    with pytest.raises(ValueError, match='three increasing reference degrees'):
        validate_reference_degrees(degrees, receiver)


def test_reference_policy_defaults_remain_unchanged():
    assert DEFAULT_REFERENCE_DEGREES == (12, 14, 16)
    assert DEFAULT_REFERENCE_ORDER_TOLERANCE == .005
    assert validate_reference_degrees(DEFAULT_REFERENCE_DEGREES, 16) == (12, 14, 16)
    assert validate_reference_degrees((4, 6, 8), 8) == (4, 6, 8)
