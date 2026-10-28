"""Face indexing for three Cartesian axes and six whole cuboid faces.

These conventions specify array ordering only. Reordering does not convert
current signs, densities/integrals, geometry, or homogeneous reference physics.
"""
import operator

import numpy as np

CARTESIAN_FACE_ORDER = ("x-", "x+", "y-", "y+", "z-", "z+")
NEM_FACE_ORDER = ("x-", "y-", "x+", "y+", "z-", "z+")
CARTESIAN_AXIS_FACE_PAIRS = ((0, 1), (2, 3), (4, 5))
NEM_AXIS_FACE_PAIRS = ((0, 2), (1, 3), (4, 5))
NEM_OPPOSITE_FACES = (2, 3, 0, 1, 5, 4)
NEM_NORMAL_SIGNS = (-1, -1, 1, 1, -1, 1)


def _face_order(order):
    try:
        labels = tuple(order)
        valid = len(labels) == 6 and set(labels) == set(CARTESIAN_FACE_ORDER)
    except TypeError as exc:
        raise ValueError("face order must contain each of x-,x+,y-,y+,z-,z+ once") from exc
    if not valid:
        raise ValueError("face order must contain each of x-,x+,y-,y+,z-,z+ once")
    return labels


def reorder_faces(values, *, source_order, target_order, axis=-1):
    """Return a CPU NumPy copy with the specified six-face axis reordered.

    Supply explicit label sequences, typically CARTESIAN_FACE_ORDER and
    NEM_FACE_ORDER. Values and dtype are preserved; no physical conversion or
    qualification of ADF factors is implied. Other axes retain their order.
    """
    source = _face_order(source_order)
    target = _face_order(target_order)
    if hasattr(values, "__cuda_array_interface__"):
        raise ValueError("face values must be a CPU array")
    array = np.asarray(values)
    try:
        face_axis = operator.index(axis)
    except TypeError as exc:
        raise ValueError("face axis must be an integer") from exc
    if not -array.ndim <= face_axis < array.ndim:
        raise ValueError("face axis is out of range")
    if array.shape[face_axis] != 6:
        raise ValueError("face axis must have length six")
    return np.take(array, [source.index(label) for label in target], axis=face_axis)
