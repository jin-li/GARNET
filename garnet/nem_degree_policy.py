"""Existing polynomial-degree and local reference qualification policies.

These finite choices describe the current implementation, not a guarantee of
convergence or permission to substitute a different receiving degree.
"""
import numpy as np

SUPPORTED_NEM_DEGREES = (4, 6, 8, 10, 12, 14, 16)
DEFAULT_REFERENCE_DEGREES = (12, 14, 16)
DEFAULT_REFERENCE_ORDER_TOLERANCE = .005


def validate_nem_degree(degree, *, name="degree"):
    """Return a supported degree as int; retain acceptance of integral floats."""
    if (isinstance(degree, (bool, np.bool_)) or not np.isscalar(degree)
            or degree not in SUPPORTED_NEM_DEGREES):
        raise ValueError(f"{name} must be one of 4,6,8,10,12,14,16")
    return int(degree)


def validate_reference_degrees(degrees, receiving_degree):
    """Require exactly three increasing levels ending at the receiver.

    The qualification uses both successive changes and conservative bounds
    derived from all three levels. Two levels or a different final receiver
    would change that prescription rather than merely tune convergence.
    """
    message = "three increasing reference degrees ending at receiving degree required"
    try:
        levels = tuple(validate_nem_degree(value) for value in degrees)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if (len(levels) != 3 or not levels[0] < levels[1] < levels[2]
            or levels[-1] != receiving_degree):
        raise ValueError(message)
    return levels
