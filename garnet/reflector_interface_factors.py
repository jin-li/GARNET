"""Explicit positive-trace extension of a frozen fuel-only ADF prescription.

This operation qualifies ratios on internal parent faces only. It does not
qualify an auxiliary reference field, the transport data, or a receiving core.
In particular, negative unused auxiliary traces are retained by the caller;
they are never clipped or used to modify exterior boundary conditions.
"""
import numpy as np


def extend_reflector_interface_factors(fuel_factors, fuel_mask, neighbors,
                                     heterogeneous_faces, homogeneous_faces,
                                     face_change_over_mean, factor_relative_change,
                                     *, order_tolerance):
    """Return frozen fuel factors plus reflector-owned internal HET/HOM ratios.

    Face ordering is x-, y-, x+, y+, z-, z+. Arrays of face quantities have
    shape (N,6,G). ``face_change_over_mean`` is the nonnegative maximum change
    over the prescribed reference-order comparisons, normalized by each native
    group mean, at each face. Its provenance is the caller's responsibility.
    ``factor_relative_change`` independently bounds the maximum relative factor
    change over those comparisons (using the final factor as denominator).
    ``neighbors`` is a reciprocal (N,6) integer array, -1 at exterior faces.

    Requires a finite positive float64 fuel-only library, unity on reflector
    and exterior faces. Every internal reflector trace must be finite positive
    and order-converged. Exterior HET/HOM values may be negative or zero but must
    remain finite. All exterior factors stay unity by explicit prescription.
    No whole-field positivity or exact reference preservation claim is made.
    """
    base = np.asarray(fuel_factors)
    fuel = np.asarray(fuel_mask)
    adjacent = np.asarray(neighbors)
    if (base.dtype != np.dtype('float64') or base.ndim != 3
            or base.shape[1] != 6 or min(base.shape[0], base.shape[2]) < 1):
        raise ValueError('fuel factors must have float64 shape (N,6,G)')
    n = base.shape[0]
    if fuel.dtype.kind != 'b' or fuel.shape != (n,):
        raise ValueError('fuel_mask must be a boolean array of shape (N,)')
    if (adjacent.dtype.kind not in 'iu' or adjacent.shape != (n, 6)
            or np.any(adjacent < -1) or np.any(adjacent >= n)):
        raise ValueError('neighbors must be integer (N,6) indices or -1')
    opposite = (2, 3, 0, 1, 5, 4)
    internal = adjacent >= 0
    for node, side in np.argwhere(internal):
        other = adjacent[node, side]
        if other == node or adjacent[other, opposite[side]] != node:
            raise ValueError('neighbors must be reciprocal without self faces')
    if (not np.all(np.isfinite(base)) or np.any(base <= 0)
            or np.any(base[~fuel] != 1) or np.any(base[~internal] != 1)):
        raise ValueError('positive fuel-only baseline with unity reflector/exterior required')
    selected = internal & ~fuel[:, None]
    if not np.any(selected):
        raise ValueError('at least one internal reflector face required')
    het, hom, changes, relative = [np.asarray(a, dtype=float) for a in
                         (heterogeneous_faces, homogeneous_faces, face_change_over_mean,
                          factor_relative_change)]
    if any(a.shape != base.shape or not np.all(np.isfinite(a))
           for a in (het, hom, changes, relative)):
        raise ValueError('finite face arrays must match baseline shape')
    if (np.ndim(order_tolerance) != 0 or isinstance(order_tolerance, (bool, np.bool_))
            or not np.isfinite(order_tolerance) or order_tolerance <= 0
            or np.any(changes < 0) or np.any(relative < 0)):
        raise ValueError('positive finite order tolerance and nonnegative changes required')
    if (np.any(het[selected] <= 0) or np.any(hom[selected] <= 0)
            or np.any(changes[selected] > order_tolerance)
            or np.any(relative[selected] > order_tolerance)):
        raise ValueError('every internal reflector trace must be positive and order-converged')
    result = base.copy()
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        result[selected] = het[selected]/hom[selected]
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError('reflector ratios must be representable finite positive values')
    result.setflags(write=False)
    return result
