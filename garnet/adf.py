"""Interior-face assembly discontinuity factors for Cartesian diffusion."""

import numpy as np


def _real_positive_factors(xp, values, *, dtype=None):
    """Validate ADF data before any cast that could discard information."""
    factors = xp.asarray(values)
    if factors.dtype.kind not in "biuf":
        raise ValueError("ADF factors must be real numbers")
    if not bool(xp.all(xp.isfinite(factors) & (factors > 0))):
        raise ValueError("ADF factors must be finite and positive")
    if dtype is not None:
        factors = xp.asarray(factors, dtype=dtype)
        if not bool(xp.all(xp.isfinite(factors) & (factors > 0))):
            raise ValueError("ADF factors must remain finite and positive in the operator dtype")
    return factors


def assembly_face_factors(regions, side_factors):
    """Build directed face arrays from cell regions and per-side factors.

    ``regions`` is a 3-D integer map; identical values identify the same
    homogenized region, independently of material. ``side_factors`` is shaped
    (G, 6, nx, ny, nz), in x-/x+/y-/y+/z-/z+ order. Each region's physical
    face factors should be extended to its boundary-adjacent cells. All
    interior faces within one region receive unity, including mesh refinement
    faces. Every array must be provided, real, finite and positive. External and
    active-mask boundaries use the solver's ordinary boundary conditions.

    Returns three pairs of arrays shaped (G, *interior_face_shape), suitable
    for ``DiffusionEigenSolver(adf=...)``. This preprocessing runs on the CPU;
    the solver transfers the resulting arrays to its selected backend.
    """
    regions = np.asarray(regions)
    factors = _real_positive_factors(np, side_factors, dtype=float)
    if regions.ndim != 3 or not np.issubdtype(regions.dtype, np.integer):
        raise ValueError("regions must be a 3-D integer array")
    if factors.ndim != 5 or factors.shape[1:] != (6, *regions.shape) or factors.shape[0] < 1:
        raise ValueError("side_factors must have shape (G, 6, *regions.shape)")
    result = []
    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        interface = regions[lo] != regions[hi]
        result.append((np.where(interface, factors[(slice(None), 2 * axis + 1, *lo)], 1.0),
                       np.where(interface, factors[(slice(None), 2 * axis, *hi)], 1.0)))
    return tuple(result)
