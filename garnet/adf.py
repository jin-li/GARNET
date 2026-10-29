"""Interior-face assembly discontinuity factors for Cartesian diffusion."""

import numpy as np
from .face_conventions import CARTESIAN_AXIS_FACE_PAIRS


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
    for axis, (minus, plus) in enumerate(CARTESIAN_AXIS_FACE_PAIRS):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        interface = regions[lo] != regions[hi]
        result.append((np.where(interface, factors[(slice(None), plus, *lo)], 1.0),
                       np.where(interface, factors[(slice(None), minus, *hi)], 1.0)))
    return tuple(result)


def expand_region_side_factors(regions, factors):
    """Expand per-region side factors to the cell array of ``assembly_face_factors``.

    ``factors`` is shaped (G, R, 6): one factor per energy group, homogenized
    region and side (x-/x+/y-/y+/z-/z+), e.g. from an isolated-assembly
    library. ``regions`` is the 3-D integer region map with values in [0, R).
    Returns the (G, 6, nx, ny, nz) array, every cell carrying its region's
    factors.
    """
    regions = np.asarray(regions)
    factors = _real_positive_factors(np, factors, dtype=float)
    if regions.ndim != 3 or not np.issubdtype(regions.dtype, np.integer):
        raise ValueError("regions must be a 3-D integer array")
    if factors.ndim != 3 or factors.shape[2] != 6:
        raise ValueError("factors must have shape (G, R, 6)")
    if regions.min() < 0 or regions.max() >= factors.shape[1]:
        raise ValueError("regions must index the factors' region axis")
    return np.moveaxis(factors[:, regions, :], -1, 1)


def region_side_traces(ops, flux, regions):
    """Area-averaged surface flux and net current on region interfaces.

    ``ops`` holds one Cartesian ``GroupOperator`` per group and ``flux`` the
    matching (G, nx, ny, nz) solution. All group operators must share the
    same active-cell mask. On each interior face between two
    different regions, the trace on each side comes from
    ``GroupOperator.face_traces``. For a solution without discontinuity
    factors, the two sides agree and give the heterogeneous surface flux.

    Returns ``(trace, current, area)`` as NumPy arrays shaped (G, R, 6),
    (G, R, 6) and (R, 6), in x-/x+/y-/y+/z-/z+ side order. ``current`` is the
    integrated net current leaving the region through that side. Sides with
    no interior interface (exterior or masked) are NaN in ``trace`` and zero
    in ``current`` and ``area``. All interface faces on one side of a region
    are pooled, so the result is a single face average only for box-shaped
    regions such as assemblies.
    """
    from .backend import asnumpy

    regions = np.asarray(regions)
    n_regions = int(regions.max()) + 1
    groups = len(ops)
    trace_sum = np.zeros((groups, n_regions, 6))
    current = np.zeros((groups, n_regions, 6))
    area = np.zeros((n_regions, 6))
    spacing = ops[0].spacing
    for axis in range(3):
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        ra, rb = regions[lo], regions[hi]
        # Masked faces have zero coupling even when their region labels differ.
        # Use the shared operator geometry rather than counting their cell traces.
        coupling = (ops[0].wx, ops[0].wy, ops[0].wz)[axis]
        interface = (ra != rb) & (asnumpy(coupling) != 0)
        ra, rb = ra[interface], rb[interface]
        face_area = float(np.prod([spacing[a] for a in range(3) if a != axis]))
        np.add.at(area, (ra, 2 * axis + 1), face_area)
        np.add.at(area, (rb, 2 * axis), face_area)
        for g, op in enumerate(ops):
            t_lo, t_hi = (asnumpy(t)[interface] for t in op.face_traces(flux[g])[axis])
            j = asnumpy(op.face_currents(flux[g])[axis])[interface] * face_area
            np.add.at(trace_sum[g], (ra, 2 * axis + 1), t_lo * face_area)
            np.add.at(trace_sum[g], (rb, 2 * axis), t_hi * face_area)
            np.add.at(current[g], (ra, 2 * axis + 1), j)
            np.add.at(current[g], (rb, 2 * axis), -j)
    with np.errstate(invalid="ignore", divide="ignore"):
        trace = np.where(area > 0, trace_sum / area, np.nan)
    return trace, current, area
