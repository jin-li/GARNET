"""Auxiliary homogeneous cuboid reconstruction for frozen scalar diffusion DFs.

This is an explicit flat within-group source closure, not a coupled transport
reconstruction. Factors inferred from a reference state are not independent
predictions. Surface currents are integrated and inward-positive.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class CuboidReconstruction:
    face_flux: np.ndarray              # (W,S,E,N,bottom,top; group)
    uniform_source: np.ndarray         # density source, fixed by compatibility
    minimum_flux: np.ndarray           # continuous minimum over the cuboid
    volume: float
    face_areas: np.ndarray
    source_defect: np.ndarray | None = None


def _axis_value(xi, t, half, diffusion, inward_sum, inward_difference):
    """Zero-mean axis contribution, stable at small and large optical size."""
    if t < 1e-3:
        even = xi*xi/2 - 1/6 + t*t*(xi**4/24-xi*xi/12+7/360)
        odd = xi + t*t*(xi**3/6-xi/2) + t**4*(xi**5/120-xi**3/12+5*xi/24)
        return half/(2*diffusion)*(inward_sum*even+inward_difference*odd)
    a = abs(t*xi)
    low, high = math.exp(-a-t), math.exp(a-t)
    cosh_ratio = (high+low)/(-math.expm1(-2*t))
    sinh_ratio = math.copysign((high-low)/(1+math.exp(-2*t)), xi)
    return half/(2*diffusion*t)*(inward_sum*(cosh_ratio-1/t)+inward_difference*sinh_ratio)


def cuboid_flat_source_reconstruction(mean_flux, diffusion, removal,
                                      net_inward_current, widths, *,
                                      physical_source=None, minimum_flux_margin=0.):
    """Reconstruct homogeneous face means from mean flux and six net currents.

    ``mean_flux``, ``diffusion``, ``removal`` have shape (G,); widths are (Lx,Ly,Lz).
    ``net_inward_current`` has shape (6,G), surface-integrated, inward-positive.
    The equation is -D Laplacian(phi)+removal*phi=q with a uniform q, prescribed
    spatially uniform normal current density on each face, and the supplied volume mean. Off-group/fission
    source variation is frozen in this auxiliary problem. q follows from the
    Neumann compatibility condition; global solver sources/XS are not changed.

    If supplied, ``physical_source`` reports the discrepancy with a separately
    computed reference-state source. Assess this against tally uncertainty.
    Reject nonpositive continuous solutions rather than clipping factors. The
    optional minimum_flux_margin (scalar or G-vector in flux units) lets the
    caller require an uncertainty-qualified lower bound. Default zero checks
    nominal positivity only; it does not establish statistical qualification.
    """
    mean = np.asarray(mean_flux, dtype=float)
    D, r = np.asarray(diffusion, dtype=float), np.asarray(removal, dtype=float)
    widths = np.asarray(widths, dtype=float)
    currents = np.asarray(net_inward_current, dtype=float)
    if mean.ndim != 1 or mean.size == 0 or D.shape != mean.shape or r.shape != mean.shape:
        raise ValueError('mean_flux, diffusion and removal must have common shape (G,)')
    if widths.shape != (3,) or currents.shape != (6, mean.size):
        raise ValueError('widths must be (3,) and currents (6,G), W/S/E/N/bottom/top')
    if not all(np.all(np.isfinite(a)) for a in [mean,D,r,widths,currents]):
        raise ValueError('reconstruction inputs must be finite')
    if np.any(mean <= 0) or np.any(D <= 0) or np.any(r < 0) or np.any(widths <= 0):
        raise ValueError('positive means, diffusion and widths; nonnegative removal required')
    volume = float(np.prod(widths))
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError('cuboid volume must be finite and positive')
    areas = np.array([widths[1]*widths[2], widths[0]*widths[2],
                      widths[1]*widths[2], widths[0]*widths[2],
                      widths[0]*widths[1], widths[0]*widths[1]])
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0):
        raise ValueError('face areas must be finite and positive')
    margin = np.asarray(minimum_flux_margin,dtype=float)
    if margin.shape not in [(),mean.shape] or not np.all(np.isfinite(margin)) or np.any(margin < 0):
        raise ValueError('minimum_flux_margin must be finite nonnegative scalar or shape (G,)')
    density = currents / areas[:,None]
    faces = np.empty_like(currents)
    minimum = mean.copy()
    for axis, (minus,plus) in enumerate([(0,2),(1,3),(4,5)]):
        half = widths[axis]/2
        for g in range(mean.size):
            t = half*math.sqrt(r[g]/D[g])
            isum, idiff = density[minus,g]+density[plus,g], density[plus,g]-density[minus,g]
            pm = _axis_value(-1,t,half,D[g],isum,idiff)
            pp = _axis_value(1,t,half,D[g],isum,idiff)
            candidates = [pm,pp]
            if isum != 0:
                ratio = -idiff/isum
                arg = ratio*math.tanh(t) if t else 0
                if t == 0:
                    xi = ratio
                elif abs(arg) < 1:
                    xi = math.atanh(arg)/t
                else:
                    xi = math.inf
                if abs(xi) <= 1:
                    candidates.append(_axis_value(xi,t,half,D[g],isum,idiff))
            minimum[g] += min(candidates)
            faces[minus,g],faces[plus,g] = mean[g]+pm,mean[g]+pp
    source = r*mean-currents.sum(axis=0)/volume
    defect = None
    if physical_source is not None:
        physical = np.asarray(physical_source,dtype=float)
        if physical.shape != mean.shape or not np.all(np.isfinite(physical)):
            raise ValueError('physical_source must have finite shape (G,)')
        defect = source-physical
    if not np.all(np.isfinite(faces)) or not np.all(np.isfinite(minimum)) or np.any(minimum <= margin):
        raise ValueError('nonpositive, uncertain or nonfinite homogeneous solution; reconstruction unsupported')
    return CuboidReconstruction(faces,source,minimum,volume,areas,defect)


def cuboid_flat_source_factors(heterogeneous_face_flux, *args, **kwargs):
    """Return frozen face factors (6,G) and their auxiliary reconstruction."""
    reconstruction = cuboid_flat_source_reconstruction(*args, **kwargs)
    surface = np.asarray(heterogeneous_face_flux,dtype=float)
    if surface.shape != reconstruction.face_flux.shape or not np.all(np.isfinite(surface)) or np.any(surface <= 0):
        raise ValueError('heterogeneous_face_flux must be finite positive shape (6,G)')
    factors = surface/reconstruction.face_flux
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0):
        raise ValueError('factors must be finite and positive')
    return factors,reconstruction
