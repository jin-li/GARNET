"""Unqualified CPU-only coupled quartic transverse-average axis reference.

This is an isolated mathematical generalization, with no solver integration or
3D point-field positivity claim. Currents and moments are prescribed local data;
neighbor footprint compatibility and missing-neighbor closure belong to callers.
Coupling acts on group columns (e.g. removal diagonal minus off-diagonal
source-row scatter transpose minus outer(chi, nu_f)/k).

With t=x/L in [-1/2,1/2], the basis is 1, t, 3t^2-1/4,
t(t^2-1/4), (t^2-1/20)(t^2-1/4). The exact quadratic leakage moments
are p1/12 and p2/20. The latter deliberately departs from the printed p2/60
in thesis Eq. 3.44 and appendix: /60 conflicts with the stated second basis.
This implements the independently integrated weak form, not a literal NEM
reproduction or a claim about the author's implementation.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class NodalPolynomialReferenceResult:
    """Moments m1,m2; faces MINUS,PLUS; coefficients in ascending order."""

    moments: np.ndarray
    face_means: np.ndarray
    basis_coefficients: np.ndarray
    power_coefficients: np.ndarray
    diagnostics: dict


class NodalPolynomialReferenceError(ValueError):
    """Rejected numerical result, retaining all solved fields and evidence."""

    def __init__(self, message, diagnostics, result=None):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.result = result


def _array(value, name):
    if hasattr(value, '__cuda_array_interface__') or hasattr(value, 'get'):
        raise TypeError(f'{name}: CPU arrays required')
    if np.iscomplexobj(value):
        raise ValueError(f'{name} must be real')
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must be finite')
    return array


def _positive_scalar(value, name):
    array = _array(value, name)
    if array.ndim != 0 or array <= 0:
        raise ValueError(f'{name} must be a positive scalar')
    return float(array)


def _equilibrate(matrix):
    """Return R A C and positive diagonal scaling vectors, without inversion."""
    scaled = matrix.copy()
    row = np.ones(len(matrix))
    column = np.ones(len(matrix))
    for _ in range(4):
        maxima = abs(scaled).max(axis=1)
        r = np.ones(len(matrix))
        r[maxima > 0] = 1 / maxima[maxima > 0]
        scaled *= r[:, None]
        row *= r
        maxima = abs(scaled).max(axis=0)
        c = np.ones(len(matrix))
        c[maxima > 0] = 1 / maxima[maxima > 0]
        scaled *= c[None, :]
        column *= c
    return scaled, row, column


def quadratic_neighbor_leakage_moments(center_avg, minus_avg, plus_avg,
                                      center_width, minus_width, plus_width):
    """Fit interval means over actual adjacent unequal widths, returning (2,G).

    Averages can have either sign. In center-scaled t coordinates the fitted
    polynomial is center_avg+p1*t+p2*(3*t^2-1/4). Its exact center-cell weighted
    moments are p1/12 and p2/20, not the source's printed p2/60. The inputs must
    already represent compatible transverse footprints; no closure is supplied.
    """
    center = _array(center_avg, 'center_avg')
    minus = _array(minus_avg, 'minus_avg')
    plus = _array(plus_avg, 'plus_avg')
    if center.ndim != 1 or not center.size or minus.shape != center.shape or plus.shape != center.shape:
        raise ValueError('averages must be matching nonempty group vectors')
    width = _positive_scalar(center_width, 'center_width')
    wm = _positive_scalar(minus_width, 'minus_width') / width
    wp = _positive_scalar(plus_width, 'plus_width') / width
    with np.errstate(over='ignore', invalid='ignore'):
        matrix = np.array([[-(1 + wm)/2, .5 + 1.5*wm + wm*wm],
                           [(1 + wp)/2, .5 + 1.5*wp + wp*wp]])
        rhs = np.stack((minus-center, plus-center))
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)):
        raise ValueError('neighbor fit exceeds finite floating-point range')
    scaled, row, column = _equilibrate(matrix)
    try:
        coefficients = column[:, None] * np.linalg.solve(scaled, row[:, None]*rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError('neighbor interval-mean fit is singular') from exc
    result = coefficients / np.array([[12.], [20.]])
    if not np.all(np.isfinite(result)):
        raise ValueError('neighbor fit has nonfinite moments')
    return result


def _stationary_points(power):
    """All interior stationary points, including repeated derivative roots.

    The derivative is monotone between roots of its own derivative. Bracketing
    those intervals catches simple/triple roots; checking the turning points
    also catches double roots. Constant and linear derivatives need no cubic
    root classification or complex-root tolerance.
    """
    derivative = np.polynomial.polynomial.polytrim(np.arange(1, 5)*power[1:])
    if len(derivative) <= 1:
        return np.empty(0)
    second = np.arange(1, len(derivative))*derivative[1:]
    turning = np.polynomial.polynomial.polyroots(second)
    turning = [float(z.real) for z in turning
               if abs(z.imag) <= 64*np.finfo(float).eps*max(1, abs(z.real))
               and -.5 < z.real < .5]
    bounds = np.unique([-.5, *turning, .5])
    evaluate = lambda t: np.polynomial.polynomial.polyval(t, derivative)
    tolerance = 64*np.finfo(float).eps*np.sum(abs(derivative))
    roots = [t for t in bounds[1:-1] if abs(evaluate(t)) <= tolerance]
    for left, right in zip(bounds[:-1], bounds[1:]):
        fl, fr = evaluate(left), evaluate(right)
        if fl == 0 and -.5 < left < .5:
            roots.append(left)
        if fr == 0 and -.5 < right < .5:
            roots.append(right)
        if fl != 0 and fr != 0 and np.signbit(fl) != np.signbit(fr):
            roots.append(brentq(evaluate, left, right, xtol=1e-15, maxiter=200))
    return np.unique(roots)


def coupled_nodal_polynomial_axis_reference(
        diffusion, coupling, width, mean, net_inward_density,
        transverse_leakage_moments, *, rtol=1e-10, condition_limit=1e12):
    """Solve the certified 4G local quartic equations for one Cartesian axis.

    All vector inputs are CPU arrays. Diffusion and mean have shape (G,),
    coupling (G,G), currents (2,G) in MINUS/PLUS inward-density order, and
    leakage moments (2,G) in ell1/ell2 order. Leakage moments must ALREADY be
    divided by transverse widths and summed. No extra width division occurs.
    M may be arbitrary finite real coupling, without a uniqueness/positivity
    promise. Conditioning uses the exact dense SVD 2-norm of the equilibrated
    4G matrix. Independent Gauss checks verify physical weak equations, endpoint
    currents, mean and weighted moments. Weak errors use absolute quadrature
    contributions plus the characteristic curvature scale (D/L^2)*mean*<|f_n|>,
    which also defines a nonzero scale for constant/linear zero-reaction fields.
    Positivity is checked at endpoints and
    every real stationary point of the 1D quartic, with no clipping or floor.

    No ell0 is supplied. The reported axis balance without transverse mean is
    M@mean+(Jplus-Jminus)/L; it is NOT a complete 3D auxiliary delta or core source.
    """
    D = _array(diffusion, 'diffusion')
    mu = _array(mean, 'mean')
    M = _array(coupling, 'coupling')
    inward = _array(net_inward_density, 'net_inward_density')
    ell = _array(transverse_leakage_moments, 'transverse_leakage_moments')
    L = _positive_scalar(width, 'width')
    rtol = _positive_scalar(rtol, 'rtol')
    condition_limit = _positive_scalar(condition_limit, 'condition_limit')
    if condition_limit < 1:
        raise ValueError('condition_limit must be at least one')
    if D.ndim != 1 or not D.size or np.any(D <= 0):
        raise ValueError('diffusion must be a positive nonempty group vector')
    G = D.size
    if mu.shape != (G,) or np.any(mu <= 0):
        raise ValueError('mean must be a positive matching group vector')
    if M.shape != (G,G) or inward.shape != (2,G) or ell.shape != (2,G):
        raise ValueError('coupling/current/leakage shapes must be (G,G)/(2,G)/(2,G)')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        H = D/L/L
        rate = max(np.max(H), np.max(abs(M)))
        h = H/rate
        reaction = (M/rate)*(mu[None,:]/mu[:,None])
        normalized_j = (inward/mu[None,:])/D[None,:]*L
        jm, jp = normalized_j[0], -normalized_j[1]
        leakage = (ell/mu[None,:])/rate
    if (not np.isfinite(rate) or rate <= 0 or np.any(H <= 0)
            or not all(np.all(np.isfinite(v)) for v in (H,h,reaction,normalized_j,leakage))):
        raise ValueError('physical scales exceed finite floating-point range')
    eye = np.eye(G)
    diagonal = np.diag(h)
    zero = np.zeros((G,G))
    matrix = np.block([[-reaction, zero, -diagonal, diagonal],
                       [zero, -reaction, -3*diagonal, -3*diagonal],
                       [60*eye, 140*eye, -16*eye, -4*eye],
                       [60*eye, -140*eye, 4*eye, 16*eye]])
    rhs = np.concatenate((h*(jp+jm)/2+leakage[0],
                          h*(jp-jm)/2-6*h+leakage[1],
                          jp-20, jm+20))
    scaled, row, column = _equilibrate(matrix)
    diagnostics = {'groups': G, 'unknowns': 4*G,
                   'leakage_second_moment_definition': 'p2/20; independent weak form, not printed p2/60'}
    try:
        singular_values = np.linalg.svd(scaled, compute_uv=False)
        condition = float(singular_values[0]/singular_values[-1]) if singular_values[-1] > 0 else float('inf')
        diagnostics['condition_number_2_scaled'] = condition
        solution_scaled = np.linalg.solve(scaled, row*rhs)
        solution = column*solution_scaled
    except np.linalg.LinAlgError as exc:
        raise NodalPolynomialReferenceError('singular nodal polynomial system', diagnostics) from exc
    moments_scaled = solution[:2*G].reshape(2,G)
    bp, bm = solution[2*G:3*G], solution[3*G:]
    a1, a2 = bp-bm, bp+bm-2
    a3, a4 = 10*a1-120*moments_scaled[0], 35*a2-700*moments_scaled[1]
    basis_scaled = np.column_stack((np.ones(G), a1, a2, a3, a4))
    power_scaled = np.column_stack((1-a2/4+a4/80, a1-a3/4,
                                    3*a2-3*a4/10, a3, a4))
    with np.errstate(over='ignore', invalid='ignore'):
        moments = moments_scaled*mu
        faces = np.stack((bm,bp))*mu
        basis = basis_scaled*mu[:,None]
        power = power_scaled*mu[:,None]
    result = NodalPolynomialReferenceResult(moments, faces, basis, power, diagnostics)
    if not all(np.all(np.isfinite(v)) for v in (solution,moments,faces,basis,power)):
        raise NodalPolynomialReferenceError('nonfinite nodal polynomial solution', diagnostics, result)
    if not np.isfinite(condition) or condition > condition_limit:
        raise NodalPolynomialReferenceError('nodal polynomial condition exceeds condition_limit', diagnostics, result)
    # Four-point Gauss would integrate all weak polynomials exactly (degree <=6).
    # Eight points provide an independent path and absolute contribution scales.
    t, weights = np.polynomial.legendre.leggauss(8)
    t, weights = t/2, weights/2
    phi = np.polynomial.polynomial.polyval(t, power.T)
    first_coeff = power[:,1:]*np.arange(1,5)
    second_coeff = power[:,2:]*np.array([2,6,12])
    second = np.polynomial.polynomial.polyval(t, second_coeff.T)
    tests = np.stack((t,3*t*t-.25))
    weighted_phi = (phi*weights)@tests.T
    weighted_second = (second*weights)@tests.T
    weak_residual = H[:,None]*weighted_second-M@weighted_phi-ell.T
    weak_scale = (H[:,None]*((abs(second)*weights)@abs(tests).T)
                  + abs(M)@((abs(phi)*weights)@abs(tests).T)+abs(ell.T)
                  + (H*mu)[:,None]*(abs(tests)@weights)[None,:])
    endpoint_derivative = np.polynomial.polynomial.polyval(np.array([-.5,.5]),first_coeff.T)
    recovered_inward = np.stack((-D/L*endpoint_derivative[:,0],D/L*endpoint_derivative[:,1]))
    current_scale = abs(inward)+abs(recovered_inward)+(D/L*mu)[None,:]
    backward_residual = scaled@solution_scaled-row*rhs
    backward_scale = abs(scaled)@abs(solution_scaled)+abs(row*rhs)
    diagnostics.update(
        scaled_backward_error=float(np.max(abs(backward_residual)/np.maximum(backward_scale,np.finfo(float).tiny))),
        physical_weak_balance_error=float(np.max(abs(weak_residual)/np.maximum(weak_scale,np.finfo(float).tiny))),
        physical_current_balance_error=float(np.max(abs(recovered_inward-inward)/np.maximum(current_scale,np.finfo(float).tiny))),
        physical_mean_balance_error=float(np.max(abs(phi@weights-mu)/mu)),
        physical_moment_balance_error=float(np.max(abs(weighted_phi.T-moments)/mu[None,:])),
        physical_weak_residual=weak_residual.T,
        physical_weak_residual_scale=weak_scale.T,
        physical_weak_error_scale_definition='absolute quadrature reaction/diffusion contributions + abs(ell_n) + (D/L^2)*mean*integral(abs(f_n))',
        physical_current_residual=recovered_inward-inward,
        axis_balance_without_transverse_mean=M@mu-(inward[0]+inward[1])/L)
    roots, candidates, minima, locations = [], [], [], []
    try:
        for g in range(G):
            stationary = _stationary_points(power_scaled[g])
            points = np.r_[-.5,stationary,.5]
            values = np.polynomial.polynomial.polyval(points,power_scaled[g])*mu[g]
            index = int(np.argmin(values))
            roots.append(stationary)
            candidates.append(points)
            minima.append(float(values[index]))
            locations.append(float(points[index]))
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        raise NodalPolynomialReferenceError('stationary-point computation failed', diagnostics, result) from exc
    group = int(np.argmin(minima))
    diagnostics.update(stationary_points=tuple(roots), positivity_candidates=tuple(candidates),
                       profile_minima=np.asarray(minima), profile_minimum_locations=np.asarray(locations),
                       minimum_polynomial_flux=minima[group], minimum_polynomial_group=group,
                       minimum_polynomial_t=locations[group],
                       minimum_face_mean=float(np.min(faces)),
                       minimum_face_index=tuple(int(i) for i in np.unravel_index(np.argmin(faces),faces.shape)))
    if not all(np.all(np.isfinite(v)) for v in (phi,second,weak_residual,weak_scale,recovered_inward,minima)):
        raise NodalPolynomialReferenceError('nonfinite physical polynomial diagnostics', diagnostics, result)
    if np.any(faces <= 0):
        raise NodalPolynomialReferenceError('nonpositive whole-face mean', diagnostics, result)
    if min(minima) <= 0:
        raise NodalPolynomialReferenceError('nonpositive polynomial profile', diagnostics, result)
    gate_keys = ('scaled_backward_error','physical_weak_balance_error',
                 'physical_current_balance_error','physical_mean_balance_error','physical_moment_balance_error')
    if any(not np.isfinite(diagnostics[k]) or diagnostics[k] > rtol for k in gate_keys):
        raise NodalPolynomialReferenceError('nodal polynomial balance tolerance exceeded', diagnostics, result)
    return result
