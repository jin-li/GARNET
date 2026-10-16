"""Independent CPU continuous reflector-axis diagnostic (no production ADf).

The auxiliary uniform source closes the supplied mean/current balance. It is
not a physical core source or a fitted parameter. Positivity concerns the
numerical cubic spline only, and is not a certified continuum statement.
"""
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import PPoly


class ReflectorAxisReferenceError(ValueError):
    """Invalid, ill-conditioned, or unconverged reference problem."""


class NonpositiveReflectorAxisError(ReflectorAxisReferenceError):
    """A converged numerical field is nonpositive; ``result`` retains it."""

    def __init__(self, result):
        super().__init__('nonpositive reflector reference field; diagnostic only')
        self.result = result
        self.diagnostics = result.diagnostics


def _readonly(value):
    value = np.array(value, dtype=float, copy=True)
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class ReflectorAxisReferenceResult:
    """Spline coefficients have scipy PPoly layout (4, intervals, groups).

    ``evaluate(t)`` returns shape ``t.shape + (G,)`` for t=x/L. Negative
    values are preserved. ``physical_admission`` is always false: this is
    an auxiliary diagnostic, including when its numerical spline is positive.
    """
    knots: np.ndarray
    coefficients: np.ndarray
    face_means: np.ndarray
    mean: np.ndarray
    delta: np.ndarray
    minima: np.ndarray
    minimum_locations: np.ndarray
    diagnostics: object

    def evaluate(self, t, derivative=0):
        t = np.asarray(t, dtype=float)
        if not np.all(np.isfinite(t)) or np.any(abs(t) > .5):
            raise ValueError('t must be finite and inside [-.5,.5]')
        return PPoly(self.coefficients, self.knots, extrapolate=False)(t, nu=derivative)


def solve_reflector_axis_reference(
        diffusion, loss, width, native_mean, currents,
        transverse_leakage_mean, transverse_leakage_moments, *,
        tol=1e-9, max_nodes=10000, require_positive=True):
    """Solve -D/L² phi_tt + R phi + ell(t) = delta with fixed currents.

    Currents (2,G) are in the global axis direction, at minus/plus faces.
    Leakage is ell0 + 12 ell1 t + 20 ell2 (3t²-.25); moments are (2,G).
    R acts on group columns and must be a nonsingular, column-diagonally
    dominant nonfission M-matrix. The dimensionless mean-scaled loss operator
    must have singular values between 1e-12 and 1e12 and condition <=1e12.
    Flux states are scaled by native means; derivative states additionally use
    the group diffusion rate L*sqrt(R_gg/D_g). This balances the first-order
    equations for stiff groups. Solver RMS residuals use these balanced states;
    first_order_residual is independently checked in the original native-mean
    coordinates, with the same acceptance threshold as the balance checks.
    No clipping, field renormalization, or adjustment of currents is applied.
    With require_positive=False, a nonpositive result is diagnostic only.
    """
    def array(x, name):
        if np.iscomplexobj(x):
            raise ReflectorAxisReferenceError(f'{name} must be real')
        try:
            a = np.asarray(x, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ReflectorAxisReferenceError(f'{name} must be numeric') from exc
        if not np.all(np.isfinite(a)):
            raise ReflectorAxisReferenceError(f'{name} must be finite')
        return a

    D, R = array(diffusion, 'diffusion'), array(loss, 'loss')
    m, J = array(native_mean, 'native_mean'), array(currents, 'currents')
    ell0 = array(transverse_leakage_mean, 'transverse_leakage_mean')
    ellmom = array(transverse_leakage_moments, 'transverse_leakage_moments')
    L, tol = array(width, 'width'), array(tol, 'tol')
    if L.ndim or tol.ndim or L <= 0 or not 1e-13 <= tol < 1:
        raise ReflectorAxisReferenceError('width must be positive; tol must be in [1e-13,1)')
    L, tol = float(L), float(tol)
    if (isinstance(max_nodes, (bool, np.bool_)) or not isinstance(max_nodes, (int, np.integer))
            or max_nodes < 5):
        raise ReflectorAxisReferenceError('max_nodes must be an integer >= 5')
    if not isinstance(require_positive, (bool, np.bool_)):
        raise ReflectorAxisReferenceError('require_positive must be boolean')
    G = D.size
    if (D.ndim != 1 or G == 0 or R.shape != (G, G) or m.shape != (G,)
            or J.shape != (2, G) or ell0.shape != (G,) or ellmom.shape != (2, G)):
        raise ReflectorAxisReferenceError('expected shapes G,GxG,G,2xG,G,2xG')
    offdiag = R.copy()
    np.fill_diagonal(offdiag, 0)
    if (np.any(D <= 0) or np.any(m <= 0) or np.any(np.diag(R) <= 0)
            or np.any(offdiag > 0) or np.any(R.sum(axis=0) < 0)):
        raise ReflectorAxisReferenceError('positive D/mean and nonfission M-matrix loss required')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        A = (L**2 / (D*m))[:, None] * R * m[None, :]
        delta = R @ m + (J[1]-J[0])/L + ell0
        boundary = -L*J/(D*m)
    if not all(np.all(np.isfinite(x)) for x in (A, delta, boundary)):
        raise ReflectorAxisReferenceError('nonfinite scaled problem')
    try:
        singular = np.linalg.svd(A, compute_uv=False)
        condition = max(1., singular[0])/min(1., singular[-1]) if singular[-1] > 0 else np.inf
    except np.linalg.LinAlgError as exc:
        raise ReflectorAxisReferenceError('loss conditioning failed') from exc
    if not np.isfinite(condition) or condition > 1e12:
        raise ReflectorAxisReferenceError('singular or ill-conditioned dimensionless loss operator')

    # Balancing u_t=v as u_t=rate*w, w_t=(A*u+forcing)/rate avoids
    # excessive refinement around zeros of large derivative states. In those
    # regions, cubic-spline differentiation can otherwise reach roundoff before
    # satisfying a tight collocation tolerance. This is a variable transform;
    # currents, source, field scaling, and physical residual gates are fixed.
    rate = np.sqrt(np.diag(A))
    scaled_boundary = boundary/rate

    def leakage(t):
        return ell0[:, None] + 12*ellmom[0, :, None]*t + 20*ellmom[1, :, None]*(3*t*t-.25)

    def fun(t, y):
        return np.vstack((rate[:, None]*y[G:], (A @ y[:G] + (L**2/(D*m))[:, None]*(leakage(t)-delta[:, None]))/rate[:, None]))

    def jac(t, y):
        result = np.zeros((2*G, 2*G, t.size))
        result[:G, G:] = np.diag(rate)[:, :, None]
        result[G:, :G] = (A/rate[:, None])[:, :, None]
        return result

    def bc(left, right):
        return np.r_[left[G:]-scaled_boundary[0], right[G:]-scaled_boundary[1]]

    nodes = np.linspace(-.5, .5, 5)
    guess = np.vstack((np.ones((G, 5)), np.linspace(scaled_boundary[0], scaled_boundary[1], 5).T))
    solution = solve_bvp(fun, bc, nodes, guess, fun_jac=jac, tol=tol, max_nodes=max_nodes)
    if not solution.success or not np.all(np.isfinite(solution.y)):
        raise ReflectorAxisReferenceError(f'BVP did not converge: {solution.message}')
    coefficients = solution.sol.c[:, :, :G]*m[None, None, :]
    spline = PPoly(coefficients, solution.x, extrapolate=False)
    faces = spline(np.array([-.5, .5]))
    mean = spline.integrate(-.5, .5)
    measured_J = -D/L*spline(np.array([-.5, .5]), nu=1)
    minima, locations = [], []
    for g in range(G):
        scalar = PPoly(coefficients[:, :, g], solution.x, extrapolate=False)
        roots = scalar.derivative().roots(extrapolate=False)
        points = np.r_[-.5, roots[np.isfinite(roots) & (roots >= -.5) & (roots <= .5)], .5]
        values = scalar(points)
        i = np.argmin(values)
        minima.append(values[i]); locations.append(points[i])
    minima, locations = np.array(minima), np.array(locations)
    # Probe all spline intervals independently of the solver's collocation nodes.
    q = np.array([.1127016653792583, .5, .8872983346207417])
    probes = (solution.x[:-1, None] + np.diff(solution.x)[:, None]*q).ravel()
    y, dy = solution.sol(probes), solution.sol(probes, 1)
    rhs = fun(probes, y)
    relative = lambda a, scale: float(np.max(abs(a)/np.maximum(scale, np.finfo(float).tiny)))
    # Undo derivative-state balancing before checking the original first-order ODE.
    dy[G:] *= rate[:, None]
    rhs[G:] *= rate[:, None]
    first_order_error = relative(dy-rhs, 1+abs(rhs))
    current_error = relative(measured_J-J, abs(J)+D*m/L)
    mean_error = relative(mean-m, m)
    balance = R @ mean + (measured_J[1]-measured_J[0])/L + ell0-delta
    balance_error = relative(balance, abs(R) @ m + abs(J).sum(axis=0)/L + abs(ell0)+abs(delta))
    positive = bool(np.all(minima > 0))
    diagnostics = dict(converged=True, solver_status=int(solution.status),
                       solver_message=solution.message, nodes=solution.x.size,
                       iterations=solution.niter, tolerance=tol,
                       condition_number_2_operator=float(singular[0]/singular[-1]),
                       dimensionless_operator_condition_guard=float(condition),
                       first_order_residual=first_order_error,
                       solver_rms_residual=float(np.max(solution.rms_residuals)),
                       current_balance_error=current_error, mean_balance_error=mean_error,
                       zeroth_balance_error=balance_error, full_profile_positive=positive,
                       negative_profile_count=int(np.count_nonzero(minima < 0)),
                       nonpositive_profile_count=int(np.count_nonzero(minima <= 0)),
                       extrema_are_interval_certified=False, physical_admission=False,
                       diagnostic_only=True, auxiliary_source='fixed algebraic balance')
    if max(first_order_error, current_error, mean_error, balance_error) > 100*tol:
        raise ReflectorAxisReferenceError('independent field residual/balance checks failed')
    result = ReflectorAxisReferenceResult(
        *map(_readonly, (solution.x, coefficients, faces, mean, delta, minima, locations)),
        MappingProxyType(diagnostics))
    if require_positive and not positive:
        raise NonpositiveReflectorAxisError(result)
    return result
