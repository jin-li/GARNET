"""Source-space operators for the matching NEM discrete adjoint.

The dual variables in this module are source importances, not sampled adjoint
fluxes. Mean and ALL retained spatial source moments are part of the state.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigs

from .nem import NEMDiffusionSolver


def _real_array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf':
        raise ValueError(f'{name} must be a real numeric array')
    return np.asarray(raw, dtype=float)


class NEMSourceResponse(LinearOperator):
    """Exact zero-boundary-drive map H: source moments -> flux moments.

    Flattened vectors contain means (N,G), then moments (N,3,K,G), in C
    order. ``rmatvec`` is the Euclidean transpose of that complete map,
    including the direct local source response. It uses the cached condensed
    loss LU; it does not assemble a dense full-core matrix.
    """

    def __init__(self, solver):
        if not isinstance(solver, NEMDiffusionSolver):
            raise TypeError('solver must be a NEMDiffusionSolver')
        solver._assert_unchanged()
        self.solver = solver
        self.mean_size = solver.n_nodes * solver.n_groups
        self.moment_size = self.mean_size * 3 * solver.n_moments
        size = self.mean_size + self.moment_size
        if hasattr(solver, '_moment_source'):
            self.moment_source = solver._moment_source
            self.face_source = solver._face_source
        else:
            # Base quartic solver stores its local moment response explicitly.
            self.moment_source = sparse.block_diag(
                [sparse.csr_matrix(b) for b in solver._local_inverse.reshape(
                    -1, solver.n_groups, solver.n_groups)],
                format='csr')
            self.face_source = solver._face_from_moments @ self.moment_source
        super().__init__(dtype=np.dtype(float), shape=(size, size))

    def _vector(self, value):
        self.solver._assert_unchanged()
        value = _real_array(value, 'NEM source vector').reshape(-1)
        if value.size != self.shape[0] or not np.all(np.isfinite(value)):
            raise ValueError('invalid NEM source vector shape or nonfinite entries')
        return value

    def _matvec(self, value):
        q = self._vector(value)
        s, n = self.solver, self.mean_size
        rhs = np.r_[q[:n], -(s._constraint_face @ (self.face_source @ q[n:]))]
        x = s._column_scale * s._lu.solve(s._row_scale * rhs)
        return np.r_[x[:n], s._moment_current @ x[n:] + self.moment_source @ q[n:]]

    def _rmatvec(self, value):
        w = self._vector(value)
        s, n = self.solver, self.mean_size
        # Reverse every operation in matvec, including local recovery and
        # source-dependent interface RHS. A_cond^T alone omits both paths.
        xbar = np.r_[w[:n], s._moment_current.T @ w[n:]]
        rhsbar = s._row_scale * s._lu.solve(s._column_scale * xbar, trans='T')
        qmbar = (self.moment_source.T @ w[n:]
                 - self.face_source.T @ (s._constraint_face.T @ rhsbar[n:]))
        return np.r_[rhsbar[:n], qmbar]

    def pack(self, result):
        """Pack means and every spatial moment from a matching NEM result."""
        s = self.solver
        if (np.shape(result.means) != (s.n_nodes, s.n_groups)
                or np.shape(result.moments) != (s.n_nodes, 3, s.n_moments, s.n_groups)):
            raise ValueError('result shape does not match this NEM operator')
        return self._vector(np.r_[result.means.ravel(), result.moments.ravel()])

    def apply_group_matrix(self, matrix, value):
        """Apply a node-local target/source group matrix to all moments."""
        s, n = self.solver, self.mean_size
        matrix = _real_array(matrix, 'group matrix')
        if matrix.shape != (s.n_nodes, s.n_groups, s.n_groups) or not np.all(np.isfinite(matrix)):
            raise ValueError('group matrix must be finite (node,target,source)')
        value = self._vector(value)
        means = value[:n].reshape(s.n_nodes, s.n_groups)
        moments = value[n:].reshape(s.n_nodes, 3, s.n_moments, s.n_groups)
        return np.r_[np.einsum('ngh,nh->ng', matrix, means).ravel(),
                     np.einsum('ngh,naph->napg', matrix, moments).ravel()]

    def fission_operator(self):
        """Return H C with its exact transpose C^T H^T."""
        C = self.solver.fission
        return LinearOperator(self.shape, dtype=float,
            matvec=lambda y: self @ self.apply_group_matrix(C, y),
            rmatvec=lambda w: self.apply_group_matrix(C.transpose(0, 2, 1), self.rmatvec(w)))


def _readonly(value):
    value = np.array(value, dtype=float, copy=True)
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class NEMAdjointResult:
    """Left source-coordinate eigenvector and recovered source importance."""
    k_eff: float
    left_state: np.ndarray
    source_importance: np.ndarray
    relative_residual: float
    forward_relative_residual: float


@dataclass(frozen=True)
class NEMEffectiveKinetics:
    """Physical static parameters; times in seconds, decay in inverse seconds.

    ``generation_time`` is the physical mass/fission ratio Lambda.
    ``neutron_lifetime`` is k*Lambda under this discrete pencil definition.
    Neither is asserted equivalent to an unqualified Serpent time entry.
    """
    k_eff: float
    generation_time: float
    neutron_lifetime: float
    beta: np.ndarray
    decay: np.ndarray
    time_importance: float
    fission_importance: float

    @property
    def beta_total(self):
        return float(self.beta.sum())

    @property
    def lambda_low_frequency(self):
        """One-family reduction preserving sum(beta_i/lambda_i)."""
        return (self.beta_total / float(np.sum(self.beta / self.decay))
                if self.beta_total > 0 else None)


@dataclass(frozen=True)
class NEMTimeParameters:
    """Physical static time projection, requiring no delayed-family input."""
    k_eff: float
    generation_time: float
    neutron_lifetime: float
    time_importance: float
    fission_importance: float


def _relative_norm(residual, reference):
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(reference)):
        return float('inf')
    rscale, scale = float(np.max(abs(residual))), float(np.max(abs(reference)))
    if rscale == 0:
        return 0.
    if scale == 0:
        return float('inf')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        ratio = (rscale/scale) * (np.linalg.norm(residual/rscale)/np.linalg.norm(reference/scale))
    return float(ratio) if np.isfinite(ratio) else float('inf')


def _normalize(value, name):
    scale = float(np.max(abs(value)))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError(f'{name} must be finite nonzero')
    return value/scale, scale


def solve_nem_adjoint(solver, forward, *, rtol=1e-9, max_iterations=10000):
    """Solve the matching complete-state adjoint, normalized by max(abs(w))=1.

    The returned source importance H^T w already includes physical equation
    scaling, local elimination, boundary and ADF responses. Multiplying it by
    node volume again would change the discrete problem.
    """
    if not np.isfinite(rtol) or rtol <= 0 or max_iterations < 1:
        raise ValueError('rtol and max_iterations must be positive')
    H = NEMSourceResponse(solver)
    y, _ = _normalize(H.pack(forward), 'forward state')
    k = float(forward.k_eff)
    if not np.isfinite(k) or k <= 0:
        raise ValueError('forward k_eff must be finite positive')
    T = H.fission_operator()
    fres = _relative_norm(T @ y - k*y, k*y)
    if fres > rtol:
        raise ValueError(f'forward eigenpair residual {fres:g} exceeds tolerance')
    initial = np.zeros_like(y)
    initial[:H.mean_size] = 1
    values, vectors = eigs(T.T, k=1, which='LR', v0=initial,
                          tol=rtol * .1, maxiter=max_iterations)
    value, vector = values[0], vectors[:, 0]
    if not np.isfinite(value) or value.real <= 0 or not np.all(np.isfinite(vector)):
        raise ValueError('dominant adjoint eigenpair must be finite with positive eigenvalue')
    if abs(value.imag) > rtol*max(1, abs(value.real)) or np.linalg.norm(vector.imag) > rtol*np.linalg.norm(vector.real):
        raise ValueError('dominant adjoint eigenpair is not real')
    w, _ = _normalize(vector.real, 'adjoint state')
    pairing = float(w @ y)
    if not np.isfinite(pairing) or abs(pairing) <= np.finfo(float).eps*np.linalg.norm(w)*np.linalg.norm(y):
        raise ValueError('forward/adjoint pairing is singular')
    if pairing < 0:
        w = -w
    res = _relative_norm(T.rmatvec(w) - k*w, k*w)
    if abs(value.real-k) > rtol*max(k, abs(value.real)) or res > rtol:
        raise ValueError(f'adjoint eigenvalue/residual does not match forward: {value.real:g}, {res:g}')
    return NEMAdjointResult(float(value.real), _readonly(w), _readonly(H.rmatvec(w)), res, fres)


def _matched_state(solver, forward, adjoint, rtol):
    if not np.isfinite(rtol) or rtol <= 0:
        raise ValueError('rtol must be finite positive')
    H = NEMSourceResponse(solver)
    y, _ = _normalize(H.pack(forward), 'forward state')
    w, wscale = _normalize(H._vector(adjoint.left_state), 'adjoint state')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        z = H._vector(adjoint.source_importance)/wscale
    k = float(forward.k_eff)
    if (not np.isfinite(k) or k <= 0 or not np.isfinite(adjoint.k_eff)
            or adjoint.k_eff <= 0 or abs(adjoint.k_eff-k) > rtol*k):
        raise ValueError('adjoint and forward k_eff differ')
    T = H.fission_operator()
    if (_relative_norm(T @ y-k*y, k*y) > rtol
            or _relative_norm(T.rmatvec(w)-k*w, k*w) > rtol
            or _relative_norm(H.rmatvec(w)-z, z) > rtol):
        raise ValueError('forward/adjoint does not match this NEM operator')
    z, _ = _normalize(z, 'source importance')
    return H, y, z, k


def _time_projection(H, y, z, k, inverse_velocities):
    solver = H.solver
    N, G = solver.n_nodes, solver.n_groups
    inv = _real_array(inverse_velocities, 'inverse_velocities')
    if inv.shape != (N,G) or not np.all(np.isfinite(inv)) or np.any(inv <= 0):
        raise ValueError('inverse_velocities must be finite positive (N,G)')
    mass = H.apply_group_matrix(np.asarray([np.diag(v) for v in inv]), y)
    F = float(z @ H.apply_group_matrix(solver.fission, y))
    time = float(z @ mass)
    if not np.isfinite(F) or not np.isfinite(time) or F <= 0 or time <= 0:
        raise ValueError('nonpositive physical fission/time importance')
    return NEMTimeParameters(k, time/F, k*time/F, time, F)


def project_nem_time(solver, forward, adjoint, *, inverse_velocities, rtol=1e-8):
    """Project physical Lambda and k*Lambda without assuming delayed yields.

    This permits a qualified time calculation when regional inverse speeds
    exist but regional delayed-family production is still missing.
    """
    return _time_projection(*_matched_state(solver, forward, adjoint, rtol), inverse_velocities)


def project_nem_kinetics(solver, forward, adjoint, *, inverse_velocities,
                         delayed_production, decay, rtol=1e-8):
    """Project raw physical kinetics data using the matching NEM adjoint.

    ``inverse_velocities`` is (N,G), s/cm, constant in each node/group.
    ``delayed_production`` is (N,I,G_target,G_source), cm^-1. Direct family
    production matrices retain incident-energy dependence and regional/family
    spectra. They are RAW production, never whole-core effective beta values.
    ``decay`` is a common family vector in s^-1; incompatible regional family
    systems require a separate precursor model.

    Total production is the solver's frozen target/source matrix, supplied
    through fission_matrix or constructed from chi*nu_sigma_f by default.
    Its remainder after all delayed families must be nonnegative. No spectrum
    is substituted or renormalized. Static interfaces/ADF are held fixed in
    this projection.
    """
    H, y, z, k = _matched_state(solver, forward, adjoint, rtol)
    time = _time_projection(H, y, z, k, inverse_velocities)
    N, G = solver.n_nodes, solver.n_groups
    lam = _real_array(decay, 'decay')
    delayed = _real_array(delayed_production, 'delayed_production')
    if lam.ndim != 1 or not lam.size or not np.all(np.isfinite(lam)) or np.any(lam <= 0):
        raise ValueError('decay must be a nonempty finite positive family vector')
    if delayed.shape != (N,lam.size,G,G) or not np.all(np.isfinite(delayed)) or np.any(delayed < 0):
        raise ValueError('delayed_production must be finite nonnegative (N,I,G,G)')
    total = solver.fission
    excess = delayed.sum(axis=1)-total
    if np.any(excess > 1e-13 * np.maximum(np.max(total), np.finfo(float).tiny)):
        raise ValueError('delayed production exceeds total production; prompt remainder is negative')
    F = time.fission_importance
    beta = np.asarray([z @ H.apply_group_matrix(delayed[:,i], y)/F for i in range(lam.size)])
    if not np.all(np.isfinite(beta)) or np.any(beta < 0) or beta.sum() >= 1:
        raise ValueError('projected delayed fractions are not physical')
    return NEMEffectiveKinetics(k, time.generation_time, time.neutron_lifetime,
                                _readonly(beta), _readonly(lam), time.time_importance, F)
