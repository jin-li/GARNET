"""Opt-in 2D group Gauss--Seidel fixed-source pilot; no eigenvalue solve."""
from dataclasses import dataclass, field
import math
from types import MappingProxyType

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from .native_angular_source import _budget, _integer, _readonly
from .native_grouped_angular import GroupAngularOperators
from .native_mesh_streaming import NativeMeshStreaming


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be finite positive')
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite positive')
    return value


@dataclass(frozen=True)
class GroupedFixedSourceResult:
    """Raw finite terminal iterate, including on iteration-limit failure.

    The returned field is an immutable owned copy; history rows are immutable.
    No positivity filter, normalization, XS adjustment or fallback is applied.
    """
    angular_flux: np.ndarray = field(repr=False, compare=False)
    converged: bool
    reason: str
    full_residual: float | None
    history: tuple
    provenance: dict

    def __post_init__(self):
        object.__setattr__(self, 'angular_flux', _readonly(self.angular_flux))
        object.__setattr__(self, 'history', tuple(MappingProxyType(dict(row)) for row in self.history))
        object.__setattr__(self, 'provenance', MappingProxyType(dict(self.provenance)))


@dataclass(frozen=True)
class GroupedFixedSource:
    """One-group all-angle GMRES within multigroup Gauss--Seidel sweeps.

    Equations are cell-integrated A psi = rhs. Ascending native group order
    visits EVERY group on EVERY sweep, retaining upscatter and all signed
    SP0..SPL coefficients. Exact simultaneous residuals determine convergence.
    The only preconditioner is the streaming/collision diagonal, one group at
    a time. No full coupled transport constructor, group factorization, fission,
    retries, automatic selection, or physical admission is provided.
    """
    streaming: NativeMeshStreaming
    boundary_conditions: tuple
    inner_tolerance: float = 1e-10
    full_tolerance: float = 1e-8
    restart: int = 30
    max_inner: int = 200
    max_sweeps: int = 100
    max_bytes: int = field(default=256*1024**2, repr=False, compare=False)
    operators: GroupAngularOperators = field(init=False, repr=False, compare=False)
    memory_estimate_bytes: int = field(init=False)

    def __post_init__(self):
        if not isinstance(self.streaming, NativeMeshStreaming):
            raise ValueError('NativeMeshStreaming required')
        for name in ('inner_tolerance', 'full_tolerance'):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name in ('restart', 'max_inner', 'max_sweeps'):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))
        nc, na, ng = self.streaming.shape
        h = self.streaming.quadrature.harmonics.shape[1]
        group_size, full_size, moments = nc*na, nc*na*ng, nc*h*ng
        restart = min(self.restart, self.max_inner, group_size)
        object.__setattr__(self, 'restart', restart)
        batch = min(nc, 512)
        # Reproduce the public operator bound before constructing its reflection
        # permutations. No large numeric allocation precedes this preflight.
        operator_bytes = 8*(3*nc*na + 2*nc*h + na*h +
                            batch*(4*h*ng+8*h+8*na+16) + 100*na+1)
        # Includes source/iterate/result immutable-copy ownership and validation
        # allowance (5 full angular fields); cache + moment conversion allowance;
        # one-group Krylov basis, Hessenberg/rotations and 32 group work vectors
        # for SciPy, RHS, diagonal, x0/candidate, true residual and norm work.
        # History term counts numeric payload, not Python object overhead.
        estimate = operator_bytes + 8*(5*full_size + 2*moments +
                    (restart+1)*group_size + 32*group_size + 6*(restart+1)**2 +
                    4*self.max_sweeps*(ng+4))
        _budget(estimate, self.max_bytes)
        object.__setattr__(self, 'memory_estimate_bytes', estimate)
        object.__setattr__(self, 'operators', GroupAngularOperators(
            self.streaming, self.boundary_conditions, max_bytes=self.max_bytes))

    def _guard(self):
        _budget(self.memory_estimate_bytes, self.max_bytes)

    def _field(self, value, name):
        if (not isinstance(value, np.ndarray) or value.shape != self.streaming.shape
                or value.dtype.kind not in 'fiu'):
            raise ValueError(f'{name} must be a real ndarray with shape {self.streaming.shape}')
        # Validate group-by-group: no full multigroup boolean temporary.
        for group in range(self.streaming.shape[2]):
            self.operators._array(value[:, :, group], self.operators.shape, name)
        return value

    @staticmethod
    def _norm(value):
        # Scale before squaring: tiny nonzero sources must not become the
        # zero-RHS shortcut merely because an unscaled sum of squares underflows.
        scale = float(np.max(np.abs(value)))
        if not math.isfinite(scale):
            raise ValueError('nonfinite vector norm')
        if scale == 0:
            return 0.
        norm = scale*float(np.linalg.norm(value/scale))
        if not math.isfinite(norm):
            raise ValueError('nonfinite vector norm')
        return norm

    def _refresh(self, flux, moments):
        for group in range(self.streaming.shape[2]):
            moments[:, :, group] = self.operators.project_group(flux[:, :, group])

    def _full_residual(self, flux, moments, rhs, denominator):
        # This is A psi - rhs with ALL groups evaluated at the same terminal
        # iterate, not the lagged Gauss--Seidel RHS of any individual group.
        total = 0.
        for group in range(self.streaming.shape[2]):
            residual = self.operators.streaming_group_action(flux[:, :, group], group)
            residual -= self.operators.scatter_to_group(moments, group)
            residual -= rhs[:, :, group]
            total = math.hypot(total, self._norm(residual))
        value = total/denominator
        if not math.isfinite(value):
            raise ValueError('nonfinite simultaneous residual')
        return value

    def _result(self, flux, converged, reason, residual, history, warm):
        return GroupedFixedSourceResult(flux, converged, reason, residual, tuple(history), dict(
            initial_field='caller_supplied' if warm else 'zero',
            group_initial_guess='latest_group_iterate', group_order=tuple(range(self.streaming.shape[2])),
            preconditioner='one_group_streaming_collision_diagonal',
            residual='simultaneous_L2_relative_rhs_or_absolute_for_zero_rhs',
            inner_tolerance=self.inner_tolerance, full_tolerance=self.full_tolerance,
            restart=self.restart, max_inner=self.max_inner, max_sweeps=self.max_sweeps,
            boundary_conditions=self.boundary_conditions, memory_estimate_bytes=self.memory_estimate_bytes,
            max_bytes=self.max_bytes, physical_admission=False, eigenvalue_solver=False))

    def solve(self, rhs, *, initial_flux=None):
        """Solve a supplied cell-integrated signed source, cold by default.

        Inputs are caller-owned and never mutated; owned float64 copies are
        budgeted. An explicit initial_flux enables a warm outer start. Each
        group starts GMRES from its latest iterate. Legacy callback semantics
        bound actual Arnoldi steps by max_inner, not restart cycles.

        Inner tests use ||A_g x-b_g||/||b_g|| (absolute for zero b_g).
        The full test uses ||A psi-rhs||/||rhs|| (absolute for zero rhs).
        Finite inner candidates are retained even when convergence fails;
        arithmetic failure retains the last finite, projected group iterate.
        """
        self._guard()
        self._field(rhs, 'rhs')
        warm = initial_flux is not None
        if warm:
            self._field(initial_flux, 'initial_flux')
        source = np.array(rhs, dtype=np.float64, copy=True)
        flux = np.array(initial_flux, dtype=np.float64, copy=True) if warm else np.zeros(self.streaming.shape)
        # A finite extended-precision input can overflow its float64 copy.
        self._field(source, 'stored rhs')
        self._field(flux, 'stored initial_flux')
        moments = np.zeros((self.streaming.shape[0], self.streaming.quadrature.harmonics.shape[1],
                            self.streaming.shape[2]))
        history, full_residual = [], None
        try:
            self._refresh(flux, moments)
            source_norm = 0.
            for group in range(self.streaming.shape[2]):
                source_norm = math.hypot(source_norm, self._norm(source[:, :, group]))
            if not math.isfinite(source_norm):
                raise ValueError('nonfinite source norm')
            denominator = source_norm if source_norm else 1.
            full_residual = self._full_residual(flux, moments, source, denominator)
            if full_residual <= self.full_tolerance:
                reason = 'zero_rhs' if source_norm == 0 and not warm else 'initial_residual_converged'
                return self._result(flux, True, reason, full_residual, history, warm)
            for sweep in range(1, self.max_sweeps+1):
                steps, infos, residuals = [], [], []
                failed = None
                for group in range(self.streaming.shape[2]):
                    group_rhs = self.operators.scatter_to_group(moments, group, exclude_self=True)
                    group_rhs += source[:, :, group]
                    self.operators._finite(group_rhs)
                    rhs_norm = self._norm(group_rhs)
                    diagonal = self.operators.diagonal_group(group)
                    if np.any(diagonal <= 0):
                        failed = 'zero_streaming_diagonal'
                        break
                    count = 0
                    def callback(_):
                        nonlocal count
                        count += 1
                    if rhs_norm == 0:
                        candidate, info = np.zeros(self.operators.shape), 0
                    else:
                        size = group_rhs.size
                        operator = LinearOperator((size, size), dtype=np.float64,
                            matvec=lambda x: self.operators.group_action(x.reshape(self.operators.shape), group).ravel())
                        preconditioner = LinearOperator((size, size), dtype=np.float64,
                            matvec=lambda x: x/diagonal.ravel())
                        candidate, info = gmres(operator, group_rhs.ravel(),
                            x0=flux[:, :, group].copy().ravel(), M=preconditioner,
                            rtol=self.inner_tolerance, atol=0., restart=self.restart,
                            maxiter=self.max_inner, callback=callback, callback_type='legacy')
                        candidate = candidate.reshape(self.operators.shape)
                    self.operators._finite(candidate)
                    true_residual = self.operators.group_action(candidate, group)-group_rhs
                    inner_residual = self._norm(true_residual)/(rhs_norm if rhs_norm else 1.)
                    # Project before committing, preserving the last finite
                    # flux/cache pair if projection arithmetic fails.
                    candidate_moments = self.operators.project_group(candidate)
                    flux[:, :, group] = candidate
                    moments[:, :, group] = candidate_moments
                    steps.append(count)
                    infos.append(int(info))
                    residuals.append(inner_residual)
                    if info != 0 or inner_residual > self.inner_tolerance:
                        failed = 'group_gmres_failed' if info != 0 else 'true_group_residual_failed'
                        break
                # Fresh cache for the simultaneous terminal field. This also
                # measures partial sweeps on explicit inner failure.
                self._refresh(flux, moments)
                full_residual = self._full_residual(flux, moments, source, denominator)
                history.append(dict(sweep=sweep, completed_groups=len(infos),
                    inner_steps=tuple(steps), gmres_info=tuple(infos), group_residuals=tuple(residuals),
                    full_residual=full_residual, reason=failed or 'full_sweep'))
                if failed:
                    return self._result(flux, False, failed, full_residual, history, warm)
                if full_residual <= self.full_tolerance:
                    return self._result(flux, True, 'converged', full_residual, history, warm)
            return self._result(flux, False, 'max_sweeps', full_residual, history, warm)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            # A cached residual might predate a partially completed sweep.
            # Never label it as the residual of the returned terminal field.
            return self._result(flux, False, f'arithmetic_failure: {error}', None, history, warm)
