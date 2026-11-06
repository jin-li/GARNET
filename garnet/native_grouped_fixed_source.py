"""Opt-in 2D/3D group Gauss--Seidel fixed-source pilot; no eigenvalue solve."""
from dataclasses import dataclass, field
import hashlib
import inspect
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.sparse.linalg._isolve import iterative as _scipy_iterative, utils as _scipy_utils

from .native_angular_source import _budget, _integer
from .native_grouped_angular import GroupAngularOperators
from .native_mesh_streaming import NativeMeshStreaming


_REVIEWED_SOURCES = (
    ('iterative.py', '0fe686ee49308a66e59a4f610346a8ba9f6d5eb7166f84748f7558420c938a70'),
    ('utils.py', '3b4aa7a964e336795a4ff2e8ee7594046ba76d9a5eba758fe1a99d81f13cb6b0'),
)


@dataclass(frozen=True)
class _WorkspaceContract:
    name: str
    group_vectors: int
    source_hashes: tuple
    reason: str


def _source_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            digest.update(block)
    return digest.hexdigest()


def _gmres_workspace_contract():
    """Bind the smaller allowance to reviewed source and the actual callable.

    Read/hash source bytes only; never evaluate source. A version string alone
    cannot authorize the reviewed 10-vector allowance. Recheck at every solve
    so replacing GMRES or its source after construction cannot keep that bound.
    """
    hashes = ()
    try:
        if (gmres is not _scipy_iterative.gmres or not inspect.isfunction(gmres)
                or gmres.__globals__ is not vars(_scipy_iterative)
                or gmres.__module__ != _scipy_iterative.__name__
                or _scipy_iterative.make_system is not _scipy_utils.make_system):
            raise ValueError('GMRES callable or make_system binding differs')
        iterative_path = Path(inspect.getsourcefile(_scipy_iterative)).resolve()
        utils_path = Path(inspect.getsourcefile(_scipy_utils)).resolve()
        if (Path(gmres.__code__.co_filename).resolve() != iterative_path
                or Path(_scipy_utils.make_system.__code__.co_filename).resolve() != utils_path):
            raise ValueError('GMRES/helper code filename differs from module source')
        hashes = (('iterative.py', _source_digest(iterative_path)),
                  ('utils.py', _source_digest(utils_path)))
        if hashes == _REVIEWED_SOURCES:
            return _WorkspaceContract('reviewed_scipy_1_18_1_10G', 10, hashes, 'reviewed_source_hashes_match')
        reason = 'GMRES/helper source hashes differ from reviewed implementation'
    except (OSError, TypeError, ValueError, AttributeError) as error:
        reason = f'GMRES source contract unavailable: {error}'
    return _WorkspaceContract('conservative_32G', 32, hashes, reason)


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be finite positive')
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite positive')
    return value


def _readonly(value):
    """Freeze the terminal field without a full coupled finite-check mask."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError('terminal field must have cell/angle/group layout')
    for group in range(array.shape[2]):
        for start in range(0, array.shape[0], 512):
            if not np.all(np.isfinite(array[start:start+512, :, group])):
                raise ValueError('terminal field must be finite')
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


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
    workspace_contract: _WorkspaceContract = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.streaming, NativeMeshStreaming):
            raise ValueError('NativeMeshStreaming required')
        for name in ('inner_tolerance', 'full_tolerance'):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name in ('restart', 'max_inner', 'max_sweeps'):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))
        nc, na, _ = self.streaming.shape
        group_size = nc*na
        restart = min(self.restart, self.max_inner, group_size)
        object.__setattr__(self, 'restart', restart)
        contract = _gmres_workspace_contract()
        estimate = self._memory_estimate(contract)
        _budget(estimate, self.max_bytes)
        object.__setattr__(self, 'workspace_contract', contract)
        object.__setattr__(self, 'memory_estimate_bytes', estimate)
        object.__setattr__(self, 'operators', GroupAngularOperators(
            self.streaming, self.boundary_conditions, max_bytes=self.max_bytes))

    def _memory_estimate(self, contract):
        nc, na, ng = self.streaming.shape
        h = self.streaming.quadrature.harmonics.shape[1]
        group_size, full_size, moments = nc*na, nc*na*ng, nc*h*ng
        restart, batch = self.restart, min(nc, 512)
        # Reproduce the public operator bound before constructing its reflection
        # permutations. No large numeric allocation precedes this preflight.
        operator_bytes = 8*(3*nc*na + 2*nc*h + na*h +
                            batch*(4*h*ng+8*h+8*na+16) + 100*na+1)
        # Three full fields: owned RHS, iterate, immutable result even while
        # exception tracebacks retain GMRES frames. Exactly one coupled cache.
        # Reviewed 10G: RHS+diagonal (2), passed x0+SciPy x (2), r/av/w (3),
        # plus three transient vectors. The operator bound separately covers
        # projection, local harmonic fields and norm temporaries. Unreviewed
        # implementations retain the conservative 32G workspace allowance.
        return operator_bytes + 8*(3*full_size + moments +
                    (restart+1)*group_size + contract.group_vectors*group_size + 6*(restart+1)**2 +
                    4*self.max_sweeps*(ng+4))

    def _guard(self):
        contract = _gmres_workspace_contract()
        estimate = self._memory_estimate(contract)
        _budget(estimate, self.max_bytes)
        return contract, estimate

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
            del residual
        value = total/denominator
        if not math.isfinite(value):
            raise ValueError('nonfinite simultaneous residual')
        return value

    def _result(self, flux, converged, reason, residual, history, warm, workspace):
        contract, estimate = workspace
        return GroupedFixedSourceResult(flux, converged, reason, residual, tuple(history), dict(
            initial_field='caller_supplied' if warm else 'zero',
            group_initial_guess='latest_group_iterate', group_order=tuple(range(self.streaming.shape[2])),
            preconditioner='one_group_streaming_collision_diagonal',
            residual='simultaneous_L2_relative_rhs_or_absolute_for_zero_rhs',
            inner_tolerance=self.inner_tolerance, full_tolerance=self.full_tolerance,
            restart=self.restart, max_inner=self.max_inner, max_sweeps=self.max_sweeps,
            boundary_conditions=self.boundary_conditions, memory_estimate_bytes=estimate,
            workspace_contract=contract.name, group_workspace_vectors=contract.group_vectors,
            gmres_source_hashes=contract.source_hashes, workspace_contract_reason=contract.reason,
            max_bytes=self.max_bytes, physical_admission=False, eigenvalue_solver=False))

    def _solve_group(self, source, flux, moments, group):
        """Commit one group and return scalars only; no arrays survive success.

        The frame bounds RHS, diagonal, closures, candidate, true residual and
        candidate moments to one group call. In particular none can overlap
        the next group's GMRES basis. Exception tracebacks may retain this
        frame through result freezing; the preflight accounts that peak.
        """
        group_rhs = self.operators.scatter_to_group(moments, group, exclude_self=True)
        group_rhs += source[:, :, group]
        self.operators._finite(group_rhs)
        rhs_norm = self._norm(group_rhs)
        diagonal = self.operators.diagonal_group(group)
        if np.any(diagonal <= 0):
            return 0, None, None, 'zero_streaming_diagonal'
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
        # Projection precedes commit, preserving the last finite flux/cache
        # pair if projection arithmetic fails.
        candidate_moments = self.operators.project_group(candidate)
        flux[:, :, group] = candidate
        moments[:, :, group] = candidate_moments
        failed = None
        if info != 0 or inner_residual > self.inner_tolerance:
            failed = 'group_gmres_failed' if info != 0 else 'true_group_residual_failed'
        return count, int(info), inner_residual, failed

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
        workspace = self._guard()
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
                return self._result(flux, True, reason, full_residual, history, warm, workspace)
            for sweep in range(1, self.max_sweeps+1):
                steps, infos, residuals = [], [], []
                failed = None
                for group in range(self.streaming.shape[2]):
                    count, info, inner_residual, failed = self._solve_group(source, flux, moments, group)
                    if info is not None:
                        steps.append(count)
                        infos.append(info)
                        residuals.append(inner_residual)
                    if failed:
                        break
                # Fresh cache for the simultaneous terminal field. This also
                # measures partial sweeps on explicit inner failure.
                self._refresh(flux, moments)
                full_residual = self._full_residual(flux, moments, source, denominator)
                history.append(dict(sweep=sweep, completed_groups=len(infos),
                    inner_steps=tuple(steps), gmres_info=tuple(infos), group_residuals=tuple(residuals),
                    full_residual=full_residual, reason=failed or 'full_sweep'))
                if failed:
                    return self._result(flux, False, failed, full_residual, history, warm, workspace)
                if full_residual <= self.full_tolerance:
                    return self._result(flux, True, 'converged', full_residual, history, warm, workspace)
            return self._result(flux, False, 'max_sweeps', full_residual, history, warm, workspace)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            # A cached residual might predate a partially completed sweep.
            # Never label it as the residual of the returned terminal field.
            return self._result(flux, False, f'arithmetic_failure: {error}', None, history, warm, workspace)
