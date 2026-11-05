"""Opt-in CPU float64 native scalar FV pilot; no physical qualification.

SP0 is [region, from, to]; F is the full [region, to, from] matrix.
Native imbalance and optional signed RABS are preserved, never repaired.
Memory admission is an estimate, not an operating-system memory limit: sparse
factorization and Python allocator peaks cannot be guaranteed in advance. For
physical-scale states (>100000 unknowns) an externally enforced process memory
cap is required. This module neither installs nor verifies that external cap.
"""

from dataclasses import dataclass
from time import perf_counter
import tracemalloc

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from .mesh import MeshGrid, MeshGroupOperator
from .corrected_mesh import CorrectedMeshGroupOperator
from .mesh_preconditioner import build_tpfa_ilu, build_corrected_ilu


def _frozen_array(value, dtype=np.float64):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError("array requires real numeric input; bool, complex, string and object are unsupported")
    a = np.asarray(raw, dtype=dtype)
    # A bytes-backed array cannot have writeability re-enabled by the caller.
    return np.frombuffer(a.tobytes(), dtype=dtype).reshape(a.shape)


@dataclass(frozen=True, eq=False)
class RegionalCoefficients:
    region_ids: tuple
    energy_bounds: np.ndarray
    D: np.ndarray
    T: np.ndarray
    SP0: np.ndarray
    F: np.ndarray
    scattering_orientation: str
    fission_orientation: str
    signed_rabs: np.ndarray | None = None
    kappa_fission: np.ndarray | None = None

    def __post_init__(self):
        ids = tuple(self.region_ids)
        if not ids or any(not isinstance(i, (int, np.integer)) or isinstance(i, bool) for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("region_ids must be explicit unique integers")
        object.__setattr__(self, "region_ids", ids)
        if self.scattering_orientation != "from,to" or self.fission_orientation != "to,from":
            raise ValueError("declare SP0 'from,to' and F 'to,from' orientation")
        for name in ("D", "T", "SP0", "F", "energy_bounds", "signed_rabs", "kappa_fission"):
            value = getattr(self, name)
            if value is not None:
                a = _frozen_array(value)
                if not np.all(np.isfinite(a)):
                    raise ValueError(f"{name} must be finite")
                object.__setattr__(self, name, a)
        if self.D.ndim != 2 or self.D.shape[0] != len(ids) or self.D.shape[1] < 1:
            raise ValueError("D requires shape (regions, groups)")
        r, g = self.D.shape
        if self.T.shape != (r, g) or np.any(self.D <= 0) or np.any(self.T <= 0):
            raise ValueError("D and T must be positive with matching shapes")
        if any(a.shape != (r, g, g) or np.any(a < 0) for a in (self.SP0, self.F)):
            raise ValueError("SP0 and F require nonnegative (regions, groups, groups)")
        if self.energy_bounds.shape != (g+1,) or np.any(self.energy_bounds < 0) or np.any(np.diff(self.energy_bounds) >= 0):
            raise ValueError("energy_bounds must be descending nonnegative group boundaries")
        if self.signed_rabs is not None and self.signed_rabs.shape != (r, g):
            raise ValueError("signed_rabs requires (regions, groups)")
        if self.kappa_fission is not None and (self.kappa_fission.shape != (r, g) or np.any(self.kappa_fission < 0)):
            raise ValueError("kappa_fission requires nonnegative (regions, groups)")


@dataclass(frozen=True)
class PilotResult:
    converged: bool
    reason: str
    k: float | None
    flux: np.ndarray
    heat: np.ndarray | None
    history: tuple
    diagnostics: dict


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class NativeScalarFV:
    """Cell-major (N,G) coupled loss/fission with bounded regional batches.

    Preconditioning is fixed per-group ILU (drop_tol=1e-6, fill_factor=20)
    or explicitly selected Jacobi; failures propagate without a fallback.
    ``external_memory_cap=True`` declares a separately enforced process cap.
    No state from an earlier inverse solve is reused as an initial guess.
    """

    def __init__(self, mesh, coefficients, cell_region_index, *, method,
                 preconditioner, max_bytes, restart, max_inner_iterations,
                 max_outer_iterations, batch_cells=1024,
                 boundary_conditions="reflective", external_memory_cap=False):
        if not isinstance(coefficients, RegionalCoefficients):
            raise TypeError("coefficients must be RegionalCoefficients")
        if method not in ("tpfa", "corrected") or preconditioner not in ("ilu", "jacobi"):
            raise ValueError("explicit method tpfa/corrected and preconditioner ilu/jacobi required")
        self.coefficients = coefficients
        self.restart = _positive_integer(restart, "restart")
        self.max_inner_iterations = _positive_integer(max_inner_iterations, "max_inner_iterations")
        self.max_outer_iterations = _positive_integer(max_outer_iterations, "max_outer_iterations")
        self.batch_cells = _positive_integer(batch_cells, "batch_cells")
        max_bytes = _positive_integer(max_bytes, "max_bytes")
        n, g = mesh.n_cells, coefficients.D.shape[1]
        index = np.asarray(cell_region_index)
        if index.shape != (n,) or index.dtype.kind not in "iu" or np.any(index < 0) or np.any(index >= len(coefficients.region_ids)):
            raise ValueError("cell_region_index requires integer (N,) regional row indices in bounds")
        self.cell_region_index = _frozen_array(index, np.int64)
        index = self.cell_region_index
        self.shape = (n, g)
        if n*g > 100000 and external_memory_cap is not True:
            raise ValueError("physical-scale setup requires an externally enforced memory cap")
        # Degree-squared bound for corrected two-hop stencils; count Python
        # dictionaries, gather matrices, GMRES vectors and factor headroom.
        degree = np.zeros(n, dtype=np.int64)
        for i, j, *_ in mesh.faces:
            degree[i] += 1
            degree[j] += 1
        width = min(n, int((degree.max(initial=0)+1)**2))
        matrix_bytes = n*g*max(1, width)*512
        factor_headroom = matrix_bytes*20 if preconditioner == "ilu" else 0
        estimate = (8*n*g*(self.restart+20) + matrix_bytes + factor_headroom
                    + sum(a.nbytes for a in (coefficients.D, coefficients.T, coefficients.SP0, coefficients.F,
                                             coefficients.energy_bounds, coefficients.signed_rabs,
                                             coefficients.kappa_fission) if a is not None)
                    + 8*min(n, self.batch_cells)*(g*g+4*g) + 16*n
                    + 16384*self.max_outer_iterations)
        if estimate > max_bytes:
            raise MemoryError(f"pre-build memory estimate {estimate} exceeds max_bytes {max_bytes}")
        self.diagnostics = dict(physical_qualification=False, method=method,
                                preconditioner=preconditioner, estimated_bytes=estimate,
                                external_memory_cap_declared=external_memory_cap,
                                factor_memory_estimate_guaranteed=False)
        start = perf_counter()
        tracing_owned = not tracemalloc.is_tracing()
        if tracing_owned:
            tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]
        try:
            self.grid = MeshGrid(mesh, boundary_conditions)
            self.volumes = _frozen_array(self.grid.integration_weights.reshape(-1))
            self.groups, self.preconditioners = [], []
            for group in range(g):
                d = coefficients.D[index, group]
                removal = coefficients.T[index, group] - coefficients.SP0[index, group, group]
                constructor = MeshGroupOperator if method == "tpfa" else CorrectedMeshGroupOperator
                op = constructor(np, self.grid, d, removal)
                self.groups.append(op)
                if preconditioner == "ilu":
                    fixed = (build_tpfa_ilu(op, np) if method == "tpfa" else
                             build_corrected_ilu(op.host_matrix, self.grid.shape, np))
                else:
                    diag = op.diag.copy()
                    if not np.all(np.isfinite(diag)) or np.any(diag == 0):
                        raise ValueError("Jacobi requires finite nonzero group diagonals")
                    def fixed(r, diag=diag):
                        return r/diag
                self.preconditioners.append(fixed)
        finally:
            current, peak = tracemalloc.get_traced_memory()
            self.diagnostics.update(setup_seconds=perf_counter()-start,
                                    setup_traced_retained_bytes=max(0, current-before),
                                    setup_traced_peak_bytes=peak,
                                    setup_tracing_exclusive=tracing_owned,
                                    memory_measurement_scope="tracemalloc; native allocations may be untracked")
            if tracing_owned:
                tracemalloc.stop()
        if self.diagnostics["setup_traced_retained_bytes"] > max_bytes or (tracing_owned and peak > max_bytes):
            raise MemoryError("measured setup memory exceeds max_bytes")
        self.diagnostics["factor_nnz"] = tuple(getattr(p, "nnz", None) for p in self.preconditioners)

    def _state(self, phi):
        raw = np.asarray(phi)
        if raw.dtype.kind not in "iuf":
            raise ValueError("state requires real numeric input")
        a = np.asarray(raw, dtype=np.float64)
        if a.shape != self.shape or not np.all(np.isfinite(a)):
            raise ValueError("state requires finite cell-major (N,G)")
        return a

    def _regional_action(self, phi, fission):
        out = np.empty(self.shape, dtype=np.float64)
        c = self.coefficients
        for first in range(0, self.shape[0], self.batch_cells):
            sl = slice(first, first+self.batch_cells)
            rows = self.cell_region_index[sl]
            if fission:
                out[sl] = np.einsum("bij,bj->bi", c.F[rows], phi[sl])
            else:
                # SP0 includes its diagonal; subtract it here because each
                # group operator already contains T-SP0[g,g].
                out[sl] = (np.einsum("bij,bi->bj", c.SP0[rows], phi[sl])
                           - np.diagonal(c.SP0[rows], axis1=1, axis2=2)*phi[sl])
            out[sl] *= self.volumes[sl, None]
        return out

    def apply_A(self, phi):
        phi = self._state(phi)
        out = -self._regional_action(phi, False)
        for g, op in enumerate(self.groups):
            out[:, g] += op.apply(phi[:, g]).reshape(-1)
        return out

    def apply_F(self, phi):
        return self._regional_action(self._state(phi), True)

    def apply_heat(self, phi):
        """Integrated energy production, unavailable without kappa_fission."""
        phi = self._state(phi)
        kappa = self.coefficients.kappa_fission
        if kappa is None:
            return None
        heat = np.empty(self.shape[0], dtype=np.float64)
        for first in range(0, self.shape[0], self.batch_cells):
            sl = slice(first, first+self.batch_cells)
            heat[sl] = self.volumes[sl]*np.sum(kappa[self.cell_region_index[sl]]*phi[sl], axis=1)
        return heat

    def solve(self):
        """Cold coupled inverse/power iteration with immutable gates.

        Flux/heat must satisfy both min >= -1e-12 and min/max(abs) >=
        -1e-12, with zero scale admitted. Rejection returns the last finite
        accepted iterate; k is None until an inverse iterate is accepted.
        """
        shape, size = self.shape, int(np.prod(self.shape))
        history = []
        phi, k = np.ones(shape, dtype=np.float64), None
        state_label = "finite_seed_no_eigenvalue"
        metadata = dict(self.diagnostics)
        def result(ok, reason):
            if not ok:
                if history:
                    history[-1]["failure"] = reason
                else:
                    history.append(dict(failure=reason))
            heat = self.apply_heat(phi)
            heat_label = "unavailable_no_kappa" if heat is None else "finite"
            if heat is not None and not np.all(np.isfinite(heat)):
                heat, heat_label = None, "unavailable_nonfinite_seed_heat"
            metadata.update(returned_state=state_label, returned_flux_finite=True,
                            returned_k_matches_flux=k is not None,
                            returned_heat_state=heat_label)
            return PilotResult(ok, reason, k, _frozen_array(phi), None if heat is None else _frozen_array(heat),
                               tuple(history), metadata)
        def precondition(v):
            r = v.reshape(shape)
            return np.column_stack([p(r[:, g].reshape(self.grid.shape)).reshape(-1)
                                    for g, p in enumerate(self.preconditioners)]).reshape(-1)
        A = LinearOperator((size, size), matvec=lambda v: self.apply_A(v.reshape(shape)).reshape(-1), dtype=np.float64)
        M = LinearOperator((size, size), matvec=precondition, dtype=np.float64)
        source = self.apply_F(phi)
        production = float(source.sum())
        metadata["initial_production"] = production
        if not np.isfinite(production) or production <= 0:
            return result(False, "nonpositive or nonfinite initial production")
        initial_phi = phi/production
        initial_source = source/production
        if not np.all(np.isfinite(initial_phi)) or not np.all(np.isfinite(initial_source)):
            return result(False, "nonfinite initial normalized flux/source")
        initial_heat = self.apply_heat(initial_phi)
        flux_ok, metadata["initial_flux_positivity"] = _positivity(initial_phi)
        heat_ok, metadata["initial_heat_positivity"] = _positivity(initial_heat)
        if not flux_ok or not heat_ok:
            return result(False, "negative or nonfinite initial normalized flux/heat")
        phi, source = initial_phi, initial_source
        state_label = "finite_normalized_seed_no_eigenvalue"
        for outer in range(1, self.max_outer_iterations+1):
            count = 0
            def callback(_):
                nonlocal count
                count += 1
                if count > self.max_inner_iterations:
                    raise _IterationBudget("inner iteration budget exhausted")
            try:
                x, info = gmres(A, source.reshape(-1), x0=np.zeros(size), M=M,
                                rtol=1e-7, atol=0., restart=self.restart,
                                maxiter=self.max_inner_iterations, callback=callback,
                                callback_type="legacy")
            except _IterationBudget as exc:
                history.append(dict(outer=outer, inner_iterations=count, failure=str(exc)))
                return result(False, str(exc))
            if not np.all(np.isfinite(x)):
                history.append(dict(outer=outer, inner_iterations=count, gmres_info=int(info)))
                return result(False, "GMRES returned nonfinite flux")
            inner = float(np.linalg.norm(A@x-source.reshape(-1))/np.linalg.norm(source))
            entry = dict(outer=outer, inner_iterations=count, inner_residual=inner, gmres_info=int(info))
            history.append(entry)
            if info != 0 or not np.isfinite(inner) or inner > 1e-7:
                return result(False, "GMRES failed true inner residual gate")
            candidate = x.reshape(shape)
            candidate_source = self.apply_F(candidate)
            candidate_k = float(candidate_source.sum())
            entry["candidate_production"] = candidate_k
            candidate_heat = self.apply_heat(candidate)
            flux_ok, entry["candidate_flux_positivity"] = _positivity(candidate)
            heat_ok, entry["candidate_heat_positivity"] = _positivity(candidate_heat)
            if not np.isfinite(candidate_k) or candidate_k <= 0:
                return result(False, "nonpositive or nonfinite production")
            entry["minimum_flux"] = float(np.min(candidate))
            entry["minimum_heat"] = None if candidate_heat is None else float(np.min(candidate_heat))
            if not flux_ok or not heat_ok:
                return result(False, "negative or nonfinite flux/heat")
            new_phi = candidate/candidate_k
            new_source = candidate_source/candidate_k
            flux_ok, entry["normalized_flux_positivity"] = _positivity(new_phi)
            entry["normalized_source_finite"] = bool(np.all(np.isfinite(new_source)))
            if not np.all(np.isfinite(new_phi)) or not np.all(np.isfinite(new_source)):
                return result(False, "nonfinite normalized flux/source")
            heat = self.apply_heat(new_phi)
            heat_ok, entry["normalized_heat_positivity"] = _positivity(heat)
            if not flux_ok or not heat_ok:
                return result(False, "negative normalized flux/heat")
            source_residual = float(np.linalg.norm(new_source-source)/np.linalg.norm(new_source))
            full = float(np.linalg.norm(self.apply_A(new_phi)-new_source/candidate_k)/np.linalg.norm(new_source/candidate_k))
            dk = float("inf") if k is None else abs(candidate_k-k)
            entry.update(k=candidate_k, k_change_absolute=dk, source_residual=source_residual, full_residual=full)
            if not np.isfinite(source_residual) or not np.isfinite(full):
                return result(False, "nonfinite candidate convergence residual")
            # Commit the complete finite, admissible iterate together. Later
            # rejection leaves this flux and its associated k untouched.
            phi, source, k = new_phi, new_source, candidate_k
            state_label = "finite_accepted_iterate"
            entry["accepted"] = True
            if dk <= 1e-6 and source_residual <= 1e-5 and np.isfinite(full) and full <= 1e-5:
                return result(True, "all strict scalar pilot gates passed")
        return result(False, "outer iteration budget exhausted")


class _IterationBudget(RuntimeError):
    pass


def _positivity(values):
    """Require absolute AND scale-relative >= -1e-12, without a unit floor."""
    if values is None:
        return True, dict(available=False)
    finite = bool(np.all(np.isfinite(values)))
    if not finite:
        return False, dict(available=True, finite=False)
    minimum = float(np.min(values))
    scale = float(np.max(np.abs(values)))
    relative_minimum = minimum/scale if scale else 0.
    relative = values/scale if scale else values
    total = float(np.sum(np.abs(relative)))
    negative_fraction = float(-np.sum(np.minimum(relative, 0.))/total) if total else 0.
    absolute_ok, relative_ok = minimum >= -1e-12, relative_minimum >= -1e-12
    return absolute_ok and relative_ok, dict(
        available=True, finite=True, minimum=minimum, scale=scale,
        minimum_over_scale=relative_minimum, negative_fraction=negative_fraction,
        negative_entry_fraction=float(np.mean(values < 0)),
        absolute_gate_passed=absolute_ok, relative_gate_passed=relative_ok)
