"""Bounded matrix-free native angular transport iteration; mathematical pilot only."""
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from .native_angular_source import _budget, _integer, _readonly
from .native_mesh_streaming import NativeMeshStreaming


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isreal(value):
        raise ValueError(f'{name} must be finite positive')
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite positive')
    return value


def _mapping(value):
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class FixedSourceResult:
    angular_flux: np.ndarray = field(repr=False)
    converged: bool
    reason: str
    gmres_info: int
    inner_steps: int
    true_residual: float | None

    def __post_init__(self):
        object.__setattr__(self, 'angular_flux', _readonly(self.angular_flux))


@dataclass(frozen=True)
class TransportResult:
    angular_flux: np.ndarray = field(repr=False)
    scalar_flux: np.ndarray = field(repr=False)
    k_eff: float | None
    converged: bool
    reason: str
    history: tuple
    diagnostics: dict
    provenance: dict

    def __post_init__(self):
        for name in ('angular_flux', 'scalar_flux'):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        object.__setattr__(self, 'history', tuple(_mapping(row) for row in self.history))
        object.__setattr__(self, 'diagnostics', _mapping(self.diagnostics))
        object.__setattr__(self, 'provenance', _mapping(self.provenance))


@dataclass(frozen=True)
class NativeMeshTransport:
    """Cold fission power iteration using zero-initial-guess GMRES for A y=F psi.

    Boundary conditions explicitly list vacuum/reflective for every boundary
    face. Only Cartesian-axis-aligned specular reflection is admitted. Scalar
    flux and cell-integrated kappa-fission heat must be >= -1e-12 times the
    declared physical scales. Angular negativity is reported separately.
    """
    streaming: NativeMeshStreaming
    boundary_conditions: tuple
    inner_tolerance: float = 1e-7
    k_tolerance: float = 1e-6
    source_tolerance: float = 1e-5
    full_tolerance: float = 1e-5
    max_inner: int = 200
    max_outer: int = 100
    restart: int = 30
    scalar_flux_scale: float = 1.
    power_scale: float = 1.
    max_bytes: int = 256*1024**2
    chi: np.ndarray = field(init=False, repr=False, compare=False)
    nu_sigma_f: np.ndarray = field(init=False, repr=False, compare=False)
    kappa_fission: np.ndarray = field(init=False, repr=False, compare=False)
    diagonal: np.ndarray = field(init=False, repr=False, compare=False)
    reflection_permutations: tuple = field(init=False, repr=False, compare=False)
    memory_estimate_bytes: int = field(init=False)
    closed_conservative: bool = field(init=False)

    def __post_init__(self):
        if not isinstance(self.streaming, NativeMeshStreaming):
            raise ValueError('NativeMeshStreaming required')
        for name in ('inner_tolerance', 'k_tolerance', 'source_tolerance', 'full_tolerance',
                     'scalar_flux_scale', 'power_scale'):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name, ceiling in (('inner_tolerance', 1e-7), ('k_tolerance', 1e-6),
                              ('source_tolerance', 1e-5), ('full_tolerance', 1e-5)):
            if getattr(self, name) > ceiling:
                raise ValueError(f'{name} may only tighten the default convergence gate')
        for name in ('max_inner', 'max_outer', 'restart'):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))
        if isinstance(self.boundary_conditions, str):
            raise ValueError('explicit boundary condition per boundary face required')
        bc = tuple(self.boundary_conditions)
        s = self.streaming
        if len(bc) != len(s.boundary_owners) or any(v not in ('vacuum', 'reflective') for v in bc):
            raise ValueError('one vacuum/reflective boundary condition per boundary face required')
        nc, na, ng = s.shape
        size = nc*na*ng
        restart = min(self.restart, self.max_inner, size)
        object.__setattr__(self, 'restart', restart)
        # Own persistent state, Krylov basis/Hessenberg, simultaneous operator,
        # fission, residual, old/new/terminal copies, diagnostics, reflection
        # workspace and scalar history (conservative numeric/Python allowance).
        a = (s.quadrature.lmax+1)**2
        numeric = 8*((restart+1)*size + 48*size + 4*(restart+1)**2 +
                     nc*a*ng + na*a + 6*na*ng + 4*len(bc)*na*ng +
                     6*len(s.source.materials)*ng + 8*nc*ng)
        estimate = numeric + 8192*self.max_outer
        _budget(estimate, self.max_bytes)
        object.__setattr__(self, 'memory_estimate_bytes', estimate)
        permutations = []
        directions = s.quadrature.directions
        for spec, normal in zip(bc, s.boundary_normals):
            if spec == 'vacuum':
                permutations.append(None)
                continue
            if abs(abs(normal[0])-1) <= 2e-13 and abs(normal[1]) <= 2e-13:
                axis, tangent = 'x', 1
            elif abs(abs(normal[1])-1) <= 2e-13 and abs(normal[0]) <= 2e-13:
                axis, tangent = 'y', 0
            else:
                raise ValueError('reflective normal must align with Cartesian x or y axis')
            permutation = s.quadrature.reflection(axis)
            speed = directions[:, :2] @ normal
            if (not np.array_equal(permutation[permutation], np.arange(na)) or
                    not np.array_equal(s.quadrature.weights[permutation], s.quadrature.weights) or
                    not np.allclose(directions[permutation, tangent], directions[:, tangent], atol=2e-13, rtol=0) or
                    not np.allclose(directions[permutation, 2], directions[:, 2], atol=2e-13, rtol=0) or
                    not np.allclose(speed[permutation], -speed, atol=2e-13, rtol=0)):
                raise ValueError('quadrature does not provide exact specular index reflection')
            permutations.append(permutation)
        chi, nsf, kappa = [], [], []
        for material in s.source.materials:
            for name, destination in (('chi', chi), ('nu_sigma_f', nsf), ('kappa_fission', kappa)):
                value = np.asarray(getattr(material, name))
                if (value.shape != (ng,) or value.dtype.kind not in 'fiu'
                        or not np.all(np.isfinite(value)) or np.any(value < 0)):
                    raise ValueError(f'native {name} must be finite nonnegative shape (n_groups,)')
                destination.append(value)
        diagonal = np.empty(s.shape)
        for cell, material in enumerate(s.source.cell_material_ids):
            diagonal[cell] = s.area[cell]*s.total[material]
        for (left, right), area, normal in zip(s.owners, s.measures, s.normals):
            coefficient = area*(directions[:, :2] @ normal)
            diagonal[left] += np.maximum(coefficient, 0)[:, None]
            diagonal[right] += np.maximum(-coefficient, 0)[:, None]
        for owner, area, normal in zip(s.boundary_owners, s.boundary_measures, s.boundary_normals):
            diagonal[owner] += np.maximum(area*(directions[:, :2] @ normal), 0)[:, None]
        for name, value in (('boundary_conditions', bc), ('reflection_permutations', tuple(permutations)),
                            ('chi', _readonly(np.stack(chi))), ('nu_sigma_f', _readonly(np.stack(nsf))),
                            ('kappa_fission', _readonly(np.stack(kappa))),
                            ('diagonal', _readonly(diagonal))):
            object.__setattr__(self, name, value)
        # A closed conservative system has a weighted particle-balance left
        # nullspace. Reject it rather than mistake roundoff inversion for a
        # physically finite fixed-source solution. This changes no native data.
        conservative = all(spec == 'reflective' for spec in bc)
        for mid in np.unique(s.source.cell_material_ids):
            sp0 = s.source.production[mid][0]
            balance = s.total[mid]-sp0.sum(axis=0)
            scale = np.abs(s.total[mid])+np.abs(sp0).sum(axis=0)
            conservative = conservative and bool(np.all(np.abs(balance) <= 64*np.finfo(float).eps*scale))
        object.__setattr__(self, 'closed_conservative', conservative)

    def _flux(self, value):
        return self.streaming._flux(value)

    def reflective_incoming(self, angular_flux):
        """Boundary traces: exact reflected outgoing ordinate, vacuum elsewhere."""
        flux = self._flux(angular_flux)
        s = self.streaming
        incoming = np.zeros((len(s.boundary_owners), s.shape[1], s.shape[2]))
        for face, permutation in enumerate(self.reflection_permutations):
            if permutation is not None:
                owner = s.boundary_owners[face]
                speed = s.quadrature.directions[:, :2] @ s.boundary_normals[face]
                indices = np.flatnonzero(speed < 0)
                incoming[face, indices] = flux[owner, permutation[indices]]
        return incoming

    def action(self, angular_flux):
        """A psi = L_stream psi - V*q_native(psi) - R_reflect psi."""
        flux = self._flux(angular_flux)
        s = self.streaming
        result = s.action(flux) - s.area[:, None, None]*s.source.scatter(flux)
        for face, permutation in enumerate(self.reflection_permutations):
            if permutation is not None:
                owner = s.boundary_owners[face]
                speed = s.quadrature.directions[:, :2] @ s.boundary_normals[face]
                indices = np.flatnonzero(speed < 0)
                result[owner, indices] += (s.boundary_measures[face]*speed[indices, None]
                                           * flux[owner, permutation[indices]])
        return s._finite(result)

    def scalar_flux(self, angular_flux):
        flux = self._flux(angular_flux)
        return self.streaming._finite(np.einsum('n,cng->cg', self.streaming.quadrature.weights, flux))

    def _cell_production(self, scalar_flux):
        s = self.streaming
        production = np.empty(len(s.area))
        for cell, material in enumerate(s.source.cell_material_ids):
            production[cell] = s.area[cell]*np.dot(self.nu_sigma_f[material], scalar_flux[cell])
        return s._finite(production)

    def production(self, angular_flux):
        """P(psi)=sum_c V[c] sum_g nuSigmaF[c,g] phi[c,g]."""
        value = float(self._cell_production(self.scalar_flux(angular_flux)).sum())
        if not np.isfinite(value):
            raise ValueError('nonfinite total fission production')
        return value

    def fission(self, angular_flux):
        """F psi = V*chi*(sum_g nuSigmaF_g*phi_g)/(4*pi), no chi normalization."""
        s = self.streaming
        production = self._cell_production(self.scalar_flux(angular_flux))
        result = np.empty(s.shape)
        for cell, material in enumerate(s.source.cell_material_ids):
            result[cell] = self.chi[material]*production[cell]/(4*np.pi)
        return s._finite(result)

    def fixed_source(self, rhs):
        """Solve A y=rhs from exactly zero; bound actual inner steps, not cycles.

        Convergence requires GMRES info==0 and an unpreconditioned true residual.
        A zero RHS denominator fails closed, as do zero streaming diagonals.
        """
        rhs = self._flux(rhs)
        shape = self.streaming.shape
        zero = np.zeros(shape)
        norm = float(np.linalg.norm(rhs.ravel()))
        if not np.isfinite(norm) or norm == 0:
            return FixedSourceResult(zero, False, 'zero_or_nonfinite_rhs_norm', -1, 0, None)
        if self.closed_conservative:
            return FixedSourceResult(zero, False, 'closed_conservative_singular_system', -1, 0, None)
        if np.any(self.diagonal <= 0):
            return FixedSourceResult(zero, False, 'zero_streaming_diagonal', -1, 0, None)
        size = rhs.size
        operator = LinearOperator((size, size), matvec=lambda x: self.action(x.reshape(shape)).ravel(), dtype=float)
        preconditioner = LinearOperator((size, size), matvec=lambda x: x/self.diagonal.ravel(), dtype=float)
        steps = 0

        def count_step(_):
            nonlocal steps
            steps += 1

        try:
            # legacy callback semantics make maxiter count inner Arnoldi steps.
            solution, info = gmres(operator, rhs.ravel(), x0=zero.ravel(), M=preconditioner,
                                   rtol=self.inner_tolerance, atol=0., restart=self.restart,
                                   maxiter=self.max_inner, callback=count_step, callback_type='legacy')
            solution = solution.reshape(shape)
            self.streaming._finite(solution)
            residual = float(np.linalg.norm((self.action(solution)-rhs).ravel())/norm)
            if not np.isfinite(residual):
                return FixedSourceResult(zero, False, 'nonfinite_true_inner_residual', int(info), steps, None)
            passed = info == 0 and residual <= self.inner_tolerance
            reason = 'converged' if passed else ('gmres_failed' if info != 0 else 'true_inner_residual_failed')
            return FixedSourceResult(solution, passed, reason, int(info), steps, residual)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            return FixedSourceResult(zero, False, f'inner_arithmetic_failure: {error}', -1, steps, None)

    def _diagnostics(self, flux):
        scalar = self.scalar_flux(flux)
        s = self.streaming
        neutron_production = self._cell_production(scalar)
        heat = np.empty(len(s.area))
        for cell, material in enumerate(s.source.cell_material_ids):
            heat[cell] = s.area[cell]*np.dot(self.kappa_fission[material], scalar[cell])
        s._finite(heat)
        negative_mass = float(np.einsum('c,n,cng->', s.area, s.quadrature.weights, np.maximum(-flux, 0)))
        result = dict(scalar_flux_min=float(scalar.min()), cell_heat_power_min=float(heat.min()),
                      cell_neutron_production_min=float(neutron_production.min()),
                      total_heat_power=float(heat.sum()), total_neutron_production=float(neutron_production.sum()),
                      angular_flux_min=float(flux.min()), weighted_negative_angular_mass=negative_mass,
                      scalar_nonnegative=bool(np.all(scalar >= -1e-12*self.scalar_flux_scale)),
                      heat_power_nonnegative=bool(np.all(heat >= -1e-12*self.power_scale)))
        if not np.all(np.isfinite([negative_mass, result['total_heat_power'], result['total_neutron_production']])):
            raise ValueError('nonfinite integrated diagnostics')
        return result

    def _result(self, flux, k, converged, reason, history):
        s = self.streaming
        try:
            scalar = self.scalar_flux(flux)
            diagnostics = self._diagnostics(flux)
        except (ValueError, FloatingPointError) as error:
            # Keep terminal arrays finite even if diagnostic arithmetic overflows.
            flux = np.zeros(s.shape)
            scalar = np.zeros((s.shape[0], s.shape[2]))
            diagnostics = dict(diagnostics_valid=False, diagnostic_error=str(error))
            converged = False
            reason = f'terminal_arithmetic_failure: {reason}'
        provenance = dict(cold_start=True, initial_angular_field='uniform_then_P_normalized',
                          linear_initial_guess='zero_every_outer_step', preconditioner='diagonal_streaming_collision',
                          boundary_conditions=self.boundary_conditions, physical_admission=False,
                          max_inner=self.max_inner, max_outer=self.max_outer, restart=self.restart,
                          inner_tolerance=self.inner_tolerance, k_tolerance=self.k_tolerance,
                          source_tolerance=self.source_tolerance, full_tolerance=self.full_tolerance,
                          scalar_flux_scale=self.scalar_flux_scale, power_scale=self.power_scale,
                          power_quantity='cell_integrated_kappa_fission_heat',
                          memory_estimate_bytes=self.memory_estimate_bytes, max_bytes=self.max_bytes,
                          energy_bounds_mev=tuple(s.source.energy_bounds_mev))
        return TransportResult(flux, scalar, k, converged, reason, tuple(history), diagnostics, provenance)

    def solve(self):
        """Cold fission power iteration with true inner and complete eigen residuals."""
        flux = np.ones(self.streaming.shape)
        history = []
        k = None
        try:
            production = self.production(flux)
        except (ValueError, FloatingPointError) as error:
            return self._result(flux, k, False, f'initial_arithmetic_failure: {error}', history)
        if production <= 0:
            return self._result(flux, k, False, 'nonpositive_initial_production', history)
        try:
            flux /= production
            diagnostics = self._diagnostics(flux)
        except (ValueError, FloatingPointError) as error:
            return self._result(flux, k, False, f'initial_arithmetic_failure: {error}', history)
        if not diagnostics['scalar_nonnegative'] or not diagnostics['heat_power_nonnegative']:
            return self._result(flux, k, False, 'initial_scalar_or_power_positivity_failed', history)
        old_k = 1.
        try:
            for iteration in range(1, self.max_outer+1):
                old_source = self.fission(flux)
                inner = self.fixed_source(old_source)
                row = dict(iteration=iteration, inner_info=inner.gmres_info,
                           inner_steps=inner.inner_steps, true_inner_residual=inner.true_residual)
                if not inner.converged:
                    row['reason'] = inner.reason
                    history.append(row)
                    return self._result(flux, k, False, f'inner_failure: {inner.reason}', history)
                next_k = self.production(inner.angular_flux)
                row['k_eff'] = next_k
                if not np.isfinite(next_k) or next_k <= 0:
                    row['reason'] = 'nonpositive_k'
                    history.append(row)
                    return self._result(flux, k, False, 'nonpositive_k', history)
                next_flux = inner.angular_flux/next_k
                diagnostics = self._diagnostics(next_flux)
                row.update(diagnostics)
                if not diagnostics['scalar_nonnegative'] or not diagnostics['heat_power_nonnegative']:
                    row['reason'] = 'scalar_or_power_positivity_failed'
                    history.append(row)
                    return self._result(next_flux, next_k, False, 'scalar_or_power_positivity_failed', history)
                next_source = self.fission(next_flux)
                source_norm = float(np.linalg.norm(next_source.ravel()))
                rhs_norm = float(np.linalg.norm((next_source/next_k).ravel()))
                if source_norm == 0 or rhs_norm == 0 or not np.isfinite(source_norm+rhs_norm):
                    row['reason'] = 'zero_or_nonfinite_outer_denominator'
                    history.append(row)
                    return self._result(next_flux, next_k, False, row['reason'], history)
                source_change = float(np.linalg.norm((next_source-old_source).ravel())/source_norm)
                full = float(np.linalg.norm((self.action(next_flux)-next_source/next_k).ravel())/rhs_norm)
                k_change = abs(next_k-old_k)/abs(next_k)
                row.update(k_change=k_change, source_change=source_change, full_residual=full)
                if not np.all(np.isfinite([source_change, full, k_change])):
                    row['reason'] = 'nonfinite_outer_residual'
                    history.append(row)
                    return self._result(next_flux, next_k, False, row['reason'], history)
                passed = (k_change <= self.k_tolerance and source_change <= self.source_tolerance
                          and full <= self.full_tolerance)
                row['reason'] = 'converged' if passed else 'continue'
                history.append(row)
                flux, k, old_k = next_flux, next_k, next_k
                if passed:
                    return self._result(flux, k, True, 'converged', history)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            return self._result(flux, k, False, f'outer_arithmetic_failure: {error}', history)
        return self._result(flux, k, False, 'outer_iteration_limit', history)
