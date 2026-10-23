"""Static physical kinetics for the standard symmetric finite-volume diffusion model.

This is a separate baseline from NEM and from the critically adjusted transient
projection in :mod:`garnet.quasistatic`.  ``Kinetics.beta`` supplies raw production
fractions independent of incident energy, globally or by material.  Its common
family decay constants and global/family delayed spectra are model assumptions,
not an adapter for arbitrary regional Serpent delayed-production data.  Blended
cells use the existing flat-flux fission-production weighting approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backend import asnumpy
from .materials import Kinetics
from .quasistatic import (_flux_array, _loss_apply, _mapped_kinetics, _source_weight,
                          project_effective_kinetics)
from .solver import DiffusionEigenSolver, Result


@dataclass(frozen=True)
class DiffusionKinetics:
    """Verified static FV projection; time units seconds, decay units s^-1.

    ``generation_time`` is physical Lambda = T/F without critical adjustment.
    ``lifetime`` is the discrete k-mode lifetime k*Lambda, not an assertion of
    equality with a particular Monte Carlo lifetime estimator.  Importance
    integrals use the assembled symmetric FV equation weights; a common cell
    volume factor cancels from all reported ratios. Shapes are normalized to
    unit maximum absolute value before diagnostics and importance integrals.
    """

    k_eff: float
    generation_time: float
    lifetime: float
    beta: np.ndarray
    decay: np.ndarray
    time_importance: float
    fission_importance: float
    forward_residual: float
    adjoint_residual: float
    adjoint_k_eff: float

    @property
    def beta_total(self) -> float:
        return float(self.beta.sum())

    @property
    def lambda_low_frequency(self) -> float | None:
        """One-family reduction matching sum(beta_i/lambda_i); absent if beta=0."""
        return (self.beta_total / float(np.sum(self.beta / self.decay))
                if self.beta_total > 0 else None)


def _eigenvalue(value, explicit, name):
    if isinstance(value, Result):
        if not value.converged:
            raise ValueError(f"{name} Result is not converged")
        native = float(value.k_eff)
        if explicit is not None and not np.isclose(
                explicit, native, rtol=1e-12, atol=0):
            raise ValueError(f"{name} explicit eigenvalue conflicts with Result")
        explicit = native
    if explicit is None or not np.isfinite(explicit) or explicit <= 0:
        raise ValueError(f"{name} eigenvalue must be supplied, finite and positive")
    return float(explicit)


def project_diffusion_kinetics(solver, forward, adjoint, kinetics: Kinetics, *,
                               k_eff=None, adjoint_k_eff=None,
                               residual_tolerance=1e-7,
                               k_tolerance=1e-7) -> DiffusionKinetics:
    """Project a verified matching static forward/adjoint FV eigenpair.

    Accept solver Results or group-first arrays. Arrays require their respective
    explicit eigenvalues. Both full equations L phi=F phi/k and
    L.T star=F.T star/k are checked, including their amplitude components.
    Only the exact standard DiffusionEigenSolver with symmetric Cartesian or
    cylindrical FV operators is qualified. NEM, SPN and unstructured mesh
    models need their own adjoints and physical forms.
    """
    if type(solver) is not DiffusionEigenSolver or not solver.symmetric_operator:
        raise TypeError("requires the standard symmetric FV DiffusionEigenSolver")
    if not isinstance(kinetics, Kinetics):
        raise TypeError("kinetics must be a Kinetics object")
    for name, tolerance in (("residual_tolerance", residual_tolerance),
                            ("k_tolerance", k_tolerance)):
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError(f"{name} must be finite and positive")
    k = _eigenvalue(forward, k_eff, "forward")
    ka = _eigenvalue(adjoint, adjoint_k_eff, "adjoint")
    if abs(k-ka) / max(k, ka) > k_tolerance:
        raise ValueError("forward and adjoint eigenvalues disagree")
    # Kinetics' transient-compatible constructor is intentionally permissive;
    # static qualification checks finite physical data before any mapping.
    for values in (kinetics.velocities, kinetics.beta, kinetics.decay):
        if np.any(~np.isfinite(values)):
            raise ValueError("kinetics inputs must be finite")
    if (np.any(kinetics.velocities <= 0) or np.any(kinetics.decay <= 0)
            or np.any(kinetics.beta < 0)):
        raise ValueError("speeds/decay must be positive and raw beta nonnegative")
    if np.any(kinetics.beta.sum(axis=-1) >= 1):
        raise ValueError("raw beta fractions must sum to less than one")
    chi = kinetics.chi_delayed
    if chi is not None and (chi.shape[-1] != solver.n_groups
                            or np.any(~np.isfinite(chi)) or np.any(chi < 0)
                            or not np.allclose(chi.sum(axis=-1), 1)):
        raise ValueError("delayed spectra must be finite normalized group vectors")
    xp = solver.xp
    shape = tuple(solver.grid.shape)
    phi = _flux_array(forward, "forward flux", solver.n_groups, shape, xp, solver.dtype)
    star = _flux_array(adjoint, "adjoint flux", solver.n_groups, shape, xp, solver.dtype)
    def normalized(vector):
        scale = float(xp.max(xp.abs(vector)))
        if not np.isfinite(scale) or scale == 0:
            raise ValueError("eigenvectors must be finite and nonzero")
        return vector / scale
    phi, star = normalized(phi), normalized(star)
    weight = _source_weight(solver)
    host_weight = asnumpy(weight)
    if np.any(~np.isfinite(host_weight)) or np.any(host_weight <= 0):
        raise ValueError("FV source weights must be finite and positive")

    _, beta_fields = _mapped_kinetics(solver, kinetics)
    # Freeze the total production operator chi_g * nuSigma_f,h. Delayed
    # production is a split of this total, not an additional source.
    total_scale = max(float(xp.max(solver.chi[g] * solver.nu_sigma_f[h]))
                      for g in range(solver.n_groups)
                      for h in range(solver.n_groups))
    for g in range(solver.n_groups):
        delayed_spectrum = sum(beta_i * (solver.chi[g] if chi is None else
                               float(chi[g] if chi.ndim == 1 else chi[i, g]))
                               for i, beta_i in enumerate(beta_fields))
        for h in range(solver.n_groups):
            remainder = (solver.chi[g] - delayed_spectrum) * solver.nu_sigma_f[h]
            if (not bool(xp.all(xp.isfinite(remainder)))
                    or bool(xp.any(remainder < -1e-13 * max(total_scale, np.finfo(float).tiny)))):
                raise ValueError("delayed production exceeds total production; prompt remainder is negative")

    def residual(vector, eigenvalue, transpose):
        loss = _loss_apply(solver, vector, adjoint=transpose)
        prod = solver.chi if transpose else solver.nu_sigma_f
        emit = solver.nu_sigma_f if transpose else solver.chi
        source = sum(prod[g] * vector[g] for g in range(solver.n_groups))
        rhs = [weight * emit[g] * source / eigenvalue
               for g in range(solver.n_groups)]
        if any(not bool(xp.all(xp.isfinite(t))) for t in loss + rhs):
            raise ValueError("nonfinite complete diffusion eigen diagnostics")
        def norm(terms):
            scale = max(float(xp.max(xp.abs(t))) for t in terms)
            return (scale * float(xp.sqrt(sum(xp.sum((t/scale)**2) for t in terms)))
                    if scale else 0.)
        denominator = max(norm(loss), norm(rhs))
        if denominator == 0:
            raise ValueError("zero eigenvector/operator importance")
        return norm([a-b for a, b in zip(loss, rhs)]) / denominator

    rf, ra = residual(phi, k, False), residual(star, ka, True)
    if not np.isfinite(rf + ra) or max(rf, ra) > residual_tolerance:
        raise ValueError(f"complete diffusion eigen residual exceeds tolerance: {rf:g}, {ra:g}")
    # k0=1 explicitly removes the legacy transient critical adjustment.
    projected = project_effective_kinetics(solver, phi, star, kinetics, k0=1.0)
    if (not np.all(np.isfinite([projected.generation_time,
                               k * projected.generation_time,
                               projected.time_importance,
                               projected.fission_importance]))
            or not np.all(np.isfinite(projected.beta))
            or np.any(projected.beta < 0) or projected.beta.sum() >= 1):
        raise ValueError("nonfinite or unphysical projected diffusion kinetics")
    return DiffusionKinetics(
        k, projected.generation_time, k * projected.generation_time,
        projected.beta.copy(), projected.decay.copy(),
        projected.time_importance, projected.fission_importance, rf, ra, ka)
