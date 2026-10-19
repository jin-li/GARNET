"""Opt-in homogeneous B→0 moment-hierarchy condensation, CPU NumPy only.

Raw production matrices use [from, to]; losses use [to, from]. This is a
solution-dependent projection of four leading homogeneous moment balances,
not a native current-weight correction or preservation at finite buckling.
Higher weights and effective coefficients may be signed. No regularization,
clipping, pseudoinverse, spectrum renormalization or physical qualification is
provided. Optional heat must be the actual heat-production vector.
"""
from dataclasses import dataclass
import numpy as np
from .moment_materials import MomentMaterial, _immutable


@dataclass(frozen=True)
class CondensedMomentHierarchy:
    total: np.ndarray
    chi: np.ndarray
    nu_sigma_f: np.ndarray
    fission: np.ndarray
    heat_production: object
    aggregation: np.ndarray
    partition: tuple
    fine_weights: dict
    weights: dict
    injections: dict
    loss: dict
    production: dict
    diagonal_correction: dict
    k_infinity: float
    diagnostics: dict
    provenance: dict

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _immutable(getattr(self, name)))


def _finite_vector(value, n, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (n,) or not np.all(np.isfinite(a)):
        raise ValueError('Invalid finite '+name)
    return a


def _condition(a, limit, name):
    if not np.all(np.isfinite(a)):
        raise ValueError('Nonfinite '+name)
    condition = float(np.linalg.cond(a))
    if not np.isfinite(condition) or condition > limit:
        raise ValueError('Singular/ill-conditioned '+name)
    return condition


def _backward(a, x, b):
    denominator = np.linalg.norm(a, 2)*np.linalg.norm(x)+np.linalg.norm(b)
    numerator = np.linalg.norm(a@x-b)
    if denominator == 0:
        return 0. if numerator == 0 else float('inf')
    return float(numerator/denominator)


def condense_moment_hierarchy(total, production, chi, nu_sigma_f, groups, *,
                              heat_production=None, cancellation_threshold=1e-10,
                              condition_limit=1e12, backward_tolerance=1e-12):
    """Condense an exhaustive disjoint index partition using orders 0..3.

    psi0 = solve(R0, chi)/k_inf, k_inf = nu_sigma_f @ solve(R0, chi).
    psi_l = l/(2*l+1) solve(R_l, psi_(l-1)), R_l = diag(T)-SP_l.T.
    D_l[i,G] = psi_l[i]/sum_G psi_l; R_lc = C R_l D_l.

    Scalar psi0 must be strictly positive in every fine group and fission must
    have support. Higher weights may be signed. Every group/order must satisfy
    rho=abs(sum(psi))/sum(abs(psi)) > cancellation_threshold, a scale-invariant
    gate. Both fine and coarse losses must meet condition_limit. Solve and
    projected-hierarchy normwise backward residuals must meet backward_tolerance.
    Defaults are numerical refusal thresholds, not uncertainty qualifications.
    """
    for value, name, low, high in ((cancellation_threshold, 'cancellation threshold', 0., 1.),
                                   (backward_tolerance, 'backward tolerance', 0., 1.)):
        if not np.isfinite(value) or not low < value < high:
            raise ValueError('Invalid '+name)
    if not np.isfinite(condition_limit) or condition_limit <= 1:
        raise ValueError('Invalid condition limit')
    total = np.asarray(total, dtype=float)
    if total.ndim != 1 or not total.size or not np.all(np.isfinite(total)) or np.any(total <= 0):
        raise ValueError('Finite positive total required')
    n = total.size
    chi = _finite_vector(chi, n, 'chi')
    nuf = _finite_vector(nu_sigma_f, n, 'nu_sigma_f')
    if np.any(chi < 0) or np.any(nuf < 0) or not np.any(chi > 0) or not np.any(nuf > 0):
        raise ValueError('Nonnegative fissile spectrum/production support required')
    if heat_production is not None:
        heat_production = _finite_vector(heat_production, n, 'actual heat production')
        if np.any(heat_production < 0):
            raise ValueError('Actual heat production must be nonnegative')
    try:
        partition = tuple(tuple(group) for group in groups)
    except TypeError as error:
        raise ValueError('Explicit index groups required') from error
    indices = [i for group in partition for i in group]
    if not partition or any(not group for group in partition) or any(
            isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in indices):
        raise ValueError('Nonempty integer index groups required')
    if sorted(indices) != list(range(n)):
        raise ValueError('Exhaustive disjoint partition required')
    C = np.zeros((len(partition), n))
    for g, group in enumerate(partition):
        C[g, list(group)] = 1.
    fine_loss = {}
    fine_condition = {}
    for order in range(4):
        if order not in production:
            raise ValueError('Production orders 0..3 required')
        sp = np.asarray(production[order], dtype=float)
        if sp.shape != (n, n) or not np.all(np.isfinite(sp)):
            raise ValueError('Invalid finite production matrix')
        fine_loss[order] = np.diag(total)-sp.T
        fine_condition[order] = _condition(fine_loss[order], condition_limit, f'fine R{order}')
    x = np.linalg.solve(fine_loss[0], chi)
    solve_residual = {0: _backward(fine_loss[0], x, chi)}
    kinf = float(nuf@x)
    if not np.isfinite(kinf) or kinf <= 0:
        raise ValueError('Finite positive k_infinity required')
    psi = {0: x/kinf}
    if not np.all(np.isfinite(psi[0])) or np.any(psi[0] <= 0):
        raise ValueError('Finite strictly positive scalar weights required')
    for order in range(1, 4):
        factor = order/(2*order+1)
        value = np.linalg.solve(fine_loss[order], psi[order-1])
        solve_residual[order] = _backward(fine_loss[order], value, psi[order-1])
        psi[order] = factor*value
    weights, injections, losses, effective, correction = {}, {}, {}, {}, {}
    ratios, coarse_condition, hierarchy_residual = {}, {}, {}
    for order in range(4):
        if not np.all(np.isfinite(psi[order])):
            raise ValueError('Nonfinite hierarchy weights')
        weights[order] = C@psi[order]
        absolute = C@np.abs(psi[order])
        if not np.all(np.isfinite(weights[order])) or not np.all(np.isfinite(absolute)) or np.any(absolute <= 0):
            raise ValueError('Exact/near cancellation or unsupported group')
        rho = np.abs(weights[order])/absolute
        if np.any(rho <= cancellation_threshold):
            raise ValueError('Exact/near cancellation in hierarchy group sums')
        ratios[order] = rho
        D = np.zeros((n, len(partition)))
        for g, group in enumerate(partition):
            D[list(group), g] = psi[order][list(group)]/weights[order][g]
        if not np.all(np.isfinite(D)):
            raise ValueError('Nonfinite injection')
        injections[order] = D
        losses[order] = C@fine_loss[order]@D
        coarse_condition[order] = _condition(losses[order], condition_limit, f'coarse R{order}')
        rhs = C@chi/kinf if order == 0 else order/(2*order+1)*weights[order-1]
        hierarchy_residual[order] = _backward(losses[order], weights[order], rhs)
    Tc = np.diag(C@np.diag(total)@injections[0])
    for order in range(4):
        projected_total = np.diag(C@np.diag(total)@injections[order])
        correction[order] = Tc-projected_total
        effective[order] = (np.diag(Tc)-losses[order]).T
    if any(not np.isfinite(r) or r > backward_tolerance
           for r in (*solve_residual.values(), *hierarchy_residual.values())):
        raise ValueError('Backward residual gate failed')
    chic = C@chi
    nufc = injections[0].T@nuf
    heatc = None if heat_production is None else injections[0].T@heat_production
    derived = [Tc, chic, nufc, np.outer(chic, nufc), *effective.values()]
    if heatc is not None:
        derived.append(heatc)
    if any(not np.all(np.isfinite(a)) for a in derived):
        raise ValueError('Nonfinite derived coefficients')
    return CondensedMomentHierarchy(
        Tc, chic, nufc, np.outer(chic, nufc), heatc, C, partition, psi, weights,
        injections, losses, effective, correction, kinf,
        {'fine_condition_2': fine_condition, 'coarse_condition_2': coarse_condition,
         'cancellation_ratio': ratios, 'fine_solve_backward_residual': solve_residual,
         'coarse_hierarchy_backward_residual': hierarchy_residual,
         'condition_limit': condition_limit, 'cancellation_threshold': cancellation_threshold,
         'backward_tolerance': backward_tolerance},
        {'method': 'homogeneous_B_to_zero_leading_moment_hierarchy',
         'production_orientation': 'from-major [from,to]', 'loss_orientation': '[to,from]',
         'chi_renormalized': False, 'native_current_weight_correction': False,
         'finite_buckling_preservation': False, 'higher_weights_may_be_signed': True,
         'derived_coefficients_may_be_signed': True, 'production_admission': False,
         'physical_admission': False, 'scientific_admission': False,
         'source_qualified': False, 'spatial_convergence_admitted': False,
         'homogeneous_scope_only': True, 'angular_qualification': False,
         'stability_qualification': False, 'uncertainty_qualification': False,
         'heat_uses_actual_supplied_vector': heat_production is not None})


def condense_moment_material(material, groups, *, heat_production=None, **kwargs):
    """Explicit opt-in bridge for a from-major MomentMaterial; no raw mutation."""
    if not isinstance(material, MomentMaterial) or material.provenance.get('orientation_contract') != 'from-major':
        raise ValueError('Explicit from-major MomentMaterial required')
    for order in range(4):
        if order not in material.production or order not in material.loss or not np.array_equal(
                material.loss[order], np.diag(material.total)-material.production[order].T):
            raise ValueError('Native production/loss orientation identity differs')
    return condense_moment_hierarchy(material.total, material.production, material.chi,
                                    material.nu_sigma_f, groups, heat_production=heat_production, **kwargs)
