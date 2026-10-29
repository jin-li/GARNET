"""Fission power density: the source term the thermal solver consumes.

A k-eigenvalue flux carries an arbitrary normalization (the power iteration
renormalizes it every outer so the mean fission source is 1), so a *shape* is
all a criticality solve can give you. Coupling to heat transfer needs the
absolute thing -- watts per cubic centimetre -- which comes from the shape plus
one externally imposed number, the rated thermal power:

    q'''(r)  =  P_rated * kappa_Sigma_f . phi(r) / sum_cells kappa_Sigma_f . phi V

Normalizing this way means the absolute units of the library's ``kappaFission``
never enter: whatever they are (J or MeV per fission times Sigma_f), they cancel
between numerator and denominator, and only the *shape* of kappa_Sigma_f is
used. That is the honest reading of the data, because multigroup libraries do
not agree on those units and the reactor's power is a design specification
rather than something the eigenvalue solve computes.
"""

from __future__ import annotations

import numpy as np

from .backend import asnumpy


def fission_energy_xs(materials):
    """(M, G) energy-release cross section table, and the attribute it came from.

    POWER is proportional to kappa*Sigma_f, not nu*Sigma_f. Both vanish outside
    the fuel, so either puts power only where it belongs -- but nu varies by
    group (on the HP-MR's 11-group set kappaSigma_f/nuSigma_f spans 1.12x), so
    nu-weighting tilts the distribution by the local spectrum. Fall back to
    nu_sigma_f only when no material carries kappaFission, and report which was
    used so callers can say so.
    """
    mats = list(materials)
    key = ("kappa_fission" if any(getattr(m, "kappa_fission", None) is not None
                                  for m in mats) else "nu_sigma_f")

    def row(m):
        v = getattr(m, key, None)
        return np.atleast_1d(m.nu_sigma_f if v is None else v)

    return np.array([row(m) for m in mats], dtype=float), key


def power_density(flux, materials, material_map=None, *, total_power=None,
                  cell_volume=1.0, volume_weight=None, mix_material=None,
                  mix_weight=None, active=None):
    """Per-cell fission power density on the host, shape ``grid.shape``.

    flux         : (G, *grid.shape), on device or host.
    material_map : integer map into ``materials``; None for a single material.
    total_power  : rated thermal power in W. Given, the result is W/cm^3 and
                   integrates to exactly this over the cell volumes. Omitted,
                   the raw kappa_Sigma_f . phi edit is returned, which carries
                   the flux's own arbitrary normalization.
    cell_volume  : scalar or per-cell cm^3 (including annuli on r-z grids).
                   Only the *product* with
                   volume_weight matters, and only up to a constant when
                   total_power is given.
    volume_weight: optional per-cell relative volume factor, shape grid.shape --
                   the radial metric ``grid.cylindrical_metrics()[0]`` on r-z
                   grids, where cells are annuli and cell_volume alone is wrong.
    mix_material / mix_weight : the solver's two-material volume blend. Cross
                   sections blend linearly (exact flat-flux reaction-rate
                   averaging), matching ``Fields``.
    active       : bool mask; excised cells produce no power.
    """
    flux = asnumpy(flux)
    groups = flux.shape[0]
    table, _ = fission_energy_xs(materials)
    if table.shape[1] != groups:
        raise ValueError(f"flux has {groups} groups, materials have {table.shape[1]}")

    if material_map is None:
        if len(table) > 1:
            raise ValueError("material_map is required with multiple materials")
        energy_release_xs = np.broadcast_to(table[0].reshape(groups, *([1] * (flux.ndim - 1))),
                             flux.shape)
    else:
        material_indices = np.asarray(material_map)
        if material_indices.shape != flux.shape[1:]:
            raise ValueError(f"material_map shape {material_indices.shape} != flux cell "
                             f"shape {flux.shape[1:]}")
        energy_release_xs = np.stack([table[:, g][material_indices] for g in range(groups)])
        if mix_material is not None:
            mixed_material = np.asarray(mix_material)
            mixed_weight = np.asarray(mix_weight, dtype=float)
            has_mixture = mixed_material >= 0
            safe_mixed_material = np.maximum(mixed_material, 0)
            for g in range(groups):
                mixed_xs = table[:, g][safe_mixed_material]
                energy_release_xs[g] = np.where(
                    has_mixture,
                    (1.0 - mixed_weight) * energy_release_xs[g] + mixed_weight * mixed_xs,
                    energy_release_xs[g],
                )

    power_density_field = (energy_release_xs * flux).sum(axis=0)
    if active is not None:
        power_density_field = np.where(np.asarray(active).astype(bool), power_density_field, 0.0)
    if total_power is None:
        return power_density_field

    cell_volumes = np.asarray(asnumpy(cell_volume), dtype=float)
    if volume_weight is not None:
        cell_volumes = cell_volumes * asnumpy(volume_weight)
    try:
        cell_volumes = np.broadcast_to(cell_volumes, power_density_field.shape)
    except ValueError as exc:
        raise ValueError("cell volumes must broadcast to the flux cell shape") from exc
    if not np.all(np.isfinite(cell_volumes)) or np.any(cell_volumes <= 0.0):
        raise ValueError("cell volumes must be finite and positive")
    integral = float(np.sum(power_density_field * cell_volumes))
    if integral <= 0.0:
        raise ValueError("no fission power in the flux: cannot normalize to "
                         "a rated power (is the flux zero, or the material "
                         "map missing its fuel?)")
    return power_density_field * (float(total_power) / integral)
