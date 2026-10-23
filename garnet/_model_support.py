"""Shared validation for the public model builders."""

from collections.abc import Mapping

import numpy as np

from .materials import Kinetics, Material


def checked_materials(materials, *, count=None, groups=None, context="materials"):
    values = list(materials)
    if not values:
        raise ValueError(f"{context} must contain at least one Material")
    if any(not isinstance(m, Material) for m in values):
        raise TypeError(f"{context} must contain Material objects")
    if count is not None and len(values) != count:
        raise ValueError(f"{context} must preserve material count/order ({count} entries)")
    groups = values[0].n_groups if groups is None else groups
    if any(m.n_groups != groups for m in values):
        raise ValueError(f"{context}: all materials must have the same number of groups ({groups})")
    return values


def replace_materials(base, replacements, *, context="replacement materials"):
    """Resolve a complete sequence or partial name/index mapping in stable order."""
    base = list(base)
    if not base:
        raise ValueError("empty model: call fill() with a material first")
    if isinstance(replacements, Mapping):
        values = list(base)
        seen = set()
        for key, material in replacements.items():
            if isinstance(key, str):
                matches = [i for i, m in enumerate(base) if m.name == key]
                if not matches:
                    raise ValueError(f"{context}: unknown material name {key!r}")
                if len(matches) != 1:
                    raise ValueError(f"{context}: ambiguous material name {key!r}; use an index")
                index = matches[0]
            elif isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_)):
                index = int(key)
                if not 0 <= index < len(base):
                    raise ValueError(f"{context}: material index {index} is out of range")
            else:
                raise TypeError(f"{context}: keys must be material names or integer indices")
            if index in seen:
                raise ValueError(f"{context}: material index {index} was specified twice")
            seen.add(index)
            values[index] = material
    else:
        values = replacements
    return checked_materials(values, count=len(base), groups=base[0].n_groups,
                             context=context)


def solver_options(defaults, options):
    """Keep construction options separate from the model's geometry and device."""
    options = dict(options or {})
    owned = {"grid", "mesh", "materials", "material_map", "cell_material", "bc",
             "active", "mask_bc", "mix_material", "mix_weight", "device",
             "alpha_boundary", "eig_solver", "group_operator", "problem_at",
             "kinetics"}
    conflicts = owned.intersection(options)
    if conflicts:
        raise ValueError("solver_kwargs cannot override model settings: "
                         + ", ".join(sorted(conflicts)))
    return dict(defaults, **options)


def checked_kinetics(kinetics, groups, *, material_count=None):
    if not isinstance(kinetics, Kinetics):
        raise TypeError("kinetics must be a Kinetics object")
    if kinetics.velocities.shape[-1] != groups:
        raise ValueError(f"kinetics needs {groups} velocities per row (one per group)")
    if material_count is not None:
        for name in ("velocities", "beta"):
            value = getattr(kinetics, name)
            if value.ndim == 2 and value.shape[0] != material_count:
                raise ValueError(f"kinetics.{name} needs {material_count} material rows")
    if kinetics.chi_delayed is not None and kinetics.chi_delayed.shape[-1] != groups:
        raise ValueError(f"kinetics.chi_delayed needs {groups} groups per spectrum")
    return kinetics


def box_range(value, axis):
    bounds = np.asarray(value, dtype=float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
        raise ValueError(f"{axis} range must be a finite (lower, upper) pair with lower <= upper")
    return bounds
