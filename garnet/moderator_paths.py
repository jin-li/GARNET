"""Caller-defined bounded temperature/density tabulations (K and g/cm³)."""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass(frozen=True)
class TabulatedModeratorPath:
    """Shape-preserving interpolation; no equation of state is inferred.

    Supply positive, increasing temperature knots, strictly monotone positive
    densities, and a caller-owned identity. ``pressure_bar`` is optional
    provenance, not an input to density interpolation. Increasing as well as
    decreasing density paths are supported. Evaluation never extrapolates.
    """
    temperatures: tuple
    densities: tuple
    path_id: str
    pressure_bar: float | None = None

    def __post_init__(self):
        def knots(values, name):
            raw = np.asarray(values)
            if raw.dtype.kind not in 'iuf' or raw.ndim != 1:
                raise ValueError(f'{name} must be a real one-dimensional array')
            result = np.asarray(raw, dtype=float)
            if not np.all(np.isfinite(result)) or np.any(result <= 0):
                raise ValueError(f'{name} must be positive finite')
            return result

        temperatures = knots(self.temperatures, 'path temperatures')
        densities = knots(self.densities, 'path densities')
        differences = np.diff(densities)
        if (len(temperatures) < 2 or densities.shape != temperatures.shape
                or np.any(np.diff(temperatures) <= 0)
                or not (np.all(differences < 0) or np.all(differences > 0))):
            raise ValueError('path requires increasing T and strictly monotone density knots')
        if not isinstance(self.path_id, str) or not self.path_id.strip():
            raise ValueError('path requires nonempty identity')
        pressure = self.pressure_bar
        if pressure is not None:
            if (isinstance(pressure, (bool, np.bool_))
                    or not isinstance(pressure, (int, float, np.number))
                    or np.iscomplexobj(pressure) or not np.isfinite(pressure) or pressure <= 0):
                raise ValueError('pressure_bar must be positive finite or None')
            object.__setattr__(self, 'pressure_bar', float(pressure))
        object.__setattr__(self, 'temperatures', tuple(float(v) for v in temperatures))
        object.__setattr__(self, 'densities', tuple(float(v) for v in densities))

    def evaluate(self, temperature):
        if (isinstance(temperature, (bool, np.bool_))
                or not isinstance(temperature, (int, float, np.number))
                or np.iscomplexobj(temperature) or not np.isfinite(temperature)
                or not self.temperatures[0] <= temperature <= self.temperatures[-1]):
            raise ValueError('moderator temperature outside tabulated path')
        return float(PchipInterpolator(self.temperatures, self.densities,
                                       extrapolate=False)(temperature))
