"""Scalar coefficients for the experimental diffusion frequency transform.

This module intentionally has no array/backend dependencies. A shared frozen
frequency applies to all neutron groups during one step. The second-order
method integrates delayed-neutron production at both endpoints. Spatial
operators still act on physical fluxes; heat transfer is not transformed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


def phi1(z: float) -> float:
    """Return (exp(z)-1)/z with its continuous value at zero."""
    if abs(z) < 1e-5:
        return 1.0 + z * (0.5 + z * (1.0/6 + z * (1.0/24 + z/120)))
    return math.expm1(z) / z


def phi2(z: float) -> float:
    """Return (exp(z)-1-z)/z**2, including its value 1/2 at zero."""
    if abs(z) < 0.5:
        term = total = 0.5
        for denominator in range(3, 18):
            term *= z / denominator
            total += term
        return total
    return (phi1(z) - 1.0) / z


def population_frequency(current: float, previous: float, width: float) -> float:
    """Logarithmic growth over an accepted interval, without ratio overflow."""
    if not all(math.isfinite(v) and v > 0.0
               for v in (current, previous, width)):
        raise ValueError("frequency history needs finite positive populations and width")
    change = (current - previous) / previous
    logarithm = (math.log1p(change) if abs(change) < 0.5
                 else math.log(current) - math.log(previous))
    return logarithm / width


@dataclass(frozen=True)
class ExponentialStep:
    """Frozen-frequency temporal coefficients in physical endpoint variables."""

    width: float
    frequency: float

    def __post_init__(self):
        if not math.isfinite(self.width) or self.width <= 0.0:
            raise ValueError("exponential step width must be finite and positive")
        if not math.isfinite(self.frequency):
            raise ValueError("exponential frequency must be finite")
        if abs(self.frequency * self.width) >= 1.0:
            raise ValueError("prototype requires abs(frequency * width) < 1")

    @property
    def diagonal_factor(self):
        return 1.0 + self.frequency * self.width

    @property
    def history_factor(self):
        return math.exp(self.frequency * self.width)

    @property
    def source_weight(self):
        """Integral weight for an exponentially varying endpoint source."""
        return self.width * phi1(-self.frequency * self.width)

    def precursor(self, decay):
        """Return old-state retention, source weight and removed-beta factor.

        C_new = retention*C_old + beta*source_weight*S_new.
        The fission emission weight is chi_total - sum chi_d*beta*removed.
        The last factor is evaluated without subtracting nearly equal numbers
        for large decay*width; at zero frequency it is exactly retention.
        """
        if not math.isfinite(decay) or decay < 0.0:
            raise ValueError("precursor decay must be finite and non-negative")
        lh = decay * self.width
        wh = self.frequency * self.width
        z = lh + wh
        source_weight = self.width * phi1(-z)
        removed = (1.0 - decay * source_weight if abs(z) < 1e-5 else
                   (wh + lh * math.exp(-z)) / z)
        return math.exp(-lh), source_weight, removed


@dataclass(frozen=True)
class ExponentialThetaStep(ExponentialStep):
    """Transformed trapezoidal neutron step with linear production history.

    For M*dphi/dt=R and frozen omega, the physical endpoint equation is

      M*(2/h+omega)*phi_new - R_new
          = exp(omega*h)*(M*(2/h-omega)*phi_old + R_old).

    These coefficients apply to neutrons and delayed-neutron production.
    No heat-transfer equation is transformed.
    """

    @property
    def diagonal_factor(self):
        return 2.0 + self.frequency * self.width

    @property
    def old_state_factor(self):
        return 2.0 - self.frequency * self.width

    def linear_precursor(self, decay):
        """Return retention, old/new production weights and removed beta.

        C_new = retention*C_old + old_weight*(beta*S)_old
                                  + new_weight*(beta*S)_new.
        Interpolation is linear after removing the frozen exponential.
        The removed-beta coefficient is 1-lambda*new_weight.
        """
        if not math.isfinite(decay) or decay < 0.0:
            raise ValueError("precursor decay must be finite and non-negative")
        lh, wh = decay * self.width, self.frequency * self.width
        z = -(lh + wh)
        new_weight = self.width * phi2(z)
        # The old moment decays as 1/z**2. Do not subtract the nearly equal
        # phi_1 and phi_2 when a precursor family is very stiff.
        old_moment = (phi1(z) - phi2(z) if abs(z) < 0.5 else
                      (((z - 1.0) * math.exp(z) + 1.0) / z) / z)
        old_weight = math.exp(wh) * self.width * old_moment
        removed = (1.0 - decay * new_weight if abs(z) < 0.5 else
                   (wh + lh * phi1(z)) / (-z))
        return math.exp(-lh), old_weight, new_weight, removed
