"""Offline integration of qualified delayed-neutron ACE LAW4 spectra.

Only continuous histogram/linear-linear outgoing PDFs and histogram/linear
incident interpolation are supported. Linear incident interpolation uses the
unit-base transformed mixture, not a mixture on the original energy axes.
Energies are MeV; densities are MeV^-1. No incident extrapolation, production
weighting, aggregate INF_CHID substitution, or implicit renormalization occurs.

Primary format and implementation references:
https://github.com/NuclearData/ACEFormat/blob/master/src/ContinuousEnergyNeutron.tex
https://github.com/openmc-dev/openmc/blob/develop/src/distribution_energy.cpp
Integration uses the piecewise PDF, with the stored ACE CDF independently checked
within the declared numeric tolerance. Tiny printing differences are exposed.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import math

import numpy as np

from .ace_precursors import (
    ACEPrecursorError, _integer, _number, _probability, read_ace_precursors,
)


@dataclass(frozen=True)
class ACEOutgoingPDF:
    energy_mev: tuple[float, ...]
    pdf_per_mev: tuple[float, ...]
    stored_cdf: tuple[float, ...]
    interpolation: int
    validation_atol: float = 1e-6

    def __post_init__(self):
        if not math.isfinite(self.validation_atol) or self.validation_atol < 0:
            raise ACEPrecursorError('Invalid spectrum validation tolerance')
        e, p, c = map(np.asarray, (self.energy_mev, self.pdf_per_mev, self.stored_cdf))
        if (len(e) < 2 or e.shape != p.shape or e.shape != c.shape or e.ndim != 1 or
                not all(np.all(np.isfinite(v)) for v in (e, p, c)) or
                e[0] < 0 or np.any(np.diff(e) <= 0) or np.any(p < 0) or
                np.any(c < 0) or np.any(c > 1+self.validation_atol) or np.any(np.diff(c) < 0)):
            raise ACEPrecursorError('Invalid LAW4 outgoing energy/PDF/CDF data')
        if self.interpolation not in (1, 2):
            raise ACEPrecursorError('Unsupported LAW4 outgoing interpolation/discrete lines')
        integral = self.integrated_cdf(e)
        if (abs(c[0]) > self.validation_atol or abs(c[-1]-1) > self.validation_atol or
                np.max(np.abs(integral-c)) > self.validation_atol or
                abs(integral[-1]-1) > self.validation_atol):
            raise ACEPrecursorError('LAW4 PDF/CDF consistency or normalization failed')

    def integrated_cdf(self, energy_mev):
        """Integral of the PDF from its lower support; zero/total outside support."""
        e, p = np.asarray(self.energy_mev), np.asarray(self.pdf_per_mev)
        query = np.asarray(energy_mev, dtype=float)
        if not np.all(np.isfinite(query)):
            raise ACEPrecursorError('Nonfinite outgoing energy')
        width = np.diff(e)
        area = width * (p[:-1] if self.interpolation == 1 else (p[:-1]+p[1:])/2)
        cumulative = np.concatenate(([0.], np.cumsum(area)))
        index = np.clip(np.searchsorted(e, query, side='right')-1, 0, len(e)-2)
        delta = np.clip(query, e[0], e[-1])-e[index]
        values = cumulative[index] + p[index]*delta
        if self.interpolation == 2:
            values += .5*(p[index+1]-p[index])/width[index]*delta**2
        return np.where(query <= e[0], 0., np.where(query >= e[-1], cumulative[-1], values))

    @property
    def integrated_probability(self):
        return float(self.integrated_cdf(self.energy_mev[-1]))

    @property
    def maximum_cdf_discrepancy(self):
        return float(np.max(np.abs(self.integrated_cdf(self.energy_mev)-self.stored_cdf)))


@dataclass(frozen=True)
class ACEGroupedSpectrum:
    group_bounds_mev: tuple[float, ...]
    probabilities: tuple[float, ...]  # same group order as supplied boundaries
    below_group_range: float
    above_group_range: float
    total_probability: float
    validation_atol: float

    def normalized_probabilities(self):
        """Explicit print-precision correction, only after complete-coverage checks."""
        if (self.below_group_range > self.validation_atol or
                self.above_group_range > self.validation_atol or
                abs(self.total_probability-1) > self.validation_atol):
            raise ACEPrecursorError('Cannot normalize an incomplete/unqualified spectrum')
        # Never discard an uncovered tail, even if smaller than validation_atol.
        if self.below_group_range > 0 or self.above_group_range > 0:
            raise ACEPrecursorError('Cannot renormalize uncovered outgoing probability')
        values = np.asarray(self.probabilities)
        return tuple(float(v) for v in values / values.sum())


@dataclass(frozen=True)
class ACEDelayedFamilySpectrum:
    incident_energy_mev: tuple[float, ...]
    breakpoints: tuple[int, ...]
    interpolation: tuple[int, ...]
    distributions: tuple[ACEOutgoingPDF, ...]
    validation_atol: float = 1e-6

    @property
    def incident_independent(self):
        """Exact equality of every stored outgoing table, not just group integrals."""
        return all(d == self.distributions[0] for d in self.distributions[1:])

    def _mixture(self, incident_energy_mev):
        energy = float(incident_energy_mev)
        grid = np.asarray(self.incident_energy_mev)
        if not math.isfinite(energy) or energy < grid[0] or energy > grid[-1]:
            raise ACEPrecursorError('Incident energy outside declared LAW4 range')
        matches = np.flatnonzero(grid == energy)
        if len(matches):
            d = self.distributions[int(matches[0])]
            return ((1., d),), (d.energy_mev[0], d.energy_mev[-1])
        i = int(np.searchsorted(grid, energy)-1)
        region = int(np.searchsorted(self.breakpoints, i+2, side='left')) if self.breakpoints else 0
        scheme = self.interpolation[region] if self.interpolation else 2
        left, right = self.distributions[i:i+2]
        if scheme == 1:
            return ((1., left),), (left.energy_mev[0], left.energy_mev[-1])
        if scheme != 2:
            raise ACEPrecursorError('Unsupported LAW4 incident interpolation')
        fraction = (energy-grid[i])/(grid[i+1]-grid[i])
        lo = (1-fraction)*left.energy_mev[0]+fraction*right.energy_mev[0]
        hi = (1-fraction)*left.energy_mev[-1]+fraction*right.energy_mev[-1]
        return ((1-fraction, left), (fraction, right)), (lo, hi)

    def integrated_cdf(self, incident_energy_mev, outgoing_energy_mev):
        mixture, (lo, hi) = self._mixture(incident_energy_mev)
        energy = np.asarray(outgoing_energy_mev, dtype=float)
        if not np.all(np.isfinite(energy)):
            raise ACEPrecursorError('Nonfinite outgoing energy')
        coordinate = (np.clip(energy, lo, hi)-lo)/(hi-lo)
        result = np.zeros_like(energy)
        for weight, distribution in mixture:
            low, high = distribution.energy_mev[0], distribution.energy_mev[-1]
            result += weight*distribution.integrated_cdf(low+coordinate*(high-low))
        total = sum(weight*d.integrated_probability for weight, d in mixture)
        return np.where(energy <= lo, 0., np.where(energy >= hi, total, result))

    def integrate_groups(self, incident_energy_mev, group_bounds_mev, *, require_complete=True):
        """Integrate ascending or descending groups; report all uncovered probability.

        Returned probabilities are raw PDF integrals. Call normalized_probabilities
        explicitly for an accepted print-precision normalization correction.
        """
        bounds = np.asarray(group_bounds_mev, dtype=float)
        if (bounds.ndim != 1 or len(bounds) < 2 or not np.all(np.isfinite(bounds)) or
                np.any(bounds < 0) or not (np.all(np.diff(bounds)>0) or np.all(np.diff(bounds)<0))):
            raise ACEPrecursorError('Group bounds must be finite, nonnegative, and strictly monotone')
        values = self.integrated_cdf(incident_energy_mev, bounds)
        probabilities = np.diff(values) * (1 if bounds[-1] > bounds[0] else -1)
        _, support = self._mixture(incident_energy_mev)
        total = float(self.integrated_cdf(incident_energy_mev, support[-1]))
        below = float(self.integrated_cdf(incident_energy_mev, min(bounds)))
        above = max(0., total-float(self.integrated_cdf(incident_energy_mev, max(bounds))))
        # Exact zero is possible and required: a nonzero tail is not discarded.
        if require_complete and (below > 0 or above > 0):
            raise ACEPrecursorError(f'Uncovered outgoing probability: below={below}, above={above}')
        if np.any(probabilities < 0) or abs(total-1) > self.validation_atol:
            raise ACEPrecursorError('Integrated LAW4 spectrum is not normalized')
        return ACEGroupedSpectrum(tuple(bounds), tuple(probabilities), below, above,
                                  total, self.validation_atol)


@dataclass(frozen=True)
class ACEDelayedSpectra:
    source_path: str
    source_sha256: str
    table_name: str
    temperature_mev: float
    families: tuple[ACEDelayedFamilySpectrum, ...]


def read_ace_delayed_spectra(path, table_name, *, validation_atol=1e-6):
    """Read single-law delayed LAW4 data from an exact legacy ASCII ACE table.

    Unsupported law chains, discrete lines, interpolation and extrapolation fail
    explicitly. Source metadata reuse the strict precursor reader. Raw library
    contents are not retained in the returned object or written to disk.
    """
    metadata = read_ace_precursors(path, table_name)
    content = Path(path).read_bytes()
    if sha256(content).hexdigest() != metadata.source_sha256:
        raise ACEPrecursorError('ACE file changed while reading delayed spectra')
    lines = content.decode('ascii').splitlines()
    # Exact table selection/ambiguity and complete headers were checked above.
    header = next(i for i, line in enumerate(lines)
                  if line.split() and line.split()[0] == table_name)
    nlines = (metadata.nxs[0]+3)//4
    xss = tuple(_number(v) for line in lines[header+12:header+12+nlines] for v in line.split())
    base = metadata.jxs[26]-1
    stop = min([p-1 for p in metadata.jxs if p > metadata.jxs[26]]+[len(xss)])

    def take(index, count):
        if index < base or index+count > stop:
            raise ACEPrecursorError('Truncated/out-of-range LAW4 data')
        return xss[index:index+count]

    result = []
    for family in metadata.families:
        if len(family.outgoing_laws) != 1 or family.outgoing_laws[0].law != 4:
            raise ACEPrecursorError('Only a single outgoing LAW4 is qualified')
        law = family.outgoing_laws[0]
        applicability, _ = _probability(xss, law.xss_index+2, stop)
        if (not applicability.linear_linear or
                any(abs(p-1) > validation_atol for p in applicability.probability)):
            raise ACEPrecursorError('LAW4 applicability must be a qualified unity probability')
        start = base+law.data_locator-1
        nr = _integer(take(start, 1)[0], 'LAW4 NR')
        bp = tuple(_integer(v, 'LAW4 breakpoint', 2) for v in take(start+1, nr))
        interpolation = tuple(_integer(v, 'LAW4 interpolation', 1)
                              for v in take(start+1+nr, nr))
        ne = _integer(take(start+1+2*nr, 1)[0], 'LAW4 NE', 2)
        energies = tuple(take(start+2+2*nr, ne))
        locators = tuple(_integer(v, 'LAW4 outgoing locator', 1)
                         for v in take(start+2+2*nr+ne, ne))
        if (energies[0] < 0 or any(a >= b for a, b in zip(energies, energies[1:])) or
                nr and (bp[-1] != ne or any(a >= b for a, b in zip(bp, bp[1:])))):
            raise ACEPrecursorError('Invalid LAW4 incident grid/interpolation breakpoints')
        if any(i not in (1, 2) for i in interpolation):
            raise ACEPrecursorError('Unsupported LAW4 incident interpolation')
        if (applicability.energy_mev[0] > energies[0] or
                applicability.energy_mev[-1] < energies[-1]):
            raise ACEPrecursorError('LAW4 applicability does not cover the incident grid')
        distributions = []
        for locator in locators:
            index = base+locator-1
            flag = _integer(take(index, 1)[0], 'LAW4 INTT', 1)
            if flag not in (1, 2):
                raise ACEPrecursorError('Unsupported LAW4 outgoing interpolation/discrete lines')
            count = _integer(take(index+1, 1)[0], 'LAW4 NP', 2)
            energy = tuple(take(index+2, count))
            pdf = tuple(take(index+2+count, count))
            cdf = tuple(take(index+2+2*count, count))
            distributions.append(ACEOutgoingPDF(energy, pdf, cdf, flag, validation_atol))
        result.append(ACEDelayedFamilySpectrum(energies, bp, interpolation,
                                              tuple(distributions), validation_atol))
    return ACEDelayedSpectra(metadata.source_path, metadata.source_sha256,
                             metadata.table_name, metadata.temperature_mev, tuple(result))
