"""Offline precursor metadata from legacy ASCII continuous-energy neutron ACE.

This reader does not evaluate outgoing spectra, infer production weights, or
convert family probabilities into regional beta. Incident energies and kT remain
in MeV. ACE BDD decay constants are inverse shakes (1 shake = 1e-8 s); the
public decay_s_inv values are inverse seconds.

Format: https://github.com/NuclearData/ACEFormat/blob/master/src/ContinuousEnergyNeutron.tex
Unit cross-check: OpenMC openmc/data/reaction.py, _get_fission_products_ace.
Only linear-linear interpolation is qualified here (NR=0, or explicit INT=2).
Other interpolation metadata are retained but evaluation is refused.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import math
import re

import numpy as np


class ACEPrecursorError(ValueError):
    """Malformed, ambiguous, or unsupported ACE precursor input."""


def _integer(value, label, minimum=0):
    if not math.isfinite(value) or value != int(value) or value < minimum:
        raise ACEPrecursorError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _number(token):
    # NJOY may omit E on very small exponents. Never execute input text.
    token = token.replace('D', 'E').replace('d', 'e')
    token = re.sub(r'(?<=[\d.])([+-]\d+)$', r'e\1', token) if 'e' not in token.lower() else token
    try:
        value = float(token)
    except ValueError as exc:
        raise ACEPrecursorError("Invalid numeric ACE token") from exc
    if not math.isfinite(value):
        raise ACEPrecursorError("Nonfinite numeric ACE value")
    return value


@dataclass(frozen=True)
class ACEProbabilityTable:
    energy_mev: tuple[float, ...]
    probability: tuple[float, ...]
    breakpoints: tuple[int, ...]
    interpolation: tuple[int, ...]

    @property
    def linear_linear(self):
        return not self.interpolation or all(i == 2 for i in self.interpolation)

    def evaluate(self, energy_mev):
        """Evaluate only qualified linear tables, without extrapolation."""
        if not self.linear_linear:
            raise ACEPrecursorError("Unqualified BDD interpolation scheme")
        energies = np.asarray(energy_mev, dtype=float)
        if (not np.all(np.isfinite(energies)) or
                np.any(energies < self.energy_mev[0]) or
                np.any(energies > self.energy_mev[-1])):
            raise ACEPrecursorError("Requested energy outside declared BDD range")
        return np.interp(energies, self.energy_mev, self.probability)


@dataclass(frozen=True)
class ACEOutgoingLaw:
    """DNED header only; all locator indices are one-based ACE indices."""
    relative_locator: int
    xss_index: int
    next_law_locator: int
    law: int
    data_locator: int


@dataclass(frozen=True)
class ACEPrecursorFamily:
    decay_inverse_shakes: float
    decay_s_inv: float
    branching: ACEProbabilityTable
    outgoing_locator: int | None
    outgoing_laws: tuple[ACEOutgoingLaw, ...]


@dataclass(frozen=True)
class ACEBranchingGrid:
    energy_mev: tuple[float, ...]
    raw_probabilities: tuple[tuple[float, ...], ...]  # family, energy
    probability_sums: tuple[float, ...]
    normalized_probabilities: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class ACEPrecursorMetadata:
    source_path: str
    source_sha256: str
    table_name: str
    atomic_weight_ratio: float
    temperature_mev: float  # kT, not temperature in Kelvin
    date: str
    nxs: tuple[int, ...]
    jxs: tuple[int, ...]
    families: tuple[ACEPrecursorFamily, ...]

    @property
    def group_count(self):
        return self.nxs[7]

    def _range(self, incident_range_mev):
        if not self.families:
            raise ACEPrecursorError("Table has no delayed precursor families")
        ranges = [(f.branching.energy_mev[0], f.branching.energy_mev[-1])
                  for f in self.families]
        if incident_range_mev is None:
            if any(r != ranges[0] for r in ranges[1:]):
                raise ACEPrecursorError("BDD ranges differ; declare a common incident range")
            result = ranges[0]
        else:
            if len(incident_range_mev) != 2:
                raise ACEPrecursorError("Incident range must have two bounds")
            result = tuple(float(e) for e in incident_range_mev)
        if (not all(math.isfinite(e) for e in result) or result[0] >= result[1] or
                any(result[0] < lo or result[1] > hi for lo, hi in ranges)):
            raise ACEPrecursorError("Invalid or uncovered common incident range")
        return result

    def branching_on_union_grid(self, incident_range_mev=None):
        """Return raw and explicitly normalized probabilities on a common grid.

        No production weighting is implied. Normalization factors are exposed;
        callers must independently qualify whether a correction is acceptable.
        """
        lo, hi = self._range(incident_range_mev)
        grid = np.unique([lo, hi] + [e for f in self.families
                         for e in f.branching.energy_mev if lo <= e <= hi])
        raw = np.array([f.branching.evaluate(grid) for f in self.families])
        sums = raw.sum(axis=0)
        if np.any(sums <= 0):
            raise ACEPrecursorError("BDD probability sum is not positive")
        return ACEBranchingGrid(tuple(grid), tuple(map(tuple, raw)), tuple(sums),
                                tuple(map(tuple, raw / sums)))

    def qualified_constant_branching(self, incident_range_mev=None, *, atol=1e-6,
                                     constancy_atol=0.):
        """Return raw family constants only when constant and summing to unity.

        ``atol`` permits probability-sum printing error only. Constancy is
        exact by default; a nonzero ``constancy_atol`` is an explicit extra
        approximation tolerance, not an automatic consequence of printing.
        Every family must cover the whole declared range. No extrapolation,
        unqualified interpolation, or hidden probability renormalization occurs.
        """
        if not math.isfinite(atol) or atol < 0:
            raise ACEPrecursorError("atol must be finite and nonnegative")
        if not math.isfinite(constancy_atol) or constancy_atol < 0:
            raise ACEPrecursorError("constancy_atol must be finite and nonnegative")
        grid = self.branching_on_union_grid(incident_range_mev)
        raw = np.asarray(grid.raw_probabilities)
        if not np.all(np.abs(raw - raw[:, :1]) <= constancy_atol):
            raise ACEPrecursorError("BDD branching varies over the incident range")
        if not np.all(np.abs(np.asarray(grid.probability_sums) - 1) <= atol):
            raise ACEPrecursorError("BDD family probabilities do not sum to unity")
        return tuple(float(x) for x in raw[:, 0])


def _probability(xss, start, stop):
    def take(n):
        nonlocal start
        if n < 0 or start + n > stop:
            raise ACEPrecursorError("Truncated or overlapping BDD probability table")
        values = xss[start:start+n]
        start += n
        return values
    nr = _integer(take(1)[0], 'BDD NR')
    bp = tuple(_integer(v, 'BDD breakpoint', 1) for v in take(nr))
    interp = tuple(_integer(v, 'BDD interpolation', 1) for v in take(nr))
    ne = _integer(take(1)[0], 'BDD NE', 2)
    energy = tuple(take(ne))
    probability = tuple(take(ne))
    if (energy[0] < 0 or any(a >= b for a, b in zip(energy, energy[1:])) or
            any(p < 0 or p > 1 for p in probability)):
        raise ACEPrecursorError("Invalid BDD energies or probabilities")
    if nr and (bp[-1] != ne or bp[0] < 2 or
               any(a >= b for a, b in zip(bp, bp[1:]))):
        raise ACEPrecursorError("Invalid BDD interpolation breakpoints")
    return ACEProbabilityTable(energy, probability, bp, interp), start


def _metadata(path, digest, header, nxs, jxs, xss):
    name, awr, kt, date = header
    if not name.endswith('c'):
        raise ACEPrecursorError("Only continuous-energy neutron ACE tables are supported")
    count = nxs[7]
    for ptr in jxs:
        if ptr > len(xss):
            raise ACEPrecursorError("JXS pointer outside XSS")
    if count == 0:
        if any(jxs[i] for i in (24, 25, 26)):
            raise ACEPrecursorError("Precursor pointers present with zero family count")
        return ACEPrecursorMetadata(str(path), digest, name, awr, kt, date, nxs, jxs, ())
    if jxs[24] == 0:
        raise ACEPrecursorError("Missing BDD pointer for delayed families")
    start = jxs[24] - 1
    stop = min([p-1 for p in jxs if p > jxs[24]] + [len(xss)])
    if bool(jxs[25]) != bool(jxs[26]):
        raise ACEPrecursorError("Incomplete DNEDL/DNED pointer pair")
    locators = ()
    dned_stop = len(xss)
    if jxs[25]:
        locator_stop = min([p-1 for p in jxs if p > jxs[25]] + [len(xss)])
        dned_stop = min([p-1 for p in jxs if p > jxs[26]] + [len(xss)])
        if jxs[25]-1 + count > locator_stop:
            raise ACEPrecursorError("Truncated DNEDL locator array")
        locators = tuple(_integer(v, 'DNEDL locator', 1)
                         for v in xss[jxs[25]-1:jxs[25]-1+count])
    families = []
    for group in range(count):
        if start >= stop:
            raise ACEPrecursorError("Truncated BDD decay constant")
        decay = xss[start]
        start += 1
        if decay <= 0 or not math.isfinite(decay * 1e8):
            raise ACEPrecursorError("BDD decay must be finite and positive")
        probability, start = _probability(xss, start, stop)
        laws = []
        locator = locators[group] if locators else None
        current = locator
        seen = set()
        while current:
            if current in seen:
                raise ACEPrecursorError("Cyclic DNED law chain")
            seen.add(current)
            offset = jxs[26] + current - 2
            if offset < jxs[26]-1 or offset+3 > dned_stop:
                raise ACEPrecursorError("DNED law header outside XSS")
            nxt, law, data = (_integer(xss[offset], 'DNED next law'),
                              _integer(xss[offset+1], 'DNED LAW', 1),
                              _integer(xss[offset+2], 'DNED data locator', 1))
            if jxs[26] + data - 2 >= dned_stop:
                raise ACEPrecursorError("DNED data locator outside XSS")
            laws.append(ACEOutgoingLaw(current, offset+1, nxt, law, data))
            current = nxt
        families.append(ACEPrecursorFamily(decay, decay*1e8, probability,
                                           locator, tuple(laws)))
    return ACEPrecursorMetadata(str(path), digest, name, awr, kt, date, nxs, jxs,
                                tuple(families))


def read_ace_precursors(path, table_name):
    """Select exactly one named table from an offline legacy ASCII ACE file.

    Whole-file SHA-256 binds provenance even in a multi-table file. ACE 2.x
    headers and binary ACE are rejected explicitly. Library contents are never
    written or retained in the metadata object.
    """
    if not isinstance(table_name, str) or not table_name.strip():
        raise ACEPrecursorError("An exact table name is required")
    path = Path(path)
    content = path.read_bytes()
    digest = sha256(content).hexdigest()
    try:
        text = content.decode('ascii')
    except UnicodeDecodeError as exc:
        raise ACEPrecursorError("Binary/non-ASCII ACE is unsupported") from exc
    if '\x00' in text:
        raise ACEPrecursorError("Binary ACE is unsupported")
    lines = text.splitlines()
    cursor = 0
    matches = []
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        words = lines[cursor].split()
        if re.match(r'^\d+\.\d+\.\d+$', words[0]):
            raise ACEPrecursorError("ACE 2.x headers are unsupported")
        if len(words) < 3 or cursor + 12 > len(lines):
            raise ACEPrecursorError("Truncated legacy ACE header")
        awr, kt = _number(words[1]), _number(words[2])
        if awr <= 0 or kt < 0:
            raise ACEPrecursorError("Invalid ACE mass ratio or kT")
        nxs_words = ' '.join(lines[cursor+6:cursor+8]).split()
        jxs_words = ' '.join(lines[cursor+8:cursor+12]).split()
        if len(nxs_words) != 16 or len(jxs_words) != 32:
            raise ACEPrecursorError("Invalid NXS/JXS array length")
        nxs = tuple(_integer(_number(v), 'NXS') for v in nxs_words)
        jxs = tuple(_integer(_number(v), 'JXS') for v in jxs_words)
        length = nxs[0]
        nlines = (length+3)//4
        end = cursor + 12 + nlines
        if length == 0 or end > len(lines):
            raise ACEPrecursorError("Missing or truncated XSS array")
        if words[0] == table_name:
            tokens = ' '.join(lines[cursor+12:end]).split()
            if len(tokens) != length:
                raise ACEPrecursorError("XSS length does not match NXS(1)")
            xss = tuple(_number(v) for v in tokens)
            matches.append((words[0], awr, kt, words[3] if len(words)>3 else '', nxs, jxs, xss))
        cursor = end
    if len(matches) != 1:
        raise ACEPrecursorError(f"Expected exactly one table named {table_name!r}; found {len(matches)}")
    name, awr, kt, date, nxs, jxs, xss = matches[0]
    return _metadata(path, digest, (name, awr, kt, date), nxs, jxs, xss)
