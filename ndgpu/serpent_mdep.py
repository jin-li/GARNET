"""Literal, non-executing import of native Serpent micro-depletion output.

Only FLUX_<selector> and XS_<selector> are imported. Static decay/yield tables
are ignored. Selectors are exact native variable suffixes, not fuzzy universe
names; multiple mdep definitions may introduce numbered suffixes. Native group
indices have the same order as INF_NSF in Serpent 2.2.2. Bounds and their order
must be supplied from the matching input/result provenance: mdx has neither.

MT452 and MT455 are microscopic total and delayed neutron-production cross
sections. Multiplication by native smeared density gives central macroscopic
production estimates. Density and cross-section errors are retained separately;
no product error or covariance is inferred. This is not an effective-beta reader.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np

_NUM = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?\Z")
_ASSIGN = re.compile(r"\s*(FLUX|XS)_([A-Za-z0-9_]+)\s*=\s*\[(.*?)\]\s*;", re.S)
_RELEVANT = re.compile(r"\b(?:FLUX|XS)_[A-Za-z0-9_]+\b")


@dataclass(frozen=True)
class MicroDepletionRow:
    zai: int
    mt: int
    flag: int
    density: float
    density_relative_error: float
    xs: tuple[float, ...]
    xs_relative_error: tuple[float, ...]

    @property
    def macroscopic_xs(self) -> tuple[float, ...]:
        """Central N*XS in cm^-1; not an uncertainty-propagated estimate."""
        return tuple(self.density * x for x in self.xs)


@dataclass(frozen=True)
class MicroDepletionRegion:
    selector: str
    flux: tuple[float, ...]
    flux_relative_error: tuple[float, ...]
    rows: Mapping[tuple[int, int, int], MicroDepletionRow]


@dataclass(frozen=True)
class SerpentMicroDepletion:
    source: Path
    source_sha256: str
    energy_bounds_mev: tuple[float, ...]
    energy_order: str
    regions: Mapping[str, MicroDepletionRegion]
    flags: tuple[str, ...]


def _numbers(text):
    tokens = text.replace(',', ' ').split()
    if any(not _NUM.fullmatch(t) for t in tokens):
        raise ValueError('expected finite numeric literals, not MATLAB expressions')
    values = [float(t.replace('D', 'E').replace('d', 'e')) for t in tokens]
    if not all(np.isfinite(values)):
        raise ValueError('finite numeric literals required')
    return values


def read_serpent_mdep(path, *, energy_bounds_mev, energy_order, selectors) -> SerpentMicroDepletion:
    """Read exact region selectors with explicit group provenance and no execution.

    Native rows have 5+2G columns. Newlines or semicolons delimit matrix rows.
    All imported assignments, including unselected regions, are checked for
    duplicates and malformed literals. A selected region needs both flux and XS.
    No missing isotope/reaction is fabricated; coverage is checked separately by
    :func:`production_xs`. Metadata/version qualification belongs to the caller.
    """
    bounds = tuple(float(x) for x in energy_bounds_mev)
    if (len(bounds) < 2 or not np.all(np.isfinite(bounds)) or min(bounds) < 0
            or energy_order not in ('ascending', 'descending')):
        raise ValueError('finite nonnegative ordered energy bounds required')
    difference = np.diff(bounds)
    if not np.all(difference > 0 if energy_order == 'ascending' else difference < 0):
        raise ValueError('bounds disagree with declared energy order')
    selectors = tuple(selectors)
    if (not selectors or any(not isinstance(s, str) or not re.fullmatch(r'[A-Za-z0-9_]+', s)
                             for s in selectors) or len(set(selectors)) != len(selectors)):
        raise ValueError('unique exact string selectors required')
    path = Path(path)
    raw = path.read_bytes()
    # Native numeric tables contain no strings. Comments cannot hide execution:
    # nothing is evaluated and relevant leftovers are rejected below.
    text = re.sub(r'%[^\n]*', '', raw.decode('utf-8'))
    assignments = {}
    spans = []
    groups = len(bounds) - 1
    for match in _ASSIGN.finditer(text):
        kind, selector, body = match.groups()
        key = kind, selector
        if key in assignments:
            raise ValueError(f'duplicate {kind}_{selector}')
        if kind == 'FLUX':
            values = _numbers(body.replace(';', ' '))
            if len(values) != 2 * groups or min(values, default=-1) < 0:
                raise ValueError(f'FLUX_{selector}: expected G nonnegative mean/error pairs')
            assignments[key] = tuple(values)
        else:
            rows = {}
            for line in re.split(r'[;\n]', body):
                if not line.strip():
                    continue
                values = _numbers(line)
                if len(values) != 5 + 2 * groups:
                    raise ValueError(f'XS_{selector}: expected 5+2G columns per row')
                if any(x != int(x) for x in values[:3]) or values[0] <= 0 or min(values[1:]) < 0:
                    raise ValueError(f'XS_{selector}: invalid integer key or nonnegative data')
                zai, mt, flag = map(int, values[:3])
                rowkey = zai, mt, flag
                if rowkey in rows:
                    raise ValueError(f'XS_{selector}: duplicate isotope/reaction/flag row')
                rows[rowkey] = MicroDepletionRow(zai, mt, flag, values[3], values[4],
                                                tuple(values[5::2]), tuple(values[6::2]))
            if not rows:
                raise ValueError(f'XS_{selector}: empty table')
            assignments[key] = MappingProxyType(rows)
        spans.append(match.span())
    remainder = text
    for start, end in reversed(spans):
        remainder = remainder[:start] + ' ' * (end - start) + remainder[end:]
    if _RELEVANT.search(remainder):
        raise ValueError('unsupported or unterminated relevant assignment')
    regions = {}
    flags = ['version_not_embedded', 'group_bounds_caller_supplied',
             'product_covariance_unavailable']
    for selector in selectors:
        if ('FLUX', selector) not in assignments or ('XS', selector) not in assignments:
            raise ValueError(f'{selector}: missing exact FLUX or XS assignment')
        flux = assignments['FLUX', selector]
        rows = assignments['XS', selector]
        regions[selector] = MicroDepletionRegion(selector, flux[::2], flux[1::2], rows)
        if any(e == 0 for e in flux[1::2]) or any(r.density_relative_error == 0 or
                0 in r.xs_relative_error for r in rows.values()):
            flags.append(f'printed_zero_relative_error:{selector}')
    return SerpentMicroDepletion(path, hashlib.sha256(raw).hexdigest(), bounds,
                                energy_order, MappingProxyType(regions), tuple(flags))


@dataclass(frozen=True)
class IsotopeProduction:
    zai: int
    total: tuple[float, ...]
    delayed: tuple[float, ...]
    prompt: tuple[float, ...]


def production_xs(region: MicroDepletionRegion, *, expected_isotopes) -> Mapping[int, IsotopeProduction]:
    """Require complete explicit paired MT452/455 coverage (flag=0).

    Additional production isotopes or nonzero special flags are errors. Other MT
    rows do not affect this extraction. Densities and their printed relative
    errors must agree exactly between paired rows. Central delayed production
    may not exceed total; no clipping or alteration of frozen steady XS occurs.
    """
    expected = tuple(expected_isotopes)
    if not expected or any(isinstance(z, bool) or not isinstance(z, (int, np.integer)) or z <= 0
           for z in expected) or len(set(expected)) != len(expected):
        raise ValueError('unique positive integer expected isotopes required')
    keys = {key for key in region.rows if key[1] in (452, 455)}
    required = {(z, mt, 0) for z in expected for mt in (452, 455)}
    if keys != required:
        raise ValueError(f'explicit isotope/reaction coverage mismatch: missing={sorted(required-keys)}, extra={sorted(keys-required)}')
    result = {}
    for zai in expected:
        total, delayed = region.rows[zai, 452, 0], region.rows[zai, 455, 0]
        if (total.density, total.density_relative_error) != (delayed.density, delayed.density_relative_error):
            raise ValueError(f'{zai}: paired production density mismatch')
        t, d = total.macroscopic_xs, delayed.macroscopic_xs
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(d)):
            raise ValueError('nonfinite macroscopic product')
        if any(b > a for a, b in zip(t, d)):
            raise ValueError(f'{zai}: delayed production exceeds total')
        result[zai] = IsotopeProduction(zai, t, d, tuple(a-b for a, b in zip(t, d)))
    return MappingProxyType(result)


@dataclass(frozen=True)
class DelayedProductionChannels:
    """Raw isotope/family source matrices, never adjoint-effective fractions."""
    labels: tuple[tuple[int, int], ...]
    decay: tuple[float, ...]
    production: tuple[tuple[tuple[float, ...], ...], ...]  # I,target,source
    qualification: Mapping[int, str]


def build_delayed_channels(region: MicroDepletionRegion, *, expected_isotopes,
                           branching, decay, chi_delayed, qualification) -> DelayedProductionChannels:
    """Expand each isotope's MT455 into explicitly qualified constant branches.

    Every mapping must cover exactly the expected isotopes. Each branching
    vector sums to unity; decay and target spectra are per isotope/family.
    Energy independence is a caller-qualified assumption, not inferred from
    group-averaged effective beta. Spectra must already be normalized; they are
    never silently renormalized. Output C[i,target,source] has units cm^-1.
    """
    products = production_xs(region, expected_isotopes=expected_isotopes)
    expected = set(products)
    if any(set(m) != expected for m in (branching, decay, chi_delayed, qualification)):
        raise ValueError('channel mappings must match explicit isotope coverage')
    labels, rates, matrices = [], [], []
    g = len(region.flux)
    for zai, product in products.items():
        arrays = tuple(np.asarray(m[zai]) for m in (branching, decay, chi_delayed))
        if any(np.iscomplexobj(a) for a in arrays):
            raise ValueError(f'{zai}: real branches, decay, and spectra required')
        b, lam, chi = (np.asarray(a, dtype=float) for a in arrays)
        if (b.ndim != 1 or not b.size or lam.shape != b.shape or chi.shape != (b.size, g)
                or not all(np.all(np.isfinite(a)) for a in (b, lam, chi))
                or np.any(b < 0) or np.any(lam <= 0) or np.any(chi < 0)
                or not np.isclose(b.sum(), 1, rtol=0, atol=1e-12)
                or not np.allclose(chi.sum(axis=1), 1, rtol=0, atol=1e-12)):
            raise ValueError(f'{zai}: invalid branches, decay, or normalized family spectra')
        if not isinstance(qualification[zai], str) or not qualification[zai].strip():
            raise ValueError(f'{zai}: explicit constant-branch qualification required')
        for i in range(b.size):
            labels.append((zai, i+1))
            rates.append(float(lam[i]))
            matrix = np.outer(chi[i], np.asarray(product.delayed) * b[i])
            if not np.all(np.isfinite(matrix)):
                raise ValueError('nonfinite delayed production matrix')
            matrices.append(tuple(tuple(float(x) for x in row) for row in matrix))
    return DelayedProductionChannels(tuple(labels), tuple(rates), tuple(matrices),
                                      MappingProxyType(dict(qualification)))
