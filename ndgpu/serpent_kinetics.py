"""Read core-global Serpent kinetics estimates without executing MATLAB.

Repeated GCU blocks are provenance, not independent observations. This module
never constructs regional ``Kinetics`` inputs from whole-system estimates.
Write timestamps are retained per record, since they can advance while output
blocks are written. Entry ordering follows Serpent's output reference. Version-qualified native
formulas do not establish equivalence with a diffusion kinetics projection.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .serpent_xs import _ASSIGNMENT, _COUNTER, _literal, _statements

_BETA = {"FWD_ANA_BETA_ZERO", "ADJ_IFP_IMP_BETA_EFF", "ADJ_IFP_ANA_BETA_EFF"}
_LAMBDA = {"FWD_ANA_LAMBDA", "ADJ_IFP_IMP_LAMBDA", "ADJ_IFP_ANA_LAMBDA"}
_TIME = {"ADJ_IFP_GEN_TIME", "ADJ_IFP_LIFETIME"}
_ESTIMATES = _BETA | _LAMBDA | _TIME
_INTEGER_METADATA = {"PRECURSOR_GROUPS", "IFP_CHAIN_LENGTH", "POP", "CYCLES", "SKIP"}
_STRING_METADATA = {"VERSION", "TITLE", "INPUT_FILE_NAME"}
_RECORD_METADATA = {"START_DATE", "COMPLETE_DATE"}
_METADATA = _INTEGER_METADATA | _STRING_METADATA
_KNOWN = _ESTIMATES | _METADATA | _RECORD_METADATA
_RELEVANT_PREFIX = re.compile(r"^(?:" + "|".join(sorted(_KNOWN)) + r")\b")


@dataclass(frozen=True)
class SerpentKineticsEstimate:
    """Immutable printed means/errors in native order, with explicit units.

    A zero relative error is retained and does not establish exactness. No
    covariance, significance, or independent sample count is inferred.
    """
    labels: tuple[str, ...]
    mean: tuple[float, ...]
    relative_error: tuple[float, ...]
    unit: str

    @property
    def absolute_error(self) -> tuple[float, ...]:
        return tuple(abs(x) * e for x, e in zip(self.mean, self.relative_error))


@dataclass(frozen=True)
class SerpentKineticsResult:
    """Core-global estimates and exact repetition provenance for one file.

    ``flags`` report unavailable or unqualified information; successful parsing
    is not a qualification of transport convergence or estimator equivalence.
    Mapping values and estimate arrays are immutable. Record indices are zero
    based file order. ``record_count`` counts result blocks, never samples.
    """
    source: Path
    source_sha256: str
    metadata: Mapping[str, str | int]
    estimates: Mapping[str, SerpentKineticsEstimate]
    record_metadata: Mapping[str, tuple[tuple[int, str], ...]]
    native_definitions: Mapping[str, str]
    field_records: Mapping[str, tuple[int, ...]]
    record_count: int
    flags: tuple[str, ...]


def _roundoff_bound(token: str) -> float:
    """Half the last printed decimal place, including scientific notation."""
    value = Decimal(token.replace("D", "E").replace("d", "e"))
    return float(Decimal(5).scaleb(value.as_tuple().exponent - 1))


def read_serpent_kinetics(path) -> SerpentKineticsResult:
    """Import available global kinetics fields, rejecting malformed/conflicting data.

    Missing fields/metadata are explicit flags, not fabricated defaults. Each
    present field is validated, and repeated global values must match exactly
    (numeric values and strings, without averaging). Beta total/family agreement
    uses only the summed half-last-digit bounds of the printed means. MATLAB
    expressions in relevant assignments are rejected; unrelated fields are
    ignored. No complete steady-XS schema is required.
    """
    path = Path(path)
    raw = path.read_bytes()
    records: list[dict] = []
    fields: dict = {}

    def finish():
        if fields:
            records.append(fields.copy())
            fields.clear()

    try:
        for statement in _statements(raw.decode("utf-8")):
            if _COUNTER.fullmatch(statement):
                finish()
                continue
            match = _ASSIGNMENT.fullmatch(statement)
            if not match:
                if _RELEVANT_PREFIX.match(statement):
                    raise ValueError("unsupported relevant assignment")
                continue
            key, _, count, rhs = match.groups()
            if key not in _KNOWN:
                continue
            if key in fields:
                raise ValueError(f"{key}: duplicate assignment in record {len(records)}")
            value = _literal(rhs)
            if isinstance(value, np.ndarray) and value.size != (int(count) if count else 1):
                raise ValueError(f"{key}: declared length differs from numeric data")
            fields[key] = (value, rhs)
        finish()
        if not records:
            raise ValueError("no recognized global kinetics or metadata fields")

        metadata, estimates, observations, locations, flags = {}, {}, {}, {}, []
        unique = {}
        for index, record in enumerate(records):
            for key, (value, rhs) in record.items():
                if key in _RECORD_METADATA:
                    if not isinstance(value, str) or not value:
                        raise ValueError(f"{key}: expected a nonempty string")
                    observations.setdefault(key, []).append((index, value))
                    locations.setdefault(key, []).append(index)
                    continue
                if key in unique:
                    old = unique[key][0]
                    equal = (old == value if isinstance(old, str) and isinstance(value, str)
                             else isinstance(old, np.ndarray) and isinstance(value, np.ndarray)
                             and np.array_equal(old, value))
                    if not equal:
                        raise ValueError(f"{key}: conflicting global repetition in record {index}")
                else:
                    unique[key] = (value, rhs)
                locations.setdefault(key, []).append(index)
        for key in sorted(_METADATA):
            if key not in unique:
                flags.append(f"missing:{key}")
                continue
            value, _ = unique[key]
            if key in _STRING_METADATA:
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{key}: expected a nonempty string")
                metadata[key] = value
            else:
                if (not isinstance(value, np.ndarray) or value.size != 1
                        or not np.isfinite(value[0]) or value[0] != int(value[0])
                        or value[0] < (0 if key == "SKIP" else 1)):
                    raise ValueError(f"{key}: expected {'nonnegative' if key == 'SKIP' else 'positive'} integer")
                metadata[key] = int(value[0])
        for key in sorted(_RECORD_METADATA):
            flags.append(f"per_record_metadata:{key}" if key in observations else f"missing:{key}")
        native_definitions = {}
        qualified_version = metadata.get("VERSION") == "Serpent 2.2.2"
        if qualified_version:
            native_definitions = {
                "ADJ_IFP_IMP_LAMBDA": "Score-weighted arithmetic decay constant: decay-weighted delayed scores divided by delayed scores, total then families. The total uses beta-effective weighting at score/batch level; independently printed means need not obey that identity exactly.",
                "ADJ_IFP_ANA_LAMBDA": "Score-weighted arithmetic decay constant using analog delayed counts, total then families. Independently printed means need not obey the score-level reduction identity exactly.",
                "FWD_ANA_LAMBDA": "Score-weighted arithmetic decay constant using native delayed-emission lambda-event score denominators, total then families. Those weights are not established by the independently printed FWD_ANA_BETA_ZERO values; the native total is authoritative.",
                "ADJ_IFP_GEN_TIME": "Corresponding total/prompt/delayed IFP lifetime divided by analog cycle k_eff during result collection. Independently printed means need not obey an exact ratio identity.",
                "ADJ_IFP_LIFETIME": "Separately scored IFP progenitor lifetime categories: total, prompt, delayed.",
            }
        families = metadata.get("PRECURSOR_GROUPS")
        for key in sorted(_ESTIMATES):
            if key not in unique:
                flags.append(f"missing:{key}")
                continue
            value, rhs = unique[key]
            if (not isinstance(value, np.ndarray) or value.size % 2 or value.size == 0
                    or not np.all(np.isfinite(value))):
                raise ValueError(f"{key}: expected finite mean/relative-error pairs")
            means, errors = value[::2], value[1::2]
            if np.any(errors < 0):
                raise ValueError(f"{key}: relative errors must be nonnegative")
            if key in _TIME:
                if means.size != 3:
                    raise ValueError(f"{key}: expected total/prompt/delayed entries")
                labels, unit = ("total", "prompt", "delayed"), "s"
                if qualified_version:
                    if key == "ADJ_IFP_GEN_TIME":
                        flags.append(f"native_time_conversion_uses_analog_cycle_k:{key}")
                else:
                    flags.append(f"unqualified_time_convention:{key}")
            else:
                if families is None:
                    flags.append(f"unqualified_family_count:{key}")
                if means.size < 2 or (families is not None and means.size != families + 1):
                    raise ValueError(f"{key}: expected total followed by PRECURSOR_GROUPS families")
                labels = ("total",) + tuple(f"family_{i}" for i in range(1, means.size))
                unit = "1" if key in _BETA else "s^-1"
            if np.any(means < 0) or (key not in _BETA and np.any(means == 0)):
                raise ValueError(f"{key}: {'nonnegative' if key in _BETA else 'positive'} means required")
            if key in _BETA:
                tokens = rhs.strip().strip("[]").replace(",", " ").split()[::2]
                bound = sum(_roundoff_bound(t) for t in tokens)
                floating = np.finfo(float).eps * max(float(np.sum(np.abs(means))), 1e-300) * 8
                if abs(float(means[0] - means[1:].sum())) > bound + floating:
                    raise ValueError(f"{key}: total differs from family sum beyond printed precision")
            if key in _LAMBDA:
                if qualified_version and key in native_definitions:
                    flags.append(f"native_score_weighted_arithmetic_lambda:{key}")
                else:
                    flags.append(f"unqualified_total_lambda_definition:{key}")
            if np.any(errors == 0):
                flags.append(f"printed_zero_relative_error:{key}")
            estimates[key] = SerpentKineticsEstimate(labels, tuple(float(x) for x in means),
                                                        tuple(float(x) for x in errors), unit)
        for key, indices in sorted(locations.items()):
            if len(indices) != len(records):
                flags.append(f"incomplete_global_repetition:{key}")
        if any(key in estimates for key in _TIME):
            flags.append("unqualified_NDgpu_MC_time_equivalence")
        native_definitions = {key: value for key, value in native_definitions.items() if key in estimates}
        if not qualified_version:
            flags.append("unqualified_version")
        return SerpentKineticsResult(
            path, hashlib.sha256(raw).hexdigest(), MappingProxyType(metadata),
            MappingProxyType(estimates),
            MappingProxyType({key: tuple(values) for key, values in observations.items()}),
            MappingProxyType(native_definitions),
            MappingProxyType({key: tuple(indices) for key, indices in locations.items()}),
            len(records), tuple(flags))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"{path.name}: {exc}") from exc
