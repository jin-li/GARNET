"""Fixed-state Serpent ``*_res.m`` group constants, parsed without execution.

The diffusion adapter uses reduced absorption and scattering *production* to
preserve (n,xn) neutron multiplication. Physical absorption/collision scattering
remain available separately in the record. This is an infinite-spectrum reader,
not a B1, depletion, interpolation or transport-correction implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import numpy as np

from .materials import Material


_VECTOR_FIELDS = {
    "INF_TOT", "INF_ABS", "INF_RABSXS", "INF_NSF", "INF_CHIT", "INF_FLX",
    "CMM_DIFFCOEF", "INF_DIFFCOEF", "TRC_DIFFCOEF", "INF_FISS", "INF_KAPPA",
}
_MATRIX_FIELDS = {"INF_S0", "INF_SP0"}
_REFERENCE_FIELDS = {"IMP_KEFF", "INF_KINF", "ANA_KEFF", "TOT_LEAKRATE"}
_ADF_VOLUME_FIELDS = {"DF_HET_VOL_FLUX", "DF_HOM_VOL_FLUX"}
_ADF_SURFACE_FIELDS = {
    "DF_HET_SURF_FLUX", "DF_HOM_SURF_FLUX", "DF_SURF_DF",
    "DF_SURF_NET_CURR", "DF_SURF_IN_CURR", "DF_SURF_OUT_CURR",
}
_ADF_SIGN_CURRENT_FIELDS = {
    "DF_SGN_SURF_IN_CURR", "DF_SGN_SURF_OUT_CURR", "DF_SGN_SURF_NET_CURR",
}
_STATISTICAL_FIELDS = (_VECTOR_FIELDS | _MATRIX_FIELDS | _REFERENCE_FIELDS
                       | _ADF_VOLUME_FIELDS | _ADF_SURFACE_FIELDS | _ADF_SIGN_CURRENT_FIELDS)
_ADF_GEOMETRY_FIELDS = {"DF_N_SURF", "DF_VOLUME", "DF_SURF_AREA", "DF_N_SGN"}
_REQUIRED = {
    "VERSION", "GC_UNIVERSE_NAME", "MACRO_NG", "MACRO_E", "INF_TOT",
    "INF_ABS", "INF_RABSXS", "INF_NSF", "INF_FLX",
    "INF_S0", "INF_SP0",
}
_KNOWN = _REQUIRED | _STATISTICAL_FIELDS | _ADF_GEOMETRY_FIELDS
_ASSIGNMENT = re.compile(
    r"^([A-Z][A-Z0-9_]*)\s*\(\s*idx\s*,\s*(1|\[\s*1\s*:\s*(\d+)\s*\])"
    r"\s*\)\s*=\s*(.*?)\s*$", re.S)
_NUMBER = re.compile(
    r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?|inf|nan)$", re.I)
_COUNTER = re.compile(r"^if\s*\(\s*exist\s*\(\s*'idx'\s*,\s*'var'\s*\)\s*\)$")


def _statements(text):
    """Split MATLAB data statements; honor quoted strings and percent comments."""
    current, quoted, comment, brackets = [], False, False, 0
    i = 0
    while i < len(text):
        c = text[i]
        if comment:
            if c == "\n":
                comment = False
                current.append(c)
        elif quoted:
            current.append(c)
            if c == "'":
                if i + 1 < len(text) and text[i + 1] == "'":
                    current.append("'")
                    i += 1
                else:
                    quoted = False
        elif c == "%":
            comment = True
        elif c == "'":
            quoted = True
            current.append(c)
        elif c == ";" and brackets == 0:
            yield "".join(current).strip()
            current = []
        else:
            brackets += (c == "[") - (c == "]")
            current.append(c)
        i += 1
    if quoted or brackets != 0:
        raise ValueError("unterminated string or numeric vector")
    if "".join(current).strip():
        raise ValueError("unterminated statement (missing semicolon)")


def _literal(rhs):
    if re.fullmatch(r"'(?:[^']|'')*'", rhs, re.S):
        return rhs[1:-1].replace("''", "'")
    if rhs.startswith("[") and rhs.endswith("]"):
        rhs = rhs[1:-1].strip()
    tokens = rhs.replace(",", " ").split()
    if not tokens or any(not _NUMBER.fullmatch(t) for t in tokens):
        raise ValueError("expected a numeric literal/vector or quoted string; expressions are unsupported")
    return np.array([float(t.replace("D", "E").replace("d", "e")) for t in tokens])


@dataclass(frozen=True)
class SerpentEstimate:
    """Means and printed relative errors; no covariance is inferred.

    A printed zero error is preserved and does not establish exactness.
    """

    mean: np.ndarray
    relative_error: np.ndarray

    @property
    def absolute_error(self):
        return np.abs(self.mean) * self.relative_error


@dataclass
class SerpentXSRecord:
    source: Path
    source_sha256: str
    record_index: int
    fields: dict

    def _error(self, field, message):
        return ValueError(f"{self.source.name}: record {self.record_index}, {field}: {message}")

    @property
    def version(self):
        return self.fields["VERSION"]

    @property
    def universe(self):
        return self.fields["GC_UNIVERSE_NAME"]

    @property
    def n_groups(self):
        return int(self.fields["MACRO_NG"][0])

    @property
    def energy_bounds_mev(self):
        return self.fields["MACRO_E"].copy()

    @property
    def n_surfaces(self):
        """Number of ADF faces, preserving Serpent's native face order."""
        if "DF_N_SURF" not in self.fields:
            raise self._error("DF_N_SURF", "missing field for ADF surface data")
        return int(self.fields["DF_N_SURF"][0])

    @property
    def n_sign_moments(self):
        """Native sign-current moment count, with no face/tangent inference."""
        if "DF_N_SGN" not in self.fields:
            raise self._error("DF_N_SGN", "missing field for ADF sign-current data")
        return int(self.fields["DF_N_SGN"][0])

    def estimate(self, field):
        """Decode a recognized paired statistical field, preserving raw values."""
        if field not in _STATISTICAL_FIELDS:
            raise self._error(field, "not a recognized mean/error field")
        if field not in self.fields:
            raise self._error(field, "missing field")
        raw = self.fields[field]
        if not isinstance(raw, np.ndarray) or raw.size % 2:
            raise self._error(field, "expected interleaved mean/relative-error pairs")
        return SerpentEstimate(raw[::2].copy(), raw[1::2].copy())

    def adf_estimate(self, field):
        """Decode native ADF means/errors without rescaling or reordering.

        Surface estimates have shape (n_surfaces, G), face-major with group
        innermost; volume estimates have shape (G,). This supports both four
        and six faces. Heterogeneous surface fluxes are surface-average
        densities; partial and net currents are surface-integrated. Serpent
        2.2.2 net current is inward minus outward, so negative values are valid.
        DF_SURF_DF is the heterogeneous/homogeneous surface-flux ratio.
        This decoder returns raw scores, not receiver-method compatibility.
        At nonzero current, a usable ratio requires a homogeneous denominator
        consistent with the receiving method's auxiliary boundary-value solve.
        In Serpent 2.2.2, disabled reconstruction (DFS=1) may leave the printed
        homogeneous surface score as a volume-flux proxy; decoding neither
        replaces that denominator nor qualifies the printed factor for use.
        Printed zero relative errors do not imply exact values.
        """
        if field not in _ADF_VOLUME_FIELDS | _ADF_SURFACE_FIELDS:
            raise self._error(field, "not a recognized ADF mean/error field")
        est = self.estimate(field)
        shape = ((self.n_surfaces, self.n_groups) if field in _ADF_SURFACE_FIELDS
                 else (self.n_groups,))
        if est.mean.size != int(np.prod(shape)):
            raise self._error(field, f"expected {int(np.prod(shape))} mean/error pairs")
        return SerpentEstimate(est.mean.reshape(shape), est.relative_error.reshape(shape))

    def adf_sign_current_estimate(self, field):
        """Decode raw signed current moments as (n_sign_moments,G), group fastest.

        All three sign-weighted IN/OUT/NET current means may be negative.
        Serpent 2.2.2 scores are returned without area scaling or physical
        qualification. The caller must establish tangent ordering and current
        normalization from its exact scorer/version/input. This generic reader
        assumes neither a cuboid nor two moments per face. Sign-weighted flux
        scores are deliberately not decoded or qualified by this method.
        """
        if field not in _ADF_SIGN_CURRENT_FIELDS:
            raise self._error(field, "not a recognized sign-current mean/error field")
        estimate = self.estimate(field)
        shape = (self.n_sign_moments, self.n_groups)
        if estimate.mean.size != int(np.prod(shape)):
            raise self._error(field, f"expected {int(np.prod(shape))} mean/error pairs")
        return SerpentEstimate(estimate.mean.reshape(shape), estimate.relative_error.reshape(shape))

    def _validate(self):
        for key in sorted(_REQUIRED):
            if key not in self.fields:
                raise self._error(key, "missing required field")
        for key in ("VERSION", "GC_UNIVERSE_NAME"):
            if not isinstance(self.fields[key], str) or not self.fields[key]:
                raise self._error(key, "expected a nonempty string")
        ng = self.fields["MACRO_NG"]
        if (not isinstance(ng, np.ndarray) or ng.size != 1
                or not np.isfinite(ng[0]) or ng[0] < 1 or ng[0] != int(ng[0])):
            raise self._error("MACRO_NG", "expected a positive integer")
        edges = self.fields["MACRO_E"]
        if (not isinstance(edges, np.ndarray) or edges.size != self.n_groups + 1
                or not np.all(np.isfinite(edges)) or np.any(edges < 0)
                or not np.all(np.diff(edges) < 0)):
            raise self._error("MACRO_E", "expected G+1 finite nonnegative descending boundaries in MeV")
        if "DF_N_SURF" in self.fields:
            ns = self.fields["DF_N_SURF"]
            if (not isinstance(ns, np.ndarray) or ns.size != 1
                    or not np.isfinite(ns[0]) or ns[0] < 1 or ns[0] != int(ns[0])):
                raise self._error("DF_N_SURF", "expected a positive integer")
        if "DF_N_SGN" in self.fields:
            ns = self.fields["DF_N_SGN"]
            if (not isinstance(ns, np.ndarray) or ns.size != 1
                    or not np.isfinite(ns[0]) or ns[0] < 0 or ns[0] != int(ns[0])):
                raise self._error("DF_N_SGN", "expected a nonnegative integer")
        for key in {"DF_VOLUME", "DF_SURF_AREA"} & self.fields.keys():
            values = self.fields[key]
            expected = 1 if key == "DF_VOLUME" else self.n_surfaces
            if (not isinstance(values, np.ndarray) or values.size != expected
                    or not np.all(np.isfinite(values)) or np.any(values <= 0)):
                raise self._error(key, f"expected {expected} finite strictly positive values")
        for key in _STATISTICAL_FIELDS & self.fields.keys():
            est = self.estimate(key)
            expected = (self.n_groups ** 2 if key in _MATRIX_FIELDS else
                        self.n_surfaces * self.n_groups if key in _ADF_SURFACE_FIELDS else
                        self.n_sign_moments * self.n_groups if key in _ADF_SIGN_CURRENT_FIELDS else
                        self.n_groups if key in _VECTOR_FIELDS | _ADF_VOLUME_FIELDS else
                        3 if key == "ANA_KEFF" else 1)
            if est.mean.size != expected:
                raise self._error(key, f"expected {expected} mean/error pairs")
            if (not np.all(np.isfinite(est.mean))
                    or (key != "DF_SURF_NET_CURR" and key not in _ADF_SIGN_CURRENT_FIELDS and np.any(est.mean < 0))
                    or not np.all(np.isfinite(est.relative_error))
                    or np.any(est.relative_error < 0)):
                raise self._error(key, "means must be finite (nonnegative except net/sign currents); "
                                  "relative errors must be finite and nonnegative")
            if key in {"CMM_DIFFCOEF", "INF_DIFFCOEF", "TRC_DIFFCOEF", "INF_TOT"} and np.any(est.mean <= 0):
                raise self._error(key, "values must be strictly positive")
        fissile = np.any(self.estimate("INF_NSF").mean > 0)
        if "INF_CHIT" not in self.fields:
            if fissile:
                raise self._error("INF_CHIT", "missing required field for fissile material")
        else:
            chi = self.estimate("INF_CHIT").mean
            if not (not fissile and np.all(chi == 0)) and abs(float(chi.sum()) - 1.) > 1e-5:
                raise self._error("INF_CHIT", "spectrum must sum to one within 1e-5")
        if ("INF_FISS" in self.fields) != ("INF_KAPPA" in self.fields):
            raise self._error("INF_FISS/INF_KAPPA", "power fields must both be present or both absent")
        if self.estimate("INF_FLX").mean.sum() <= 0:
            raise self._error("INF_FLX", "flux must have a positive sum")

    def to_diffusion_material(self, diffusion="cmm"):
        """Map INF production balance, with reduced (not physical) absorption.

        This adapter is validated for diffusion eigenpairs. Do not interpret
        ``Material.sigma_a`` as a physical absorption tally, or assume these
        uncorrected data have been validated for SPN/SN or thermal feedback.
        CMM is the default; ``diffusion='inf'`` explicitly opts into Serpent's
        out-scattering diffusion approximation. ``diffusion='trc'`` selects native
        TRC_DIFFCOEF and requires documented correction data/provenance.
        No selection falls back or establishes physical applicability by itself.
        Paired INF_FISS and INF_KAPPA supply kappa_fission in MeV/cm for
        normalized power shapes; raw estimates and errors remain unchanged.
        Nonfissile regions with absent/zero chi use a benign first-group
        spectrum, which contributes no fission source.
        """
        self._validate()
        keys = {"cmm": "CMM_DIFFCOEF", "inf": "INF_DIFFCOEF", "trc": "TRC_DIFFCOEF"}
        if diffusion not in keys:
            raise self._error("diffusion", "use 'cmm', 'inf' or 'trc'")
        chi = (self.estimate("INF_CHIT").mean if "INF_CHIT" in self.fields
               else np.zeros(self.n_groups))
        if chi.sum() == 0:
            chi[0] = 1.
        else:
            chi /= chi.sum()
        kappa_fission = (self.estimate("INF_FISS").mean * self.estimate("INF_KAPPA").mean
                         if "INF_FISS" in self.fields else None)
        return Material(
            diffusion=self.estimate(keys[diffusion]).mean,
            sigma_a=self.estimate("INF_RABSXS").mean,
            nu_sigma_f=self.estimate("INF_NSF").mean,
            sigma_s=self.estimate("INF_SP0").mean.reshape(self.n_groups, self.n_groups),
            chi=chi, total=self.estimate("INF_TOT").mean,
            kappa_fission=kappa_fission,
            name=f"Serpent universe {self.universe} (reduced absorption, production scattering)")

    def balance_diagnostics(self):
        """Report closure without fitting data or assigning a covariance model."""
        g = self.n_groups
        total = self.estimate("INF_TOT").mean
        absorb = self.estimate("INF_ABS").mean
        reduced = self.estimate("INF_RABSXS").mean
        s = self.estimate("INF_S0").mean.reshape(g, g)
        sp = self.estimate("INF_SP0").mean.reshape(g, g)
        residuals = {
            "collision_total_minus_absorption_scatter_out": total - absorb - s.sum(axis=1),
            "production_total_minus_reduced_absorption_production_out": total - reduced - sp.sum(axis=1),
            "reduction_minus_scattering_multiplication": absorb - reduced - (sp - s).sum(axis=1),
        }
        return {
            name: {"per_group_cm_inv": values.tolist(), "max_abs_cm_inv": float(np.max(np.abs(values)))}
            for name, values in residuals.items()
        }


def read_serpent_results(path):
    """Read generated result blocks as literals; never execute their control code.

    Record indices are zero-based file order, independent of MATLAB's ``idx``.
    Unsupported expressions in relevant fields and duplicate assignments fail.
    Unrelated output fields are retained when literal and otherwise ignored.
    """
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    records, fields = [], {}

    def finish():
        if fields:
            record = SerpentXSRecord(path, digest, len(records), fields.copy())
            record._validate()
            records.append(record)
            fields.clear()

    try:
        for statement in _statements(raw.decode("utf-8")):
            if _COUNTER.fullmatch(statement):
                finish()
                continue
            match = _ASSIGNMENT.fullmatch(statement)
            if not match:
                # Unsupported indexing/assignment syntax must not hide a
                # relevant field (including after a malicious prefix).
                if re.match(r"^(?:" + "|".join(sorted(_KNOWN)) + r")\b", statement):
                    raise ValueError(f"record {len(records)}: unsupported field assignment: {statement.split('=')[0].strip()}")
                continue
            key, spec, count, rhs = match.groups()
            try:
                # Empty optional sign-current arrays are meaningful at DF_N_SGN=0.
                # Ordinary-field literal parsing and validation remain unchanged.
                value = (np.array([], dtype=float) if key in _ADF_SIGN_CURRENT_FIELDS and re.fullmatch(r"\[\s*\]", rhs)
                         else _literal(rhs))
            except ValueError as exc:
                if key in _KNOWN:
                    raise ValueError(f"record {len(records)}, {key}: {exc}") from exc
                continue
            if key in fields:
                raise ValueError(f"record {len(records)}, {key}: duplicate assignment")
            if isinstance(value, np.ndarray) and value.size != (int(count) if count else 1):
                raise ValueError(f"record {len(records)}, {key}: declared length differs from numeric data")
            fields[key] = value
        finish()
    except ValueError as exc:
        raise ValueError(f"{path.name}: {exc}") from exc
    if not records:
        raise ValueError(f"{path.name}: no Serpent XS records found")
    return records


def read_serpent_xs(path, *, universe=None, record_index=None):
    """Select exactly one universe/record; ambiguous selections are errors."""
    records = read_serpent_results(path)
    selected = [r for r in records
                if (universe is None or r.universe == str(universe))
                and (record_index is None or r.record_index == record_index)]
    if len(selected) != 1:
        available = [(r.record_index, r.universe) for r in records]
        raise ValueError(f"{Path(path).name}: selection matched {len(selected)} records; "
                         f"choose universe/record_index; available {available}")
    return selected[0]
