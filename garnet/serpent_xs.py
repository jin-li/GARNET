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
# Preserve historical private names for callers that used the scalar parser.
from ._serpent_literals import _ASSIGNMENT, _COUNTER, _NUMBER, _literal, _statements


_VECTOR_FIELDS = {
    "INF_TOT", "INF_ABS", "INF_RABSXS", "INF_NSF", "INF_CHIT", "INF_FLX",
    "CMM_DIFFCOEF", "INF_DIFFCOEF", "TRC_DIFFCOEF", "INF_FISS", "INF_KAPPA",
}
_MATRIX_FIELDS = {"INF_S0", "INF_SP0"}
_MAX_MOMENT_ORDER = 7
_REFERENCE_FIELDS = {"IMP_KEFF", "INF_KINF", "ANA_KEFF", "TOT_LEAKRATE"}
_ADF_VOLUME_FIELDS = {"DF_HET_VOL_FLUX", "DF_HOM_VOL_FLUX"}
_ADF_SURFACE_FIELDS = {
    "DF_HET_SURF_FLUX", "DF_HOM_SURF_FLUX", "DF_SURF_DF",
    "DF_SURF_NET_CURR", "DF_SURF_IN_CURR", "DF_SURF_OUT_CURR",
}
_STATISTICAL_FIELDS = (_VECTOR_FIELDS | _MATRIX_FIELDS | _REFERENCE_FIELDS
                       | _ADF_VOLUME_FIELDS | _ADF_SURFACE_FIELDS)
_ADF_GEOMETRY_FIELDS = {"DF_N_SURF", "DF_VOLUME", "DF_SURF_AREA"}
_REQUIRED = {
    "VERSION", "GC_UNIVERSE_NAME", "MACRO_NG", "MACRO_E", "INF_TOT",
    "INF_ABS", "INF_RABSXS", "INF_NSF", "INF_FLX",
    "INF_S0", "INF_SP0",
}
_KNOWN = _REQUIRED | _STATISTICAL_FIELDS | _ADF_GEOMETRY_FIELDS


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
                        self.n_groups if key in _VECTOR_FIELDS | _ADF_VOLUME_FIELDS else
                        3 if key == "ANA_KEFF" else 1)
            if est.mean.size != expected:
                raise self._error(key, f"expected {expected} mean/error pairs")
            if (not np.all(np.isfinite(est.mean))
                    or (key != "DF_SURF_NET_CURR" and np.any(est.mean < 0))
                    or not np.all(np.isfinite(est.relative_error))
                    or np.any(est.relative_error < 0)):
                raise self._error(key, "means must be finite (nonnegative except net current); "
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


def read_serpent_results(path, *, moment_orders=None):
    """Read generated result blocks as literals; never execute their control code.

    Record indices are zero-based file order, independent of MATLAB's ``idx``.
    Unsupported expressions in relevant fields and duplicate assignments fail.
    Unrelated output fields are retained when literal and otherwise ignored.
    ``moment_orders`` explicitly enables higher signed moment validation; use
    ``read_serpent_moment_results`` for the documented opt-in API.
    """
    orders = _moment_orders(moment_orders) if moment_orders is not None else None
    moment_fields = (
        {f"INF_{kind}{order}"
         for order in range(1, _MAX_MOMENT_ORDER + 1) for kind in ("S", "SP")}
        if orders is not None else set()
    )
    known = _KNOWN | moment_fields
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    records, fields = [], {}

    def finish():
        if fields:
            record = (SerpentXSRecord(path, digest, len(records), fields.copy()) if orders is None
                      else SerpentMomentRecord(path, digest, len(records), fields.copy(), orders))
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
                if re.match(r"^(?:" + "|".join(sorted(known)) + r")\b", statement):
                    raise ValueError(f"record {len(records)}: unsupported field assignment: {statement.split('=')[0].strip()}")
                continue
            key, spec, count, rhs = match.groups()
            if orders is not None and re.fullmatch(r"INF_SP?\d+", key) and key not in _MATRIX_FIELDS | moment_fields:
                raise ValueError(f"record {len(records)}, {key}: unsupported moment order")
            try:
                value = _literal(rhs)
            except ValueError as exc:
                if key in known:
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


def _select_record(records, path, *, universe, record_index):
    """Apply the same exact selection contract to scalar and moment records."""
    selected = [record for record in records
                if (universe is None or record.universe == str(universe))
                and (record_index is None or record.record_index == record_index)]
    if len(selected) != 1:
        available = [(record.record_index, record.universe) for record in records]
        raise ValueError(f"{Path(path).name}: selection matched {len(selected)} records; "
                         f"choose universe/record_index; available {available}")
    return selected[0]


def read_serpent_xs(path, *, universe=None, record_index=None):
    """Select exactly one universe/record; ambiguous selections are errors."""
    return _select_record(read_serpent_results(path), path,
                          universe=universe, record_index=record_index)


def _moment_orders(orders):
    orders = tuple(orders)
    if (not orders or any(type(order) is not int for order in orders)
            or orders != tuple(range(max(orders) + 1))
            or max(orders) > _MAX_MOMENT_ORDER):
        raise ValueError(f"moment orders must be consecutive integers 0..L, L <= {_MAX_MOMENT_ORDER}")
    return orders


@dataclass(frozen=True)
class NativeMomentEstimate:
    """Finite signed native means/reported pairs; error sign meaning unresolved.

    No absolute error, valid SE or covariance interpretation is supplied.
    """
    mean: np.ndarray
    reported_error: np.ndarray


@dataclass
class SerpentMomentRecord(SerpentXSRecord):
    required_orders: tuple = (0, 1, 2, 3)

    def moment_estimate(self, kind, order):
        if kind not in ("S", "SP") or type(order) is not int or not 0 <= order <= _MAX_MOMENT_ORDER:
            raise self._error("moment", f"use S/SP and integer order 0..{_MAX_MOMENT_ORDER}")
        key = f"INF_{kind}{order}"
        if key not in self.fields:
            raise self._error(key, "missing required moment")
        raw = self.fields[key]
        if (not isinstance(raw, np.ndarray) or raw.size != 2 * self.n_groups ** 2
                or not np.all(np.isfinite(raw))):
            raise self._error(key, "expected G² finite signed mean/reported-error pairs")
        return NativeMomentEstimate(raw[::2].reshape(self.n_groups, self.n_groups).copy(),
                                    raw[1::2].reshape(self.n_groups, self.n_groups).copy())

    def _validate(self):
        super()._validate()
        for order in range(1, _MAX_MOMENT_ORDER + 1):
            for kind in ("S", "SP"):
                if order in self.required_orders or f"INF_{kind}{order}" in self.fields:
                    self.moment_estimate(kind, order)

    def to_moment_material(self, *, orientation, orders=None):
        from .moment_materials import moment_material_from_record
        return moment_material_from_record(self, orientation=orientation, orders=orders)


def read_serpent_moment_results(path, *, required_orders=(0,1,2,3)):
    """Explicit opt-in signed S1..7/SP1..7 literal validation.

    Legacy readers remain unchanged. Field availability is not physical admission;
    native orientation must be supplied explicitly to the material adapter.
    """
    return read_serpent_results(path, moment_orders=required_orders)


def read_serpent_moment_xs(path, *, universe=None, record_index=None, required_orders=(0,1,2,3)):
    records = read_serpent_moment_results(path, required_orders=required_orders)
    return _select_record(records, path, universe=universe, record_index=record_index)
