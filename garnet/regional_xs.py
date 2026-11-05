"""Opt-in regional isotropic constants; no fitting or physical qualification.

Matrices are from-group major, as in the scalar Serpent reader. Native means
and printed relative errors are retained separately from effective constants.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import re
from types import MappingProxyType

import numpy as np

from .serpent_xs import SerpentXSRecord, read_serpent_results


def _freeze(value):
    if isinstance(value, np.ndarray):
        return np.frombuffer(value.astype(float).tobytes(), dtype=float).reshape(value.shape)
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (dict, MappingProxyType)):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError("source_sha256 must contain 64 hexadecimal digits")


@dataclass(frozen=True)
class RegionMapping:
    """Caller-supplied identity, never inferred from a universe name."""
    region_id: str
    material: str
    pin: str
    ring: int
    axial: int
    component: str = "fuel"

    def __post_init__(self):
        for key in ("region_id", "material", "pin", "component"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"{key} must be a nonempty string")
        for key in ("ring", "axial"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"{key} must be a nonnegative integer")


@dataclass(frozen=True)
class CurveProvenance:
    """Explicit correction evidence supplied by the caller, not verified here.

    The digest identifies the correction source, independently of the result
    file digest. Evidence must describe its temperature/correction applicability.
    """
    source: str
    source_sha256: str
    temperature_k: float
    temperature_evidence: str

    def __post_init__(self):
        _sha(self.source_sha256)
        for key in ("source", "temperature_evidence"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"curve {key} must be supplied explicitly")
        if not np.isfinite(self.temperature_k) or self.temperature_k <= 0:
            raise ValueError("curve temperature_k must be finite and positive")


@dataclass(frozen=True)
class RegionalXS:
    mapping: RegionMapping
    mode: str
    total: np.ndarray
    diffusion: np.ndarray
    reduced_absorption: np.ndarray
    production: np.ndarray
    nu_sigma_f: np.ndarray
    chi: np.ndarray
    kappa_fission: np.ndarray | None
    raw_means: dict
    raw_relative_errors: dict
    provenance: dict

    def __post_init__(self):
        if not isinstance(self.mapping, RegionMapping):
            raise ValueError("explicit RegionMapping required")
        if self.mode not in ("transport-corrected", "physical-isotropic"):
            raise ValueError("unknown regional mode")
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        g = self.total.size
        for name in ("total", "diffusion", "reduced_absorption", "nu_sigma_f", "chi"):
            array = getattr(self, name)
            if array.shape != (g,) or not np.all(np.isfinite(array)) or np.any(array < 0):
                raise ValueError(f"invalid {name} shape/values")
        if g < 1 or np.any(self.total <= 0) or np.any(self.diffusion <= 0):
            raise ValueError("total and diffusion must be positive")
        if self.production.shape != (g, g) or not np.all(np.isfinite(self.production)) or np.any(self.production < 0):
            raise ValueError("negative or invalid effective production")
        if self.kappa_fission is not None and (self.kappa_fission.shape != (g,)
                or not np.all(np.isfinite(self.kappa_fission)) or np.any(self.kappa_fission < 0)):
            raise ValueError("invalid kappa_fission")
        edges = np.asarray(self.provenance["energy_bounds_mev"])
        if edges.shape != (g + 1,) or not np.all(np.isfinite(edges)) or np.any(edges < 0) or np.any(np.diff(edges) >= 0):
            raise ValueError("group boundaries must be descending G+1 values")
        _sha(self.provenance["source_sha256"])

    def as_material(self):
        """Copy effective arrays into the existing diffusion Material interface.

        Normalize native nonzero chi as the scalar adapter does. An all-zero
        fissile chi is rejected; an all-zero nonfissile chi gets group zero.
        Native chi in this immutable snapshot remains unchanged.
        """
        from .materials import Material
        chi = self.chi.copy()
        if chi.sum() == 0:
            if np.any(self.nu_sigma_f > 0):
                raise ValueError("fissile regional material requires nonzero chi")
            chi[0] = 1.
        else:
            chi /= chi.sum()
        return Material(diffusion=self.diffusion.copy(), sigma_a=self.reduced_absorption.copy(),
                        nu_sigma_f=self.nu_sigma_f.copy(), sigma_s=self.production.copy(),
                        chi=chi, total=self.total.copy(),
                        kappa_fission=None if self.kappa_fission is None else self.kappa_fission.copy(),
                        name=f"Experimental regional {self.mapping.region_id} ({self.mode})")


def regional_material_from_record(record, *, mapping, mode,
                                  balance_sigma, balance_atol=0.,
                                  curve_provenance=None):
    """Construct effective constants with an explicit mode and balance policy.

    Acceptance: |TOT-RABSXS-sum(SP0,out)| <= balance_atol + balance_sigma
    * (|TOT|err_TOT + |RABSXS|err_RABSXS + sum(|SP0|err_SP0)). This is a
    caller-declared absolute-error envelope, not a covariance or confidence
    model. The same residual is preserved by the effective diagonal transform.
    """
    if mode not in ("transport-corrected", "physical-isotropic"):
        raise ValueError("mode must be transport-corrected or physical-isotropic")
    if not isinstance(mapping, RegionMapping):
        raise ValueError("explicit RegionMapping required")
    if (not np.isfinite(balance_sigma) or balance_sigma < 0
            or not np.isfinite(balance_atol) or balance_atol < 0):
        raise ValueError("balance tolerances must be finite and nonnegative")
    if mode == "transport-corrected" and not isinstance(curve_provenance, CurveProvenance):
        raise ValueError("explicit CurveProvenance with source/hash/temperature evidence required")
    record._validate()
    g = record.n_groups
    means, errors = {}, {}
    # Retain every recognized native paired estimate, including physical S0.
    for key in record.fields:
        try:
            estimate = record.estimate(key)
        except ValueError:
            continue
        means[key] = estimate.mean
        errors[key] = estimate.relative_error
    total = means["INF_TOT"]
    reduced = means["INF_RABSXS"]
    production = means["INF_SP0"].reshape(g, g).copy()
    residual = total - reduced - production.sum(axis=1)
    envelope = (np.abs(total) * errors["INF_TOT"] + np.abs(reduced) * errors["INF_RABSXS"]
                + (np.abs(production) * errors["INF_SP0"].reshape(g, g)).sum(axis=1))
    tolerance = balance_atol + balance_sigma * envelope
    if np.any(np.abs(residual) > tolerance):
        raise ValueError("scalar production balance exceeds declared raw-error tolerance")
    if mode == "transport-corrected":
        diffusion = record.estimate("TRC_DIFFCOEF").mean
        if diffusion.shape != (g,) or not np.all(np.isfinite(diffusion)) or np.any(diffusion <= 0):
            raise ValueError("TRC_DIFFCOEF must contain G positive finite values")
        effective_total = 1. / (3. * diffusion)
        production[np.diag_indices(g)] -= total - effective_total
    else:
        effective_total = total.copy()
        diffusion = 1. / (3. * total)
    if np.any(production < 0):
        raise ValueError("negative effective production; no clipping or fallback")
    chi = means.get("INF_CHIT", np.zeros(g))
    kappa = (means["INF_FISS"] * means["INF_KAPPA"] if "INF_FISS" in means else None)
    provenance = dict(source=str(record.source), source_sha256=record.source_sha256,
                      record_index=record.record_index, universe=record.universe,
                      version=record.version, energy_bounds_mev=record.energy_bounds_mev,
                      orientation="from-major", covariance_inferred=False,
                      physical_applicability_verified=False,
                      balance_policy="absolute-error envelope; no covariance model",
                      balance_sigma=float(balance_sigma), balance_atol=float(balance_atol),
                      balance_residual=residual, balance_tolerance=tolerance,
                      curve_provenance=(vars(curve_provenance).copy() if curve_provenance else None))
    return RegionalXS(mapping, mode, effective_total, diffusion, reduced, production,
                      means["INF_NSF"], chi, kappa, means, errors, provenance)


@dataclass(frozen=True)
class RegionDataset:
    regions: tuple

    def __post_init__(self):
        object.__setattr__(self, "regions", tuple(self.regions))
        if not self.regions:
            raise ValueError("regional dataset must not be empty")
        ids, locations, records = set(), set(), set()
        edges = self.regions[0].provenance["energy_bounds_mev"]
        for region in self.regions:
            identity = region.mapping
            location = (identity.pin, identity.component, identity.ring, identity.axial)
            source_record = (region.provenance["source_sha256"], region.provenance["record_index"])
            if identity.region_id in ids or location in locations or source_record in records:
                raise ValueError("ambiguous duplicate region, pin/component/ring/axial, or source record mapping")
            if not np.array_equal(edges, region.provenance["energy_bounds_mev"]):
                raise ValueError("regional group boundaries differ")
            ids.add(identity.region_id)
            locations.add(location)
            records.add(source_record)

    def write_manifest(self, path):
        """Write portable means, raw printed errors and provenance, no binaries."""
        regions = []
        for region in self.regions:
            item = {name: _plain(getattr(region, name)) for name in region.__dataclass_fields__ if name != "mapping"}
            item["mapping"] = vars(region.mapping).copy()
            regions.append(item)
        Path(path).write_text(json.dumps({"schema": "garnet.regional-xs.v1", "regions": regions},
                                         indent=2, allow_nan=False) + "\n")

    @classmethod
    def read_manifest(cls, path):
        payload = json.loads(Path(path).read_text())
        if payload.get("schema") not in ("garnet.regional-xs.v1", "ndgpu.regional-xs.v1"):
            raise ValueError("unsupported regional manifest schema")
        regions = []
        for raw in payload["regions"]:
            item = dict(raw)
            item["mapping"] = RegionMapping(**item["mapping"])
            for name in ("total", "diffusion", "reduced_absorption", "production", "nu_sigma_f", "chi", "kappa_fission"):
                if item[name] is not None:
                    item[name] = np.asarray(item[name], dtype=float)
            for name in ("raw_means", "raw_relative_errors"):
                item[name] = {key: np.asarray(value, dtype=float) for key, value in item[name].items()}
            region = RegionalXS(**item)
            _validate_manifest_region(region)
            regions.append(region)
        return cls(tuple(regions))


def _validate_manifest_region(region):
    """Reconstruct adapter admission and require exact derived means/policy."""
    provenance = region.provenance
    if (type(provenance.get("record_index")) is not int or provenance["record_index"] < 0
            or provenance.get("physical_applicability_verified") is not False
            or provenance.get("covariance_inferred") is not False):
        raise ValueError("manifest provenance cannot assert physical verification or covariance")
    fields = dict(VERSION=provenance["version"], GC_UNIVERSE_NAME=provenance["universe"],
                  MACRO_NG=np.array([region.total.size]),
                  MACRO_E=np.asarray(provenance["energy_bounds_mev"]))
    if set(region.raw_means) != set(region.raw_relative_errors):
        raise ValueError("manifest native means/errors fields differ")
    for key, mean in region.raw_means.items():
        error = region.raw_relative_errors[key]
        if (mean.ndim != 1 or mean.shape != error.shape or not np.all(np.isfinite(mean))
                or not np.all(np.isfinite(error)) or np.any(error < 0)):
            raise ValueError("invalid manifest native means/errors")
        # ADF geometry is not part of this means-only regional manifest; its
        # estimates remain auditable, but are not needed for scalar admission.
        if not key.startswith("DF_"):
            fields[key] = np.column_stack((mean, error)).ravel()
    evidence = provenance.get("curve_provenance")
    try:
        curve = CurveProvenance(**evidence) if evidence is not None else None
    except (TypeError, KeyError) as exc:
        raise ValueError("invalid manifest curve evidence") from exc
    record = SerpentXSRecord(Path(provenance["source"]), provenance["source_sha256"],
                             provenance["record_index"], fields)
    expected = regional_material_from_record(record, mapping=region.mapping, mode=region.mode,
        curve_provenance=curve, balance_sigma=provenance["balance_sigma"],
        balance_atol=provenance["balance_atol"])
    for key in ("total", "diffusion", "reduced_absorption", "production", "nu_sigma_f", "chi", "kappa_fission"):
        if not np.array_equal(getattr(region, key), getattr(expected, key)):
            raise ValueError(f"manifest {key} differs from native data and declared mode")
    if _plain(provenance) != _plain(expected.provenance):
        raise ValueError("manifest provenance differs from reconstructed admission policy")


def read_regional_dataset(path, selections, **adapter_options):
    """Read explicitly selected (RegionMapping, zero-based record_index) pairs.

    Universe names are retained for audit, never used to guess pin identities.
    One adapter policy applies to the dataset; use the record adapter directly
    for datasets whose regions require different correction evidence.
    """
    records = read_serpent_results(path)
    regions = []
    for mapping, index in selections:
        if type(index) is not int or not 0 <= index < len(records):
            raise ValueError("record_index must select an existing record explicitly")
        regions.append(regional_material_from_record(records[index], mapping=mapping, **adapter_options))
    return RegionDataset(tuple(regions))


def aggregate_power_bins(flux, kappa_fission, volumes, bins):
    """Return bin volume-average power density sum(V*sum_g(phi*kSf))/sum(V).

    Flux normalization is supplied by the caller. No diffusion coefficients or
    cross sections are averaged by this tally-only operation.
    """
    flux, kappa_fission, volumes = (np.asarray(x, dtype=float) for x in (flux, kappa_fission, volumes))
    bins = tuple(bins)
    if (flux.ndim != 2 or kappa_fission.shape != flux.shape or volumes.shape != (flux.shape[0],)
            or len(bins) != flux.shape[0] or not np.all(np.isfinite(flux))
            or not np.all(np.isfinite(kappa_fission)) or not np.all(np.isfinite(volumes))
            or np.any(flux < 0) or np.any(kappa_fission < 0) or np.any(volumes <= 0)):
        raise ValueError("power aggregation requires finite nonnegative N×G flux/kappa and N positive volumes")
    numerator, denominator = {}, {}
    for label, volume, power in zip(bins, volumes, (flux * kappa_fission).sum(axis=1)):
        numerator[label] = numerator.get(label, 0.) + volume * power
        denominator[label] = denominator.get(label, 0.) + volume
    return {label: numerator[label] / denominator[label] for label in numerator}
