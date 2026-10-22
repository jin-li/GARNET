"""Synthetic regional adapter algebra, identity and provenance checks."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from ndgpu.regional_xs import (CurveProvenance, RegionDataset, RegionMapping,
                              aggregate_power_bins, read_regional_dataset,
                              regional_material_from_record)
from ndgpu.serpent_xs import SerpentXSRecord


def paired(mean, error=.01):
    mean = np.asarray(mean).ravel()
    return np.column_stack((mean, np.broadcast_to(error, mean.shape))).ravel()


def record(g=2):
    total = np.array([.5, 1.])[:g]
    reduced = np.array([.1, .2])[:g]
    scatter = np.array([[.3, .1], [.05, .75]]) if g == 2 else np.array([[.4]])
    fields = dict(VERSION="synthetic", GC_UNIVERSE_NAME="fuel",
                  MACRO_NG=np.array([g]), MACRO_E=np.array([20., 1.e-6, 0.]) if g == 2 else np.array([20., 0.]),
                  INF_TOT=paired(total), INF_ABS=paired(reduced),
                  INF_RABSXS=paired(reduced), INF_S0=paired(scatter),
                  INF_SP0=paired(scatter), INF_NSF=paired(np.full(g, .02)),
                  INF_CHIT=paired(np.eye(1, g).ravel()), INF_FLX=paired(np.ones(g)),
                  TRC_DIFFCOEF=paired(1. / (3. * np.array([.4, .8])[:g])),
                  INF_FISS=paired(np.full(g, .01)), INF_KAPPA=paired(np.full(g, 200.)))
    return SerpentXSRecord(Path("synthetic_res.m"), "a" * 64, 0, fields)


def mapping(**changes):
    return RegionMapping(**(dict(region_id="r0", material="fuel", pin="p0", ring=0, axial=0) | changes))


def curve():
    return CurveProvenance("correction-table.json", "b" * 64, 900., "900 K curve tabulated in source, section 2")


def adapt(rec=None, **options):
    return regional_material_from_record(record() if rec is None else rec,
        **(dict(mapping=mapping(), mode="transport-corrected", curve_provenance=curve(),
                balance_sigma=1., balance_atol=1.e-14) | options))


@pytest.mark.parametrize("g", [1, 2])
def test_effective_algebra_retains_native_and_scalar_balance(g):
    native = record(g)
    before = {k: v.copy() for k, v in native.fields.items() if isinstance(v, np.ndarray)}
    result = adapt(native)
    physical = native.estimate("INF_TOT").mean
    sp0 = native.estimate("INF_SP0").mean.reshape(g, g)
    np.testing.assert_allclose(result.total, 1. / (3 * result.diffusion))
    np.testing.assert_allclose(result.production, sp0 - np.diag(physical - result.total))
    np.testing.assert_allclose(result.total - result.production.sum(axis=1), result.reduced_absorption)
    np.testing.assert_allclose(np.diag(result.total) - result.production.T,
                               np.diag(physical) - sp0.T)
    np.testing.assert_array_equal(result.nu_sigma_f, native.estimate("INF_NSF").mean)
    np.testing.assert_array_equal(result.chi, native.estimate("INF_CHIT").mean)
    np.testing.assert_allclose(result.kappa_fission, 2.)
    np.testing.assert_array_equal(result.raw_relative_errors["TRC_DIFFCOEF"], [.01] * g)
    for key, value in before.items():
        np.testing.assert_array_equal(native.fields[key], value)


def test_physical_isotropic_is_separate_explicit_diagnostic():
    native = record()
    result = adapt(native, mode="physical-isotropic", curve_provenance=None)
    np.testing.assert_array_equal(result.total, native.estimate("INF_TOT").mean)
    np.testing.assert_allclose(result.diffusion, 1. / (3 * result.total))
    np.testing.assert_array_equal(result.production, native.estimate("INF_SP0").mean.reshape(2, 2))
    assert result.provenance["physical_applicability_verified"] is False
    assert result.provenance["curve_provenance"] is None


def test_explicit_curve_evidence_and_mode_required():
    with pytest.raises(ValueError, match="CurveProvenance"):
        adapt(curve_provenance=None)
    with pytest.raises(ValueError, match="CurveProvenance"):
        adapt(curve_provenance={"qualified": True})
    with pytest.raises(ValueError, match="temperature_evidence"):
        CurveProvenance("file", "b" * 64, 900., "")
    with pytest.raises(ValueError, match="sha256"):
        CurveProvenance("file", "no digest", 900., "evidence")
    with pytest.raises(ValueError, match="mode"):
        adapt(mode="auto")


@pytest.mark.parametrize("bad", [0., -1., np.nan])
def test_bad_or_absent_transport_coefficient(bad):
    native = record()
    native.fields["TRC_DIFFCOEF"] = paired([bad, .4])
    with pytest.raises(ValueError, match="TRC_DIFFCOEF"):
        adapt(native)
    del native.fields["TRC_DIFFCOEF"]
    with pytest.raises(ValueError, match="TRC_DIFFCOEF.*missing"):
        adapt(native)
    adapt(native, mode="physical-isotropic", curve_provenance=None)


@pytest.mark.parametrize("key, value", [("TRC_DIFFCOEF", paired([1.])),
    ("INF_SP0", paired([.3, .1])), ("INF_RABSXS", paired([-.1, .2])),
    ("INF_SP0", paired([.3, -.1, .05, .75]))])
def test_bad_shape_and_negative_native_production_or_reduced_absorption(key, value):
    native = record()
    native.fields[key] = value
    with pytest.raises(ValueError, match=key):
        adapt(native)


def test_negative_effective_diagonal_rejected_without_clipping():
    native = record()
    native.fields["TRC_DIFFCOEF"] = paired([10., 10.])
    with pytest.raises(ValueError, match="negative effective production"):
        adapt(native)


def test_balance_accepts_only_declared_raw_error_envelope():
    native = record()
    native.fields["INF_TOT"][0] += .02
    with pytest.raises(ValueError, match="scalar production balance"):
        adapt(native)
    accepted = adapt(native, balance_sigma=3.)
    assert accepted.provenance["balance_sigma"] == 3.
    assert accepted.provenance["covariance_inferred"] is False
    native.fields["INF_TOT"][1::2] = 0
    native.fields["INF_SP0"][1::2] = 0
    native.fields["INF_RABSXS"][1::2] = 0
    with pytest.raises(ValueError, match="scalar production balance"):
        adapt(native, balance_sigma=1.e6)


def test_deep_immutability_and_snapshot():
    native = record()
    result = adapt(native)
    with pytest.raises(ValueError):
        result.production[0, 0] = 99
    with pytest.raises(ValueError):
        result.production.setflags(write=True)
    with pytest.raises(TypeError):
        result.raw_relative_errors["new"] = 1
    with pytest.raises(TypeError):
        result.provenance["curve_provenance"]["source"] = "other"
    native.fields["INF_NSF"][0] = 99
    assert result.nu_sigma_f[0] == .02


def test_dataset_manifest_roundtrip(tmp_path):
    result = adapt()
    path = tmp_path / "regions.json"
    RegionDataset((result,)).write_manifest(path)
    restored = RegionDataset.read_manifest(path).regions[0]
    assert restored.mapping == result.mapping
    np.testing.assert_array_equal(restored.production, result.production)
    np.testing.assert_array_equal(restored.raw_means["INF_SP0"], result.raw_means["INF_SP0"])
    assert restored.provenance["source_sha256"] == "a" * 64
    assert restored.provenance["curve_provenance"]["temperature_k"] == 900.
    with pytest.raises(ValueError):
        restored.raw_means["INF_SP0"][0] = 10


@pytest.mark.parametrize("identity", [mapping(), mapping(region_id="other"), mapping(pin="p1")])
def test_duplicate_id_location_or_record_is_ambiguous(identity):
    one = adapt()
    with pytest.raises(ValueError, match="ambiguous"):
        RegionDataset((one, replace(one, mapping=identity)))


def test_multiple_regions_and_boundary_mismatch():
    one = adapt()
    two = replace(one, mapping=mapping(region_id="r1", pin="p1"),
                  provenance=dict(one.provenance) | {"record_index": 1})
    assert len(RegionDataset((one, two)).regions) == 2
    two = replace(two, provenance=dict(two.provenance) | {"energy_bounds_mev": [21., 1.e-6, 0.]})
    with pytest.raises(ValueError, match="boundaries differ"):
        RegionDataset((one, two))
    with pytest.raises(ValueError, match="ring"):
        mapping(ring=-1)


def test_pin_components_and_fuel_rings_have_distinct_locations():
    one = adapt()
    identities = [mapping(), mapping(region_id="fuel1", ring=1)]
    identities.extend(mapping(region_id=name, material=name, component=name)
                      for name in ("gap", "clad", "inner-coolant", "outer-coolant"))
    regions = tuple(replace(one, mapping=identity,
                           provenance=dict(one.provenance) | {"record_index": index})
                    for index, identity in enumerate(identities))
    assert len(RegionDataset(regions).regions) == 6
    duplicate = replace(regions[-1], mapping=mapping(region_id="duplicate", component="outer-coolant"),
                        provenance=dict(one.provenance) | {"record_index": 6})
    with pytest.raises(ValueError, match="ambiguous"):
        RegionDataset((*regions, duplicate))


def test_as_material_copies_effective_arrays_and_normalizes_chi():
    native = adapt()
    region = replace(native, chi=np.array([.9, .099999]))
    material = region.as_material()
    np.testing.assert_array_equal(material.total, region.total)
    np.testing.assert_array_equal(material.diffusion, region.diffusion)
    np.testing.assert_array_equal(material.sigma_s, region.production)
    np.testing.assert_array_equal(material.sigma_a, region.reduced_absorption)
    np.testing.assert_allclose(material.chi, region.chi / region.chi.sum())
    material.sigma_s[0, 0] = 99
    assert region.production[0, 0] != 99
    with pytest.raises(ValueError, match="nonzero chi"):
        replace(native, chi=np.zeros(2)).as_material()
    nonfissile = replace(native, chi=np.zeros(2), nu_sigma_f=np.zeros(2)).as_material()
    np.testing.assert_array_equal(nonfissile.chi, [1., 0.])


@pytest.mark.parametrize("tamper", ["total", "production", "diffusion", "raw-balance",
    "curve-missing", "curve-assertion", "physical-assertion", "balance-policy", "mode"])
def test_manifest_rechecks_derived_arrays_balance_and_curve_admission(tmp_path, tamper):
    path = tmp_path / "regions.json"
    RegionDataset((adapt(),)).write_manifest(path)
    payload = json.loads(path.read_text())
    region = payload["regions"][0]
    if tamper in ("total", "diffusion"):
        region[tamper][0] *= 1.01
    elif tamper == "production":
        region["production"][0][1] += .001
    elif tamper == "raw-balance":
        region["raw_means"]["INF_TOT"][0] += .1
    elif tamper == "curve-missing":
        region["provenance"]["curve_provenance"] = None
    elif tamper == "curve-assertion":
        region["provenance"]["curve_provenance"] = {"qualified": True}
    elif tamper == "physical-assertion":
        region["provenance"]["physical_applicability_verified"] = True
    elif tamper == "balance-policy":
        region["provenance"]["balance_policy"] = "trusted"
    elif tamper == "mode":
        region["mode"] = "physical-isotropic"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        RegionDataset.read_manifest(path)


def test_reader_requires_existing_explicit_record_index(tmp_path):
    native = record()
    path = tmp_path / "regions_res.m"
    lines = []
    for key, value in native.fields.items():
        literal = repr(value) if isinstance(value, str) else "[ " + " ".join(str(x) for x in value) + " ]"
        count = len(value)
        lines.append(f"{key} (idx, [1:{count}]) = {literal};")
    path.write_text("\n".join(lines))
    dataset = read_regional_dataset(path, [(mapping(), 0)], mode="transport-corrected",
        curve_provenance=curve(), balance_sigma=1., balance_atol=1.e-14)
    assert dataset.regions[0].provenance["source"] == str(path)
    with pytest.raises(ValueError, match="record_index"):
        read_regional_dataset(path, [(mapping(), -1)])


def test_power_bin_volume_weighted_density():
    powers = aggregate_power_bins([[1., 2.], [3., 4.], [5., 6.]],
                                  [[2., 3.]] * 3, [1., 3., 2.], ["a", "a", "b"])
    assert powers == {"a": (8. + 3 * 18.) / 4., "b": 28.}
    with pytest.raises(ValueError, match="power aggregation"):
        aggregate_power_bins([[1.]], [[1.]], [0.], ["a"])
