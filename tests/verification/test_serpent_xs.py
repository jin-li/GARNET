"""Literal-only Serpent result parsing and reflective multigroup verification.

Real Serpent files are optional local inputs; no generated result is vendored.
"""

import hashlib
from pathlib import Path
import re

import numpy as np
import pytest

from ndgpu import DiffusionEigenSolver, Grid
from ndgpu.serpent_xs import read_serpent_results, read_serpent_xs


_COUNTER = """if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;
"""
_FIELDS = {
    "VERSION": "'Serpent 2.2.0'",
    "GC_UNIVERSE_NAME": "'fuel'",
    "MACRO_NG": "2",
    "MACRO_E": "[ 20 6.25D-7 0 ]",
    "INF_RABSXS": "[ 0.01 0.1 0.10 0.02 ]",
    "INF_ABS": "[ 0.01 0.1 0.10 0.02 ]",
    "INF_SP0": "[ 0.44 0.01 0.05 0.02 0.001 0.03 0.899 0.04 ]",
    "INF_S0": "[ 0.44 0.01 0.05 0.02 0.001 0.03 0.899 0.04 ]",
    "INF_NSF": "[ 0.005 0.02 0.15 0.03 ]",
    "INF_CHIT": "[ 0.95 0.01 0.05 0.02 ]",
    "INF_TOT": "[ 0.50 0.01 1.00 0.02 ]",
    "CMM_DIFFCOEF": "[ 1.25 0.01 0.25 0.02 ]",
    "INF_DIFFCOEF": "[ 1.5 0.01 0.3 0.02 ]",
    "INF_FLX": "[ 3.0 0.01 1.0 0.02 ]",
    "IMP_KEFF": "[ 1.0 0.001 ]",
}


def _block(overrides=None):
    fields = _FIELDS | (overrides or {})
    lines = []
    for name, value in fields.items():
        if value is None:
            continue
        if value.startswith("'"):
            count = len(value[1:-1].replace("''", "'"))
        else:
            literal = re.sub(r"%[^\n]*", "", value).strip("[] ")
            count = len(literal.split())
        lines.append(f"{name} (idx, [1: {count}]) = {value};")
    return _COUNTER + "\n".join(lines) + "\n"


def _write(tmp_path, text=None, overrides=None):
    path = tmp_path / "synthetic_res.m"
    path.write_text(_block(overrides) if text is None else text)
    return path


def _dense(material):
    scatter = material.sigma_s.copy()
    np.fill_diagonal(scatter, 0.0)
    loss = np.diag(material.removal) - scatter.T
    production = np.outer(material.chi, material.nu_sigma_f)
    values, vectors = np.linalg.eig(np.linalg.solve(loss, production))
    idx = np.argmax(values.real)
    spectrum = vectors[:, idx].real
    spectrum /= spectrum.sum()
    return values[idx].real, spectrum


def _reflective(material, size=1.0):
    grid = Grid(shape=(1, 1, 1), size=(size,) * 3)
    result = DiffusionEigenSolver(
        grid, material, device="cpu", bc="reflective"
    ).solve(max_outer=2000, tol_k=1e-11, tol_source=1e-10)
    assert result.converged
    spectrum = result.flux_numpy[:, 0, 0, 0]
    return result.k_eff, spectrum / spectrum.sum()


def test_metadata_statistics_and_material_mapping(tmp_path):
    path = _write(tmp_path)
    record = read_serpent_xs(path)
    assert record.n_groups == 2
    assert record.version == "Serpent 2.2.0"
    assert record.universe == "fuel"
    assert record.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    np.testing.assert_allclose(record.energy_bounds_mev, [20, 6.25e-7, 0])
    estimate = record.estimate("INF_RABSXS")
    np.testing.assert_allclose(estimate.mean, [0.01, 0.10])
    np.testing.assert_allclose(estimate.relative_error, [0.1, 0.02])
    np.testing.assert_allclose(estimate.absolute_error, [0.001, 0.002])
    material = record.to_diffusion_material()
    np.testing.assert_allclose(material.diffusion, [1.25, 0.25])
    np.testing.assert_allclose(material.sigma_a, estimate.mean)
    np.testing.assert_allclose(material.sigma_t, [0.5, 1.0])
    np.testing.assert_allclose(material.nu_sigma_f, [0.005, 0.15])
    np.testing.assert_allclose(material.chi, [0.95, 0.05])
    # Source-major order preserves the small but nonzero upscatter entry.
    np.testing.assert_allclose(material.sigma_s, [[0.44, 0.05], [0.001, 0.899]])
    np.testing.assert_allclose(record.to_diffusion_material(diffusion="inf").diffusion,
                               [1.5, 0.3])
    assert isinstance(record.balance_diagnostics(), dict)


def test_numeric_multiline_comments_and_escaped_strings(tmp_path):
    path = _write(tmp_path, overrides={
        "GC_UNIVERSE_NAME": "'fuel''s % universe'",
        "INF_NSF": "[ +5D-3 2d-2 % comment\n .15 3E-2 ]",
    })
    record = read_serpent_xs(path)
    assert record.universe == "fuel's % universe"
    np.testing.assert_allclose(record.estimate("INF_NSF").mean, [0.005, 0.15])


def test_declared_numeric_length_must_match_data(tmp_path):
    text = _block().replace("MACRO_E (idx, [1: 3])", "MACRO_E (idx, [1: 4])")
    with pytest.raises(ValueError, match="MACRO_E"):
        read_serpent_xs(_write(tmp_path, text=text))


def test_production_multiplication_and_informational_balance(tmp_path):
    record = read_serpent_xs(_write(tmp_path, overrides={
        "INF_ABS": "[ 0.02 0.1 0.10 0.02 ]",
        "INF_S0": "[ 0.43 0.01 0.05 0.02 0.001 0.03 0.899 0.04 ]",
    }))
    material = record.to_diffusion_material()
    assert material.sigma_a[0] == pytest.approx(0.01)
    assert material.sigma_s[0, 0] == pytest.approx(0.44)
    assert record.estimate("INF_ABS").mean[0] == pytest.approx(0.02)
    assert record.estimate("INF_S0").mean[0] == pytest.approx(0.43)
    for diagnostic in record.balance_diagnostics().values():
        assert diagnostic["max_abs_cm_inv"] < 1e-14
    # A finite balance discrepancy is reported rather than silently fitted.
    discrepant = read_serpent_xs(_write(tmp_path, overrides={
        "INF_TOT": "[ 0.6 0.01 1.0 0.02 ]",
    }))
    diagnostic = discrepant.balance_diagnostics()[
        "production_total_minus_reduced_absorption_production_out"]
    assert diagnostic["max_abs_cm_inv"] == pytest.approx(0.1)
    assert discrepant.to_diffusion_material().sigma_t[0] == pytest.approx(0.6)


def test_repeated_counter_blocks_require_explicit_selection(tmp_path):
    path = _write(tmp_path, text=_block() + _block({"GC_UNIVERSE_NAME": "'other'"}))
    records = read_serpent_results(path)
    assert len(records) == 2
    with pytest.raises(ValueError):
        read_serpent_xs(path)
    assert read_serpent_xs(path, universe="other").universe == "other"
    assert read_serpent_xs(path, record_index=records[1].record_index).universe == "other"
    repeated = _write(tmp_path, text=_block() + _block())
    with pytest.raises(ValueError):
        read_serpent_xs(repeated, universe="fuel")
    with pytest.raises(ValueError):
        read_serpent_xs(repeated, universe="absent")


@pytest.mark.parametrize("field,value", [
    ("MACRO_NG", None), ("MACRO_NG", "0"), ("MACRO_NG", "2.5"),
    ("MACRO_E", "[ 0 1 20 ]"), ("MACRO_E", "[ 20 0 ]"),
    ("INF_RABSXS", None), ("INF_SP0", "[ 0.4 0.01 ]"),
    ("INF_NSF", "[ 0.1 0.01 0.2 ]"),
    ("INF_RABSXS", "[ -0.01 0.1 0.10 0.02 ]"),
    ("INF_SP0", "[ 0.44 0.01 -0.05 0.02 0.001 0.03 0.899 0.04 ]"),
    ("INF_NSF", "[ NaN 0.1 0.15 0.02 ]"),
    ("INF_TOT", "[ Inf 0.1 1.0 0.02 ]"),
    ("CMM_DIFFCOEF", "[ 0 0.01 0.25 0.02 ]"),
    ("CMM_DIFFCOEF", "[ -1 0.01 0.25 0.02 ]"),
    ("CMM_DIFFCOEF", "[ 1e999 0.01 0.25 0.02 ]"),
    ("INF_RABSXS", "[ 0.01 -0.1 0.10 0.02 ]"),
    ("INF_RABSXS", "[ 0.01 NaN 0.10 0.02 ]"),
])
def test_malformed_required_data_rejected(tmp_path, field, value):
    with pytest.raises(ValueError):
        read_serpent_xs(_write(tmp_path, overrides={field: value})).to_diffusion_material()


def test_no_silent_diffusion_fallback(tmp_path):
    record = read_serpent_xs(_write(tmp_path, overrides={"CMM_DIFFCOEF": None}))
    with pytest.raises(ValueError):
        record.to_diffusion_material()
    np.testing.assert_allclose(record.to_diffusion_material(diffusion="inf").diffusion,
                               [1.5, 0.3])
    with pytest.raises(ValueError):
        record.to_diffusion_material(diffusion="unknown")


def test_chi_roundoff_normalized_without_mutating_estimates(tmp_path):
    record = read_serpent_xs(_write(tmp_path, overrides={
        "INF_CHIT": "[ 0.95 0.01 0.050005 0.02 ]",
    }))
    original = record.estimate("INF_CHIT").mean.copy()
    material = record.to_diffusion_material()
    assert material.chi.sum() == pytest.approx(1.0, abs=1e-15)
    np.testing.assert_allclose(material.chi, original / original.sum())
    np.testing.assert_array_equal(record.estimate("INF_CHIT").mean, original)


@pytest.mark.parametrize("chi", ["[ .95 0 .051 0 ]", "[ 0 0 0 0 ]",
                                  "[ 1.01 0 -.01 0 ]"])
def test_invalid_chi_rejected(tmp_path, chi):
    with pytest.raises(ValueError):
        read_serpent_xs(_write(tmp_path, overrides={"INF_CHIT": chi})).to_diffusion_material()


def test_required_expressions_fail_and_payloads_never_execute(tmp_path):
    sentinel = tmp_path / "executed"
    payload = f"__import__('pathlib').Path('{sentinel}').touch()"
    path = _write(tmp_path, overrides={"INF_NSF": payload})
    with pytest.raises(ValueError):
        read_serpent_xs(path).to_diffusion_material()
    assert not sentinel.exists()
    path = _write(tmp_path, text=_block() + f"UNKNOWN (idx, 1) = {payload};\n")
    try:
        read_serpent_results(path)
    except ValueError:
        pass  # Rejecting unsupported unknown syntax is also safe.
    assert not sentinel.exists()


def test_dense_production_and_reflective_diffusion_invariance(tmp_path):
    material = read_serpent_xs(_write(tmp_path)).to_diffusion_material()
    expected_k, expected_flux = _dense(material)
    # This checks the fission outer product as well as scatter orientation.
    scatter = material.sigma_s.copy()
    np.fill_diagonal(scatter, 0)
    loss = np.diag(material.removal) - scatter.T
    np.testing.assert_allclose(loss @ expected_flux,
                               material.chi * np.dot(material.nu_sigma_f, expected_flux)
                               / expected_k, atol=1e-14)
    for size, scale in [(1.0, 1.0), (0.01, 10.0), (100.0, 0.1)]:
        material.diffusion *= scale
        k, flux = _reflective(material, size)
        assert k == pytest.approx(expected_k, abs=1e-8)
        np.testing.assert_allclose(flux, expected_flux, atol=1e-8, rtol=0)


@pytest.mark.parametrize("groups", [2, 4, 7, 14, 23])
def test_local_point_reactor_results(groups):
    path = (Path(__file__).resolve().parents[2] / "examples" / "serpent-xs"
            / "point-reactor" / f"i2_cro_e71_{groups}g_res.m")
    if not path.exists():
        pytest.skip(f"optional local Serpent result absent: {path.name}")
    record = read_serpent_xs(path)
    assert record.n_groups == groups
    material = record.to_diffusion_material()
    dense_k, dense_flux = _dense(material)
    k, flux = _reflective(material)
    assert k == pytest.approx(dense_k, abs=1e-8)
    np.testing.assert_allclose(flux, dense_flux, atol=1e-8, rtol=0)
    serpent_k = record.estimate("IMP_KEFF")
    reference_k = float(np.ravel(serpent_k.mean)[0])
    stderr = float(np.ravel(serpent_k.absolute_error)[0])
    assert abs(k - reference_k) <= max(1e-5, 3 * stderr)
    reference_flux = record.estimate("INF_FLX").mean
    reference_flux = reference_flux / reference_flux.sum()
    assert np.max(np.abs(flux - reference_flux)) <= 1e-4
    assert np.linalg.norm(flux - reference_flux) / np.linalg.norm(reference_flux) <= 5e-4
