"""C5G7 colorset ADF workflow: library generation and reference equivalence."""

import numpy as np
import pytest

import garnet.benchmarks.c5g7 as c5g7
from examples.c5g7_colorset_adf import (COVERAGE_SAMPLES, LAYOUTS, assembly_library, node_map,
                                        solve_coarse, solve_fine, study)


@pytest.fixture(scope="module")
def library():
    return assembly_library(2, "cpu")


def test_isolated_assembly_homogenization_preserves_k_inf(library):
    for kind, entry in library.items():
        assemblies = node_map([kind], 1, 3)
        _, result = solve_coarse([kind], 3, [entry["material"]], assemblies, assemblies,
                                 None, "cpu")
        assert result.k_eff == pytest.approx(entry["k_inf"], abs=1e-8)


def test_library_factors_are_symmetric_and_physical(library):
    for entry in library.values():
        adf = entry["adf"]
        assert np.allclose(adf[:, :4], adf[:, :1], rtol=1e-10)
        assert np.all(adf[:, 4:] == 1)
    # Thermal group: the UO2 interior holds the water-filled guide tubes, so
    # its surface flux is below average; the MOX edge pins have the lowest
    # enrichment and absorb less, so its surface flux is above average.
    assert library["U"]["adf"][-1, 0] < 1 < library["M"]["adf"][-1, 0]


def test_reference_factors_reproduce_reference_and_report_inadmissible_nodes(library):
    res = study("checker2", 2, [1, 17], "cpu", library)
    rows = {(r["case"], r["nodes"]): r for r in res["rows"]}
    # Two-point traces on one 21.42 cm node per assembly turn negative in the
    # thermal groups, so no positive reference factor exists there.
    coarse = rows[("reference XS + DF", 1)]
    assert coarse["admissible"] is False and coarse["min_trace_ratio"] < 0
    exact = rows[("reference XS + DF", 17)]
    assert exact["min_trace_ratio"] > 0
    assert abs(exact["dpcm"]) < 1e-3
    assert exact["power_max_pct"] < 1e-4
    assert exact["current_max_rel"] < 1e-6
    # Without the factors the same reference cross sections are not exact.
    assert abs(rows[("reference XS, no DF", 17)]["dpcm"]) > 10
    assert len(res["reference_power"]) == len("".join(LAYOUTS["checker2"]))


@pytest.mark.parametrize("kind", ["U", "M"])
def test_one_cell_pin_blend_equals_pin_cell_homogenization(kind, monkeypatch):
    # At one cell per pin the pin-resolved raster is a single blended cell:
    # linear cross sections and harmonic D at the sampled fuel fraction are
    # exactly the pin-cell homogenization at that fraction.
    monkeypatch.setattr(c5g7, "FUEL_FRACTION",
                        c5g7.pin_coverage(1, sub=COVERAGE_SAMPLES).mean())
    homogenized = solve_fine([kind], 1, "cpu")["result"].k_eff
    blended = solve_fine([kind], 1, "cpu", pin_resolved=True)["result"].k_eff
    assert blended == pytest.approx(homogenized, abs=1e-11)


@pytest.fixture(scope="module")
def pin_resolved_library():
    return assembly_library(4, "cpu", pin_resolved=True)


def test_pin_resolved_homogenization_preserves_k_inf(pin_resolved_library):
    for kind, entry in pin_resolved_library.items():
        assemblies = node_map([kind], 1, 3)
        _, result = solve_coarse([kind], 3, [entry["material"]], assemblies, assemblies,
                                 None, "cpu")
        assert result.k_eff == pytest.approx(entry["k_inf"], abs=1e-8)
    adf = pin_resolved_library["U"]["adf"]
    assert np.allclose(adf[:, :4], adf[:, :1], rtol=1e-10)


def test_pin_resolved_reference_factors_reproduce_reference(pin_resolved_library):
    # Blended cells must be homogenized into their own node: attributing the
    # fuel fraction to a region numbered after its material breaks exactness.
    res = study("checker2", 4, [17], "cpu", pin_resolved_library, pin_resolved=True)
    assert res["pin_resolved"]
    rows = {r["case"]: r for r in res["rows"]}
    exact = rows["reference XS + DF"]
    assert exact["min_trace_ratio"] > 0
    assert abs(exact["dpcm"]) < 1e-3
    assert exact["power_max_pct"] < 1e-4
    assert exact["current_max_rel"] < 1e-6
    # Pin-cell homogenization is now a real approximation.
    assert abs(rows["reference XS, no DF"]["dpcm"]) > 10
