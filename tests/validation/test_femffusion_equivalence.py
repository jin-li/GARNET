"""GARNET's IAEA-3D and Langenbuch builders pose FEMFFUSION's own inputs.

The GARNET vs FEMFFUSION performance report runs FEMFFUSION on its example
inputs (examples/3D_IAEA, examples/3D_Langenbuch) and GARNET on its builders,
so the two must describe the same problem: identical cross sections, the same
material in every node, the same holes and the same boundary condition. The
FEMFFUSION node map is rebuilt from the .prm's Geometry_Points and the .xsec's
PLANE blocks (row 1 = lowest y, x fastest); GARNET's first array axis is
FEMFFUSION's y, so the GARNET map is the FEMFFUSION map transposed in plan -- a
mirror image across the diagonal, which leaves k unchanged and maps node
powers by the same transpose.
"""

import re

import numpy as np
import pytest

from garnet.benchmarks.iaea import ALBEDO, build_iaea
from garnet.benchmarks.langenbuch import build_langenbuch, rod_tip
from garnet.femffusion import read_xsec
from tests._femffusion import femffusion_path, requires_femffusion

IAEA = ("examples/3D_IAEA/IAEA_FE3.prm", "examples/3D_IAEA/IAEA.xsec")
LMW = ("examples/3D_Langenbuch/3D_Langenbuch_ds_fe2.prm",
       "examples/3D_Langenbuch/3D_Langenbuch.xsec",
       "examples/3D_Langenbuch/3D_Langenbuch.bar")


def _prm(path, key):
    text = open(path).read()
    return re.search(rf"^\s*set\s+{key}\s*=\s*(.+?)\s*$", text, re.M).group(1)


def _femffusion_map(prm, xsec):
    """Material id per node, [x, y, z], 0 = hole, from Geometry_Points + PLANE maps."""
    pts = [int(v) for v in _prm(prm, "Geometry_Points").split()]
    rows = list(zip(pts[0::2], pts[1::2]))
    nx = max(hi for _, hi in rows)
    xs = read_xsec(xsec)
    planes = []
    for plane in xs.core_map:
        a = np.zeros((nx, len(rows)), dtype=np.int64)
        for j, ((lo, hi), ids) in enumerate(zip(rows, plane)):
            assert len(ids) == hi - lo + 1
            a[lo - 1:hi, j] = ids
        planes.append(a)
    return xs, np.stack(planes, axis=2)


def _same_xs(garnet_mats, femffusion_mats):
    assert len(garnet_mats) == len(femffusion_mats)
    for g, f in zip(garnet_mats, femffusion_mats):
        np.testing.assert_allclose(g.diffusion, f.diffusion, rtol=1e-12)
        np.testing.assert_allclose(g.sigma_a, f.sigma_a, rtol=1e-12)
        np.testing.assert_allclose(g.nu_sigma_f, f.nu_sigma_f, rtol=1e-12)
        np.testing.assert_allclose(g.sigma_s, f.sigma_s, rtol=1e-12)


@requires_femffusion(*IAEA)
def test_iaea_builder_is_femffusions_input():
    prm, xsec = (femffusion_path(p) for p in IAEA)
    xs, fmap = _femffusion_map(prm, xsec)
    p = build_iaea(1)
    _same_xs(p.materials[1:], xs.materials)
    assert np.array_equal(p.material_map, fmap.transpose(1, 0, 2))
    assert np.array_equal(p.active, fmap.transpose(1, 0, 2) > 0)
    # albedo (BC code 3) on all six faces, factor shared by both groups
    assert _prm(prm, "Boundary_Conditions").split() == ["3"] * 6
    assert [float(a) for a in _prm(prm, "Albedo_Factors").split()] == [ALBEDO, ALBEDO]
    assert p.bc == p.mask_bc == ALBEDO


def _apply_bars_static(fmap, bar_file, pitch_z=20.0):
    """FEMFFUSION's move_bars_static at t = 0: fully covered nodes take the bar material."""
    lines = [ln.split("#")[0].split() for ln in open(bar_file)]
    lines = [ln for ln in lines if ln]
    at = {ln[0]: n for n, ln in enumerate(lines)}
    ny = fmap.shape[1]
    conf = lines[at["Bank_Configuration"] + 1: at["Bank_Configuration"] + 1 + ny]
    top = float(lines[at["BarsTopPosition"]][1])
    banks = {}
    for ln in lines[at["Move_Banks"] + 1: at["Move_Banks"] + 1 + int(lines[at["Move_Banks"]][1])]:
        banks[int(ln[0])] = (int(ln[1]), float(ln[4]))        # material, h0
    out = fmap.copy()
    for j, row in enumerate(conf):
        xs_present = np.nonzero(fmap[:, j, 0] > 0)[0]
        for x, bank in zip(xs_present, (int(v) for v in row[1:])):
            if bank == 0:
                continue
            mat, h0 = banks[bank]
            for k in range(fmap.shape[2]):
                lo, hi = k * pitch_z, (k + 1) * pitch_z
                covered = max(0.0, min(hi, top) - max(lo, h0)) / pitch_z
                assert covered in (0.0, 1.0), "partly rodded node at t = 0"
                if covered == 1.0:
                    out[x, j, k] = mat
    return out


@requires_femffusion(*LMW)
def test_langenbuch_builder_is_femffusions_input():
    prm, xsec, bar = (femffusion_path(p) for p in LMW)
    xs, fmap = _femffusion_map(prm, xsec)
    fmap = _apply_bars_static(fmap, bar)
    p = build_langenbuch(1, excise_corners=True)
    mats, mmap = p.problem_at(0.0)
    _same_xs(mats[:len(xs.materials)], xs.materials)
    expect = fmap.transpose(1, 0, 2)
    assert np.array_equal(p.active, expect > 0)
    assert np.array_equal(np.where(p.active, mmap + 1, 0), expect)
    # bank 1 starts half inserted, bank 2 withdrawn, as the .bar file says
    assert rod_tip(1, 0.0) == 100.0 and rod_tip(2, 0.0) == 180.0
    assert _prm(prm, "Boundary_Conditions").split() == ["0"] * 6
    assert p.bc == p.mask_bc == "zero-flux"


def test_langenbuch_default_keeps_the_corner_reflector():
    p = build_langenbuch(1)
    assert p.active is None
    q = build_langenbuch(2, excise_corners=True)
    assert int((~q.active).sum()) == 4 * 2 * 2 * 20
