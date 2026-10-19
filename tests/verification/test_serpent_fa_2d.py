"""Assembly geometry and tally conventions, independent of local Serpent files."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    'serpent_fa_2d', Path(__file__).resolve().parents[2]/'examples/serpent-xs/fa-2d/run.py')
fa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fa)


def test_exact_nonuniform_rectangles_and_refined_region_areas(tmp_path):
    widths = np.full(17, 1.2618)
    widths[[0, -1]] = 1.30215
    edges = np.r_[0, np.cumsum(widths)]
    path = tmp_path/'geometry'
    lines = [f'surf {1001+17*r+c} rect {edges[c]} {edges[c+1]} {edges[r]} {edges[r+1]} % pin'
             for r in range(17) for c in range(17)]
    path.write_text('\n'.join(lines))
    x, y = fa.rectangles(path)
    mesh, region = fa.make_mesh(x, y, 2)
    areas = np.bincount(region, weights=mesh.area)
    np.testing.assert_allclose(areas, np.outer(widths, widths).ravel(), rtol=1e-12)
    assert mesh.n_cells == 1156
    assert mesh.area.sum() == pytest.approx(21.5313**2)
    assert mesh.cell_tag[0] == 0 and mesh.cell_tag[-1] == 288
    # Reordering labels must not silently rotate/transpose the input map.
    lines[0], lines[1] = lines[0].replace('1001', '1002'), lines[1].replace('1002', '1001')
    path.write_text('\n'.join(lines))
    with pytest.raises(ValueError, match='increase with x'):
        fa.rectangles(path)


def test_volume_integrals_and_power_use_energy_not_neutron_yield():
    flux = np.array([[1., 3., 2.], [2., 1., 4.]])
    area = np.array([2., 1., 3.])
    region = np.array([0, 0, 1])
    integrals = fa.region_integrals(flux, area, region)
    np.testing.assert_array_equal(integrals[:2], [[5, 5], [6, 12]])
    energy = np.zeros((289, 2))
    energy[:2] = [[10, 20], [30, 40]]
    spectrum, power = fa.edits(integrals, energy)
    np.testing.assert_allclose(spectrum, [11/28, 17/28])
    np.testing.assert_allclose(power[:2], [150/810, 660/810])
    assert power[2:].sum() == 0


def test_independent_balance_has_zero_reflective_leakage_and_reciprocal_faces():
    from ndgpu import Material
    # make_mesh uses 17x17 mapping; this helper checks a rectangular two-cell
    # mesh directly, isolating nonuniform interface coupling from parsing.
    from ndgpu.mesh import assemble_mesh
    mesh = assemble_mesh(np.array([[0,0],[1,0],[3,0],[0,1],[1,1],[3,1]]),
                         [(0,1,4,3),(1,2,5,4)], [0,1])
    mats = [Material([d], [0.1], [0.2]) for d in (1., 4.)]
    loss, production = fa.balance_matrix(mesh, mats, np.array([0,1]))
    np.testing.assert_allclose(loss.toarray(), [[0.1+4/3, -4/3], [-4/3, 0.2+4/3]])
    np.testing.assert_allclose(loss.toarray().sum(axis=1), mesh.area*0.1)
    np.testing.assert_allclose(production.diagonal(), mesh.area*0.2)
