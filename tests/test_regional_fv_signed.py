"""Signed regional diagonals cancel in existing corrected FV operators."""
from dataclasses import replace

import numpy as np
import pytest

from garnet import Material, MeshDiffusionEigenSolver, MeshSP3EigenSolver, MeshSDP1EigenSolver
from garnet.grid import Grid
from garnet.mesh import assemble_mesh
from garnet.regional_xs import RegionDataset
from garnet.sn import SNTransportSolver
from garnet.tri import TriGrid
from garnet.tri_sn import TriSNTransportSolver
from garnet.hybrid_sn import HybridSNDiffusionSolver
from garnet.hybrid_tri_sn import HybridTriSNDiffusionSolver
from test_regional_xs import adapt, paired, record


def signed_region(residual=0., scale=1.):
    native = record()
    for field in ("INF_TOT", "INF_RABSXS", "INF_ABS", "INF_S0", "INF_SP0", "INF_NSF", "INF_FISS"):
        native.fields[field][::2] *= scale
    # One negative and one positive corrected diagonal; off-diagonals nonzero.
    native.fields["TRC_DIFFCOEF"] = paired(1. / (3. * np.array([.1, .8]) * scale))
    native.fields["INF_TOT"][0] += residual
    return adapt(native, balance_atol=.03)


def test_signed_manifest_roundtrip_and_strict_legacy_conversion(tmp_path):
    region = signed_region()
    assert region.production[0, 0] < 0 < region.production[1, 1]
    path = tmp_path / "signed.json"
    RegionDataset((region,)).write_manifest(path)
    restored = RegionDataset.read_manifest(path).regions[0]
    np.testing.assert_array_equal(restored.production, region.production)
    np.testing.assert_array_equal(restored.raw_means["INF_SP0"], region.raw_means["INF_SP0"])
    np.testing.assert_array_equal(restored.provenance["balance_residual"], region.provenance["balance_residual"])
    with pytest.raises(ValueError, match="non-negative"):
        restored.as_material()
    ordinary = adapt().as_material()
    assert not ordinary.fv_only
    np.testing.assert_array_equal(ordinary.sigma_s, adapt().production)


@pytest.mark.parametrize("field, value", [
    ("mode", "physical-isotropic"),
    ("production", np.array([[-.1, -.02], [.05, .55]])),
    ("diffusion", np.array([-.1, .5])),
    ("total", np.array([0., .8])),
])
def test_signed_exception_does_not_relax_other_validation(field, value):
    with pytest.raises(ValueError):
        replace(signed_region(), **{field: value})


@pytest.mark.parametrize("mode", ["transport-corrected", "physical-isotropic"])
@pytest.mark.parametrize("entry", [0, 1])
def test_native_production_stays_nonnegative(mode, entry):
    native = record()
    native.fields["INF_SP0"][2*entry] = -.01
    with pytest.raises(ValueError, match="INF_SP0"):
        adapt(native, mode=mode)


def test_signed_diagonal_does_not_bypass_raw_balance_gate():
    native = record()
    native.fields["TRC_DIFFCOEF"] = paired([10., 10.])
    native.fields["INF_TOT"][0] += .2
    with pytest.raises(ValueError, match="scalar production balance"):
        adapt(native)


def test_fv_conversion_eliminates_both_diagonal_signs_and_copies():
    region = signed_region()
    material = region.as_fv_material()
    assert adapt().as_fv_material().fv_only
    np.testing.assert_array_equal(np.diag(adapt().as_fv_material().sigma_s), [0., 0.])
    assert material.fv_only
    expected = region.production.copy()
    np.fill_diagonal(expected, 0.)
    np.testing.assert_array_equal(material.sigma_s, expected)
    for field, expected in (("total", region.total), ("diffusion", region.diffusion),
                            ("sigma_a", region.reduced_absorption)):
        np.testing.assert_array_equal(getattr(material, field), expected)
    copied = material.replace(name="independent FV copy")
    assert copied.fv_only
    copied.sigma_s[0, 1] = 99.
    assert material.sigma_s[0, 1] == region.production[0, 1]
    material.total[0] = 99.
    assert region.total[0] != 99.
    with pytest.raises(ValueError, match="non-negative"):
        Material([1.], [.1], [.02], [[-.1]])


@pytest.mark.parametrize("solver, grid", [
    (SNTransportSolver, Grid((2, 2, 1), (1., 1., 1.))),
    (TriSNTransportSolver, TriGrid((2, 2, 2), 1.)),
    (HybridSNDiffusionSolver, Grid((2, 2, 1), (1., 1., 1.))),
    (HybridTriSNDiffusionSolver, TriGrid((2, 2, 2), 1.)),
])
@pytest.mark.parametrize("as_list", [False, True])
def test_sn_rejects_fv_only_before_reconstructing_self_scatter(solver, grid, as_list):
    material = signed_region().as_fv_material().replace(name="copied")
    with pytest.raises(ValueError, match="FV-only.*SN"):
        solver(grid, [material] if as_list else material)


def square_mesh():
    coords = [(x + .25*y, y) for y in (0., .4, 1.) for x in (0., .3, 1.)]
    cells = [(j*3+i, j*3+i+1, (j+1)*3+i+1, (j+1)*3+i) for j in range(2) for i in range(2)]
    return assemble_mesh(coords, cells, [0]*4)


def full_loss_matrix(solver):
    """Probe the coupled FV loss operator including off-diagonal in-scatter."""
    template = solver._initial_state()
    n = template[0].size
    matrix = np.empty((2*n, 2*n))
    for col in range(2*n):
        state = [np.zeros_like(u) for u in template]
        state[col // n].flat[col % n] = 1.
        rows = []
        for g in range(2):
            other = 1-g
            q = solver.sigma_s[other][g] * solver._phi(state[other])
            if solver._src_weight is not None:
                q = q * solver._src_weight
            rows.append((solver.ops[g].apply(state[g]) - solver._rhs(g, q)).ravel())
        matrix[:, col] = np.concatenate(rows)
    return matrix


@pytest.mark.parametrize("solver_cls", [MeshDiffusionEigenSolver, MeshSP3EigenSolver, MeshSDP1EigenSolver])
@pytest.mark.parametrize("residual", [0., .02])
@pytest.mark.parametrize("heterogeneous", [False, True])
def test_corrected_fv_full_operator_invariant_under_signed_diagonal_elimination(solver_cls, residual, heterogeneous):
    regions = [signed_region(residual)]
    if heterogeneous:
        regions.append(signed_region(residual / 2., scale=1.6))
        for field in ("diffusion", "total", "reduced_absorption", "production"):
            assert not np.array_equal(getattr(regions[0], field), getattr(regions[1], field))
    materials, references = [], []
    for index, region in enumerate(regions):
        expected_residual = residual / (index + 1)
        effective_loss = np.diag(region.total) - region.production.T
        native = region.raw_means
        np.testing.assert_allclose(effective_loss,
            np.diag(native["INF_TOT"]) - native["INF_SP0"].reshape(2, 2).T)
        np.testing.assert_allclose(region.provenance["balance_residual"],
                                   [expected_residual, 0.], atol=1e-15)
        material = region.as_fv_material()
        # Mathematical reference only: strict Material construction remains intact.
        # Insert each signed native-derived array after validation to probe the
        # pre-elimination FV algebra, including both sides of the interface.
        reference = material.replace()
        reference.sigma_s[:] = region.production
        materials.append(material)
        references.append(reference)
        # FV uses RABS + off-diagonal row sums; the admitted raw balance
        # residual is retained, not repaired by diagonal elimination.
        fv_loss = np.diag(material.removal) - material.sigma_s.T
        np.testing.assert_allclose(effective_loss - fv_loss,
                                   np.diag([expected_residual, 0.]), atol=1e-15)
        assert np.all(region.diffusion > 0) and np.all(region.total > 0)
        assert region.production[0, 0] < 0 < region.production[1, 1]
        assert region.production[0, 1] > 0 and region.production[1, 0] > 0
    material_map = [0, 1, 0, 1] if heterogeneous else [0]*4
    kw = dict(device="cpu", boundary_conditions="vacuum", mesh_discretization="corrected")
    mesh = square_mesh()
    if heterogeneous:
        assert any(material_map[i] != material_map[j] for i, j, _, _ in mesh.faces)
    before = solver_cls(mesh, references, material_map, **kw)
    after = solver_cls(mesh, materials, material_map, **kw)
    a, b = full_loss_matrix(before), full_loss_matrix(after)
    np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-14)
    assert np.any(a[:a.shape[0]//2, a.shape[1]//2:] != 0)
