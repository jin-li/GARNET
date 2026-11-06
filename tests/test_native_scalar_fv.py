"""Independent tiny synthetic checks; no physical qualification implied."""
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest

from garnet.mesh import assemble_mesh
from garnet.native_scalar_fv import NativeScalarFV, RegionalCoefficients, estimate_memory


def mesh_grid(n=2, shear=0., unequal=False):
    xs = np.linspace(0, 1, n+1)
    if unequal:
        xs = xs**1.4
    coords = np.array([(x+shear*y, y) for y in np.linspace(0, 1, n+1) for x in xs])
    cells = [(j*(n+1)+i, j*(n+1)+i+1, (j+1)*(n+1)+i+1, (j+1)*(n+1)+i)
             for j in range(n) for i in range(n)]
    return assemble_mesh(coords, cells, np.zeros(n*n, dtype=int))


def coefficients(regions=1):
    return RegionalCoefficients(
        region_ids=tuple(19+7*i for i in range(regions)), energy_bounds=[10., 1., 0.],
        D=np.array([[.7, 1.1]])*np.arange(1, regions+1)[:, None],
        T=np.tile([2.4, 1.3], (regions, 1)),
        SP0=np.tile([[.4, .1], [.2, .3]], (regions, 1, 1)),
        F=np.tile([[1., .8], [.3, .6]], (regions, 1, 1)),
        scattering_orientation="from,to", fission_orientation="to,from",
        signed_rabs=np.tile([-.2, .7], (regions, 1)))


def pilot(mesh=None, c=None, index=None, **kw):
    mesh = mesh_grid() if mesh is None else mesh
    c = coefficients() if c is None else c
    opts = dict(method="tpfa", preconditioner="ilu", max_bytes=100_000_000,
                restart=16, max_inner_iterations=100, max_outer_iterations=200,
                batch_cells=2)
    opts.update(kw)
    return NativeScalarFV(mesh, c, np.zeros(mesh.n_cells, dtype=int) if index is None else index, **opts)


def dense_oracle(mesh, c, index):
    """Assemble algebra directly, independent of both pilot actions/stencils."""
    n, g = mesh.n_cells, c.D.shape[1]
    a, f = np.zeros((n*g, n*g)), np.zeros((n*g, n*g))
    for i, r in enumerate(index):
        sl = slice(i*g, (i+1)*g)
        a[sl, sl] = mesh.area[i]*(np.diag(c.T[r])-c.SP0[r].T)
        f[sl, sl] = mesh.area[i]*c.F[r]
    for face, distances in zip(mesh.faces, mesh.face_distances):
        i, j, measure, _ = face
        for group in range(g):
            w = measure/(distances[0]/c.D[index[i], group]+distances[1]/c.D[index[j], group])
            p, q = i*g+group, j*g+group
            a[p, p] += w
            a[q, q] += w
            a[p, q] -= w
            a[q, p] -= w
    return a, f


@pytest.mark.parametrize("method", ["tpfa", "corrected"])
@pytest.mark.parametrize("preconditioner", ["ilu", "jacobi"])
def test_exact_homogeneous_reflective_full_fission(method, preconditioner):
    c = coefficients()
    assert np.linalg.det(c.F[0]) != 0
    op = pilot(c=c, method=method, preconditioner=preconditioner)
    result = op.solve()
    assert result.converged, (result.reason, result.history)
    assert abs(result.k-1.) < 1e-6
    np.testing.assert_allclose(result.flux/result.flux[0, 0], 1., atol=1e-6)
    assert result.history[-1]["inner_residual"] <= 1e-7
    assert result.history[-1]["full_residual"] <= 1e-5
    assert result.history[-1]["k_change_absolute"] <= 1e-6
    assert not result.diagnostics["physical_qualification"]
    assert result.heat is None


def test_heat_is_distinct_from_neutron_production():
    op = pilot(c=replace(coefficients(), kappa_fission=[[7., 13.]]))
    result = op.solve()
    assert result.converged
    expected = op.volumes*np.sum(result.flux*[7., 13.], axis=1)
    np.testing.assert_allclose(result.heat, expected)
    assert not np.allclose(result.heat, op.apply_F(result.flux).sum(axis=1))


def test_independent_dense_actions_and_eigenpair_shuffled_unequal():
    mesh, c = mesh_grid(unequal=True), coefficients(2)
    c = replace(c, T=[[2.4, 1.3], [2.8, 1.5]],
                SP0=[[[.4, .1], [.2, .3]], [[.7, .23], [.41, .2]]],
                F=[[[1., .8], [.3, .6]], [[.8, .5], [.1, .4]]])
    index = np.array([1, 0, 0, 1])
    op = pilot(mesh, c, index)
    a, f = dense_oracle(mesh, c, index)
    x = np.arange(1., 9.).reshape(4, 2)
    np.testing.assert_allclose(op.apply_A(x).ravel(), a@x.ravel(), atol=2e-14)
    np.testing.assert_allclose(op.apply_F(x).ravel(), f@x.ravel(), atol=2e-14)
    values, vectors = np.linalg.eig(np.linalg.solve(a, f))
    selected = np.argmax(values.real)
    expected = vectors[:, selected].real
    expected /= (f@expected).sum()
    result = op.solve()
    assert result.converged, result.reason
    np.testing.assert_allclose(result.k, values[selected].real, rtol=2e-6)
    np.testing.assert_allclose(result.flux.ravel(), expected, rtol=3e-5)
    for previous, current in zip(result.history, result.history[1:]):
        assert current["k_change_absolute"] == abs(current["k"]-previous["k"])


def test_native_imbalance_signed_removal_rabs_and_immutability():
    c = replace(coefficients(), T=[[.2, 1.3]], signed_rabs=[[-71., 9.]])
    saved = [a.copy() for a in (c.T, c.SP0, c.F, c.signed_rabs)]
    index = np.zeros(4, dtype=int)
    op = pilot(c=c, index=index, preconditioner="jacobi")
    index[:] = 99
    x = np.ones((4, 2))
    np.testing.assert_allclose(op.apply_A(x), op.volumes[:, None]*(c.T[0]-c.SP0[0].sum(axis=0)))
    for actual, expected in zip((c.T, c.SP0, c.F, c.signed_rabs), saved):
        np.testing.assert_array_equal(actual, expected)
        with pytest.raises(ValueError):
            actual.setflags(write=True)
    with pytest.raises(ValueError):
        op.cell_region_index[0] = 2
    with pytest.raises(FrozenInstanceError):
        c.T = np.ones((1, 2))
    other = pilot(c=replace(c, signed_rabs=[[5., -3.]]), preconditioner="jacobi")
    np.testing.assert_array_equal(op.apply_A(x), other.apply_A(x))


@pytest.mark.parametrize("method", ["tpfa", "corrected"])
def test_conservative_interface_constant_and_orthogonal_agreement(method):
    mesh, c = mesh_grid(3), coefficients(2)
    index = np.arange(9) % 2
    op = pilot(mesh, c, index, method=method)
    ref = pilot(mesh, c, index, method="tpfa")
    phi = np.linspace(.3, 2., 18).reshape(9, 2)
    np.testing.assert_allclose(op.apply_A(phi), ref.apply_A(phi), atol=2e-13)
    for group, block in enumerate(op.groups):
        removal = c.T[index, group]-c.SP0[index, group, group]
        constant = block.apply(np.ones(9)).ravel()-mesh.area*removal
        np.testing.assert_allclose(constant, 0., atol=2e-14)
        leakage = block.apply(phi[:, group]).ravel()-mesh.area*removal*phi[:, group]
        assert abs(leakage.sum()) < 2e-13


def test_skew_discontinuous_interface_current_oracle():
    mesh = mesh_grid(6, shear=.35)
    normal = np.array([1., -.35])/np.hypot(1., .35)
    tangent = np.array([.35, 1.])/np.hypot(1., .35)
    coordinate = mesh.centroid@normal
    interface = .5/np.hypot(1., .35)
    index = (coordinate >= interface).astype(int)
    c = replace(coefficients(2), D=[[.6, .6], [2.7, 2.7]])
    op = pilot(mesh, c, index, method="corrected")
    # Existing corrected MMS hook tests face law independently of the patch's
    # reconstruction. Build supplied exact gradients for this fixed linear u.
    from garnet.corrected_mesh import CorrectedMeshGroupOperator
    D = c.D[index, 0]
    u = 3.+1.2*(coordinate-interface)/D+.3*(mesh.centroid@tangent)
    gradients = 1.2/D[:, None]*normal+.3*tangent
    block = CorrectedMeshGroupOperator(np, op.grid, D, np.zeros(36),
                                      reconstruction_policy="supplied",
                                      gradient_stencils=[{0: grad/u[0]} for grad in gradients])
    currents, _ = block.face_currents(u)
    checked = 0
    for face_id, (i, j, area, _) in enumerate(mesh.faces):
        if index[i] != index[j]:
            np.testing.assert_allclose(currents[face_id], -area*1.2*np.dot(normal, mesh.face_normals[face_id]), atol=3e-14)
            checked += 1
    assert checked == 6


@pytest.mark.parametrize("index", [[0., 0., 0., 0.], [-1, 0, 0, 0], [1, 0, 0, 0], [[0, 0], [0, 0]], [True]*4])
def test_bad_maps(index):
    with pytest.raises(ValueError, match="cell_region_index"):
        pilot(index=index)


@pytest.mark.parametrize("changes", [dict(region_ids=(1, 1)), dict(scattering_orientation="to,from"),
    dict(fission_orientation="from,to"), dict(energy_bounds=[0., 1., 10.]),
    dict(energy_bounds=[10., 1., 1.]), dict(D=[[0., 1.]]), dict(T=[[0., 1.]]),
    dict(SP0=[[[-.1, 0.], [0., 0.]]]), dict(F=[[[np.nan, 0.], [0., 0.]]])])
def test_bad_coefficients(changes):
    with pytest.raises(ValueError):
        replace(coefficients(), **changes)


@pytest.mark.parametrize("name", ["D", "T", "SP0", "F", "energy_bounds", "signed_rabs", "kappa_fission"])
@pytest.mark.parametrize("kind", ["bool", "complex", "string", "object"])
def test_coefficients_require_real_numeric_input(name, kind):
    c = coefficients()
    original = np.ones_like(c.T) if name == "kappa_fission" else getattr(c, name)
    if kind == "bool":
        invalid = np.ones(original.shape, dtype=bool)
    elif kind == "complex":
        invalid = original.astype(complex)+1j
    elif kind == "string":
        invalid = original.astype(str)
    else:
        invalid = original.astype(object)
    with pytest.raises(ValueError, match="real numeric"):
        replace(c, **{name: invalid})


@pytest.mark.parametrize("invalid", [np.ones((4, 2), dtype=complex)*(1+2j),
    np.ones((4, 2), dtype=bool), np.full((4, 2), "1"), np.ones((4, 2), dtype=object)])
def test_states_require_real_numeric_input(invalid):
    op = pilot()
    for action in (op.apply_A, op.apply_F, op.apply_heat):
        with pytest.raises(ValueError, match="real numeric"):
            action(invalid)


def test_zero_fission_memory_and_iteration_budgets():
    zero = pilot(c=replace(coefficients(), F=np.zeros((1, 2, 2)))).solve()
    assert not zero.converged and "production" in zero.reason
    with pytest.raises(MemoryError, match="pre-build"):
        pilot(max_bytes=1)
    with pytest.raises(ValueError, match="restart"):
        pilot(restart=0)
    outer = pilot(max_outer_iterations=1).solve()
    assert not outer.converged and "outer iteration budget" in outer.reason
    mesh, c = mesh_grid(), coefficients(2)
    c = replace(c, T=[[2.4, 1.3], [2.8, 1.6]])
    inner = pilot(mesh, c, np.array([1, 0, 1, 0]), preconditioner="jacobi",
                  max_inner_iterations=1, restart=1).solve()
    assert not inner.converged and "inner residual" in inner.reason
    assert inner.history[-1]["inner_iterations"] <= 1


def test_nonfinite_initial_normalization_is_terminal():
    op = pilot(c=replace(coefficients(), F=np.full((1, 2, 2), 1e-320)))
    with np.errstate(over="ignore"):
        result = op.solve()
    assert not result.converged
    assert result.reason == "nonfinite initial normalized flux/source"
    assert result.history[-1]["failure"] == result.reason
    assert np.all(np.isfinite(result.flux))
    assert result.heat is None
    assert result.k is None
    assert result.diagnostics["returned_state"] == "finite_seed_no_eigenvalue"


def test_singular_reflective_loss_and_negative_flux_fail_closed():
    c = replace(coefficients(), T=[[.4, .3]], SP0=[[[.4, 0.], [0., .3]]])
    singular = pilot(c=c, preconditioner="jacobi").solve()
    assert not singular.converged
    # Invertible local loss maps a positive fission source to a negative
    # component while retaining positive total production.
    negative = replace(coefficients(), T=[[1., 1.]], SP0=[[[2., 0.], [0., 0.]]],
                       F=[[[.1, 0.], [0., 1.]]])
    result = pilot(c=negative, preconditioner="jacobi").solve()
    assert not result.converged and "negative" in result.reason


@pytest.mark.parametrize("scale", [1., 1e15])
def test_scaled_negative_eigenvector_is_rejected(scale):
    c = replace(coefficients(), D=np.array([[.7, 1.1]])*scale,
                T=np.array([[1., 1.]])*scale,
                SP0=np.array([[[2., 0.], [0., 0.]]])*scale,
                F=np.array([[[.1, .1], [.1, 1.]]])*scale,
                kappa_fission=[[1., 0.]])
    result = pilot(c=c, preconditioner="jacobi").solve()
    assert not result.converged and "negative" in result.reason
    stats = result.history[-1]["candidate_flux_positivity"]
    assert stats["minimum_over_scale"] < -.01
    assert stats["negative_fraction"] > .01
    assert not stats["relative_gate_passed"]
    if scale == 1e15:
        assert stats["absolute_gate_passed"]
    heat_stats = result.history[-1]["candidate_heat_positivity"]
    assert not heat_stats["relative_gate_passed"]
    assert result.k is None
    assert np.all(np.isfinite(result.flux))
    assert np.all(np.isfinite(result.heat))
    assert result.diagnostics["returned_state"] == "finite_normalized_seed_no_eigenvalue"


@pytest.mark.parametrize("failure", ["production", "normalization", "positivity"])
def test_rejected_candidate_preserves_last_accepted_flux_and_k(monkeypatch, failure):
    import garnet.native_scalar_fv as native
    c = replace(coefficients(), kappa_fission=[[7., 13.]])
    baseline = pilot(c=c, max_outer_iterations=1).solve()
    op = pilot(c=c)
    real_gmres, real_fission = native.gmres, op.apply_F
    gmres_calls, fission_calls = 0, 0

    def controlled_gmres(A, rhs, **kwargs):
        nonlocal gmres_calls
        gmres_calls += 1
        if gmres_calls == 1:
            return real_gmres(A, rhs, **kwargs)
        if failure == "positivity":
            # Inject an inverse result which passes the measured inner gate;
            # the independent positivity gate must reject before commit.
            op.apply_A = lambda _: rhs.reshape(op.shape).copy()
            return np.tile([-.01, 1.], op.shape[0]), 0
        return real_gmres(A, rhs, **kwargs)

    def controlled_fission(phi):
        nonlocal fission_calls
        fission_calls += 1
        if fission_calls == 3:
            if failure == "production":
                return np.zeros(op.shape)
            if failure == "normalization":
                return np.full(op.shape, 1e-320)
        return real_fission(phi)

    monkeypatch.setattr(native, "gmres", controlled_gmres)
    monkeypatch.setattr(op, "apply_F", controlled_fission)
    with np.errstate(over="ignore"):
        result = op.solve()
    assert not result.converged
    assert result.k == baseline.k
    np.testing.assert_array_equal(result.flux, baseline.flux)
    np.testing.assert_array_equal(result.heat, baseline.heat)
    assert np.all(np.isfinite(result.flux)) and np.all(np.isfinite(result.heat))
    assert result.diagnostics["returned_state"] == "finite_accepted_iterate"
    assert result.diagnostics["returned_k_matches_flux"]
    assert result.history[-1]["candidate_production"] is not None
    assert "candidate_flux_positivity" in result.history[-1]
    assert not result.history[-1].get("accepted", False)
    assert result.history[-1]["failure"] == result.reason


def test_history_storage_is_in_prebuild_memory_admission():
    with pytest.raises(MemoryError, match="pre-build"):
        pilot(max_outer_iterations=10_000_000)


def memory_coefficients(groups):
    return RegionalCoefficients(region_ids=(17,), energy_bounds=np.arange(groups, -1, -1, dtype=float),
        D=np.ones((1, groups)), T=np.ones((1, groups)), SP0=np.zeros((1, groups, groups)),
        F=np.eye(groups)[None], scattering_orientation="from,to", fission_orientation="to,from")


def memory_mesh(nx=18, ny=20):
    """Topology/array inputs only: estimator must never assemble a stencil."""
    faces = []
    for j in range(ny):
        for i in range(nx):
            cell = j*nx+i
            if i+1 < nx:
                faces.append((cell, cell+1, 1., 1.))
            if j+1 < ny:
                faces.append((cell, cell+nx, 1., 1.))
    bfaces = [(j*nx+i, 1., .5) for j in range(ny) for i in range(nx)
              for side in (i == 0, i == nx-1, j == 0, j == ny-1) if side]
    return SimpleNamespace(n_cells=nx*ny, faces=faces, bfaces=bfaces,
        centroid=np.array([(i+.5, j+.5) for j in range(ny) for i in range(nx)]),
        area=np.ones(nx*ny))


def memory_estimate(mesh, groups=23, **kw):
    settings = dict(method="corrected", preconditioner="ilu", restart=20,
                    batch_cells=256, max_outer_iterations=100)
    settings.update(kw)
    return estimate_memory(mesh, memory_coefficients(groups), **settings)


def test_estimator_linear_group_storage_and_single_group_python_peak():
    mesh = memory_mesh(4, 5)
    estimates = [memory_estimate(mesh, groups) for groups in (2, 4, 6)]
    # Regional dense coefficient/batch sizes are quadratic in G by design;
    # the remaining total grows linearly in group count, with one setup peak.
    corrected_totals = [e["estimated_bytes"]-e["shared_input_bytes"]-e["regional_batch_bytes"]
                        for e in estimates]
    assert corrected_totals[1]-corrected_totals[0] == corrected_totals[2]-corrected_totals[1]
    for field in ("one_group_python_construction_bytes", "one_group_factor_workspace_bytes",
                  "one_group_construction_peak_bytes", "per_group_operator_bytes", "per_group_factor_bytes"):
        assert len({e[field] for e in estimates}) == 1
    e = estimates[0]
    assert e["one_group_construction_peak_bytes"] == max(e["one_group_python_construction_bytes"],
                                                       e["one_group_factor_workspace_bytes"])
    assert not e["factor_fill_and_workspace_guaranteed"]


def test_estimator_small_pin_retained_arrays_and_fixed_budget():
    mesh = memory_mesh()
    e = memory_estimate(mesh)
    assert mesh.n_cells == 360
    assert 0 < e["estimated_bytes"] < 4*1024**3
    assert e["per_group_csr_bytes"] > 0
    assert e["per_group_gather_bytes"] > 0
    assert e["per_group_gradient_bytes"] > 0
    assert e["per_group_tpfa_bytes"] > 0
    tpfa = memory_estimate(mesh, method="tpfa")
    assert tpfa["per_group_gradient_bytes"] == tpfa["per_group_gather_bytes"] == 0
    assert tpfa["estimated_bytes"] < e["estimated_bytes"]
    jacobi = memory_estimate(mesh, preconditioner="jacobi")
    assert jacobi["per_group_operator_bytes"] == e["per_group_operator_bytes"]
    assert jacobi["one_group_python_construction_bytes"] == e["one_group_python_construction_bytes"]
    assert jacobi["one_group_factor_workspace_bytes"] == 0
    assert jacobi["estimated_bytes"] < e["estimated_bytes"]


def test_estimator_high_degree_rejects_before_operator_construction(monkeypatch):
    import garnet.native_scalar_fv as native
    n = 2048
    star = SimpleNamespace(n_cells=n, faces=[(0, i, 1., 1.) for i in range(1, n)],
                           bfaces=[], centroid=np.zeros((n, 2)), area=np.ones(n))
    e = memory_estimate(star)
    assert e["maximum_corrected_row_width"] == n
    assert e["estimated_bytes"] > 4*1024**3
    def unexpected(*args, **kwargs):
        pytest.fail("high-degree memory rejection must precede group construction")
    monkeypatch.setattr(native, "MeshGrid", unexpected)
    with pytest.raises(MemoryError, match="pre-build"):
        native.NativeScalarFV(star, memory_coefficients(23), np.zeros(n, dtype=int),
            method="corrected", preconditioner="ilu", max_bytes=4*1024**3,
            restart=20, max_inner_iterations=1000, max_outer_iterations=100,
            batch_cells=256, external_memory_cap=True)


def test_estimator_history_is_additive_and_admission_uses_total(monkeypatch):
    mesh = mesh_grid()
    c = coefficients()
    settings = dict(method="corrected", preconditioner="ilu", restart=16,
                    batch_cells=2, max_outer_iterations=200)
    low = estimate_memory(mesh, c, **settings)
    high = estimate_memory(mesh, c, **{**settings, "max_outer_iterations": 201})
    assert high["estimated_bytes"]-low["estimated_bytes"] == 16384
    assert high["history_bytes"]-low["history_bytes"] == 16384
    with pytest.raises(MemoryError, match="pre-build"):
        pilot(mesh, c, method="corrected", max_bytes=low["estimated_bytes"]-1)


def test_corrected_geometry_rank_failure():
    mesh = mesh_grid()
    keep = [k for k, normal in enumerate(mesh.face_normals) if abs(normal[0]) > .9]
    thin = replace(mesh, faces=[mesh.faces[k] for k in keep],
                   face_centers=mesh.face_centers[keep], face_normals=mesh.face_normals[keep],
                   face_distances=mesh.face_distances[keep], bfaces=[],
                   bface_centers=np.empty((0, 2)), bface_normals=np.empty((0, 2)))
    with pytest.raises(ValueError, match="rank-deficient"):
        pilot(thin, method="corrected")
