"""Current-state balance and independent dense eigenvalue references."""
import numpy as np
import pytest

from ndgpu import (Grid, Material, DiffusionEigenSolver, SP3EigenSolver,
                   SDP1EigenSolver, MeshDiffusionEigenSolver)
from ndgpu.nodal_mesh import build_rectilinear_mesh_3d
from ndgpu import solver as solver_module


def material():
    return Material(diffusion=[1., .3], sigma_a=[.2, 0.], nu_sigma_f=[.2, 0.],
                    sigma_s=[[0., .1], [.1, 0.]], chi=[.5, .5], total=[1., 1.])


def masked_mesh():
    x = [-110., -65., -20., 20., 65., 110.]
    y = [-90., -45., -10., 15., 55., 100.]
    z = [0., 20., 20.5613, 80., 160., 220.5613, 240.]
    mask = np.ones((6, 5, 5), dtype=bool)
    mask[:, 0, 0] = mask[:, 0, -1] = mask[:, -1, 0] = mask[:, -1, -1] = False
    return build_rectilinear_mesh_3d(x, y, z, active_mask=mask, tags=np.zeros(mask.shape, dtype=int))[0]


def small_masked_mesh():
    mask = np.ones((2, 2, 3), dtype=bool)
    mask[0, 1, 0] = False
    return build_rectilinear_mesh_3d([0, 3, 8, 12], [0, 2, 7], [0, 1, 4],
                                   active_mask=mask, tags=np.zeros(mask.shape, dtype=int))[0]


def dense_pencil(mesh, mat, material_map=None):
    """Independent integrated FV balance with source-row scattering."""
    mats = mat if isinstance(mat, list) else [mat]
    indices = np.zeros(mesh.n_cells, dtype=int) if material_map is None else material_map
    cells = [mats[i] for i in indices]
    n, groups = mesh.n_cells, mats[0].n_groups
    loss = np.zeros((groups*n, groups*n))
    fission = np.zeros_like(loss)
    for g in range(groups):
        offset = g*n
        loss[offset:offset+n, offset:offset+n] = np.diag(mesh.area*np.array([m.removal[g] for m in cells]))
        for face, distances in zip(mesh.faces, mesh.face_distances):
            i, j, area, _ = face
            di, dj = distances
            conductance = area/(di/cells[i].diffusion[g]+dj/cells[j].diffusion[g])
            loss[offset+i, offset+i] += conductance
            loss[offset+j, offset+j] += conductance
            loss[offset+i, offset+j] -= conductance
            loss[offset+j, offset+i] -= conductance
        # Vacuum alpha=.5; exact half-cell normal boundary resistance.
        for i, area, distance in mesh.bfaces:
            diffusion = cells[i].diffusion[g]
            loss[offset+i, offset+i] += area*.5*diffusion/(diffusion+.5*distance)
        for origin in range(groups):
            if origin != g:
                loss[offset:offset+n, origin*n:(origin+1)*n] -= np.diag(mesh.area*np.array([m.sigma_s[origin, g] for m in cells]))
            fission[offset:offset+n, origin*n:(origin+1)*n] = np.diag(mesh.area*np.array([m.chi[g]*m.nu_sigma_f[origin] for m in cells]))
    return loss, fission


def actual_balance(solver, result, adjoint=False):
    prod = solver.chi if adjoint else solver.nu_sigma_f
    emit = solver.nu_sigma_f if adjoint else solver.chi
    source = sum(prod[g]*solver._phi(solver.state[g]) for g in range(solver.n_groups))
    residual, norm = 0., 0.
    for g in range(solver.n_groups):
        q = emit[g]*source/result.k_eff
        for origin in range(solver.n_groups):
            scatter = solver.sigma_s[g][origin] if adjoint else solver.sigma_s[origin][g]
            if origin != g and scatter is not None:
                q = q+scatter*solver._phi(solver.state[origin])
        if solver._src_weight is not None:
            q = q*solver._src_weight
        loss = solver.ops[g].apply(solver.state[g])
        residual += np.sum((loss-solver._rhs(g, q))**2)
        norm += np.sum(loss**2)
    return np.sqrt(residual/norm)


@pytest.mark.parametrize('unity_df', [False, True])
@pytest.mark.parametrize('factory', [masked_mesh, small_masked_mesh])
def test_default_anderson_reaches_dense_truth_and_current_balance(unity_df, factory):
    mesh, mat = factory(), material()
    loss, fission = dense_pencil(mesh, mat)
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(loss, fission))
    selected = np.argmax(eigenvalues.real)
    k = eigenvalues[selected].real
    truth = np.abs(eigenvectors[:, selected].real)
    kwargs = {} if not unity_df else {'face_df': np.ones((2, len(mesh.faces), 2)),
                                    'boundary_df': np.ones((2, len(mesh.bfaces)))}
    solver = MeshDiffusionEigenSolver(mesh, mat, bc='vacuum', device='cpu', **kwargs)
    result = solver.solve(tol_k=1e-10, tol_source=1e-9, inner_rtol_floor=1e-12, max_outer=2000)
    assert result.converged
    assert actual_balance(solver, result) < 1e-9
    assert result.balance_error == pytest.approx(actual_balance(solver, result), rel=1e-10)
    assert observed_positive(result)
    assert result.k_eff == pytest.approx(k, abs=3e-9)
    observed = result.flux_numpy.ravel()
    np.testing.assert_allclose(observed/np.linalg.norm(observed), truth/np.linalg.norm(truth), rtol=2e-6, atol=2e-8)


def observed_positive(result):
    return np.all(np.isfinite(result.flux_numpy)) and np.all(result.flux_numpy > 0) and result.k_eff > 0


def test_plain_iteration_small_mesh_reaches_same_dense_reference():
    mesh, _ = build_rectilinear_mesh_3d([0, 2, 5], [0, 3], [0, 4],
        active_mask=np.ones((1, 1, 2), dtype=bool), tags=np.zeros((1, 1, 2), dtype=int))
    mat = material()
    loss, fission = dense_pencil(mesh, mat)
    k = np.linalg.eigvals(np.linalg.solve(loss, fission)).real.max()
    solver = MeshDiffusionEigenSolver(mesh, mat, bc='vacuum', device='cpu')
    result = solver.solve(tol_k=1e-10, tol_source=1e-9, inner_rtol_floor=1e-12, anderson_depth=1)
    assert result.converged and actual_balance(solver, result) < 1e-9
    assert result.k_eff == pytest.approx(k, abs=1e-9)


@pytest.mark.parametrize('bad', ['negative', 'nan', 'infinite', 'zero'])
def test_invalid_anderson_proposals_rejected_whole_and_history_reset(monkeypatch, bad):
    mesh, mat = small_masked_mesh(), material()
    loss, fission = dense_pencil(mesh, mat)
    truth = np.linalg.eigvals(np.linalg.solve(loss, fission)).real.max()
    lengths = []
    def proposal(hist, raw, xp, integration_weights):
        lengths.append(len(hist))
        values = raw.copy()
        if bad == 'negative':
            values.flat[0] = -1000
        else:
            values[:] = {'nan': np.nan, 'infinite': np.inf, 'zero': 0}[bad]
        return values
    monkeypatch.setattr(solver_module, '_anderson_source', proposal)
    solver = MeshDiffusionEigenSolver(mesh, mat, bc='vacuum', device='cpu')
    result = solver.solve(tol_k=1e-10, tol_source=1e-9, inner_rtol_floor=1e-12)
    assert lengths and set(lengths) == {1}
    assert result.converged and observed_positive(result)
    assert result.k_eff == pytest.approx(truth, abs=1e-9)
    assert actual_balance(solver, result) < 1e-9


def test_unsettled_fixed_source_budget_never_updates_k_or_anderson(monkeypatch):
    monkeypatch.setattr(solver_module, '_anderson_source', lambda *a: pytest.fail('Unsettled solve entered Anderson history'))
    solver = MeshDiffusionEigenSolver(small_masked_mesh(), material(), bc='vacuum', device='cpu')
    result = solver.solve(max_group_sweeps=1, max_outer=20, k_guess=1.234)
    assert not result.converged and result.outer_iterations == 1
    assert result.k_eff == 1.234 and result.k_history == [] and result.source_error_history == []
    assert result.balance_error == pytest.approx(actual_balance(solver, result))
    assert result.failure_reason == 'fixed-source group budget exhausted before requested balance'
    assert result.fixed_source_error > 1e-12


def test_full_balance_gate_rejects_loose_source_and_k_candidate():
    solver = MeshDiffusionEigenSolver(small_masked_mesh(), material(), bc='vacuum', device='cpu')
    result = solver.solve(tol_k=.1, tol_source=.1, tol_balance=1e-30, max_outer=30)
    assert not result.converged and result.balance_error > 1e-30
    assert result.balance_error == pytest.approx(actual_balance(solver, result))


@pytest.mark.parametrize('scatter', [[[0, .1], [0, 0]], [[0, 0], [.1, 0]], [[0, .1], [.1, 0]]])
def test_up_down_and_bidirectional_scattering_dense_reference(scatter):
    mesh = small_masked_mesh()
    mat = material()
    mat.sigma_s = np.array(scatter, dtype=float)
    loss, fission = dense_pencil(mesh, mat)
    truth = np.linalg.eigvals(np.linalg.solve(loss, fission)).real.max()
    solver = MeshDiffusionEigenSolver(mesh, mat, bc='vacuum', device='cpu')
    result = solver.solve(tol_k=1e-10, tol_source=1e-9, inner_rtol_floor=1e-12)
    assert result.converged and result.k_eff == pytest.approx(truth, abs=1e-9)
    assert result.balance_error < 1e-9


@pytest.mark.parametrize('cls', [DiffusionEigenSolver, SP3EigenSolver, SDP1EigenSolver])
def test_moment_rhs_hooks_adjoint_and_warm_start(cls):
    grid = Grid((3, 2, 1), (9., 7., 3.))
    solver = cls(grid, material(), bc=('vacuum', 'vacuum', 'reflective'), device='cpu')
    result = solver.solve(tol_k=1e-10, tol_source=1e-9, inner_rtol_floor=1e-12)
    assert result.converged and observed_positive(result)
    assert actual_balance(solver, result) < 1e-9
    initial = [state.copy() for state in solver.state]
    warm = solver.solve(state0=initial, tol_k=1e-10, tol_source=1e-9, inner_rtol_floor=1e-12)
    assert warm.converged and warm.k_eff == pytest.approx(result.k_eff, abs=1e-9)
    for before, after in zip(initial, solver.state):
        np.testing.assert_allclose(before, after, rtol=1e-8, atol=1e-8)
    adjoint = solver.solve(adjoint=True, tol_k=1e-10, tol_source=1e-9, inner_rtol_floor=1e-12)
    assert adjoint.converged and adjoint.k_eff == pytest.approx(result.k_eff, abs=1e-9)
    assert actual_balance(solver, adjoint, adjoint=True) < 1e-9


def test_cylindrical_source_metric_weight_preserved():
    solver = DiffusionEigenSolver(Grid((3, 1, 2), (8., 1., 6.), geometry='cylindrical'),
        material(), bc=('vacuum', 'reflective', 'vacuum'), device='cpu')
    result = solver.solve(tol_k=1e-10, tol_source=1e-9, inner_rtol_floor=1e-12)
    assert solver._src_weight is not None and result.converged
    assert actual_balance(solver, result) < 1e-9


@pytest.mark.parametrize('argument,value', [('tol_balance', 0), ('tol_balance', np.nan),
    ('tol_fixed_source', 0), ('tol_fixed_source', np.inf), ('max_group_sweeps', 0),
    ('max_group_sweeps', True), ('inner_rtol_floor', 0), ('k_guess', -1)])
def test_invalid_fixed_source_and_balance_controls(argument, value):
    solver = MeshDiffusionEigenSolver(small_masked_mesh(), material(), bc='vacuum', device='cpu')
    with pytest.raises(ValueError):
        solver.solve(**{argument: value})


def test_explicit_unattainable_fp32_fixed_source_tolerance_preserves_dtype():
    solver = MeshDiffusionEigenSolver(small_masked_mesh(), material(), bc='vacuum',
                                     device='cpu', dtype=np.float32)
    result = solver.solve(tol_fixed_source=1e-12, max_group_sweeps=3)
    assert not result.converged
    assert result.flux.dtype == np.float32
    assert all(s.dtype == np.float32 for s in solver.state)
    assert result.fixed_source_error > 1e-12
    assert result.fixed_source_tolerance == 1e-12
    assert 'fixed-source' in result.failure_reason


def test_attainable_fp32_precision_preserves_dtype_and_balance():
    mesh, mat = small_masked_mesh(), material()
    other = Material(diffusion=[.8, .4], sigma_a=[.18, .03], nu_sigma_f=[.24, .01],
                     sigma_s=[[0., .08], [.06, 0.]], chi=[.6, .4], total=[1., 1.])
    mats = [mat, other]
    mapping = np.arange(mesh.n_cells) % 2
    loss, fission = dense_pencil(mesh, mats, mapping)
    truth = np.linalg.eigvals(np.linalg.solve(loss, fission)).real.max()
    solver = MeshDiffusionEigenSolver(mesh, mats, material_map=mapping,
                                     bc='vacuum', device='cpu', dtype=np.float32)
    result = solver.solve(tol_fixed_source=None, tol_k=1e-6,
                          tol_source=1e-5, tol_balance=1e-5)
    assert result.converged and result.failure_reason is None
    assert result.flux.dtype == np.float32
    assert result.balance_error < 1e-5
    assert actual_balance(solver, result) < 1e-5
    assert result.fixed_source_tolerance == 100*np.finfo(np.float32).eps
    assert result.group_rtol == 8*np.finfo(np.float32).eps
    assert result.k_eff == pytest.approx(truth, abs=3e-5)


def test_zero_source_group_is_exactly_empty():
    mat = material()
    mat.chi = np.array([1., 0.])
    mat.sigma_s = np.zeros((2, 2))
    mat.sigma_a = np.array([.2, .2])
    solver = DiffusionEigenSolver(Grid((2, 1, 1), (3., 1., 1.)), mat,
                                 bc='reflective', device='cpu')
    result = solver.solve()
    assert result.converged
    assert result.scalar_flux_min == 0
    assert np.all(result.flux[1] == 0)


@pytest.mark.parametrize('negative,accepted', [(-1e-16, True), (-95.7, False)])
def test_scalar_trace_roundoff_guard_preserves_values(monkeypatch, negative, accepted):
    mat = material()
    mat.chi = np.array([1., 0.])
    mat.sigma_s = np.zeros((2, 2))
    mat.sigma_a = np.array([.2, .2])
    solver = DiffusionEigenSolver(Grid((2, 1, 1), (3., 1., 1.)), mat,
                                 bc='reflective', device='cpu')
    # Emulate cancellation in a physical trace of an otherwise empty group.
    monkeypatch.setattr(solver, '_phi', lambda s: s + negative if not np.any(s) else s)
    result = solver.solve()
    assert result.converged == accepted
    assert np.all(result.flux[1] < 0)
    if accepted:
        assert result.scalar_flux_min >= -result.scalar_flux_negative_tolerance
    else:
        assert result.scalar_flux_min < -result.scalar_flux_negative_tolerance
        assert result.failure_reason == 'nonphysical fixed-source iterate'
