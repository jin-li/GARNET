"""Independent checks for Cartesian two-point discontinuity factors."""

import warnings

import numpy as np
import pytest
import scipy.linalg

from garnet import (DiffusionEigenSolver, Grid, Material,
                    projected_adjoint_residual, projected_shape_residual)
from garnet.adf import assembly_face_factors
from garnet.stencil import GroupOperator


def faces(shape, groups=1, value=1.):
    arrays = []
    for axis in range(3):
        fshape = list(shape)
        fshape[axis] -= 1
        a = np.full((groups, *fshape), value)
        arrays.append((a.copy(), a.copy()))
    return tuple(arrays)


def single_group(adf):
    return tuple((a[0], b[0]) for a, b in adf)


def test_two_cell_interface_traces_and_balance():
    grid = Grid((2, 1, 1), (4., 1., 1.))
    D = np.array([2., 0.5]).reshape(grid.shape)
    adf = faces(grid.shape)
    adf[0][0][:] = 0.8
    adf[0][1][:] = 1.5
    op = GroupOperator(np, grid, D, np.zeros(grid.shape), bc='reflective', adf=single_group(adf))
    phi = np.array([3., 1.]).reshape(grid.shape)
    current = (0.8 * 3 - 1.5 * 1) / (0.8 * 1 / 2 + 1.5 * 1 / 0.5)
    assert op.apply(phi).ravel() == pytest.approx([current / 2, -current / 2])
    assert op.face_currents(phi)[0].ravel() == pytest.approx([current])
    assert 0.8 * (3 - current / 2) == pytest.approx(1.5 * (1 + current / 0.5))
    assert op.assemble().toarray() @ phi.ravel() == pytest.approx(op.apply(phi).ravel())


def test_directed_matrix_matches_independent_face_assembly_and_masks():
    rng = np.random.default_rng(41)
    shape = (3, 4, 2)
    grid = Grid(shape, (3., 8., 6.))
    D = 0.3 + rng.random(shape)
    removal = 0.1 + rng.random(shape)
    adf = faces(shape)
    for lo, hi in adf:
        lo[:] = 0.5 + rng.random(lo.shape)
        hi[:] = 0.5 + rng.random(hi.shape)
    active = rng.random(shape) > 0.2
    op = GroupOperator(np, grid, D, removal.copy(), bc='reflective', active=active,
                       mask_bc='reflective', adf=single_group(adf))
    mat = np.diag(np.where(active, removal, 1).ravel())
    ids = np.arange(np.prod(shape)).reshape(shape)
    for axis, (lo, hi) in enumerate(single_group(adf)):
        for ix in np.ndindex(lo.shape):
            iy = list(ix)
            iy[axis] += 1
            iy = tuple(iy)
            if not (active[ix] and active[iy]):
                continue
            d = grid.spacing[axis]
            den = d*d/2 * (lo[ix]/D[ix] + hi[ix]/D[iy])
            a, b = lo[ix]/den, hi[ix]/den
            i, j = ids[ix], ids[iy]
            mat[i, i] += a
            mat[i, j] -= b
            mat[j, i] -= a
            mat[j, j] += b
    phi = rng.random(shape)
    assert op.assemble().toarray() == pytest.approx(mat)
    assert op.apply(phi).ravel() == pytest.approx(mat @ phi.ravel())
    assert op.apply_transpose(phi).ravel() == pytest.approx(mat.T @ phi.ravel())
    divergence = np.zeros(shape)
    for axis, current in enumerate(op.face_currents(phi)):
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        divergence[tuple(lo)] += current / grid.spacing[axis]
        divergence[tuple(hi)] -= current / grid.spacing[axis]
    assert divergence == pytest.approx(op.apply(phi) - np.where(active, removal * phi, phi))
    assert np.sum(op.apply(phi) - np.where(active, removal * phi, phi)) == pytest.approx(0, abs=1e-12)


def test_equal_factors_and_region_refinement():
    shape = (4, 2, 1)
    region = np.zeros(shape, dtype=int)
    region[2:] = 1
    side = np.full((2, 6, *shape), 2.)
    side[:, 0, 2:] = 3.
    adf = assembly_face_factors(region, side)
    assert np.all(adf[0][0][:, 0] == 1)
    assert np.all(adf[0][0][:, 1] == 2)
    assert np.all(adf[0][1][:, 1] == 3)
    assert np.all(adf[1][0] == 1)
    grid = Grid(shape, (4., 2., 1.))
    D = np.arange(8).reshape(shape) + 1.
    plain = GroupOperator(np, grid, D, np.ones(shape), bc='vacuum')
    equal = GroupOperator(np, grid, D, np.ones(shape), bc='vacuum', adf=single_group(faces(shape, value=2)))
    assert equal.assemble().toarray() == pytest.approx(plain.assemble().toarray())


def test_facewise_common_factor_scale_invariance():
    shape = (3, 2, 2)
    rng = np.random.default_rng(27)
    grid = Grid(shape, (6., 4., 6.))
    adf = faces(shape)
    scaled = faces(shape)
    for (lo, hi), (lo_scaled, hi_scaled) in zip(adf, scaled):
        lo[:] = .5 + rng.random(lo.shape)
        hi[:] = .5 + rng.random(hi.shape)
        common = .2 + 3 * rng.random(lo.shape)
        lo_scaled[:] = lo * common
        hi_scaled[:] = hi * common
    D = .3 + rng.random(shape)
    removal = .1 + rng.random(shape)
    phi = rng.random(shape)
    a = GroupOperator(np, grid, D, removal, adf=single_group(adf))
    b = GroupOperator(np, grid, D, removal, adf=single_group(scaled))
    assert a.apply(phi) == pytest.approx(b.apply(phi))
    for current, scaled_current in zip(a.face_currents(phi), b.face_currents(phi)):
        assert current == pytest.approx(scaled_current)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('scale', [1e-308, 1e308])
def test_extreme_common_factor_scale_preserves_operator_and_current(dtype, scale):
    shape = (2, 1, 1)
    grid = Grid(shape, (2., 1., 1.))
    D = np.ones(shape, dtype=dtype)
    removal = np.full(shape, .1, dtype=dtype)
    phi = np.array([3., 1.], dtype=dtype).reshape(shape)
    baseline = faces(shape)
    baseline[0][0][:] = .8
    baseline[0][1][:] = 1.5
    scaled = tuple((lo * scale, hi * scale) for lo, hi in baseline)
    with np.errstate(over='raise', divide='raise', invalid='raise'):
        plain = GroupOperator(np, grid, D, removal, bc='reflective', adf=single_group(baseline))
        extreme = GroupOperator(np, grid, D, removal, bc='reflective', adf=single_group(scaled))
    tolerance = 3e-7 if dtype == np.float32 else 1e-13
    np.testing.assert_allclose(extreme.assemble().toarray(), plain.assemble().toarray(), rtol=tolerance)
    np.testing.assert_allclose(extreme.apply(phi), plain.apply(phi), rtol=tolerance)
    np.testing.assert_allclose(extreme.apply_transpose(phi), plain.apply_transpose(phi), rtol=tolerance)
    np.testing.assert_allclose(extreme.face_currents(phi)[0], plain.face_currents(phi)[0], rtol=tolerance)


def test_unrepresentable_factor_ratio_is_rejected_instead_of_decoupling():
    shape = (2, 1, 1)
    adf = faces(shape)
    adf[0][0][:] = 1e-300
    adf[0][1][:] = 1e300
    with pytest.raises(ValueError, match='finite and positive'):
        GroupOperator(np, Grid(shape, (2., 1., 1.)), np.ones(shape),
                      np.full(shape, .1), adf=single_group(adf))


@pytest.mark.parametrize('imaginary_part', [0., 3.])
def test_complex_adf_input_is_rejected_before_any_real_cast(imaginary_part):
    shape = (2, 1, 1)
    grid = Grid(shape, (2., 1., 1.))
    sides = np.full((1, 6, *shape), 1 + imaginary_part * 1j)
    adf = faces(shape)
    adf = ((adf[0][0].astype(complex) + imaginary_part * 1j, adf[0][1]), *adf[1:])
    # Treat warnings as errors to ensure invalid complex inputs cannot first
    # pass through NumPy's lossy-cast warning on any entry path.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError, match='real numbers'):
            assembly_face_factors(np.arange(2).reshape(shape), sides)
        with pytest.raises(ValueError, match='real numbers'):
            GroupOperator(np, grid, np.ones(shape), np.full(shape, .1), adf=single_group(adf))
        with pytest.raises(ValueError, match='real numbers'):
            DiffusionEigenSolver(grid, Material([1.], [.1], [.2]), device='cpu',
                                  adf=adf, linear_solver='gmres')


def two_group_case():
    grid = Grid((2, 1, 1), (6., 1., 1.))
    mats = [Material([1., .4], [.01, .05], [.01, .08], [[0, .02], [0, 0]]),
            Material([.8, .2], [.02, .07], [.015, .11], [[0, .03], [0, 0]])]
    adf = faces(grid.shape, 2)
    adf[0][0][:, 0, 0, 0] = [.8, 1.5]
    adf[0][1][:, 0, 0, 0] = [1.2, .7]
    return grid, mats, adf


def two_group_solver(device):
    grid, mats, adf = two_group_case()
    return DiffusionEigenSolver(grid, mats, np.array([0, 1]).reshape(grid.shape),
                                bc='reflective', device=device, adf=adf, linear_solver='gmres',
                                precond_degree=1)


TIGHT = dict(tol_k=1e-11, tol_source=1e-10, inner_rtol_floor=1e-12)


def test_two_group_eigenvalue_against_dense_independent_system():
    grid, mats, adf = two_group_case()
    s = two_group_solver('cpu')
    result = s.solve(**TIGHT)
    assert result.converged
    loss = np.zeros((4, 4))
    for g in range(2):
        D = [m.diffusion[g] for m in mats]
        f, h = adf[0][0][g, 0, 0, 0], adf[0][1][g, 0, 0, 0]
        den = 4.5 * (f/D[0] + h/D[1])
        a, b = f/den, h/den
        loss[2*g:2*g+2, 2*g:2*g+2] = [[a+mats[0].removal[g], -b], [-a, b+mats[1].removal[g]]]
    loss[2:, :2] = -np.diag([m.sigma_s[0, 1] for m in mats])
    fiss = np.zeros((4, 4))
    fiss[:2, :2] = np.diag([m.nu_sigma_f[0] for m in mats])
    fiss[:2, 2:] = np.diag([m.nu_sigma_f[1] for m in mats])
    k = max(scipy.linalg.eigvals(fiss, loss).real)
    assert result.k_eff == pytest.approx(k, rel=1e-8)
    adjoint = s.solve(adjoint=True, **TIGHT)
    assert adjoint.converged
    assert adjoint.k_eff == pytest.approx(k, rel=1e-8)
    psi = adjoint.flux_numpy.ravel()
    assert np.linalg.norm(loss.T @ psi - fiss.T @ psi / k) / np.linalg.norm(loss.T @ psi) < 1e-8
    assert projected_shape_residual(s, result, adjoint) < 1e-9
    assert projected_adjoint_residual(s, result, adjoint) < 1e-9
    assert projected_adjoint_residual(s, 7 * result.flux, .03 * adjoint.flux) < 1e-9
    distorted = adjoint.flux_numpy.copy()
    distorted[0, 0, 0, 0] *= 1.2
    assert projected_adjoint_residual(s, result, distorted) > 1e-3


def test_reject_invalid_combinations():
    grid = Grid((2, 1, 1), (2., 1., 1.))
    mat = Material([1.], [.1], [.2])
    adf = faces(grid.shape)
    with pytest.raises(ValueError, match='nonsymmetric'):
        DiffusionEigenSolver(grid, mat, device='cpu', adf=adf)
    adf[0][0][:] = 0
    with pytest.raises(ValueError, match='positive'):
        DiffusionEigenSolver(grid, mat, device='cpu', adf=adf, linear_solver='gmres')
    adf = faces(grid.shape)
    DiffusionEigenSolver(grid, mat, device='cpu', adf=adf, linear_solver='gmres')
    with pytest.raises(ValueError, match='shape'):
        DiffusionEigenSolver(grid, mat, device='cpu', adf=faces(grid.shape, 2), linear_solver='gmres')


def test_fused_dispatch_uses_correct_directed_weights(monkeypatch):
    from garnet import kernels
    shape = (3, 2, 2)
    grid = Grid(shape, (3., 4., 6.))
    rng = np.random.default_rng(13)
    adf = faces(shape)
    for lo, hi in adf:
        lo[:] = .5 + rng.random(lo.shape)
        hi[:] = .5 + rng.random(hi.shape)
    op = GroupOperator(np, grid, .4 + rng.random(shape), .1 + rng.random(shape), adf=single_group(adf))
    phi = rng.random(shape)
    expected = op.apply(phi)
    expected_t = op.apply_transpose(phi)
    calls = []
    def emulate(xp, p, diag, wx, wy, wz, xh, yh, zh, out=None):
        calls.append(True)
        result = diag * p
        for axis, (low, high) in enumerate(((wx, xh), (wy, yh), (wz, zh))):
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            result[b] -= low.reshape(tuple(shape[i] - (i == axis) for i in range(3))) * p[a]
            result[a] -= high.reshape(tuple(shape[i] - (i == axis) for i in range(3))) * p[b]
        if out is not None:
            out[...] = result
            return out
        return result
    monkeypatch.setattr(kernels, 'use_fused', lambda *args: True)
    monkeypatch.setattr(kernels, 'stencil7_adf_apply', emulate)
    assert op.apply(phi) == pytest.approx(expected)
    assert op.apply_transpose(phi) == pytest.approx(expected_t)
    assert calls


def test_bernal_case_setup_and_base_solve():
    from examples.bernal_adf_benchmark import make_problem, provisional_map, run
    base = provisional_map()
    assert np.count_nonzero(base) == 1170
    solver, mmap = make_problem(base, 2, True)
    assert np.all(solver.ops[0].wx[::2] == solver.ops[0]._high_weights[0][::2])
    assert np.all(solver.fields.nu_sigma_f[0][mmap == 0] == 0)
    result = run(base, 1, True)
    assert result['relative_eigen_residual'] < 1e-7




def test_adf_kernel_source_edits_must_match_exactly_once():
    from garnet import kernels
    assert kernels._stencil7_adf_src().count('_hi[') == 3
    broken = kernels._STENCIL7_SRC.replace('wz[wzb + k]     * phi[i + 1]', 'wz[wzb + k] * phi[i + 1]')
    with pytest.raises(RuntimeError, match='exactly once'):
        kernels._stencil7_adf_src(broken)


def test_face_traces_reconstruct_both_sides():
    grid = Grid((2, 1, 1), (4., 1., 1.))
    D = np.array([2., 0.5]).reshape(grid.shape)
    phi = np.array([3., 1.]).reshape(grid.shape)
    adf = faces(grid.shape)
    adf[0][0][:] = 0.8
    adf[0][1][:] = 1.5
    op = GroupOperator(np, grid, D, np.zeros(grid.shape), bc='reflective', adf=single_group(adf))
    current = (0.8 * 3 - 1.5 * 1) / (0.8 * 1 / 2 + 1.5 * 1 / 0.5)
    low, high = op.face_traces(phi)[0]
    assert low.ravel() == pytest.approx([3 - current / 2])
    assert high.ravel() == pytest.approx([1 + current / 0.5])
    assert 0.8 * low.ravel() == pytest.approx(1.5 * high.ravel())
    plain = GroupOperator(np, grid, D, np.zeros(grid.shape), bc='reflective')
    for low, high in plain.face_traces(np.random.default_rng(3).random(grid.shape)):
        assert low == pytest.approx(high)


def test_region_side_traces_and_factor_expansion():
    from garnet.adf import expand_region_side_factors, region_side_traces
    rng = np.random.default_rng(5)
    shape = (4, 3, 2)
    grid = Grid(shape, (4., 6., 2.))
    regions = np.zeros(shape, dtype=int)
    regions[2:] = 1
    side = np.ones((2, 2, 6))
    side[:, 0, 1] = [.9, 1.3]
    side[:, 1, 0] = [1.1, .6]
    cell_side = expand_region_side_factors(regions, side)
    assert cell_side.shape == (2, 6, *shape)
    assert np.all(cell_side[1, 0, 2:] == .6) and np.all(cell_side[0, 1, :2] == .9)
    adf = assembly_face_factors(regions, cell_side)
    D = .5 + rng.random((2, *shape))
    ops = [GroupOperator(np, grid, D[g], .1 + rng.random(shape), bc='reflective',
                         adf=tuple((lo[g], hi[g]) for lo, hi in adf)) for g in range(2)]
    flux = rng.random((2, *shape))
    trace, current, area = region_side_traces(ops, flux, regions)
    assert area[0, 1] == area[1, 0] == pytest.approx(3 * 2 * (2. * 1.))  # 3x2 faces, 2 cm x 1 cm
    assert np.isnan(trace[:, 0, 0]).all() and np.all(current[:, 0, 0] == 0)
    assert current[:, 0, 1] == pytest.approx(-current[:, 1, 0])
    # Corrected surface flux is continuous across the region interface.
    assert side[:, 0, 1] * trace[:, 0, 1] == pytest.approx(side[:, 1, 0] * trace[:, 1, 0])
    for g, op in enumerate(ops):
        j = op.face_currents(flux[g])[0][1]
        assert current[g, 0, 1] == pytest.approx(j.sum() * 2.)


@pytest.fixture
def cuda():
    cp = pytest.importorskip('cupy')
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('no CUDA device')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('no CUDA runtime')
    return cp


@pytest.mark.parametrize('shape', [(4, 3, 2), (17, 13, 11), (1, 9, 7)])
@pytest.mark.parametrize('dtype', [np.float64, np.float32])
@pytest.mark.parametrize('masked', [False, True])
def test_gpu_adf_fused_matches_unfused(cuda, shape, dtype, masked):
    cp = cuda
    from garnet import kernels
    rng = np.random.default_rng(19)
    grid = Grid(shape, tuple(2. * n for n in shape))
    adf = faces(shape)
    for lo, hi in adf:
        lo[:] = .5 + rng.random(lo.shape)
        hi[:] = .5 + rng.random(hi.shape)
    active = cp.asarray(rng.random(shape) > .2) if masked else None
    D = cp.asarray(.5 + rng.random(shape), dtype=dtype)
    removal = cp.asarray(.1 + rng.random(shape), dtype=dtype)
    op = GroupOperator(cp, grid, D, removal, active=active, adf=single_group(adf))
    phi = cp.asarray(rng.random(shape), dtype=dtype)
    tol = 2e-6 if dtype == np.float32 else 1e-13
    previous = kernels.set_fused(True)
    try:
        for apply in (op.apply, op.apply_transpose):
            fused = apply(phi)
            assert fused.dtype == dtype
            out = cp.full_like(phi, cp.nan)
            assert apply(phi, out=out) is out
            kernels.set_fused(False)
            unfused = cp.asnumpy(apply(phi))
            kernels.set_fused(True)
            scale = np.abs(unfused).max()
            np.testing.assert_allclose(cp.asnumpy(fused), unfused, rtol=tol, atol=tol * scale)
            np.testing.assert_allclose(cp.asnumpy(out), unfused, rtol=tol, atol=tol * scale)
    finally:
        kernels.set_fused(previous)
    # The CPU operator is an independent reference for the GPU arithmetic.
    cpu = GroupOperator(np, grid, cp.asnumpy(D), cp.asnumpy(removal),
                        active=None if active is None else cp.asnumpy(active),
                        adf=single_group(adf))
    reference = cpu.apply(cp.asnumpy(phi))
    np.testing.assert_allclose(cp.asnumpy(op.apply(phi)), reference,
                               rtol=tol, atol=tol * np.abs(reference).max())


def test_gpu_two_group_adf_solve_matches_cpu(cuda):
    cpu, gpu = two_group_solver('cpu'), two_group_solver('gpu')
    assert gpu.device != 'cpu'
    fwd = {d: s.solve(**TIGHT) for d, s in (('cpu', cpu), ('gpu', gpu))}
    adj = {d: s.solve(adjoint=True, **TIGHT) for d, s in (('cpu', cpu), ('gpu', gpu))}
    for results in (fwd, adj):
        assert all(r.converged for r in results.values())
        assert results['gpu'].k_eff == pytest.approx(results['cpu'].k_eff, abs=1e-9)
        a, b = (results[d].flux_numpy for d in ('cpu', 'gpu'))
        np.testing.assert_allclose(b / np.linalg.norm(b), a / np.linalg.norm(a), atol=1e-7)
    assert projected_shape_residual(gpu, fwd['gpu'], adj['gpu']) < 1e-9
    assert projected_adjoint_residual(gpu, fwd['gpu'], adj['gpu']) < 1e-9


def test_gpu_bernal_base_case_matches_cpu(cuda):
    from examples.bernal_adf_benchmark import provisional_map, run
    base = provisional_map()
    cpu, gpu = run(base, 1, True), run(base, 1, True, device='gpu')
    assert gpu['k_eff'] == pytest.approx(cpu['k_eff'], abs=1e-8)
    np.testing.assert_allclose(gpu['normalized_assembly_fission_production'],
                               cpu['normalized_assembly_fission_production'], atol=1e-7)
