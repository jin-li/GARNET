"""Independent continuum, dense-saddle, conservation and contract checks."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ndgpu.reflector_cuboid_reference import (
    solve_reflector_cuboid_reference, ReflectorCuboidReferenceError,
    NonpositiveReflectorCuboidError,
)
from ndgpu.reflector_vacuum_reference import solve_reflector_vacuum_reference as solve

D = np.array([1.2, .7])
R = np.array([[.8, -.12], [-.2, .5]])
L = np.array([1.3, 2.1, .8])
MU = np.array([3., 4.])
I = (np.arange(12).reshape(6, 2)-5)*.004


def test_analytic_mixed_slab_refinement():
    d, r, length, mu, jin, alpha = 1.1, .6, 2.3, 2., .14, .5
    k = np.sqrt(r/d)
    b = -jin/(d*k)
    sh, ch = np.sinh(k*length), np.cosh(k*length)
    c, a = np.linalg.solve([[1, sh/(k*length)],
        [alpha, d*k*sh+alpha*ch]],
        [mu-b*(ch-1)/(k*length), -b*(d*k*ch+alpha*sh)])
    errors = []
    for n in (8, 16, 32):
        currents = np.zeros((6, 1)); currents[0, 0] = jin
        result = solve([d], [[r]], [length, 1, 1], [mu], currents,
                       [False, False, True, False, False, False], subdivisions=(n, 1, 1))
        edges = np.linspace(0, length, n+1)
        exact = c+a*np.diff(np.sinh(k*edges))/(k*length/n)+b*np.diff(np.cosh(k*edges))/(k*length/n)
        errors.append(np.max(abs(result.cell_flux.ravel()-exact)))
        assert abs(result.delta[0]-r*c) < .07/n**2
        assert_allclose(result.mean, [mu], rtol=2e-14)
    assert errors[0]/errors[1] > 3.4
    assert errors[1]/errors[2] > 3.7


@pytest.mark.parametrize('affine', [False, True])
def test_independent_dense_saddle_non_cubic(affine):
    counts = (2, 4, 6) if affine else (2, 3, 4)
    mask = np.array([True, False, False, True, False, True])
    currents = I.copy(); currents[mask] = 0
    moments = (np.arange(24).reshape(6, 2, 2)-11)*.0003 if affine else np.zeros((6, 2, 2))
    moments[mask] = 0
    result = solve(D, R, L, MU, currents, mask, subdivisions=counts,
                   net_inward_sign_moments=moments)
    nx, ny, nz = counts; n = nx*ny*nz; gcount = 2
    h = L/counts; cv = np.prod(h)
    matrix = np.zeros((gcount*n+gcount, gcount*n+gcount))
    rhs = np.zeros(gcount*n+gcount)
    tangents = ((1, 2), (0, 2), (1, 2), (0, 2), (0, 1), (0, 1))
    specs = ((0, 0), (1, 0), (0, nx-1), (1, ny-1), (2, 0), (2, nz-1))
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                pos = (x, y, z); cell = x+nx*(y+ny*z)
                for g in range(gcount):
                    row = g*n+cell
                    for other in range(gcount):
                        matrix[row, other*n+cell] += cv*R[g, other]
                    matrix[row, gcount*n+g] = -cv
                    matrix[gcount*n+g, row] = 1/n
                    for axis, stride in enumerate((1, nx, nx*ny)):
                        for direction in (-1, 1):
                            if 0 <= pos[axis]+direction < counts[axis]:
                                conductance = cv*D[g]/h[axis]**2
                                matrix[row, row] += conductance
                                matrix[row, row+direction*stride] -= conductance
                    for s, (axis, coordinate) in enumerate(specs):
                        if pos[axis] != coordinate:
                            continue
                        patch_area = cv/h[axis]
                        if mask[s]:
                            # Half-cell diffusion resistance in series with 1/alpha.
                            matrix[row, row] += patch_area/(2+h[axis]/(2*D[g]))
                        else:
                            area = np.prod(L)/L[axis]
                            density = currents[s, g]/area
                            for t, a in enumerate(tangents[s]):
                                coordinate_t = (pos[a]+.5)*h[a]-L[a]/2
                                density += 4*moments[s, t, g]*coordinate_t/(area*L[a])
                            rhs[row] += density*patch_area
    rhs[-gcount:] = MU
    exact = np.linalg.solve(matrix, rhs)
    assert_allclose(result.cell_flux.ravel(), exact[:-gcount], atol=5e-14)
    assert_allclose(result.delta, exact[-gcount:], atol=5e-14)
    assert result.diagnostics['operator_condition_number_1'] > 1
    # Independent 1-norm condition from this small assembled integrated operator.
    assert_allclose(result.diagnostics['operator_condition_number_1'],
                    np.linalg.cond(matrix[:-gcount, :-gcount], 1), rtol=2e-13)
    assert result.diagnostics['factorization_count'] == 1


@pytest.mark.parametrize('affine', [False, True])
def test_no_vacuum_matches_dct(affine):
    moments = (np.arange(24).reshape(6, 2, 2)-11)*.001 if affine else None
    options = dict(subdivisions=(2, 4, 6), net_inward_sign_moments=moments)
    expected = solve_reflector_cuboid_reference(D, R, L, MU, I, **options)
    result = solve(D, R, L, MU, I, [False]*6, **options)
    assert_allclose(result.cell_flux, expected.cell_flux, rtol=2e-14)
    assert_allclose(result.delta, expected.delta, rtol=2e-14)
    for actual, reference in zip(result.face_traces, expected.face_traces):
        assert_allclose(actual, reference, rtol=2e-14)


@pytest.mark.parametrize('sides', [(i,) for i in range(6)]+[(0, 1, 4), tuple(range(6))])
def test_orientation_corner_masks_and_volume_scaling(sides):
    mask = np.array([s in sides for s in range(6)])
    currents = I.copy(); currents[mask] = 0
    moments = np.full((6, 2, 2), .003); moments[mask] = 0
    base = solve(D, R, L, MU, currents, mask, subdivisions=(2, 4, 6), net_inward_sign_moments=moments)
    # Geometric similarity: D -> s² D, alpha -> s alpha, integrated I -> s³ I.
    scale = 3.7
    other = solve(D*scale**2, R, L*scale, MU, currents*scale**3, mask,
                  alpha=.5*scale, subdivisions=(2, 4, 6), net_inward_sign_moments=moments*scale**3)
    assert_allclose(base.cell_flux, other.cell_flux, rtol=3e-14)
    assert_allclose(base.delta, other.delta, rtol=3e-14)
    for s in range(6):
        current = base.face_patch_currents[s]
        if mask[s]:
            assert_allclose(current, -.5*base.face_traces[s]*base.face_patch_areas[s], rtol=1e-15)
        else:
            assert_allclose(current.sum(axis=(1, 2)), currents[s], atol=2e-17)
            for t, sign in enumerate(base.diagnostics['face_patch_signs'][s]):
                assert_allclose((current*sign).sum(axis=(1, 2)), moments[s, t], atol=2e-17)
    assert_allclose(np.prod(L)*(R@MU-base.delta),
                    sum(p.sum(axis=(1, 2)) for p in base.face_patch_currents), atol=1e-13)


def test_negative_retained_and_immutable():
    currents = np.zeros((6, 1)); currents[0, 0] = -100
    args = ([1.], [[.1]], [1, 1, 1], [1.], currents, [False, False, True, False, False, False])
    with pytest.raises(NonpositiveReflectorCuboidError) as caught:
        solve(*args, subdivisions=4)
    result = solve(*args, subdivisions=4, require_positive=False)
    assert_allclose(result.cell_flux, caught.value.result.cell_flux)
    assert result.cell_flux.min() < 0
    assert_allclose(result.mean, [1.], atol=1e-13)
    for array in (result.cell_flux, result.delta, result.mean, result.face_means,
                  *result.face_traces, *result.face_patch_areas, *result.face_patch_currents,
                  result.diagnostics['loss_positive_scaling']):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        result.diagnostics['alpha'] = 1


@pytest.mark.parametrize('kwargs', [
    dict(diffusion=[1, 0]), dict(diffusion=[1+1j, 1]), dict(widths=[1, np.inf, 1]),
    dict(widths=[1e-300]*3), dict(native_mean=[1, np.nan]),
    dict(loss=[[1, .1], [0, 1]]), dict(loss=[[1, -2], [-2, 1]]),
    dict(loss=[[1, -1], [-1, 1]]), dict(vacuum_faces=[1]*6),
    dict(vacuum_faces=[False]*5), dict(subdivisions=True), dict(subdivisions=(2, 3, 0)),
    dict(alpha=0), dict(alpha=[.5]), dict(require_positive=1), dict(rtol=0),
    dict(condition_limit=1), dict(prescribed_net_inward_integrated=np.ones((6, 2))),
    dict(net_inward_sign_moments=np.ones((6, 2, 2))),
])
def test_invalid_inputs(kwargs):
    args = dict(diffusion=D, loss=R, widths=L, native_mean=MU,
                prescribed_net_inward_integrated=np.zeros((6, 2)), vacuum_faces=[True]*6,
                subdivisions=2)
    args.update(kwargs)
    with pytest.raises(ReflectorCuboidReferenceError):
        solve(**args)


def test_affine_requires_even_tangents():
    moments = np.zeros((6, 2, 2)); moments[0, 0] = .1
    with pytest.raises(ReflectorCuboidReferenceError, match='even tangential'):
        solve(D, R, L, MU, I, [False]*6, subdivisions=3, net_inward_sign_moments=moments)


def test_non_column_dominant_mmatrix_and_tiny_group_mean():
    result = solve([1, .8], [[1, -2], [-1e-13, 1]], [1, 2, 3], [1, 1e-12],
                   np.zeros((6, 2)), [True]*6, subdivisions=4, require_positive=False)
    assert_allclose(result.mean, [1, 1e-12], rtol=1e-10)
