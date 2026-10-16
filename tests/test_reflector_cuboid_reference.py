"""Independent operator, continuum refinement, and diagnostic contract checks."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ndgpu.reflector_cuboid_reference import (
    NonpositiveReflectorCuboidError, ReflectorCuboidReferenceError,
    solve_reflector_cuboid_reference as solve,
)


D = np.array([1.2, .7])
R = np.array([[.8, -.12], [-.2, .5]])
L = np.array([1.3, 2.1, .8])
M = np.array([3., 4.])
I = np.array([[.03, -.01], [.02, .04], [-.02, .01],
              [.01, -.03], [.04, .02], [-.01, .05]])


@pytest.mark.parametrize('affine', [False, True])
def test_independent_uncondensed_fv_operator(affine):
    counts = (2, 4, 6) if affine else (2, 3, 4)
    moments = (np.arange(24).reshape(6, 2, 2)-11)*.001 if affine else np.zeros((6, 2, 2))
    result = solve(D, R, L, M, I, subdivisions=counts,
                   net_inward_sign_moments=moments)
    nx, ny, nz = counts
    n = nx*ny*nz
    h, volume = L/counts, np.prod(L)
    cv = volume/n
    matrix = np.zeros((2*n, 2*n))
    rhs = np.tile(result.delta[:, None], (1, n))
    # Assemble integrated group/cell equations with explicit edge conductances.
    face_spec = ((0, 0), (1, 0), (0, nx-1), (1, ny-1), (2, 0), (2, nz-1))
    areas = volume/L[[0, 1, 0, 1, 2, 2]]
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                cell = x+nx*(y+ny*z)
                pos = (x, y, z)
                for g in range(2):
                    row = g*n+cell
                    for other in range(2):
                        matrix[row, other*n+cell] += cv*R[g, other]
                    for axis, stride in enumerate((1, nx, nx*ny)):
                        for direction in (-1, 1):
                            if 0 <= pos[axis]+direction < counts[axis]:
                                conductance = cv*D[g]/h[axis]**2
                                matrix[row, row] += conductance
                                matrix[row, row+direction*stride] -= conductance
                    for side, (axis, coordinate) in enumerate(face_spec):
                        if pos[axis] == coordinate:
                            prescribed = I[side, g]/areas[side]
                            tangents = [a for a in range(3) if a != axis]
                            for tangent, a in enumerate(tangents):
                                coordinate = (pos[a]+.5)*h[a]-L[a]/2
                                prescribed += 4*moments[side, tangent, g]*coordinate/(areas[side]*L[a])
                            rhs[g, cell] += prescribed/h[axis]
    expected = np.linalg.solve(matrix, rhs.ravel()*cv).reshape(2, nz, ny, nx)
    assert_allclose(result.cell_flux, expected, rtol=3e-14, atol=3e-14)
    assert_allclose(result.mean, M, rtol=1e-14)
    assert_allclose(result.delta, R@M-I.sum(axis=0)/volume)
    for side, axis in enumerate((0, 1, 0, 1, 2, 2)):
        assert_allclose(result.face_patch_areas[side].sum(), areas[side])
        assert_allclose(result.face_patch_currents[side].sum(axis=(1, 2)), I[side])
        assert_allclose(result.face_means[side], result.face_traces[side].mean(axis=(1, 2)))
    assert result.diagnostics['residual_relative'] < 1e-12
    assert result.diagnostics['balance_relative'] < 1e-13
    assert result.diagnostics['uniform_block_in_condition_gate']


def test_scalar_cosh_separable_refinement():
    diffusion, reaction = 1.1, .6
    lengths = np.array([1.3, 2.1, .8])
    amplitude = np.array([.08, -.03, .05])
    rate = np.sqrt(reaction/diffusion)
    face_area = np.prod(lengths)/lengths
    density = diffusion*amplitude*rate*np.sinh(rate*lengths/2)
    currents = (face_area*density)[[0, 1, 0, 1, 2, 2], None]
    errors, face_errors = [], []
    for n in (8, 16, 32):
        result = solve([diffusion], [[reaction]], lengths, [2.], currents, subdivisions=n)
        profiles = [a*(np.cosh(rate*((np.arange(n)+.5)*width/n-width/2))-
                      np.sinh(rate*width/2)/(rate*width/2))
                    for a, width in zip(amplitude, lengths)]
        exact = 2+profiles[0][None, None, :]+profiles[1][None, :, None]+profiles[2][:, None, None]
        errors.append(np.max(abs(result.cell_flux[0]-exact)))
        exact_faces = (2+amplitude*(np.cosh(rate*lengths/2)-
                         np.sinh(rate*lengths/2)/(rate*lengths/2)))[[0, 1, 0, 1, 2, 2]]
        face_errors.append(np.max(abs(result.face_means[:, 0]-exact_faces)))
    # Max-norm samples move toward the boundary as n increases; the final
    # ratio approaches four from below.
    assert np.all(np.array(errors[:-1])/errors[1:] > 3.5)
    assert errors[-2]/errors[-1] > 3.8
    assert np.all(np.array(face_errors[:-1])/face_errors[1:] > 3.8)
    assert errors[-1] < 1e-5


def test_manufactured_constant_and_immutable_result():
    result = solve(D, R, L, M, np.zeros((6, 2)), subdivisions=(1, 3, 2))
    assert_allclose(result.cell_flux, np.broadcast_to(M[:, None, None, None], (2, 2, 3, 1)), atol=1e-14)
    assert_allclose(result.delta, R@M)
    arrays = [result.cell_flux, result.face_means, result.delta, result.mean,
              *result.face_traces, *result.face_patch_areas, *result.face_patch_currents,
              result.diagnostics['recovered_net_inward_integrated']]
    for array in arrays:
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        result.diagnostics['changed'] = True
    assert result.diagnostics['physical_admission'] is False


def test_decoupled_limit_and_linearity():
    diagonal = np.diag([.8, .5])
    coupled = solve(D, diagonal, L, M, I, subdivisions=(3, 2, 4))
    for group in range(2):
        scalar = solve(D[group:group+1], diagonal[group:group+1, group:group+1],
                       L, M[group:group+1], I[:, group:group+1], subdivisions=(3, 2, 4))
        assert_allclose(coupled.cell_flux[group], scalar.cell_flux[0], atol=2e-14)
    a = solve(D, R, L, M, I, subdivisions=3)
    b = solve(D, R, L, M*.7, -I*.3, subdivisions=3)
    total = solve(D, R, L, M*1.7, I*.7, subdivisions=3)
    assert_allclose(total.cell_flux, a.cell_flux+b.cell_flux, atol=2e-14)
    assert_allclose(total.face_means, a.face_means+b.face_means, atol=2e-14)


def test_negative_field_retained_without_admission():
    currents = np.array([[-20.], [0.], [20.], [0.], [0.], [0.]])
    with pytest.raises(NonpositiveReflectorCuboidError) as caught:
        solve([1.], [[.4]], [1., 1., 1.], [1.], currents)
    result = caught.value.result
    diagnostic = solve([1.], [[.4]], [1., 1., 1.], [1.], currents, require_positive=False)
    assert result.cell_flux.min() < 0
    assert result.face_means.min() < 0
    assert_allclose(result.cell_flux, diagnostic.cell_flux, rtol=0, atol=0)
    assert_allclose(result.face_means, diagnostic.face_means, rtol=0, atol=0)
    assert_allclose(result.mean, [1.], atol=2e-15)
    assert not result.diagnostics['full_field_positive']
    assert not result.diagnostics['physical_admission']


@pytest.mark.parametrize('kwargs', [
    {'loss': [[1., -1.], [-1., 1.]]},
    {'loss': [[1., -2.], [-2., 1.]]},
    {'loss': [[1., .1], [0., 1.]]},
    {'diffusion': [0., 1.]}, {'native_mean': [1., 0.]},
    {'widths': [1., np.inf, 2.]}, {'net_inward_integrated': np.zeros((2, 6))},
    {'subdivisions': True}, {'subdivisions': (2, 2., 3)},
    {'subdivisions': (2, 0, 3)}, {'subdivisions': (2, 3)},
    {'condition_limit': .9}, {'rtol': 0}, {'rtol': np.nan},
    {'require_positive': 'yes'}, {'diffusion': np.array([1+1j, 2])},
])
def test_invalid_and_singular_rejection(kwargs):
    args = dict(diffusion=D, loss=R, widths=L, native_mean=M, net_inward_integrated=I)
    args.update(kwargs)
    with pytest.raises(ReflectorCuboidReferenceError):
        solve(**args)


def test_condition_gate_covers_uniform_and_all_nonconstant_modes():
    # Fixed mean does not exempt a nearly singular uniform block.
    with pytest.raises(ReflectorCuboidReferenceError, match='condition_limit'):
        solve([1., 1.], np.diag([1e-14, 1.]), L, M, I, condition_limit=1e12)
    with pytest.raises(ReflectorCuboidReferenceError, match='condition_limit'):
        solve([1.], [[1.]], [1., 1., 1.], [1.], np.zeros((6, 1)),
              subdivisions=8, condition_limit=10)


def test_zero_and_tiny_currents_do_not_fail_trace_cancellation_guard():
    currents = np.zeros((6, 2))
    currents[0, 0] = 1e-100
    result = solve(D, R, L, M, currents, subdivisions=(2, 3, 4))
    assert result.diagnostics['trace_current_relative'] < 1e-14
    assert np.all(np.isfinite(result.face_means))
    assert_allclose(result.face_patch_currents[0].sum(axis=(1, 2)), currents[0], rtol=1e-14, atol=0)


AFFINE_MOMENTS = (np.arange(24).reshape(6, 2, 2)-11)*.002


def test_affine_exact_patch_half_and_quadrant_integrals():
    result = solve(D, R, L, M, I, subdivisions=(2, 4, 6),
                   net_inward_sign_moments=AFFINE_MOMENTS)
    tangents = ((1, 2), (0, 2), (1, 2), (0, 2), (0, 1), (0, 1))
    counts = (2, 4, 6)
    recovered_moments = []
    for side, tangent_axes in enumerate(tangents):
        current = result.face_patch_currents[side]
        signs = result.diagnostics['face_patch_signs'][side]
        u, v = tangent_axes
        # Independent integral of the affine polynomial over each rectangle.
        ub = np.linspace(-L[u]/2, L[u]/2, counts[u]+1)
        vb = np.linspace(-L[v]/2, L[v]/2, counts[v]+1)
        area = L[u]*L[v]
        for j in range(counts[v]):
            for i in range(counts[u]):
                du, dv = ub[i+1]-ub[i], vb[j+1]-vb[j]
                expected = (I[side]*du*dv/area
                    +2*AFFINE_MOMENTS[side, 0]*(ub[i+1]**2-ub[i]**2)*dv/(area*L[u])
                    +2*AFFINE_MOMENTS[side, 1]*(vb[j+1]**2-vb[j]**2)*du/(area*L[v]))
                assert_allclose(current[:, j, i], expected, rtol=1e-13, atol=1e-17)
                assert signs[0, j, i] == (-1 if i < counts[u]//2 else 1)
                assert signs[1, j, i] == (-1 if j < counts[v]//2 else 1)
        for tangent in range(2):
            plus_half = current[:, signs[tangent] > 0].sum(axis=1)
            assert_allclose(plus_half, (I[side]+AFFINE_MOMENTS[side, tangent])/2, atol=2e-17)
        for su in (-1, 1):
            for sv in (-1, 1):
                quadrant = current[:, (signs[0] == su) & (signs[1] == sv)].sum(axis=1)
                expected = (I[side]+su*AFFINE_MOMENTS[side, 0]+sv*AFFINE_MOMENTS[side, 1])/4
                assert_allclose(quadrant, expected, atol=2e-17)
        recovered_moments.append([(current*sign).sum(axis=(1, 2)) for sign in signs])
        with pytest.raises(ValueError):
            signs.setflags(write=True)
    assert_allclose(recovered_moments, AFFINE_MOMENTS, atol=3e-17)
    assert_allclose(result.diagnostics['recovered_net_inward_sign_moments'], AFFINE_MOMENTS, atol=3e-17)
    assert_allclose(result.delta, R@M-I.sum(axis=0)/np.prod(L))
    assert_allclose(result.mean, M, atol=2e-15)
    assert result.diagnostics['boundary_current_profile_model'] == 'affine_sign_moments'


@pytest.mark.parametrize('axis', range(3))
def test_affine_global_axis_reflection(axis):
    counts = (2, 4, 6)
    base = solve(D, R, L, M, I, subdivisions=counts, net_inward_sign_moments=AFFINE_MOMENTS)
    permutation = np.arange(6)
    minus, plus = ((0, 2), (1, 3), (4, 5))[axis]
    permutation[minus], permutation[plus] = plus, minus
    reflected = AFFINE_MOMENTS[permutation].copy()
    for side, tangent_axes in enumerate(((1, 2), (0, 2), (1, 2), (0, 2), (0, 1), (0, 1))):
        for tangent, a in enumerate(tangent_axes):
            if a == axis:
                reflected[side, tangent] *= -1
    result = solve(D, R, L, M, I[permutation], subdivisions=counts,
                   net_inward_sign_moments=reflected)
    assert_allclose(result.cell_flux, np.flip(base.cell_flux, axis=3-axis), atol=3e-14)
    assert_allclose(result.face_means, base.face_means[permutation], atol=3e-14)


def test_zero_moments_preserve_uniform_arrays_bitwise():
    uniform = solve(D, R, L, M, I, subdivisions=(2, 3, 4))
    zero = solve(D, R, L, M, I, subdivisions=(2, 3, 4), net_inward_sign_moments=np.zeros((6, 2, 2)))
    for name in ('cell_flux', 'face_means', 'delta', 'mean'):
        np.testing.assert_array_equal(getattr(uniform, name), getattr(zero, name))
    for name in ('face_traces', 'face_patch_areas', 'face_patch_currents'):
        for a, b in zip(getattr(uniform, name), getattr(zero, name)):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('side,tangent', [(s, t) for s in range(6) for t in range(2)])
def test_reject_unresolved_nonzero_moment(side, tangent):
    moments = np.zeros((6, 2, 2))
    moments[side, tangent, 0] = .01
    axis = ((1, 2), (0, 2), (1, 2), (0, 2), (0, 1), (0, 1))[side][tangent]
    for unresolved in (1, 3):
        counts = [2, 4, 6]
        counts[axis] = unresolved
        with pytest.raises(ReflectorCuboidReferenceError, match='even tangential'):
            solve(D, R, L, M, I, subdivisions=counts, net_inward_sign_moments=moments)


def test_only_active_tangent_requires_even_count():
    moments = np.zeros((6, 2, 2))
    moments[0, 0] = [.01, -.02]  # W y tangent; x/z may stay odd.
    result = solve(D, R, L, M, I, subdivisions=(3, 2, 3), net_inward_sign_moments=moments)
    assert_allclose(result.diagnostics['recovered_net_inward_sign_moments'], moments, atol=2e-17)


@pytest.mark.parametrize('moments', [np.zeros((6, 2)), np.full((6, 2, 2), np.nan),
                                     np.full((6, 2, 2), 1j)])
def test_invalid_moments(moments):
    with pytest.raises(ReflectorCuboidReferenceError):
        solve(D, R, L, M, I, net_inward_sign_moments=moments)


def test_affine_negative_exception_retains_full_patch_profiles():
    moments = np.zeros((6, 2, 1))
    moments[0, 0, 0] = 30.
    args = ([1.], [[.4]], [1., 1., 1.], [1.], np.zeros((6, 1)))
    with pytest.raises(NonpositiveReflectorCuboidError) as caught:
        solve(*args, subdivisions=(2, 4, 6), net_inward_sign_moments=moments)
    diagnostic = solve(*args, subdivisions=(2, 4, 6), net_inward_sign_moments=moments, require_positive=False)
    assert diagnostic.cell_flux.min() < 0
    np.testing.assert_array_equal(caught.value.result.cell_flux, diagnostic.cell_flux)
    for name in ('face_traces', 'face_patch_currents'):
        for actual, expected in zip(getattr(caught.value.result, name), getattr(diagnostic, name)):
            np.testing.assert_array_equal(actual, expected)
    assert not diagnostic.diagnostics['physical_admission']
