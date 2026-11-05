"""Synthetic native-P0 reaction identities; no cohort or physical qualification."""
import numpy as np
import pytest

from garnet.nem import NEMDiffusionSolver, NEMSolverError
from garnet.nem_higher_order import NEMHigherOrderDiffusionSolver
from test_nem import tensor_grid, PAIRS


def grid(boundary='reflective'):
    return tensor_grid(([0., 1.], [0., 1.], [0., 1.]), boundary)


def native(solver=NEMDiffusionSolver, *, mesh=None, **overrides):
    args = dict(D=[.7, .4], sigma_a=None, sigma_s=None,
                nu_sigma_f=[0., 0.], chi=[0., 0.],
                native_total=np.array([[1., 1.8]]),
                native_scattering_production=np.array([[[.2, 1.1], [.1, .3]]]))
    args.update(overrides)
    return solver(grid() if mesh is None else mesh, **args)


def test_native_direct_reaction_local_blocks_and_signed_compatibility_diagnostic():
    op = native()
    # Independent emitted/source equations, including asymmetric and nonzero
    # self-scattering. In particular R00 is T0-SP00, not T0 or signed_a0.
    expected = np.array([[.8, -.1], [-1.1, 1.5]])
    np.testing.assert_allclose(op.R[0], expected, atol=2e-16, rtol=0)
    np.testing.assert_allclose(op.signed_reduced_absorption, [[-.3, 1.4]])
    np.testing.assert_array_equal(op.sigma_a, op.signed_reduced_absorption)
    assert op.signed_reduced_absorption[0, 0] < 0
    np.testing.assert_array_equal(np.diag(op.sigma_s[0]), [0., 0.])
    for axis in range(3):
        for order, shift in enumerate((10, 42)):
            response = expected+np.diag(shift*np.array([.7, .4]))
            np.testing.assert_allclose(op._local_inverse[0, axis, order], np.linalg.inv(response), atol=2e-16)
    np.testing.assert_allclose(op._loss_matrix[:2, :2].toarray(), expected, atol=2e-16)
    means = np.array([2., 3.])
    result = op.solve_fixed_source(expected @ means)
    np.testing.assert_allclose(result.means[0], means, atol=2e-14)
    np.testing.assert_allclose(result.currents, 0., atol=2e-14)
    assert result.diagnostics['reaction_mode'] == 'native_p0_unqualified'
    np.testing.assert_array_equal(result.diagnostics['native_total'], op.native_total)
    np.testing.assert_array_equal(result.diagnostics['native_scattering_production'], op.native_scattering_production)
    assert 'not physical ABS' in result.diagnostics['reduced_absorption_semantics']


def test_default_negative_absorption_still_rejected():
    with pytest.raises(ValueError, match='nonnegative'):
        NEMDiffusionSolver(grid(), [.7, .4], [-.3, 1.4], [[0., 1.1], [.1, 0.]], [0., 0.], [0., 0.])


@pytest.mark.parametrize('overrides', [
    dict(native_total=None), dict(native_scattering_production=None),
    dict(sigma_a=[.1, .2]), dict(sigma_s=[[0., .1], [.2, 0.]]),
    dict(native_total=[[0., 1.8]]), dict(native_total=[[np.inf, 1.8]]),
    dict(native_total=[[1., 1.8], [1., 1.8]]),
    dict(native_scattering_production=[[[-.1, .2], [.1, .3]]]),
    dict(native_scattering_production=[[[.2, np.nan], [.1, .3]]]),
    dict(native_scattering_production=np.ones((2, 3))),
    dict(native_scattering_production=[[[1., 0.], [0., .3]]]),
])
def test_partial_mixed_bad_shape_sign_finiteness_and_loss_diagonal_rejected(overrides):
    with pytest.raises(ValueError):
        native(**overrides)


@pytest.mark.parametrize('solver', [NEMDiffusionSolver, NEMHigherOrderDiffusionSolver])
def test_raw_copies_caller_mutation_isolation_and_no_coefficient_changes(solver):
    total = np.array([[1., 1.8]])
    production = np.array([[[.2, 1.1], [.1, .3]]])
    before_t, before_p = total.copy(), production.copy()
    op = native(solver, native_total=total, native_scattering_production=production)
    np.testing.assert_array_equal(total, before_t)
    np.testing.assert_array_equal(production, before_p)
    signature = op._signature()
    total[:] = 999.
    production[:] = 999.
    np.testing.assert_array_equal(op.native_total, before_t)
    np.testing.assert_array_equal(op.native_scattering_production, before_p)
    assert op._signature() == signature
    for value in (op.native_total, op.native_scattering_production, op.signed_reduced_absorption):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)
    op._assert_unchanged()


@pytest.mark.parametrize('field', ['native_total', 'native_scattering_production', 'signed_reduced_absorption', 'reaction_mode'])
def test_cache_rejects_native_attribute_replacement(field):
    op = native()
    value = getattr(op, field)
    setattr(op, field, 'absorption_scattering' if field == 'reaction_mode' else value.copy()+.01)
    with pytest.raises(ValueError, match='changed after cached'):
        op.solve_fixed_source([1., 1.])


def test_identical_loss_operators_have_distinct_native_raw_and_mode_cache_identities():
    total = np.array([[1., 1.8]])
    production = np.array([[[.2, .1], [.1, .3]]])
    native_op = native(native_total=total, native_scattering_production=production)
    conventional = NEMDiffusionSolver(grid(), [.7, .4], [.7, 1.4], [[0., .1], [.1, 0.]], [0., 0.], [0., 0.])
    np.testing.assert_allclose(native_op.R, conventional.R, atol=2e-16)
    assert native_op._signature() != conventional._signature()
    # Add exactly representable self-scattering and total increments together:
    # the actual loss is unchanged, but raw input provenance remains distinct.
    diagonal = np.array([[[.5, 0.], [0., .5]]])
    changed = native(native_total=total+.5, native_scattering_production=production+diagonal)
    np.testing.assert_allclose(changed.R, native_op.R, atol=2e-16)
    assert changed._signature() != native_op._signature()
    assert conventional._backend_evidence['reaction_mode'] == 'absorption_scattering'


@pytest.mark.parametrize('solver', [NEMDiffusionSolver, NEMHigherOrderDiffusionSolver])
def test_manufactured_quadratic_reconstruction_uses_native_R_in_all_moments(solver):
    mesh = grid('vacuum')
    op = native(solver, mesh=mesh)
    reaction = np.array([[.8, -.1], [-1.1, 1.5]])
    diffusion = np.array([.7, .4])
    linear = np.array([[.12, -.07], [.08, .11], [-.02, .03]])
    quadratic = np.array([[.02, .03], [-.01, .01], [.015, -.01]])
    center = np.full(3, .5)
    means = np.array([2., 1.5])+center @ linear+(center**2+1/12) @ quadratic
    moments = np.zeros((1, 3, op.n_moments, 2))
    moments[0, :, 0] = (linear+2*center[:, None]*quadratic)/12
    moments[0, :, 1] = quadratic/60
    currents = np.empty((len(mesh.face_nodes), 2))
    faces = np.empty((1, 6, 2))
    for face, axis in enumerate(mesh.face_axes):
        x = mesh.face_planes[face]
        currents[face] = -diffusion*(linear[axis]+2*x*quadratic[axis])
    for axis, (minus, plus) in enumerate(PAIRS):
        for side, x in ((minus, 0.), (plus, 1.)):
            faces[0, side] = means+linear[axis]*(x-.5)+quadratic[axis]*(x*x-.25-1/12)
    mean_source = reaction @ means-2*diffusion*quadratic.sum(axis=0)
    moment_source = np.empty_like(moments)
    for axis in range(3):
        for order in range(op.n_moments):
            moment_source[0, axis, order] = reaction @ moments[0, axis, order]
    drive = (mesh.normal_signs[mesh.boundary_sides, None]*currents[mesh.boundary_faces]
             -mesh.boundary_alpha[:, None]*faces[mesh.boundary_nodes, mesh.boundary_sides])
    result = op.solve_fixed_source(mean_source, moment_source, drive)
    np.testing.assert_allclose(result.means[0], means, atol=2e-12, rtol=1e-12)
    np.testing.assert_allclose(result.moments, moments, atol=2e-12, rtol=1e-10)
    np.testing.assert_allclose(result.currents, currents, atol=2e-12, rtol=1e-10)
    np.testing.assert_allclose(result.side_face_means, faces, atol=2e-12, rtol=1e-12)
    assert result.diagnostics['reaction_mode'] == 'native_p0_unqualified'


def test_full_F_guard_and_native_chi_are_unchanged():
    fission = np.array([[[.07, .12], [.03, .08]]])
    op = native(nu_sigma_f=[.1, .2], chi=[.6, .4], fission_matrix=fission)
    np.testing.assert_array_equal(op.fission, fission)
    np.testing.assert_array_equal(op.chi, [[.6, .4]])
    with pytest.raises(ValueError, match='column sums'):
        native(nu_sigma_f=[.1, .3], chi=[.6, .4], fission_matrix=fission)
    with pytest.raises(ValueError, match='nonnegative'):
        native(nu_sigma_f=[.1, .2], chi=[.6, .4], fission_matrix=[[.11, .12], [-.01, .08]])
    with pytest.raises(ValueError, match='chi must sum'):
        native(nu_sigma_f=[.1, .2], chi=[.3, .2], fission_matrix=fission)


def test_native_singular_and_condition_guards_remain_active():
    with pytest.raises(NEMSolverError, match='condition'):
        native(condition_limit=1.)
    # Positive diagonal alone is insufficient: reflective loss has a nullspace.
    with pytest.raises(NEMSolverError, match='singular|condition'):
        native(native_total=[[1., 1.]], native_scattering_production=[[[0., 1.], [1., 0.]]])
