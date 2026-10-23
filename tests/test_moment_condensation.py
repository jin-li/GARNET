"""Synthetic CPU dense references; no source data, GPU or transport claim."""
import numpy as np
import pytest
from garnet.moment_condensation import condense_moment_hierarchy, condense_moment_material
from garnet.moment_materials import MomentMaterial
from garnet.coupled_moments import CoupledMomentOperator, OrthogonalMesh


def fixture():
    total = np.array([1.3, 1.8, 2.2])
    losses = {0: np.array([[.6, -.12, -.04], [-.18, .9, -.07], [-.03, -.11, 1.1]]),
              1: np.array([[1.1, .08, -.03], [-.04, 1.5, .06], [.02, -.05, 1.9]]),
              2: np.array([[1.2, -.03, .02], [.06, 1.6, -.04], [-.01, .03, 2.]]),
              3: np.array([[1.25, .04, -.02], [-.05, 1.7, .03], [.03, -.01, 2.1]])}
    production = {l: (np.diag(total)-a).T for l, a in losses.items()}
    chi = np.array([.85, .3, .05])  # sum 1.2 must remain unnormalized
    nuf = np.array([.3, .4, .2])
    heat = np.array([20., 30., 17.])
    return total, production, chi, nuf, heat, losses


def condense(groups=((0, 1), (2,)), **kwargs):
    total, sp, chi, nuf, heat, _ = fixture()
    return condense_moment_hierarchy(total, sp, chi, nuf, groups, heat_production=heat, **kwargs)


def test_noncommuting_3_to_2_independent_dense_eigenpair_and_all_balances():
    total, sp, chi, nuf, heat, R = fixture()
    assert np.linalg.norm(R[1]@R[2]-R[2]@R[1]) > .01
    out = condense()
    eig, vectors = np.linalg.eig(np.linalg.solve(R[0], np.outer(chi, nuf)))
    index = np.argmax(eig.real)
    assert out.k_infinity == pytest.approx(eig[index].real, rel=1e-13)
    expected = vectors[:, index].real
    expected /= nuf@expected
    np.testing.assert_allclose(out.fine_weights[0], expected, rtol=1e-13)
    np.testing.assert_allclose(R[0]@out.fine_weights[0], chi/out.k_infinity, rtol=1e-13)
    np.testing.assert_allclose(out.loss[0]@out.weights[0], out.fission@out.weights[0]/out.k_infinity, rtol=1e-13)
    for l in range(4):
        np.testing.assert_allclose(out.aggregation@out.injections[l], np.eye(2), atol=1e-15)
        np.testing.assert_allclose(out.injections[l]@out.weights[l], out.fine_weights[l], atol=1e-15)
        if l:
            np.testing.assert_allclose(R[l]@out.fine_weights[l], l/(2*l+1)*out.fine_weights[l-1], rtol=1e-13)
            np.testing.assert_allclose(out.loss[l]@out.weights[l], l/(2*l+1)*out.weights[l-1], rtol=1e-13)
    np.testing.assert_allclose(out.heat_production, out.injections[0].T@heat)
    assert not np.allclose(out.heat_production, out.nu_sigma_f)
    assert out.chi.sum() == pytest.approx(1.2)
    assert not out.provenance['chi_renormalized']
    assert not out.provenance['physical_admission']
    assert max(out.diagnostics['fine_solve_backward_residual'].values()) < 1e-14
    assert max(out.diagnostics['coarse_hierarchy_backward_residual'].values()) < 1e-14


def test_diagonal_correction_matches_projected_loss_not_raw_scattering_projection():
    total, sp, *_ = fixture()
    out = condense()
    assert np.linalg.norm(out.diagonal_correction[1]) > .001
    for l in range(4):
        projected = out.aggregation@sp[l].T@out.injections[l]
        np.testing.assert_allclose(out.production[l].T, projected+np.diag(out.diagonal_correction[l]), atol=1e-15)
        np.testing.assert_allclose(np.diag(out.total)-out.production[l].T, out.loss[l], atol=1e-15)
    np.testing.assert_allclose(out.diagonal_correction[0], 0., atol=1e-15)


def test_identity_partition_preserves_original_coefficients_and_no_input_mutation():
    total, sp, chi, nuf, heat, R = fixture()
    before = {l: a.copy() for l, a in sp.items()}
    out = condense(((0,), (1,), (2,)))
    for l in range(4):
        np.testing.assert_allclose(out.loss[l], R[l], atol=1e-15)
        np.testing.assert_allclose(out.production[l], sp[l], atol=1e-15)
        np.testing.assert_array_equal(sp[l], before[l])
    for actual, expected in ((out.total, total), (out.chi, chi), (out.nu_sigma_f, nuf), (out.heat_production, heat)):
        np.testing.assert_allclose(actual, expected)
    with pytest.raises(ValueError):
        out.total.setflags(write=True)


def test_fine_and_coarse_permutations():
    total, sp, chi, nuf, heat, _ = fixture()
    out = condense()
    perm = np.array([2, 0, 1])
    reordered = condense_moment_hierarchy(total[perm], {l: a[perm][:, perm] for l, a in sp.items()},
                                         chi[perm], nuf[perm], ((1, 2), (0,)), heat_production=heat[perm])
    swapped = condense(((2,), (0, 1)))
    for l in range(4):
        np.testing.assert_allclose(reordered.loss[l], out.loss[l], atol=1e-15)
        np.testing.assert_allclose(reordered.fine_weights[l], out.fine_weights[l][perm], atol=1e-15)
        np.testing.assert_allclose(swapped.loss[l], out.loss[l][::-1, ::-1], atol=1e-15)
        np.testing.assert_allclose(swapped.weights[l], out.weights[l][::-1], atol=1e-15)
    assert reordered.k_infinity == pytest.approx(out.k_infinity)


def test_nested_composition_equals_direct_all_orders_and_heat():
    first = condense()
    nested = condense_moment_hierarchy(first.total, first.production, first.chi, first.nu_sigma_f,
                                       ((0, 1),), heat_production=first.heat_production)
    direct = condense(((0, 1, 2),))
    for l in range(4):
        np.testing.assert_allclose(nested.loss[l], direct.loss[l], atol=1e-15)
        np.testing.assert_allclose(nested.production[l], direct.production[l], atol=1e-15)
        np.testing.assert_allclose(first.injections[l]@nested.injections[l], direct.injections[l], atol=1e-15)
    for name in ('total', 'chi', 'nu_sigma_f', 'fission', 'heat_production'):
        np.testing.assert_allclose(getattr(nested, name), getattr(direct, name), atol=1e-14)
    assert nested.k_infinity == pytest.approx(direct.k_infinity, rel=1e-13)


def cancellation_fixture(epsilon=0.):
    total = np.ones(3)
    R = {l: np.eye(3) for l in range(4)}
    R[1] = np.diag([1., -1/(1-epsilon), 1.])
    return total, {l: (np.diag(total)-a).T for l, a in R.items()}, np.ones(3), np.ones(3)


@pytest.mark.parametrize('epsilon', [0., 1e-12])
def test_exact_and_scale_invariant_near_cancellation_fail_closed(epsilon):
    total, sp, chi, nuf = cancellation_fixture(epsilon)
    for scale in (1e-100, 1., 1e100):
        with pytest.raises(ValueError, match='cancellation'):
            condense_moment_hierarchy(total, sp, chi, nuf*scale, ((0, 1), (2,)))


def test_signed_higher_weights_and_coefficients_are_retained_without_clipping():
    total, sp, chi, nuf = cancellation_fixture(.5)
    out = condense_moment_hierarchy(total, sp, chi, nuf, ((0, 1), (2,)))
    assert out.injections[1][1, 0] < 0
    assert out.production[1][0, 0] < 0
    assert out.provenance['higher_weights_may_be_signed']
    assert not out.provenance['angular_qualification']
    assert not out.provenance['stability_qualification']


@pytest.mark.parametrize('epsilon', [0., 1e-13])
def test_well_conditioned_fine_but_singular_or_ill_conditioned_coarse_refused(epsilon):
    R0 = np.array([[1., 0., 0.], [0., 1., 1.], [.25, .75, .5+epsilon]])
    assert np.linalg.cond(R0) < 20
    total = np.ones(3)*5
    sp = {l: (np.diag(total)-(R0 if l == 0 else np.eye(3))).T for l in range(4)}
    with pytest.raises(ValueError, match='coarse R0'):
        condense_moment_hierarchy(total, sp, R0@np.ones(3), np.ones(3), ((0, 1), (2,)), condition_limit=1e10)


@pytest.mark.parametrize('bad', [np.zeros((3, 3)), np.diag([1., 1., 1e-14]), np.full((3, 3), np.nan)])
def test_singular_ill_conditioned_nonfinite_fine_refused(bad):
    total, sp, chi, nuf, *_ = fixture()
    sp[2] = (np.diag(total)-bad).T
    with pytest.raises(ValueError):
        condense_moment_hierarchy(total, sp, chi, nuf, ((0, 1), (2,)))


@pytest.mark.parametrize('groups', [((0, 1),), ((0, 1), (1, 2)), ((), (0, 1, 2)), ((True, 1), (2,)), ((0, 1), (3,))])
def test_partition_invalid_refused(groups):
    with pytest.raises(ValueError):
        condense(groups)


def test_nonfissile_nonpositive_scalar_and_invalid_threshold_refused():
    total, sp, chi, nuf, *_ = fixture()
    for c, f in ((np.zeros(3), nuf), (chi, np.zeros(3)), (-chi, nuf)):
        with pytest.raises(ValueError):
            condense_moment_hierarchy(total, sp, c, f, ((0, 1), (2,)))
    bad = dict(sp); bad[0] = (np.diag(total)-np.diag([-1., 1., 1.])).T
    with pytest.raises(ValueError):
        condense_moment_hierarchy(total, bad, chi, nuf, ((0, 1), (2,)))
    for key, value in (('cancellation_threshold', 0), ('cancellation_threshold', 1),
                       ('condition_limit', 1), ('backward_tolerance', float('nan'))):
        with pytest.raises(ValueError):
            condense(**{key: value})
    with pytest.raises(ValueError, match='Backward residual'):
        condense(backward_tolerance=1e-30)


def native(total, sp, chi, nuf, R):
    return MomentMaterial(total, chi, nuf, np.ones_like(total)*20., {}, sp, R, {}, {},
                          {'orientation_contract': 'from-major'}, {})


def test_opt_in_native_bridge_orientation_and_heat_are_explicit():
    total, sp, chi, nuf, heat, R = fixture()
    m = native(total, sp, chi, nuf, R)
    out = condense_moment_material(m, ((0, 1), (2,)))
    assert out.heat_production is None
    wrong = MomentMaterial(total, chi, nuf, heat, {}, sp, R, {}, {}, {'orientation_contract': 'to-major'}, {})
    with pytest.raises(ValueError, match='from-major'):
        condense_moment_material(wrong, ((0, 1), (2,)))
    badR = dict(R); badR[1] = badR[1].T
    wrong = native(total, sp, chi, nuf, badR)
    with pytest.raises(ValueError, match='identity'):
        condense_moment_material(wrong, ((0, 1), (2,)))


def fourier_eigenpair(material, family, B):
    op = CoupledMomentOperator(OrthogonalMesh([1.], [[0.]]), [material], family=family)
    g = material.n_groups
    A = op.reaction[0]+B**2*op.diffusion[0]
    F = np.column_stack([op.fission_action(np.eye(2*g)[:, i]) for i in range(2*g)])
    eig, vectors = np.linalg.eig(np.linalg.solve(A, F))
    index = np.argmax(eig.real)
    u = vectors[:, index].real
    phi0 = u[:g]-2/3*u[g:]
    u /= material.nu_sigma_f@phi0
    phi0 = u[:g]-2/3*u[g:]
    phi1 = -1j*B/3*np.linalg.solve(material.loss[1], u[:g])
    phi2 = u[g:]/3
    phi3 = -1j*B/7*np.linalg.solve(material.loss[3], u[g:])
    return eig[index].real, (phi0, phi1, phi2, phi3), op


@pytest.mark.parametrize('family', ['sp3', 'sdp1'])
def test_coupled_homogeneous_B_zero_leading_hierarchy_and_finite_B_negative_control(family):
    total, sp, chi, nuf, _, R = fixture()
    out = condense()
    fine = native(total, sp, chi, nuf, R)
    coarse = native(out.total, dict(out.production), out.chi, out.nu_sigma_f, dict(out.loss))
    k0, moments, op = fourier_eigenpair(fine, family, 0.)
    assert k0 == pytest.approx(out.k_infinity, rel=1e-13)
    np.testing.assert_allclose(moments[0], out.fine_weights[0], atol=1e-14)
    a = -op.c1[1, 0]; b = op.c2[1, 1]
    assert a/(3*b) == pytest.approx(2/5, rel=1e-15)
    B = 1e-4
    _, moments, _ = fourier_eigenpair(fine, family, B)
    for l in range(4):
        np.testing.assert_allclose(moments[l]/(-1j*B)**l, out.fine_weights[l], rtol=1e-6, atol=1e-8)
    kc0, _, _ = fourier_eigenpair(coarse, family, 0.)
    assert kc0 == pytest.approx(k0, rel=1e-13)
    kfinite, _, _ = fourier_eigenpair(fine, family, .4)
    kcfinite, _, _ = fourier_eigenpair(coarse, family, .4)
    assert abs(kfinite-kcfinite) > 1e-5
    assert not out.provenance['finite_buckling_preservation']
