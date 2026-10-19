import numpy as np
import pytest

from ndgpu.reflector_interface_factors import extend_reflector_interface_factors as extend


def fixture():
    base = np.ones((3, 6, 2))
    base[0, 2] = [1.2, .8]
    fuel = np.array([True, False, False])
    neighbors = np.full((3, 6), -1)
    neighbors[0, 2], neighbors[1, 0] = 1, 0
    neighbors[1, 2], neighbors[2, 0] = 2, 1
    hom = np.ones_like(base)*2
    het = np.ones_like(base)*3
    changes = np.zeros_like(base)
    return base, fuel, neighbors, het, hom, changes, changes.copy()


def test_all_reflector_owners_and_frozen_fuel():
    args = fixture()
    before = [a.copy() for a in args]
    result = extend(*args, order_tolerance=.005)
    assert np.array_equal(result[0], args[0][0])
    assert np.all(result[1, [0, 2]] == 1.5)
    assert np.all(result[2, 0] == 1.5)
    assert np.all(result[args[2] < 0] == 1)
    assert not result.flags.writeable
    for a, b in zip(args, before):
        assert np.array_equal(a, b)


def test_negative_unused_auxiliary_faces_retained():
    args = fixture()
    args[4][args[2] < 0] = -5
    args[3][args[2] < 0] = 0
    result = extend(*args, order_tolerance=.005)
    assert np.all(result[args[2] < 0] == 1)
    assert np.all(args[4][args[2] < 0] == -5)


@pytest.mark.parametrize('value', [0, -1, np.inf, np.nan])
@pytest.mark.parametrize('index', [3, 4])
def test_never_skips_bad_internal_reflector_trace(index, value):
    args = fixture()
    args[index][2, 0, 1] = value
    with pytest.raises(ValueError):
        extend(*args, order_tolerance=.005)


def test_order_failure():
    args = fixture()
    args[5][1, 2, 0] = .005001
    with pytest.raises(ValueError, match='order-converged'):
        extend(*args, order_tolerance=.005)


def test_small_denominator_factor_error_cannot_hide_behind_native_mean():
    args = fixture()
    args[5][:] = .0001
    args[6][1, 0, 0] = .01
    with pytest.raises(ValueError, match='order-converged'):
        extend(*args, order_tolerance=.005)


def test_baseline_scope_failure():
    args = fixture()
    args[0][1, 0, 0] = 1.1
    with pytest.raises(ValueError, match='fuel-only'):
        extend(*args, order_tolerance=.005)


def test_rejects_nonreciprocal_geometry():
    args = fixture()
    args[2][2, 0] = -1
    with pytest.raises(ValueError, match='reciprocal'):
        extend(*args, order_tolerance=.005)


def test_asymmetric_factors_preserve_corrected_flux_and_opposite_current():
    # Independent two-half-cell diffusion interface law. HET is continuous;
    # HOM need not be. Keeping each owner factor is essential.
    args = fixture()
    args[3][0, 2] = [3, 4]
    args[4][0, 2] = args[3][0, 2]/args[0][0, 2]
    args[3][1, 0] = [3, 4]
    args[4][1, 0] = [1, 8]
    f = extend(*args, order_tolerance=.005)
    left, right = f[0, 2], f[1, 0]
    hleft, hright, conductance = .3, .7, 2.
    current = np.array([.2, -.4])
    surface_left = np.array([3, 4])/left
    surface_right = np.array([3, 4])/right
    cell_left = surface_left + hleft*current/conductance
    cell_right = surface_right - hright*current/conductance
    recovered = (left*cell_left-right*cell_right)/(left*hleft/conductance+right*hright/conductance)
    np.testing.assert_allclose(recovered, current, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(left*(cell_left-hleft*recovered/conductance),
                               right*(cell_right+hright*recovered/conductance))
