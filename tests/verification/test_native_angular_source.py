"""Mathematical verification only; no transport or material qualification."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.special import eval_legendre

from garnet.moment_materials import MomentMaterial
from garnet.native_angular_source import NativeAngularSource, ProductQuadrature, real_harmonics


def material(matrices, orientation='to-major', edges=(20., 1., 0.)):
    groups = len(edges)-1
    production = {l: matrix.T if orientation == 'from-major' else matrix
                  for l, matrix in enumerate(matrices)}
    return MomentMaterial(np.ones(groups), np.zeros(groups), np.zeros(groups),
                          np.zeros(groups), {}, production, {}, {}, {},
                          {'orientation_contract': orientation, 'energy_bounds_mev': edges}, {})


def source(lmax=1, orientation='to-major'):
    matrices = [np.array([[.3, .2], [.1, .4]])]
    matrices += [np.array([[-.07, .03], [.09, -.11]]) * l for l in range(1, lmax+1)]
    return NativeAngularSource(ProductQuadrature(lmax), (material(matrices, orientation),), np.array([0, 0]))


@pytest.mark.parametrize('lmax', [0, 1, 3, 7])
def test_gram_addition_and_isotropic_moments(lmax):
    q = ProductQuadrature(lmax)
    basis = q.harmonics
    np.testing.assert_allclose(basis.T @ (q.weights[:, None]*basis),
                               4*np.pi*np.eye((lmax+1)**2), atol=2e-12, rtol=0)
    np.testing.assert_array_equal(basis[:, 0], 1)
    np.testing.assert_allclose(q.weights.sum(), 4*np.pi, atol=2e-14, rtol=0)
    # Arbitrary independent unit vectors exercise the continuous addition theorem.
    vectors = np.array([[1., 2., -3.], [-.5, 4., 1.], [0., 0., 1.]])
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    h = real_harmonics(vectors, lmax)
    for l in range(lmax+1):
        block = h[:, l*l:(l+1)**2]
        np.testing.assert_allclose(block @ block.T, (2*l+1)*eval_legendre(l, vectors @ vectors.T),
                                   atol=2e-12, rtol=0)
    op = source(lmax)
    moments = op.moments(np.ones((2, q.n_angles, 2)))
    np.testing.assert_allclose(moments[:, 0], 4*np.pi, atol=2e-14)
    np.testing.assert_allclose(moments[:, 1:], 0, atol=1e-13)


def test_p0_p1_two_group_analytic_oracle_and_particle_production():
    op = source()
    q = op.quadrature
    a = np.array([2., 3.])
    b = np.array([[.4, -.1], [.2, .5], [-.3, .7]])
    flux = np.broadcast_to(a + q.directions @ b, (2, q.n_angles, 2)).copy()
    expected = a @ op.production[0][0].T + (q.directions @ b) @ op.production[0][1].T
    result = op.scatter(flux)
    np.testing.assert_allclose(result, np.broadcast_to(expected, result.shape), atol=8e-16)
    integrated = np.einsum('n,cng->cg', q.weights, result)
    np.testing.assert_allclose(integrated, op.moments(flux)[:, 0] @ op.production[0][0].T, atol=2e-14)


def test_orientation_equivalence_signed_higher_orders_and_heterogeneous_cells():
    op = source(3)
    transpose = source(3, 'from-major')
    q = op.quadrature
    coefficients = np.arange(32, dtype=float).reshape(16, 2)/10 - 1
    flux = np.broadcast_to(q.harmonics @ coefficients, (2, q.n_angles, 2)).copy()
    expected = q.harmonics @ np.stack([op.production[0][l] @ coefficients[a]
                                      for a, l in enumerate(q.orders)])
    np.testing.assert_allclose(op.scatter(flux)[0], expected, atol=2e-14)
    np.testing.assert_allclose(op.scatter(flux), transpose.scatter(flux), atol=0, rtol=0)
    # A second material scales every signed coefficient without any clipping.
    second = material([2*matrix for matrix in op.production[0]])
    heterogeneous = NativeAngularSource(q, (op.materials[0], second), np.array([0, 1]))
    result = heterogeneous.scatter(flux)
    np.testing.assert_allclose(result[1], 2*result[0], atol=2e-14)
    np.testing.assert_array_equal(heterogeneous.production[0], op.production[0])
    assert len(heterogeneous.production) == 2  # matrices per material, never per cell
    assert np.any(result < 0)


@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
def test_exact_reflection_permutations(axis):
    q = ProductQuadrature(7)
    permutation = q.reflection(axis)
    reflected = q.directions.copy()
    reflected[:, 'xyz'.index(axis)] *= -1
    np.testing.assert_allclose(q.directions[permutation], reflected, atol=2e-15, rtol=0)
    np.testing.assert_array_equal(permutation[permutation], np.arange(q.n_angles))
    np.testing.assert_array_equal(q.weights[permutation], q.weights)


def test_immutability():
    op = source()
    arrays = [op.quadrature.directions, op.quadrature.weights, op.quadrature.harmonics,
              op.quadrature.orders, op.cell_material_ids, op.production[0], op.energy_bounds_mev]
    for array in arrays:
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(AttributeError):
        op.quadrature.lmax = 7


@pytest.mark.parametrize('args', [(-1,), (8,), (True,), (1, 1, 4), (1, 2, 3), (1, 2, 2)])
def test_bad_quadrature(args):
    with pytest.raises(ValueError):
        ProductQuadrature(*args)


def test_bad_harmonics():
    for vectors in (np.zeros((2, 3)), [[1, 0, np.nan]], [[1, 0]], [[1j, 0, 0]]):
        with pytest.raises(ValueError):
            real_harmonics(vectors, 1)
    with pytest.raises(ValueError):
        ProductQuadrature(0).reflection('w')


def test_invalid_materials_and_ids():
    op = source()
    q, m = op.quadrature, op.materials[0]
    for ids in ([0.], [-1], [1], [], [[0]], [True]):
        with pytest.raises(ValueError):
            NativeAngularSource(q, (m,), np.asarray(ids))
    for bad in (replace(m, production={0: m.production[0]}),
                replace(m, production={0: np.ones((2, 3)), 1: m.production[1]}),
                replace(m, production={0: np.full((2, 2), np.nan), 1: m.production[1]}),
                replace(m, provenance={}),
                replace(m, provenance={'orientation_contract': 'guess', 'energy_bounds_mev': [20, 1, 0]}),
                replace(m, provenance={'orientation_contract': 'to-major', 'energy_bounds_mev': [20, 20, 0]})):
        with pytest.raises(ValueError):
            NativeAngularSource(q, (bad,), np.array([0]))
    other = material([np.ones((2, 2))]*2, edges=(10., 1., 0.))
    with pytest.raises(ValueError, match='energy bounds mismatch'):
        NativeAngularSource(q, (m, other), np.array([0, 1]))
    other = material([np.ones((1, 1))]*2, edges=(20., 0.))
    with pytest.raises(ValueError, match='group count mismatch'):
        NativeAngularSource(q, (m, other), np.array([0, 1]))


def test_invalid_flux_and_allocation_budgets():
    op = source()
    shape = (2, op.quadrature.n_angles, 2)
    for flux in (np.ones((2, 2)), np.full(shape, np.inf), np.ones(shape, dtype=complex)):
        with pytest.raises(ValueError):
            op.scatter(flux)
    with pytest.raises(ValueError, match='allocation estimate'):
        ProductQuadrature(7, max_bytes=1)
    with pytest.raises(ValueError, match='allocation estimate'):
        NativeAngularSource(op.quadrature, op.materials, np.array([0]), max_bytes=1)
    small = NativeAngularSource(ProductQuadrature(0),
                               (material([np.ones((1, 1))], edges=(20., 0.)),),
                               np.array([0]), max_bytes=50)
    with pytest.raises(ValueError, match='allocation estimate'):
        small.scatter(np.ones((1, 2, 1)))
