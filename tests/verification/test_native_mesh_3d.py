"""Manufactured native FV3D algebra only; no physical qualification or pin solve."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.special import eval_legendre

from garnet.mesh import assemble_mesh, assemble_mesh_3d
from garnet.moment_materials import MomentMaterial
from garnet.native_angular_source import NativeAngularSource, ProductQuadrature
from garnet.native_mesh_streaming import NativeMeshStreaming
from garnet.native_mesh_transport import NativeMeshTransport
from garnet.native_angular_p0 import P0GalerkinTransport, _half_range
from garnet.native_angular_sweep import (
    StreamingBlockP0Transport, build_streaming_block, factor_streaming_block,
)


def material(lmax, index=0):
    # Signed higher moments, asymmetric transfer and deliberately raw chi.
    moments = {0: np.array([[.2, .07], [.11, .3]])}
    moments.update({l: np.array([[.013, -.004], [.008, -.009]])/l
                    for l in range(1, lmax+1)})
    return MomentMaterial(np.array([1.3, 1.7])+.1*index,
                          np.array([.9, .4]), np.array([.12, .31]),
                          np.array([2., 5.]), {}, moments, {}, {}, {},
                          {'orientation_contract': 'to-major',
                           'energy_bounds_mev': [20., 1., 0.]}, {})


def bind(mesh, lmax=1, quadrature=None, **kwargs):
    q = ProductQuadrature(lmax) if quadrature is None else quadrature
    source = NativeAngularSource(q, tuple(material(lmax, c) for c in range(mesh.n_cells)),
                                 np.arange(mesh.n_cells))
    return NativeMeshStreaming(mesh, source, **kwargs)


def prisms(two=True):
    # Base area=1; heights .7 and 1.6. Shared horizontal face area=1.
    levels = [0., .7, 2.3] if two else [0., .7]
    points = np.array([[x, y, z] for z in levels for x, y in [(0., 0.), (2., 0.), (0., 1.)]])
    cells = [tuple(range(3*c, 3*c+6)) for c in range(len(levels)-1)]
    return assemble_mesh_3d(points, cells, np.arange(len(cells)), normal_distances=True)


def independent_faces(s, flux, incoming=None):
    """Scalar face equations, independent of vector actions and CSR assembly."""
    result = np.zeros(s.shape)
    interior = np.zeros((len(s.owners), s.shape[2]))
    boundary = np.zeros((len(s.boundary_owners), s.shape[2]))
    rhs = np.zeros(s.shape)
    for c, mid in enumerate(s.source.cell_material_ids):
        result[c] = s.area[c]*s.source.materials[mid].total*flux[c]
    for f, ((left, right), measure, normal) in enumerate(zip(s.owners, s.measures, s.normals)):
        for n, omega in enumerate(s.quadrature.directions):
            transfer = measure*sum(omega[d]*normal[d] for d in range(s.dimension))
            trace = flux[left if transfer >= 0 else right, n]
            result[left, n] += transfer*trace
            result[right, n] -= transfer*trace
            interior[f] += s.quadrature.weights[n]*transfer*trace
    for f, (owner, measure, normal) in enumerate(zip(
            s.boundary_owners, s.boundary_measures, s.boundary_normals)):
        for n, omega in enumerate(s.quadrature.directions):
            transfer = measure*sum(omega[d]*normal[d] for d in range(s.dimension))
            if transfer > 0:
                result[owner, n] += transfer*flux[owner, n]
                boundary[f] += s.quadrature.weights[n]*transfer*flux[owner, n]
            elif transfer < 0 and incoming is not None:
                rhs[owner, n] -= transfer*incoming[f, n]
                boundary[f] += s.quadrature.weights[n]*transfer*incoming[f, n]
    return result, rhs, interior, boundary


@pytest.mark.parametrize('two', [False, True])
def test_prism_analytic_streaming_inflow_currents_and_assembled(two):
    s = bind(prisms(two))
    assert s.dimension == 3
    np.testing.assert_allclose(s.area, [.7, 1.6] if two else [.7], atol=1e-15)
    if two:
        np.testing.assert_allclose(s.measures, [1.], atol=1e-15)
        np.testing.assert_allclose(s.normals, [[0., 0., 1.]], atol=1e-15)
    rng = np.random.default_rng(925)
    flux = rng.normal(size=s.shape)
    incoming = rng.normal(size=(len(s.boundary_owners), s.shape[1], s.shape[2]))
    expected, rhs, interior, boundary = independent_faces(s, flux, incoming)
    np.testing.assert_allclose(s.action(flux), expected, atol=3e-15)
    np.testing.assert_allclose(s.boundary_rhs(incoming), rhs, atol=3e-15)
    masked = incoming.copy()
    for f, normal in enumerate(s.boundary_normals):
        masked[f, s.quadrature.directions @ normal >= 0] = np.nan
    np.testing.assert_allclose(s.boundary_rhs(masked), rhs, atol=3e-15)
    np.testing.assert_allclose(s.residual(flux, incoming), expected-rhs, atol=3e-15)
    actual = s.face_currents(flux, incoming)
    np.testing.assert_allclose(actual[0], interior, atol=5e-15)
    np.testing.assert_allclose(actual[1], boundary, atol=5e-15)
    for n in range(s.shape[1]):
        for g in range(s.shape[2]):
            np.testing.assert_allclose(s.assembled(n, g) @ flux[:, n, g], expected[:, n, g], atol=3e-15)
    # Integrated balance counts every interior transfer with opposite signs.
    collision = np.stack([s.area[c]*s.total[mid]*flux[c]
                          for c, mid in enumerate(s.source.cell_material_ids)])
    balance = np.einsum('n,cng->g', s.quadrature.weights, s.residual(flux, incoming)-collision)
    np.testing.assert_allclose(balance, boundary.sum(axis=0), atol=2e-14)


def test_nonzero_z_gradient_sign_and_interior_cancellation():
    s = bind(prisms())
    flux = np.ones(s.shape)
    flux[1] *= 3
    # Match boundary traces to owner values, leaving only the axial jump.
    incoming = flux[s.boundary_owners]
    residual = s.residual(flux, incoming)
    collision = np.stack([s.area[c]*s.total[mid]*flux[c]
                          for c, mid in enumerate(s.source.cell_material_ids)])
    leakage = residual-collision
    mu = s.quadrature.directions[:, 2]
    np.testing.assert_allclose(leakage[0], np.broadcast_to(2*np.minimum(mu, 0)[:, None], leakage[0].shape), atol=4e-15)
    np.testing.assert_allclose(leakage[1], np.broadcast_to(2*np.maximum(mu, 0)[:, None], leakage[1].shape), atol=4e-15)
    # Strip boundary outflow from the actual action: the remaining interior
    # transfers cancel between owners before any angular integration.
    interior_only = s.action(flux)-collision
    for owner, area, normal in zip(s.boundary_owners, s.boundary_measures, s.boundary_normals):
        interior_only[owner] -= area*np.maximum(s.quadrature.directions @ normal, 0)[:, None]*flux[owner]
    np.testing.assert_allclose(interior_only.sum(axis=0), 0, atol=4e-15)
    np.testing.assert_allclose(interior_only[0], mu[:, None]*np.where(mu[:, None] >= 0, flux[0], flux[1]), atol=4e-15)


def test_tetrahedron_oblique_xyz_face_checked_measures():
    mesh = assemble_mesh_3d(np.array([[0., 0., 0.], [2., 0., 0.],
                                     [0., 3., 0.], [0., 0., 4.]]), [(0, 1, 2, 3)], [0], normal_distances=True)
    s = bind(mesh)
    np.testing.assert_allclose(s.area, [4.], atol=2e-15)
    np.testing.assert_allclose(s.centroid, [[.5, .75, 1.]], atol=2e-15)
    oblique = np.flatnonzero(np.all(np.abs(s.boundary_normals) > .1, axis=1))
    assert len(oblique) == 1
    f = oblique[0]
    np.testing.assert_allclose(s.boundary_measures[f], np.sqrt(61.), atol=2e-15)
    np.testing.assert_allclose(s.boundary_normals[f], np.array([6., 4., 3.])/np.sqrt(61.), atol=2e-15)
    np.testing.assert_allclose(np.sort(s.boundary_measures), np.sort([3., 4., 6., np.sqrt(61.)]), atol=2e-15)
    flux = np.arange(np.prod(s.shape), dtype=float).reshape(s.shape)/7
    incoming = np.full((4, s.shape[1], 2), .3)
    expected, rhs, _, boundary = independent_faces(s, flux, incoming)
    np.testing.assert_allclose(s.action(flux), expected, atol=2e-14)
    np.testing.assert_allclose(s.boundary_rhs(incoming), rhs, atol=2e-15)
    np.testing.assert_allclose(s.face_currents(flux, incoming)[1], boundary, atol=2e-14)
    for n in range(s.shape[1]):
        np.testing.assert_allclose(s.assembled(n, 0) @ flux[:, n, 0], expected[:, n, 0], atol=2e-14)


@pytest.mark.parametrize('lmax', [1, 7])
def test_signed_native_scattering_commutes_with_z_reflection(lmax):
    s = bind(prisms(), lmax)
    q = s.quadrature
    permutation = q.reflection('z')
    flux = np.random.default_rng(93).normal(size=s.shape)
    # Exercise z-odd angular content and signed asymmetric cross-group moments.
    flux += q.directions[None, :, 2, None]*np.array([1., -.4])
    scattered = s.source.scatter(flux)
    np.testing.assert_allclose(s.source.scatter(flux[:, permutation]), scattered[:, permutation], atol=8e-14, rtol=2e-13)
    # Independent addition-theorem action; P7 stays action-only (no full factor).
    dot = q.directions @ q.directions.T
    expected = np.zeros(s.shape)
    for c, mid in enumerate(s.source.cell_material_ids):
        for l, matrix in s.source.materials[mid].production.items():
            kernel = (2*l+1)*eval_legendre(l, dot)*q.weights[None, :]/(4*np.pi)
            expected[c] += kernel @ flux[c] @ matrix.T
    np.testing.assert_allclose(scattered, expected, atol=8e-14, rtol=2e-13)


def test_z_reflection_preserves_all_tangents_weights_and_zero_current():
    s = bind(prisms())
    bc = tuple('reflective' if abs(n[2]) > .9 else 'vacuum' for n in s.boundary_normals)
    op = NativeMeshTransport(s, bc)
    flux = np.random.default_rng(19).normal(size=s.shape)
    incoming = op.reflective_incoming(flux)
    _, current = s.face_currents(flux, incoming)
    for f, permutation in enumerate(op.reflection_permutations):
        if permutation is None:
            continue
        q = s.quadrature
        np.testing.assert_array_equal(permutation[permutation], np.arange(q.n_angles))
        np.testing.assert_array_equal(q.weights[permutation], q.weights)
        np.testing.assert_allclose(q.directions[permutation, :2], q.directions[:, :2], atol=2e-15)
        np.testing.assert_allclose(q.directions[permutation, 2], -q.directions[:, 2], atol=2e-15)
        np.testing.assert_allclose(current[f], 0, atol=4e-15)
    expected, _, _, _ = independent_faces(s, flux)
    np.testing.assert_allclose(op.action(flux), expected-s.boundary_rhs(incoming)
                               -s.area[:, None, None]*s.source.scatter(flux), atol=4e-15)


def extrusion(height=2.7):
    points = np.array([[0., 0.], [1., 0.], [3., 0.], [0., 1.], [1., 1.], [3., 1.]])
    cells = [(0, 1, 4, 3), (1, 2, 5, 4)]
    planar = assemble_mesh(points, cells, [0, 1])
    xyz = np.vstack((np.column_stack((points, np.zeros(6))),
                     np.column_stack((points, np.full(6, height)))))
    spatial = assemble_mesh_3d(xyz, [c+tuple(i+6 for i in c) for c in cells], [0, 1], normal_distances=True)
    return bind(planar), bind(spatial)


def test_zeven_extrusion_physical_a_f_raw_chi_and_zodd_counterexample():
    h = 2.7
    s2, s3 = extrusion(h)
    op2 = NativeMeshTransport(s2, ('vacuum',)*len(s2.boundary_owners))
    bc3 = tuple('reflective' if abs(n[2]) > .9 else 'vacuum' for n in s3.boundary_normals)
    op3 = NativeMeshTransport(s3, bc3)
    flux = np.random.default_rng(75).normal(size=s2.shape)
    reflection = s2.quadrature.reflection('z')
    flux = (flux+flux[:, reflection])/2
    np.testing.assert_allclose(op3.action(flux), h*op2.action(flux), atol=8e-15)
    np.testing.assert_allclose(op3.fission(flux), h*op2.fission(flux), atol=3e-15)
    for c, mid in enumerate(s3.source.cell_material_ids):
        m = s3.source.materials[mid]
        phi = np.einsum('n,ng->g', s3.quadrature.weights, flux[c])
        expected = s3.area[c]*(np.outer(m.chi, m.nu_sigma_f) @ phi)/(4*np.pi)
        np.testing.assert_allclose(op3.fission(flux)[c], np.broadcast_to(expected, flux[c].shape), atol=3e-15)
    odd = np.broadcast_to(s3.quadrature.directions[None, :, 2, None], s3.shape)
    assert np.linalg.norm(op3.action(odd)-h*op2.action(odd)) > .1
    # Cap outflow remains in streaming D and L0, even when physical A cancels it.
    assert np.all(op3.diagonal > h*op2.diagonal)


def dense_a_f_r_p(s, bc):
    """Independent tiny dense physical A/F, including specular cap coupling."""
    nc, na, ng = s.shape
    size = nc*na*ng
    a, f = np.zeros((size, size)), np.zeros((size, size))
    r, p = np.zeros((nc*ng, size)), np.zeros((size, nc*ng))
    index = lambda c, n, g: (c*na+n)*ng+g
    q = s.quadrature
    for c, mid in enumerate(s.source.cell_material_ids):
        m = s.source.materials[mid]
        for n in range(na):
            for g in range(ng):
                row = index(c, n, g)
                a[row, row] += s.area[c]*m.total[g]
                r[c*ng+g, row] = q.weights[n]
                p[row, c*ng+g] = 1/(4*np.pi)
                for k in range(na):
                    dot = sum(q.directions[n, d]*q.directions[k, d] for d in range(3))
                    for j in range(ng):
                        col = index(c, k, j)
                        kernel = sum((2*l+1)*eval_legendre(l, dot)*matrix[g, j]
                                     for l, matrix in m.production.items())
                        a[row, col] -= s.area[c]*q.weights[k]*kernel/(4*np.pi)
                        f[row, col] += s.area[c]*m.chi[g]*m.nu_sigma_f[j]*q.weights[k]/(4*np.pi)
    for (left, right), area, normal in zip(s.owners, s.measures, s.normals):
        for n in range(na):
            coefficient = area*sum(q.directions[n, d]*normal[d] for d in range(3))
            for g in range(ng):
                col = index(left if coefficient >= 0 else right, n, g)
                a[index(left, n, g), col] += coefficient
                a[index(right, n, g), col] -= coefficient
    for spec, owner, area, normal in zip(bc, s.boundary_owners, s.boundary_measures, s.boundary_normals):
        for n in range(na):
            coefficient = area*sum(q.directions[n, d]*normal[d] for d in range(3))
            reflected = q.directions[n]-2*np.dot(q.directions[n], normal)*normal
            target = int(np.argmin(np.linalg.norm(q.directions-reflected, axis=1)))
            for g in range(ng):
                row = index(owner, n, g)
                if coefficient > 0:
                    a[row, row] += coefficient
                elif coefficient < 0 and spec == 'reflective':
                    assert np.linalg.norm(q.directions[target]-reflected) < 1e-12
                    a[row, index(owner, target, g)] += coefficient
    return a, f, r, p


def test_extrusion_converged_eigenpair_heat_and_independent_dense_spectrum():
    h = 2.7
    s2, s3 = extrusion(h)  # P1 Q8, unequal cell volumes; 32 unknowns each.
    bc2 = ('vacuum',)*len(s2.boundary_owners)
    bc3 = tuple('reflective' if abs(n[2]) > .9 else 'vacuum' for n in s3.boundary_normals)
    # A tiny manufactured eigenvector comparison needs stronger controls than
    # the default residual gate: a residual bound alone does not bound every
    # angular component's relative error. Production defaults remain unchanged;
    # the separate homogeneous axial leakage test exercises those defaults.
    controls = dict(inner_tolerance=1e-11, k_tolerance=1e-10,
                    source_tolerance=1e-9, full_tolerance=1e-9)
    op2 = NativeMeshTransport(s2, bc2, **controls)
    op3 = NativeMeshTransport(s3, bc3, **controls)
    result2, result3 = op2.solve(), op3.solve()
    for result in (result2, result3):
        assert result.converged, (result.reason, result.history[-1])
        assert result.diagnostics['scalar_nonnegative']
        assert result.diagnostics['heat_power_nonnegative']
        assert result.angular_flux.min() > 0
        assert result.history[-1]['k_change'] <= controls['k_tolerance']
        assert result.history[-1]['source_change'] <= controls['source_tolerance']
        assert result.history[-1]['full_residual'] <= controls['full_tolerance']
    assert result3.k_eff == pytest.approx(result2.k_eff, rel=2e-9)
    # P-normalization gives psi3=psi2/h, so integrated cell heat is equal.
    np.testing.assert_allclose(h*result3.angular_flux, result2.angular_flux, rtol=3e-5, atol=1e-8)
    heat = []
    for s, result in ((s2, result2), (s3, result3)):
        heat.append(np.array([s.area[c]*np.dot(s.source.materials[mid].kappa_fission,
                                               result.scalar_flux[c])
                              for c, mid in enumerate(s.source.cell_material_ids)]))
        assert result.diagnostics['total_heat_power'] == pytest.approx(heat[-1].sum(), rel=2e-14)
    np.testing.assert_allclose(heat[1], heat[0], rtol=3e-5, atol=1e-8)
    # Cold isotropic flux and both operators preserve Jz parity; no condition
    # on equality of the 2D and 3D preconditioners is needed for this eigenpair.
    reflected = s3.quadrature.reflection('z')
    np.testing.assert_allclose(result3.angular_flux[:, reflected], result3.angular_flux, rtol=3e-5, atol=1e-8)
    a, f, _, _ = dense_a_f_r_p(s3, bc3)
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(a, f))
    chosen = int(np.argmax(np.abs(eigenvalues)))
    value = eigenvalues[chosen]
    assert abs(value.imag) < 1e-12 and value.real > 0
    others = np.delete(eigenvalues, chosen)
    assert np.max(np.abs(others)) < value.real*(1-1e-6)
    assert result3.k_eff == pytest.approx(value.real, rel=2e-9)
    reference = eigenvectors[:, chosen].real.reshape(s3.shape)
    scalar = np.einsum('n,cng->cg', s3.quadrature.weights, reference)
    normalization = sum(s3.area[c]*np.dot(s3.source.materials[mid].nu_sigma_f, scalar[c])
                        for c, mid in enumerate(s3.source.cell_material_ids))
    reference /= normalization
    assert reference.min() > 0
    np.testing.assert_allclose(result3.angular_flux, reference, rtol=3e-5, atol=1e-8)
    reference_scalar = np.einsum('n,cng->cg', s3.quadrature.weights, reference)
    reference_heat = np.array([s3.area[c]*np.dot(s3.source.materials[mid].kappa_fission,
                                                reference_scalar[c])
                               for c, mid in enumerate(s3.source.cell_material_ids)])
    np.testing.assert_allclose(heat[1], reference_heat, rtol=3e-5, atol=1e-8)
    field = result3.angular_flux.ravel()
    rhs = f @ field/result3.k_eff
    independent_residual = np.linalg.norm(a @ field-rhs)/np.linalg.norm(rhs)
    assert independent_residual <= controls['full_tolerance']
    assert result3.history[-1]['full_residual'] == pytest.approx(independent_residual, abs=3e-14)


def test_homogeneous_axial_vacuum_leakage_reduces_converged_k():
    # One homogeneous hex, with identical side conditions and only caps varied.
    points = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
                       [0., 0., 2.7], [1., 0., 2.7], [1., 1., 2.7], [0., 1., 2.7]])
    s = bind(assemble_mesh_3d(points, [tuple(range(8))], [0], normal_distances=True))
    reflected = NativeMeshTransport(s, ('reflective',)*len(s.boundary_owners))
    bc = tuple('vacuum' if abs(n[2]) > .9 else 'reflective' for n in s.boundary_normals)
    leaking = NativeMeshTransport(s, bc)
    control, result = reflected.solve(), leaking.solve()
    assert control.converged, control.reason
    assert result.converged, result.reason
    assert 0 < result.k_eff < control.k_eff
    _, control_currents = s.face_currents(control.angular_flux, reflected.reflective_incoming(control.angular_flux))
    _, currents = s.face_currents(result.angular_flux, leaking.reflective_incoming(result.angular_flux))
    np.testing.assert_allclose(control_currents, 0, atol=5e-15)
    for face, spec in enumerate(bc):
        if spec == 'vacuum':
            assert np.all(currents[face] > 0)
        else:
            np.testing.assert_allclose(currents[face], 0, atol=5e-15)
    assert currents.sum() > 0


def test_axial_interior_coarse_rap_sweep_inverse_and_declared_correction():
    s = bind(prisms())  # P1 Q8: 32 angular unknowns, four coarse unknowns.
    bc = tuple('reflective' if n[2] < -.9 else 'vacuum' for n in s.boundary_normals)
    op = StreamingBlockP0Transport(s, bc, ilu_drop_tolerance=1e-12)
    a, f, r, p = dense_a_f_r_p(s, bc)
    np.testing.assert_allclose(op.coarse_matrix.toarray(), r @ a @ p, atol=5e-15)
    vector = np.random.default_rng(501).normal(size=s.shape)
    np.testing.assert_allclose(op.action(vector).ravel(), a @ vector.ravel(), atol=5e-15)
    np.testing.assert_allclose(op.fission(vector).ravel(), f @ vector.ravel(), atol=3e-15)
    np.testing.assert_allclose(op.restrict(op.action(op.prolong(np.ones((2, 2))))).ravel(),
                               r @ a @ p @ np.ones(4), atol=5e-15)
    sweep = np.empty(s.shape)
    for n in range(s.shape[1]):
        for g in range(s.shape[2]):
            block = s.assembled(n, g).toarray()
            sweep[:, n, g] = np.linalg.solve(block, vector[:, n, g])
    np.testing.assert_allclose(op.streaming_inverse(vector), sweep, atol=3e-15)
    # Verify a separately built/factored 3D block, including original cap outflow.
    block = build_streaming_block(s, 0, 0)
    factor = factor_streaming_block(block)
    np.testing.assert_allclose(factor.solve(vector[:, 0, 0]), sweep[:, 0, 0], atol=3e-15)
    z = p @ np.linalg.solve(r @ a @ p, r @ vector.ravel())
    remainder = (vector.ravel()-a @ z).reshape(s.shape)
    correction = np.empty(s.shape)
    for n in range(s.shape[1]):
        for g in range(s.shape[2]):
            correction[:, n, g] = np.linalg.solve(s.assembled(n, g).toarray(), remainder[:, n, g])
    np.testing.assert_allclose(op.precondition(vector).ravel(), z+correction.ravel(), atol=8e-15)
    diagonal_op = P0GalerkinTransport(s, bc, ilu_drop_tolerance=1e-12)
    np.testing.assert_allclose(diagonal_op.precondition(vector).ravel(),
                               z+remainder.ravel()/diagonal_op.diagonal.ravel(), atol=8e-15)
    # Use discrete weighted half ranges; polar Q8 value differs from area/4.
    normal = np.array([0., 0., 1.])
    expected = np.dot(s.quadrature.weights, np.maximum(s.quadrature.directions[:, 2], 0))/(4*np.pi)
    np.testing.assert_allclose(_half_range(s.quadrature, 1., normal), expected, atol=1e-15)
    assert abs(expected-.25) > .01


@pytest.mark.parametrize('change', ['dimension', 'normal_shape', 'normal_unit', 'closure', 'center'])
def test_malformed_3d_geometry_rejected(change):
    mesh = prisms()
    if change == 'dimension':
        bad = replace(mesh, centroid=np.column_stack((mesh.centroid, np.ones(2))))
    elif change == 'normal_shape':
        bad = replace(mesh, face_normals=mesh.face_normals[:, :2])
    elif change == 'normal_unit':
        bad = replace(mesh, bface_normals=2*mesh.bface_normals)
    elif change == 'closure':
        faces = list(mesh.bfaces)
        owner, area, distance = faces[0]
        faces[0] = (owner, 1.2*area, distance)
        bad = replace(mesh, bfaces=faces)
    else:
        bad = replace(mesh, bface_centers=mesh.bface_centers[:, :2])
    with pytest.raises(ValueError):
        bind(bad)


@pytest.mark.parametrize('polar,azimuth', [(None, None), (1, 2), (1, 4), (2, 2)])
def test_3d_degenerate_quadrature_rejected_without_changing_2d(polar, azimuth):
    q = ProductQuadrature(0, n_polar=polar, n_azimuth=azimuth)
    with pytest.raises(ValueError, match='3D streaming requires'):
        bind(prisms(False), 0, q)
    planar = assemble_mesh(np.array([[0., 0.], [1., 0.], [0., 1.]]), [(0, 1, 2)], [0])
    assert bind(planar, 0, q).dimension == 2
    assert bind(prisms(False), 0, ProductQuadrature(0, n_polar=2, n_azimuth=4)).dimension == 3


def test_dimension_aware_geometry_preflight_before_copy():
    mesh = prisms()
    # Dimension-three charge: 8*(14*nc+28*nf+24*nb+3*nm*ng).
    required = 8*(14*mesh.n_cells+28*len(mesh.faces)+24*len(mesh.bfaces)+3*mesh.n_cells*2)
    with pytest.raises(ValueError, match='allocation estimate'):
        bind(mesh, max_bytes=required-1)
    assert bind(mesh, max_bytes=required).dimension == 3


@pytest.mark.parametrize('kind', ['vacuum', 'reflective', 'mixed'])
def test_grouped_3d_actions_and_fixed_source_against_dense_faces(kind):
    from garnet.native_grouped_angular import GroupAngularOperators
    from garnet.native_grouped_fixed_source import GroupedFixedSource

    _, s = extrusion()
    bc = tuple('vacuum' if kind == 'vacuum' or (kind == 'mixed' and n[2] > .9)
               else 'reflective' for n in s.boundary_normals)
    grouped = GroupAngularOperators(s, bc)
    reference = NativeMeshTransport(s, bc)
    rng = np.random.default_rng(319)
    flux = rng.normal(size=s.shape)
    moments = s.source.moments(flux)
    rebuilt = np.empty(s.shape)
    for g in range(s.shape[2]):
        rebuilt[:, :, g] = (grouped.streaming_group_action(flux[:, :, g], g)
                            - grouped.scatter_to_group(moments, g))
        np.testing.assert_allclose(grouped.diagonal_group(g), reference.diagonal[:, :, g], atol=3e-15)
    matrix, _, _, _ = dense_a_f_r_p(s, bc)
    np.testing.assert_allclose(rebuilt, reference.action(flux), atol=5e-14)
    np.testing.assert_allclose(rebuilt.ravel(), matrix @ flux.ravel(), atol=5e-14)
    if kind == 'reflective':
        assert {int(np.argmax(np.abs(n))) for n in s.boundary_normals} == {0, 1, 2}
        for f, permutation in enumerate(grouped.reflection_permutations):
            np.testing.assert_array_equal(permutation, reference.reflection_permutations[f])
    source = rng.normal(size=s.shape)
    solver = GroupedFixedSource(s, bc, inner_tolerance=1e-11, full_tolerance=1e-9,
                                restart=100, max_inner=200, max_sweeps=100)
    result = solver.solve(source)
    assert result.converged, result.reason
    expected = np.linalg.solve(matrix, source.ravel()).reshape(s.shape)
    np.testing.assert_allclose(result.angular_flux, expected, atol=3e-9, rtol=3e-9)
    assert not result.provenance['physical_admission']


def test_grouped_3d_oblique_reflection_fails_closed():
    from garnet.native_grouped_angular import GroupAngularOperators

    s = bind(prisms())
    with pytest.raises(ValueError, match='Cartesian'):
        GroupAngularOperators(s, ('reflective',)*len(s.boundary_owners))


def test_legacy_3d_mesh_without_geometry_is_rejected():
    mesh = assemble_mesh_3d(np.array([[0., 0., 0.], [1., 0., 0.],
                                     [0., 1., 0.], [0., 0., 1.]]), [(0, 1, 2, 3)], [0])
    assert mesh.face_normals is None
    assert mesh.bface_normals is None
    with pytest.raises(ValueError, match='interior normals must be finite real shape'):
        bind(mesh)
