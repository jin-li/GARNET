"""Independent synthetic SP5 computational checks; no physical admission."""
from fractions import Fraction as F

import numpy as np
import pytest


def exact_reference():
    """Derive transformed reaction and half-range boundary before native imports."""
    transform = [[F(1), F(2), F(0)], [F(0), F(3), F(4)], [F(0), F(0), F(5)]]
    augmented = [row[:] + [F(i == j) for j in range(3)] for i, row in enumerate(transform)]
    for i in range(3):
        pivot = augmented[i][i]
        augmented[i] = [v/pivot for v in augmented[i]]
        for j in range(3):
            if j != i:
                multiplier = augmented[j][i]
                augmented[j] = [a-multiplier*b for a, b in zip(augmented[j], augmented[i])]
    Q = [row[3:] for row in augmented]
    polynomials = [[F(1)], [F(0), F(1)]]
    for n in range(1, 5):
        polynomial = [F(0)]*(n+2)
        for k, value in enumerate(polynomials[n]):
            polynomial[k+1] += F(2*n+1, n+1)*value
        for k, value in enumerate(polynomials[n-1]):
            polynomial[k] -= F(n, n+1)*value
        polynomials.append(polynomial)
    half = [[(4*j+1)*sum(a*b/F(k+l+1) for k, a in enumerate(polynomials[2*i+1])
                        for l, b in enumerate(polynomials[2*j])) for j in range(3)] for i in range(3)]
    boundary = [[sum(half[i][k]*Q[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    reaction = [[[F(4*m+1)*Q[m][i]*Q[m][j] for j in range(3)] for i in range(3)] for m in range(3)]
    return np.asarray(reaction, dtype=float), np.asarray(boundary, dtype=float), np.asarray(Q[0], dtype=float)


C_REF, G_REF, WEIGHTS = exact_reference()

# None of the expected SP5 coefficients above use the production tables.
from dataclasses import replace
from scipy import sparse
from scipy.linalg import eig
import garnet.coupled_moment_solver as driver
from garnet.coupled_moments import CoupledMomentOperator, OrthogonalMesh
from garnet.corrected_native_moments import CorrectedNativeMomentOperator
from garnet.mesh import MeshGrid
from garnet.moment_materials import MomentMaterial
from test_corrected_mesh import mesh_grid
from test_corrected_native_moments import orthogonal, partial
from test_coupled_moments import mesh as orthogonal_pair


def material(which=0, groups=3, representable=False):
    total = np.arange(groups)*.3+1.3+.1*which
    loss = {0: np.diag(np.arange(groups)*.2+.5+.05*which)}
    rng = np.random.default_rng(100+which)
    loss[0] -= .03*(np.ones((groups, groups))-np.eye(groups))
    for order in range(1, 6):
        loss[order] = np.diag(total+.03*order)+rng.uniform(-.035, .035, (groups, groups))
    if representable:
        for order in range(1, 6):
            loss[order] = np.diag(total*(.8 if order == 1 else 1.))
    chi = np.arange(groups, 0, -1, dtype=float)
    chi /= chi.sum()
    return MomentMaterial(total, chi, np.arange(groups)*.05+.3, np.arange(groups)*5.+20,
        {}, {}, loss, {}, {}, {'orientation_contract': 'from-major'}, {})


def operator(corrected, *, bc='vacuum', skew=0., mats=None, n=2, **kwargs):
    mesh = mesh_grid(n, skew)
    mats = mats if mats is not None else [material(i % 2) for i in range(mesh.n_cells)]
    if corrected:
        return CorrectedNativeMomentOperator(MeshGrid(mesh, bc), mats, family='sp5', **kwargs)
    return CoupledMomentOperator(orthogonal(mesh, bc), mats, family='sp5', **kwargs)


def dense_pencil(mesh, materials):
    """Independent face-unknown conservation system and its Schur complement."""
    cells, groups = len(materials), materials[0].n_groups
    q = 3*groups
    vacuum = [f for f in mesh.boundary if f.kind == 'vacuum']
    count = len(mesh.interior)+len(vacuum)
    H = np.zeros(((cells+count)*q,)*2)
    fission = np.zeros((cells*q,)*2)
    diffusion = []
    for i, (volume, m) in enumerate(zip(mesh.volumes, materials)):
        sl = slice(i*q, (i+1)*q)
        H[sl, sl] = volume*sum(np.kron(C_REF[k], m.loss[2*k]) for k in range(3))
        fission[sl, sl] = volume*np.kron(np.outer(WEIGHTS, WEIGHTS), np.outer(m.chi, m.nu_sigma_f))
        D = np.zeros((q, q))
        for k in range(3):
            s = slice(k*groups, (k+1)*groups)
            D[s, s] = np.linalg.solve((4*k+3)*m.loss[2*k+1], np.eye(groups))
        diffusion.append(D)
    for face_index, face in enumerate(mesh.interior):
        z = slice((cells+face_index)*q, (cells+face_index+1)*q)
        for i, distance in ((face.left, face.dleft), (face.right, face.dright)):
            s = slice(i*q, (i+1)*q)
            conductance = diffusion[i]/distance
            H[s, s] += face.area*conductance
            H[s, z] -= face.area*conductance
            H[z, s] -= conductance
            H[z, z] += conductance
    for face_index, face in enumerate(vacuum, start=len(mesh.interior)):
        z = slice((cells+face_index)*q, (cells+face_index+1)*q)
        s = slice(face.cell*q, (face.cell+1)*q)
        conductance = diffusion[face.cell]/face.distance
        H[s, s] += face.area*conductance
        H[s, z] -= face.area*conductance
        H[z, s] -= conductance
        H[z, z] += conductance+np.kron(G_REF, np.eye(groups))
    N = cells*q
    reaction = H[:N, :N]-H[:N, N:]@np.linalg.solve(H[N:, N:], H[N:, :N]) if count else H
    return reaction, fission


@pytest.mark.parametrize('corrected', [False, True])
@pytest.mark.parametrize('bc', ['reflective', 'vacuum'])
def test_independent_dense_noncommuting_pencil(corrected, bc):
    a = operator(corrected, bc=bc)
    mesh = orthogonal(a.grid.mesh, bc) if corrected else a.mesh
    expected, fission = dense_pencil(mesh, a.materials)
    assert a.n_moments == 3 and a.shape == (4, 3, 3)
    assert a.required_orders == tuple(range(6))
    for actual, reference in zip(a.C, C_REF):
        np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(a.B, np.kron(G_REF, np.eye(3)))
    assert a.c1 is a.C[0] and a.c2 is a.C[1]
    assert np.linalg.norm(a.diffusion[0]@a.diffusion[1]-a.diffusion[1]@a.diffusion[0]) > 1e-5
    np.testing.assert_allclose(a.assembled.toarray(), expected, atol=4e-14)
    u = np.random.default_rng(35).normal(size=a.shape)
    np.testing.assert_allclose(a.action(u).ravel(), expected@u.ravel(), atol=4e-14)
    np.testing.assert_allclose(a.fission_action(u).ravel(), fission@u.ravel(), atol=2e-14)
    np.testing.assert_array_equal(a.action(u).ravel(), a.action(u.ravel()))


@pytest.mark.parametrize('corrected', [False, True])
def test_orthogonal_pair_face_unknown_oracle(corrected):
    # Unequal volumes and face distances prevent cancellation by mesh symmetry.
    mats = [material(), material(1)]
    mesh = orthogonal_pair()
    if corrected:
        # Corrected equality is covered above on a complete geometry patch.
        a = operator(True)
        mesh, mats = orthogonal(a.grid.mesh), a.materials
    else:
        a = CoupledMomentOperator(mesh, mats, family='sp5')
    A, Fission = dense_pencil(mesh, mats)
    np.testing.assert_allclose(a.assembled.toarray(), A, atol=4e-14)
    np.testing.assert_allclose(np.column_stack([a.fission_action(e) for e in np.eye(a.size)]), Fission, atol=2e-14)


@pytest.mark.parametrize('corrected', [False, True])
def test_third_moment_only_scalar_and_every_fission_row(corrected):
    a = operator(corrected)
    u = np.zeros(a.shape)
    u[:, 2] = np.arange(a.n_cells*a.n_groups).reshape(a.n_cells, a.n_groups)+1
    phi = 8*u[:, 2]/15
    np.testing.assert_array_equal(a.scalar_flux(u), phi)
    volumes = a.volumes if corrected else a.mesh.volumes
    expected = np.stack([v*WEIGHTS[:, None]*m.chi[None, :]*(m.nu_sigma_f@p)
                         for v, m, p in zip(volumes, a.materials, phi)])
    np.testing.assert_allclose(a.fission_action(u), expected, atol=1e-15)
    assert np.all(expected[:, 0] > 0) and np.all(expected[:, 1] < 0) and np.all(expected[:, 2] > 0)


@pytest.mark.parametrize('corrected', [False, True])
@pytest.mark.parametrize('preconditioner', ['none', 'ilu', 'group_ilu'])
def test_eigenpair_matches_independent_dense_pencil(corrected, preconditioner):
    a = operator(corrected)
    A, Fission = dense_pencil(orthogonal(a.grid.mesh) if corrected else a.mesh, a.materials)
    values, vectors = eig(Fission, A)
    index = np.argmax(values.real)
    state = vectors[:, index].real
    volumes = a.volumes if corrected else a.mesh.volumes
    phi = np.einsum('m,cmg->cg', WEIGHTS, state.reshape(a.shape))
    state /= sum(v*(m.nu_sigma_f@p) for v, m, p in zip(volumes, a.materials, phi))
    result = driver.CoupledMomentEigenSolver(a).solve(preconditioner=preconditioner)
    assert result.converged and result.reconstructed_nonnegative
    assert result.k_eff == pytest.approx(values[index].real, rel=2e-9)
    np.testing.assert_allclose(result.moments.ravel(), state, rtol=2e-8, atol=1e-10)
    assert result.full_residual < 1e-9
    assert result.provenance['moments_per_cell'] == 3
    assert result.provenance['required_orders'] == tuple(range(6))
    if preconditioner == 'group_ilu':
        assert [g['dimension'] for g in result.provenance['preconditioner_metadata']['groups']] == [3*a.n_cells]*3
    a.assembled = a.assembled*1.1
    failed = driver.CoupledMomentEigenSolver(a).solve(preconditioner=preconditioner)
    assert not failed.converged and failed.reason == 'inner_failure_or_full_linear_residual'


@pytest.mark.parametrize('corrected', [False, True])
def test_homogeneous_reflective_kinf_and_positive_guards(corrected):
    m = material()
    a = operator(corrected, bc='reflective', mats=[m]*4)
    initial = np.zeros(a.shape)
    initial[:, 0] = 2.
    initial[:, 2] = -.1  # Signed higher moments are allowed without clipping.
    result = driver.CoupledMomentEigenSolver(a).solve(initial=initial, preconditioner='group_ilu')
    assert result.converged
    assert result.k_eff == pytest.approx(m.nu_sigma_f@np.linalg.solve(m.loss[0], m.chi), rel=1e-10)
    assert np.min(result.scalar_flux) > 0
    bad = initial.copy();bad[:, 2] = -10
    with pytest.raises(ValueError, match='scalar flux is negative'):
        driver.CoupledMomentEigenSolver(a).solve(initial=bad)
    zero = replace(m, nu_sigma_f=np.zeros(3))
    with pytest.raises(ValueError, match='fission source'):
        driver.CoupledMomentEigenSolver(operator(corrected, mats=[zero]*4)).solve()
    negative = replace(m, kappa_fission=-m.kappa_fission)
    rejected = driver.CoupledMomentEigenSolver(operator(corrected, mats=[negative]*4)).solve()
    assert not rejected.converged and rejected.reason == 'negative_reconstructed_scalar_or_power'
    assert np.all(rejected.power < 0)  # Returned output is preserved, never clipped.


@pytest.mark.parametrize('corrected', [False, True])
def test_full_group_permutation(corrected):
    mats = [material(i % 2) for i in range(4)]
    order = np.array([2, 0, 1])
    def perm(m):
        return replace(m, total=m.total[order], chi=m.chi[order], nu_sigma_f=m.nu_sigma_f[order],
                       kappa_fission=m.kappa_fission[order], loss={k:v[np.ix_(order, order)] for k,v in m.loss.items()})
    a, b = operator(corrected, mats=mats), operator(corrected, mats=[perm(m) for m in mats])
    indices = np.arange(a.size).reshape(a.shape)[:, :, order].ravel()
    np.testing.assert_allclose(b.assembled.toarray(), a.assembled.toarray()[np.ix_(indices, indices)], atol=4e-14)
    u = np.random.default_rng(15).normal(size=a.size)
    np.testing.assert_allclose(b.action(u[indices]), a.action(u)[indices], atol=4e-14)
    np.testing.assert_allclose(b.fission_action(u[indices]), a.fission_action(u)[indices], atol=2e-14)
    if corrected:
        np.testing.assert_allclose(b.gradients(u[indices]), a.gradients(u)[:, np.arange(9).reshape(3,3)[:,order].ravel()], atol=4e-14)
        for actual, expected in zip(b.face_currents(u[indices]), a.face_currents(u)):
            np.testing.assert_allclose(actual, expected[:, :, order], atol=4e-14)
    ra, rb = (driver.CoupledMomentEigenSolver(op).solve(preconditioner='group_ilu') for op in (a,b))
    assert ra.converged and rb.converged
    assert ra.k_eff == pytest.approx(rb.k_eff, rel=2e-10)
    np.testing.assert_allclose(rb.scalar_flux, ra.scalar_flux[:,order], rtol=2e-9)


def test_group_ilu_all_moments_offgroup_matrix_and_frozen_factors(monkeypatch):
    a = CoupledMomentOperator(OrthogonalMesh([1., 2.], [[0.], [1.]]), [material()]*2, family='sp5')
    indices = [np.arange(a.size).reshape(a.shape)[:, :, group].ravel() for group in range(3)]
    matrix = np.full((a.size, a.size), .017)
    blocks = []
    for group, index in enumerate(indices):
        block = np.random.default_rng(group).uniform(-.2, .2, (6,6))+np.eye(6)*(group+1)
        matrix[np.ix_(index,index)] = block
        blocks.append(block)
    a.assembled = sparse.csr_matrix(matrix)
    full_matrix = a.assembled
    original_factor = driver.spilu
    calls = []
    def factor(block, **kwargs):
        np.testing.assert_array_equal(block.toarray(), blocks[len(calls)])
        assert block.format == 'csc' and not np.shares_memory(block.data, a.assembled.data)
        calls.append(block.shape)
        return original_factor(block, **kwargs)
    def gmres(operator, rhs, *, M, **kwargs):
        assert operator is full_matrix
        np.testing.assert_array_equal(operator.toarray(), matrix)
        vector = np.arange(a.size, dtype=float)+1
        expected = np.empty(a.size)
        for index, block in zip(indices, blocks):
            expected[index] = np.linalg.solve(block, vector[index])
        np.testing.assert_allclose(M@vector, expected, rtol=1e-13, atol=1e-13)
        operator.data *= 1.7
        np.testing.assert_allclose(M@vector, expected, rtol=1e-13, atol=1e-13)
        return np.linalg.solve(operator.toarray(), rhs), 0
    monkeypatch.setattr(driver, 'spilu', factor)
    monkeypatch.setattr(driver, 'gmres', gmres)
    result = driver.CoupledMomentEigenSolver(a).solve(preconditioner='group_ilu')
    assert calls == [(6,6)]*3
    assert not result.converged and result.reason == 'inner_failure_or_full_linear_residual'
    assert 'M*G*i+G*m+g' in result.provenance['preconditioner_metadata']['group_cell_ordering']


@pytest.mark.parametrize('bc', ['reflective', 'vacuum'])
def test_existing_scalar_sp5_orthogonal_reduction(bc):
    from garnet import Grid
    from garnet.spn import SDPNGroupOperator
    # The scalar path is representable only when L2..5 equal diag(total).
    mesh = mesh_grid(2, 0.)
    mats = [material(i % 2, representable=True) for i in range(4)]
    a = operator(True, bc=bc, mats=mats)
    grid = Grid(shape=(2,2,1), size=(1.,1.,1.))
    state = np.random.default_rng(25).normal(size=a.shape)
    expected = np.zeros_like(state)
    coefficients = {2: C_REF.tolist()}
    for group in range(3):
        st = np.array([m.total[group] for m in mats]).reshape(grid.shape)
        rem = np.array([m.loss[0][group,group] for m in mats]).reshape(grid.shape)
        D = np.array([1/(3*m.loss[1][group,group]) for m in mats]).reshape(grid.shape)
        boundary = ((bc,bc),(bc,bc),('reflective','reflective'))  # Match the planar mesh: no axial leakage.
        old = SDPNGroupOperator(np, grid, D, st, rem, 2, bc=boundary, coeffs=coefficients, boundary_g=G_REF)
        expected[:,:,group] = old.apply(state[:,:,group].T.reshape((3,)+grid.shape)).reshape(3,4).T*mesh.area[:,None]
        for other in range(3):
            if other != group:
                off = np.array([m.loss[0][group,other] for m in mats])
                expected[:,:,group] += mesh.area[:,None]*off[:,None]*(state[:,:,other]@C_REF[0].T)
    np.testing.assert_allclose(a.action(state), expected, atol=5e-14)


def test_noncommuting_piecewise_affine_interface_and_conservation():
    mesh = partial(mesh_grid(4,.35))
    normal = np.array([1.,-.35])/np.hypot(1,.35)
    tangent = np.array([.35,1.])/np.hypot(1,.35)
    signed = mesh.centroid@normal-.5/np.hypot(1,.35)
    pair = [material(), material(1)]
    mats = [pair[int(v>0)] for v in signed]
    a = CorrectedNativeMomentOperator(MeshGrid(mesh,'vacuum'), mats, family='sp5')
    current = np.linspace(-.3,.7,9)
    ht = np.linspace(.2,-.4,9)
    base = np.linspace(1.,2.,9)
    H = np.stack([np.linalg.solve(D,current)[:,None]*normal+ht[:,None]*tangent for D in a.diffusion])
    state = np.stack([base+np.linalg.solve(D,current)*sn+ht*tn for D,sn,tn in zip(a.diffusion,signed,mesh.centroid@tangent)])
    np.testing.assert_allclose(a.gradients(state.ravel()), H, atol=5e-13)
    currents, boundary = a.face_currents(state.ravel())
    checked = 0
    for k, ((i,j,area,_), center, n) in enumerate(zip(mesh.faces, mesh.face_centers, mesh.face_normals)):
        np.testing.assert_allclose(currents[k].ravel(), -area*a.diffusion[i]@H[i]@n, atol=5e-14)
        if mats[i] is not mats[j] and signed[i]*signed[j]<0:
            np.testing.assert_allclose(state[i]+H[i]@(center-mesh.centroid[i]), state[j]+H[j]@(center-mesh.centroid[j]), atol=5e-14)
            np.testing.assert_allclose(a.diffusion[i]@H[i]@n, a.diffusion[j]@H[j]@n, atol=5e-14)
            checked += 1
    assert checked == 4
    reaction = np.stack([r@u for r,u in zip(a.reaction,state)])
    np.testing.assert_allclose((a.action(state.ravel()).reshape(-1,9)-reaction).sum(axis=0), boundary.reshape(-1,9).sum(axis=0), atol=5e-14)
    np.testing.assert_allclose(a.assembled@state.ravel(), a.action(state.ravel()), atol=5e-14)


def test_affine_coupled_vacuum_identity():
    mesh = partial(mesh_grid(3,.35), [0])
    a = CorrectedNativeMomentOperator(MeshGrid(mesh,'vacuum'), [material()]*mesh.n_cells, family='sp5')
    n = mesh.bface_normals[0]
    tangent = np.array([-n[1],n[0]])
    trace = np.linspace(-.2,1.1,9)
    H = -np.linalg.solve(a.diffusion[0],np.kron(G_REF,np.eye(3))@trace)[:,None]*n+np.linspace(.1,.4,9)[:,None]*tangent
    state = trace+(mesh.centroid-mesh.bface_centers[0])@H.T
    np.testing.assert_allclose(a.gradients(state.ravel()), np.broadcast_to(H,(mesh.n_cells,)+H.shape), atol=5e-13)
    _, boundary = a.face_currents(state.ravel())
    area = mesh.bfaces[0][1]
    np.testing.assert_allclose(boundary[0].ravel(), area*np.kron(G_REF,np.eye(3))@trace, atol=5e-14)
    np.testing.assert_allclose(boundary[0].ravel(), -area*a.diffusion[0]@H@n, atol=5e-14)


@pytest.mark.parametrize('corrected', [False, True])
def test_required_orders_finite_even_losses_and_odd_conditioning(corrected):
    m = material()
    for order in range(6):
        loss = dict(m.loss);del loss[order]
        with pytest.raises(ValueError, match='0..5'):
            operator(corrected, mats=[replace(m,loss=loss)]*4)
        loss = dict(m.loss);loss[order] = np.full((3,3),np.nan)
        with pytest.raises(ValueError, match='finite'):
            operator(corrected, mats=[replace(m,loss=loss)]*4)
    # Even reactions need not be invertible: vacuum leakage defines this case.
    loss = dict(m.loss)
    for order in (0,2,4):
        loss[order] = np.zeros((3,3))
    a = operator(corrected, mats=[replace(m,loss=loss)]*4)
    assert np.linalg.matrix_rank(a.assembled.toarray()) == a.size
    for order in (1,3,5):
        loss = dict(m.loss);loss[order] = np.diag([1.,1.,1e-14])
        with pytest.raises(ValueError, match='ill-conditioned'):
            operator(corrected, mats=[replace(m,loss=loss)]*4)
    with pytest.raises(ValueError, match='unknowns'):
        a.scalar_flux(np.ones((a.n_cells,2,a.n_groups)))


def test_sp5_default_budget_counts_all_three_moments_before_geometry(monkeypatch):
    grid = MeshGrid(mesh_grid(2,.35),'vacuum')
    grid.n_cells = 2223  # 2223*3*3=20007, whereas a two-moment count is below cap.
    def forbidden(*args, **kwargs):
        raise AssertionError('Geometry/patch allocation must not precede budget guard')
    import garnet.corrected_native_moments as native
    monkeypatch.setattr(native,'_validate_corrected_mesh',forbidden)
    with pytest.raises(ValueError, match='20000'):
        CorrectedNativeMomentOperator(grid,[material()]*2223,family='sp5')
    for budget in (0,False,1.5):
        with pytest.raises(ValueError, match='positive integer'):
            CorrectedNativeMomentOperator(grid,[material()]*2223,family='sp5',max_unknowns=budget)


def test_sp5_explicit_budget_and_immutable_coefficients(monkeypatch):
    a = operator(True)
    monkeypatch.setattr(CorrectedNativeMomentOperator,'prototype_unknown_limit',a.size-1)
    with pytest.raises(ValueError, match='limited'):
        operator(True)
    b = operator(True,max_unknowns=a.size)
    np.testing.assert_array_equal(a.assembled.toarray(),b.assembled.toarray())
    for array in (*a.C,a.B):
        with pytest.raises(ValueError):
            array.setflags(write=True)
