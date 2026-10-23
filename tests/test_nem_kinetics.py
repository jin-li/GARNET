"""Complete source response/transpose checks on unequal nodes and ADFs."""
import numpy as np
import pytest
from garnet.nem import NEMDiffusionSolver
from garnet.nem_higher_order import NEMHigherOrderDiffusionSolver
from garnet.nem_kinetics import (NEMSourceResponse, solve_nem_adjoint,
                                project_nem_kinetics, project_nem_time)
from test_nem import tensor_grid, materials


@pytest.mark.parametrize('degree', [None, 4, 8, 16])
def test_source_response_and_transpose(degree):
    grid = tensor_grid(([0., .7, 1.9], [0., 1.3], [0., .8]))
    mat = materials(grid, 2)
    factors = np.ones((grid.n_nodes, 6, 2))
    factors[0, 2] = [.91, 1.07]
    factors[1, 0] = [1.05, .94]
    solver = (NEMDiffusionSolver(grid, *mat, factors) if degree is None else
              NEMHigherOrderDiffusionSolver(grid, *mat, factors, degree=degree))
    H = NEMSourceResponse(solver)
    rng = np.random.default_rng(912)
    q, w = rng.normal(size=(2, H.shape[0]))
    # Signed small sources need no physical positivity condition.
    q[:H.mean_size] = 1 + abs(q[:H.mean_size])
    q[H.mean_size:] *= .00001
    result = solver.solve_fixed_source(q[:H.mean_size].reshape(2, 2),
                                     q[H.mean_size:].reshape(2, 3, solver.n_moments, 2))
    np.testing.assert_allclose(H @ q, H.pack(result), rtol=2e-12, atol=1e-13)
    # Explicit columns are an independent evaluation of the actual forward
    # response; transpose must include local direct and boundary RHS terms.
    dense = np.column_stack([H @ e for e in np.eye(H.shape[0])])
    np.testing.assert_allclose(H.rmatvec(w), dense.T @ w, rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(w @ (H @ q), H.rmatvec(w) @ q, rtol=2e-12)
    T = H.fission_operator()
    Tdense = np.column_stack([T @ e for e in np.eye(H.shape[0])])
    np.testing.assert_allclose(T.rmatvec(w), Tdense.T @ w, rtol=2e-11, atol=2e-12)


def test_response_rejects_replaced_cached_material():
    grid = tensor_grid(([0., 1.], [0., 1.], [0., 1.]))
    solver = NEMDiffusionSolver(grid, *materials(grid, 2))
    H = NEMSourceResponse(solver)
    solver.D = solver.D * 2
    with pytest.raises(ValueError, match='changed'):
        H @ np.ones(H.shape[0])


@pytest.mark.parametrize('degree', [4, 8, 16])
@pytest.mark.parametrize('policy', ['one_sided', 'primary_rule'])
@pytest.mark.parametrize('general_production', [False, True])
def test_physical_projection_against_independent_augmented_pencil(degree, policy, monkeypatch, general_production):
    from garnet.nem_geometry import CuboidNodalGrid
    from test_nem_higher_order import independent_pencil, recovered_vector
    if policy == 'primary_rule':
        import test_nem_higher_order
        from test_nem_boundary_leakage import independent_weights
        monkeypatch.setattr(test_nem_higher_order, 'independent_weights', independent_weights)
    grid0 = tensor_grid(([0., .7, 1.9], [0., 1.3], [0., .8]))
    grid = CuboidNodalGrid(grid0.bounds, leakage_boundary=policy)
    mat = materials(grid, 2)
    mat[4][:] = [[.8,.2],[.65,.35]]
    factors = np.ones((2, 6, 2))
    factors[0, 2] = [.91, 1.07]
    factors[1, 0] = [1.05, .94]
    kernel = None
    if general_production:
        # Incident-dependent spectra cannot be represented as chi outer nuF.
        kernel = np.array([[[.83,.51],[.17,.49]],
                           [[.62,.37],[.38,.63]]])*mat[3][:,None,:]
    solver = NEMHigherOrderDiffusionSolver(grid, *mat, factors, degree=degree,
                                          fission_matrix=kernel)
    fwd = solver.solve_eigenvalue(rtol=2e-12, max_iterations=2000)
    adj = solve_nem_adjoint(solver, fwd, rtol=1e-9)
    inv = np.array([[1e-7, 2e-5], [1.3e-7, 2.6e-5]])
    # Family spectra and delayed yields depend on region and incident group.
    delayed = solver.fission[:,None] * np.array([[[[.002, .004], [.003, .005]],
                                                [[.001, .002], [.002, .001]]],
                                               [[[.004, .003], [.002, .004]],
                                                [[.002, .001], [.001, .002]]]])
    decay = np.array([.08, .3])
    got = project_nem_kinetics(solver, fwd, adj, inverse_velocities=inv,
                              delayed_production=delayed, decay=decay)
    A, Q = independent_pencil(grid, *mat, degree, factors, fission_matrix=kernel)
    x = recovered_vector(fwd, degree)
    # Build the physical mass/delayed operators independently by Gauss
    # integration of the retained Legendre weak test functions.
    u, weights = np.polynomial.legendre.leggauss(2*degree+3)
    basis = np.polynomial.legendre.legvander(u, degree).T
    overlap = ((basis[1:degree-1]/2)*(weights/2)) @ basis.T
    overlap[abs(overlap) < 1e-13] = 0
    B = np.zeros_like(A)
    Qi = np.zeros((2, *A.shape))
    for node in range(2):
        sl = slice(node*2, (node+1)*2)
        B[sl, sl] = np.diag(inv[node])
        Qi[:, sl, sl] = delayed[node]
        for axis in range(3):
            offset = 4 + (node*3+axis)*degree*2
            for n in range(degree-2):
                row = slice(offset+n*2, offset+(n+1)*2)
                for j in range(1, degree+1):
                    col = slice(offset+(j-1)*2, offset+j*2)
                    B[row, col] = np.diag(inv[node]) * overlap[n,j]
                    Qi[:,row,col] = delayed[node] * overlap[n,j]
    vals, vecs = np.linalg.eig(np.linalg.solve(A.T, Q.T))
    idx = np.argmax(vals.real)
    ell = vecs[:,idx].real
    assert abs(vals[idx].imag) < 1e-12
    np.testing.assert_allclose(vals[idx].real, fwd.k_eff, rtol=2e-10)
    expected_time = (ell @ B @ x)/(ell @ Q @ x)
    expected_beta = np.array([(ell @ q @ x)/(ell @ Q @ x) for q in Qi])
    np.testing.assert_allclose(got.generation_time, expected_time, rtol=3e-9)
    np.testing.assert_allclose(got.beta, expected_beta, rtol=3e-9)
    np.testing.assert_allclose(got.neutron_lifetime, fwd.k_eff*expected_time, rtol=3e-9)
    assert np.max(abs(fwd.moments[:,:,2:])) > 1e-12 if degree > 4 else True
    # Equation units/scaling must cancel in all physical bilinear ratios.
    scale = np.geomspace(.02, 50, len(A))
    As, Qs, Bs = scale[:,None]*A, scale[:,None]*Q, scale[:,None]*B
    sv, se = np.linalg.eig(np.linalg.solve(As.T, Qs.T))
    left_scaled = se[:,np.argmax(sv.real)].real
    np.testing.assert_allclose((left_scaled @ Bs @ x)/(left_scaled @ Qs @ x),
                               expected_time, rtol=2e-9)
    # Independent alpha perturbation: A+alpha B, k'(0)=-k^2 Lambda.
    step = .02
    kval = lambda alpha: np.max(np.linalg.eigvals(np.linalg.solve(A+alpha*B,Q)).real)
    derivative = (kval(step)-kval(-step))/(2*step)
    np.testing.assert_allclose(derivative, -fwd.k_eff**2*expected_time, rtol=2e-5)
    # Both eigenvectors may be normalized arbitrarily.
    from dataclasses import replace
    changed_fwd = replace(fwd, means=fwd.means*7, moments=fwd.moments*7)
    changed_adj = replace(adj, left_state=adj.left_state*.17, source_importance=adj.source_importance*.17)
    again = project_nem_kinetics(solver, changed_fwd, changed_adj,
                               inverse_velocities=inv, delayed_production=delayed, decay=decay)
    np.testing.assert_allclose(again.generation_time, got.generation_time, rtol=1e-13)
    np.testing.assert_allclose(again.beta, got.beta, rtol=1e-13)


def test_extreme_normalization_and_invalid_nonfinite_adjoint_rejected():
    from dataclasses import replace
    grid = tensor_grid(([0., .7, 2.3], [0., 1.3], [0., .8]), boundary='reflective')
    solver = NEMHigherOrderDiffusionSolver(grid, [.5], [.07], [[0.]], [.1], [1.], degree=8)
    forward = solver.solve_eigenvalue(rtol=1e-12)
    adjoint = solve_nem_adjoint(solver, forward)
    inv = np.array([[1e-5],[2e-5]])
    expected = project_nem_time(solver, forward, adjoint, inverse_velocities=inv)
    np.testing.assert_allclose(expected.generation_time,
        float(np.sum(grid.volumes[:,None]*inv))/float(np.sum(grid.volumes)*.1), rtol=1e-11)
    for fs, ads in [(1e200,1e-200),(1e-200,1e200)]:
        f = replace(forward, means=forward.means*fs, moments=forward.moments*fs)
        a = replace(adjoint, left_state=adjoint.left_state*ads,
                    source_importance=adjoint.source_importance*ads)
        got = project_nem_time(solver, f, a, inverse_velocities=inv)
        np.testing.assert_allclose(got.generation_time, expected.generation_time, rtol=1e-12)
    invalid = replace(forward, means=np.array([[1e200],[2e200]]))
    with pytest.raises(ValueError, match='does not match'):
        project_nem_time(solver, invalid, adjoint, inverse_velocities=inv)
    for value in [float('nan'), float('inf'), -1.]:
        with pytest.raises(ValueError, match='k_eff'):
            project_nem_time(solver, forward, replace(adjoint,k_eff=value), inverse_velocities=inv)


def test_zero_production_reflector_and_different_family_spectra():
    grid = tensor_grid(([0., 5., 9.], [0., 5.], [0., 5.]), boundary='reflective')
    mat = list(materials(grid,2))
    mat[3][1] = 0
    mat[4][:] = [[.8,.2],[.65,.35]]
    solver = NEMHigherOrderDiffusionSolver(grid,*mat,degree=8)
    f = solver.solve_eigenvalue(rtol=1e-12)
    a = solve_nem_adjoint(solver,f)
    delayed = np.zeros((2,2,2,2))
    delayed[0,0] = solver.fission[0]*np.array([[.001,.002],[.002,.004]])
    delayed[0,1] = solver.fission[0]*np.array([[.003,.001],[.001,.002]])
    got = project_nem_kinetics(solver,f,a,inverse_velocities=np.full((2,2),1e-5),
                               delayed_production=delayed,decay=[.08,.3])
    assert 0 < got.beta_total < .01
    assert got.generation_time > 0


@pytest.mark.parametrize('absorption', [.07, .1, .14])
def test_offcritical_one_group_unequal_volume_analytical(absorption):
    grid = tensor_grid(([0., .7, 2.3], [0., 1.3], [0., .8]), boundary='reflective')
    solver = NEMHigherOrderDiffusionSolver(grid, [.5], [absorption], [[0.]], [.1], [1.], degree=8)
    forward = solver.solve_eigenvalue(rtol=1e-12)
    adjoint = solve_nem_adjoint(solver, forward)
    got = project_nem_kinetics(solver, forward, adjoint,
        inverse_velocities=np.full((2,1), 1/220000.),
        delayed_production=np.broadcast_to(np.array([.0002,.0005])[None,:,None,None], (2,2,1,1)),
        decay=[.08,.3])
    np.testing.assert_allclose(forward.k_eff, .1/absorption, rtol=1e-10)
    np.testing.assert_allclose(got.generation_time, 1/(220000*.1), rtol=1e-10)
    np.testing.assert_allclose(got.neutron_lifetime, 1/(220000*absorption), rtol=1e-10)
    np.testing.assert_allclose(got.beta, [.002,.005], rtol=1e-10)
    np.testing.assert_allclose(got.lambda_low_frequency, .007/(.002/.08+.005/.3))
    with pytest.raises(ValueError, match='exceeds total'):
        project_nem_kinetics(solver, forward, adjoint,
            inverse_velocities=np.full((2,1), 1/220000.),
            delayed_production=np.ones((2,2,1,1)), decay=[.08,.3])
