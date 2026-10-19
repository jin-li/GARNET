"""Independent full augmented checks of the condensed CPU NEM prototype."""
import itertools

import numpy as np
import pytest

from ndgpu.nem import NEMDiffusionSolver, NEMSolverError
from ndgpu.nem_geometry import CuboidNodalGrid

PAIRS = ((0,2),(1,3),(4,5))


def tensor_grid(edges,boundary='vacuum'):
    bounds = []
    for indices in itertools.product(*(range(len(e)-1) for e in edges)):
        bounds.append([[edges[a][i],edges[a][i+1]] for a,i in enumerate(indices)])
    return CuboidNodalGrid(bounds,boundary)


def independent_weights(grid,node,axis):
    """Interval means and weighted targets integrated with independent Gauss."""
    sm,sp = PAIRS[axis]
    left,right = grid.neighbors[node,[sm,sp]]
    samples = [node]
    if left >= 0 and right >= 0:
        samples += [left,right]
    elif left >= 0 or right >= 0:
        side = sm if left >= 0 else sp
        first = grid.neighbors[node,side]
        samples.append(first)
        if grid.neighbors[first,side] >= 0:
            samples.append(grid.neighbors[first,side])
    x,w = np.polynomial.legendre.leggauss(10)
    degree = len(samples)-1
    center = grid.bounds[node,axis].mean()
    width = grid.widths[node,axis]
    averages = []
    for sample in samples:
        lo,hi = (grid.bounds[sample,axis]-center)/width
        t = (lo+hi)/2+(hi-lo)/2*x
        averages.append([np.sum(w*t**p)/2 for p in range(degree+1)])
    t = x/2
    tests = [t,3*t*t-.25]
    targets = np.array([[np.sum(w*test*t**p)/2 for p in range(degree+1)] for test in tests])
    # Exact oddness and zero-mean identities, avoiding Gauss rounding of zero.
    targets[:,0] = 0
    if degree >= 1:
        targets[1,1] = 0
    if degree >= 2:
        targets[0,2] = 0
    return np.array(samples),np.linalg.solve(np.array(averages).T,targets.T).T


def full_augmented(grid,D,absorption,scatter,nuf,chi,factors=None):
    """Assemble uncondensed mu, m, J, b and physical fission pencil independently."""
    N,G = D.shape;F = len(grid.face_nodes)
    ng = N*G
    joff,boff = 7*ng,7*ng+F*G
    size = 13*ng+F*G
    A,Q = np.zeros((size,size)),np.zeros((size,size))
    def m(i,a,p,g):return ng+((i*3+a)*2+p)*G+g
    def j(f,g):return joff+f*G+g
    def b(i,s,g):return boff+(i*6+s)*G+g
    if factors is None:factors = np.ones((N,6,G))
    for i in range(N):
        R = np.zeros((G,G))
        for g in range(G):
            R[g,g] = absorption[i,g]+sum(scatter[i,g,h] for h in range(G) if h != g)
            for h in range(G):
                if h != g:R[g,h] = -scatter[i,h,g]
        for g in range(G):
            for h in range(G):
                A[i*G+g,i*G+h] = R[g,h]
                Q[i*G+g,i*G+h] = chi[i,g]*nuf[i,h]
        for a,(sm,sp) in enumerate(PAIRS):
            L = grid.widths[i,a]
            fm,fp = grid.node_faces[i,[sm,sp]]
            samples,weights = independent_weights(grid,i,a)
            for g in range(G):
                H = D[i,g]/L**2
                A[i*G+g,j(fp,g)] += 1/L;A[i*G+g,j(fm,g)] -= 1/L
                for p in range(2):
                    row = m(i,a,p,g)
                    for h in range(G):
                        A[row,m(i,a,p,h)] += R[g,h]
                        Q[row,m(i,a,p,h)] = chi[i,g]*nuf[i,h]
                    for k,sample in enumerate(samples):
                        for transverse,(tm,tp) in enumerate(PAIRS):
                            if transverse != a:
                                w = weights[p,k]/grid.widths[i,transverse]
                                A[row,j(grid.node_faces[sample,tp],g)] += w
                                A[row,j(grid.node_faces[sample,tm],g)] -= w
                row1,row2 = m(i,a,0,g),m(i,a,1,g)
                A[row1,b(i,sp,g)] += H;A[row1,b(i,sm,g)] -= H
                A[row1,j(fp,g)] += 1/(2*L);A[row1,j(fm,g)] += 1/(2*L)
                A[row2,b(i,sp,g)] += 3*H;A[row2,b(i,sm,g)] += 3*H
                A[row2,j(fp,g)] += 1/(2*L);A[row2,j(fm,g)] -= 1/(2*L)
                A[row2,i*G+g] -= 6*H
                for p in range(2):
                    row = 7*ng+((i*3+a)*2+p)*G+g
                    A[row,m(i,a,0,g)] = 60
                    A[row,m(i,a,1,g)] = 140 if p == 0 else -140
                    A[row,b(i,sp,g)] = -16 if p == 0 else 4
                    A[row,b(i,sm,g)] = -4 if p == 0 else 16
                    A[row,i*G+g] = 20 if p == 0 else -20
                    A[row,j(fp if p == 0 else fm,g)] = -L/D[i,g]
    boundary = {int(f):k for k,f in enumerate(grid.boundary_faces)}
    for f,(low,high) in enumerate(grid.face_nodes):
        sm,sp = PAIRS[grid.face_axes[f]]
        for g in range(G):
            row = 13*ng+f*G+g
            if low >= 0 and high >= 0:
                A[row,b(low,sp,g)] = factors[low,sp,g]
                A[row,b(high,sm,g)] = -factors[high,sm,g]
            else:
                k = boundary[f];i,s = grid.boundary_nodes[k],grid.boundary_sides[k]
                A[row,j(f,g)] = grid.normal_signs[s]
                A[row,b(i,s,g)] = -grid.boundary_alpha[k]
    return A,Q


def full_state(result):
    return np.r_[result.means.ravel(),result.moments.ravel(),result.currents.ravel(),result.side_face_means.ravel()]


def physical_rhs(grid,q0,qm,drive):
    N,G = q0.shape;F = len(grid.face_nodes)
    rhs = np.zeros(13*N*G+F*G)
    rhs[:N*G] = q0.ravel();rhs[N*G:7*N*G] = qm.ravel()
    face_rhs = rhs[13*N*G:].reshape(F,G)
    face_rhs[grid.boundary_faces] = drive
    return rhs


def materials(grid,G):
    N = grid.n_nodes
    D = np.linspace(.8,1.2,N)[:,None]*np.linspace(1.,.3,G)[None,:]
    absorption = np.linspace(.12,.21,N)[:,None]*np.linspace(1.,1.3,G)[None,:]
    scatter = np.zeros((N,G,G))
    for g in range(G):
        for h in range(G):
            scatter[:,g,h] = .01*(g+1)/(h+1) if g != h else 7.  # diagonal must be ignored
    nuf = np.broadcast_to(np.linspace(.08,.04,G),(N,G)).copy()
    chi = np.zeros((N,G));chi[:,0] = 1
    return D,absorption,scatter,nuf,chi


@pytest.mark.parametrize('G',[2,4])
def test_condensed_vs_independent_full_augmented_nonzero_moments(G):
    grid = tensor_grid(([0.,.7,2.],[0.,.4,1.5],[0.,1.]))
    D,sa,ss,nuf,chi = materials(grid,G)
    rng = np.random.default_rng(814)
    q0 = .4+rng.random((grid.n_nodes,G))
    qm = rng.normal(size=(grid.n_nodes,3,2,G))*.003
    drive = rng.normal(size=(len(grid.boundary_faces),G))*.02
    factors = np.ones((grid.n_nodes,6,G))
    for f,(low,high) in enumerate(grid.face_nodes):
        if low >= 0 and high >= 0:
            sm,sp = PAIRS[grid.face_axes[f]]
            factors[low,sp] = np.linspace(.9,1.05,G)
            factors[high,sm] = np.linspace(1.1,.95,G)
    solver = NEMDiffusionSolver(grid,D,sa,ss,nuf,chi,factors)
    result = solver.solve_fixed_source(q0,qm,drive)
    A,_ = full_augmented(grid,D,sa,ss,nuf,chi,factors)
    rhs = physical_rhs(grid,q0,qm,drive)
    expected = np.linalg.solve(A,rhs)
    np.testing.assert_allclose(full_state(result),expected,rtol=2e-10,atol=2e-12)
    residual = A@full_state(result)-rhs
    scale = abs(A)@abs(full_state(result))+abs(rhs)
    assert np.max(abs(residual)/np.maximum(scale,np.finfo(float).tiny)) < 1e-11
    for key in solver._residual_keys:
        assert result.diagnostics[key] < 1e-10
    assert result.diagnostics['backend'] == 'NumPy/SciPy CPU'
    raw_weak_ratio = abs(result.diagnostics['weak_residual'])/result.diagnostics['weak_residual_scale']
    assert result.diagnostics['weak_balance_error'] == pytest.approx(raw_weak_ratio.max(),rel=1e-13)
    assert result.diagnostics['linear_solver'] == 'SuperLU'
    assert type(result.means) is np.ndarray and type(result.currents) is np.ndarray
    assert type(solver._lu).__name__ == 'SuperLU'
    for f,(low,high) in enumerate(grid.face_nodes):
        if low >= 0 and high >= 0:
            sm,sp = PAIRS[grid.face_axes[f]]
            np.testing.assert_allclose(factors[low,sp]*result.side_face_means[low,sp],
                                       factors[high,sm]*result.side_face_means[high,sm],rtol=1e-12)
    # Volume-integrated outgoing currents cancel internally, independently of ADF.
    loss = np.einsum('ngh,nh->ng',solver.R,result.means)
    boundary_out = (grid.normal_signs[grid.boundary_sides,None]*result.currents[grid.boundary_faces]
                    *grid.face_areas[grid.boundary_faces,None]).sum(axis=0)
    np.testing.assert_allclose((grid.volumes[:,None]*(q0-loss)).sum(axis=0),boundary_out,rtol=1e-11,atol=1e-12)


def test_quadratic_multidimensional_leakage_independent_widths():
    grid = tensor_grid(([0.,.3,1.,2.],[0.,.8,2.],[0.,1.2]))
    solver = NEMDiffusionSolver(grid,*materials(grid,2))
    J = np.empty((len(grid.face_nodes),2))
    for f in range(len(J)):
        a = grid.face_axes[f]
        # Evaluate an axis-oriented polynomial at the face center.
        owners = grid.face_nodes[f];node = next(i for i in owners if i >= 0)
        center = grid.bounds[node].mean(axis=1)
        center[a] = grid.face_planes[f]
        J[f] = np.array([.1,-.2])*(center[a]*(1+center[(a+1)%3]**2))
    actual = solver._leakage(J)
    expected = np.zeros_like(actual)
    for i in range(grid.n_nodes):
        for a in range(3):
            samples,weights = independent_weights(grid,i,a)
            for t,(sm,sp) in enumerate(PAIRS):
                if t != a:
                    expected[i,a] += weights@(J[grid.node_faces[samples,sp]]-J[grid.node_faces[samples,sm]])/grid.widths[i,t]
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-14)
    assert np.max(abs(actual[:,:,1])) > 1e-4


def test_uniform_reflective_k_infinity_and_complete_normalization():
    grid = tensor_grid(([0.,.5,1.5],[0.,1.],[0.,1.]),'reflective')
    D = np.array([1.,.4]);sa = np.array([.08,.12]);ss = np.array([[3.,.03],[.01,5.]])
    nuf = np.array([.11,.09]);chi = np.array([1.,0.])
    solver = NEMDiffusionSolver(grid,D,sa,ss,nuf,chi)
    R = np.array([[.11,-.01],[-.03,.13]])
    exact = float(nuf@np.linalg.solve(R,chi))
    result = solver.solve_eigenvalue()
    assert result.k_eff == pytest.approx(exact,rel=2e-10)
    np.testing.assert_allclose(result.currents,0,atol=1e-12)
    np.testing.assert_allclose(result.moments,0,atol=1e-12)
    assert solver._production(result) == pytest.approx(1.,rel=1e-13)
    assert result.diagnostics['full_eigen_residual'] < 1e-10
    np.testing.assert_allclose(result.side_face_means,np.repeat(result.means[:,None,:],6,axis=1),atol=1e-12)


@pytest.mark.parametrize('groups',[2,4])
def test_eigenpair_full_dense_pencil_nonzero_spatial_fission_moments(groups):
    grid = tensor_grid(([0.,.7,1.8],[0.,1.],[0.,1.]))
    D,sa,ss,nuf,chi = materials(grid,groups)
    nuf[1] *= 1.4
    solver = NEMDiffusionSolver(grid,D,sa,ss,nuf,chi)
    A,F = full_augmented(grid,D,sa,ss,nuf,chi)
    values,vectors = np.linalg.eig(np.linalg.solve(A,F))
    dominant = int(np.argmax(abs(values)))
    assert abs(values[dominant].imag) < 1e-12 and values[dominant].real > 0
    result = solver.solve_eigenvalue(max_iterations=2000)
    assert result.k_eff == pytest.approx(values[dominant].real,rel=2e-9)
    expected = vectors[:,dominant].real
    N,G = result.means.shape
    production = np.sum(grid.volumes[:,None]*nuf*expected[:N*G].reshape(N,G))
    expected /= production
    np.testing.assert_allclose(full_state(result),expected,rtol=2e-7,atol=1e-10)
    assert result.diagnostics['minimum_mean'] > 0
    assert result.diagnostics['minimum_side_face_mean'] > 0
    assert result.diagnostics['minimum_profile_flux'] > 0
    assert np.max(abs(result.moments)) > 1e-5
    qmom = np.einsum('ngh,naph->napg',solver.fission,result.moments)
    assert np.max(abs(qmom)) > 1e-6
    residual = A@full_state(result)-F@full_state(result)/result.k_eff
    scale = abs(A)@abs(full_state(result))+abs(F)@abs(full_state(result))/result.k_eff
    assert np.max(abs(residual)/np.maximum(scale,np.finfo(float).tiny)) < 1e-9
    # Dropping spatial fission moments defines a measurably different eigenproblem.
    Fmean = F.copy();Fmean[N*G:,:] = 0
    wrong = np.max(np.linalg.eigvals(np.linalg.solve(A,Fmean)).real)
    assert abs(wrong-result.k_eff) > 1e-5


def test_reflective_zero_absorption_is_singular():
    grid = tensor_grid(([0.,1.],[0.,1.],[0.,1.]),'reflective')
    with pytest.raises(NEMSolverError,match='singular|condition'):
        NEMDiffusionSolver(grid,[1.],[0.],[[0.]],[0.],[0.])


def test_loss_lu_cached_and_configuration_guard(monkeypatch):
    import ndgpu.nem as module
    original = module.splu;calls = []
    def count(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(module,'splu',count)
    grid = tensor_grid(([0.,1.],[0.,1.],[0.,1.]))
    D,sa,ss,nuf,chi = materials(grid,2)
    solver = NEMDiffusionSolver(grid,D,sa,ss,nuf,chi)
    D[:] = 999  # caller-owned data cannot mutate the cached operator
    first = solver.solve_fixed_source([1.,.5])
    second = solver.solve_fixed_source([2.,1.])
    np.testing.assert_allclose(second.means,2*first.means,rtol=1e-12)
    assert len(calls) == 1
    with pytest.raises(ValueError):solver.D[0,0] = 2
    grid.widths = grid.widths.copy()*2
    with pytest.raises(ValueError,match='changed after cached'):
        solver.solve_fixed_source([1.,1.])


def test_negative_states_are_flagged_not_clipped():
    grid = tensor_grid(([0.,1.],[0.,1.],[0.,1.]))
    solver = NEMDiffusionSolver(grid,*materials(grid,1))
    result = solver.solve_fixed_source([-1.])
    assert result.diagnostics['minimum_mean'] < 0
    assert result.diagnostics['minimum_side_face_mean'] < 0
    assert result.diagnostics['minimum_profile_flux'] < 0
    assert result.diagnostics['negative_profile_count'] == 3
    assert np.all(result.means < 0)


def test_eigen_nonconvergence_preserves_latest_state():
    grid = tensor_grid(([0.,.7,1.8],[0.,1.],[0.,1.]))
    solver = NEMDiffusionSolver(grid,*materials(grid,2))
    with pytest.raises(NEMSolverError,match='did not converge') as exc:
        solver.solve_eigenvalue(max_iterations=1)
    assert exc.value.result is not None
    assert exc.value.result.k_eff > 0
    assert len(exc.value.diagnostics['power_iteration_history']) == 1
    assert 'minimum_profile_flux' in exc.value.diagnostics


@pytest.mark.parametrize('field,value',[
    ('D',[0.]),('sigma_a',[-.1]),('sigma_s',[[-.2]]),('nu_sigma_f',[-.1]),
    ('chi',[.5]),('D',[[1.,2.],[3.,4.]]),('D',[np.nan]),
])
def test_invalid_materials(field,value):
    grid = tensor_grid(([0.,1.],[0.,1.],[0.,1.]))
    args = dict(D=[1.],sigma_a=[.1],sigma_s=[[0.]],nu_sigma_f=[.1],chi=[1.])
    args[field] = value
    with pytest.raises(ValueError):NEMDiffusionSolver(grid,**args)


def test_external_factors_and_bad_sources():
    grid = tensor_grid(([0.,1.],[0.,1.],[0.,1.]))
    with pytest.raises(ValueError,match='external'):
        NEMDiffusionSolver(grid,*materials(grid,1),node_face_factors=np.ones((1,6,1))*1.1)
    solver = NEMDiffusionSolver(grid,*materials(grid,1))
    with pytest.raises(ValueError):solver.solve_fixed_source([[1.,2.]])
    with pytest.raises(ValueError):solver.solve_fixed_source([1.],moment_source=np.zeros((1,2,1)))
    with pytest.raises(ValueError):solver.solve_fixed_source([1.],boundary_drive=np.zeros((5,1)))
    with pytest.raises(ValueError):solver.solve_eigenvalue(max_iterations=0)
    with pytest.raises(ValueError):solver.solve_eigenvalue(initial_means=[-1.])


def test_cpu_only_inputs():
    class Device:
        __cuda_array_interface__ = {}
    grid = tensor_grid(([0.,1.],[0.,1.],[0.,1.]))
    with pytest.raises(TypeError,match='CPU'):
        NEMDiffusionSolver(grid,Device(),[.1],[[0.]],[.1],[1.])


def test_analytic_separable_quadratic_with_driven_robin_boundaries():
    grid = tensor_grid(([0.,.7,2.],[0.,.4,1.5],[0.,1.]))
    N,G = grid.n_nodes,2
    D = np.broadcast_to([1.,.4],(N,G)).copy()
    sa = np.broadcast_to([.1,.2],(N,G)).copy()
    ss = np.broadcast_to([[0.,.03],[.01,0.]],(N,G,G)).copy()
    nuf = np.zeros((N,G));chi = np.zeros((N,G))
    solver = NEMDiffusionSolver(grid,D,sa,ss,nuf,chi)
    linear = np.array([[.12,-.07],[.08,.11],[-.02,.03]])
    quadratic = np.array([[.02,.03],[-.01,.01],[.015,-.01]])
    centers = grid.bounds.mean(axis=2)
    means = np.broadcast_to([2.,1.5],(N,G)).copy()
    means += np.einsum('na,ag->ng',centers,linear)
    means += np.einsum('na,ag->ng',centers**2+grid.widths**2/12,quadratic)
    moments = np.empty((N,3,2,G))
    for a in range(3):
        moments[:,a,0] = grid.widths[:,a,None]*(linear[a]+2*centers[:,a,None]*quadratic[a])/12
        moments[:,a,1] = grid.widths[:,a,None]**2*quadratic[a]/60
    current = np.empty((len(grid.face_nodes),G))
    for f,a in enumerate(grid.face_axes):
        current[f] = -D[0]*(linear[a]+2*grid.face_planes[f]*quadratic[a])
    side_means = np.empty((N,6,G))
    for i in range(N):
        for a,(sm,sp) in enumerate(PAIRS):
            for side,end in ((sm,0),(sp,1)):
                x = grid.bounds[i,a,end]
                side_means[i,side] = means[i]+linear[a]*(x-centers[i,a])+quadratic[a]*(x*x-centers[i,a]**2-grid.widths[i,a]**2/12)
    q0 = np.einsum('ngh,nh->ng',solver.R,means)-2*D*quadratic.sum(axis=0)
    qm = np.einsum('ngh,naph->napg',solver.R,moments)
    drive = (grid.normal_signs[grid.boundary_sides,None]*current[grid.boundary_faces]
             -grid.boundary_alpha[:,None]*side_means[grid.boundary_nodes,grid.boundary_sides])
    result = solver.solve_fixed_source(q0,qm,drive)
    np.testing.assert_allclose(result.means,means,rtol=1e-12)
    np.testing.assert_allclose(result.moments,moments,rtol=1e-10,atol=1e-13)
    np.testing.assert_allclose(result.currents,current,rtol=1e-11,atol=1e-13)
    np.testing.assert_allclose(result.side_face_means,side_means,rtol=1e-12)


def test_local_response_condition_guard_and_material_replacement():
    grid = tensor_grid(([0.,1.],[0.,1.],[0.,1.]))
    with pytest.raises(NEMSolverError,match='local loss response condition'):
        NEMDiffusionSolver(grid,*materials(grid,2),condition_limit=1.)
    solver = NEMDiffusionSolver(grid,*materials(grid,2))
    solver.sigma_a = solver.sigma_a.copy()*2
    with pytest.raises(ValueError,match='changed after cached'):
        solver.solve_fixed_source([1.,1.])


def test_fixed_source_residual_rejection_preserves_state():
    grid = tensor_grid(([0.,.7,1.8],[0.,1.],[0.,1.]))
    solver = NEMDiffusionSolver(grid,*materials(grid,2),rtol=1e-20)
    with pytest.raises(NEMSolverError,match='physical residual') as exc:
        solver.solve_fixed_source([.3,.7],moment_source=np.full((3,2,2),.003))
    assert exc.value.result is not None
    assert 'profile_minima' in exc.value.diagnostics
