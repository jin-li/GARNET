import numpy as np
import pytest
from garnet.kinetics_production import assemble_kinetics_production
from garnet.nem import NEMDiffusionSolver
from garnet.nem_higher_order import NEMHigherOrderDiffusionSolver
from test_nem import tensor_grid, materials


def test_zero_sampled_bin_and_delayed_tail_through_nem_projection():
    from garnet.nem_kinetics import solve_nem_adjoint, project_nem_kinetics
    grid = tensor_grid(([0.,.7,1.9],[0.,1.3],[0.,.8]))
    mat = materials(grid,2)
    mat[4][:] = [1.,0.]
    delayed = np.empty((2,2,2,2))
    delayed[:,0] = np.array([1-1e-8,1e-8])[None,:,None] * (mat[3]*[.005,.006])[:,None,:]
    delayed[:,1] = np.array([1-2e-8,2e-8])[None,:,None] * (mat[3]*[.001,.002])[:,None,:]
    production = assemble_kinetics_production(mat[3],mat[4],delayed)
    inv = np.array([[1e-7,2e-5],[1.3e-7,2.6e-5]])
    for kernel, accepted in ((None,False),(production.total,True)):
        solver = NEMHigherOrderDiffusionSolver(grid,*mat,degree=8,fission_matrix=kernel)
        fwd = solver.solve_eigenvalue(rtol=1e-12,max_iterations=2000)
        adj = solve_nem_adjoint(solver,fwd,rtol=1e-9)
        if accepted:
            result = project_nem_kinetics(solver,fwd,adj,inverse_velocities=inv,
                                          delayed_production=delayed,decay=[.08,.3])
            assert np.all(np.asarray(result.beta) > 0)
            assert result.generation_time > 0
        else:
            with pytest.raises(ValueError,match='prompt'):
                project_nem_kinetics(solver,fwd,adj,inverse_velocities=inv,
                                     delayed_production=delayed,decay=[.08,.3])


def test_delayed_tail_is_preserved_without_negative_prompt():
    nuf = np.array([[.02,.04], [0.,0.]])
    delayed = np.zeros((2,2,2,2))
    delayed[0,0] = np.outer([1-1e-8,1e-8], [.0001,.0003])
    delayed[0,1] = np.outer([.9,.1], [.00002,.00008])
    got = assemble_kinetics_production(nuf, [[1.,0.],[0.,0.]], delayed)
    np.testing.assert_allclose(got.total.sum(axis=1), nuf, rtol=1e-15)
    np.testing.assert_array_equal(got.delayed, delayed)
    assert np.all(got.prompt >= 0)
    assert got.total[0,1,0] > 0
    assert abs(np.linalg.det(got.total[0])) > 1e-10
    for x in (got.prompt, got.total, got.delayed):
        assert not x.flags.writeable
    delayed[:] = 0
    assert got.delayed.max() > 0


@pytest.mark.parametrize('kind', ['negative','excess','nan','shape','spectrum'])
def test_invalid_production_rejected(kind):
    nuf = np.array([[.02,.04]])
    chi = np.array([[1.,0.]])
    delayed = np.ones((1,2,2,2))*.0001
    if kind == 'negative': delayed[0,0,0,0] = -.01
    if kind == 'excess': delayed *= 1000
    if kind == 'nan': delayed[0,0,0,0] = np.nan
    if kind == 'shape': delayed = delayed[0]
    if kind == 'spectrum': chi *= .9
    with pytest.raises(ValueError):
        assemble_kinetics_production(nuf, chi, delayed)


@pytest.mark.parametrize('cls', [NEMDiffusionSolver,NEMHigherOrderDiffusionSolver])
def test_matrix_validation_and_legacy_equivalence(cls):
    grid = tensor_grid(([0.,.7,1.9],[0.,1.3],[0.,.8]))
    mat = materials(grid,2)
    legacy = cls(grid,*mat)
    explicit = cls(grid,*mat,fission_matrix=legacy.fission)
    assert not explicit.fission.flags.writeable
    np.testing.assert_array_equal(legacy.fission, explicit.fission)
    a,b = legacy.solve_eigenvalue(), explicit.solve_eigenvalue()
    assert a.k_eff == b.k_eff
    np.testing.assert_array_equal(a.means,b.means)
    for bad in (legacy.fission*1.01, -legacy.fission,
                legacy.fission.astype(complex)+1j, np.full_like(legacy.fission,np.nan)):
        with pytest.raises(ValueError): cls(grid,*mat,fission_matrix=bad)
    explicit.fission = explicit.fission.copy()
    explicit.fission[0,0,0] *= .9
    with pytest.raises(ValueError,match='changed'):
        explicit.solve_eigenvalue()
