"""Independent pencil and compatibility checks for separate NEM controls."""
import numpy as np
import pytest
from garnet.nem import NEMDiffusionSolver
from test_nem import tensor_grid, materials, full_augmented


def solver():
    grid = tensor_grid(([0.,.7,1.8],[0.,1.],[0.,1.]))
    data = materials(grid,2)
    data[3][1] *= 1.4
    return NEMDiffusionSolver(grid,*data,rtol=1e-5),data


def test_requested_controls_against_independent_dense_pencil():
    s,data = solver()
    A,F = full_augmented(s.grid,*data)
    exact = np.linalg.eigvals(np.linalg.solve(A,F))
    exact = exact[np.argmax(abs(exact))].real
    result = s.solve_eigenvalue(tol_k=1e-6,tol_source=1e-5,tol_balance=1e-5)
    assert abs(result.k_eff-exact) < 1e-6
    d = result.diagnostics
    assert d['full_eigen_residual'] <= 1e-5
    assert d['absolute_k_change'] <= 1e-6
    assert d['relative_fission_source_change'] <= 1e-5
    assert np.max(abs(result.moments)) > 1e-5
    assert d['convergence_controls']['eigenvalue_change'] == 'absolute'
    assert d['power_iteration_history'][0]['absolute_k_change'] is None


def test_legacy_unchanged_default_and_explicit_none():
    s,_ = solver()
    a = s.solve_eigenvalue(rtol=1e-9)
    b = s.solve_eigenvalue(rtol=1e-9,tol_k=None,tol_source=None,tol_balance=None)
    assert a.k_eff == b.k_eff and a.iterations == b.iterations
    assert a.diagnostics['power_iteration_history'] == b.diagnostics['power_iteration_history']
    assert 'convergence_controls' not in a.diagnostics
    np.testing.assert_array_equal(a.moments,b.moments)


@pytest.mark.parametrize('name',['tol_k','tol_source','tol_balance'])
@pytest.mark.parametrize('value',[0,-1,np.nan,np.inf,True,[1.]])
def test_invalid_controls(name,value):
    s,_ = solver()
    with pytest.raises(ValueError,match=name):s.solve_eigenvalue(**{name:value})


def test_spatial_source_change_is_actually_gated():
    s,_ = solver()
    result = s.solve_eigenvalue(tol_k=1,tol_balance=1,tol_source=1e-9)
    history = result.diagnostics['power_iteration_history']
    assert len(history) > 2
    assert history[-2]['relative_fission_source_change'] > 1e-9
    assert history[-1]['relative_fission_source_change'] <= 1e-9
