"""Synthetic CPU group-Jacobi ILU checks; no physical-admission claim."""
import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import eig

import garnet.coupled_moment_solver as driver
from garnet.coupled_moments import CoupledMomentOperator, OrthogonalMesh
from test_coupled_moments import material, oracle
from test_corrected_native_moments import op, orthogonal, representable
from test_corrected_mesh import mesh_grid


@pytest.mark.parametrize('group_order',[(0,1,2),(2,0,1)])
def test_group_indexing_permutation_scatter_and_frozen_factors(monkeypatch,group_order):
    # Distinct blocks require pivoting; off-group entries must reach GMRES.
    groups=3;cells=2
    a=CoupledMomentOperator(OrthogonalMesh([1.,2.],[[0.],[1.]]),
                            [representable(groups,0)]*cells)
    base=np.array([[0.,3.,0.,1.],[2.,0.,4.,0.],[0.,5.,6.,1.],[4.,0.,2.,8.]])
    indices=[np.arange(a.size).reshape(a.shape)[:,:,g].ravel() for g in range(groups)]
    blocks=[base*(g+1)+np.diag([0.,0.,g*.2,g*.4]) for g in group_order]
    matrix=np.full((a.size,a.size),.013)
    for index,block in zip(indices,blocks):matrix[np.ix_(index,index)]=block
    a.assembled=sparse.csr_matrix(matrix)
    original_matrix=a.assembled
    original_factor=driver.spilu;calls=[];applications=[]

    class Factor:
        def __init__(self,factor):self.factor=factor
        @property
        def nnz(self):return self.factor.nnz
        def solve(self,rhs):return self.factor.solve(rhs)
        @property
        def L(self):raise AssertionError('must not materialize L')
        @property
        def U(self):raise AssertionError('must not materialize U')

    def factor(block,**kwargs):
        assert block.format=='csc' and block.shape==(2*cells,2*cells)
        assert not np.shares_memory(block.data,a.assembled.data)
        assert kwargs==dict(drop_tol=1e-6,fill_factor=20)
        np.testing.assert_array_equal(block.toarray(),blocks[len(calls)])
        fact=original_factor(block,**kwargs)
        assert np.any(fact.perm_r!=np.arange(2*cells))
        calls.append(block)
        return Factor(fact)

    # A full-size CSC snapshot would defeat the memory purpose of this mode.
    original_csc=sparse.csr_matrix.tocsc
    def csc(matrix,*args,**kwargs):
        assert matrix.shape!=(a.size,a.size)
        return original_csc(matrix,*args,**kwargs)

    def gmres(matrix,rhs,*,M,**kwargs):
        assert matrix is original_matrix
        np.testing.assert_array_equal(matrix.toarray(),matrix_before)
        vector=np.arange(1.,a.size+1)
        expected=np.empty(a.size)
        for index,block in zip(indices,blocks):expected[index]=np.linalg.solve(block,vector[index])
        np.testing.assert_allclose(M@vector,expected,rtol=1e-13,atol=1e-13)
        # Matrix changes after setup cannot change any of the group solves.
        matrix.data*=1.7
        np.testing.assert_allclose(M@vector,expected,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(M@(2*vector),2*expected,rtol=1e-13,atol=1e-13)
        applications.append(True)
        return np.linalg.solve(matrix.toarray(),rhs),0

    matrix_before=matrix.copy()
    monkeypatch.setattr(driver,'spilu',factor)
    monkeypatch.setattr(driver,'gmres',gmres)
    monkeypatch.setattr(sparse.csr_matrix,'tocsc',csc)
    result=driver.CoupledMomentEigenSolver(a).solve(preconditioner='group_ilu')
    assert len(calls)==groups and len(applications)==1
    assert not result.converged  # Deliberately unrelated assembly fails true action.
    metadata=result.provenance['preconditioner_metadata']
    assert '2*G*i+g' in metadata['group_cell_ordering']
    assert metadata['matrix_nnz']==np.count_nonzero(matrix_before)
    assert metadata['setup_seconds']>0
    assert [s['group'] for s in metadata['groups']]==list(range(groups))
    for stats,block in zip(metadata['groups'],blocks):
        assert stats['dimension']==2*cells
        assert stats['matrix_nnz']==np.count_nonzero(block)
        assert stats['superlu_nnz']>0 and stats['setup_seconds']>0
    assert 'not an explicit' in metadata['superlu_nnz_caveat']


@pytest.mark.parametrize('family',['sp3','sdp1'])
def test_corrected_multigroup_dense_monolithic_and_exactly_g_setups(family,monkeypatch):
    mesh=mesh_grid(2,0)
    # Introduce genuine intergroup coupling in all native loss orders.
    a=op(mesh,[material(i%2) for i in range(mesh.n_cells)],family)
    original=driver.spilu;calls=[]
    def factor(matrix,**kwargs):
        calls.append(matrix.shape)
        return original(matrix,**kwargs)
    monkeypatch.setattr(driver,'spilu',factor)
    result=driver.CoupledMomentEigenSolver(a).solve(preconditioner='group_ilu')
    assert result.converged and result.iterations>1
    assert calls==[(2*mesh.n_cells,)*2]*a.shape[2]
    assert result.full_residual<=1e-9
    assert all(h['linear_residual']<=1e-11 for h in result.history)
    assert not result.provenance['physical_admission']
    assert not result.provenance['fallback'] and not result.provenance['scattering_RHS']
    mono=driver.CoupledMomentEigenSolver(a).solve(preconditioner='ilu')
    assert mono.converged
    F=np.column_stack([a.fission_action(e) for e in np.eye(a.size)])
    values,vectors=eig(F,oracle(orthogonal(mesh),a.materials,family))
    index=np.argmax(values.real);state=vectors[:,index].real
    norm=sum(v*np.dot(m.nu_sigma_f,p) for v,m,p in zip(a.volumes,a.materials,a.scalar_flux(state)))
    assert result.k_eff==pytest.approx(values[index].real,rel=2e-9)
    assert result.k_eff==pytest.approx(mono.k_eff,rel=2e-9)
    np.testing.assert_allclose(result.moments.ravel(),state/norm,rtol=2e-8,atol=1e-10)
    np.testing.assert_allclose(result.moments,mono.moments,rtol=2e-8,atol=1e-10)
    a.assembled=a.assembled*1.1
    failed=driver.CoupledMomentEigenSolver(a).solve(preconditioner='group_ilu')
    assert not failed.converged and failed.reason=='inner_failure_or_full_linear_residual'
    assert failed.history[0]['gmres_info']==0
    assert failed.history[0]['linear_residual']>.01


@pytest.mark.parametrize('family',['sp3','sdp1'])
def test_one_group_equivalence_to_monolithic_ilu(family,monkeypatch):
    a=op(mesh_grid(2,.2),[representable(1,i) for i in range(4)],family)
    original=driver.gmres;applications=[]
    def gmres(*args,**kwargs):
        applications.append(kwargs['M']@np.arange(1.,a.size+1))
        return original(*args,**kwargs)
    monkeypatch.setattr(driver,'gmres',gmres)
    group=driver.CoupledMomentEigenSolver(a).solve(preconditioner='group_ilu')
    count=len(applications)
    mono=driver.CoupledMomentEigenSolver(a).solve(preconditioner='ilu')
    assert group.converged and mono.converged
    for application in applications:
        np.testing.assert_array_equal(application,applications[0])
    assert count==group.iterations and len(applications)-count==mono.iterations
    assert group.k_eff==pytest.approx(mono.k_eff,rel=1e-13)
    np.testing.assert_allclose(group.moments,mono.moments,rtol=1e-13,atol=1e-13)


@pytest.mark.parametrize('failure',[RuntimeError,ValueError])
def test_group_failure_propagates_without_gmres_or_fallback(monkeypatch,failure):
    a=op(mesh_grid(2,0),[material()]*4)
    original=driver.spilu;calls=[]
    def factor(matrix,**kwargs):
        calls.append(matrix.shape)
        if len(calls)==2:raise failure('synthetic group factor failure')
        return original(matrix,**kwargs)
    def forbidden(*args,**kwargs):raise AssertionError('GMRES must not run after setup failure')
    monkeypatch.setattr(driver,'spilu',factor)
    monkeypatch.setattr(driver,'gmres',forbidden)
    with pytest.raises(RuntimeError,match='group 1; no solver fallback') as caught:
        driver.CoupledMomentEigenSolver(a).solve(preconditioner='group_ilu')
    assert isinstance(caught.value.__cause__,failure)
    assert 'synthetic group factor failure' in str(caught.value.__cause__)
    assert len(calls)==2
