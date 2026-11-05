"""Exact sparse group blocks and bounded temporary workspace; no assembly job."""
import weakref

import numpy as np
import pytest
from scipy import sparse

import garnet.coupled_moment_solver as driver
from garnet.corrected_native_moments import CorrectedNativeMomentOperator
from garnet.coupled_moments import CoupledMomentOperator, OrthogonalMesh
from garnet.mesh import MeshGrid
from test_corrected_mesh import mesh_grid
from test_corrected_native_moments import representable
from test_coupled_moments import material
from test_native_sp5 import material as sp5_material


def exact_arrays(actual, expected):
    assert actual.shape == expected.shape and actual.format == expected.format
    for key in ('data', 'indices', 'indptr'):
        a, b = getattr(actual,key), getattr(expected,key)
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a,b)
        assert a.tobytes() == b.tobytes()  # Includes signed explicit zero bits.


def compare(matrix, groups, group, **kwargs):
    rows = np.arange(group,matrix.shape[0],groups)
    expected = matrix[rows,:][:,rows]
    actual = driver._group_principal_block(matrix,groups,group,**kwargs)
    exact_arrays(actual,expected)
    exact_arrays(actual.tocsc(copy=True),expected.tocsc(copy=True))
    assert not np.shares_memory(actual.data,matrix.data)


@pytest.mark.parametrize('family', ['sp3','sdp1','sp5'])
@pytest.mark.parametrize('bc', ['reflective','vacuum'])
def test_native_noncommuting_blocks_exact_scipy_slices(family,bc):
    mesh = mesh_grid(3,.31)
    make = sp5_material if family == 'sp5' else material
    pair = [make(0),make(1)]
    operator = CorrectedNativeMomentOperator(MeshGrid(mesh,bc),[pair[i%2] for i in range(mesh.n_cells)],family=family)
    assert np.linalg.norm(operator.diffusion[0]@operator.diffusion[1]-operator.diffusion[1]@operator.diffusion[0]) > 1e-5
    for group in range(operator.n_groups):
        compare(operator.assembled,operator.n_groups,group,max_rows=3,max_nnz=50)


@pytest.mark.parametrize('groups,moments', [(1,2),(3,2),(7,2),(7,3)])
@pytest.mark.parametrize('index64', [False,True])
def test_random_sparse_blocks_explicit_zeros_empty_rows_and_index_width(groups,moments,index64):
    size = 4*groups*moments
    rng = np.random.default_rng(57)
    matrix = sparse.random(size,size,density=.24,random_state=rng,format='csr')
    matrix.data[::4] = -0.
    # Retain an empty row and a row with entries but no principal-group columns.
    row = size//2
    matrix.data = np.delete(matrix.data,np.s_[matrix.indptr[row]:matrix.indptr[row+1]])
    matrix.indices = np.delete(matrix.indices,np.s_[matrix.indptr[row]:matrix.indptr[row+1]])
    count = matrix.indptr[row+1]-matrix.indptr[row]
    matrix.indptr[row+1:] -= count
    if index64:
        matrix.indices = matrix.indices.astype(np.int64)
        matrix.indptr = matrix.indptr.astype(np.int64)
    before = [getattr(matrix,key).copy() for key in ('data','indices','indptr')]
    for group in range(groups):
        compare(matrix,groups,group,max_rows=4,max_nnz=40)
    for key,expected in zip(('data','indices','indptr'),before):
        np.testing.assert_array_equal(getattr(matrix,key),expected)


@pytest.mark.parametrize('index64', [False,True])
def test_unsorted_duplicates_missing_diagonals_and_all_empty(index64):
    # Several empty rows, unsorted and duplicate columns, explicit +/- zeros.
    matrix = sparse.csr_matrix((
        np.array([0.,-0.,3.,4.,0.,1.,-2.,-0.,4.,9.,5.,7.,8.]),
        np.array([9,0,6,0,11,2,4,6,0,9,1,10,0]),
        np.array([0,4,4,6,7,7,7,10,11,11,11,13,13])),shape=(12,12))
    if index64:
        matrix.indices = matrix.indices.astype(np.int64)
        matrix.indptr = matrix.indptr.astype(np.int64)
    for group in range(3):
        compare(matrix,3,group,max_rows=2,max_nnz=2)
        compare(sparse.csr_matrix((12,12)),3,group,max_rows=2,max_nnz=2)
    compare(sparse.csr_matrix((0,0)),1,0)


def test_selected_row_chunks_obey_row_and_source_nnz_bounds(monkeypatch):
    matrix = sparse.csr_matrix(np.ones((84,84)))
    original = sparse.csr_matrix.__getitem__
    calls = []
    def getitem(self,key):
        assert self is matrix  # No second wide column slice on an intermediate.
        rows,columns = key
        assert columns == slice(None)
        assert len(rows) <= 5
        source_nnz = int(np.sum(self.indptr[rows+1]-self.indptr[rows]))
        assert source_nnz <= 180 or len(rows) == 1
        calls.append(tuple(rows))
        return original(self,key)
    monkeypatch.setattr(sparse.csr_matrix,'__getitem__',getitem)
    actual = driver._group_principal_block(matrix,7,3,max_rows=5,max_nnz=180)
    assert actual.shape == (12,12) and actual.nnz == 144
    assert len(calls) == 12  # Six two-row chunks on each of two passes.
    assert calls[:6] == calls[6:]


@pytest.mark.parametrize('kwargs', [
    {'groups':0,'group':0}, {'groups':True,'group':0}, {'groups':3.,'group':0},
    {'groups':3,'group':-1}, {'groups':3,'group':3}, {'groups':3,'group':True},
    {'groups':3,'group':1.5}, {'groups':3,'group':0,'max_rows':0},
    {'groups':3,'group':0,'max_nnz':False}, {'groups':3,'group':0,'max_rows':1.5},
])
def test_invalid_group_or_chunk_bounds(kwargs):
    with pytest.raises(ValueError):
        driver._group_principal_block(sparse.eye(12,format='csr'),**kwargs)


@pytest.mark.parametrize('matrix', [sparse.eye(12,format='csc'), sparse.csr_matrix((12,11)), sparse.eye(11,format='csr')])
def test_invalid_native_csr_shape(matrix):
    with pytest.raises(ValueError,match='Square native CSR'):
        driver._group_principal_block(matrix,3,0)


@pytest.mark.parametrize('phase', ['extraction','factorization'])
def test_memory_failure_has_group_completed_count_and_releases_prior_block(monkeypatch,phase):
    a = CoupledMomentOperator(OrthogonalMesh([1.,2.],[[0.],[1.]]),[representable(3,0)]*2)
    extract = driver._group_principal_block
    extract_calls, factor_calls, block_refs = [], [], []
    cause = MemoryError('synthetic capped allocation')
    def extraction(*args,**kwargs):
        if extract_calls:
            assert block_refs[0]() is None  # Prior CSC gone before next allocation.
            if phase == 'extraction':
                raise cause
        extract_calls.append(True)
        return extract(*args,**kwargs)
    class Factor:
        nnz = 1
        def solve(self,rhs):
            return rhs
    def factor(block,**kwargs):
        if factor_calls and phase == 'factorization':
            raise cause
        assert kwargs == dict(drop_tol=1e-6,fill_factor=20)
        block_refs.append(weakref.ref(block));factor_calls.append(True)
        return Factor()
    def forbidden(*args,**kwargs):
        raise AssertionError('No GMRES or fallback after failed group setup')
    monkeypatch.setattr(driver,'_group_principal_block',extraction)
    monkeypatch.setattr(driver,'spilu',factor)
    monkeypatch.setattr(driver,'gmres',forbidden)
    with pytest.raises(MemoryError,match=f'{phase} failed for group 1 after 1 completed factors; no solver fallback') as caught:
        driver.CoupledMomentEigenSolver(a).solve(preconditioner='group_ilu')
    assert caught.value.__cause__ is cause
