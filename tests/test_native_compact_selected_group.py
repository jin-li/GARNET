"""Exact selected-channel extraction; synthetic only, no assembly resource claim."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from garnet.native_compact import CompactNativeStorage
from garnet.corrected_native_moments import CorrectedNativeMomentOperator as Operator
from garnet.mesh import MeshGrid
from test_native_compact import fixture
from test_native_sp5 import material
from test_corrected_mesh import mesh_grid


def old_group(storage, group):
    """Pre-specialization full-block stream and identical sparse accumulation."""
    select = np.arange(group, storage.q, storage.op.n_groups)
    rows = [{} for _ in range(storage.op.n_cells)]
    for ii, jj, blocks in storage.iter_full_blocks():
        for i, j, block in zip(ii, jj, blocks):
            value = block[np.ix_(select, select)]
            rows[i][j] = rows[i].get(j, 0)+value
    data, indices, indptr = [], [], [0]
    for row in rows:
        for col in sorted(row):
            indices.append(col); data.append(row[col])
        indptr.append(len(data))
    n = storage.op.n_cells*storage.op.n_moments
    result = sparse.bsr_matrix((np.asarray(data), np.asarray(indices), np.asarray(indptr)), shape=(n, n)).tocsr()
    result.eliminate_zeros()
    return result


def assert_contribution_parity(storage, group):
    select = np.arange(group, storage.q, storage.op.n_groups)
    for (i,j,full),(a,b,selected) in zip(storage._contributions(),storage._group_contributions(select),strict=True):
        assert (i,j)==(a,b)
        assert selected.shape==(storage.op.n_moments,storage.op.n_moments)
        np.testing.assert_allclose(selected,full[np.ix_(select,select)],rtol=2e-13,atol=3e-14)
    expected=old_group(storage,group);actual=storage.group_principal_block(group)
    np.testing.assert_allclose(actual.toarray(),expected.toarray(),rtol=2e-13,atol=3e-14)
    assert not np.any(actual.data==0)


@pytest.mark.parametrize('family',['sp3','sp5'])
@pytest.mark.parametrize('chunk,batch',[(1,1),(3,5)])
def test_physical_skew_mixed_boundaries_every_group(family,chunk,batch):
    grid,mats=fixture(family)
    # Strengthen nonsymmetric cross-group coupling while preserving odd invertibility.
    mats=[replace(m,loss={order: matrix+np.array([[0.,.11,-.07],[-.19,0.,.16],[.09,-.13,0.]])
                          for order,matrix in m.loss.items()}) for m in mats]
    op=Operator(grid,mats,family=family,storage='deferred',gradient_chunk_cells=chunk,batch_size=batch)
    assert set(grid.classify_boundary())=={'reflective','vacuum'}
    assert any(np.linalg.norm(t)>1e-3 for _,_,_,t,_ in op.compact.faces)
    assert np.linalg.norm(op.diffusion[0]-op.diffusion[0].T)>1e-3
    for group in range(op.n_groups):assert_contribution_parity(op.compact,group)
    assert not hasattr(op,'_assembled')


def synthetic(groups,moments):
    """Arbitrary dense nonsymmetric channels, independent of a moment family."""
    rng=np.random.default_rng(43+groups+moments);q=groups*moments
    obj=CompactNativeStorage.__new__(CompactNativeStorage)
    obj.q=q;obj.dim=2;obj.batch_size=3
    obj.op=SimpleNamespace(n_groups=groups,n_moments=moments,n_cells=2,
        reaction=[rng.normal(size=(q,q)),rng.normal(size=(q,q))])
    obj.faces=((0,1,rng.normal(size=(q,q)),np.array([.3,-.2]),np.array([-.1,.4])),)
    obj.boundary=((0,rng.normal(size=(q,q)),np.array([.1,.2])),
                  (1,np.zeros((q,q)),np.array([-.2,.1])))
    gradients={i:[(j,rng.normal(size=(q,2,q))) for j in (0,1)] for i in (0,1)}
    obj._gradient_row=lambda cell:iter(gradients[cell])
    return obj


@pytest.mark.parametrize('groups,moments',[(1,1),(4,1),(4,2),(5,3),(3,4),(23,2)])
def test_generic_moment_counts_and_23_groups(groups,moments):
    storage=synthetic(groups,moments)
    for group in range(groups):assert_contribution_parity(storage,group)


def test_adversarial_internal_offgroup_path_survives():
    storage=synthetic(2,2)
    T=np.zeros((4,4));T[0,1]=2.
    H=np.zeros((4,2,4));H[1,0,0]=3.
    storage.faces=((0,1,T,np.array([1.,0.]),np.zeros(2)),)
    storage.boundary=();storage.op.reaction=[np.zeros((4,4))]*2
    storage._gradient_row=lambda cell:iter([(0,H)]) if cell==0 else iter(())
    select=np.array([0,2]);projected=np.einsum('adk,d->ak',H,np.array([1.,0.]))
    assert (T[np.ix_(select,select)]@projected[np.ix_(select,select)])[0,0]==0.
    result=storage.group_principal_block(0).toarray()
    assert result[0,0]==6. and result[2,0]==-6.
    assert_contribution_parity(storage,0)


def test_selected_path_never_calls_full_stream_and_projects_before_contraction(monkeypatch):
    storage=synthetic(23,3)
    expected=old_group(storage,7).toarray()
    def forbidden(*args,**kwargs):raise AssertionError('full contribution path forbidden')
    monkeypatch.setattr(storage,'iter_full_blocks',forbidden)
    monkeypatch.setattr(storage,'_contributions',forbidden)
    original=np.einsum;seen=[]
    def einsum(expression,coefficients,displacement):
        assert expression=='adk,d->ak'
        assert coefficients.shape==(69,2,3)  # q x dimension x selected inputs, never q x dimension x q
        seen.append(coefficients.shape)
        return original(expression,coefficients,displacement)
    monkeypatch.setattr(np,'einsum',einsum)
    actual=storage.group_principal_block(7).toarray()
    assert seen
    np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=3e-14)


@pytest.mark.parametrize('family',['sp3','sp5'])
def test_23_group_skew_operator_and_assembled_reference(family):
    grid=MeshGrid(mesh_grid(2,.31),'reflective')
    mats=[material(i%2,groups=23) for i in range(grid.n_cells)]
    op=Operator(grid,mats,family=family,storage='deferred',gradient_chunk_cells=1,batch_size=2)
    for group in range(23):assert_contribution_parity(op.compact,group)
    # A small full reference is permitted only here; the production selected path remains deferred.
    reference=Operator(grid,mats,family=family)
    for group in (0,11,22):
        np.testing.assert_allclose(op.group_principal_block(group).toarray(),
            reference.group_principal_block(group).toarray(),rtol=2e-13,atol=3e-14)


@pytest.mark.parametrize('bad',[-1,2,True,.5])
def test_invalid_group_rejected_before_stream(bad):
    with pytest.raises(ValueError,match='group'):synthetic(2,2).group_principal_block(bad)
