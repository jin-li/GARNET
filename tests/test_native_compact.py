"""Bounded synthetic deferred operator checks; no production solve admission."""
from dataclasses import replace
import numpy as np
import pytest
from scipy import sparse
from garnet.corrected_native_moments import CorrectedNativeMomentOperator as Operator
from garnet.mesh import MeshGrid
from garnet.moment_materials import MomentMaterial
from test_corrected_mesh import mesh_grid
from test_corrected_native_moments import partial, cube3d
from test_native_sp5 import material


def fixture(family, permutation=None):
    mesh = mesh_grid(2, .37)
    # Two segments per physical face exercise repeated-owner and repeated-pair sums.
    faces = [(i, j, area/2, d) for i, j, area, d in mesh.faces for _ in range(2)]
    bfaces = [(i, area/2, d) for i, area, d in mesh.bfaces for _ in range(2)]
    mesh = replace(mesh, faces=faces, bfaces=bfaces,
        face_centers=np.repeat(mesh.face_centers, 2, axis=0),
        face_distances=np.repeat(mesh.face_distances, 2, axis=0),
        face_normals=np.repeat(mesh.face_normals, 2, axis=0),
        bface_centers=np.repeat(mesh.bface_centers, 2, axis=0),
        bface_normals=np.repeat(mesh.bface_normals, 2, axis=0),
        bface_tags=np.repeat(mesh.bface_tags, 2))
    mats = [material(i % 2, groups=3) for i in range(mesh.n_cells)]
    if permutation is not None:
        p = permutation
        mats = [MomentMaterial(m.total[p], m.chi[p], m.nu_sigma_f[p], m.kappa_fission[p], {}, {},
                {k: a[np.ix_(p, p)] for k, a in m.loss.items()}, {}, {}, dict(m.provenance), {}) for m in mats]
    bc = ['vacuum' if k//2 % 2 else 'reflective' for k in range(len(bfaces))]
    return MeshGrid(mesh, bc), mats


def reference_action(op, state):
    """Fit each field's gradient directly with lstsq, without stored stencils.

    This independently forms equations from geometry and materials and evaluates
    integrated currents with scalar loops, independent of compact block streams.
    """
    x = state.reshape(op.n_cells, -1)
    mesh = op.grid.mesh
    q, dim = x.shape[1], mesh.centroid.shape[1]
    I = np.eye(q)
    invD = [np.linalg.inv(D) for D in op.diffusion]
    patches = [[] for _ in x]
    def equation(blocks):
        return np.stack(blocks, axis=2).reshape(q, q*dim)
    for (i, j, _, _), normal, (di, dj) in zip(mesh.faces, mesh.face_normals, mesh.face_distances):
        r = mesh.centroid[j]-mesh.centroid[i]
        for owner, neighbor, vector, n, d in ((i, j, r, normal, dj), (j, i, -r, -normal, di)):
            jump = invD[neighbor]@op.diffusion[owner]-I
            A = equation([vector[k]*I+d*n[k]*jump for k in range(dim)])
            patches[owner].append((A, x[neighbor]-x[owner]))
    for (i, _, _), center, normal, spec in zip(mesh.bfaces, mesh.bface_centers, mesh.bface_normals,
                                               op.grid.classify_boundary()):
        r = center-mesh.centroid[i]
        B = op.B if spec == 'vacuum' else np.zeros((q, q))
        A = equation([normal[k]*op.diffusion[i]+r[k]*B for k in range(dim)]
                     if spec == 'vacuum' else [normal[k]*I for k in range(dim)])
        patches[i].append((A, -B@x[i]))
    H = []
    for patch in patches:
        A = np.concatenate([a for a, _ in patch]); rhs = np.concatenate([b for _, b in patch])
        scales = np.linalg.norm(A, axis=1)
        H.append(np.linalg.lstsq(A/scales[:, None], rhs/scales, rcond=None)[0].reshape(q, dim))
    H = np.array(H)
    out = np.array([r@v for r, v in zip(op.reaction, x)])
    for (i, j, area, _), center, n, (di, dj) in zip(mesh.faces, mesh.face_centers,
                                                   mesh.face_normals, mesh.face_distances):
        T = area*np.linalg.inv(di*invD[i]+dj*invD[j])
        current = T@(x[i]+H[i]@(center-mesh.centroid[i]-di*n)
                     -x[j]-H[j]@(center-mesh.centroid[j]+dj*n))
        out[i] += current; out[j] -= current
    for (i, area, d), center, n, spec in zip(mesh.bfaces, mesh.bface_centers, mesh.bface_normals,
                                            op.grid.classify_boundary()):
        if spec == 'vacuum':
            C = op.diffusion[i]/d
            K = area*op.B@np.linalg.inv(C+op.B)@C
            out[i] += K@(x[i]+H[i]@(center-mesh.centroid[i]-d*n))
    return out.reshape(state.shape)


@pytest.mark.parametrize('family', ['sp3', 'sp5'])
@pytest.mark.parametrize('chunk,batch', [(1, 1), (3, 5)])
def test_action_groups_and_stream(family, chunk, batch):
    grid, mats = fixture(family)
    assembled = Operator(grid, mats, family=family)
    compact = Operator(grid, mats, family=family, storage='deferred',
                       gradient_chunk_cells=chunk, batch_size=batch)
    assert np.linalg.norm(assembled.diffusion[0]@assembled.diffusion[1]
                          -assembled.diffusion[1]@assembled.diffusion[0]) > 1e-6
    rng = np.random.default_rng(316)
    for shape in ((compact.size,), compact.shape):
        u = rng.normal(size=shape)
        np.testing.assert_allclose(compact.action(u), assembled.assembled@u.ravel()
                                   if len(shape) == 1 else (assembled.assembled@u.ravel()).reshape(shape), rtol=1e-12, atol=3e-14)
        np.testing.assert_allclose(compact.action(u), reference_action(assembled, u), rtol=1e-12, atol=3e-14)
        np.testing.assert_allclose(compact.gradients(u), assembled.gradients(u), rtol=1e-12, atol=3e-14)
        for a, b in zip(compact.face_currents(u), assembled.face_currents(u)):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=3e-14)
        np.testing.assert_allclose(compact.fission_action(u), assembled.fission_action(u), rtol=0, atol=0)
        np.testing.assert_allclose(compact.linear_operator()@u.ravel(), assembled.action(u).ravel(), rtol=1e-12, atol=3e-14)
    full = np.zeros((compact.size, compact.size)); q = compact.n_groups*compact.n_moments
    for rows, cols, blocks in compact.iter_full_blocks():
        assert len(rows) <= batch
        for i, j, block in zip(rows, cols, blocks):
            full[i*q:(i+1)*q, j*q:(j+1)*q] += block
    np.testing.assert_allclose(full, assembled.assembled.toarray(), rtol=1e-12, atol=3e-14)
    for g in range(compact.n_groups):
        idx = np.arange(compact.size).reshape(compact.shape)[:, :, g].ravel()
        np.testing.assert_allclose(compact.group_principal_block(g).toarray(), full[np.ix_(idx, idx)], rtol=1e-12, atol=3e-14)
        np.testing.assert_allclose(compact.group_principal_block(g).toarray(),
                                   assembled.group_principal_block(g).toarray(), rtol=1e-12, atol=3e-14)
    # An actual internal cross-group product is nonzero and affects a selected block.
    candidates = [(T, np.einsum('adk,d->ak', h, t)) for a, b, T, t, _ in compact.compact.faces
                  for _, h in compact.compact._gradient_row(a)]
    sel = np.arange(0, q, compact.n_groups)
    assert max(np.linalg.norm((T@h)[np.ix_(sel, sel)]-T[np.ix_(sel, sel)]@h[np.ix_(sel, sel)])
               for T, h in candidates) > 1e-9


@pytest.mark.parametrize('family', ['sp3', 'sp5'])
def test_group_permutation(family):
    p = np.array([2, 0, 1])
    grid, mats = fixture(family); pgrid, pmats = fixture(family, p)
    a = Operator(grid, mats, family=family, storage='deferred', gradient_chunk_cells=1)
    b = Operator(pgrid, pmats, family=family, storage='deferred', gradient_chunk_cells=1)
    u = np.random.default_rng(8).normal(size=a.shape)
    np.testing.assert_allclose(b.action(u[:, :, p]), a.action(u)[:, :, p], rtol=1e-12, atol=3e-14)
    for g in range(3):
        np.testing.assert_allclose(b.group_principal_block(g).toarray(), a.group_principal_block(p[g]).toarray(), rtol=1e-12, atol=3e-14)


def test_allocation_contract_and_immutability(monkeypatch):
    import garnet.corrected_native_moments as native
    import garnet.native_compact as compact_module
    grid, mats = fixture('sp5')
    original_bsr = sparse.bsr_matrix
    seen = []
    def bsr(*args, **kwargs):
        matrix = original_bsr(*args, **kwargs)
        seen.append(matrix.shape)
        assert matrix.blocksize == (18, 9)
        assert matrix.shape == (18, 36)
        return matrix
    def no_csr(*args, **kwargs):
        raise AssertionError('global CSR forbidden')
    original_freeze = native._immutable
    def freeze(value):
        assert not isinstance(value, dict), 'gradient dictionary freeze forbidden'
        return original_freeze(value)
    original_svd = np.linalg.svd
    patch_shapes = []
    def svd(patch, **kwargs):
        # Four interior and four boundary equations per cell in this fixture;
        # reject a fit spanning more than one cell's patch.
        assert patch.shape == (72, 18)
        patch_shapes.append(patch.shape)
        return original_svd(patch, **kwargs)
    original_chunk_freeze = compact_module._frozen
    def freeze_chunk(value):
        if value.ndim == 3:
            assert value.shape[0] <= 5, 'whole-gradient freeze forbidden'
        return original_chunk_freeze(value)
    monkeypatch.setattr(sparse, 'bsr_matrix', bsr)
    monkeypatch.setattr(sparse, 'csr_matrix', no_csr)
    monkeypatch.setattr(native, '_immutable', freeze)
    monkeypatch.setattr(compact_module, '_immutable', freeze)
    monkeypatch.setattr(np.linalg, 'svd', svd)
    monkeypatch.setattr(compact_module, '_frozen', freeze_chunk)
    a = Operator(grid, mats, family='sp5', storage='deferred', gradient_chunk_cells=1, batch_size=2)
    assert len(seen) == len(patch_shapes) == grid.n_cells
    assert not hasattr(a, 'gradient_stencils') and not hasattr(a, '_assembled')
    for _, _, matrix in a.compact.gradient_chunks:
        for array in (matrix.data, matrix.indices, matrix.indptr):
            with pytest.raises(ValueError): array.setflags(write=True)
    for access in (lambda: a.assembled, lambda: setattr(a, 'assembled', None)):
        with pytest.raises(NotImplementedError, match='Global assembled'): access()
    a.action(np.ones(a.size))
    accounting = a.compact.payload_accounting()
    assert accounting['total_array_bytes'] == sum(v for k, v in accounting.items() if k != 'total_array_bytes')
    assert accounting['gradient_bytes'] > 0
    assert a.compact.provenance['solver_integration'] is False
    with pytest.raises(TypeError): a.compact.provenance['storage'] = 'assembled'


@pytest.mark.parametrize('storage', ['assembled', 'deferred'])
def test_failures(storage):
    grid, mats = fixture('sp3')
    for name in ('gradient_chunk_cells', 'batch_size'):
        for bad in (0, -1, True, 1.2):
            with pytest.raises(ValueError, match=name): Operator(grid, mats, storage=storage, **{name: bad})
    with pytest.raises(ValueError, match='ill-conditioned'):
        Operator(grid, mats, storage=storage, condition_limit=1.01)
    with pytest.raises(ValueError, match='rank-deficient'):
        Operator(MeshGrid(partial(mesh_grid(1, .3))), [mats[0]], storage=storage)
    a = Operator(grid, mats, storage=storage)
    for group in (-1, 3, True, .5):
        with pytest.raises(ValueError, match='group'): a.group_principal_block(group)
    with pytest.raises(ValueError, match='unknowns'): a.action(np.zeros(3))
    loss = dict(mats[0].loss); loss[1] = np.diag([1., 1., 1e-14])
    with pytest.raises(ValueError, match='L1'):
        Operator(grid, [replace(mats[0], loss=loss)]*grid.n_cells, storage=storage)


@pytest.mark.parametrize('mesh', [cube3d(), partial(mesh_grid(2, .3))])
@pytest.mark.parametrize('bc', ['vacuum', 'reflective'])
def test_empty_face_sets_and_three_dimensions(mesh, bc):
    grid = MeshGrid(mesh, bc)
    mats = [material(i % 2) for i in range(mesh.n_cells)]
    a = Operator(grid, mats, family='sp5')
    b = Operator(grid, mats, family='sp5', storage='deferred', gradient_chunk_cells=1, batch_size=1)
    u = np.random.default_rng(2).normal(size=a.size)
    np.testing.assert_allclose(b.action(u), a.assembled@u, rtol=1e-12, atol=3e-14)
    for x, y in zip(a.face_currents(u), b.face_currents(u)):
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=3e-14)
    for g in range(3):
        np.testing.assert_allclose(b.group_principal_block(g).toarray(), a.group_principal_block(g).toarray(), rtol=1e-12, atol=3e-14)


def test_group_block_exact_zero_pattern_matches_orthogonal_reflective_sp5():
    grid = MeshGrid(mesh_grid(3, 0), 'reflective')
    mats = [material(0, groups=3)]*grid.n_cells
    assembled = Operator(grid, mats, family='sp5')
    deferred = Operator(grid, mats, family='sp5', storage='deferred',
                        gradient_chunk_cells=2, batch_size=3)
    for group in range(3):
        expected = assembled.group_principal_block(group)
        actual = deferred.group_principal_block(group)
        expected.sort_indices(); actual.sort_indices()
        assert actual.nnz == expected.nnz
        assert not np.any(actual.data == 0)
        np.testing.assert_array_equal(actual.indptr, expected.indptr)
        np.testing.assert_array_equal(actual.indices, expected.indices)
        np.testing.assert_allclose(actual.toarray(), expected.toarray(), rtol=1e-12, atol=3e-14)
