"""Tiny manufactured fixed-source proofs; no reactor data or eigeniterations."""
from dataclasses import FrozenInstanceError
import hashlib
import sys
import weakref

import numpy as np
import pytest
from scipy.special import eval_legendre

import garnet.native_grouped_fixed_source as grouped_module
from garnet.mesh import assemble_mesh
from garnet.moment_materials import MomentMaterial
from garnet.native_angular_source import NativeAngularSource, ProductQuadrature
from garnet.native_grouped_fixed_source import GroupedFixedSource
from garnet.native_mesh_streaming import NativeMeshStreaming


def manufactured(lmax=1, orientation='to-major', cells=2, total=None):
    points = np.array([(x, y) for y in (0., 1.) for x in range(cells+1)], dtype=float)
    mesh = assemble_mesh(points, [(c, c+1, cells+2+c, cells+1+c) for c in range(cells)],
                         np.arange(cells))
    materials = []
    for mid in range(cells):
        p = [np.array([[.25, .19], [.11, .31]])*(1+mid*.1)]
        p += [np.array([[.003, -.001], [.002, -.004]])*l for l in range(1, lmax+1)]
        materials.append(MomentMaterial(
            np.array([1.2, 1.7]) if total is None else np.asarray(total, dtype=float),
            np.array([1.4, .3]), np.zeros(2), np.zeros(2), {},
            {l: matrix.T.copy() if orientation == 'from-major' else matrix.copy()
             for l, matrix in enumerate(p)}, {}, {}, {},
            {'orientation_contract': orientation, 'energy_bounds_mev': [20., 1., 0.]}, {}))
    source = NativeAngularSource(ProductQuadrature(lmax), tuple(materials), np.arange(cells))
    return NativeMeshStreaming(mesh, source)


def boundary(s, kind):
    return tuple('vacuum' if kind == 'vacuum' or (kind == 'mixed' and n[0] < -.5)
                 else 'reflective' for n in s.boundary_normals)


def dense_matrix(s, bc):
    """Literal physical angular kernel + face rows, independent of group ops.

    The Legendre addition theorem builds scattering directly from directions,
    never via project_group, source.scatter, or either transport action.
    """
    nc, na, ng = s.shape
    matrix = np.zeros((nc*na*ng, nc*na*ng))
    q = s.quadrature
    def index(c, n, g):
        return (c*na+n)*ng+g
    for c, mid in enumerate(s.source.cell_material_ids):
        material = s.source.materials[mid]
        for n in range(na):
            for g in range(ng):
                matrix[index(c, n, g), index(c, n, g)] += s.area[c]*material.total[g]
            for incident in range(na):
                cosine = np.dot(q.directions[n], q.directions[incident])
                for l in range(q.lmax+1):
                    coefficient = s.area[c]*q.weights[incident]*(2*l+1)*eval_legendre(l, cosine)/(4*np.pi)
                    raw = material.production[l]
                    canonical = raw.T if material.provenance['orientation_contract'] == 'from-major' else raw
                    for g in range(ng):
                        for gp in range(ng):
                            matrix[index(c, n, g), index(c, incident, gp)] -= coefficient*canonical[g, gp]
    for (left, right), area, normal in zip(s.owners, s.measures, s.normals):
        for n in range(na):
            speed = sum(q.directions[n, a]*normal[a] for a in (0, 1))
            upstream = left if speed >= 0 else right
            for g in range(ng):
                matrix[index(left, n, g), index(upstream, n, g)] += area*speed
                matrix[index(right, n, g), index(upstream, n, g)] -= area*speed
    for face, (owner, area, normal) in enumerate(zip(s.boundary_owners, s.boundary_measures, s.boundary_normals)):
        for n in range(na):
            speed = sum(q.directions[n, a]*normal[a] for a in (0, 1))
            if speed < 0 and bc[face] == 'reflective':
                reflected = q.directions[n].copy()
                reflected[:2] -= 2*np.dot(reflected[:2], normal)*normal
                column = int(np.argmin(np.linalg.norm(q.directions-reflected, axis=1)))
                assert np.linalg.norm(q.directions[column]-reflected) < 1e-12
            elif speed > 0:
                column = n
            else:
                continue
            for g in range(ng):
                matrix[index(owner, n, g), index(owner, column, g)] += area*speed
    return matrix


def solver(s, kind='mixed', **kwargs):
    controls = dict(inner_tolerance=1e-11, full_tolerance=1e-9, restart=100,
                    max_inner=200, max_sweeps=100)
    controls.update(kwargs)
    return GroupedFixedSource(s, boundary(s, kind), **controls)


def test_one_cell_two_group_reflective_exact_upscatter_and_cold_start():
    s = manufactured(1, cells=1)
    op = solver(s, 'reflective')
    source = np.broadcast_to(np.array([.8, .5]), s.shape).copy()
    expected = np.linalg.solve(np.diag(s.total[0])-s.source.production[0][0], [.8, .5])
    result = op.solve(source)
    assert result.converged, result.reason
    np.testing.assert_allclose(result.angular_flux, np.broadcast_to(expected, s.shape), atol=2e-9, rtol=2e-9)
    assert len(result.history) > 1  # Upstream group must revisit later-group scatter.
    assert result.history[0]['full_residual'] > op.full_tolerance
    assert all(row['completed_groups'] == 2 for row in result.history)
    assert all(all(step <= op.max_inner for step in row['inner_steps']) for row in result.history)
    assert result.provenance['initial_field'] == 'zero'
    assert result.provenance['group_order'] == (0, 1)
    assert not result.provenance['physical_admission']


@pytest.mark.parametrize('lmax,orientation', [(0, 'to-major'), (1, 'from-major'), (7, 'to-major')])
@pytest.mark.parametrize('kind', ['mixed', 'reflective'])
def test_signed_heterogeneous_against_independent_dense(lmax, orientation, kind):
    # Keep the P7 dense oracle at one cell (256 coupled unknowns).
    s = manufactured(lmax, orientation, cells=1 if lmax == 7 else 2)
    op = solver(s, kind)
    matrix = dense_matrix(s, op.boundary_conditions)
    rng = np.random.default_rng(913)
    storage = rng.normal(size=(s.shape[0], s.shape[1], 4))
    source = storage[:, ::-1, ::2]
    source.setflags(write=False)
    original = storage.copy()
    raw = tuple({l: p.copy() for l, p in m.production.items()} for m in s.source.materials)
    chi = tuple(m.chi.copy() for m in s.source.materials)
    expected = np.linalg.solve(matrix, source.ravel()).reshape(s.shape)
    result = op.solve(source)
    assert result.converged, (result.reason, result.history[-1] if result.history else None)
    np.testing.assert_allclose(result.angular_flux, expected, atol=3e-9, rtol=3e-9)
    residual = np.linalg.norm(matrix @ result.angular_flux.ravel()-source.ravel())/np.linalg.norm(source)
    assert residual <= op.full_tolerance
    assert result.full_residual == pytest.approx(residual, abs=2e-13)
    assert np.min(result.angular_flux) < 0  # Raw signed field is preserved.
    np.testing.assert_array_equal(storage, original)
    for mid, material in enumerate(s.source.materials):
        np.testing.assert_array_equal(material.chi, chi[mid])
        for l, p in material.production.items():
            np.testing.assert_array_equal(p, raw[mid][l])
    np.testing.assert_array_equal(s.source.energy_bounds_mev, [20., 1., 0.])


def test_group_accuracy_alone_cannot_accept_one_sweep_and_terminal_is_retained():
    s = manufactured(1, cells=1)
    op = solver(s, 'reflective', max_sweeps=1)
    source = np.ones(s.shape)
    result = op.solve(source)
    assert not result.converged and result.reason == 'max_sweeps'
    assert result.history[0]['completed_groups'] == 2
    assert max(result.history[0]['group_residuals']) <= op.inner_tolerance
    assert result.full_residual > op.full_tolerance
    assert np.any(result.angular_flux != 0)
    matrix = dense_matrix(s, op.boundary_conditions)
    residual = np.linalg.norm(matrix @ result.angular_flux.ravel()-source.ravel())/np.linalg.norm(source)
    assert result.full_residual == pytest.approx(residual, abs=3e-15)


def test_inner_step_limit_retains_finite_candidate_without_retry():
    s = manufactured(1)
    op = solver(s, restart=1, max_inner=1)
    source = np.random.default_rng(6).normal(size=s.shape)
    result = op.solve(source)
    assert not result.converged
    assert result.reason in ('group_gmres_failed', 'true_group_residual_failed')
    assert len(result.history) == 1
    assert result.history[0]['inner_steps'] == (1,)
    assert result.history[0]['completed_groups'] == 1
    assert np.any(result.angular_flux[:, :, 0] != 0)
    np.testing.assert_array_equal(result.angular_flux[:, :, 1], 0)
    assert np.all(np.isfinite(result.angular_flux))
    matrix = dense_matrix(s, op.boundary_conditions)
    residual = np.linalg.norm(matrix @ result.angular_flux.ravel()-source.ravel())/np.linalg.norm(source)
    assert result.full_residual == pytest.approx(residual, abs=3e-15)


def test_zero_rhs_and_explicit_warm_start_and_result_immutability(monkeypatch):
    s = manufactured(1, cells=1)
    op = solver(s, 'reflective')
    source = np.broadcast_to(np.array([.8, .5]), s.shape).copy()
    expected = np.linalg.solve(dense_matrix(s, op.boundary_conditions), source.ravel()).reshape(s.shape)
    before = expected.copy()
    def forbidden(*args, **kwargs):
        raise AssertionError('unexpected GMRES for already-converged input')
    monkeypatch.setattr(grouped_module, 'gmres', forbidden)
    zero = op.solve(np.zeros(s.shape))
    assert zero.converged and zero.reason == 'zero_rhs' and zero.full_residual == 0
    assert zero.history == ()
    np.testing.assert_array_equal(zero.angular_flux, 0)
    warm = op.solve(source, initial_flux=expected)
    assert warm.converged and warm.reason == 'initial_residual_converged'
    assert warm.provenance['initial_field'] == 'caller_supplied'
    np.testing.assert_array_equal(expected, before)
    assert not np.shares_memory(warm.angular_flux, expected)
    with pytest.raises(ValueError):
        warm.angular_flux.setflags(write=True)
    with pytest.raises(TypeError):
        warm.provenance['initial_field'] = 'changed'
    with pytest.raises(FrozenInstanceError):
        warm.reason = 'changed'


def test_nonempty_history_is_immutable():
    s = manufactured(1, cells=1)
    result = solver(s, 'reflective', max_sweeps=1).solve(np.ones(s.shape))
    with pytest.raises(TypeError):
        result.history[0]['full_residual'] = 0.


def test_invalid_source_warm_start_controls_and_explicit_boundaries():
    s = manufactured(1, cells=1)
    op = solver(s)
    good = np.ones(s.shape)
    for bad in (good.tolist(), np.ones((1, 2)), good.astype(complex), good.astype(bool),
                np.full(s.shape, np.nan), np.full(s.shape, np.inf)):
        with pytest.raises(ValueError):
            op.solve(bad)
        with pytest.raises(ValueError):
            op.solve(good, initial_flux=bad)
    for controls in ({'inner_tolerance': 0}, {'full_tolerance': -1}, {'full_tolerance': np.nan},
                     {'inner_tolerance': np.inf}, {'full_tolerance': True}, {'inner_tolerance': 1j},
                     {'restart': 0}, {'max_inner': 1.2}, {'max_sweeps': True}, {'max_sweeps': -1}):
        with pytest.raises(ValueError):
            solver(s, **controls)
    for bc in ('reflective', (), ['vacuum']*len(s.boundary_owners), ('periodic',)*len(s.boundary_owners)):
        with pytest.raises(ValueError):
            GroupedFixedSource(s, bc)
    with pytest.raises(ValueError, match='NativeMeshStreaming'):
        GroupedFixedSource(None, ())


def test_budget_preflight_before_operator_construction_and_large_allocations(monkeypatch):
    s = manufactured(1)
    op = solver(s)
    source = np.ones(s.shape)
    assert op.memory_estimate_bytes > op.operators.allocation_estimate_bytes
    def forbidden(*args, **kwargs):
        raise AssertionError('allocation/operator construction preceded guard')
    monkeypatch.setattr(grouped_module, 'GroupAngularOperators', forbidden)
    monkeypatch.setattr(np, 'empty', forbidden)
    monkeypatch.setattr(np, 'zeros', forbidden)
    monkeypatch.setattr(np, 'array', forbidden)
    with pytest.raises(ValueError, match='allocation estimate'):
        solver(s, max_bytes=op.memory_estimate_bytes-1)
    # Repeat the guard at solve entry, before copies and initialization.
    object.__setattr__(op, 'max_bytes', op.memory_estimate_bytes-1)
    with pytest.raises(ValueError, match='allocation estimate'):
        op.solve(source)


def test_arithmetic_failure_preserves_last_finite_terminal(monkeypatch):
    s = manufactured(1, cells=1)
    op = solver(s, 'reflective')
    initial = np.full(s.shape, .2)
    before = initial.copy()
    def broken(*args, **kwargs):
        raise FloatingPointError('manufactured GMRES arithmetic failure')
    monkeypatch.setattr(grouped_module, 'gmres', broken)
    result = op.solve(np.ones(s.shape), initial_flux=initial)
    assert not result.converged and result.reason.startswith('arithmetic_failure:')
    assert result.full_residual is None  # Never report a stale residual.
    np.testing.assert_array_equal(result.angular_flux, before)
    np.testing.assert_array_equal(initial, before)


def test_group_arrays_and_closures_die_before_next_group_across_sweeps():
    s = manufactured(1, cells=1)
    op = solver(s, 'reflective', max_sweeps=2, full_tolerance=1e-13)
    code = GroupedFixedSource._solve_group.__code__
    previous, completed = [], []
    names = ('group_rhs', 'diagonal', 'candidate', 'true_residual',
             'candidate_moments', 'operator', 'preconditioner')
    def profile(frame, event, arg):
        if frame.f_code is not code:
            return
        if event == 'call':
            # No GC collection: these must have no surviving owner before
            # the next group can allocate its RHS/diagonal/Krylov basis.
            assert all(reference() is None for reference in previous)
        elif event == 'return':
            values = frame.f_locals
            assert all(name in values for name in names)
            previous[:] = [weakref.ref(values[name]) for name in names]
            completed.append(values['group'])
    old_profile = sys.getprofile()
    try:
        sys.setprofile(profile)
        result = op.solve(np.ones(s.shape))
    finally:
        sys.setprofile(old_profile)
    assert completed == [0, 1, 0, 1]
    assert all(reference() is None for reference in previous)
    assert len(result.history) == 2
    assert result.reason == 'max_sweeps'


def test_callback_failure_retains_gmres_basis_during_immutable_result_copy(monkeypatch):
    s = manufactured(1, cells=1)
    op = solver(s, 'reflective')
    source, initial = np.ones(s.shape), np.full(s.shape, .2)
    actual_gmres, actual_readonly = grouped_module.gmres, grouped_module._readonly
    retained, observed = [], []
    def wrapped_gmres(*args, **kwargs):
        callback = kwargs['callback']
        def throwing_callback(value):
            callback(value)
            try:
                raise FloatingPointError('manufactured callback failure')
            except FloatingPointError as error:
                retained.append(error)  # Deliberately keep traceback frames.
                raise
        kwargs['callback'] = throwing_callback
        return actual_gmres(*args, **kwargs)
    def observe_copy(value, *args, **kwargs):
        frames = []
        traceback = retained[0].__traceback__
        while traceback is not None:
            frames.append(traceback.tb_frame)
            traceback = traceback.tb_next
        gmres_frame = next(frame for frame in frames if frame.f_code is actual_gmres.__code__)
        group_frame = next(frame for frame in frames if frame.f_code is GroupedFixedSource._solve_group.__code__)
        solve_frame = next(frame for frame in frames if frame.f_code is GroupedFixedSource.solve.__code__)
        basis = gmres_frame.f_locals['v']
        assert basis.shape == (op.restart+1, np.prod(op.operators.shape))
        assert group_frame.f_locals['diagonal'].shape == op.operators.shape
        assert solve_frame.f_locals['flux'] is value
        assert not np.shares_memory(solve_frame.f_locals['source'], source)
        assert not np.shares_memory(value, initial)
        frozen = actual_readonly(value, *args, **kwargs)
        assert not np.shares_memory(frozen, value)
        assert basis.size > 0 and not frozen.flags.writeable
        observed.append(weakref.ref(basis))
        return frozen
    monkeypatch.setattr(grouped_module, 'gmres', wrapped_gmres)
    monkeypatch.setattr(grouped_module, '_readonly', observe_copy)
    result = op.solve(source, initial_flux=initial)
    assert not result.converged and result.reason.startswith('arithmetic_failure:')
    assert observed and observed[0]() is not None  # Held exception owns GMRES.
    assert result.full_residual is None
    assert result.provenance['workspace_contract'] == 'conservative_32G'
    np.testing.assert_array_equal(result.angular_flux, initial)
    np.testing.assert_array_equal(source, 1.)


def test_source_contract_is_hash_bound_not_version_only(monkeypatch):
    expected = dict(grouped_module._REVIEWED_SOURCES)
    monkeypatch.setattr(grouped_module, '_source_digest', lambda path: expected[path.name])
    reviewed = grouped_module._gmres_workspace_contract()
    assert reviewed.name == 'reviewed_scipy_1_18_1_10G'
    assert reviewed.group_vectors == 10
    assert reviewed.source_hashes == grouped_module._REVIEWED_SOURCES
    for altered in ('iterative.py', 'utils.py'):
        monkeypatch.setattr(grouped_module, '_source_digest',
            lambda path, altered=altered: '0'*64 if path.name == altered else expected[path.name])
        fallback = grouped_module._gmres_workspace_contract()
        assert fallback.name == 'conservative_32G'
        assert fallback.group_vectors == 32
        assert 'hashes differ' in fallback.reason
    def unavailable(path):
        raise OSError('manufactured missing source')
    monkeypatch.setattr(grouped_module, '_source_digest', unavailable)
    assert grouped_module._gmres_workspace_contract().group_vectors == 32


def test_source_digest_reads_bytes_without_evaluating_source(tmp_path):
    contents = b"raise RuntimeError('this source must never execute')\n"*4096
    source = tmp_path / 'manufactured.py'
    source.write_bytes(contents)
    assert grouped_module._source_digest(source) == hashlib.sha256(contents).hexdigest()


def test_replaced_callable_cannot_inherit_reviewed_source_allowance(monkeypatch):
    expected = dict(grouped_module._REVIEWED_SOURCES)
    monkeypatch.setattr(grouped_module, '_source_digest', lambda path: expected[path.name])
    actual = grouped_module.gmres
    def wrapper(*args, **kwargs):
        return actual(*args, **kwargs)
    monkeypatch.setattr(grouped_module, 'gmres', wrapper)
    assert grouped_module._gmres_workspace_contract().group_vectors == 32


def test_source_contract_change_rechecks_cap_before_solve_copies(monkeypatch):
    s = manufactured(1)
    expected = dict(grouped_module._REVIEWED_SOURCES)
    monkeypatch.setattr(grouped_module, '_source_digest', lambda path: expected[path.name])
    reviewed = solver(s)
    # Give the instance precisely the already-reviewed contract's budget.
    limited = solver(s, max_bytes=reviewed.memory_estimate_bytes)
    source = np.ones(s.shape)
    monkeypatch.setattr(grouped_module, '_source_digest', lambda path: '0'*64)
    fallback = solver(s)
    assert fallback.workspace_contract.group_vectors == 32
    assert fallback.memory_estimate_bytes > limited.max_bytes
    def forbidden(*args, **kwargs):
        raise AssertionError('numeric allocation preceded updated contract cap')
    monkeypatch.setattr(np, 'array', forbidden)
    monkeypatch.setattr(np, 'zeros', forbidden)
    monkeypatch.setattr(np, 'empty', forbidden)
    with pytest.raises(ValueError, match='allocation estimate'):
        limited.solve(source)
    with pytest.raises(ValueError, match='allocation estimate'):
        solver(s, max_bytes=limited.max_bytes)


def test_owned_rhs_copy_and_single_moment_cache_through_result(monkeypatch):
    s = manufactured(1, cells=1)
    op = solver(s, 'reflective', max_sweeps=2)
    caller = np.ones(s.shape)
    old_result = GroupedFixedSource._result
    captured = []
    old_group = GroupedFixedSource._solve_group
    def observe_group(self, source, flux, moments, group):
        assert source is not caller and not np.shares_memory(source, caller)
        assert source.dtype == np.float64
        # One persistent coupled cache and iterate across groups and sweeps.
        identity = (id(source), id(flux), id(moments))
        if captured:
            assert identity == captured[0]
        captured.append(identity)
        return old_group(self, source, flux, moments, group)
    def observe_result(self, flux, *args):
        result = old_result(self, flux, *args)
        assert not np.shares_memory(result.angular_flux, flux)
        assert not np.shares_memory(result.angular_flux, caller)
        return result
    monkeypatch.setattr(GroupedFixedSource, '_solve_group', observe_group)
    monkeypatch.setattr(GroupedFixedSource, '_result', observe_result)
    result = op.solve(caller)
    assert len(captured) >= 4
    assert not result.angular_flux.flags.writeable
    np.testing.assert_array_equal(caller, 1.)


def test_result_freezing_does_not_allocate_a_full_multigroup_finite_mask(monkeypatch):
    terminal = np.ones((513, 4, 7))
    original = np.isfinite
    seen = []
    def bounded(value, *args, **kwargs):
        assert value.ndim == 2 and value.shape[0] <= 512
        seen.append(value.shape)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(np, 'isfinite', bounded)
    frozen = grouped_module._readonly(terminal)
    assert len(seen) == 14  # Seven groups, one full and one partial batch.
    assert not np.shares_memory(frozen, terminal)
    assert not frozen.flags.writeable
