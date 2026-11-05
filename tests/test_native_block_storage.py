"""Vacuum storage oracle plus reflective runtime invariants.

The old reflective gradient metric intentionally differs after reflection folding
was corrected; test_native_reflection_folding checks its replacement invariant.
"""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

from garnet.corrected_native_moments import CorrectedNativeMomentOperator
from garnet.mesh import MeshGrid
from test_corrected_mesh import mesh_grid
from test_coupled_moments import material


@pytest.fixture(scope='module')
def frozen_operator():
    # This local evidence snapshot is deliberately not a second production
    # implementation. Source archives without it still run the independent
    # dense/face/eigen oracles in test_corrected_native_moments.py.
    root = Path(__file__).resolve().parents[4]
    snapshot = root / '.scratch/pratic-subpin/evidence/software-snapshots/c613ef7'
    if not snapshot.is_dir():
        pytest.skip('local immutable c613ef7 evidence snapshot unavailable')
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    assert manifest['commit'] == 'c613ef7d3380c6c1b6c86427dc1cc7d63e2ffb21'
    # Freeze both operators together: the old corrected operator must retain
    # its original coupled value/scalar helpers when the live layout evolves.
    for name in ('coupled_moments.py', 'corrected_native_moments.py'):
        assert hashlib.sha256((snapshot / 'garnet' / name).read_bytes()).hexdigest() == manifest['files']['garnet/' + name]
    # Only unchanged, hash-verified helpers share live identities. In particular
    # MeshGrid/MomentMaterial must be the same types for isinstance contracts.
    package = Path(__file__).resolve().parents[1] / 'garnet'
    helpers = ('mesh.py', 'corrected_mesh.py', 'moment_materials.py', 'spn.py')
    for name in helpers:
        if hashlib.sha256((package / name).read_bytes()).hexdigest() != manifest['files']['garnet/' + name]:
            pytest.skip(f'local frozen oracle requires unchanged helper {name}')
    private_name = '_garnet_frozen_c613ef7_storage'
    private = ModuleType(private_name)
    private.__package__ = private_name
    private.__path__ = [str(snapshot / 'garnet')]
    private.__spec__ = importlib.util.spec_from_loader(private_name, loader=None, is_package=True)
    # Restore sys.modules after this module-scoped fixture, including on failure.
    with pytest.MonkeyPatch.context() as aliases:
        aliases.setitem(sys.modules, private_name, private)
        for filename in helpers:
            name = filename[:-3]
            module = importlib.import_module('garnet.' + name)
            aliases.setitem(sys.modules, private_name + '.' + name, module)
            setattr(private, name, module)
        for name in ('coupled_moments', 'corrected_native_moments'):
            fullname = private_name + '.' + name
            source = snapshot / 'garnet' / (name + '.py')
            spec = importlib.util.spec_from_file_location(fullname, source)
            module = importlib.util.module_from_spec(spec)
            aliases.setitem(sys.modules, fullname, module)
            spec.loader.exec_module(module)
            setattr(private, name, module)
        yield module.CorrectedNativeMomentOperator


@pytest.mark.parametrize('family', ['sp3', 'sdp1'])
@pytest.mark.parametrize('bc', ['reflective', 'vacuum'])
def test_block_storage_frozen_noncommuting_skew(frozen_operator, family, bc):
    mesh = mesh_grid(3, .41)
    grid = MeshGrid(mesh, bc)
    # Repeated immutable objects exercise the existing material cache, while
    # alternating materials retain full noncommuting native group matrices.
    pair = [material(0), material(1)]
    materials = [pair[i % 2] for i in range(mesh.n_cells)]
    current = CorrectedNativeMomentOperator(grid, materials, family=family)
    frozen = frozen_operator(grid, materials, family=family)
    D0, D1 = current.diffusion[:2]
    assert np.linalg.norm(D0 @ D1 - D1 @ D0) > 1e-5
    assert current.diffusion[0] is current.diffusion[2]
    assert current.assembled.format == 'csr'
    assert current.assembled.has_canonical_format
    assert not np.any(current.assembled.data == 0)
    if bc == 'vacuum':
        for name in ('indptr', 'indices', 'data'):
            np.testing.assert_array_equal(getattr(current.assembled, name), getattr(frozen.assembled, name))
    for name in ('diffusion','reaction'):
        for actual,expected in zip(getattr(current,name),getattr(frozen,name)):
            np.testing.assert_array_equal(actual,expected)
    u = np.random.default_rng(613).normal(size=current.shape)
    for state in (u, u.ravel()):
        np.testing.assert_array_equal(current.fission_action(state), frozen.fission_action(state))
        if bc == 'vacuum':
            np.testing.assert_array_equal(current.action(state), frozen.action(state))
            np.testing.assert_array_equal(current.gradients(state), frozen.gradients(state))
            for actual, expected in zip(current.face_currents(state), frozen.face_currents(state)):
                np.testing.assert_array_equal(actual, expected)
        else:
            gradients=current.gradients(state)
            assert gradients.shape==(mesh.n_cells,4,2) and np.all(np.isfinite(gradients))
            interior,boundary=current.face_currents(state)
            np.testing.assert_array_equal(boundary,0)
            reaction=np.stack([r@u for r,u in zip(current.reaction,state.reshape(mesh.n_cells,4))])
            independent=reaction.copy()
            for (i,j,*_),flux in zip(mesh.faces,interior.reshape(-1,4)):
                independent[i]+=flux;independent[j]-=flux
            np.testing.assert_array_equal(current.action(state).reshape(mesh.n_cells,4),independent)
            np.testing.assert_allclose((independent-reaction).sum(axis=0),0,atol=3e-14)
        np.testing.assert_allclose(current.assembled @ state.ravel(), current.action(state).ravel(), atol=3e-14)


def test_explicit_unknown_budget_replaces_default(monkeypatch):
    grid = MeshGrid(mesh_grid(2, .41), 'vacuum')
    materials = [material()] * grid.n_cells
    reference = CorrectedNativeMomentOperator(grid, materials)
    assert reference.max_unknowns == 20_000
    # Lower the class default so testing the opt-in requires no large domain.
    monkeypatch.setattr(CorrectedNativeMomentOperator, 'prototype_unknown_limit', reference.size - 1)
    for kwargs in ({}, {'max_unknowns': None}, {'max_unknowns': reference.size - 1}):
        with pytest.raises(ValueError, match=f'limited to {reference.size - 1} unknowns'):
            CorrectedNativeMomentOperator(grid, materials, **kwargs)
    for budget in (reference.size, np.int64(reference.size + 1)):
        current = CorrectedNativeMomentOperator(grid, materials, max_unknowns=budget)
        assert current.max_unknowns == budget
        np.testing.assert_array_equal(current.assembled.toarray(), reference.assembled.toarray())
        u = np.random.default_rng(20).normal(size=current.size)
        np.testing.assert_array_equal(current.action(u), reference.action(u))


@pytest.mark.parametrize('budget', [True, False, np.bool_(True), 0, -1, 16., .5, np.nan, np.inf, '16', [16]])
def test_invalid_unknown_budget_rejected_before_geometry_or_material_matrices(monkeypatch, budget):
    import garnet.corrected_native_moments as native
    grid = MeshGrid(mesh_grid(2, .41), 'vacuum')
    materials = [material()] * grid.n_cells
    def forbidden(*args, **kwargs):
        raise AssertionError('geometry validation or material matrix allocation must not run')
    monkeypatch.setattr(native, '_validate_corrected_mesh', forbidden)
    monkeypatch.setattr(CorrectedNativeMomentOperator, '_inverse', forbidden)
    with pytest.raises(ValueError, match='max_unknowns must be a positive integer'):
        CorrectedNativeMomentOperator(grid, materials, max_unknowns=budget)
