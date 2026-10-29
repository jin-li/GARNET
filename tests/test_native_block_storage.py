"""Storage regression against the immutable c613ef7 numerical implementation."""
import hashlib
import importlib.util
import json
from pathlib import Path

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
    source = snapshot / 'garnet/corrected_native_moments.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['files'][
        'garnet/corrected_native_moments.py']
    # The frozen module uses the live package's shared types/helpers. Verify
    # those sources are identical to the snapshot before comparing operators.
    package = Path(__file__).resolve().parents[1] / 'garnet'
    for name in ('mesh.py', 'corrected_mesh.py', 'moment_materials.py',
                 'coupled_moments.py', 'spn.py'):
        assert hashlib.sha256((package / name).read_bytes()).hexdigest() == manifest['files']['garnet/' + name]
    spec = importlib.util.spec_from_file_location('garnet._frozen_c613ef7_native', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CorrectedNativeMomentOperator


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
    for name in ('indptr', 'indices', 'data'):
        np.testing.assert_array_equal(getattr(current.assembled, name), getattr(frozen.assembled, name))
    u = np.random.default_rng(613).normal(size=current.shape)
    for state in (u, u.ravel()):
        np.testing.assert_array_equal(current.action(state), frozen.action(state))
        np.testing.assert_array_equal(current.fission_action(state), frozen.fission_action(state))
        np.testing.assert_array_equal(current.gradients(state), frozen.gradients(state))
        for actual, expected in zip(current.face_currents(state), frozen.face_currents(state)):
            np.testing.assert_array_equal(actual, expected)
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
