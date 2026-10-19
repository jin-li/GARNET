"""Analytical smoke checks for the portable synthetic feature examples."""
from pathlib import Path
import runpy
import numpy as np


EXAMPLES = Path(__file__).resolve().parents[1]/'examples'/'isonees_features'


def example(name):
    return runpy.run_path(str(EXAMPLES/f'{name}.py'))


def test_common_mesh_reflected_homogeneous_limit():
    mesh, _, result = example('common_mesh_fv')['solve']()
    assert result.converged
    np.testing.assert_allclose(result.k_eff, .035/.03, rtol=1e-10)
    # Reflective homogeneous material has constant flux despite unequal volumes.
    phi = np.asarray(result.flux).reshape(-1)
    assert np.all(phi > 0)
    np.testing.assert_allclose(phi/phi[0], 1, rtol=1e-9)
    np.testing.assert_allclose(mesh.area, [2000., 3000.])


def test_manufactured_adf_jump_preserves_corrected_trace_and_balance():
    solver, factors, forward = example('nem_local_reference')['solve']()
    np.testing.assert_allclose(forward.k_eff, .035/.03, rtol=1e-9)
    np.testing.assert_allclose(forward.means[1]/forward.means[0], 1.1, rtol=1e-8)
    # The manufactured jump supports zero leakage; physical interface traces match.
    corrected = forward.side_face_means*factors.factors
    np.testing.assert_allclose(corrected[0, 2], corrected[1, 0], rtol=1e-9)
    np.testing.assert_allclose(forward.currents, 0, atol=1e-10)
    np.testing.assert_allclose(factors.references[-1].auxiliary_mean_source, 0, atol=1e-12)
    # Global production/absorption balance is independent of arbitrary normalization.
    inventory = np.sum(solver.grid.volumes[:, None]*forward.means)
    np.testing.assert_allclose(.03*inventory, .035*inventory/forward.k_eff, rtol=1e-9)


def test_raw_contract_matching_adjoint_has_physical_time_and_family_limits():
    namespace = example('static_kinetics')
    _, _, result = namespace['solve']()
    beta, decay = namespace['RAW_BETA'], namespace['DECAY']
    np.testing.assert_allclose(result.beta, beta, rtol=1e-10)
    np.testing.assert_allclose(result.generation_time,
                               1/(namespace['SPEED']*namespace['NU_SIGMA_F']), rtol=1e-10)
    # This off-critical problem distinguishes Lambda from k*Lambda.
    np.testing.assert_allclose(result.neutron_lifetime, result.k_eff*result.generation_time,
                               rtol=1e-12)
    assert result.neutron_lifetime > result.generation_time
    np.testing.assert_allclose(result.lambda_low_frequency, beta.sum()/np.sum(beta/decay),
                               rtol=1e-12)
