import cupy as cp
import numpy as np
from ndgpu import DiffusionEigenSolver, Grid, PWR_TWO_GROUP

cp.show_config()
props = cp.cuda.runtime.getDeviceProperties(0)
print('GPU:', props['name'].decode(), 'compute capability:', cp.cuda.Device(0).compute_capability, flush=True)
np.testing.assert_allclose(cp.asnumpy(cp.arange(100, dtype=cp.float64) ** 2), np.arange(100) ** 2)
grid = Grid(shape=(32, 32, 32), size=(150., 150., 150.))
results = {}
for device in ('cpu', 'gpu'):
    results[device] = DiffusionEigenSolver(grid, PWR_TWO_GROUP, device=device).solve(tol_k=1e-10, tol_source=1e-9)
    assert results[device].converged
    print(device, results[device], flush=True)
np.testing.assert_allclose(results['gpu'].k_eff, results['cpu'].k_eff, rtol=0, atol=1e-8)
np.testing.assert_allclose(results['gpu'].flux_numpy, results['cpu'].flux_numpy, rtol=1e-5, atol=1e-9)
print('PASS: CPU/GPU k_eff and full flux comparison', flush=True)
