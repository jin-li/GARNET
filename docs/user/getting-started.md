# Installation and first calculation

Run commands from the repository root. A CPU environment needs Python 3.10 or newer, NumPy and SciPy:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test,docs]'
```

On NixOS use the repository's documented development shell before creating or using the environment; numerical wheels need its native runtime libraries. For GPU installation follow [AMD/ROCm setup](../amd_rocm.md) or the CUDA extras in `pyproject.toml`. Install one CuPy distribution matching the runtime. CUDA and ROCm distributions share a Python module and should not coexist in one environment.

## A complete CPU baseline

This reflected one-group synthetic problem has the analytic result $k=\nu\Sigma_f/\Sigma_a=1.2$. Reflection removes leakage, so it is useful for checking installation and group conventions.

```python
import numpy as np
from ndgpu import Grid, Material, DiffusionEigenSolver

fuel = Material(name="synthetic fuel", diffusion=[1.2],
                sigma_a=[0.025], nu_sigma_f=[0.030],
                sigma_s=[[0.0]], chi=[1.0])
grid = Grid(shape=(8, 1, 1), size=(40.0, 1.0, 1.0))
result = DiffusionEigenSolver(grid, fuel, bc="reflective",
                             device="cpu").solve(
                                 tol_k=1e-9, tol_source=1e-8)
print(result)
assert np.isclose(result.k_eff, 1.2, atol=1e-7)
phi = result.flux_numpy
print(phi.shape)
```

`Grid.shape` and `Grid.size` always have three entries. The low-level flux is `(groups,nx,ny,nz)`. `result.flux` lives on the solve backend; `flux_numpy` is the explicit host conversion. Save the reported method, device, tolerances and status with the answer.

## A finite leaking reactor

```python
from ndgpu import Model
model = (Model(size=(40.0,), cells=(40,))
         .fill(fuel)
         .set_boundary("vacuum"))
report = model.run(device="cpu")
print(report)
print(report.k_eff, report.reactivity_pcm)
```

This block continues the preceding material definition. A vacuum surface leaks neutrons, so its eigenvalue differs from the reflected case. `Model` handles the collapsed axes. For analytic bare-box comparisons use `zero-flux` deliberately; it represents a different surface condition.

Existing runnable starting points are [bare reactor](https://github.com/jin-li/NDgpu/blob/main/examples/bare_reactor.py), [reflected core](https://github.com/jin-li/NDgpu/blob/main/examples/reflected_core.py), and [adjoint importance](https://github.com/jin-li/NDgpu/blob/main/examples/adjoint_importance.py). Run each script's documented command or `--help` before scaling its mesh.
