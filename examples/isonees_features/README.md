# Synthetic ISONEES feature examples

From the repository root, install the CPU dependencies and run:

```sh
python -m pip install -e '.[test]'
python examples/isonees_features/common_mesh_fv.py
python examples/isonees_features/nem_local_reference.py
python examples/isonees_features/static_kinetics.py
python examples/isonees_features/branch_xs.py
python -m pytest -q tests/test_isonees_examples.py
```

On NixOS, enter the repository's `nix develop` environment first. These examples
use NumPy/SciPy on CPU, finish on small synthetic problems, and require no GPU,
Serpent executable, proprietary reactor data, ACE files or external downloads
beyond installing Python dependencies. They write no persistent output.

| Example | Demonstrated behavior | Analytical check |
|---|---|---|
| `common_mesh_fv.py` | Common-mesh diffusion on two unequal reflected cuboids | Constant flux and k = 0.035/0.03 |
| `nem_local_reference.py` | Matching local NEM reference at degrees 4/6/8, internal ADF construction, rebuilt receiver | Mean ratio 1.1, equal corrected traces, zero current, same k |
| `static_kinetics.py` | Version-1 raw regional JSON contract, repeated region mapping, matching NEM forward/adjoint projection | beta = (0.002, 0.004), physical Lambda = 1/(220000 × 0.035) s |
| `branch_xs.py` | Explicit temperature triangles, dependent moderator density, absolute XS evaluation and fresh NEM | Homogeneous reflected k = nuSigma_f/Sigma_a; outside-diamond requests rejected |

The NEM/reference/ADF and matching kinetics paths are experimental CPU paths.
The interface traces in the ADF example are manufactured to illustrate the
contract. Local reference order checks do not establish transport accuracy or
qualify a receiving core. The auxiliary reference source is a balance diagnostic
and does not enter physical core production or kinetics.

The kinetics example declares **raw family production**, not global beta_eff as
an input. Its provenance hash identifies the synthetic definition, not a measured
or qualified nuclear data source. It keeps both decay channels and reports the
explicit low-frequency reduction sum(beta)/sum(beta/lambda). Physical Lambda and
the discrete lifetime k × Lambda differ because k is not one. These are static
parameters; the example does not establish a transient NEM model.
