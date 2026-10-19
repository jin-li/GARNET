# Validation, diagnostics and reproducible runs

## A practical sequence

1. Run a small CPU synthetic case with a known limit before loading a large dataset.
2. Confirm geometry, dimensions, material ordering, energy boundaries and boundary conditions.
3. Require iterative convergence and inspect algebraic residuals and conservation.
4. Refine space, angular order, polynomial order or time as appropriate; vary one error source at a time.
5. Compare against an independent reference with the same production model and normalization.
6. Repeat on the requested GPU and retain backend/version evidence if GPU execution is part of the objective.

The integrated synthetic feature examples are:

```bash
python examples/isonees_features/common_mesh_fv.py
python examples/isonees_features/nem_local_reference.py
python examples/isonees_features/static_kinetics.py
python examples/kinetics/static_nem.py
```

Use the examples' README for available options and numerical expectations. They exercise common mesh, local NEM reference and static kinetics without proprietary inputs. They do not certify a private reactor calculation.

Focused regression checks from the root include:

```bash
python -m pytest tests/test_nem.py tests/test_nem_higher_order.py tests/test_nem_kinetics.py
python -m pytest tests/verification/test_conduction.py tests/verification/test_feedback.py
```

Run the full ordinary suite with `python -m pytest`. Expensive cases are marked `slow` and excluded by the repository default; select them deliberately with `-m slow`. A skip for missing GPU/preCICE is evidence that the relevant environment was unavailable, not a passing test of that capability.

## Diagnose by the failed contract

| Symptom | First checks |
| --- | --- |
| Ambiguous Serpent selection | Inventory records; specify universe or record index |
| Unexpected spectrum or neutron balance | Source/destination orientation, production vs collision fields, reduced absorption, chi and normalization |
| Mesh/factor mismatch | Cell/side ordering, active mask, exact shared coordinates, whole-face vs subface representation |
| NEM refusal or negative traces | Condition estimates, local profile diagnostics, leakage policy and polynomial convergence; retain negative values |
| Invalid delayed input | Raw production vs effective beta, energy/region mapping, prompt remainder and common channel identities |
| Flat or stale transient response | Callback ordering, object caching/replacement, critical adjustment and state rebuild triggers |
| Coupling residual stalls | Inner neutron accuracy, thermal units/power, relaxation and useful outer tolerance floor |
| GPU path unavailable | Actual CuPy family/runtime/device and path-specific version requirements |

Do not repair a refused physical input by clipping, silently changing orientation, replacing missing data, loosening positivity or fitting to a desired eigenvalue. Resolve the contract and preserve diagnostics.

## What to archive

Store the software revision, environment versions, device/precision, source hashes, selected records, energy/region maps, boundaries, production kernel, factors and their reference qualification, solver settings, convergence/refinement results and diagnostic flags. Separate timing from correctness checks. If Monte Carlo uncertainty is quoted, retain its estimator definition and covariance limitations.

The [methodology evidence chapter](../methodology/data-verification.md) explains what each check proves. Existing detailed notes remain useful for advanced controls, but a historical example's success is not automatic qualification of a changed material library, kernel, geometry or factor reconstruction.
