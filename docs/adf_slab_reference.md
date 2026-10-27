# Heterogeneous slab reference-factor verification

This reproducible, deliberately synthetic case verifies the finite-volume
interface treatment against a separately computed analytical heterogeneous
**diffusion** solution. It is not a published reactor benchmark or a neutron
transport reference. Its structure follows the reference-cross-section and
reference-discontinuity-factor approach discussed by
[Vidal-Ferràndiz et al.](https://doi.org/10.1016/j.anucene.2015.06.021).
The reference factors are generated using knowledge of the whole-core answer;
this is an equivalence test and does not demonstrate that practical
isolated-assembly ADFs improve a different environment.

## Definition

A 30 cm slab contains three 10 cm homogenization regions, each made from two
heterogeneous layers. Both boundaries reflect. In order from left to right,
the parameters are:

| Region | Width (cm) | D (cm) | Σa (cm⁻¹) | νΣf (cm⁻¹) |
|---:|---:|---:|---:|---:|
| 1 | 4 | 1.30 | 0.008 | 0.014 |
| 1 | 6 | 0.25 | 0.065 | 0.035 |
| 2 | 6 | 0.40 | 0.035 | 0.040 |
| 2 | 4 | 1.00 | 0.012 | 0.025 |
| 3 | 5 | 1.50 | 0.009 | 0.013 |
| 3 | 5 | 0.18 | 0.080 | 0.055 |

There is one energy group. Within a layer the exact equation is
`phi' = -J/D`, `J' = (nuSigma_f/k - Sigma_a) phi`. Starting at
`phi(0)=1, J(0)=0`, propagate `(phi,J,integral(phi))` through each layer by
matrix exponentials and find the root for `J(30)=0`. Flux and current are
continuous at layer interfaces. The chosen root has positive flux throughout
the slab. It gives `k = 0.9964473442616343`, region-average fluxes
`[1.2426133211, 4.6998097069, 3.5469294233]`, and oriented inter-region
currents `[-0.2293095799, 0.1831278353]` under that normalization.
An independent GARNET calculation of the six-layer heterogeneous problem
converges towards that k at approximately second order when the aligned mesh
is refined; this tests the analytical reference without any ADFs.

The analytical flux weights Σa and νΣf in each of the three homogenized
regions, preserving the reference reaction rates for a given region-average
flux. D is also flux weighted as a stated modeling choice; reaction-rate
preservation does not uniquely determine D. For each desired resolution, solve
an **independent tridiagonal homogeneous finite-volume problem in each
region** at the reference k with the analytical boundary currents imposed.
At each region interface, divide the heterogeneous flux by each adjacent
homogeneous two-point reconstructed trace. These two positive factors enforce
the reference current and corrected interface trace. They are regenerated for
each resolution because the finite-volume reconstruction changes with the
mesh. On the single-cell mesh the two directed pairs are `(0.87607666,
0.93653961)` and `(1.55331975, 1.18091183)`. The helper assigns unity to
subfaces internal to each region.

The independent reference-factor construction supplies the answer it is meant
to reproduce, so this is a strong check of implementation and normalization,
not an independent prediction of accuracy. The homogenized shapes within a
region are not the heterogeneous shapes, and the case does not establish
accuracy for two-group, 2-D or transport-generated ADFs.

## Results

Run `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python3 examples/adf_slab_reference.py`.
Here pcm is `1e5*(k-k_ref)/k_ref`; maximum flux error is the largest relative
error in the three region-average fluxes after matching their summed average.

| cells per region | No-factor k | No-factor pcm | No-factor max flux error | Reference-factor k | Reference-factor max flux error |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.01736677 | +2099 | 13.85% | 0.99644734 | <1e-9 |
| 2 | 0.96861093 | −2794 | 24.09% | 0.99644734 | <1e-9 |
| 4 | 0.95576621 | −4083 | 30.17% | 0.99644734 | <1e-9 |
| 8 | 0.95253279 | −4407 | 31.74% | 0.99644734 | <1e-9 |

For all four corrected cases the maximum absolute interface-current error is
below 1.1e-12 under the reference normalization, and the algebraic relative
residual is below 1.5e-12. The no-factor errors need not decrease with mesh
refinement: the heterogeneous layers have been replaced by fixed homogenized
region data, so refinement approaches the wrong continuous homogenized
problem more accurately. This example deliberately uses strong heterogeneity.

The tests in [test_adf_slab_reference.py](../tests/verification/test_adf_slab_reference.py)
check the continuous reference, mesh convergence, eigenvalue, region averages,
interface currents and residuals. Independent two-group matrix and transpose
checks are in [test_cartesian_adf.py](../tests/verification/test_cartesian_adf.py).

## Two-group synthetic extension

[adf_slab_2g_reference.py](../examples/adf_slab_2g_reference.py) uses the U,
M1, M2 and W materials from Dall'Osso's Table 1, but places them in this same
synthetic 30 cm, three-region slab: `(4 cm U, 6 cm W)`, `(6 cm M1, 4 cm M2)`,
`(5 cm U, 5 cm W)`. Both boundaries reflect. Fission neutrons enter group 1;
the only intergroup scattering is downscatter. This is **not** Dall'Osso's
published geometry. A coupled four-component diffusion transfer matrix and
integrals yield `k = 0.9437610266`. Two-group region-weighted cross sections
and independent coupled fixed-current FV region solves generate the reference
factors. The region averages and group currents agree to below 1e-8 and 1e-9
respectively on 4, 8 and 16 cells per region. With no factors, k differs by
−1670, −731 and −416 relative pcm respectively; maximum region/group flux
errors are about 18–19%. A separately refined heterogeneous FV solve converges
towards the transfer solution at second order.

The corresponding fixed-current homogeneous reconstruction produces a
**negative surface trace** on the 1- and 2-cell-per-region meshes. A positive
factor cannot reproduce the positive heterogeneous trace there, so the example
refuses to construct those reference factors. This is a limitation of this
coarse two-point reconstruction for the chosen strongly heterogeneous case,
not a general statement about ADFs. To run the supported meshes:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python3 examples/adf_slab_2g_reference.py
```

The separately reconstructed published Dall'Osso geometry and the remaining
reference discrepancy are documented in [dallosso_1d_reference.md](dallosso_1d_reference.md).
