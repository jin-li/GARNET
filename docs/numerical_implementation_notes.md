# Numerical implementation notes

These notes retain historical measurements and failure investigations moved
out of the numerical implementations during the readability review. They refer
to the original workloads and code state; they are not new benchmark results.
The source code keeps the equations and invariants needed to follow each method.

## Energy-group sweeps before amplitude rebalance

The diffusion transient defaults to six group subsweeps for upscattering
libraries. An amplitude rebalance assumes the swept flux satisfies the source
balance. Additional group sweeps settle the spectral distribution before that
amplitude correction.

The original 11-group HP-MR tolerance study compared three, six, and twelve
subsweeps. At refine 2, the converged powers were 1.3990072, 1.3990070, and
1.3990067 at `tol_step=1e-9`.

| Subsweeps | Outer sweeps, refine 2 / 3 | Inner CG work, refine 2 | Production-tolerance error |
|---|---|---|---|
| 3 | 248 / 247 | 43,371 | 2.5e-4 |
| 6 | 27 / 26 | 11,449 | 7.1e-5 |
| 12 | not recorded | 26,387 | 1.8e-4 |

For C5G7-TD1-1, raising the count from three to eight reduced outer sweeps
from 399 to 162 and inner CG work by a factor of 1.9. Errors changed from
1.24e-4 to 6.4e-5; tightly converged powers were 0.7814288 and 0.7814287.
Without rebalance, raising HP-MR subsweeps from three to six only reduced
outer sweeps from 1,347 to 1,080. The benefit depends on both mechanisms.

## SPH nonlinear solves

A non-positive trial region flux makes the logarithmic SPH residual undefined.
The recovery residual is proportional to `log(mu)`, pulling factors toward
`mu=1`. Its nonzero Jacobian gives Newton a recovery direction. The previous
constant penalty had a zero Jacobian: an HP-MR run stalled at a residual
magnitude of 106.826 while its line-search step changed without improving it.

The small HP-MR SPH problem had roughly twelve active region/group unknowns.
With production normalization, a Jacobian-free solve required 2,017 residual
evaluations without converging. A dense numerical Jacobian cost thirteen
coarse solves per construction and let the trust-region method resolve the
poor row scaling. This motivates offering dense and Krylov methods separately;
the appropriate choice depends on problem size and conditioning.

## Triangular equivalence currents

Partial-current reconstruction must use the same boundary law as the diffusion
operator. A previous reconstruction used `phi/4` independently of diffusion,
albedo, and the fitted boundary coefficient. The fit residual reached
1.2e-28 while the eigenvalue remained about 350 pcm away from the reference.

Per-cell discontinuity factors provide one unknown for a region's several
surface-current conditions. An HP-MR least-squares fit stalled at cost 0.678
with drum factors at their `exp(±1.2)` bounds. Per-face factors provide
independent unknowns for the ordered region interfaces.

Coarse and transport reference currents must also use the same face measure.
Omitting the edge length from coarse currents produced separate normalization
clusters for flux, interface current, and boundary current: approximately 31,
10, and 5. Current reconstruction now includes the face area explicitly.

## Transport sweep contractions

The level sweep uses persistent buffers for weighted angular reductions and
small corner-block products. The earlier broadcast-multiply implementation
was allocation-free and supported graph capture, but moved roughly three
times the data of a fused contraction. The fused kernels retain that storage
contract. Their broadcast fallbacks remain available, and the selected path
is fixed at setup so graph capture and replay use the same operations.

## Coupled-transient acceleration

The original 11-group HP-MR comparison used refine 3, one 50 ms step, and a
`tol=1e-10` rebalanced reference.

| Anderson depth | Rebalance | Tolerance | Power error | Time | Inner iterations |
|---|---|---|---|---|---|
| 1 | on | 1e-6 | -0.034% | 16.7 s | 112k |
| 1 | off | 1e-6 | -0.034% | 75.3 s | 535k |
| 5 | off | 1e-6 | -0.693% | 38.5 s | 273k |
| 5 | off | 1e-8 | -0.002% | 102.8 s | 652k |

Rebalanced Picard and plain Picard agreed within 1.2e-6 in that comparison.
A small change between Anderson-mixed iterates understated the residual on
the stiff, stateful group map; combining depth-five Anderson with rebalance
failed to converge. The transient therefore confirms apparent Anderson
convergence with an unaccelerated sweep.

The two-group demonstration behaved differently: Anderson took 2.0 s versus
15.0 s for rebalanced Picard, with answers agreeing within 1.5e-4. The coupled
driver retains a group-count heuristic and allows callers to override it
through `transient_kwargs`.
