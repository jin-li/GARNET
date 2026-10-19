# Angular approximations and native moments

## SPN and SDPN

SPN replaces the transport angular dependence by coupled diffusion-type moment equations. SP1 has the diffusion limit. SP3 has two even-moment unknowns, commonly written using $u_1=\phi_0+2\phi_2$ and $u_2=\phi_2$, with physical scalar flux $\phi_0=u_1-2u_2$. SP5 and SP7 extend the block to three and four unknowns. Fission and scalar scattering act through the physical scalar-flux recovery, rather than treating every transformed unknown as an independent scalar flux.

The implementation's SP3 formulation follows [Brantley and Larsen](https://doi.org/10.13182/NSE134-01). SDPN uses a half-range angular closure, following the [simplified double-PN derivation](https://arxiv.org/abs/2212.02140). SDP1/2/3 have block sizes matching SP3/5/7. Equal block size does not imply identical coefficients, interface conditions or error.

The package can transform appropriate moment blocks to symmetric forms. The availability depends on grid, boundaries and closure. `marshak_vacuum=True` requests moment-coupled vacuum treatment in supported low-level paths. A scalar Robin condition on every moment is not automatically the same boundary model.

Standard `Material`-based SPN/SDPN solvers use their documented scalar scattering and total/transport inputs. They do not automatically consume all Serpent $S_\ell$ or $SP_\ell$ matrices. Increasing SPN order does not restore missing native intergroup anisotropic scattering.

## Discrete ordinates and hybrids

SN solves for directional angular flux $\psi_{g,m}$ at quadrature directions $\Omega_m$:

```{math}
\Omega_m\cdot\nabla\psi_{g,m}+\Sigma_{t,g}\psi_{g,m}
=Q_{g,m},\qquad
\phi_g=\sum_m w_m\psi_{g,m}.
```

`SNTransportSolver` and `TriSNTransportSolver` implement geometry-specific streaming/sweeps and angular quadrature. Reflective boundaries require the matching direction maps. Vacuum removes incoming angular flux. Source iteration, Anderson acceleration and diffusion acceleration are numerical convergence tools; report both full residuals and iteration status.

Angular quadrature, spatial resolution and iterative tolerance are independent convergence dimensions. Ray effects and highly absorbing regions can persist after iteration converges. Compare direction sets and mesh refinements rather than treating an SN label as an exact reference.

Hybrid solvers assign selected cells to transport and others to diffusion. The interface exchanges currents/partial-current information through the implemented coupling. It is a mixed model with an interface approximation. Move or refine the interface in a convergence study; agreement of two solves with the same interface does not qualify the approximation. See `hybrid_sn.py`, `hybrid_tri_sn.py` and the corresponding runnable examples.

## Full native moment scattering

The opt-in Serpent moment reader retains signed higher-order collision and production estimates independently, along with native reported errors. It requires an explicit `from-major` or `to-major` flattening contract. For each angular order,

```{math}
R_\ell=\operatorname{diag}(\Sigma_t)-SP_\ell^T
```

for source-major input. The native SP3/SDP1 operator uses cell-major `(N,2,G)` state with $U=(\phi_0+2\phi_2,3\phi_2)$; this scaling differs from the standard SP3 variables above. Its full loss blocks need not commute. It is separate from the common-mesh moment solver interfaces. The coupled operator retains full intergroup matrices rather than a diagonal current correction or scattering source iteration approximation. Immutable `MomentMaterial` data preserve provenance and do not infer covariance, reinterpret signed reported errors or establish estimator consistency with deterministic transport.

`CoupledMomentEigenSolver` uses a CPU full coupled loss solve. `CoupledMomentGPUEigenSolver` transfers the assembled CSR operator once and performs matvec/Arnoldi work on a real GPU; fission, normalization, reconstruction and independent residuals remain on the host. Its reviewed CuPy API/version and explicit device requirements are refusal conditions, not fallback suggestions. Neither path fits factors or silently regularizes the imported moments.

## Homogeneous hierarchy condensation

The opt-in CPU `condense_moment_hierarchy` uses an exhaustive disjoint fine-group partition and native orders 0 through 3. At homogeneous $B\to0$ it constructs leading hierarchy weights:

```{math}
\psi_0=R_0^{-1}\chi/k_\infty,\quad
k_\infty=\nu\Sigma_f^T R_0^{-1}\chi,\quad
\psi_\ell=\frac{\ell}{2\ell+1}R_\ell^{-1}\psi_{\ell-1}.
```

For aggregation $C$ and order-dependent injections $D_\ell$ built from within-coarse-group weights, $R_{\ell,c}=C R_\ell D_\ell$. This preserves the projected leading hierarchy for that solution and partition. It is not a general finite-buckling condensation, a heterogeneous equivalence theorem or a native current-weight correction.

Higher-order weights and effective coefficients can be signed. Near cancellation, singular/ill-conditioned losses or excessive backward residuals cause refusal. The implementation provides no clipping, pseudoinverse, renormalization or physical admission. Optional heat condensation requires an actual heat-production vector; $\nu\Sigma_f$ is not a substitute for heat.
