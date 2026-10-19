# Governing equations and conventions

## Multigroup balance

For fast-to-thermal group ordering, the standard diffusion eigenproblem is

```{math}
-\nabla\cdot D_g\nabla\phi_g+\Sigma_{r,g}\phi_g
=\sum_{h\ne g}\Sigma_{s,h\to g}\phi_h+
\frac{\chi_g}{k}\sum_h\nu\Sigma_{f,h}\phi_h,
\qquad
\Sigma_{r,g}=\Sigma_{a,g}+\sum_{h\ne g}\Sigma_{s,g\to h}.
```

In matrix notation $A\phi=F\phi/k$. The `Material.sigma_s` convention is **source group, destination group**. Thus scattering action uses its transpose. Self-scattering cancels between removal and source and is not counted twice. Macroscopic cross sections have units cm$^{-1}$; diffusion coefficients have units cm; velocities have units cm/s and time has units s. An eigenvector has arbitrary magnitude. Ratios, reaction rates normalized to production, and imposed-power shapes must state their normalization.

A separable fission kernel is $F_{gh}=\chi_g\nu\Sigma_{f,h}$. Specialized NEM production interfaces also accept a nonnegative emitted-group/source-group matrix whose column sums equal $\nu\Sigma_f$. A changed kernel is a changed physical model, even if its column sums are preserved.

## Boundaries and adjoints

Reflective boundaries impose zero normal current. Zero-flux boundaries impose a Dirichlet condition. The diffusion vacuum approximation is a Robin law $J_n=\alpha\phi_s$ with the package's vacuum coefficient; it is not the same boundary as zero flux. Collapsed dimensions must not introduce artificial leakage. A cylindrical axis at $r=0$ has a natural zero-area boundary.

The discrete adjoint satisfies $A^T\phi^*=F^T\phi^*/k$ in the coordinates and weighting of the actual solver. Reversing group scattering alone is insufficient when geometry weighting, factors or source recovery introduce other transformations. An importance solution is not a physical neutron density, so its balance must not be interpreted as an absorption/leakage fraction.

For a small perturbation, first-order eigenvalue sensitivity follows the bilinear contraction of forward and adjoint states. With $\rho=1-1/k$,

```{math}
\delta\rho\simeq
\frac{\langle\phi^*,(\delta F/k-\delta A)\phi\rangle}
{\langle\phi^*,F\phi\rangle}.
```

This expression is for the unadjusted physical pencil with compatible dual coordinates. Compare finite perturbations with separately converged eigenproblems; linear sensitivity is only a local approximation.

## Iteration and execution

Standard eigen solvers use fission-source power iteration with multigroup source sweeps and within-group Krylov solves. Both eigenvalue change and fission-source convergence matter: a nearly stationary $k$ can coexist with an unconverged flux shape. Appropriate linear tolerances tighten with the outer solution. Upscattering and fission couple energy groups; acceleration changes iteration, not the target equations. For three or more groups, an unsettled fixed-source solve may use flexible GMRES on the full group/moment scattering system, preconditioned by a lower-group sweep. Every preconditioner sweep consumes the declared group-sweep budget. Restart storage is estimated from complete block sizes and capped at 256 MiB (and one eighth of free device memory on GPU); insufficient or unknown device memory retains ordinary sweeps. This is conservative allocation planning, not a measured peak-memory guarantee. The true fixed-source residual and physical scalar checks still determine acceptance.

Matrix-free operators store face coefficients and diagonals and apply local stencils through NumPy or CuPy. CG requires a symmetric positive definite formulation. Cylindrical volume weighting and suitable moment transformations can supply it; nonsymmetric closures require the supported nonsymmetric Krylov path. Do not select CG merely because a continuum diffusion equation looks symmetric.

`device="auto"` is a convenience policy, not evidence of GPU execution. Record the reported backend, device, precision and library versions. The native-moment hybrid GPU solver is a separate assembled CSR implementation with host-side outer physics and residual checks. NEM sparse LU remains CPU-only.
