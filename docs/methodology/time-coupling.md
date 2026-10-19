# Time dependence, heat and neutron noise

## Spatial neutron kinetics

With delayed family production $\beta_i\nu\Sigma_f\phi$, conventional diffusion kinetics is

```{math}
\frac{1}{v_g}\frac{\partial\phi_g}{\partial t}+A_g\phi
=\chi_{p,g}(1-\beta)\sum_h\nu\Sigma_{f,h}\phi_h
+\sum_i\chi_{d,i,g}\lambda_i C_i,
\qquad
\frac{\partial C_i}{\partial t}=\beta_i\sum_h\nu\Sigma_{f,h}\phi_h-\lambda_iC_i.
```

Here $A$ includes spatial loss and intergroup scattering. The exact input capabilities vary by transient class. `Kinetics` specifies velocities, delayed fractions, decay constants and optional delayed spectra under that class's shape contract. General raw regional family matrices used by the static NEM projection are not automatically accepted by these transient solvers.

A transient initialized from an arbitrary eigenstate uses the documented critical adjustment: fission is divided by its initial $k_0$ so an unchanged equilibrium stays stationary. This is a deliberate transient reference model. It differs from evolving an unadjusted off-critical physical reactor.

Backward Euler is the baseline implicit scheme. BDF pathways extend the history approximation, with startup/restart and optional adaptive controllers. Abrupt state changes require aligned time points and restart controls where appropriate. Decreasing a linear tolerance does not remove time error. Compare time refinements and inspect rejected steps, startup order and discontinuity handling. Coupled drivers may impose stronger schedule restrictions than a standalone transient solver.

## Quasi-static reduction

Factor the spatial flux as $\phi(\mathbf r,E,t)=a(t)\psi(\mathbf r,E,t)$. Importance-weighted projection yields a small amplitude/precursor system. Fixed-shape reduction freezes $\psi$; IQS updates it through macro corrections while retaining fine-step amplitude and spatial precursor history. Adiabatic mode instead uses instantaneous eigen shapes.

Shape replacement changes the population coordinate. The old/new $\langle\phi^*,v^{-1}\psi\rangle$ ratio is needed to preserve the intended amplitude normalization. Consequently physical `.power` is `.amplitude * .power_shape_factor`, and should not be inferred from `.amplitude` alone.

Residual thresholds can force early shape correction or full-diffusion fallback. `adjoint_every` limits reuse age and `adjoint_residual_tol` detects an inadequate reused importance shape. Predictor disagreement can reduce later macro intervals. These guards provide evidence about the approximation; they do not replace comparisons with full spatial kinetics at representative rapid maneuvers.

## Conduction and feedback

The thermal model is conduction with an optional volumetric sink:

```{math}
c_V\frac{\partial T}{\partial t}-\nabla\cdot\kappa\nabla T
+h(T-T_{\rm sink})=q''',\qquad
-\kappa\partial_nT=\alpha_T(T_s-T_\infty).
```

It uses the same conservative leakage machinery with conductivity replacing diffusion. A source with entirely adiabatic boundaries and no sink has no steady solution. Discontinuity-factor diffusion operators do not preserve a constant thermal field and are rejected by the conduction interface.

An eigenflux defines a heat shape. Imposing domain power $P$ gives

```{math}
q'''_c=P\frac{\sum_g\kappa\Sigma_{f,cg}\phi_{cg}}
{\sum_jV_j\sum_g\kappa\Sigma_{f,jg}\phi_{jg}}.
```

The preferred weight is fission energy production. The documented fallback to $\nu\Sigma_f$ is an approximation. A 2-D model's prescribed power belongs to its modeled thickness, not automatically the full reactor rating.

Feedback can add absorption changes proportional to $\sqrt T-\sqrt{T_{\rm ref}}$ and scale macroscopic constants by a linearized density factor. The latter changes density on a frozen mesh; it does not model structural expansion. In steady coupling the power level is imposed while temperature changes shape and $k$. Picard relaxation or acceleration converges the fixed point. Inner neutron noise sets a useful floor on the outer tolerance.

Coupled transients keep supported fields on the solve device and subcycle heat on larger windows. Fixed schedules require compatible neutron/thermal intervals. Cached control frames avoid unnecessary operator reconstruction; changed physics must trigger reconstruction. Opt-in mixed precision changes preconditioning, while state and true convergence checks remain in FP64 and fallback counts remain reportable.

External preCICE examples exchange volume power and temperature. The matched-centroid setup tests coupling infrastructure by sharing the same physics half-steps as the internal driver. Identity mapping evidence in that setup does not establish accuracy for dissimilar meshes. Consult the [preCICE documentation](https://precice.org/docs.html) when changing mapping or acceleration.

## Frequency-domain noise

Linearize about a static critical-adjusted reference and use $\partial_t\mapsto i\omega$. Eliminating delayed precursors gives

```{math}
\chi_{\rm eff,g}(\omega)=\chi_g-\sum_i\chi_{d,i,g}\beta_i
\frac{i\omega}{i\omega+\lambda_i}.
```

The complex diffusion/SPN fixed-source solve includes $i\omega/v_g$ and cross-section perturbations multiplied by the reference flux. The sign convention is the positive $i\omega$ Fourier convention. Complex-symmetric operators use the implemented complex Krylov machinery; their behavior differs from real SPD CG.

`NoiseSolver` supports the documented global kinetics data, not every per-material transient table. Perturbations must remain small enough for linearization. At zero frequency a critical unfeedbacked system has a neutral mode; a DC solve is not an ordinary finite transfer-function evaluation. The reflected homogeneous limit is checked against

```{math}
G(\omega)=\left[i\omega\left(\Lambda+
\sum_i\frac{\beta_i}{i\omega+\lambda_i}\right)\right]^{-1}.
```
