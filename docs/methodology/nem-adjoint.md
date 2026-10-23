# Matching NEM adjoint and static kinetics

## Conventional forward and adjoint equations

Write the multigroup forward scalar flux as $\phi$ and the adjoint flux
(neutron importance) as $\phi^*$. With loss-minus-scattering operator
$\mathcal L$ and total fission production $\mathcal F$,

```{math}
\mathcal L\phi=\frac{1}{k}\mathcal F\phi,\qquad
\mathcal L^\dagger\phi^*=\frac{1}{k}\mathcal F^\dagger\phi^*.
```

The familiar physical definitions are

```{math}
\beta_{\mathrm{eff},c}=\frac{\langle\phi^*,\mathcal F_{d,c}\phi\rangle}
{\langle\phi^*,\mathcal F\phi\rangle},\qquad
\Lambda=\frac{\langle\phi^*,v^{-1}\phi\rangle}
{\langle\phi^*,\mathcal F\phi\rangle},\qquad \ell=k\Lambda.
```

The inner product is the spatial integral and energy-group sum. GARNET evaluates
matching discrete NEM bilinear forms, including all retained moments and the
adjoint interface treatment for fixed ADFs. It computes the diffusion-model
importance, not an angle-dependent transport adjoint. The implementation
coordinates below explain how these physical pairings are evaluated without
mistaking dual source weights for point samples of $\phi^*$.

## Why the complete source response matters

The frozen uncondensed physical NEM eigenpencil is $A x=Qx/k$. The state $x$ contains means, Legendre coefficients and shared face currents. Let $P$ extract means and every retained weak spatial moment, $E$ inject physical source terms, and $C$ be the local production kernel. Then

```{math}
y=Px,\quad H=PA^{-1}E,\quad Q=ECP,\quad T=HC,\quad Ty=ky.
```

Fission acts on every retained source moment. A mean-only transpose misses pathways and generally does not give the physical static adjoint. With condensed mean/current matrix $A_c$, recovered moment maps $M_J,M_q$, source-dependent face map $F_q$ and face constraints $W$, the forward response is

```{math}
\begin{bmatrix}\bar\phi\\J\end{bmatrix}
=A_c^{-1}\begin{bmatrix}q_{\rm mean}\\-WF_q q_{\rm moment}\end{bmatrix},
\qquad m=M_JJ+M_q q_{\rm moment}.
```

Every path must be reversed. For dual input $w$,

```{math}
s=A_c^{-T}\begin{bmatrix}w_{\rm mean}\\M_J^T w_{\rm moment}\end{bmatrix},\qquad
H^T w=\begin{bmatrix}s_{\rm mean}\\M_q^T w_{\rm moment}-F_q^TW^Ts_J\end{bmatrix}.
```

If LU factors the equilibrated matrix $\operatorname{diag}(r)A_c\operatorname{diag}(c)$, transpose application is $r\,\mathrm{LU}^{-T}(c\,b)$. Transposing the visible condensed matrix without recovery/direct-source terms is insufficient.

Solve $T^Tw=kw$ and recover $z=H^Tw$. The full physical left state is $\ell=A^{-T}P^Tw$, and $E^T\ell=z$. The familiar continuous notation $\phi^*$ represents importance, but **$z$ is source-coordinate importance**, not a node-center adjoint flux. Equation scaling and volumes are already included in its dual action.

## Physical parameters

For inverse velocity $v^{-1}$ constant within each node/group, the physical mass is $B=E v^{-1}P$. The two highest Legendre coefficients are orthogonal to the retained weak tests and do not enter separately. A delayed-family matrix gives $Q_i=EC_iP$. Therefore

```{math}
\Lambda=\frac{z^T v^{-1}y}{z^TCy},\qquad
\beta_{{\rm eff},i}=\frac{z^TC_i y}{z^TCy},\qquad
\ell_{\rm neutron}=k\Lambda.
```

Do not multiply these contractions by node volume again. Arbitrary compatible forward/adjoint scale cancels. The API checks eigen residuals, matching state, positive time/fission importance and nonnegative projected fractions rather than repairing them.

The physical static generation time is $T/F$. The existing critical-adjusted quasi-static transient projection reports $k_0T/F$ under its transient convention. These quantities need not agree off critical; their names alone do not establish equal definitions.

Keep the complete positive decay vector. The optional one-family low-frequency reduction is

```{math}
\lambda_{\rm LF}=\frac{\sum_i\beta_i}{\sum_i\beta_i/\lambda_i}.
```

It is undefined without delayed production and is not claimed to equal a Monte Carlo code's reported scalar decay constant. Distinct isotope/native-family channels can have distinct decay poles. Summing beta for reporting is exact; replacing several physical decay poles by an arithmetic mean is a transient approximation.

## Required production evidence

`read_regional_kinetics` requires raw regional production, exact descending energy boundaries, stable region/family IDs, inverse speeds, common channel decay data, source hashes and declared approximations. Total production is `(region,target,source)` and delayed production is `(region,family,target,source)`. Prompt remainder must be nonnegative within roundoff. Global effective beta is never a substitute for raw regional family production.

`for_nodes` explicitly maps physical nodes to regions and verifies energy and frozen total-kernel compatibility. Repeated IDs permit a declared symmetry mapping; they do not prove symmetry. `project_nem_time` is useful when only inverse speeds are qualified, but its output is a time-only result.

Native micro-depletion and ACE readers provide additional explicit building blocks. Isotope coverage, MT452/455 pairing, branch applicability, table identity, spectrum law and energy domain remain caller qualifications. Missing family data are not synthesized from a global IFP estimate. Yield transfer between homologous score domains is an explicit approximate modeling operation with declared domain identities and qualification.

`assemble_kinetics_production` builds a positive prompt-plus-delayed kernel from explicit spectra and yields. If a frozen sampled total spectrum has zero tails incompatible with a delayed library, subtracting delayed channels can legitimately fail. Do not clip the tails or weaken positivity. A rebuilt total kernel requires a matching homogeneous reference and factor reconstruction.

## Numerical evidence and limits

`tests/test_nem_kinetics.py` checks signed $H/H^T$ dot products, complete source columns, independent uncondensed pencils, quadrature mass/delayed matrices, unequal nodes and factors, scaling and normalization, leakage policies and polynomial degrees. The independent alpha perturbation identity $k'(0)=-k^2\Lambda$ tests time weighting.

These are strong discrete verification tests. They do not validate a new reactor dataset or show Monte Carlo/diffusion estimator equivalence. This NumPy/SciPy implementation is static and CPU-only. It supplies no three-dimensional transient NEM evolution. Material or geometry replacement invalidates cached responses and requires a rebuilt solver.
