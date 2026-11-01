# Native P0 reaction input for experimental NEM

`NEMDiffusionSolver` and `NEMHigherOrderDiffusionSolver` accept an explicit pair
of keyword arguments, `native_total` and `native_scattering_production`. Pass
`None` for both ordinary `sigma_a` and `sigma_s` arguments in this mode:

```python
solver = NEMDiffusionSolver(
    grid, D, None, None, nu_sigma_f, chi,
    native_total=total,
    native_scattering_production=sp0_from_major,
    fission_matrix=full_fission_matrix,
)
```

Native total has shape `(N,G)` and production has shape `(N,G,G)`, with production
indexed `[node, from_group, to_group]`. Group-only inputs also broadcast across
nodes, following the existing NEM field convention. Values must be finite and
real, total strictly positive, and production nonnegative. The solver copies the
native arrays into immutable storage and forms the reaction operator directly:

```text
R[node, to, from] = diag(total[node])[to, from] - SP0[node, from, to]
```

Self-scattering is included. Every reaction diagonal must be strictly positive;
this condition alone does not establish nonsingularity, conditioning, stability,
positivity, or physical validity. Existing local/global condition, factorization,
ADF, fission, and residual checks still apply. No coefficients are clipped,
renormalized, shifted, fitted, or inferred.

`reaction_mode` is `native_p0_unqualified`. The signed compatibility diagnostic
`total - SP0.sum(axis=2)` is available as `signed_reduced_absorption` and
`sigma_a`. It is **not physical absorption or the reported Serpent `INF_RABSXS` estimate**. The
compatibility `sigma_s` contains off-diagonal SP0; mathematical assembly and
reconstruction use the directly constructed `R`. Higher-order local polynomial
responses, volume/weak residuals, and all spatial moments use that same operator.

The mode, native arrays, and signed diagnostic are included in the cached
configuration fingerprint and backend/result diagnostics. Caller mutation cannot
alter the copied native arrays. Replacing solver attributes after assembly is
rejected by the existing cache guard, even when a change only affects raw
provenance while leaving `R` unchanged.

Without either native keyword, the ordinary absorption/scattering behavior is
unchanged, including rejection of negative absorption. Supplying just one native
keyword, or mixing native inputs with ordinary cross sections, is rejected.
The complete emitted/source `fission_matrix` API retains its existing nonnegative
column-sum requirement against `nu_sigma_f`; native reaction mode does not change
or normalize fission data or chi. This feature performs no transport correction,
diffusion-coefficient derivation, cohort preparation, or physical qualification.
