# Feature integration inventory

This inventory records the local `feature/*` tips inspected on 2026-10-19,
against pre-integration `dev` at `a8859c5`. The selected ancestry closures are
`feature/kinetics-parameters` (`222c132`),
`feature/moment-hierarchy-condensation` (`c0acb76`) and corner verification
(`93392d0`). Selection means integration scope, not a blanket claim of production
readiness. The platform, Serpent fixed-state interface, common-mesh FV and
convergence work form the shared base. NEM/reference/ADF and matching static
kinetics are experimental CPU paths. Independent moment CPU/GPU and energy
condensation remain explicitly opt-in experimental capabilities.

The ancestry column below records actual commit containment, rather than
inferring ordering from branch names. “Archived” means retain the original
research history on their existing branches, outside the default integrated API;
it does not authorize deleting the original evidence. These are source feature
names at audit time; archival names may gain that namespace during integration.

| Local feature branch | Tip | Actual ancestry or divergence | Disposition |
|---|---|---|---|
| `feature/amd-rocm-support` | `13a59cc` | Already contained in baseline dev and every other listed tip | Integrated platform base |
| `feature/serpent-xs-interface` | `2457d27` | Ancestor of mesh-moments and serpent-adf | Integrate fixed-state XS interface |
| `feature/serpent-adf` | `4f41715` | Diverges from mesh-moments after the XS base; three commits outside selected closures | Archived; later common-mesh reader/DF implementation selected |
| `feature/isonees2027-mesh-moments` | `d628746` | Ancestor of rectilinear FV, kinetics, moment condensation and corner tips | Integrate common-mesh solvers; moment trace factors opt-in experimental |
| `feature/rectilinear-fv-mesh` | `8e723e8` | Ancestor of eigen-source convergence and kinetics | Integrate FV geometry |
| `feature/eigen-source-convergence` | `9972990` | Ancestor of nodal-polynomial and coupled-cuboid references | Integrate source-convergence correction |
| `feature/nodal-polynomial-reference` | `38e5ce7` | Ancestor of NEM diffusion prototype and kinetics | Integrate experimental local reference |
| `feature/nem-diffusion-prototype` | `a60f997` | Ancestor of NEM boundary leakage and kinetics | Integrate experimental CPU NEM |
| `feature/nem-boundary-leakage` | `5cb5ab8` | Ancestor of polynomial order and kinetics | Integrate explicit leakage policy |
| `feature/nem-polynomial-order` | `45a9f7a` | Ancestor of polynomial receiver and kinetics | Integrate experimental reference order support |
| `feature/nem-polynomial-receiver` | `c0af0e2` | Ancestor of degree16 and kinetics | Integrate experimental CPU higher-order receiver |
| `feature/nem-degree16` | `e7b5f5f` | Ancestor of reflector-adf-nem and kinetics | Integrate experimental degree/order checks |
| `feature/reflector-adf-nem` | `c833b5b` | Ancestor of kinetics; shares reflector work with later divergent reflector branches | Integrate experimental reflector/NEM baseline |
| `feature/kinetics-parameters` | `222c132` | Selected NEM/kinetics tip; five commits beyond reflector-adf-nem | Integrate matching static forward/adjoint projection, raw data readers, reference/production contracts |
| `feature/serpent-moment-scattering` | `db56b6b` | Ancestor of serpent-moment-gpu and condensation | Integrate opt-in experimental independent CPU moment path |
| `feature/serpent-moment-gpu` | `f90fa88` | Ancestor of condensation | Integrate opt-in experimental CuPy moment path |
| `feature/moment-hierarchy-condensation` | `c0acb76` | Selected independent moment tip | Integrate opt-in experimental energy condensation |
| `feature/isonees2027-corner-verification` | `93392d0` | Diverges after mesh-moments; one additional verification commit | Integrate Robin rectangle/reentrant-corner verification |
| `feature/coupled-cuboid-reference` | `9164d9f` | Diverges after eigen-source convergence; ancestor of modal | Archived superseded coupled reference |
| `feature/coupled-reference-modal` | `260d9a3` | Descends from coupled-cuboid; ancestor of face-moments | Archived superseded modal backend |
| `feature/coupled-reference-face-moments` | `a3e5a8f` | Descends from modal; ancestor of native-mean | Archived superseded measured face-moment path |
| `feature/coupled-reference-native-mean` | `61f195e` | Descends from face-moments; ancestor of receiver-shaped | Archived superseded native-mean constraint |
| `feature/receiver-shaped-reference` | `408cbd0` | Descends from native-mean; five commits outside selected closures | Archive nonuniform reference experiments; selectively integrate `ndgpu/diffusion_currents.py` only |
| `feature/face-average-adf` | `72f39c9` | Diverges after eigen-source convergence | Archived face-average coupling prototype |
| `feature/native-face-trace-adf` | `57a7d55` | Diverges from selected NEM line; four commits outside selected closures | Archived native-face perturbation prototype |
| `feature/reflector-inclusive-adf` | `b8acadd` | Diverges after shared reflector diagnostics; ancestor of interface-factors and vacuum | Archived superseded affine reflector/reference path |
| `feature/reflector-interface-factors` | `cf5eb26` | Descends from reflector-inclusive; tip patch replayed as `93fb621` in selected NEM ancestry | Archived branch; equivalent factor extension already selected |
| `feature/reflector-vacuum-reference` | `5bf9aef` | Descends from reflector-inclusive; ancestor of incoming | Archived receiver-vacuum reference experiment |
| `feature/reflector-incoming-reference` | `7965cc6` | Descends from reflector-vacuum | Archived incoming-current reference experiment |

## Patch duplication and selective integration

The audit enumerated all 29 local feature refs, used `git merge-base
--is-ancestor` to test containment, and compared stable patch IDs for all
non-merge commits reachable from those refs outside baseline dev. There is one
pair of distinct commits with an identical stable patch ID:
`cf5eb261f5b7` = `93fb6211f46f` (internal reflector factor extension).
Shared ancestor commits are already the same objects and are not duplicates to
cherry-pick again. No other distinct patch-identical pair was found in that
scope. Later merges do not change this fixed-baseline audit.

In particular, the three divergent Serpent ADF commits `067324d`, `89eaf79` and
`4f41715` are **not** patch-identical to the selected common-mesh commits.
The selected line reworks their functionality through `4ccca9e` (scalar DF),
`dd5c2a7` (native four/six-face ADF reader) and subsequent mesh work. Semantic
overlap is not proof of patch duplication or a reason to merge both versions.

`408cbd0` combines conservative current export with nonuniform local-reference
work. Only the reusable `ndgpu/diffusion_currents.py` is selected, together with
appropriate verification. The full commit and its coupled/nonuniform ancestry
remain archived. The selected NEM kinetics line instead contains the matching
local reference and factor construction in `3f909ac`; it does not require
reviving the superseded coupled/modal/native-mean branches.

Reproduce the audit using the recorded baseline, even after integration:

```sh
git for-each-ref --format='%(refname:short) %(objectname)' refs/heads/feature/
git merge-base --is-ancestor <tip> <selected-tip>
git cherry a8859c5 <tip>
git show --pretty=format: <commit> | git patch-id --stable
```

`git cherry` compares patch equivalence with the specified upstream only;
ancestry checks and the union of the three selected closures are also needed.
Do not infer commit equivalence solely from filenames or subject lines.

## Portable verification entry points

`examples/isonees_features/` supplies small synthetic common-mesh FV,
local-reference/ADF and raw two-family static kinetics examples. They require
NumPy/SciPy only and no Serpent/ACE files. See that directory's README for exact
commands and physical limits. `tests/test_isonees_examples.py` checks reflected
homogeneous eigenvalues, the manufactured corrected-trace/mean jump, global
balance, raw family fractions and physical time ratios. These analytical smoke
checks supplement subsystem verification; they do not establish reactor-data,
full-core, transport or transient qualification. GPU execution requires its own
backend/device evidence and is not implied by these CPU examples.
