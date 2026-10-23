# Diffusion API audit

Reviewed on 23 October 2026. The audit followed a user through defining
materials, painting a core, solving criticality and adjoints, changing cross
sections, inspecting power, running kinetics, and adding thermal feedback.
It covered `Model`, `MeshModel`, `HexLattice`/`TriReactor`, their tests and
examples, and the underlying diffusion, transient, power and coupling code.

The existing development plans concern performance, quasi-static acceleration
and multi-GPU diffusion. No separate API development plan was found. The
triangular reactor interface already provided most of the required workflows;
the main gaps were inconsistent access from Cartesian and mesh models, fragile
material ordering, and results that required private fields or manual array
bookkeeping.

## Findings and changes

| Finding | Change |
|---|---|
| Cartesian and mesh models exposed `run`, while the reusable triangular reactor used `steady`. | All three provide `steady` and retain `run`. Existing solve calls continue to work. |
| Changing one material required rebuilding a list in internal index order; Cartesian and mesh models had no geometry-preserving replacement method. | All three accept `with_materials({"fuel": new_fuel})` or a complete ordered list. Integer selectors handle repeated names. Maps are independent copies. |
| A small cross-section change required repeating every field, risking loss of scattering, spectrum or energy-release data. | `Material.replace(**changes)` validates the new material, preserves other fields and copies arrays. |
| Transient callbacks required ordered lists, and Cartesian moving states required raw maps. | Cartesian and triangular `materials_at` callbacks also accept partial name/index mappings. Cartesian `state_at` accepts cached `Model` frames. |
| Cached frames were checked only by array shape; physical dimensions could change silently. | Frames must preserve grid dimensions, physical size, boundaries, active cells and material count. |
| Cartesian and mesh users could not select the Neumann preconditioner through the model API. | `solver_kwargs` consistently carries constructor settings such as `precond_degree`, `dtype` and `linear_solver`. Geometry/device overrides are rejected so the report describes the solved model. |
| Existing Cartesian thermal support required assembling a low-level coupling context. | `Model` and `TriReactor` share the named thermal configuration, coupled steady, coupled transient and quasi-static methods. The Cartesian context selects its matching diffusion solver. |
| Material fractions were private, and power edits required manually supplying maps and volumes. | Steady results expose `material_summary`, `cell_volumes`, `power_density(total_power=...)` and the underlying `.raw` result. Power edits snapshot solved materials/maps. |
| Cartesian kinetics setup did not accept an existing `Kinetics` object; a public steady result could not be reused directly in a neutron transient. | Both are accepted, preserving the component-array and low-level-result forms. |
| Invalid geometry, empty paints, conflicting callbacks and material-name typos failed late or were ignored. | Public entry points reject them with targeted errors before changing assignments or constructing operators. Nonfinite material/kinetics data and zero delayed spectra are rejected. |
| Repeating `fill` retained unused materials, and failed paints could register unused entries. | Filling resets assignments and the material list. Painting validates the selection before registering a material. |
| `MeshModel` accepted string paths but failed for `pathlib.Path`. A one-cell mesh crashed when the solver indexed its empty interior-face array. | Path-like inputs work. Empty adjacency now produces zero interior-face weights, leaving the boundary/removal terms intact. |

## Use the improved workflow

The runnable [diffusion example](../examples/diffusion_workflow.py) covers a
Cartesian reflected core, power edits, a named absorption perturbation and a
hot coupled equilibrium:

```bash
python examples/diffusion_workflow.py --device cpu
```

The same example accepts `--device gpu`. See the
[model reference](model_api.md) for signatures, units and the triangular/mesh
equivalents. A typical parameter change is now:

```python
fuel_hot = fuel.replace(sigma_a=fuel.sigma_a * 1.01)
hot_core = core.with_materials({"fuel": fuel_hot})
result = hot_core.steady(solver_kwargs={"precond_degree": 1})
q = result.power_density(total_power=2e6)
```

## Verification

The original Model/TriReactor API and TWIGL tests passed before the changes
(27 cases). The new [workflow tests](../tests/verification/test_diffusion_workflow_api.py)
check the resulting behavior through actual solves:

- Named and ordered material changes give matching transient histories;
  increased fuel absorption lowers the eigenvalue and transient power.
- Model-level preconditioner options reproduce equivalent low-level solves.
- Power fields integrate to the requested watts within relative `1e-12`,
  including unequal mesh volumes and inactive triangular padding. A separate
  calculation checks energy-release weighting. Later model/material edits
  leave saved-result power unchanged.
- Cartesian zero-feedback hot equilibrium reproduces the uncoupled eigenvalue;
  the steady thermal balance closes below `1e-9`. Unperturbed full and
  quasi-static coupled runs retain unit power. Positive Doppler absorption
  feedback lowers the hot eigenvalue.
- Invalid names, group counts, material counts, grid sizes, selectors,
  callback combinations and kinetics data fail explicitly. A real one-cell
  Gmsh file exercises the empty-adjacency fix.

The latest full CPU result and machine-readable evidence are recorded in the
[V&V report](verification_validation.md#diffusion-api-check).
CUDA execution requires a GPU and was not repeated for this API change.

## Compatibility and remaining work

`run`, complete material lists, component-array kinetics and the raw solver
classes remain available. Invalid inputs that previously slipped through now
raise errors. A box or mesh selector must hit at least one cell centre;
calling `fill` deliberately resets the painting and material list.

Partial material callbacks resolve against the base model at every time: `{}`
selects the base constants. Return cached material/state objects to preserve
operator reuse. A supplied `initial_steady` must represent the same t=0
geometry, material data and boundary conditions; matching array shapes alone
do not establish physical compatibility.

The next useful API work is bounded:

1. Cylindrical r-z geometry still uses `Grid` and the low-level diffusion
   solvers. A dedicated builder should make radial/axial coordinates and true
   annular volumes explicit rather than reinterpreting Cartesian arguments.
2. `MeshModel` remains steady diffusion-only. Transient, adjoint and thermal
   methods need mesh-specific solver support and verification before exposure.
3. Cartesian neutron transients retain `TransientModelResult`; triangular
   transients retain their existing low-level history result. A common history
   interface and restart format would improve consistency, but should preserve
   adaptive-step and device telemetry used by existing studies.

SPH generation, discrete-ordinates transport and specialized feedback hooks
continue to use their existing lower-level interfaces. This change improves
access to established diffusion methods without changing their discretization
or promoting experimental solver settings to defaults.
