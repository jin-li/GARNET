"""Public diffusion workflows: stable physics, named edits and usable results."""

import numpy as np
import pytest

from garnet import (DiffusionEigenSolver, FeedbackSpec, HexLattice, Kinetics,
                    Material, MeshModel, Model, PWR_TWO_GROUP, ThermalMaterial,
                    UnstructuredDiffusionSolver, assemble_mesh)


FUEL = PWR_TWO_GROUP.replace(name="fuel", kappa_fission=[0.004, 0.05])
REFLECTOR = Material(name="reflector", diffusion=[1.2, 0.3], sigma_a=[0.001, 0.02],
                     nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.03], [0.0, 0.0]])
KINETICS = Kinetics([1e7, 3e5], [0.0065], [0.08])
TOL = dict(tol_k=1e-9, tol_source=1e-8)


def cartesian():
    return (Model(size=(80, 80), cells=(8, 8)).fill(REFLECTOR)
            .add_box(FUEL, x=(20, 60), y=(20, 60)))


def triangular():
    return (HexLattice(pitch=20, refine=1).set_disk(1, REFLECTOR)
            .set_site((0, 0), FUEL).build())


def unstructured():
    # Two cells of unequal area: fractions must be volume-weighted.
    mesh = assemble_mesh([(0, 0), (20, 0), (60, 0), (0, 40), (20, 40), (60, 40)],
                         [(0, 1, 4, 3), (1, 2, 5, 4)], [1, 2])
    return MeshModel(mesh).fill(REFLECTOR).assign(FUEL, tag=2)


@pytest.mark.parametrize("build", [cartesian, triangular, unstructured])
def test_named_replacement_preserves_original_geometry_and_changes_reactivity(build):
    core = build()
    old_map = core.material_map.copy()
    old_materials = core.materials[:]
    poisoned = FUEL.replace(sigma_a=FUEL.sigma_a * 1.01)
    changed = core.with_materials({"fuel": poisoned})
    assert core.materials == old_materials
    np.testing.assert_array_equal(changed.material_map, old_map)
    assert changed.steady(device="cpu", **TOL).k_eff < core.steady(device="cpu", **TOL).k_eff
    # Edits to a copied geometry must not leak back into the original.
    if hasattr(changed, "_map"):
        changed._map[...] = 0
    elif hasattr(changed, "_cellmat"):
        changed._cellmat[...] = 0
    else:
        changed.material_map[...] = 0
    np.testing.assert_array_equal(core.material_map, old_map)


@pytest.mark.parametrize("build", [cartesian, triangular, unstructured])
def test_power_and_material_edits_use_solved_state(build):
    core = build()
    result = core.steady(device="cpu", **TOL)
    assert result.converged and result.raw.k_eff == result.k_eff
    q = result.power_density(total_power=1234.0)
    assert q.shape == result.flux.shape[1:]
    assert np.sum(q * result.cell_volumes) == pytest.approx(1234.0, rel=1e-12)
    rows = {row.name: row for row in result.material_summary}
    assert sum(row.volume_fraction for row in rows.values()) == pytest.approx(1.0)
    assert rows["fuel"].fission_fraction == pytest.approx(1.0)
    assert rows["reflector"].fission_fraction == 0.0
    if build is unstructured:
        assert rows["fuel"].volume_fraction == pytest.approx(2 / 3)
    if hasattr(core, "fill"):
        core.fill(REFLECTOR)
    else:
        core.material_map[...] = 0
    np.testing.assert_array_equal(result.power_density(1234.0), q)


def test_result_power_density_matches_energy_cross_sections():
    core = cartesian()
    result = core.steady(device="cpu", **TOL)
    fuel_cells = core.material_map == 1
    density = (np.asarray(FUEL.kappa_fission)[:, None, None, None]
               * result.flux).sum(axis=0)
    density *= fuel_cells[..., None]
    expected = density * (100.0 / np.sum(density * result.cell_volumes))
    np.testing.assert_allclose(result.power_density(100.0), expected, rtol=1e-13)
    with pytest.raises(ValueError, match="adjoint"):
        core.steady(adjoint=True, device="cpu").power_density(100.0)
    for invalid in (-1, np.nan, np.inf):
        with pytest.raises(ValueError, match="total_power"):
            result.power_density(invalid)


def test_later_cross_section_edits_do_not_change_saved_result_power():
    fuel = FUEL.replace()
    core = Model((80, 80), (8, 8)).fill(REFLECTOR).add_box(fuel, x=(20, 60), y=(20, 60))
    result = core.steady(device="cpu")
    q = result.power_density(100.0)
    fuel.kappa_fission[:] = [100, 0]
    np.testing.assert_array_equal(result.power_density(100.0), q)


@pytest.mark.parametrize("build", [cartesian, triangular, unstructured])
def test_replacements_reject_wrong_names_types_groups_and_counts(build):
    core = build()
    for replacements, message in [({"typo": FUEL}, "unknown"),
                                   ({999: FUEL}, "out of range"),
                                   ([FUEL], "count/order"),
                                   ({"fuel": "not a Material"}, "Material")]:
        with pytest.raises((ValueError, TypeError), match=message):
            core.with_materials(replacements)
    one_group = Material([1.0], [0.01], [0.02])
    with pytest.raises(ValueError, match="groups"):
        core.with_materials({"fuel": one_group})


def test_ambiguous_names_require_an_index_and_duplicate_selectors_fail():
    core = Model(80, 8).fill(FUEL).add_box(FUEL.replace(), x=(0, 40))
    with pytest.raises(ValueError, match="ambiguous"):
        core.with_materials({"fuel": FUEL})
    assert core.with_materials({1: REFLECTOR}).materials[1] is REFLECTOR
    unique = cartesian()
    with pytest.raises(ValueError, match="twice"):
        unique.with_materials({"fuel": FUEL, 1: FUEL})


def test_material_replace_copies_arrays_and_preserves_other_data():
    original = FUEL.replace()
    changed = original.replace(sigma_a=original.sigma_a * 1.01)
    np.testing.assert_array_equal(changed.kappa_fission, original.kappa_fission)
    assert changed.name == original.name
    for name in ("diffusion", "sigma_a", "sigma_s", "nu_sigma_f", "chi", "kappa_fission"):
        assert not np.shares_memory(getattr(original, name), getattr(changed, name))
    with pytest.raises(ValueError, match="non-negative"):
        original.replace(sigma_a=[-1, 0.1])
    with pytest.raises(TypeError):
        original.replace(sig_a=[0.01, 0.1])


@pytest.mark.parametrize("field", ["diffusion", "sigma_a", "nu_sigma_f", "chi",
                                  "total", "kappa_fission"])
def test_nonfinite_cross_sections_name_the_invalid_field(field):
    with pytest.raises(ValueError, match=field):
        FUEL.replace(**{field: [np.nan, 0.1]})


@pytest.mark.parametrize("field", ["velocities", "beta", "decay"])
def test_nonfinite_kinetics_name_the_invalid_field(field):
    values = dict(velocities=[1e7, 3e5], beta=[0.0065], decay=[0.08])
    values[field][0] = np.nan
    with pytest.raises(ValueError, match=field):
        Kinetics(**values)


def test_delayed_spectrum_and_kinetics_shape_errors_are_clear():
    with pytest.raises(ValueError, match="chi_delayed"):
        Kinetics([1e7, 3e5], [0.0065], [0.08], chi_delayed=[0, 0])
    with pytest.raises(ValueError, match="velocities"):
        cartesian().set_kinetics(Kinetics([1e7], [0.0065], [0.08]))
    with pytest.raises(ValueError, match="material rows"):
        cartesian().set_kinetics(Kinetics([[1e7, 3e5]], [0.0065], [0.08]))


def test_constructor_options_reach_cartesian_and_mesh_solvers():
    core = cartesian()
    actual = core.steady(device="cpu", solver_kwargs={"precond_degree": 1}, **TOL)
    reference = DiffusionEigenSolver(core.grid, core.materials, core._map,
        bc=core._full_bc(), device="cpu", precond_degree=1).solve(**TOL)
    assert actual.k_eff == pytest.approx(reference.k_eff, abs=1e-12)
    mesh = unstructured()
    actual = mesh.steady(device="cpu", solver_kwargs={"precond_degree": 1}, **TOL)
    reference = UnstructuredDiffusionSolver(mesh.mesh, mesh.materials, mesh.material_map,
        device="cpu", precond_degree=1).solve(**TOL)
    assert actual.k_eff == pytest.approx(reference.k_eff, abs=1e-12)


@pytest.mark.parametrize("build", [cartesian, triangular, unstructured])
def test_solver_options_cannot_change_geometry_behind_the_report(build):
    with pytest.raises(ValueError, match="cannot override model settings"):
        build().steady(device="cpu", solver_kwargs={"material_map": np.zeros(1)})


@pytest.mark.parametrize("build", [cartesian, triangular])
def test_named_transient_equals_ordered_materials_and_accepts_steady_result(build):
    core = build().set_kinetics(KINETICS)
    absorber = FUEL.replace(sigma_a=FUEL.sigma_a * [1.0, 1.003])
    steady = core.steady(device="cpu", **TOL)
    named = core.transient(t_end=0.10, dt=0.02, device="cpu", initial_steady=steady,
        materials_at=lambda t: {} if t == 0 else {"fuel": absorber},
        solver_kwargs={"precond_degree": 1})
    ordered = core.transient(t_end=0.10, dt=0.02, device="cpu", initial_steady=steady.raw,
        materials_at=lambda t: [absorber if m is FUEL and t > 0 else m for m in core.materials],
        solver_kwargs={"precond_degree": 1})
    np.testing.assert_allclose(named.power, ordered.power, rtol=1e-12)
    assert named.power[-1] < 1.0
    with pytest.raises(ValueError, match="unknown"):
        core.transient(t_end=0.02, dt=0.02, materials_at=lambda t: {"typo": absorber})


def test_cartesian_state_frames_equal_named_material_changes():
    core = cartesian().set_kinetics(KINETICS)
    absorber = FUEL.replace(sigma_a=FUEL.sigma_a * [1.0, 1.003])
    changed = core.with_materials({"fuel": absorber})
    framed = core.transient(t_end=0.10, dt=0.02, device="cpu",
                            state_at=lambda t: core if t == 0 else changed)
    named = core.transient(t_end=0.10, dt=0.02, device="cpu",
        materials_at=lambda t: {} if t == 0 else {"fuel": absorber})
    np.testing.assert_allclose(framed.power, named.power, rtol=1e-12)
    with pytest.raises(ValueError, match="only one"):
        core.transient(0.1, 0.02, materials_at=lambda t: {}, at=lambda t: None)


def test_transient_initial_report_uses_initial_callback_materials():
    core = cartesian().set_kinetics(KINETICS)
    absorber = FUEL.replace(name="initial fuel", sigma_a=FUEL.sigma_a * 1.01)
    result = core.transient(t_end=0.04, dt=0.02, device="cpu",
                            materials_at=lambda t: {"fuel": absorber})
    reference = core.with_materials({"fuel": absorber}).steady(device="cpu",
                                                            tol_k=1e-9, tol_source=1e-8)
    assert result.k0 == pytest.approx(reference.k_eff, abs=1e-7)
    assert "initial fuel" in {row.name for row in result.steady.material_summary}
    np.testing.assert_allclose(result.power, 1.0, atol=1e-6)


@pytest.mark.parametrize("build", [cartesian, triangular])
def test_state_frames_must_preserve_physical_geometry_and_boundary(build):
    core = build().set_kinetics(KINETICS)
    if build is cartesian:
        other = Model(size=(160, 80), cells=(8, 8)).fill(REFLECTOR).add_box(FUEL)
    else:
        other = HexLattice(pitch=40, refine=1).set_disk(1, REFLECTOR).set_site((0, 0), FUEL).build()
    with pytest.raises(ValueError, match="physical size"):
        core.transient(0.02, 0.02, state_at=lambda t: other, device="cpu")


@pytest.mark.parametrize("size,cells", [((np.nan,), (8,)), ((np.inf,), (8,)),
                                       ((80,), (8.5,)), ((80,), (True,))])
def test_invalid_grid_input_is_rejected(size, cells):
    with pytest.raises(ValueError):
        Model(size, cells)


@pytest.mark.parametrize("build", [cartesian, unstructured])
def test_failed_paint_does_not_register_a_material(build):
    core = build()
    before = core.materials
    with pytest.raises(ValueError, match="range|selects no"):
        core.add_box(FUEL.replace(name="unused"), x=(1000, 2000))
    assert core.materials == before
    with pytest.raises(ValueError, match="range"):
        core.add_box(FUEL, x=(30, 10))
    core.fill(FUEL)
    assert core.materials == [FUEL]
    assert not core.material_map.any()


def test_mesh_assignment_errors_and_pathlike_input(tmp_path):
    core = unstructured()
    with pytest.raises(ValueError, match="single face"):
        core.set_boundary(("vacuum", "reflective", "vacuum"))
    with pytest.raises(ValueError, match="NaN"):
        core.set_boundary(np.nan)
    with pytest.raises(ValueError, match="not both"):
        core.assign(FUEL, tag=2, where=[True, True])
    with pytest.raises(ValueError, match="shape"):
        core.assign(FUEL, where=[True])
    with pytest.raises(ValueError, match="no cells"):
        core.assign(FUEL, tag=999)
    path = tmp_path / "one-quad.msh"
    path.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n4\n"
                    "1 0 0 0\n2 40 0 0\n3 40 40 0\n4 0 40 0\n$EndNodes\n"
                    "$Elements\n1\n1 3 2 1 1 1 2 3 4\n$EndElements\n")
    loaded = MeshModel(path).fill(FUEL).steady(device="cpu")
    assert loaded.converged and loaded.flux.shape == (2, 1)


def test_cartesian_coupling_has_correct_power_and_unperturbed_equilibrium():
    core = Model(size=(80, 60), cells=(8, 6)).fill(FUEL).set_kinetics(KINETICS)
    core.configure_thermal({"fuel": ThermalMaterial(0.25, sink_coeff=0.02,
        sink_temperature=650, heat_capacity=2.5)}, total_power=100.0,
        feedback={"fuel": FeedbackSpec(reference_temperature=700, doppler=0.0)})
    steady = core.coupled_steady(device="cpu")
    assert steady.converged
    assert steady.k_eff == pytest.approx(core.steady(device="cpu", **TOL).k_eff, abs=1e-7)
    assert steady.thermal.source_watts == pytest.approx(100.0, rel=1e-12)
    assert steady.thermal.balance_residual < 1e-9
    transient = core.coupled_transient(0.10, 0.02, dt_thermal=0.10, device="cpu",
                                     initial_coupled_steady=steady)
    np.testing.assert_allclose(transient.power, 1.0, atol=2e-6)
    qs = core.quasistatic_transient(0.10, 0.02, shape_dt=0.10, dt_thermal=0.10, device="cpu")
    np.testing.assert_allclose(qs.power, 1.0, atol=2e-6)
    clone = core.with_materials({"fuel": FUEL.replace()})
    assert clone.coupling_context(device="cpu").total_power == 100.0


def test_cartesian_doppler_feedback_lowers_hot_eigenvalue():
    core = Model(80, 8).fill(FUEL)
    cold = core.steady(device="cpu", **TOL)
    core.configure_thermal({"fuel": ThermalMaterial(0.25, sink_coeff=0.02,
        sink_temperature=650, heat_capacity=2.5)}, total_power=100.0,
        feedback={"fuel": FeedbackSpec(reference_temperature=650, doppler=[2e-4, 0.0])})
    hot = core.coupled_steady(device="cpu")
    assert hot.converged and hot.k_eff < cold.k_eff
    assert np.min(hot.temperature) > 650


@pytest.mark.parametrize("build", [cartesian, triangular])
def test_thermal_name_typos_fail_instead_of_being_ignored(build):
    core = build()
    with pytest.raises(ValueError, match="unknown thermal"):
        core.configure_thermal({"typo": ThermalMaterial(0.2)}, total_power=1.0)
    thermal = {m.name: ThermalMaterial(0.2) for m in core.materials}
    with pytest.raises(ValueError, match="unknown feedback"):
        core.configure_thermal(thermal, total_power=1.0, feedback={"typo": FeedbackSpec(700)})
