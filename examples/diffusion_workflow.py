"""Steady diffusion, named material edits, kinetics and thermal feedback.

Install GARNET, then run:
    python examples/diffusion_workflow.py --device cpu
    python examples/diffusion_workflow.py --device gpu

The two-dimensional core represents a 1 cm thick slice; 100 W below is the
power of that slice. Cross sections are demonstration data.
"""

import argparse

import numpy as np

from garnet import (FeedbackSpec, Kinetics, Material, Model, PWR_TWO_GROUP,
                    ThermalMaterial)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    args = parser.parse_args()

    fuel = PWR_TWO_GROUP.replace(name="fuel")
    reflector = Material(
        name="reflector", diffusion=[1.2, 0.3], sigma_a=[0.001, 0.02],
        nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.03], [0.0, 0.0]])
    core = (Model(size=(80, 80), cells=(12, 12)).fill(reflector)
            .add_box(fuel, x=(20, 60), y=(20, 60))
            .set_boundary("vacuum")
            .set_kinetics(Kinetics([1e7, 3e5], [0.0065], [0.08])))

    steady = core.steady(device=args.device, solver_kwargs={"precond_degree": 1})
    print(steady)
    q = steady.power_density(total_power=100.0)
    print(f"Integrated slice power: {np.sum(q * steady.cell_volumes):.6f} W")
    for row in steady.material_summary:
        print(f"{row.name}: {100 * row.volume_fraction:.1f}% of volume, "
              f"{100 * row.fission_fraction:.1f}% of fission production")

    # Preserve the scattering, spectrum and optional energy-release data.
    perturbed_fuel = fuel.replace(sigma_a=fuel.sigma_a * [1.0, 1.003])
    perturbed = core.with_materials({"fuel": perturbed_fuel})
    print(f"Perturbed k_eff: {perturbed.steady(device=args.device).k_eff:.6f}")
    transient = core.transient(
        t_end=0.1, dt=0.02, device=args.device, initial_steady=steady,
        materials_at=lambda t: {} if t == 0 else {"fuel": perturbed_fuel})
    print(transient)

    core.configure_thermal(
        {"fuel": ThermalMaterial(0.25, sink_coeff=0.02, sink_temperature=650,
                                 heat_capacity=2.5),
         "reflector": ThermalMaterial(0.35, sink_coeff=0.01, sink_temperature=650,
                                      heat_capacity=2.0)},
        total_power=100.0,
        feedback={"fuel": FeedbackSpec(reference_temperature=650,
                                       doppler=[2e-4, 0.0])})
    hot = core.coupled_steady(device=args.device)
    print(hot)
    coupled = core.coupled_transient(
        t_end=0.1, dt=0.02, dt_thermal=0.1, device=args.device,
        initial_coupled_steady=hot)
    print(f"Unperturbed hot-core final power: {coupled.power[-1]:.6f} P/P0")


if __name__ == "__main__":
    main()
