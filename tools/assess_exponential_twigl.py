"""Resolved TWIGL error/work gate for the experimental exponential schemes.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m tools.assess_exponential_twigl

FV trajectories use the public Model API. The reference independently
assembles a physical neutron/precursor generator and integrates it with
sparse Radau, with tolerance refinement and separate spatial comparisons.
No matrix-free stencil or time-scheme coefficients enter that reference.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy
from scipy.integrate import solve_ivp, trapezoid
from scipy.interpolate import interp1d
from scipy.sparse import bmat, coo_matrix, diags, eye

from garnet import Model
from garnet.benchmarks import build_twigl
from garnet.benchmarks.twigl import K_REFERENCE, P_REFERENCE


ROOT = Path(__file__).resolve().parents[1]
METHODS = ('ftm2-history', 'ftm2-doubling', 'ftm-bdf2-history',
           'ftm-bdf2-doubling', 'adaptive-be', 'adaptive-bdf')


def loss_matrix(diffusion, absorption, outscatter, shape, spacing):
    """Independent Cartesian FV loss: harmonic faces, reflecting low faces,
    zero-flux high x/y faces, and reflective axial faces (TWIGL geometry).
    """
    diffusion = np.asarray(diffusion).reshape(shape)
    diagonal = np.asarray(absorption + outscatter).ravel().copy()
    indices = np.arange(diffusion.size).reshape(shape)
    rows, columns, values = [], [], []
    for axis, width in enumerate(spacing):
        if shape[axis] < 2:
            continue
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        a, b = diffusion[tuple(lo)], diffusion[tuple(hi)]
        coupling = (2 * a * b / (a + b) / width**2).ravel()
        left, right = indices[tuple(lo)].ravel(), indices[tuple(hi)].ravel()
        np.add.at(diagonal, left, coupling)
        np.add.at(diagonal, right, coupling)
        rows.extend((left, right))
        columns.extend((right, left))
        values.extend((-coupling, -coupling))
    for axis in (0, 1):
        face = [slice(None)] * 3
        face[axis] = -1
        cells = indices[tuple(face)].ravel()
        diagonal[cells] += 2 * diffusion[tuple(face)].ravel() / spacing[axis]**2
    rows.append(indices.ravel())
    columns.append(indices.ravel())
    values.append(diagonal)
    return coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                      shape=(diffusion.size, diffusion.size)).tocsc()


@dataclass
class TwiglSystem:
    problem: object
    model: object
    steady: object
    matrix: object
    change: object
    initial: np.ndarray
    fission: np.ndarray
    losses: list
    materials: list

    @classmethod
    def build(cls, kind, refinement):
        problem = build_twigl(kind, refinement)
        materials, mmap = problem.problem_at(0.)
        model = Model.from_grid(problem.grid, materials, mmap, bc=problem.bc,
                                kinetics=problem.kinetics)
        steady = model.steady(device='cpu', tol_k=1e-12, tol_source=1e-11)
        if not steady.converged:
            raise RuntimeError('TWIGL initial eigenpair did not converge')
        count = mmap.size
        fission = np.array([m.nu_sigma_f for m in materials])[mmap.ravel()].T / steady.k_eff
        source = fission * steady.flux_numpy.reshape(2, count)
        flux = steady.flux_numpy.reshape(2, count) / source.sum()
        beta, decay = float(problem.kinetics.beta[0]), float(problem.kinetics.decay[0])
        initial = np.r_[flux.ravel(), beta / decay * np.sum(fission * flux, axis=0)]
        speeds = problem.kinetics.velocities
        diffusion = np.array([m.diffusion for m in materials])[mmap.ravel()].T
        absorption = np.array([m.sigma_a for m in materials])[mmap.ravel()].T
        scatter = np.array([m.sigma_s for m in materials])[mmap.ravel()]
        spectrum = np.array([m.chi for m in materials])[mmap.ravel()].T
        losses = [loss_matrix(diffusion[g], absorption[g], scatter[:, g, :].sum(axis=1),
                              problem.grid.shape, problem.grid.spacing) for g in range(2)]
        blocks = [[None] * 3 for _ in range(3)]
        for group in range(2):
            for other in range(2):
                block = diags(speeds[group] * ((1 - beta) * spectrum[group] * fission[other]
                              + (scatter[:, other, group] if group != other else 0)))
                blocks[group][other] = block - speeds[group] * losses[group] if group == other else block
            blocks[group][2] = diags(speeds[group] * spectrum[group] * decay)
            blocks[2][group] = diags(beta * fission[group])
        blocks[2][2] = -decay * eye(count, format='csc')
        matrix = bmat(blocks, format='csc')
        # Only absorption in the central seed's thermal group changes.
        change = np.zeros(3 * count)
        change[count:2*count] = speeds[1] * absorption[1] * (mmap.ravel() == 2)
        return cls(problem, model, steady, matrix, diags(change, format='csc'),
                   initial, fission, losses, materials)

    @property
    def count(self):
        return self.problem.material_map.size

    def generator(self, t, kind):
        # Right-hand derivative for a step at zero; the time-zero eigenpair
        # and the public FV material callback still use unperturbed fields.
        reduction = 1 - 0.976667 if kind == 'step' else 1 - round(1 - 0.11667 * min(t, 0.2), 12)
        return self.matrix + reduction * self.change

    def power(self, state):
        state = np.asarray(state)
        flux = state[..., :2*self.count].reshape(*state.shape[:-1], 2, self.count)
        return np.sum(flux * self.fission, axis=(-2, -1))


@dataclass
class Reference:
    system: TwiglSystem
    segments: list
    duration: float
    rtol: float

    @classmethod
    def solve(cls, system, kind, *, duration=0.5, rtol=3e-10):
        initial, segments = system.initial, []
        scale = np.sqrt(np.mean(initial.reshape(3, -1)**2, axis=1))
        atol = np.repeat(rtol * 1e-3 * scale, system.count)
        bounds = [0., 0.2, duration] if kind == 'ramp' and duration > 0.2 else [0., duration]
        for start, end in zip(bounds[:-1], bounds[1:]):
            frozen = kind == 'step' or start >= 0.2
            matrix = system.generator(start, kind) if frozen else None
            rhs = ((lambda t, y: matrix @ y) if frozen else
                   (lambda t, y: system.generator(t, kind) @ y))
            jac = matrix if frozen else lambda t, y: system.generator(t, kind)
            solution = solve_ivp(rhs, (start, end), initial, method='Radau', jac=jac,
                                 rtol=rtol, atol=atol, dense_output=True)
            if not solution.success:
                raise RuntimeError(solution.message)
            segments.append(solution)
            initial = solution.y[:, -1]
        return cls(system, segments, duration, rtol)

    def __call__(self, times):
        times = np.asarray(times, dtype=float)
        flat = times.ravel()
        epsilon = 1e-13 * max(1., self.duration)
        if np.any(flat < -epsilon) or np.any(flat > self.duration + epsilon):
            raise ValueError('reference sample outside integration span')
        flat = np.clip(flat, 0., self.duration)
        result = np.empty((flat.size, self.system.initial.size))
        for index, solution in enumerate(self.segments):
            selected = ((flat >= solution.t[0]) & (flat <= solution.t[-1])
                        if index == 0 else (flat > solution.t[0]) & (flat <= solution.t[-1]))
            if np.any(selected):
                result[selected] = solution.sol(flat[selected]).T
        return result.reshape(*times.shape, self.system.initial.size)

    def integral(self):
        # Radau's dense interpolant is cubic; three-point Gauss integration
        # integrates its power exactly within each internal reference step.
        nodes, weights = np.polynomial.legendre.leggauss(3)
        total = 0.
        for solution in self.segments:
            widths = np.diff(solution.t)
            times = solution.t[:-1, None] + 0.5 * widths[:, None] * (1 + nodes)
            powers = self.system.power(solution.sol(times.ravel()).T).reshape(times.shape)
            total += np.sum(0.5 * widths * (powers @ weights))
        return float(total)


def sample_times(duration=0.5):
    return np.unique(np.r_[np.linspace(0, duration, 2001),
                           np.geomspace(1e-9, min(0.02, duration), 401),
                           [0.2] if duration >= 0.2 else []])


def relative_blocks(actual, expected):
    """Per-group/family volume RMS relative to the reference field norm.
    TWIGL cells have equal volume, so the volume factor cancels exactly.
    """
    return np.sqrt(np.sum((actual - expected)**2, axis=-1)
                   / np.sum(expected**2, axis=-1))


def restrict_field(field, coarse_n):
    """Conservative area averaging on nested aligned Cartesian meshes."""
    n = field.shape[-2]
    if field.shape[-1] != n or n % coarse_n:
        raise ValueError('spatial comparison needs nested square meshes')
    ratio = n // coarse_n
    return field.reshape(*field.shape[:-2], coarse_n, ratio, coarse_n, ratio).mean(axis=(-3, -1))


def measure(system, reference, times, powers, snapshots, final_precursors):
    common = sample_times(reference.duration)
    samples = np.unique(np.r_[common, times])
    expected = reference(samples)
    expected_flux = expected[:, :2*system.count].reshape(-1, 2, system.count)
    expected_power = system.power(expected)
    actual_flux = interp1d(times, snapshots, axis=0, assume_sorted=True)(samples)
    actual_power = np.interp(samples, times, powers)
    flux_errors = relative_blocks(actual_flux, expected_flux)
    shape_errors = relative_blocks(actual_flux / actual_power[:, None, None],
                                    expected_flux / expected_power[:, None, None])
    expected_final = reference(reference.duration)[2*system.count:].reshape(1, system.count)
    precursor_errors = relative_blocks(final_precursors.reshape(1, system.count), expected_final)
    regional_errors = []
    for region in range(3):
        cells = system.problem.material_map.ravel() == region
        actual_regional = np.sum(actual_flux[:, :, cells] * system.fission[:, cells], axis=(1, 2))
        expected_regional = np.sum(expected_flux[:, :, cells] * system.fission[:, cells], axis=(1, 2))
        regional_errors.append(float(np.max(np.abs(actual_regional / expected_regional - 1))))
    return {
        'maximum_relative_power_error': float(np.max(np.abs(actual_power / expected_power - 1))),
        'maximum_endpoint_relative_power_error': float(np.max(np.abs(
            powers / system.power(reference(times)) - 1))),
        'maximum_relative_flux_rms_error_by_group': np.max(flux_errors, axis=0).tolist(),
        'maximum_relative_shape_rms_error_by_group': np.max(shape_errors, axis=0).tolist(),
        'maximum_relative_regional_power_error': regional_errors,
        'relative_power_integral_error': float(abs(trapezoid(powers, times) / reference.integral() - 1)),
        'final_relative_precursor_rms_error_by_family': precursor_errors.tolist(),
        'final_relative_precursor_inventory_error_by_family': np.abs(
            final_precursors.reshape(1, system.count).sum(axis=1) / expected_final.sum(axis=1) - 1).tolist(),
        'minimum_recorded_flux': float(np.min(snapshots)),
        'minimum_final_precursor': float(np.min(final_precursors)),
    }


def error_gate(record, budget):
    fields = ('maximum_relative_power_error', 'maximum_relative_flux_rms_error_by_group',
              'maximum_relative_shape_rms_error_by_group', 'maximum_relative_regional_power_error',
              'relative_power_integral_error', 'final_relative_precursor_rms_error_by_family',
              'final_relative_precursor_inventory_error_by_family')
    return (all(np.max(record[key]) <= budget for key in fields)
            and record['minimum_recorded_flux'] >= 0 and record['minimum_final_precursor'] >= 0)


def run(system, reference, kind, method, tolerance, *, estimator=None, widths=None, frequency=None):
    snapshots = [system.initial[:2*system.count].reshape(2, system.count)]
    controls = dict(rtol=tolerance, min_dt=1e-11, max_dt=0.02, max_rejections=40)
    if method in ('adaptive-bdf', 'adaptive-be'):
        options = dict(time_scheme='bdf5' if method == 'adaptive-bdf' else 'bdf1',
                       adaptive_bdf=dict(controls, automatic_order=method == 'adaptive-bdf'))
    else:
        scheme = 'ftm-bdf2' if method.startswith('ftm-bdf2') else 'ftm2'
        options = dict(time_scheme=scheme)
        if widths is None:
            controls.update(output_rtol=1e-4,
                            estimator=estimator or ('history' if method.endswith('history') else 'step-doubling'))
            options['adaptive_ftm'] = controls
        if frequency is not None:
            options['ftm_kwargs'] = dict(frequency=frequency)
    result = system.model.transient(reference.duration, 0.005 if widths is None else widths,
        at=system.problem.problem_at, initial_steady=system.steady, device='cpu',
        event_times=[0.2] if kind == 'ramp' and reference.duration > 0.2 else None,
        on_step=lambda t, phi, power: snapshots.append(np.asarray(phi).reshape(2, system.count).copy()),
        step_solver='monolithic', tol_step=1e-10, multigroup_kwargs=dict(rtol=1e-11), **options)
    row = measure(system, reference, result.times, result.power, np.asarray(snapshots), result.precursors_numpy)
    row.update(configuration=dict(method=method, rtol=tolerance, estimator=estimator,
                                  frequency=frequency, fixed_widths=widths is not None),
        accepted_steps=result.n_steps, rejected_steps=result.rejected_steps,
        implicit_solves=sum(result.feedback_iterations) + sum(result.rejected_feedback_iterations),
        krylov_iterations=result.total_inner_iterations,
        fgmres_iterations=sum(result.step_iterations) + sum(result.rejected_step_iterations),
        operator_rebuilds=result.operator_rebuilds,
        diagnostic_solve_seconds=result.solve_seconds,
        minimum_accepted_width_s=float(min(result.step_widths)),
        maximum_accepted_width_s=float(max(result.step_widths)),
        final_power=float(result.power[-1]), power_at_repository_times={
            str(t): float(result.power_at(t)) for t in (0.1, 0.5) if t <= reference.duration})
    attempts = result.ftm_trial_diagnostics + result.rejected_ftm_trial_diagnostics
    row['frequency_refinements'] = sum(trial['frequency_refinements'] for trials in attempts for trial in trials)
    row['fallback_trial_steps'] = sum(bool(trial['fallback_reason']) for trials in attempts for trial in trials)
    if estimator == 'history-audit':
        pairs = [(max(d['history_error_blocks']), max(blocks)) for d, blocks in zip(
            result.ftm_estimator_diagnostics + result.rejected_ftm_estimator_diagnostics,
            result.local_error_blocks + result.rejected_error_blocks) if d['history_error_blocks'] is not None]
        row['audit'] = dict(paired_trials=len(pairs),
            oracle_failed_state_checks=sum(oracle > 1 for history, oracle in pairs),
            history_pass_oracle_fail=sum(history <= 1 < oracle for history, oracle in pairs),
            oracle_pass_history_fail=sum(oracle <= 1 < history for history, oracle in pairs))
    print(kind, system.problem.grid.shape[0], method, tolerance,
          f"power={100*row['maximum_relative_power_error']:.5f}% shape={100*max(row['maximum_relative_shape_rms_error_by_group']):.5f}% solves={row['implicit_solves']}",flush=True)
    return row


def graded_widths(duration=0.5, maximum=0.005, event=None):
    time, width, endpoints = 0., 1e-7, []
    while time < duration:
        limit = min(duration, event) if event is not None and time < event else duration
        end = min(time + width, limit)
        endpoints.append(end)
        time = end
        width = 1e-7 if event is not None and time == event else min(width * 1.15, maximum)
    return np.diff(np.r_[0., endpoints])


def spatial_comparison(coarse, fine):
    """Compare physical density shapes after conservative restriction.

    Engine normalization sums cell fission sources to one. Multiplying
    fields by cell count puts both meshes in the same density normalization.
    """
    a, b = coarse.system, fine.system
    times = np.unique(np.r_[np.linspace(0, 0.5, 201), np.geomspace(1e-8, 0.02, 101)])
    ca, fb = coarse(times), fine(times)
    cp, fp = a.power(ca), b.power(fb)
    n, nf = a.problem.grid.shape[0], b.problem.grid.shape[0]
    cs = ca[:, :2*a.count].reshape(-1, 2, n, n) * a.count / cp[:, None, None, None]
    fs = fb[:, :2*b.count].reshape(-1, 2, nf, nf) * b.count / fp[:, None, None, None]
    restricted = restrict_field(fs, n)
    precursor = ca[-1, 2*a.count:].reshape(n,n) * a.count
    fine_precursor = restrict_field(fb[-1, 2*b.count:].reshape(nf,nf) * b.count, n)
    return dict(coarse_shape=[n,n], fine_shape=[nf,nf],
        maximum_relative_power_difference=float(np.max(np.abs(cp / fp - 1))),
        maximum_relative_shape_rms_difference_by_group=np.max(relative_blocks(
            cs.reshape(-1,2,a.count), restricted.reshape(-1,2,a.count)),axis=0).tolist(),
        final_relative_precursor_rms_difference=float(relative_blocks(
            precursor.ravel(),fine_precursor.ravel())),
        final_relative_precursor_inventory_difference=float(abs(
            ca[-1,2*a.count:].sum()/fb[-1,2*b.count:].sum()-1)),
        relative_power_integral_difference=abs(coarse.integral()/fine.integral()-1),
        k0_difference=float(a.steady.k_eff-b.steady.k_eff))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refinements', nargs='+', type=int, default=[2, 4])
    parser.add_argument('--kinds', nargs='+', choices=['step', 'ramp'], default=['step', 'ramp'])
    parser.add_argument('--rtols', nargs='+', type=float, default=[1e-3, 1e-4, 1e-5])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--no-controls', action='store_true')
    parser.add_argument('--spatial-refinement', type=int, default=8)
    parser.add_argument('--skip-spatial', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmark-results/exponential-second-order/chunk7-twigl.json')
    args = parser.parse_args()
    if (any(r < 1 for r in args.refinements) or args.spatial_refinement < 1
            or any(not np.isfinite(t) or t <= 0 for t in args.rtols)):
        parser.error('refinements and tolerances must be positive')
    if not args.skip_spatial and any(args.spatial_refinement % r for r in args.refinements):
        parser.error('spatial refinement must be a multiple of each comparison refinement')
    sources = ['garnet/transient.py', 'garnet/timescheme.py', 'garnet/benchmarks/twigl.py',
               'garnet/stencil.py', 'tools/assess_exponential_twigl.py']
    report = dict(date='2026-10-26', scope='TWIGL spatial neutron/precursor temporal gate; no thermal transform or timing promotion',
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        reference='Independent harmonic-face Cartesian physical generator, sparse Radau with tolerance refinement',
        metrics='Maximum on common graded/linear grid and own endpoints: power, raw group flux RMS, power-normalized shape RMS and three region powers; trapezoid power integral; final precursor RMS and inventory',
        precursor_scope='Final precursor field and inventory only; no claim about unsaved precursor trajectories',
        work='All constituent/rejected/refined solves, Krylov work and rebuilds; single-run seconds diagnostic only',
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}, cases={})
    references = {}
    for refinement in args.refinements:
        for kind in args.kinds:
            print('reference',kind,refinement,flush=True)
            system = TwiglSystem.build(kind, refinement)
            start = perf_counter()
            coarse_reference = Reference.solve(system, kind, rtol=3e-9)
            reference = Reference.solve(system, kind, rtol=3e-11)
            references[kind,refinement] = reference
            samples = sample_times()
            fine, coarse = reference(samples), coarse_reference(samples)
            refinement_errors = dict(
                maximum_relative_power_error=float(np.max(np.abs(system.power(coarse) / system.power(fine) - 1))),
                maximum_relative_block_rms_error_by_field=np.max(relative_blocks(
                    coarse.reshape(-1, 3, system.count), fine.reshape(-1, 3, system.count)), axis=0).tolist(),
                relative_integral_error=abs(coarse_reference.integral() / reference.integral() - 1))
            if max(refinement_errors['maximum_relative_block_rms_error_by_field']) > 1e-7:
                raise RuntimeError(f'reference refinement unresolved: {refinement_errors}')
            case = dict(refinement=refinement, shape=list(system.problem.grid.shape), cells=system.count,
                k0=system.steady.k_eff, repository_k=K_REFERENCE,
                repository_power=P_REFERENCE[kind],
                reference_power={str(t):float(system.power(reference(t))) for t in (0.1,0.5)},
                reference_integral=reference.integral(),
                reference_refinement=refinement_errors, reference_seconds=perf_counter()-start,
                initial_generator_derivative_norm_ratio_per_s=float(
                    np.linalg.norm(system.matrix @ system.initial) / np.linalg.norm(system.initial)),
                initial_generator_scaled_residual=float(np.linalg.norm(system.matrix @ system.initial)
                    / np.linalg.norm(abs(system.matrix) @ abs(system.initial))),
                records=[], controls=[])
            for method in args.methods:
                for tolerance in args.rtols:
                    case['records'].append(run(system, reference, kind, method, tolerance))
            if not args.no_controls:
                widths = graded_widths(event=0.2 if kind == 'ramp' else None)
                for method in ('ftm2', 'ftm-bdf2'):
                    for frequency in (None, 0.):
                        case['controls'].append(run(system, reference, kind, method, None,
                                                    widths=widths, frequency=frequency))
                    case['controls'].append(run(system, reference, kind, method, 1e-4, estimator='history-audit'))
            case['fewest_tested_solves_at_budget'] = {str(budget):{
                method:min((row for row in case['records'] if row['configuration']['method']==method and error_gate(row,budget)),
                           key=lambda row:row['implicit_solves'],default=None)
                for method in args.methods} for budget in (0.01,0.001)}
            report['cases'][f'{kind}-{refinement}'] = case
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not args.skip_spatial:
        report['spatial_comparisons'] = {}
        for kind in args.kinds:
            print('spatial reference',kind,args.spatial_refinement,flush=True)
            system = TwiglSystem.build(kind,args.spatial_refinement)
            coarse = Reference.solve(system,kind,rtol=3e-9)
            fine = Reference.solve(system,kind,rtol=3e-10)
            times = sample_times()
            defects = np.max(relative_blocks(coarse(times).reshape(-1,3,system.count),
                fine(times).reshape(-1,3,system.count)),axis=0).tolist()
            if max(defects)>1e-7:
                raise RuntimeError(f'spatial reference temporal refinement unresolved: {defects}')
            report['spatial_comparisons'][kind] = dict(
                fine_refinement=args.spatial_refinement,k0=system.steady.k_eff,
                reference_power={str(t):float(system.power(fine(t))) for t in (0.1,0.5)},
                temporal_refinement_rms_by_field=defects,
                comparisons=[spatial_comparison(references[kind,r],fine) for r in args.refinements])
            args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
