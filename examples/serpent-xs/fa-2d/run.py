#!/usr/bin/env python3
"""Reflective pin-homogenized assembly with exact Serpent rectangular bounds."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigs, splu

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ndgpu import kernels
from ndgpu.backend import asnumpy, get_backend, is_rocm
from ndgpu.mesh import assemble_mesh, UnstructuredDiffusionSolver
from ndgpu.serpent_xs import read_serpent_results


def rectangles(path):
    """Read only literal region rectangles from the supplied Serpent input."""
    bounds = {}
    for line in Path(path).read_text().splitlines():
        tokens = line.split('%', 1)[0].split()
        if len(tokens) == 7 and tokens[0] == 'surf' and tokens[2] == 'rect':
            if tokens[1].isdigit() and 1001 <= int(tokens[1]) <= 1289:
                bounds[int(tokens[1])] = tuple(map(float, tokens[3:]))
    if set(bounds) != set(range(1001, 1290)):
        raise ValueError('Geometry must contain exactly regions 1001..1289')
    x = sorted({v for b in bounds.values() for v in b[:2]})
    y = sorted({v for b in bounds.values() for v in b[2:]})
    if len(x) != 18 or len(y) != 18:
        raise ValueError('Geometry is not a conforming 17x17 rectangle grid')
    for r in range(17):
        for c in range(17):
            if bounds[1001 + 17*r + c] != (x[c], x[c+1], y[r], y[r+1]):
                raise ValueError('Region IDs must increase with x, then y')
    return np.array(x), np.array(y)


def make_mesh(x, y, refinement):
    if refinement < 1:
        raise ValueError('Refinement must be positive')
    def split(edges):
        return np.concatenate([np.linspace(a, b, refinement, endpoint=False)
                               for a, b in zip(edges[:-1], edges[1:])] + [edges[-1:]])
    xx, yy = split(x), split(y)
    nx, ny = len(xx)-1, len(yy)-1
    coords = np.array([(a, b) for b in yy for a in xx])
    cells, regions = [], []
    for r in range(ny):
        for c in range(nx):
            n = r*(nx+1)+c
            cells.append((n, n+1, n+nx+2, n+nx+1))
            regions.append((r//refinement)*17+c//refinement)
    region = np.array(regions)
    return assemble_mesh(coords, cells, region), region


def balance_matrix(mesh, mats, region):
    """Independent sparse assembly from rectangular geometry and XS tables."""
    n, g = mesh.n_cells, mats[0].n_groups
    ds = np.array([m.diffusion for m in mats])[region].T
    removal = np.array([m.removal for m in mats])[region].T
    blocks = []
    for group in range(g):
        rows, cols, vals = list(range(n)), list(range(n)), list(removal[group]*mesh.area)
        for i, j, length, distance in mesh.faces:
            # Rectangles share an axis-aligned face. Its coordinate is halfway
            # between the nearest opposite bounds, not necessarily centroids.
            axis = int(np.argmax(abs(mesh.centroid[j]-mesh.centroid[i])))
            half_i = np.ptp(mesh.coords[list(mesh.cells[i]), axis])/2
            half_j = np.ptp(mesh.coords[list(mesh.cells[j]), axis])/2
            w = length/(half_i/ds[group, i]+half_j/ds[group, j])
            rows.extend((i, j, i, j)); cols.extend((i, j, j, i))
            vals.extend((w, w, -w, -w))
        blocks.append(sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc())
    scatter = np.array([m.sigma_s for m in mats])[region]
    loss = sparse.bmat([[blocks[to] if to == source else
                        sparse.diags(-scatter[:, source, to]*mesh.area)
                        for source in range(g)] for to in range(g)], format='csc')
    chi = np.array([m.chi for m in mats])[region]
    nsf = np.array([m.nu_sigma_f for m in mats])[region]
    production = sparse.bmat([[sparse.diags(chi[:, to]*nsf[:, source]*mesh.area)
                              for source in range(g)] for to in range(g)], format='csc')
    return loss, production


def normalized(a):
    a = np.asarray(a, dtype=float)
    if not np.all(np.isfinite(a)) or np.any(a < 0) or a.sum() <= 0:
        raise ValueError('Expected finite nonnegative values with positive sum')
    return a/a.sum()


def region_integrals(flux, area, region):
    return np.array([np.bincount(region, weights=f*area, minlength=289) for f in flux]).T


def edits(integrals, energy_xs):
    return normalized(integrals.sum(axis=0)), normalized((integrals*energy_xs).sum(axis=1))


def compare(k, spectrum, power, ref_k, ref_spectrum, ref_power):
    fuel = ref_power > 0
    relative = power[fuel]/ref_power[fuel]-1
    return {'k_eff': float(k), 'delta_k': float(k-ref_k),
            'delta_k_times_1e5': float((k-ref_k)*1e5),
            'reactivity_difference_pcm': float((1/ref_k-1/k)*1e5),
            'normalized_spectrum': spectrum.tolist(),
            'spectrum_max_absolute_error': float(np.max(abs(spectrum-ref_spectrum))),
            'spectrum_relative_l2_error': float(np.linalg.norm(spectrum-ref_spectrum)/np.linalg.norm(ref_spectrum)),
            'fuel_power_relative_rms': float(np.sqrt(np.mean(relative**2))),
            'fuel_power_relative_max': float(np.max(abs(relative))),
            'fuel_power_total_variation': float(abs(power-ref_power).sum()/2)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parent/'i2_cro_e71_2d_4g'
    p.add_argument('--input', type=Path, default=default)
    p.add_argument('--results', type=Path, default=Path(str(default)+'_res.m'))
    p.add_argument('--output-dir', type=Path, default=Path('results/serpent-fa-2d'))
    p.add_argument('--refinements', nargs='+', type=int, default=[1, 2, 4])
    p.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    p.add_argument('--require-rocm', action='store_true')
    p.add_argument('--diffusion', choices=['inf', 'cmm'], default='inf')
    args = p.parse_args(argv)
    if args.require_rocm and args.device != 'gpu':
        p.error('--require-rocm requires --device gpu')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    x, y = rectangles(args.input)
    records = read_serpent_results(args.results)
    by_id = {r.universe: r for r in records}
    if len(records) != 289 or set(by_id) != {str(i) for i in range(1001,1290)}:
        raise ValueError('Expected one record for each universe 1001..1289')
    records = [by_id[str(i)] for i in range(1001,1290)]
    if any(r.n_groups != 4 or not np.array_equal(r.energy_bounds_mev, records[0].energy_bounds_mev) for r in records):
        raise ValueError('All records must share the same four-group structure')
    if any(r.estimate('TOT_LEAKRATE').mean[0] != 0 for r in records):
        raise ValueError('This case requires zero external leakage')
    refs_k = [r.estimate('IMP_KEFF').mean[0] for r in records]
    if not np.allclose(refs_k, refs_k[0], rtol=0, atol=0):
        raise ValueError('Inconsistent assembly IMP_KEFF across records')
    mats = [r.to_diffusion_material(args.diffusion) for r in records]
    if any(m.kappa_fission is None for m in mats):
        raise ValueError('Power comparison requires INF_FISS and INF_KAPPA')
    energy = np.array([m.kappa_fission for m in mats])
    # Serpent INF_FLX is a region integral with common source normalization.
    # Do not multiply it by area a second time or normalize each region.
    reference_integrals = np.array([r.estimate('INF_FLX').mean for r in records])
    ref_spectrum, ref_power = edits(reference_integrals, energy)
    reconstructed = {
        'TOT_FLUX': reference_integrals.sum(),
        'TOT_GENRATE': (reference_integrals*np.array([m.nu_sigma_f for m in mats])).sum(),
        'TOT_FISSRATE': (reference_integrals*np.array([r.estimate('INF_FISS').mean for r in records])).sum(),
        'TOT_POWER': (reference_integrals*energy).sum()*1.602176634e-13,
    }
    closure = {name: {'reconstructed': float(value), 'global': float(records[0].fields[name][0]),
                      'relative_error': float(value/records[0].fields[name][0]-1)}
               for name, value in reconstructed.items()}
    if any(abs(item['relative_error']) > 1e-4 for item in closure.values()):
        raise ValueError('Regional reference integrals fail global Serpent closure')
    report = {'input_file': args.input.name, 'source_file': args.results.name,
              'geometry_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
              'source_sha256': records[0].source_sha256,
              'software': {'python': sys.version.split()[0], 'numpy': np.__version__, 'scipy': scipy.__version__},
              'geometry': {'x_edges_cm': x.tolist(), 'y_edges_cm': y.tolist(),
                           'height_convention': 'per unit axial depth', 'region_order': 'x increases first, then y'},
              'diffusion': args.diffusion, 'boundary': 'reflective',
              'warnings': ['Local CMM is not a validated regional diffusion coefficient.' if args.diffusion == 'cmm' else
                           'INF_DIFFCOEF is an out-scattering approximation; local transport corrections are not supplied.',
                           'Power reference is reconstructed from group constants and INF_FLX; no independent power detector is supplied.',
                           'No SPH or discontinuity corrections are applied; Serpent differences are reported, not fitted.'],
              'reference': {'IMP_KEFF': float(refs_k[0]),
                            'global_tally_closure': closure,
                            'k_relative_standard_error': float(records[0].estimate('IMP_KEFF').relative_error[0]),
                            'normalized_spectrum': ref_spectrum.tolist(),
                            'flux_integrals': reference_integrals.tolist(),
                            'flux_relative_errors': [r.estimate('INF_FLX').relative_error.tolist() for r in records],
                            'normalized_power': ref_power.tolist()},
              'solver': {'tol_k': 1e-10, 'tol_source': 1e-9, 'max_outer': 2000,
                         'dtype': 'float64', 'backend_k_limit': 1e-8, 'backend_flux_limit': 1e-7,
                         'balance_relative_residual_limit': 1e-8},
              'acceptance_note': 'passed verifies numerical solution and backend agreement; Serpent physics differences are reported without a fitted acceptance threshold.',
              'solves': [], 'passed': True}
    previous = kernels.set_fused(False)
    try:
        for refinement in args.refinements:
            mesh, region = make_mesh(x, y, refinement)
            loss, production = balance_matrix(mesh, mats, region)
            baseline = None
            options = [('cpu', False)]
            if args.device == 'gpu':
                options += [('gpu', False), ('gpu', True)]
            for device, fused in options:
                kernels.set_fused(fused)
                solver = UnstructuredDiffusionSolver(mesh, mats, region, alpha_boundary=0., device=device)
                if device == 'gpu':
                    xp = get_backend('gpu')
                    if solver.xp is not xp or not isinstance(solver.area, xp.ndarray):
                        raise RuntimeError('Actual CuPy device arrays required')
                    if args.require_rocm and not is_rocm(xp):
                        raise RuntimeError('HIP CuPy required')
                    report['software']['cupy'] = xp.__version__
                    report['gpu_device'] = solver.device
                result = solver.solve(tol_k=1e-10, tol_source=1e-9, max_outer=2000)
                flux = asnumpy(result.flux)
                integrals = region_integrals(flux, mesh.area, region)
                spectrum, power = edits(integrals, energy)
                flat = flux.ravel()
                lhs = loss@flat
                residual = float(np.linalg.norm(lhs-production@flat/result.k_eff)/np.linalg.norm(lhs))
                current = compare(result.k_eff, spectrum, power, refs_k[0], ref_spectrum, ref_power)
                current.update(refinement=refinement, cells=mesh.n_cells, device=result.device,
                               fusion_enabled=fused, converged=result.converged,
                               outer_iterations=result.outer_iterations, solve_seconds=result.solve_seconds,
                               balance_relative_residual=residual)
                current['passed'] = bool(result.converged and residual <= 1e-8 and np.all(flux >= 0))
                shape = flat/(flat@np.tile(mesh.area, 4))
                if baseline is None:
                    baseline = (result.k_eff, shape)
                    if refinement == 1:
                        lu = splu(loss)
                        op = LinearOperator(loss.shape, matvec=lambda v: lu.solve(production@v))
                        values, vectors = eigs(op, k=1, which='LM', v0=np.ones(loss.shape[0]), tol=1e-11)
                        independent_k = float(values[0].real)
                        independent_shape = vectors[:, 0].real
                        independent_shape /= independent_shape@np.tile(mesh.area, 4)
                        current['independent_sparse_k'] = independent_k
                        current['independent_k_error'] = abs(result.k_eff-independent_k)
                        current['independent_flux_max_error'] = float(np.max(abs(shape-independent_shape)))
                        current['passed'] &= current['independent_k_error'] <= 1e-8 and current['independent_flux_max_error'] <= 1e-7
                else:
                    current['cpu_k_error'] = abs(result.k_eff-baseline[0])
                    current['cpu_flux_max_error'] = float(np.max(abs(shape-baseline[1])))
                    current['passed'] &= current['cpu_k_error'] <= 1e-8 and current['cpu_flux_max_error'] <= 1e-7
                report['passed'] &= current['passed']
                report['solves'].append(current)
                label = f'r{refinement}_{device}_fusion_{int(fused)}'
                with (args.output_dir/f'{label}.csv').open('w', newline='') as stream:
                    writer = csv.writer(stream, lineterminator='\n')
                    writer.writerow(['universe', 'row_y', 'column_x', 'area_cm2', 'serpent_power', 'ndgpu_power',
                                     *[f'serpent_flux_fraction_g{g+1}' for g in range(4)],
                                     *[f'ndgpu_flux_fraction_g{g+1}' for g in range(4)]])
                    total = integrals.sum()
                    for i in range(289):
                        r, c = divmod(i, 17)
                        writer.writerow([1001+i, r, c, (x[c+1]-x[c])*(y[r+1]-y[r]), ref_power[i], power[i],
                                         *(reference_integrals[i]/reference_integrals.sum()), *(integrals[i]/total)])
                print(f'{label}: k={result.k_eff:.10f}, delta_k*1e5={current["delta_k_times_1e5"]:.2f}, '
                      f'power RMS={current["fuel_power_relative_rms"]:.3%}, max={current["fuel_power_relative_max"]:.3%}, '
                      f'spectrum L2={current["spectrum_relative_l2_error"]:.3%}, residual={residual:.2e}', flush=True)
    except Exception as exc:
        report['passed'] = False
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        kernels.set_fused(previous)
        (args.output_dir/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
