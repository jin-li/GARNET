"""Physical acceptance checks for static CPU branch NEM eigen solutions."""
import numpy as np


def _array(value, name, shape):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'biuf' or raw.shape != shape:
        raise ValueError(f'{name} must be a real array of shape {shape}')
    array = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must be finite')
    return array


def _relative(residual, scale):
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(scale)):
        return float('inf')
    return float(np.max(abs(residual) / np.maximum(scale, np.finfo(float).tiny), initial=0.))


def assess_nem_solution(solver, result, *, balance_tolerance=1e-5):
    """Return scalar diagnostics or raise ValueError on a nonphysical solution.

    Independently rebuild mean volume, shared corrected-face, and zero-drive
    Robin balances from result arrays and the solver's full emitted/source F.
    Recompute the solver's degree-specific weak/current/profile diagnostics,
    including F on every spatial moment; stored result diagnostics are ignored.
    Profile stationary-point minima are numerical, not interval certificates.
    Positive total volume-integrated fission production is required; arbitrary
    positive flux normalization is accepted. This checks a supplied eigenpair,
    not transport accuracy or nuclear-data/ADF qualification.
    """
    solver._assert_unchanged()
    if (isinstance(balance_tolerance, (bool, np.bool_))
            or not np.isscalar(balance_tolerance)
            or not isinstance(balance_tolerance, (int, float, np.number))
            or np.iscomplexobj(balance_tolerance)
            or not np.isfinite(balance_tolerance) or balance_tolerance <= 0):
        raise ValueError('balance_tolerance must be finite positive')
    k = result.k_eff
    if (isinstance(k, (bool, np.bool_)) or not np.isscalar(k)
            or not isinstance(k, (int, float, np.number)) or np.iscomplexobj(k)
            or not np.isfinite(k) or k <= 0):
        raise ValueError('k_eff must be finite positive')
    grid = solver.grid
    N, G, M, F = solver.n_nodes, solver.n_groups, solver.n_moments, solver.n_faces
    means = _array(result.means, 'means', (N, G))
    moments = _array(result.moments, 'moments', (N, 3, M, G))
    currents = _array(result.currents, 'currents', (F, G))
    faces = _array(result.side_face_means, 'side_face_means', (N, 6, G))
    corrected = _array(result.corrected_face_means, 'corrected_face_means', (F, G))
    if any(np.any(array < 0) for array in (means, faces, corrected)):
        raise ValueError('negative physical mean or face flux')
    q0 = np.einsum('ngh,nh->ng', solver.fission, means) / k
    qm = np.einsum('ngh,naph->napg', solver.fission, moments) / k
    production = float(np.sum(grid.volumes[:, None] * q0) * k)
    if not np.isfinite(production) or production <= 0:
        raise ValueError('fission production normalization must be finite positive')
    pairs = ((0, 2), (1, 3), (4, 5))
    divergence = np.stack([
        (currents[grid.node_faces[:, sp]] - currents[grid.node_faces[:, sm]])
        / grid.widths[:, axis, None] for axis, (sm, sp) in enumerate(pairs)], axis=1)
    reaction = np.einsum('ngh,nh->ng', solver.R, means)
    volume_residual = reaction + divergence.sum(axis=1) - q0
    volume_scale = (np.einsum('ngh,nh->ng', abs(solver.R), abs(means))
                    + abs(divergence).sum(axis=1) + abs(q0))
    interface_residuals, interface_scales, stored_residuals, stored_scales = [], [], [], []
    for face, (low, high) in enumerate(grid.face_nodes):
        sm, sp = pairs[grid.face_axes[face]]
        values = [solver.node_face_factors[node, side] * faces[node, side]
                  for node, side in ((low, sp), (high, sm)) if node >= 0]
        for value in values:
            stored_residuals.append(corrected[face] - value)
            stored_scales.append(abs(corrected[face]) + abs(value))
        if len(values) == 2:
            interface_residuals.append(values[0] - values[1])
            interface_scales.append(abs(values[0]) + abs(values[1]))
    outward = grid.normal_signs[grid.boundary_sides, None] * currents[grid.boundary_faces]
    robin = grid.boundary_alpha[:, None] * faces[grid.boundary_nodes, grid.boundary_sides]
    boundary_residual = outward - robin
    boundary_scale = (abs(outward) + abs(robin)
                      + solver.D[grid.boundary_nodes] * abs(means[grid.boundary_nodes])
                      / grid.widths[grid.boundary_nodes, grid.face_axes[grid.boundary_faces], None])
    fresh = solver._assess(result, q0, qm, np.zeros_like(outward), profiles=True)
    minima = _array(fresh['profile_minima'], 'profile_minima', (N, 3, G))
    if np.any(minima < 0):
        raise ValueError('negative reconstructed profile flux')
    report = {
        'status': 'pass', 'balance_tolerance': float(balance_tolerance),
        'fission_production': production,
        'minimum_mean': float(means.min()), 'minimum_side_face_mean': float(faces.min()),
        'minimum_corrected_face_mean': float(corrected.min()),
        'minimum_profile_flux': float(minima.min()), 'extrema_are_interval_certified': False,
        'volume_balance_error': _relative(volume_residual, volume_scale),
        'interface_balance_error': _relative(np.asarray(interface_residuals), np.asarray(interface_scales)),
        'corrected_face_balance_error': _relative(np.asarray(stored_residuals), np.asarray(stored_scales)),
        'boundary_balance_error': _relative(boundary_residual, boundary_scale),
        'maximum_absolute_volume_residual': float(np.max(abs(volume_residual))),
        'maximum_absolute_boundary_residual': float(np.max(abs(boundary_residual), initial=0.)),
        'maximum_absolute_interface_residual': float(np.max(abs(np.asarray(interface_residuals)), initial=0.)),
    }
    for name in ('weak_balance_error', 'current_balance_error', 'mean_balance_error', 'moment_balance_error'):
        report[name] = float(fresh[name])
    errors = {name: value for name, value in report.items() if name.endswith('_balance_error')}
    report['maximum_balance_error'] = max(errors.values())
    failed = [name for name, value in errors.items()
              if not np.isfinite(value) or value > balance_tolerance]
    if failed:
        raise ValueError('NEM physical balance exceeds tolerance: ' + ', '.join(failed))
    return report
