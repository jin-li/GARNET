"""CPU constant-material native-mean references on nonuniform tensor FV grids.

This module is standalone: import its function directly. The uniform auxiliary
source enforces prescribed means only in this local reference, never in a core
operator. No factors, clipping, pseudoinverse or criticality search are used.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.linalg import eigh_tridiagonal

from .materials import Material
from .cuboid_reference import CuboidReferenceError, _array, _scalar

__all__ = ['coupled_neumann_tensor_reference']
_NORMAL_AXES = (0, 1, 0, 1, 2, 2)
_TANGENTS = ((1, 2), (0, 2), (1, 2), (0, 2), (0, 1), (0, 1))


def _axis_basis(widths):
    """Diagonalize W^-1/2 K W^-1/2, K from center-to-center FV resistances."""
    n = len(widths)
    if n == 1:
        return np.zeros(1), np.ones((1, 1)), 0.
    conductance = 2/(widths[:-1]+widths[1:])
    diagonal = np.zeros(n)
    diagonal[:-1] += conductance/widths[:-1]
    diagonal[1:] += conductance/widths[1:]
    off = -conductance/np.sqrt(widths[:-1]*widths[1:])
    values, vectors = eigh_tridiagonal(diagonal, off, lapack_driver='stev')
    scale = float(np.max(abs(diagonal))+2*np.max(abs(off)))
    tolerance = 256*np.finfo(float).eps*n
    if abs(values[0]) > tolerance*scale or np.any(values[1:] <= 0):
        raise CuboidReferenceError('Invalid Neumann axis spectrum', {'stage': 'axis_basis'})
    # Structural null mode is known analytically; this is not eigenvalue flooring.
    constant = np.sqrt(widths/widths.sum())
    if vectors[:, 0]@constant < 0:
        vectors[:, 0] *= -1
    constant_error = float(np.max(abs(vectors[:, 0]-constant)))
    vectors[:, 0] = constant
    values[0] = 0.
    action = diagonal[:, None]*vectors
    action[:-1] += off[:, None]*vectors[1:]
    action[1:] += off[:, None]*vectors[:-1]
    error = max(float(np.max(abs(vectors.T@vectors-np.eye(n)))),
                float(np.max(abs(action-vectors*values[None]))/scale), constant_error)
    if not math.isfinite(error) or error > tolerance:
        raise CuboidReferenceError('Neumann axis basis accuracy failed', {'stage': 'axis_basis', 'basis_error': error})
    return values, vectors, error


def _transform(field, bases, inverse=False):
    for axis, basis in zip((3, 2, 1), bases):
        matrix = basis.T if inverse else basis
        field = np.moveaxis(np.tensordot(field, matrix, axes=(axis, 0)), -1, axis)
    return field


def _geometry(edges, widths):
    nx, ny, nz = map(len, widths)
    z, y, x = np.meshgrid(*(e[:-1]+w/2 for e, w in zip(edges[::-1], widths[::-1])), indexing='ij')
    centers = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    volumes = widths[2][:, None, None]*widths[1][None, :, None]*widths[0][None, None, :]
    grid = np.arange(nx*ny*nz).reshape(nz, ny, nx)
    owners = (grid[:, :, 0], grid[:, 0, :], grid[:, :, -1], grid[:, -1, :], grid[0], grid[-1])
    face_areas = (widths[2][:, None]*widths[1][None], widths[2][:, None]*widths[0][None],
                  widths[1][:, None]*widths[0][None])
    faces = []
    for side, (axis, indices) in enumerate(zip(_NORMAL_AXES, owners)):
        plus = side in (2, 3, 5)
        area = face_areas[axis].copy()
        distance = np.full(area.shape, widths[axis][-1 if plus else 0]/2)
        fc = centers[indices].copy()
        fc[..., axis] = edges[axis][-1 if plus else 0]
        faces.append(dict(side=side, owners=indices.copy(), areas=area, distances=distance, centers=fc))
    return centers, volumes, faces


def _apply(flux, matrix, diffusion, widths, volumes):
    """Independent conservative integrated physical-space FV action."""
    out = (matrix@flux.reshape(len(diffusion), -1)).reshape(flux.shape)*volumes
    for axis, w in zip((3, 2, 1), widths):
        left, right = [slice(None)]*4, [slice(None)]*4
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        shape = [1]*3
        shape[axis-1] = len(w)
        areas = volumes/w.reshape(shape)
        distance_shape = [1]*3
        distance_shape[axis-1] = len(w)-1
        distance = ((w[:-1]+w[1:])/2).reshape(distance_shape)
        coefficient = areas[left[1:]]/distance
        current = diffusion[:, None, None, None]*(flux[left]-flux[right])*coefficient
        out[left] += current
        out[right] -= current
    return out


def coupled_neumann_tensor_reference(material, k, edges, native_mean_flux, *,
                                      inward_current_density=None, inward_current_integral=None,
                                      condition_limit=1e12, rtol=1e-10):
    """Return a diagnostic dict and arrays for a nonuniform tensor FV reference.

    ``edges=(x_edges,y_edges,z_edges)`` are finite, increasing physical positions.
    Exactly one of the two boundary arguments is a six-array sequence in
    W,S,E,N,bottom,top order. Group is LAST: W/E arrays (Nz,Ny,G), S/N
    (Nz,Nx,G), bottom/top (Ny,Nx,G). Values are signed inward-positive
    densities j or integrated subface currents I=A*j, constant on each subface.

    M=diag(removal)-offdiag(sigma_s).T-outer(chi,nu_sigma_f)/k. The auxiliary
    equation is -D*laplace(H)+M*H=delta, delta=M*mu-sum(I)/total_volume.
    The constant mass-weighted mode is prescribed as sqrt(total_volume)*mu;
    all other group modes solve the boundary-driven problem. Full modal
    conditioning, including the uniform block, must pass even though its
    coefficient is prescribed. Returned cell arrays have shape (G,C), x
    fastest then y then z. Boundary dict arrays retain the face-grid ordering.
    Flux traces are H_cell+d*j/D, with positive normal distance d to the face.
    Sign moments integrate sign of each centered POSITIVE GLOBAL tangent
    exactly over the constant-current subfaces, including midpoint crossings.

    Singular-value and condition diagnostics are conservative congruence
    BOUNDS for integrated A, not exact nonuniform singular values. The gated
    backward error is an UPPER bound using a lower bound on ||A||2. A loose
    bound can reject an otherwise solvable mesh. Positivity is discrete only;
    no production accuracy, continuum positivity or factor admission follows.
    """
    if not isinstance(material, Material):
        raise ValueError('material must be one constant Material')
    g = material.n_groups
    diffusion = _array(material.diffusion, (g,), 'material.diffusion')
    removal = _array(material.removal, (g,), 'material.removal')
    absorption = _array(material.sigma_a, (g,), 'material.sigma_a')
    scatter = _array(material.sigma_s, (g, g), 'material.sigma_s')
    production = _array(material.nu_sigma_f, (g,), 'material.nu_sigma_f')
    chi = _array(material.chi, (g,), 'material.chi')
    if g < 1 or np.any(diffusion <= 0) or any(np.any(a < 0) for a in (removal, absorption, scatter, production, chi)):
        raise ValueError('Invalid material coefficients')
    if np.any(production > 0) and not np.isclose(chi.sum(), 1):
        raise ValueError('Fissile material chi must sum to1')
    k = _scalar(k, 'k')
    rtol = _scalar(rtol, 'rtol')
    condition_limit = _scalar(condition_limit, 'condition_limit', lower=1, inclusive=True)
    if rtol >= 1:
        raise ValueError('rtol must be less than1')
    native = _array(native_mean_flux, (g,), 'native_mean_flux')
    if np.any(native <= 0):
        raise ValueError('native_mean_flux must be strictly positive')
    try:
        if len(edges) != 3:
            raise ValueError('Three edge arrays required')
    except TypeError as exc:
        raise ValueError('Three edge arrays required') from exc
    parsed = []
    for value in edges:
        raw = np.asarray(value)
        if raw.ndim != 1 or raw.size < 2 or raw.dtype.kind not in 'iuf' or not np.all(np.isfinite(raw)):
            raise ValueError('Edges must be finite real vectors of length >=2')
        e = np.array(raw, dtype=float, copy=True)
        if not np.all(np.isfinite(np.diff(e))) or np.any(np.diff(e) <= 0):
            raise ValueError('Edges must be strictly increasing with finite intervals')
        parsed.append(e)
    edges = tuple(parsed)
    widths = tuple(np.diff(e) for e in edges)
    if any(not math.isfinite(float(w.sum())) for w in widths):
        raise ValueError('Nonrepresentable axis extent')
    centers, volumes, faces = _geometry(edges, widths)
    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0) or not np.all(np.isfinite(centers)) or any(
            not np.all(np.isfinite(f[key])) or np.any(f[key] <= 0) for f in faces for key in ('areas', 'distances')):
        raise ValueError('Nonrepresentable cell geometry')
    total_volume = float(volumes.sum())
    if not math.isfinite(total_volume) or total_volume <= 0:
        raise ValueError('Nonrepresentable total volume')
    if (inward_current_density is None) == (inward_current_integral is None):
        raise ValueError('Supply exactly one boundary current density or integral sequence')
    boundary = inward_current_density if inward_current_density is not None else inward_current_integral
    try:
        if len(boundary) != 6:
            raise ValueError('Six boundary current arrays required')
    except TypeError as exc:
        raise ValueError('Six boundary current arrays required') from exc
    rhs = np.zeros((g, volumes.size))
    totals = []
    for face, value in zip(faces, boundary):
        current = _array(value, (*face['areas'].shape, g), 'boundary current')
        if inward_current_density is not None:
            density = current
            current = density*face['areas'][..., None]
        else:
            density = current/face['areas'][..., None]
        if not np.all(np.isfinite(current)) or not np.all(np.isfinite(density)):
            raise ValueError('Nonrepresentable boundary current')
        face.update(inward_current_density=density, inward_current_integral=current)
        totals.append(current.sum(axis=(0, 1)))
        for group in range(g):
            np.add.at(rhs[group], face['owners'].ravel(), current[..., group].ravel())
    totals = np.asarray(totals)
    np.fill_diagonal(scatter, 0)
    coupling = scatter.T+np.outer(chi, production)/k
    matrix = np.diag(removal)-coupling
    delta = matrix@native-totals.sum(axis=0)/total_volume
    diagnostics = dict(device='cpu (numpy/scipy)', rtol=rtol, condition_limit=condition_limit,
        native_mean_imposed=True, continuous_positivity_certified=False,
        delta_uniform_auxiliary_source_density=delta, input_integrated_inward_currents=totals,
        conditioning_method='mass-weighted tensor modal block singular values and volume congruence bounds')
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(delta)) or not np.all(np.isfinite(rhs)):
        raise CuboidReferenceError('Nonfinite coupled material/source/RHS', diagnostics)
    try:
        decomposition = [_axis_basis(w) for w in widths]
        eigenvalues = [value[0] for value in decomposition]
        bases = [value[1] for value in decomposition]
        diagnostics['axis_basis_max_errors'] = [value[2] for value in decomposition]
        nx, ny, nz = map(len, widths)
        minimum, maximum = math.inf, 0.
        for start in range(0, volumes.size, 256):
            indices = np.arange(start, min(start+256, volumes.size))
            lam = eigenvalues[0][indices % nx]+eigenvalues[1][indices//nx % ny]+eigenvalues[2][indices//(nx*ny)]
            singular = np.linalg.svd(matrix[None]+lam[:, None, None]*np.diag(diffusion)[None], compute_uv=False)
            minimum = min(minimum, float(singular[:, -1].min()))
            maximum = max(maximum, float(singular[:, 0].max()))
        lower = float(volumes.min())*minimum
        upper = float(volumes.max())*maximum
        norm_lower = float(volumes.min())*maximum
        condition = upper/lower if lower > 0 else math.inf
        diagnostics.update(modal_sigma_min=minimum, modal_sigma_max=maximum, operator_sigma_min_lower_bound=lower,
            operator_sigma_max_upper_bound=upper, operator_sigma_max_lower_bound=norm_lower,
            condition_number_2_upper_bound=condition, spatial_modes_checked=int(volumes.size))
        if not all(math.isfinite(v) and v > 0 for v in (lower, upper, norm_lower)) or condition > condition_limit:
            raise CuboidReferenceError('Singular or excessively conditioned integrated operator bound', diagnostics)
        shape = (nz, ny, nx)
        modal = _transform(rhs.reshape((g, *shape))/np.sqrt(volumes), bases).reshape(g, -1)
        modal[:, 0] = math.sqrt(total_volume)*native
        for start in range(1, volumes.size, 256):
            indices = np.arange(start, min(start+256, volumes.size))
            lam = eigenvalues[0][indices % nx]+eigenvalues[1][indices//nx % ny]+eigenvalues[2][indices//(nx*ny)]
            blocks = matrix[None]+lam[:, None, None]*np.diag(diffusion)[None]
            modal[:, start:start+len(indices)] = np.linalg.solve(blocks, modal[:, start:start+len(indices)].T[..., None])[..., 0].T
        tensor = _transform(modal.reshape((g, *shape)), bases, inverse=True)/np.sqrt(volumes)
    except CuboidReferenceError as exc:
        raise CuboidReferenceError(str(exc), {**diagnostics, **exc.diagnostics}) from exc
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise CuboidReferenceError('Nonuniform tensor eigensystem/modal solve failed', diagnostics) from exc
    flux = tensor.reshape(g, -1)
    if not np.all(np.isfinite(flux)):
        raise CuboidReferenceError('Nonfinite nonuniform reference flux', diagnostics)
    physical = _apply(tensor, matrix, diffusion, widths, volumes).reshape(g, -1)-rhs
    auxiliary = physical-delta[:, None]*volumes.ravel()
    auxiliary_rhs = rhs+delta[:, None]*volumes.ravel()
    backward_scale_lower = norm_lower*np.linalg.norm(flux)+np.linalg.norm(auxiliary_rhs)
    backward = float(np.linalg.norm(auxiliary)/backward_scale_lower) if backward_scale_lower > 0 else math.inf
    measured = flux@volumes.ravel()/total_volume
    physical_balance = matrix@measured*total_volume-totals.sum(axis=0)
    auxiliary_balance = physical_balance-delta*total_volume
    scale = (abs(removal*measured)+abs(coupling@measured)+abs(delta))*total_volume+abs(totals).sum(axis=0)
    balance_relative = np.divide(abs(auxiliary_balance), scale, out=np.zeros(g), where=scale > 0)
    mean_relative = measured/native-1
    diagnostics.update(physical_algebra_residual_l2=float(np.linalg.norm(physical)), auxiliary_algebra_residual_l2=float(np.linalg.norm(auxiliary)),
        physical_algebra_residual_max_absolute=float(np.max(abs(physical))), auxiliary_algebra_residual_max_absolute=float(np.max(abs(auxiliary))),
        normwise_backward_error_upper_bound=backward, physical_integrated_group_balance_residual=physical_balance,
        auxiliary_integrated_group_balance_residual=auxiliary_balance, auxiliary_group_balance_max_relative=float(balance_relative.max()),
        measured_volume_mean_flux=measured, native_mean_relative_difference=mean_relative,
        minimum_cell_mean_flux_per_group=flux.min(axis=1))
    if not math.isfinite(backward) or backward > rtol or not np.all(np.isfinite(balance_relative)) or balance_relative.max() > rtol or np.max(abs(mean_relative)) > rtol:
        raise CuboidReferenceError('Nonuniform auxiliary backward error/balance/native mean gate failed', diagnostics)
    face_means, recovered, moments, recovered_moments = [], [], [], []
    current_passed = True
    for face in faces:
        owner = flux[:, face['owners']].transpose(1, 2, 0)
        density, area, distance = face['inward_current_density'], face['areas'], face['distances']
        trace = owner+distance[..., None]*density/diffusion
        trace_current = (trace-owner)*diffusion/distance[..., None]*area[..., None]
        roundoff = 8*np.finfo(float).eps*(abs(trace)+abs(owner))*diffusion/distance[..., None]*area[..., None]
        crossing_scale = abs(face['inward_current_integral']).sum(axis=(0, 1))
        error = trace_current.sum(axis=(0, 1))-face['inward_current_integral'].sum(axis=(0, 1))
        budget = rtol*crossing_scale+roundoff.sum(axis=(0, 1))
        current_passed &= bool(np.all(np.isfinite(trace)) and np.all(np.isfinite(trace_current)) and np.all(abs(error) <= budget))
        prescribed_m, trace_m = [], []
        for axis in _TANGENTS[face['side']]:
            mid = edges[axis][0]+(edges[axis][-1]-edges[axis][0])/2
            average_sign = (abs(edges[axis][1:]-mid)-abs(edges[axis][:-1]-mid))/widths[axis]
            # Face grid is descending global axis order: (z,y), (z,x), (y,x).
            tangent_order = sorted(_TANGENTS[face['side']], reverse=True)
            reshaped = average_sign.reshape((-1, 1) if tangent_order.index(axis) == 0 else (1, -1))
            prescribed_m.append((face['inward_current_integral']*reshaped[..., None]).sum(axis=(0, 1)))
            trace_m.append((trace_current*reshaped[..., None]).sum(axis=(0, 1)))
            moment_error = trace_m[-1]-prescribed_m[-1]
            current_passed &= bool(np.all(abs(moment_error) <= budget))
        face.update(flux=trace, recovered_inward_current_integral=trace_current, current_recovery_roundoff_allowance=roundoff)
        face_means.append((trace*area[..., None]).sum(axis=(0, 1))/area.sum())
        recovered.append(trace_current.sum(axis=(0, 1)))
        moments.append(prescribed_m); recovered_moments.append(trace_m)
    diagnostics.update(boundary_current_and_sign_moment_recovery_passed=current_passed,
        recovered_integrated_inward_currents=np.asarray(recovered), input_integrated_sign_moments=np.asarray(moments),
        recovered_integrated_sign_moments=np.asarray(recovered_moments),
        minimum_boundary_trace_flux_per_group=np.min(np.concatenate([f['flux'].reshape(-1, g) for f in faces]), axis=0),
        current_recovery_qualification='Trace-derived totals/moments checked against rtol*sum(abs(input I)) plus explicit floating-point trace subtraction allowance')
    if not current_passed:
        raise CuboidReferenceError('Nonuniform boundary current/sign-moment recovery failed', diagnostics)
    if np.any(flux <= 0) or any(np.any(f['flux'] <= 0) for f in faces):
        raise CuboidReferenceError('Nonpositive cell mean or FV boundary trace; no clipping', diagnostics)
    return dict(cell_mean_flux=flux, cell_centers=centers, cell_volumes=volumes.ravel(), volume_mean_flux=measured,
        physical_integrated_residual=physical, auxiliary_integrated_residual=auxiliary,
        face_mean_flux=np.asarray(face_means), boundary_profiles=faces, edges=edges, subdivisions=(nx, ny, nz),
        diagnostics=diagnostics, discrete_positive=True, continuous_positivity_certified=False)
