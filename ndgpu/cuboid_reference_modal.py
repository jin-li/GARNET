"""Uniform-tensor DCT backend for the existing fixed-k cuboid reference PDE.

The spatial transform is orthonormal DCT-II. Every group block is conditioned
and solved, including nonconstant modes and modes with a zero transformed RHS.
No mesh objects, sparse factorization, source offsets or factors are created.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.fft import dctn, idctn

from .cuboid_reference import (CoupledCuboidReference, CuboidBoundaryProfiles,
                               CuboidReferenceError, _array, _counts,
                               _modal_conditioning, _scalar)
from .materials import Material

__all__ = ['coupled_neumann_cuboid_reference_modal']


def _laplacian(field, steps):
    """Independent conservative cell-centered Neumann graph stencil, -laplace."""
    result = np.zeros_like(field)
    for axis, step in zip((-1, -2, -3), steps):
        left, right = [slice(None)]*4, [slice(None)]*4
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        difference = (field[left]-field[right])/step**2
        result[left] += difference
        result[right] -= difference
    return result


def _tensor_geometry(counts, steps, widths, cell_volume):
    """Vectorized factory-compatible cell/boundary order, without a Mesh."""
    nx, ny, nz = counts
    indices = np.arange(nx*ny*nz, dtype=np.int64)
    centers = np.empty((indices.size, 3))
    centers[:, 0] = (indices % nx+0.5)*steps[0]
    centers[:, 1] = (indices//nx % ny+0.5)*steps[1]
    centers[:, 2] = (indices//(nx*ny)+0.5)*steps[2]
    grid = indices.reshape(nz, ny, nx)
    selections = (grid[:, :, 0], grid[:, :, -1], grid[:, 0, :], grid[:, -1, :], grid[0], grid[-1])
    side_order = (0, 2, 1, 3, 4, 5)  # factory: axis x/y/z, minus then plus
    owners = np.concatenate([selection.ravel() for selection in selections])
    sides = np.concatenate([np.full(selection.size, side, dtype=np.int64)
                            for selection, side in zip(selections, side_order)])
    axes = np.array([0, 1, 0, 1, 2, 2])[sides]
    area, distance = cell_volume/steps[axes], steps[axes]/2
    face_centers = centers[owners].copy()
    plus = np.isin(sides, [2, 3, 5])
    face_centers[np.arange(len(owners)), axes] = np.where(plus, widths[axes], 0)
    return centers, owners, sides, area, distance, face_centers


def coupled_neumann_cuboid_reference_modal(material, k, net_inward_current, widths, *, subdivisions=8,
                                           native_mean_flux=None, condition_limit=1e12, rtol=1e-10):
    """Solve the same fixed-k Neumann problem as coupled_neumann_cuboid_reference.

    Arguments, return dataclasses and failure semantics match the sparse-LU
    implementation. Native means remain diagnostic only. All transforms use
    float64 CPU arrays, orthonormal DCT-II and workers=1. Spatial modes are
    coupled only within their GxG blocks M+lambda_p*diag(D); none are omitted.
    Boundary profiles follow the rectilinear factory's owner/side ordering.

    Full all-mode structural2-norm conditioning is checked before solving.
    Backward error uses an independently evaluated physical-space FV stencil;
    integrated group balance and strict cell/boundary positivity are separate
    gates. There is no clipping, source adjustment, eigenvalue search, fitted
    factor or continuous positivity certificate.
    """
    if not isinstance(material, Material):
        raise ValueError('material must be one constant Material')
    groups = material.n_groups
    if groups < 1:
        raise ValueError('material must have at least one group')
    diffusion = _array(material.diffusion, (groups,), 'material.diffusion')
    absorption = _array(material.sigma_a, (groups,), 'material.sigma_a')
    removal = _array(material.removal, (groups,), 'material.removal')
    scatter = _array(material.sigma_s, (groups, groups), 'material.sigma_s')
    production = _array(material.nu_sigma_f, (groups,), 'material.nu_sigma_f')
    chi = _array(material.chi, (groups,), 'material.chi')
    if np.any(diffusion <= 0) or any(np.any(a < 0) for a in (absorption, removal, scatter, production, chi)):
        raise ValueError('diffusion must be positive and material coefficients nonnegative')
    if np.any(production > 0) and not np.isclose(chi.sum(), 1):
        raise ValueError('fissile material chi must sum to1')
    k = _scalar(k, 'k')
    condition_limit = _scalar(condition_limit, 'condition_limit', lower=1, inclusive=True)
    rtol = _scalar(rtol, 'rtol')
    if rtol >= 1:
        raise ValueError('rtol must be less than1')
    counts = _counts(subdivisions)
    widths = _array(widths, (3,), 'widths')
    currents = _array(net_inward_current, (6, groups), 'net_inward_current')
    if np.any(widths <= 0):
        raise ValueError('widths must be strictly positive')
    native = None if native_mean_flux is None else _array(native_mean_flux, (groups,), 'native_mean_flux')
    if native is not None and np.any(native <= 0):
        raise ValueError('native_mean_flux must be strictly positive when supplied')
    steps = widths/np.asarray(counts)
    cell_volume, native_volume = float(np.prod(steps)), float(np.prod(widths))
    if not np.all(np.isfinite(steps)) or np.any(steps <= 0) or not math.isfinite(cell_volume) or cell_volume <= 0 or not math.isfinite(native_volume):
        raise ValueError('widths/subdivisions must define finite positive representable volumes and steps')
    np.fill_diagonal(scatter, 0)
    coupling = scatter.T+np.outer(chi, production)/k
    matrix = np.diag(removal)-coupling
    if not np.all(np.isfinite(matrix)):
        raise ValueError('material/k produces nonfinite coupled coefficients')
    diagnostics = _modal_conditioning(matrix, diffusion, counts, steps, cell_volume)
    diagnostics.update(condition_limit=condition_limit, rtol=rtol, dtype='float64', device='cpu (numpy/scipy)',
        continuous_positivity_certified=False, linear_solver='orthonormal_DCT_II_batched_group_solve',
        transform_workers=1, mesh_object_constructed=False)
    if not math.isfinite(diagnostics['condition_number_2']) or diagnostics['condition_number_2'] > condition_limit:
        raise CuboidReferenceError('Full coupled Neumann operator is singular or exceeds condition_limit', diagnostics)
    if not all(math.isfinite(diagnostics[key]) and diagnostics[key] > 0 for key in
               ('minimum_singular_value', 'maximum_singular_value')):
        raise CuboidReferenceError('Full operator singular values are outside representable range', diagnostics)
    # Geometry is canonical h*index, with no nonuniform/general-mesh path.
    geometry_error = max(float(np.max(abs(np.diff(np.arange(n+1)*h)/h-1)))
                         for n, h in zip(counts, steps))
    diagnostics['uniform_tensor_geometry_relative_roundoff'] = geometry_error
    if geometry_error > 128*np.finfo(float).eps*max(counts):
        raise CuboidReferenceError('Geometry is not the assumed uniform tensor', diagnostics)
    centers, owners, sides, areas, distances, boundary_centers = _tensor_geometry(counts, steps, widths, cell_volume)
    ncell, shape = centers.shape[0], tuple(reversed(counts))
    native_areas = np.array([widths[1]*widths[2], widths[0]*widths[2], widths[1]*widths[2],
                            widths[0]*widths[2], widths[0]*widths[1], widths[0]*widths[1]])
    inward_density = currents[sides]/native_areas[sides, None]
    rhs = np.zeros((groups, ncell))
    for group in range(groups):
        np.add.at(rhs[group], owners, inward_density[:, group]*areas)
    rhs_density = rhs.reshape((groups, *shape))/cell_volume
    if not np.all(np.isfinite(rhs_density)):
        raise CuboidReferenceError('Nonfinite discrete boundary RHS', diagnostics)
    # Transform b/V, solve every modal group block, then invert the transform.
    modal = dctn(rhs_density, type=2, axes=(1, 2, 3), norm='ortho', workers=1).reshape(groups, ncell)
    lambdas = [4*np.sin(np.pi*np.arange(n)/(2*n))**2/h**2 for n, h in zip(counts, steps)]
    nx, ny, nz = counts
    diagonal_diffusion = np.diag(diffusion)
    try:
        for start in range(0, ncell, 256):
            indices = np.arange(start, min(start+256, ncell))
            px, py, pz = indices % nx, indices//nx % ny, indices//(nx*ny)
            lam = lambdas[0][px]+lambdas[1][py]+lambdas[2][pz]
            blocks = matrix[None]+lam[:, None, None]*diagonal_diffusion[None]
            modal[:, start:start+indices.size] = np.linalg.solve(
                blocks, modal[:, start:start+indices.size].T[:, :, None])[:, :, 0].T
    except np.linalg.LinAlgError as exc:
        raise CuboidReferenceError('Direct modal group solve failed', diagnostics) from exc
    flux_tensor = idctn(modal.reshape((groups, *shape)), type=2, axes=(1, 2, 3), norm='ortho', workers=1)
    flux = flux_tensor.reshape(groups, ncell)
    if not np.all(np.isfinite(flux)):
        raise CuboidReferenceError('DCT solve returned nonfinite flux', diagnostics)
    # Independent physical-space stencil, rather than a transformed residual.
    applied = (diffusion[:, None]*_laplacian(flux_tensor, steps).reshape(groups, ncell)+matrix@flux)*cell_volume
    algebra_residual = applied-rhs
    backward_scale = diagnostics['maximum_singular_value']*float(np.linalg.norm(flux))+float(np.linalg.norm(rhs))
    backward_error = float(np.linalg.norm(algebra_residual)/backward_scale) if backward_scale > 0 else 0.0
    volume_mean = flux.mean(axis=1)
    integrated_residual = matrix@volume_mean*native_volume-currents.sum(axis=0)
    integrated_scale = (np.abs(removal*volume_mean)*native_volume+
                        np.abs(coupling@volume_mean)*native_volume+np.abs(currents).sum(axis=0))
    integrated_relative = np.divide(integrated_residual, integrated_scale, out=np.zeros(groups), where=integrated_scale > 0)
    diagnostics.update(normwise_backward_error=backward_error,
        algebra_residual_l2=float(np.linalg.norm(algebra_residual)),
        integrated_group_balance_residual=integrated_residual,
        integrated_group_balance_relative=integrated_relative,
        integrated_group_balance_max_relative=float(np.max(abs(integrated_relative))),
        residual_evaluation='independent conservative physical-space Neumann stencil plus coupled material matrix')
    if not math.isfinite(backward_error) or backward_error > rtol or np.max(abs(integrated_relative)) > rtol:
        raise CuboidReferenceError('Modal backward error or independent integrated group balance exceeds rtol', diagnostics)
    boundary_flux = flux[:, owners].T+distances[:, None]*inward_density/diffusion[None]
    diagnostics.update(minimum_cell_mean_flux=float(flux.min()), minimum_boundary_trace_flux=float(boundary_flux.min()),
        minimum_cell_mean_flux_per_group=flux.min(axis=1), minimum_boundary_trace_flux_per_group=boundary_flux.min(axis=0),
        nonpositive_cell_group_bins=int(np.count_nonzero(flux <= 0)),
        nonpositive_boundary_group_bins=int(np.count_nonzero(boundary_flux <= 0)))
    if not np.all(np.isfinite(boundary_flux)) or np.any(flux <= 0) or np.any(boundary_flux <= 0):
        raise CuboidReferenceError('Nonpositive/nonfinite cell mean or FV boundary trace; no clipping applied', diagnostics)
    native_diagnostics = None
    if native is not None:
        native_defect = matrix@native-currents.sum(axis=0)/native_volume
        native_scale = np.linalg.norm(matrix, 2)*np.linalg.norm(native)+np.linalg.norm(currents.sum(axis=0)/native_volume)
        native_diagnostics = dict(native_mean_flux=native, compatibility_defect_density=native_defect,
            normwise_compatibility_backward_error=float(np.linalg.norm(native_defect)/native_scale) if native_scale > 0 else 0.0,
            computed_minus_native_mean_flux=volume_mean-native, computed_relative_difference=volume_mean/native-1,
            computed_vs_native_relative_l2=float(np.linalg.norm(volume_mean-native)/np.linalg.norm(native)),
            native_mean_imposed=False)
    face_means = np.array([(boundary_flux[sides == side]*areas[sides == side, None]).sum(axis=0)/
                           areas[sides == side].sum() for side in range(6)])
    profiles = CuboidBoundaryProfiles(owners, sides, areas, distances, boundary_centers, inward_density, boundary_flux)
    return CoupledCuboidReference(flux, face_means, profiles, volume_mean, centers, np.full(ncell, cell_volume),
                                  counts, diagnostics, native_diagnostics)
