"""CPU fixed-k coupled Neumann references on uniform tensor cuboid meshes.

A constant material and prescribed uniform inward currents define a discrete
multigroup boundary-value problem. Native mean flux is diagnostic only. This
module constructs no discontinuity factors and performs no criticality search.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .materials import Material
from .mesh import MeshGrid, MeshGroupOperator
from .nodal_mesh import build_rectilinear_mesh_3d

__all__ = ["CuboidBoundaryProfiles", "CoupledCuboidReference", "CuboidReferenceError",
           "coupled_neumann_cuboid_reference"]


class CuboidReferenceError(RuntimeError):
    """Numerical conditioning, balance or discrete-positivity gate failure."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


@dataclass(frozen=True)
class CuboidBoundaryProfiles:
    """Subface profiles, ordered as the rectilinear factory's boundary faces.

    Side order is W,S,E,N,bottom,top. Flux/current arrays have shape (B,G).
    ``flux`` is the FV extrapolation H_cell+d*j_in/D, not a continuous trace
    certificate. ``areas`` and ``distances`` use the canonical uniform tensor.
    """

    owners: np.ndarray
    sides: np.ndarray
    areas: np.ndarray
    distances: np.ndarray
    centers: np.ndarray
    inward_current_density: np.ndarray
    flux: np.ndarray


@dataclass(frozen=True)
class CoupledCuboidReference:
    """Accepted discrete reference; no continuous positivity claim is made.

    ``cell_mean_flux`` has shape (G,C), ``face_mean_flux`` shape (6,G), and
    ``volume_mean_flux`` shape (G,). Cell order is z,y,x with x changing first.
    All calculations and returned arrays use float64 on the CPU.
    """

    cell_mean_flux: np.ndarray
    face_mean_flux: np.ndarray
    boundary_profiles: CuboidBoundaryProfiles
    volume_mean_flux: np.ndarray
    cell_centers: np.ndarray
    cell_volumes: np.ndarray
    subdivisions: tuple[int, int, int]
    diagnostics: dict
    native_mean_compatibility: dict | None
    discrete_positive: bool = True
    continuous_positivity_certified: bool = False


def _scalar(value, name, *, lower=0.0, inclusive=False):
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in 'iuf' or not np.isfinite(raw):
        raise ValueError(f'{name} must be a finite real scalar')
    result = float(raw)
    if result < lower or (not inclusive and result == lower):
        raise ValueError(f'{name} must be '+('at least ' if inclusive else 'greater than ')+str(lower))
    return result


def _array(value, shape, name):
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in 'iuf' or not np.all(np.isfinite(raw)):
        raise ValueError(f'{name} must be finite real data with shape {shape}')
    return np.array(raw, dtype=float, copy=True)


def _counts(value):
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        values = (value,)*3
    else:
        try:
            values = tuple(value)
        except TypeError as exc:
            raise ValueError('subdivisions must be a positive integer or three positive integers') from exc
    if len(values) != 3 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 1
                               for v in values):
        raise ValueError('subdivisions must be a positive integer or three positive integers')
    return tuple(int(v) for v in values)


def _modal_conditioning(matrix, diffusion, counts, steps, volume):
    """Exact structural2-norm condition via every orthogonal cosine mode.

    For uniform cell volumes, the spatial Neumann graph Laplacian is orthogonally
    diagonalized by the tensor DCT-II basis. In group-major order a permutation
    then gives blocks volume*(M+lambda_p*diag(D)). No symmetry of M is assumed:
    singular values, rather than eigenvalue magnitudes, determine conditioning.
    """
    nx, ny, nz = counts
    lambdas = [4*np.sin(np.pi*np.arange(n)/(2*n))**2/h**2 for n, h in zip(counts, steps)]
    smallest, largest, minimum_mode = math.inf, 0.0, None
    diagonal_diffusion = np.diag(diffusion)
    try:
        for start in range(0, nx*ny*nz, 256):
            indices = np.arange(start, min(start+256, nx*ny*nz))
            px, py, pz = indices % nx, indices//nx % ny, indices//(nx*ny)
            lam = lambdas[0][px]+lambdas[1][py]+lambdas[2][pz]
            blocks = matrix[None, :, :]+lam[:, None, None]*diagonal_diffusion[None, :, :]
            values = np.linalg.svd(blocks, compute_uv=False)
            at = int(np.argmin(values[:, -1]))
            if values[at, -1] < smallest:
                smallest = float(values[at, -1])
                minimum_mode = (int(px[at]), int(py[at]), int(pz[at]))
            largest = max(largest, float(np.max(values[:, 0])))
        uniform_singular = np.linalg.svd(matrix, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise CuboidReferenceError('Modal singular-value decomposition failed', {'stage': 'conditioning'}) from exc
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        condition = float(np.divide(largest, smallest)) if smallest > 0 else math.inf
        uniform_condition = float(np.divide(uniform_singular[0], uniform_singular[-1])) if uniform_singular[-1] > 0 else math.inf
    return dict(condition_number_2=condition, minimum_singular_value=volume*smallest,
                maximum_singular_value=volume*largest, minimum_singular_spatial_mode=minimum_mode,
                uniform_mode_condition_number_2=uniform_condition,
                spatial_modes_checked=nx*ny*nz, conditioning_method='all uniform-tensor cosine-mode block singular values')


def coupled_neumann_cuboid_reference(material, k, net_inward_current, widths, *, subdivisions=8,
                                     native_mean_flux=None, condition_limit=1e12, rtol=1e-10):
    """Solve one constant-material cuboid at fixed positive ``k`` on the CPU.

    ``net_inward_current`` is surface-integrated (6,G), inward-positive, in
    W,S,E,N,bottom,top order; it is spread uniformly over each native surface.
    ``widths`` is (Lx,Ly,Lz). ``subdivisions`` is N or (Nx,Ny,Nz).

    The source-free coupled equation is -D*Laplacian(H)+M*H=0, with
    M=diag(removal)-offdiagonal(sigma_s).T-outer(chi,nu_sigma_f)/k. The positive
    boundary RHS is j_in*subface_area. No source offsets, source iteration,
    normalization, eigenvalue search, clipping or fitted factors are used.

    All spatial cosine modes enter the full discrete operator2-norm condition
    gate, including nonconstant resonances. ``rtol`` bounds the direct solve's
    normwise backward error and separately checked integrated group balances.
    Positive cell means and positive FV boundary extrapolations are required;
    these checks establish discrete positivity only. A supplied positive native
    mean is compared to the computed mean and never imposed on the solve.

    Invalid inputs raise ValueError. Numerical/conditioning/positivity failures
    raise CuboidReferenceError with a ``diagnostics`` dictionary.
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
    cell_volume = float(np.prod(steps))
    native_volume = float(np.prod(widths))
    if not np.all(np.isfinite(steps)) or np.any(steps <= 0) or not math.isfinite(cell_volume) or cell_volume <= 0 or not math.isfinite(native_volume):
        raise ValueError('widths/subdivisions must define finite positive representable volumes and steps')
    np.fill_diagonal(scatter, 0)
    coupling = scatter.T+np.outer(chi, production)/k
    matrix = np.diag(removal)-coupling
    if not np.all(np.isfinite(matrix)):
        raise ValueError('material/k produces nonfinite coupled coefficients')
    diagnostics = _modal_conditioning(matrix, diffusion, counts, steps, cell_volume)
    diagnostics.update(condition_limit=condition_limit, rtol=rtol, dtype='float64', device='cpu (numpy/scipy)',
                       continuous_positivity_certified=False)
    if not math.isfinite(diagnostics['condition_number_2']) or diagnostics['condition_number_2'] > condition_limit:
        raise CuboidReferenceError('Full coupled Neumann operator is singular or exceeds condition_limit', diagnostics)
    if not all(math.isfinite(diagnostics[key]) and diagnostics[key] > 0 for key in
               ('minimum_singular_value', 'maximum_singular_value')):
        raise CuboidReferenceError('Full operator singular values are outside representable range', diagnostics)
    edges = [np.linspace(0, width, n+1) for width, n in zip(widths, counts)]
    shape = tuple(reversed(counts))
    mesh, topology = build_rectilinear_mesh_3d(*edges, active_mask=np.ones(shape, dtype=bool),
                                               tags=np.zeros(shape, dtype=int))
    ncell = mesh.n_cells
    # Canonical h and volume define the uniform tensor matrix exactly. The
    # factory's geometry is checked to differ only by floating-point roundoff.
    geometry_error = max(float(np.max(abs(topology['widths']/steps-1))),
                         float(np.max(abs(mesh.area/cell_volume-1))))
    diagnostics['uniform_tensor_geometry_relative_roundoff'] = geometry_error
    if geometry_error > 128*np.finfo(float).eps*max(counts):
        raise CuboidReferenceError('Factory geometry is not the assumed uniform tensor', diagnostics)
    axes_by_side = np.array([0, 1, 0, 1, 2, 2])
    owners = topology['face_owners']
    axes = axes_by_side[topology['face_sides'][:, 0]]
    conductance = cell_volume/steps[axes]**2
    i, j = owners[:, 0], owners[:, 1]
    laplacian = sparse.coo_matrix((np.concatenate((conductance, conductance, -conductance, -conductance)),
                    (np.concatenate((i, j, i, j)), np.concatenate((i, j, j, i)))), shape=(ncell, ncell)).tocsc()
    operator = sparse.kron(sparse.diags(diffusion), laplacian, format='csc') + cell_volume*sparse.kron(
        sparse.csc_matrix(matrix), sparse.eye(ncell, format='csc'), format='csc')
    boundary_owner, sides = topology['boundary_owners'], topology['boundary_sides']
    boundary_axes = axes_by_side[sides]
    boundary_area = cell_volume/steps[boundary_axes]
    boundary_distance = steps[boundary_axes]/2
    native_areas = np.array([widths[1]*widths[2], widths[0]*widths[2], widths[1]*widths[2],
                             widths[0]*widths[2], widths[0]*widths[1], widths[0]*widths[1]])
    inward_density = currents[sides]/native_areas[sides, None]
    rhs = np.zeros((groups, ncell))
    for group in range(groups):
        np.add.at(rhs[group], boundary_owner, inward_density[:, group]*boundary_area)
    if not np.all(np.isfinite(operator.data)) or not np.all(np.isfinite(rhs)):
        raise CuboidReferenceError('Nonfinite discrete operator or boundary RHS', diagnostics)
    # Independent action: current reusable mesh operators plus coupled sources.
    grid = MeshGrid(mesh, 'reflective')
    probe = 1.5+np.sin(np.arange(groups*ncell).reshape(groups, ncell)*math.sqrt(2))
    independent = np.array([MeshGroupOperator(np, grid, np.full(grid.shape, diffusion[g]),
                              np.full(grid.shape, removal[g])).apply(probe[g].reshape(grid.shape)).ravel()
                            for g in range(groups)])-mesh.area*(coupling@probe)
    assembled = (operator@probe.ravel()).reshape(groups, ncell)
    action_scale = float(np.linalg.norm(independent)+np.linalg.norm(assembled))
    action_error = float(np.linalg.norm(independent-assembled)/action_scale) if action_scale > 0 else 0.0
    diagnostics['independent_mesh_operator_action_relative_error'] = action_error
    if not math.isfinite(action_error) or action_error > 4096*np.finfo(float).eps:
        raise CuboidReferenceError('Sparse action disagrees with existing coupled MeshGroupOperator action', diagnostics)
    try:
        flux = splu(operator).solve(rhs.ravel()).reshape(groups, ncell)
    except (RuntimeError, ValueError) as exc:
        raise CuboidReferenceError('Sparse direct LU failed', diagnostics) from exc
    if not np.all(np.isfinite(flux)):
        raise CuboidReferenceError('Sparse LU returned nonfinite flux', diagnostics)
    algebra_residual = (operator@flux.ravel()-rhs.ravel()).reshape(groups, ncell)
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
        integrated_group_balance_max_relative=float(np.max(abs(integrated_relative))))
    if not math.isfinite(backward_error) or backward_error > rtol or np.max(abs(integrated_relative)) > rtol:
        raise CuboidReferenceError('Direct LU backward error or independent integrated group balance exceeds rtol', diagnostics)
    boundary_flux = flux[:, boundary_owner].T+boundary_distance[:, None]*inward_density/diffusion[None, :]
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
    face_means = np.array([(boundary_flux[sides == side]*boundary_area[sides == side, None]).sum(axis=0)/
                           boundary_area[sides == side].sum() for side in range(6)])
    boundary_centers = mesh.centroid[boundary_owner]+topology['side_normals'][sides]*boundary_distance[:, None]
    profiles = CuboidBoundaryProfiles(boundary_owner.copy(), sides.copy(), boundary_area, boundary_distance,
                                      boundary_centers, inward_density, boundary_flux)
    return CoupledCuboidReference(flux, face_means, profiles, volume_mean, mesh.centroid.copy(),
                                  np.full(ncell, cell_volume), counts, diagnostics, native_diagnostics)
