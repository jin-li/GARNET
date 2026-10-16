"""Independent CPU tensor-FV reflector diagnostic with prescribed uniform currents.

The native mean fixes the constant mode using an explicit auxiliary delta. This
is neither a physical core source nor a continuum positivity qualification.
There are no leakage moments, neighbor interpolation, or fitted coefficients.
"""
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.fft import dctn, idctn


class ReflectorCuboidReferenceError(ValueError):
    """Invalid, singular, ill-conditioned, or unconverged reference problem."""


class NonpositiveReflectorCuboidError(ReflectorCuboidReferenceError):
    """Complete unmodified nonpositive field is retained in ``result``."""

    def __init__(self, result):
        super().__init__('nonpositive cuboid reference field; diagnostic only')
        self.result = result
        self.diagnostics = result.diagnostics


def _readonly(a):
    a = np.asarray(a, dtype=float)
    return np.frombuffer(a.tobytes(), dtype=float).reshape(a.shape)


@dataclass(frozen=True)
class ReflectorCuboidReferenceResult:
    """Face tuples follow W/S/E/N/bottom/top; patches retain tensor order.

    Flux/traces/currents have group first. Patch areas have only tangent axes.
    Currents are integrated inward currents per patch. All arrays are immutable.
    """
    cell_flux: np.ndarray
    face_means: np.ndarray
    face_traces: tuple
    face_patch_areas: tuple
    face_patch_currents: tuple
    delta: np.ndarray
    mean: np.ndarray
    diagnostics: object


def _array(value, name):
    if np.iscomplexobj(value):
        raise ReflectorCuboidReferenceError(f'{name} must be real')
    try:
        a = np.asarray(value, dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ReflectorCuboidReferenceError(f'{name} must be numeric') from exc
    if not np.all(np.isfinite(a)):
        raise ReflectorCuboidReferenceError(f'{name} must be finite')
    return a


def _relative(error, scale):
    if not np.all(np.isfinite(error)) or not np.all(np.isfinite(scale)):
        raise ReflectorCuboidReferenceError('nonfinite diagnostic')
    ratio = np.divide(abs(error), scale, out=np.zeros_like(error), where=scale > 0)
    if np.any((scale == 0) & (error != 0)):
        raise ReflectorCuboidReferenceError('nonzero residual at zero scale')
    return float(np.max(ratio))


def solve_reflector_cuboid_reference(
        diffusion, loss, widths, native_mean, net_inward_integrated, *,
        subdivisions=8, condition_limit=1e12, rtol=1e-10, require_positive=True):
    """Solve -diag(D) laplace(phi)+R phi=delta with fixed native mean.

    D is (G,), R is a nonsingular nonfission M-matrix (G,G): a Z-matrix
    with positive-stable eigenvalues; column diagonal dominance is not required.
    Widths are
    (Lx,Ly,Lz), and currents are (6,G), positive inward, W/S/E/N/bottom/top.
    Every face has the explicitly prescribed UNIFORM density I/face_area.
    delta=R@native_mean-sum(I)/volume is an auxiliary constant source.
    Subdivisions are a positive integer or (nx,ny,nz). The graph-Neumann
    tensor operator uses orthonormal DCT-II, one thread, float64 CPU arrays.
    All modal blocks, including the fixed uniform block R, enter the full
    operator singular-value/condition gate. No clipping or normalization occurs.
    ``require_positive=False`` returns negative fields for diagnosis only.
    """
    D, R = _array(diffusion, 'diffusion'), _array(loss, 'loss')
    L, mean = _array(widths, 'widths'), _array(native_mean, 'native_mean')
    currents = _array(net_inward_integrated, 'net_inward_integrated')
    G = D.size
    if (D.ndim != 1 or G == 0 or R.shape != (G, G) or L.shape != (3,)
            or mean.shape != (G,) or currents.shape != (6, G)):
        raise ReflectorCuboidReferenceError('expected shapes G,GxG,3,G,6xG')
    if np.any(D <= 0) or np.any(L <= 0) or np.any(mean <= 0):
        raise ReflectorCuboidReferenceError('diffusion, widths and native_mean must be positive')
    offdiag = R.copy()
    np.fill_diagonal(offdiag, 0)
    if np.any(offdiag > 0) or np.any(np.diag(R) <= 0):
        raise ReflectorCuboidReferenceError('loss must be a nonfission M-matrix')
    if not isinstance(require_positive, (bool, np.bool_)):
        raise ReflectorCuboidReferenceError('require_positive must be boolean')
    condition_limit, rtol = _array(condition_limit, 'condition_limit'), _array(rtol, 'rtol')
    if condition_limit.ndim or rtol.ndim or condition_limit < 1 or not 0 < rtol < 1:
        raise ReflectorCuboidReferenceError('condition_limit >=1 and 0<rtol<1 required')
    condition_limit, rtol = float(condition_limit), float(rtol)
    counts = (subdivisions,)*3 if isinstance(subdivisions, (int, np.integer)) else subdivisions
    try:
        counts = tuple(counts)
    except TypeError as exc:
        raise ReflectorCuboidReferenceError('subdivisions must be integer or three integers') from exc
    if len(counts) != 3 or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1 for n in counts):
        raise ReflectorCuboidReferenceError('subdivisions must be positive integers')
    nx, ny, nz = counts
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        h = L/np.asarray(counts)
        volume, cv = float(np.prod(L)), float(np.prod(h))
        areas = volume/L[np.array([0, 1, 0, 1, 2, 2])]
        density = currents/areas[:, None]
        delta = R@mean-currents.sum(axis=0)/volume
        eig = [4*np.sin(np.pi*np.arange(n)/(2*n))**2/step**2 for n, step in zip(counts, h)]
        lambdas = (eig[2][:, None, None]+eig[1][None, :, None]+eig[0][None, None, :]).ravel()
        blocks = R[None]+lambdas[:, None, None]*np.diag(D)[None]
    if (volume <= 0 or cv <= 0 or np.any(h <= 0)
            or not all(np.all(np.isfinite(a)) for a in (h, volume, cv, areas, density, delta, blocks))):
        raise ReflectorCuboidReferenceError('nonrepresentable tensor geometry or coefficients')
    try:
        singular = np.linalg.svd(blocks, compute_uv=False)
        condition = float(singular.max()/singular.min()) if singular.min() > 0 else np.inf
        stable_loss = bool(np.all(np.linalg.eigvals(R).real > 0))
    except np.linalg.LinAlgError as exc:
        raise ReflectorCuboidReferenceError('modal conditioning failed') from exc
    if not stable_loss or not np.isfinite(condition) or condition > condition_limit:
        raise ReflectorCuboidReferenceError('singular/non-M-matrix loss or full modal operator exceeds condition_limit')
    shape = (G, nz, ny, nx)
    rhs = np.broadcast_to(delta[:, None, None, None], shape).copy()
    faces = ((slice(None), slice(None), slice(None), 0),
             (slice(None), slice(None), 0, slice(None)),
             (slice(None), slice(None), slice(None), -1),
             (slice(None), slice(None), -1, slice(None)),
             (slice(None), 0, slice(None), slice(None)),
             (slice(None), -1, slice(None), slice(None)))
    axes = (0, 1, 0, 1, 2, 2)
    for side, (selection, axis) in enumerate(zip(faces, axes)):
        rhs[selection] += density[side, :, None, None]/h[axis]
    if not np.all(np.isfinite(rhs)):
        raise ReflectorCuboidReferenceError('nonfinite boundary right hand side')
    transformed = dctn(rhs, axes=(1, 2, 3), norm='ortho', workers=1)
    try:
        modes = np.linalg.solve(blocks, transformed.reshape(G, -1).T[..., None])[..., 0].T.reshape(shape)
    except np.linalg.LinAlgError as exc:
        raise ReflectorCuboidReferenceError('modal solve failed') from exc
    modes[:, 0, 0, 0] = mean*np.sqrt(nx*ny*nz)
    phi = idctn(modes, axes=(1, 2, 3), norm='ortho', workers=1)
    # Independently evaluate conservative physical-space flux differences.
    lap = np.zeros_like(phi)
    lap_scale = np.zeros_like(phi)
    for axis, step in zip((3, 2, 1), h):
        left, right = [slice(None)]*4, [slice(None)]*4
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        term = D[:, None, None, None]*(phi[left]-phi[right])/step**2
        lap[left] += term
        lap[right] -= term
        lap_scale[left] += abs(term)
        lap_scale[right] += abs(term)
    reaction = np.einsum('gh,hzyx->gzyx', R, phi)
    residual = lap+reaction-rhs
    residual_error = _relative(residual, lap_scale+np.einsum('gh,hzyx->gzyx', abs(R), abs(phi))+abs(rhs))
    measured_mean = phi.mean(axis=(1, 2, 3))
    mean_error = _relative(measured_mean-mean, mean)
    traces, patch_areas, patch_currents, recovered, trace_current_errors = [], [], [], [], []
    for side, (selection, axis) in enumerate(zip(faces, axes)):
        owner = phi[selection]
        trace = owner+(h[axis]/2)*density[side, :, None, None]/D[:, None, None]
        area = np.full(owner.shape[1:], cv/h[axis])
        current = np.broadcast_to(density[side, :, None, None], owner.shape)*area
        reconstructed_density = D[:, None, None]*(trace-owner)/(h[axis]/2)
        trace_current_errors.append(_relative(reconstructed_density-density[side, :, None, None],
            abs(density[side, :, None, None])+D[:, None, None]*(abs(trace)+abs(owner))/(h[axis]/2)))
        traces.append(_readonly(trace)); patch_areas.append(_readonly(area)); patch_currents.append(_readonly(current))
        recovered.append(current.sum(axis=(1, 2)))
    recovered = np.array(recovered)
    current_error = _relative(recovered-currents, abs(currents))
    balance = volume*(R@measured_mean-delta)-recovered.sum(axis=0)
    balance_error = _relative(balance, volume*(abs(R)@abs(measured_mean)+abs(delta))+abs(recovered).sum(axis=0))
    face_means = np.array([t.mean(axis=(1, 2)) for t in traces])
    if not all(np.all(np.isfinite(a)) for a in (phi, face_means, measured_mean, residual, balance)):
        raise ReflectorCuboidReferenceError('nonfinite solved field or diagnostics')
    if max(residual_error, mean_error, current_error, balance_error, *trace_current_errors) > rtol:
        raise ReflectorCuboidReferenceError('independent FV residual/current/balance/mean checks failed')
    positive = bool(np.all(phi > 0) and all(np.all(t > 0) for t in traces))
    diagnostics = MappingProxyType(dict(
        condition_number_2=condition, minimum_singular_value=float(singular.min()),
        maximum_singular_value=float(singular.max()), modal_blocks=nx*ny*nz,
        uniform_block_in_condition_gate=True, residual_relative=residual_error,
        current_relative=current_error, trace_current_relative=max(trace_current_errors),
        balance_relative=balance_error, mean_relative=mean_error,
        recovered_net_inward_integrated=_readonly(recovered),
        integrated_balance_residual=_readonly(balance), minimum_cell_flux=float(phi.min()),
        minimum_boundary_trace=min(float(t.min()) for t in traces),
        full_field_positive=positive, physical_admission=False, diagnostic_only=True,
        continuous_positivity_certified=False, boundary_current_profile_model='uniform_face_density',
        auxiliary_source='fixed native-mean algebraic balance', transform_workers=1,
        linear_solver='orthonormal_DCT_II_group_blocks', rtol=rtol, condition_limit=condition_limit))
    result = ReflectorCuboidReferenceResult(_readonly(phi), _readonly(face_means), tuple(traces),
        tuple(patch_areas), tuple(patch_currents), _readonly(delta), _readonly(measured_mean), diagnostics)
    if require_positive and not positive:
        raise NonpositiveReflectorCuboidError(result)
    return result
