"""Auxiliary nonfission tensor-FV diagnostic with declared receiver vacuum faces.

Prescribed internal currents and native group means are retained. Exposed faces
instead obey J_out=alpha*phi_face. Delta is a jointly solved constant auxiliary
source, never core physics. This is not an exact MC equivalence or production ADF.
"""
from types import MappingProxyType

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .reflector_cuboid_reference import (
    _array, _readonly, _relative, _FACE_TANGENTS,
    ReflectorCuboidReferenceError, NonpositiveReflectorCuboidError,
    ReflectorCuboidReferenceResult,
)


def solve_reflector_vacuum_reference(
        diffusion, loss, widths, native_mean, prescribed_net_inward_integrated,
        vacuum_faces, *, subdivisions=8, alpha=.5, condition_limit=1e12,
        rtol=1e-10, require_positive=True, net_inward_sign_moments=None):
    """Solve A phi=b+B delta and C phi=mean without field normalization.

    Arrays use the cuboid reference conventions: (G,nz,ny,nx), faces
    W/S/E/N/bottom/top, inward integrated currents (6,G), optional signed
    affine-current moments (6,2,G), positive global tangents. Vacuum mask is
    strictly boolean (6,); prescribed currents/moments there must be zero.
    D and widths are positive, R is a nonsingular nonfission M-matrix.
    Sparse LU is reused for G+1 primary right hand sides and the transpose
    condition certificate. Returned arrays and diagnostics are immutable.
    Negative fields are retained on NonpositiveReflectorCuboidError.result;
    require_positive=False returns them explicitly for diagnosis.
    """
    D, R = _array(diffusion, 'diffusion'), _array(loss, 'loss')
    L, mu = _array(widths, 'widths'), _array(native_mean, 'native_mean')
    I = _array(prescribed_net_inward_integrated, 'prescribed_net_inward_integrated')
    G = D.size
    if (D.ndim != 1 or not G or R.shape != (G, G) or L.shape != (3,)
            or mu.shape != (G,) or I.shape != (6, G)):
        raise ReflectorCuboidReferenceError('expected shapes G,GxG,3,G,6xG')
    if np.any(D <= 0) or np.any(L <= 0) or np.any(mu <= 0):
        raise ReflectorCuboidReferenceError('diffusion, widths and native_mean must be positive')
    mask = np.asarray(vacuum_faces)
    if mask.shape != (6,) or mask.dtype.kind != 'b':
        raise ReflectorCuboidReferenceError('vacuum_faces must be boolean (6,)')
    if not isinstance(require_positive, (bool, np.bool_)):
        raise ReflectorCuboidReferenceError('require_positive must be boolean')
    scalars = [_array(v, name) for v, name in
               ((alpha, 'alpha'), (condition_limit, 'condition_limit'), (rtol, 'rtol'))]
    if any(v.ndim for v in scalars):
        raise ReflectorCuboidReferenceError('alpha, condition_limit and rtol must be scalar')
    alpha, condition_limit, rtol = map(float, scalars)
    if alpha <= 0 or condition_limit < 1 or not 0 < rtol < 1:
        raise ReflectorCuboidReferenceError('alpha>0, condition_limit>=1, 0<rtol<1 required')
    counts = (subdivisions,)*3 if isinstance(subdivisions, (int, np.integer)) else subdivisions
    try:
        counts = tuple(counts)
    except TypeError as exc:
        raise ReflectorCuboidReferenceError('subdivisions must be integer or three integers') from exc
    if len(counts) != 3 or any(isinstance(n, (bool, np.bool_)) or
            not isinstance(n, (int, np.integer)) or n < 1 for n in counts):
        raise ReflectorCuboidReferenceError('subdivisions must be positive integers')
    moments = (np.zeros((6, 2, G)) if net_inward_sign_moments is None else
               _array(net_inward_sign_moments, 'net_inward_sign_moments'))
    if moments.shape != (6, 2, G):
        raise ReflectorCuboidReferenceError('net_inward_sign_moments must be (6,2,G)')
    if np.any(I[mask] != 0) or np.any(moments[mask] != 0):
        raise ReflectorCuboidReferenceError('vacuum faces require zero prescribed currents and moments')
    for s, tangents in enumerate(_FACE_TANGENTS):
        for t, a in enumerate(tangents):
            if np.any(moments[s, t] != 0) and (counts[a] < 2 or counts[a] % 2):
                raise ReflectorCuboidReferenceError('nonzero sign moment requires even tangential subdivisions >=2')
    offdiag = R.copy()
    np.fill_diagonal(offdiag, 0)
    if np.any(offdiag > 0) or np.any(np.diag(R) <= 0):
        raise ReflectorCuboidReferenceError('loss must be a nonfission M-matrix')
    try:
        certificate = np.linalg.solve(R.T, np.ones(G))
    except np.linalg.LinAlgError as exc:
        raise ReflectorCuboidReferenceError('singular loss') from exc
    certificate_error = _relative(R.T@certificate-1, abs(R.T)@abs(certificate)+1)
    if np.any(certificate <= 0) or certificate_error > rtol:
        raise ReflectorCuboidReferenceError('loss M-matrix positive scaling certificate failed')
    nx, ny, nz = counts
    n = nx*ny*nz
    with np.errstate(all='ignore'):
        h = L/np.asarray(counts)
        volume, cv = float(np.prod(L)), float(np.prod(h))
        areas = volume/L[[0, 1, 0, 1, 2, 2]]
        diffusion_steps = D[:, None]/h[None, :]**2
        robin = alpha/(1+alpha*h[None, :]/(2*D[:, None]))
    if (min(volume, cv) <= 0 or np.any(h <= 0) or np.any(areas <= 0) or np.any(robin <= 0)
            or not all(np.all(np.isfinite(a)) for a in
                       (h, volume, cv, areas, diffusion_steps, robin))):
        raise ReflectorCuboidReferenceError('nonrepresentable geometry or coefficients')
    shape = (G, nz, ny, nx)
    faces = ((slice(None), slice(None), slice(None), 0),
             (slice(None), slice(None), 0, slice(None)),
             (slice(None), slice(None), slice(None), -1),
             (slice(None), slice(None), -1, slice(None)),
             (slice(None), 0, slice(None), slice(None)),
             (slice(None), -1, slice(None), slice(None)))
    axes = (0, 1, 0, 1, 2, 2)
    ids = np.arange(G*n).reshape(shape)
    A = sparse.kron(sparse.csc_matrix(R), sparse.eye(n), format='csc')
    # Conservative graph diffusion, assembled separately from residual evaluation.
    for a, dim in enumerate((3, 2, 1)):
        lo, hi = [slice(None)]*4, [slice(None)]*4
        lo[dim], hi[dim] = slice(None, -1), slice(1, None)
        p, q = ids[tuple(lo)].ravel(), ids[tuple(hi)].ravel()
        weights = np.broadcast_to(diffusion_steps[:, a, None, None, None],
                                  ids[tuple(lo)].shape).ravel()
        A += sparse.coo_matrix((np.concatenate((weights, weights, -weights, -weights)),
             (np.concatenate((p, q, p, q)), np.concatenate((p, q, q, p)))), shape=A.shape).tocsc()
    b = np.zeros(shape)
    densities, signs = [], []
    for s, (selection, a) in enumerate(zip(faces, axes)):
        tangents = _FACE_TANGENTS[s]
        centered = [2*np.arange(counts[t])+1-counts[t] for t in tangents]
        v, u = np.meshgrid(centered[1], centered[0], indexing='ij')
        signs.append(np.stack((np.sign(u), np.sign(v))))
        density = np.broadcast_to(I[s, :, None, None]/areas[s], ids[selection].shape).copy()
        for t, coord in enumerate((u, v)):
            axis = tangents[t]
            density += 4*moments[s, t, :, None, None]*(coord*h[axis]/2)/(areas[s]*L[axis])
        densities.append(density)
        if mask[s]:
            p = ids[selection].ravel()
            weights = np.broadcast_to(robin[:, a, None, None]/h[a], ids[selection].shape).ravel()
            A += sparse.coo_matrix((weights, (p, p)), shape=A.shape).tocsc()
        else:
            b[selection] += density/h[a]
    if not np.all(np.isfinite(A.data)) or not np.all(np.isfinite(b)):
        raise ReflectorCuboidReferenceError('nonfinite assembled operator/right hand side')
    operator_scaling_image = A.T@np.repeat(certificate, n)
    if not np.all(np.isfinite(operator_scaling_image)) or np.any(operator_scaling_image <= 0):
        raise ReflectorCuboidReferenceError('operator M-matrix positive scaling failed')
    B = np.repeat(np.eye(G), n, axis=0)
    try:
        lu = splu(A)
        solved = lu.solve(np.column_stack((b.ravel(), B)))
        inv_column_sums = lu.solve(np.ones(G*n), trans='T')
    except (RuntimeError, ValueError) as exc:
        raise ReflectorCuboidReferenceError('sparse factorization/solve failed') from exc
    # R.T v>0 proves A.T (v tensor 1)>0: diffusion cancels and Robin is positive.
    # Thus A is an M-matrix and its inverse is nonnegative: this is its 1-norm.
    inverse_certificate_error = _relative(A.T@inv_column_sums-1,
                                          abs(A.T)@abs(inv_column_sums)+1)
    operator_condition = float(np.asarray(abs(A).sum(axis=0)).max()*inv_column_sums.max())
    if (np.any(inv_column_sums <= 0) or inverse_certificate_error > rtol
            or not np.isfinite(operator_condition) or operator_condition > condition_limit):
        raise ReflectorCuboidReferenceError('operator exceeds condition_limit or inverse certificate failed')
    z, Y = solved[:, 0], solved[:, 1:]
    C = B.T/n
    S = C@Y
    # Scale constraints by each native mean and sources by the resulting
    # response. This keeps tiny native groups represented in the Schur solve.
    row_scale = mu.copy()
    if np.any(row_scale <= 0):
        raise ReflectorCuboidReferenceError('singular Schur system')
    Sr = S/row_scale[:, None]
    col_scale = np.max(abs(Sr), axis=0)
    if np.any(col_scale <= 0):
        raise ReflectorCuboidReferenceError('singular Schur system')
    scaled = Sr/col_scale[None, :]
    try:
        schur_condition = float(np.linalg.cond(scaled))
        if not np.isfinite(schur_condition) or schur_condition > condition_limit:
            raise ReflectorCuboidReferenceError('scaled Schur exceeds condition_limit')
        target = (mu-C@z)/row_scale
        unknown = np.linalg.solve(scaled, target)
        delta = unknown/col_scale
    except np.linalg.LinAlgError as exc:
        raise ReflectorCuboidReferenceError('singular Schur system') from exc
    schur_error = _relative(scaled@unknown-target, abs(scaled)@abs(unknown)+abs(target))
    phi = (z+Y@delta).reshape(shape)
    mean = phi.mean(axis=(1, 2, 3))
    reaction = np.einsum('gh,hzyx->gzyx', R, phi)
    residual = reaction-delta[:, None, None, None]
    scale = np.einsum('gh,hzyx->gzyx', abs(R), abs(phi))+abs(delta[:, None, None, None])
    for a, dim in enumerate((3, 2, 1)):
        lo, hi = [slice(None)]*4, [slice(None)]*4
        lo[dim], hi[dim] = slice(None, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        term = D[:, None, None, None]*(phi[lo]-phi[hi])/h[a]**2
        residual[lo] += term; residual[hi] -= term
        scale[lo] += abs(term); scale[hi] += abs(term)
    traces, patch_areas, patch_currents, recovered, recovered_moments = [], [], [], [], []
    robin_errors, trace_errors = [], []
    for s, (selection, a) in enumerate(zip(faces, axes)):
        owner = phi[selection]
        if mask[s]:
            trace = owner/(1+alpha*h[a]/(2*D[:, None, None]))
            density = -alpha*trace
        else:
            density = densities[s]
            trace = owner+density*h[a]/(2*D[:, None, None])
        reconstructed = 2*D[:, None, None]*(trace-owner)/h[a]
        trace_errors.append(_relative(reconstructed-density,
            abs(density)+2*D[:, None, None]*(abs(trace)+abs(owner))/h[a]))
        if mask[s]:
            robin_errors.append(_relative(reconstructed+alpha*trace,
                                          abs(reconstructed)+alpha*abs(trace)))
        residual[selection] -= density/h[a]
        scale[selection] += abs(density)/h[a]
        area = np.full(owner.shape[1:], cv/h[a])
        current = density*area
        traces.append(_readonly(trace)); patch_areas.append(_readonly(area)); patch_currents.append(_readonly(current))
        recovered.append(current.sum(axis=(1, 2)))
        recovered_moments.append([(current*sign).sum(axis=(1, 2)) for sign in signs[s]])
    recovered, recovered_moments = np.array(recovered), np.array(recovered_moments)
    normalization = abs(I)+2*abs(moments).sum(axis=1)
    current_error = _relative((recovered-I)[~mask], normalization[~mask]) if np.any(~mask) else 0.
    moment_error = _relative((recovered_moments-moments)[~mask], normalization[~mask, None, :]) if np.any(~mask) else 0.
    balance = volume*(R@mean-delta)-recovered.sum(axis=0)
    balance_error = _relative(balance, volume*(abs(R)@abs(mean)+abs(delta))+abs(recovered).sum(axis=0))
    residual_error = _relative(residual, scale)
    mean_error = _relative(mean-mu, mu)
    if max(residual_error, mean_error, balance_error, current_error, moment_error,
           schur_error, *trace_errors, *robin_errors) > rtol:
        raise ReflectorCuboidReferenceError(
            'independent FV/current/Robin/mean/Schur/balance checks failed: '
            f'FV={residual_error:.3g}, mean={mean_error:.3g}, balance={balance_error:.3g}, '
            f'current={current_error:.3g}, moment={moment_error:.3g}, '
            f'Schur={schur_error:.3g}, Robin={max(robin_errors, default=0.):.3g}')
    positive = bool(np.all(phi > 0) and all(np.all(t > 0) for t in traces))
    diagnostics = MappingProxyType(dict(
        operator_condition_number_1=operator_condition, scaled_schur_condition_number_2=schur_condition,
        loss_positive_scaling=_readonly(certificate), loss_certificate_relative=certificate_error,
        operator_positive_scaling_minimum=float(operator_scaling_image.min()),
        inverse_certificate_relative=inverse_certificate_error, residual_relative=residual_error,
        mean_relative=mean_error, balance_relative=balance_error, current_relative=current_error,
        sign_moment_relative=moment_error, robin_relative=max(robin_errors, default=0.),
        trace_current_relative=max(trace_errors), scaled_schur_relative=schur_error,
        recovered_net_inward_integrated=_readonly(recovered),
        recovered_net_inward_sign_moments=_readonly(recovered_moments),
        input_net_inward_sign_moments=_readonly(moments),
        integrated_balance_residual=_readonly(balance), face_patch_signs=tuple(_readonly(s) for s in signs),
        vacuum_faces=tuple(bool(v) for v in mask), alpha=alpha,
        full_field_positive=positive, minimum_cell_flux=float(phi.min()),
        minimum_boundary_trace=min(float(t.min()) for t in traces),
        diagnostic_only=True, physical_admission=False, continuous_positivity_certified=False,
        auxiliary_source='joint fixed-native-mean algebraic constraint; never core physics',
        linear_solver='sparse_LU_joint_mean_Schur', factorization_count=1,
        primary_right_hand_sides=G+1, rtol=rtol, condition_limit=condition_limit))
    result = ReflectorCuboidReferenceResult(_readonly(phi), _readonly(np.array([t.mean(axis=(1, 2)) for t in traces])),
        tuple(traces), tuple(patch_areas), tuple(patch_currents), _readonly(delta), _readonly(mean), diagnostics)
    if require_positive and not positive:
        raise NonpositiveReflectorCuboidError(result)
    return result
