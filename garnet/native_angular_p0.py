"""Fixed P0 Galerkin ILU plus multiplicative diagonal angular preconditioning."""
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spilu

from .native_angular_source import _budget, _readonly
from .native_mesh_transport import NativeMeshTransport, _positive


def _freeze_sparse(matrix):
    matrix = matrix.copy()
    for name in ('data', 'indices', 'indptr'):
        value = getattr(matrix, name)
        setattr(matrix, name, _readonly(value, value.dtype))
    return matrix


def _sparse_bytes(matrix):
    return sum(getattr(matrix, name).nbytes for name in ('data', 'indices', 'indptr'))


def _validate_factor_storage(storage):
    if storage not in ('superlu', 'packed'):
        raise ValueError("factor_storage must be 'superlu' or 'packed'")
    if storage == 'packed':
        from .packed_superlu import _check_backend
        _check_backend()


def _pack_factor(factor, *, reserved_bytes, max_bytes, source_array_bytes):
    """Guard conversion, verify the original action, and retain only owned arrays."""
    from .packed_superlu import PackedSuperLU
    size = factor.shape[0]
    # Explicit L/U, packed copies, inverse/permutations, normalization scratch,
    # and bounded probe vectors coexist with the original opaque factor. The
    # caller reserves its original factor allowance; native overhead still
    # requires an external process memory cap.
    packing_workspace = 3*source_array_bytes+256*size
    _budget(reserved_bytes+packing_workspace, max_bytes)
    packed = PackedSuperLU.from_superlu(factor)
    comparison = 0.
    for signed in (False, True):
        rhs = 1.+np.arange(size, dtype=float)/size
        if signed:
            rhs[::2] *= -1
        reference, actual = factor.solve(rhs), packed.solve(rhs)
        scale = float(np.max(np.abs(reference)))
        error = float(np.max(np.abs(actual-reference)))
        relative = error/scale if scale else error
        if (not np.all(np.isfinite(actual)) or not np.all(np.isfinite(reference))
                or not np.isfinite(relative) or relative > 1e-12):
            raise ValueError('packed factor action comparison exceeds 1e-12 or is nonfinite')
        comparison = max(comparison, relative)
    digest = hashlib.sha256()
    for array in packed.owned_arrays:
        digest.update(str(array.shape).encode('ascii'))
        digest.update(array.dtype.str.encode('ascii'))
        digest.update(memoryview(array).cast('B'))
    return packed, dict(
        factor_storage='packed', factor_backend='cpu_scipy_private_superlu_gstrs',
        factor_backend_abi='scipy-1.18.1:_superlu.gstrs',
        packed_payload_array_bytes=packed.payload_nbytes,
        packed_payload_sha256=digest.hexdigest(), packed_payload_nnz=int(packed.nnz),
        packed_action_probe_count=2, packed_action_relative_error=comparison,
        packing_workspace_estimate_bytes=packing_workspace,
        packing_peak_estimate_bytes=reserved_bytes+packing_workspace,
        packed_implementation_sha256=hashlib.sha256(
            Path(__file__).with_name('packed_superlu.py').read_bytes()).hexdigest(),
        original_superlu_retained=False, private_gstrs_workspace_bounded=False,
        external_hard_memory_cap_required=True)


def _half_range(quadrature, area, normal):
    directions = quadrature.directions[:, :len(normal)]
    speed = directions @ normal
    positive = float(area*np.dot(quadrature.weights, np.maximum(speed, 0))/(4*np.pi))
    negative = float(area*np.dot(quadrature.weights, np.maximum(-speed, 0))/(4*np.pi))
    reference = float(area*np.dot(quadrature.weights,
                                  np.linalg.norm(directions, axis=1))/(4*np.pi))
    if (not np.all(np.isfinite([positive, negative, reference])) or
            abs(positive-negative) > 128*np.finfo(float).eps*reference):
        raise ValueError('discrete weighted positive/negative face half ranges differ')
    return positive


@dataclass(frozen=True)
class P0GalerkinTransport(NativeMeshTransport):
    """Opt-in fixed linear left preconditioner; A/F and convergence gates unchanged.

    R psi=sum_n w_n psi_n, P phi=phi/(4*pi), Ac=R A P. Factor Ac once;
    z=P*ILU^-1*R*r, M^-1*r=z+D_stream^-1*(r-A*z). No fission in Ac.
    Constructor/factor failures raise ValueError with no fallback or shift.
    factor_storage='packed' explicitly selects SciPy 1.18.1's private CPU
    triangular application after roundoff checks; unsupported ABIs fail closed.
    """
    ilu_drop_tolerance: float = 1e-6
    ilu_fill_factor: float = 20.
    factor_storage: str = field(default='superlu', kw_only=True)
    _coarse: object = field(init=False, repr=False, compare=False)
    _factor: object = field(init=False, repr=False, compare=False)
    _p0_provenance: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        _validate_factor_storage(self.factor_storage)
        super().__post_init__()
        drop = _positive(self.ilu_drop_tolerance, 'ilu_drop_tolerance')
        fill = _positive(self.ilu_fill_factor, 'ilu_fill_factor')
        if drop > 1 or fill < 1:
            raise ValueError('need ilu_drop_tolerance <= 1 and ilu_fill_factor >= 1')
        object.__setattr__(self, 'ilu_drop_tolerance', drop)
        object.__setattr__(self, 'ilu_fill_factor', fill)
        if self.closed_conservative:
            raise ValueError('P0 coarse factor rejected: closed conservative singular system')
        if not np.all(np.isfinite(self.diagonal)) or np.any(self.diagonal <= 0):
            raise ValueError('P0 preconditioner requires finite positive streaming diagonal')
        s = self.streaming
        nc, na, ng = s.shape
        size, coarse_size = nc*na*ng, nc*ng
        entry_bound = nc*ng*ng + 4*len(s.owners)*ng + len(s.boundary_owners)*ng
        base_estimate = self.memory_estimate_bytes
        # CSR/CSC/COO and Python triplet construction, coarse vectors/permutations,
        # retained LU estimate, extra angular residual/action workspace. Fill is
        # an estimate, not a SuperLU allocation cap; external limits remain vital.
        other_extra = (192*entry_bound + 8*(12*size + 16*coarse_size + 4*na*ng)
                       + 64*coarse_size)
        if fill > self.max_bytes/(32*entry_bound):
            raise ValueError('allocation estimate for P0 ILU fill exceeds max_bytes')
        factor_estimate = 32*math.ceil(fill*entry_bound)
        _budget(base_estimate+other_extra+factor_estimate, self.max_bytes)
        rows, columns, values = [], [], []
        for cell, mid in enumerate(s.source.cell_material_ids):
            reaction = s.area[cell]*(np.diag(s.total[mid])-s.source.production[mid][0])
            if not np.all(np.isfinite(reaction)):
                raise ValueError('nonfinite native P0 reaction block')
            for to_group in range(ng):
                for from_group in range(ng):
                    rows.append(cell*ng+to_group)
                    columns.append(cell*ng+from_group)
                    values.append(reaction[to_group, from_group])
        for (left, right), area, normal in zip(s.owners, s.measures, s.normals):
            transfer = _half_range(s.quadrature, area, normal)
            for group in range(ng):
                left_row, right_row = left*ng+group, right*ng+group
                rows.extend((left_row, right_row, left_row, right_row))
                columns.extend((left_row, right_row, right_row, left_row))
                values.extend((transfer, transfer, -transfer, -transfer))
        for spec, owner, area, normal in zip(self.boundary_conditions, s.boundary_owners,
                                             s.boundary_measures, s.boundary_normals):
            transfer = _half_range(s.quadrature, area, normal)
            if spec == 'vacuum':
                for group in range(ng):
                    row = owner*ng+group
                    rows.append(row)
                    columns.append(row)
                    values.append(transfer)
        coarse = coo_matrix((values, (rows, columns)), shape=(coarse_size, coarse_size)).tocsr()
        coarse.sum_duplicates()
        coarse.eliminate_zeros()
        coarse.sort_indices()
        if not np.all(np.isfinite(coarse.data)):
            raise ValueError('nonfinite P0 Galerkin matrix')
        try:
            factor = spilu(coarse.tocsc(), drop_tol=drop, fill_factor=fill,
                           drop_rule='basic,area', permc_spec='COLAMD', diag_pivot_thresh=1.,
                           options={'Equil': False})
        except (RuntimeError, ValueError, MemoryError) as error:
            raise ValueError(f'P0 coarse ILU factorization failed without fallback: {error}') from error
        lower, upper = factor.L, factor.U
        if not np.all(np.isfinite(lower.data)) or not np.all(np.isfinite(upper.data)):
            raise ValueError('nonfinite retained P0 ILU factors')
        pivots = upper.diagonal()
        if not np.all(np.isfinite(pivots)) or np.any(pivots == 0):
            raise ValueError('P0 coarse ILU factorization failed: zero/nonfinite pivot')
        actual_bytes = (_sparse_bytes(lower)+_sparse_bytes(upper)
                        + factor.perm_r.nbytes+factor.perm_c.nbytes)
        total_estimate = base_estimate+other_extra+max(factor_estimate, actual_bytes)
        _budget(total_estimate, self.max_bytes)
        packed_config = {}
        if self.factor_storage == 'packed':
            factor, packed_config = _pack_factor(
                factor, reserved_bytes=total_estimate, max_bytes=self.max_bytes,
                source_array_bytes=actual_bytes)
            actual_bytes = factor.payload_nbytes
            total_estimate = max(total_estimate, packed_config['packing_peak_estimate_bytes'])
        object.__setattr__(self, 'memory_estimate_bytes', total_estimate)
        object.__setattr__(self, '_coarse', _freeze_sparse(coarse))
        object.__setattr__(self, '_factor', factor)
        implementations = []
        for name in ('native_angular_p0.py', 'native_mesh_transport.py', 'native_mesh_streaming.py',
                     'native_angular_source.py', 'moment_materials.py'):
            path = Path(__file__).with_name(name)
            implementations.append((name, hashlib.sha256(path.read_bytes()).hexdigest()))
        config = dict(restriction='sum_physical_weights', prolongation='isotropic_over_4pi',
                      coarse_operator='R_A_P_without_fission', correction='z_plus_Dinv_r_minus_Az',
                      factorization_count=1, ilu_drop_tolerance=drop, ilu_fill_factor=fill,
                      ilu_drop_rule='basic,area', ilu_permc_spec='COLAMD', ilu_diag_pivot_thresh=1.,
                      ilu_equil=False, factor_lower_nnz=int(lower.nnz), factor_upper_nnz=int(upper.nnz),
                      pivot_rule='reject_exact_zero_or_nonfinite',
                      retained_factor_array_bytes=int(actual_bytes), coarse_size=coarse_size,
                      coarse_nnz=int(coarse.nnz), coarse_csr_array_bytes=_sparse_bytes(coarse),
                      coarse_csr_sha256=hashlib.sha256(coarse.data.tobytes()+coarse.indices.tobytes()
                                                     +coarse.indptr.tobytes()).hexdigest(),
                      base_memory_estimate_bytes=base_estimate, extra_nonfactor_estimate_bytes=other_extra,
                      factor_preflight_estimate_bytes=factor_estimate, memory_estimate_bytes=total_estimate,
                      fill_factor_is_not_memory_bound=True, opaque_superlu_overhead_included=False,
                      scipy_version=scipy.__version__, numpy_version=np.__version__,
                      implementation_sha256=tuple(implementations), physical_admission=False,
                      regularization=False, fallback=False, adaptive=False)
        config.update(packed_config)
        object.__setattr__(self, '_p0_provenance', MappingProxyType(config))

    @property
    def coarse_matrix(self):
        """Independent read-only CSR copy, in cell-major/group-minor order."""
        return _freeze_sparse(self._coarse)

    def restrict(self, angular_field):
        return self.scalar_flux(angular_field)

    def prolong(self, scalar_field):
        scalar = np.asarray(scalar_field)
        expected = (self.streaming.shape[0], self.streaming.shape[2])
        if scalar.shape != expected or scalar.dtype.kind not in 'fiu':
            raise ValueError(f'P0 scalar field must have real shape {expected}')
        scalar = np.asarray(scalar, dtype=float)
        self.streaming._finite(scalar)
        return np.broadcast_to(scalar[:, None, :]/(4*np.pi), self.streaming.shape).copy()

    def precondition(self, residual):
        residual = self._flux(residual)
        coarse_rhs = self.restrict(residual)
        try:
            correction = self._factor.solve(coarse_rhs.ravel()).reshape(coarse_rhs.shape)
        except (RuntimeError, ValueError) as error:
            raise ValueError(f'fixed P0 ILU application failed without fallback: {error}') from error
        self.streaming._finite(correction)
        z = self.prolong(correction)
        return self.streaming._finite(z+(residual-self.action(z))/self.diagonal)

    def preconditioner_provenance(self):
        return dict(preconditioner='fixed_p0_galerkin_ilu_then_streaming_diagonal',
                    p0_galerkin=self._p0_provenance)
