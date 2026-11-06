"""Opt-in fixed P0 correction followed by native streaming-block LU solves.

L0 retains collision and the original outflow diagonal, dropping only reflected
incoming angular coupling. Scattering, fission and reflection remain in A/F.
Memory estimates cannot bound opaque SuperLU workspaces: an external hard
address-space/RSS limit remains authoritative for every factorization.
"""
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np
import scipy
from scipy.sparse import isspmatrix_csc
from scipy.sparse.linalg import splu

from .native_angular_p0 import (P0GalerkinTransport, _freeze_sparse, _sparse_bytes,
                                _pack_factor, _validate_factor_storage)
from .native_angular_source import _budget, _integer
from .native_mesh_streaming import NativeMeshStreaming
from .native_mesh_transport import _positive


def _matrix_sha256(matrix):
    digest = hashlib.sha256(str(matrix.shape).encode('ascii'))
    for name in ('indptr', 'indices', 'data'):
        array = getattr(matrix, name)
        digest.update(array.dtype.str.encode('ascii'))
        digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def same_csc(left, right):
    """Exact complete canonical CSC identity; hashes alone never authorize reuse."""
    return (isspmatrix_csc(left) and isspmatrix_csc(right) and left.shape == right.shape
            and all(getattr(left, name).dtype == getattr(right, name).dtype
                    and np.array_equal(getattr(left, name), getattr(right, name))
                    for name in ('indptr', 'indices', 'data')))


def _assembly_workspace(streaming):
    cells = streaming.shape[0]
    entries = cells+4*len(streaming.owners)+len(streaming.boundary_owners)+1
    # Python triplets, COO/CSR/CSC conversion, canonicalization and immutable
    # construction copies coexist. Use wide indices even when SciPy uses int32.
    return 256*entries+192*cells


def build_streaming_block(streaming, angle, group, *, max_bytes=256*1024**2,
                          reserved_bytes=0):
    """Build one immutable canonical CSC L0, without factors or transport setup.

    reserved_bytes accounts for caller-owned live arrays/factors. This helper
    reserves triplets, sparse conversion copies and validation workspace before
    calling NativeMeshStreaming.assembled; it never changes boundary diagonals.
    """
    if not isinstance(streaming, NativeMeshStreaming):
        raise ValueError('NativeMeshStreaming required')
    angle, group = _integer(angle, 'angle'), _integer(group, 'group')
    reserved_bytes = _integer(reserved_bytes, 'reserved_bytes')
    if angle >= streaming.shape[1] or group >= streaming.shape[2]:
        raise ValueError('angle or group outside layout')
    _budget(reserved_bytes+_assembly_workspace(streaming), max_bytes)
    csr = streaming.assembled(angle, group)
    matrix = csr.tocsc()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    streaming._finite(matrix.data)
    return _freeze_sparse(matrix)


@dataclass(frozen=True)
class StreamingBlockFactor:
    """One fixed factor and private immutable coefficient snapshot."""
    _matrix: object = field(repr=False, compare=False)
    _factor: object = field(repr=False, compare=False)
    provenance: object = field(repr=False, compare=False)

    @property
    def matrix(self):
        """Independent immutable CSC copy; changing its attributes cannot alter L0."""
        return _freeze_sparse(self._matrix)

    def solve(self, rhs):
        array = np.asarray(rhs)
        if array.shape != (self._matrix.shape[0],) or array.dtype.kind not in 'fiu':
            raise ValueError('streaming block RHS must be a real cell vector')
        array = np.asarray(array, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError('nonfinite streaming block RHS')
        try:
            result = self._factor.solve(array)
        except (RuntimeError, ValueError, MemoryError) as error:
            raise ValueError(f'fixed streaming block solve failed without fallback: {error}') from error
        if not np.all(np.isfinite(result)):
            raise ValueError('nonfinite streaming block solution')
        return result


def factor_streaming_block(matrix, *, max_bytes=256*1024**2, reserved_bytes=0,
                           fill_estimate=20., factor_storage='superlu'):
    """Factor ONE canonical CSC from build_streaming_block for setup/cost gates.

    No shift, retry, fallback, or physics repair is applied. fill_estimate is an
    advisory preflight allowance, never a cap on SuperLU's internal allocations.
    reserved_bytes excludes this matrix and includes all other caller live state.
    Retained charging includes matrix arrays, twice materialized L/U arrays and
    permutations; transient copies, vectors and estimated fill are also guarded.
    Explicit factor_storage='packed' instead retains one owned packed payload
    plus the matrix; conversion and action comparison are separately guarded.
    """
    _validate_factor_storage(factor_storage)
    reserved_bytes = _integer(reserved_bytes, 'reserved_bytes')
    fill_estimate = _positive(fill_estimate, 'fill_estimate')
    if fill_estimate < 1:
        raise ValueError('fill_estimate must be >= 1')
    if (not isspmatrix_csc(matrix) or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]
            or matrix.dtype != np.float64 or not matrix.has_canonical_format
            or np.any(matrix.data == 0) or not np.all(np.isfinite(matrix.data))):
        raise ValueError('finite zero-eliminated canonical float64 square CSC required')
    cells, entries = matrix.shape[0], max(1, matrix.nnz)
    matrix_bytes = _sparse_bytes(matrix)
    # Original CSC, immutable copy, factorization CSC/COO/CSR scratch, numeric
    # and index copies, solve verification vectors and column/row permutations.
    workspace = 6*matrix_bytes+192*cells+128*entries
    if fill_estimate > int(max_bytes)/(64*entries):
        raise ValueError('allocation estimate for streaming LU fill exceeds max_bytes')
    estimated_factors = 64*math.ceil(fill_estimate*entries)+64*cells
    _budget(reserved_bytes+workspace+estimated_factors, max_bytes)
    canonical = _freeze_sparse(matrix)
    try:
        factor = splu(canonical, permc_spec='COLAMD', diag_pivot_thresh=1., options={'Equil': False})
    except (RuntimeError, ValueError, MemoryError) as error:
        raise ValueError(f'streaming block LU factorization failed without fallback: {error}') from error
    lower, upper = factor.L, factor.U
    pivots = upper.diagonal()
    if (not np.all(np.isfinite(lower.data)) or not np.all(np.isfinite(upper.data))
            or not np.all(np.isfinite(pivots)) or np.any(pivots == 0)):
        raise ValueError('streaming block LU has zero/nonfinite pivot or factors')
    factor_bytes = _sparse_bytes(lower)+_sparse_bytes(upper)
    permutation_bytes = factor.perm_r.nbytes+factor.perm_c.nbytes
    retained = matrix_bytes+2*factor_bytes+permutation_bytes
    # Factorization has already allocated opaque memory. This measured guard
    # rejects oversized retained factors and never admits a fallback factor.
    _budget(reserved_bytes+workspace+max(estimated_factors, retained), max_bytes)
    probe = 1.+np.arange(cells, dtype=float)/cells
    rhs = canonical @ probe
    try:
        solution = factor.solve(rhs)
    except (RuntimeError, ValueError, MemoryError) as error:
        raise ValueError(f'streaming block verification failed without fallback: {error}') from error
    residual = canonical @ solution-rhs
    scale = float(np.max(np.abs(rhs)))
    error = float(np.max(np.abs(residual)))
    relative_residual = error/scale if scale else error
    if not np.all(np.isfinite(solution)) or not np.isfinite(relative_residual) or relative_residual > 1e-10:
        raise ValueError('streaming block verification residual exceeds 1e-10 or is nonfinite')
    packed_config = {}
    peak = reserved_bytes+workspace+max(estimated_factors, retained)
    if factor_storage == 'packed':
        factor, packed_config = _pack_factor(
            factor, reserved_bytes=peak, max_bytes=max_bytes,
            source_array_bytes=factor_bytes+permutation_bytes)
        retained = matrix_bytes+factor.payload_nbytes
        peak = max(peak, packed_config['packing_peak_estimate_bytes'])
        solution = factor.solve(rhs)
        packed_residual = float(np.max(np.abs(canonical @ solution-rhs)))
        packed_residual = packed_residual/scale if scale else packed_residual
        if (not np.all(np.isfinite(solution)) or not np.isfinite(packed_residual)
                or packed_residual > 1e-10):
            raise ValueError('packed streaming block verification residual exceeds 1e-10 or is nonfinite')
        packed_config['packed_verification_relative_residual'] = packed_residual
    config = dict(
        matrix_csc_sha256=_matrix_sha256(canonical), matrix_shape=canonical.shape,
        matrix_nnz=int(canonical.nnz), matrix_array_bytes=matrix_bytes,
        factor_lower_nnz=int(lower.nnz), factor_upper_nnz=int(upper.nnz),
        materialized_lu_array_bytes=factor_bytes, permutation_array_bytes=permutation_bytes,
        retained_charge_bytes=retained, reserved_bytes=reserved_bytes,
        transient_workspace_estimate_bytes=workspace, factor_preflight_estimate_bytes=estimated_factors,
        peak_estimate_bytes=peak,
        verification_relative_residual=relative_residual,
        permc_spec='COLAMD', diag_pivot_thresh=1., equil=False,
        fill_estimate=fill_estimate, fill_estimate_is_not_allocation_cap=True,
        opaque_superlu_workspace_bounded=False, external_hard_memory_cap_required=True,
        scipy_version=scipy.__version__, numpy_version=np.__version__,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        regularization=False, fallback=False, adaptive=False, physical_admission=False)
    config.update(packed_config)
    return StreamingBlockFactor(canonical, factor, MappingProxyType(config))


@dataclass(frozen=True)
class StreamingBlockP0Transport(P0GalerkinTransport):
    """Fixed z + L0^-1(r-Az), z=P ILU(Ac)^-1 Rr; opt-in only.

    Ac, its ILU and A/F are inherited unchanged. Each L0 block uses native
    streaming/collision without incoming reflection or scattering/fission.
    Complete CSC equality is required for sharing any retained block factor.
    Constructor failures stop immediately; no alternative preconditioner runs.
    """
    streaming_fill_estimate: float = 20.
    _streaming_blocks: tuple = field(init=False, repr=False, compare=False)
    _streaming_block_map: tuple = field(init=False, repr=False, compare=False)
    _sweep_provenance: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        super().__post_init__()
        fill = _positive(self.streaming_fill_estimate, 'streaming_fill_estimate')
        if fill < 1:
            raise ValueError('streaming_fill_estimate must be >= 1')
        object.__setattr__(self, 'streaming_fill_estimate', fill)
        nc, na, ng = self.streaming.shape
        p0_constructor_peak = self.memory_estimate_bytes
        base = self._post_setup_reservation_bytes
        # Additional correction, action, residual, output and packing/validation
        # vectors plus bounded block-map/cache metadata. Keep all GMRES and P0
        # allowances from the superclass, including its coarse retained factors.
        angular_workspace = 8*(6*nc*na*ng+8*nc)+1024*na*ng
        _budget(base+angular_workspace+_assembly_workspace(self.streaming), self.max_bytes)
        unique, block_map, candidates = [], [], {}
        retained = 0
        assembly_peak = base+angular_workspace+_assembly_workspace(self.streaming)
        peak = max(p0_constructor_peak, base+angular_workspace)
        for angle in range(na):
            groups = []
            for group in range(ng):
                reserved = base+angular_workspace+retained
                assembly_peak = max(assembly_peak, reserved+_assembly_workspace(self.streaming))
                peak = max(peak, assembly_peak)
                matrix = build_streaming_block(self.streaming, angle, group, max_bytes=self.max_bytes,
                                               reserved_bytes=reserved)
                key = _matrix_sha256(matrix)
                shared = next((index for index in candidates.get(key, ())
                               if same_csc(matrix, unique[index]._matrix)), None)
                if shared is None:
                    block = factor_streaming_block(matrix, max_bytes=self.max_bytes,
                                                   reserved_bytes=reserved, fill_estimate=fill,
                                                   factor_storage=self.factor_storage)
                    shared = len(unique)
                    unique.append(block)
                    candidates.setdefault(key, []).append(shared)
                    retained += block.provenance['retained_charge_bytes']
                    peak = max(peak, block.provenance['peak_estimate_bytes'])
                    del block
                groups.append(shared)
                del matrix
            block_map.append(tuple(groups))
        if len(block_map) != na or any(len(row) != ng for row in block_map):
            raise ValueError('incomplete angle/group streaming block map')
        _budget(base+angular_workspace+retained, self.max_bytes)
        object.__setattr__(self, '_streaming_blocks', tuple(unique))
        object.__setattr__(self, '_streaming_block_map', tuple(block_map))
        object.__setattr__(self, 'memory_estimate_bytes', max(peak, base+angular_workspace+retained))
        object.__setattr__(self, '_sweep_provenance', MappingProxyType(dict(
            correction='z_plus_L0inv_r_minus_Az',
            L0='native_streaming_collision_original_outflow_without_reflected_incoming',
            scattering_in_L0=False, fission_in_L0=False, reflected_coupling_retained_in_A=True,
            angle_group_layout='flux[cell,angle,group]; block_map[angle][group]',
            block_map=tuple(block_map), required_block_count=na*ng, unique_factor_count=len(unique),
            exact_csc_identity_required_for_reuse=True,
            memory_accounting_version='p0_phase_liveness_v1',
            base_p0_memory_estimate_bytes=p0_constructor_peak,
            base_p0_post_setup_reservation_bytes=base,
            streaming_assembly_peak_estimate_bytes=assembly_peak,
            angular_workspace_estimate_bytes=angular_workspace, retained_unique_charge_bytes=retained,
            memory_estimate_bytes=self.memory_estimate_bytes,
            factors=tuple(block.provenance for block in unique),
            opaque_superlu_workspace_bounded=False, external_hard_memory_cap_required=True,
            regularization=False, fallback=False, adaptive=False, physical_admission=False)))

    def streaming_inverse(self, residual):
        """Apply all fixed spatial blocks in native cell/angle/group packing."""
        residual = self._flux(residual)
        result = np.empty(self.streaming.shape)
        for angle, groups in enumerate(self._streaming_block_map):
            for group, index in enumerate(groups):
                result[:, angle, group] = self._streaming_blocks[index].solve(residual[:, angle, group])
        return self.streaming._finite(result)

    def precondition(self, residual):
        residual = self._flux(residual)
        coarse_rhs = self.restrict(residual)
        try:
            correction = self._factor.solve(coarse_rhs.ravel()).reshape(coarse_rhs.shape)
        except (RuntimeError, ValueError, MemoryError) as error:
            raise ValueError(f'fixed P0 ILU application failed without fallback: {error}') from error
        self.streaming._finite(correction)
        z = self.prolong(correction)
        return self.streaming._finite(z+self.streaming_inverse(residual-self.action(z)))

    def preconditioner_provenance(self):
        return dict(preconditioner='fixed_p0_galerkin_ilu_then_streaming_block_lu',
                    p0_galerkin=self._p0_provenance, streaming_blocks=self._sweep_provenance)
