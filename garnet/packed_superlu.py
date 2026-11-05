"""Experimental CPU SuperLU application with tightly owned explicit factors.

This adapter deliberately supports only SciPy 1.18.1's private ``gstrs`` ABI.
It retains no SuperLU object, so its caller can release SuperLU's internal
capacity after packing. Packing temporarily coexists with that capacity.

No entries are dropped. SciPy's private triangular encoding requires
``U = U_unit @ diag(d)``: the strictly upper entries are divided by ``d``
once during packing and the solution is scaled by ``1/d`` on application.
This preserves the factor's numerical action to roundoff, not bitwise.
"""

from __future__ import annotations

import numpy as np
import scipy
from scipy.sparse.linalg._dsolve import _superlu


_SUPPORTED_SCIPY = "1.18.1"
_DTYPES = tuple(np.dtype(t) for t in ("float32", "float64", "complex64", "complex128"))


def _check_backend():
    if scipy.__version__ != _SUPPORTED_SCIPY or not callable(getattr(_superlu, "gstrs", None)):
        raise RuntimeError(
            "PackedSuperLU requires CPU SciPy 1.18.1 private _superlu.gstrs; "
            f"found SciPy {scipy.__version__}. No fallback is enabled.")


class PackedSuperLU:
    """Owned CSC triangular payload for repeated normal CPU solves.

    Construct with :meth:`from_superlu`. Only ``trans='N'`` is supported.
    ``nnz`` counts the explicit L/U slots (including U's encoded zero diagonal),
    not SuperLU's reported capacity. ``payload_nbytes`` includes every retained
    array, including permutations and inverse diagonal, but excludes Python
    object overhead and temporary workspace allocated inside private ``gstrs``.
    """

    __slots__ = ("shape", "dtype", "_lower", "_upper", "_inverse_diagonal",
                 "_row_inverse", "_perm_c")

    @classmethod
    def from_superlu(cls, factor):
        """Copy a SciPy CPU SuperLU factor without retaining any source views."""
        _check_backend()
        if not isinstance(factor, _superlu.SuperLU):
            raise TypeError("PackedSuperLU requires a SciPy CPU SuperLU factor")
        lower, upper = factor.L, factor.U
        dtype = np.dtype(lower.dtype)
        if dtype not in _DTYPES or upper.dtype != dtype:
            raise TypeError(f"Unsupported SuperLU factor dtype: {dtype}")
        n, m = factor.shape
        if n != m or n == 0 or n > np.iinfo(np.intc).max:
            raise ValueError("PackedSuperLU requires a nonempty square int32-sized factor")
        for matrix in (lower, upper):
            if matrix.nnz > np.iinfo(np.intc).max:
                raise ValueError("PackedSuperLU factor nnz exceeds int32 capacity")
            if not np.all(np.isfinite(matrix.data)):
                raise ValueError("PackedSuperLU factors must be finite")
        if not np.all(lower.diagonal() == 1):
            raise ValueError("PackedSuperLU requires a unit lower diagonal")
        diagonal = upper.diagonal()
        if np.any(diagonal == 0):
            raise ValueError("PackedSuperLU requires a nonzero upper diagonal")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            inverse = np.array(1 / diagonal, dtype=dtype, copy=True)
        if not np.all(np.isfinite(inverse)) or np.any(inverse == 0):
            raise ValueError("PackedSuperLU inverse upper diagonal must be finite and nonzero")

        def own_csc(matrix):
            return (np.array(matrix.data, copy=True),
                    np.array(matrix.indices, dtype=np.intc, copy=True),
                    np.array(matrix.indptr, dtype=np.intc, copy=True))

        result = cls()
        result.shape, result.dtype = (n, n), dtype
        result._lower, result._upper = own_csc(lower), own_csc(upper)
        values, indices, indptr = result._upper
        # Normalize columns in place without an nnz-sized repeat/scaling array.
        with np.errstate(over="ignore", invalid="ignore"):
            for column in range(n):
                start, end = indptr[column:column + 2]
                values[start:end] *= inverse[column]
                values[start:end][indices[start:end] == column] = 0
        if not np.all(np.isfinite(values)):
            raise ValueError("PackedSuperLU normalized upper factor must be finite")
        # Reject entries lost to underflow rather than silently dropping them.
        for column in range(n):
            start, end = indptr[column:column + 2]
            lost = ((values[start:end] == 0) & (upper.data[start:end] != 0)
                    & (indices[start:end] != column))
            if np.any(lost):
                raise ValueError("PackedSuperLU normalized upper factor underflowed")
        result._inverse_diagonal = inverse
        # Pr A Pc = L U: Pr b = b[argsort(perm_r)], x = z[perm_c].
        result._row_inverse = np.array(np.argsort(factor.perm_r), dtype=np.intc, copy=True)
        result._perm_c = np.array(factor.perm_c, dtype=np.intc, copy=True)
        for array in result.owned_arrays:
            array.flags.writeable = False
        return result

    @property
    def owned_arrays(self):
        """Read-only, tightly allocated arrays comprising the retained payload."""
        return (*self._lower, *self._upper, self._inverse_diagonal,
                self._row_inverse, self._perm_c)

    @property
    def payload_nbytes(self):
        return sum(array.nbytes for array in self.owned_arrays)

    @property
    def nnz(self):
        return self._lower[0].size + self._upper[0].size

    def solve(self, rhs, trans="N"):
        """Apply the packed factor to a vector or multiple column RHS.

        Sparse factors are neither copied nor rescaled here. SciPy 1.18.1's
        C wrapper borrows these arrays and allocates dimension-sized metadata
        and an RHS copy, plus native solve workspace. The RHS and stored arrays
        are never mutated.
        """
        _check_backend()
        if trans != "N":
            raise NotImplementedError("PackedSuperLU supports only trans='N'")
        if not isinstance(rhs, np.ndarray):
            raise TypeError("PackedSuperLU RHS must use the NumPy CPU backend")
        if rhs.ndim not in (1, 2) or rhs.shape[0] != self.shape[0]:
            raise ValueError(f"PackedSuperLU RHS shape {rhs.shape} is incompatible with {self.shape}")
        if not np.can_cast(rhs.dtype, self.dtype, casting="safe"):
            raise TypeError(f"Cannot safely cast RHS dtype {rhs.dtype} to factor dtype {self.dtype}")
        if not np.all(np.isfinite(rhs)):
            raise ValueError("PackedSuperLU RHS must be finite")
        if rhs.ndim == 2 and rhs.shape[1] == 0:
            return np.empty(rhs.shape, dtype=self.dtype)
        # The private wrapper performs its own dtype/Fortran-order RHS copy.
        work = rhs[self._row_inverse]
        n = self.shape[0]
        values, info = _superlu.gstrs(
            "N", n, self._lower[0].size, *self._lower,
            n, self._upper[0].size, *self._upper, work)
        if info:
            raise np.linalg.LinAlgError(f"PackedSuperLU gstrs failed with info={info}")
        scale = self._inverse_diagonal
        values *= scale if values.ndim == 1 else scale[:, None]
        return values[self._perm_c]
