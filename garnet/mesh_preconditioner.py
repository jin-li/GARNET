"""Fixed ILU preconditioning of assembled mesh operators.

Assembly and incomplete factorization happen on the CPU once. GPU application
uses device-resident sparse triangular solves, including SuperLU permutations.
Factorization and backend errors propagate; there is no fallback preconditioner.
"""

from time import perf_counter

import numpy as np

from .backend import asnumpy


def _assemble_tpfa(operator):
    """Return the host CSR matrix and the operator's moment-major state shape."""
    from scipy import sparse

    if hasattr(operator, "moments"):
        moments = operator.moments
        if not moments:
            raise ValueError("TPFA moment operator must have at least one moment")
        assembled = [_assemble_tpfa(moment) for moment in moments]
        state_shape = assembled[0][1]
        if any(shape != state_shape for _, shape in assembled):
            raise ValueError("TPFA moment operators must have matching shapes")
        n, m = assembled[0][0].shape[0], len(moments)
        blocks = [[None] * m for _ in range(m)]
        for i, (matrix, _) in enumerate(assembled):
            blocks[i][i] = matrix
        for (i, j), value in operator.coupling.items():
            values = np.broadcast_to(asnumpy(value), state_shape).reshape(-1)
            blocks[i][j] = sparse.diags(values, format="csr")
        matrix = sparse.bmat(blocks, format="csr")
        # Mesh boundary blocks index cells in the original spatial state shape.
        # Include every entry, including the diagonals: moments themselves do
        # not contain the Marshak contribution stored in operator.inv_diag.
        cell_indices = np.arange(n).reshape(state_shape)
        for selection, values in operator.marshak or ():
            if isinstance(selection, tuple):
                selection = tuple(asnumpy(s) if hasattr(s, "shape") else s
                                  for s in selection)
            elif hasattr(selection, "shape"):
                selection = asnumpy(selection)
            cells = np.asarray(cell_indices[selection]).reshape(-1)
            values = np.broadcast_to(asnumpy(values), (len(cells), m, m))
            rows = cells[:, None, None] + n * np.arange(m)[None, :, None]
            cols = cells[:, None, None] + n * np.arange(m)[None, None, :]
            matrix += sparse.coo_matrix(
                (values.ravel(), (np.broadcast_to(rows, values.shape).ravel(),
                                  np.broadcast_to(cols, values.shape).ravel())),
                shape=(m * n, m * n)).tocsr()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        return matrix, (m, *state_shape)

    flat = getattr(operator, "_flat", operator)
    if not all(hasattr(flat, attr) for attr in ("nbr", "w_ell", "diag")):
        raise TypeError("TPFA ILU requires a mesh ELL operator or mesh moment operator")
    diag = asnumpy(flat.diag).reshape(-1)
    nbr, weights = asnumpy(flat.nbr), asnumpy(flat.w_ell)
    n = diag.size
    rows = np.broadcast_to(np.arange(n), weights.shape)
    matrix = sparse.coo_matrix(
        (-weights.ravel(), (rows.ravel(), nbr.ravel())), shape=(n, n)).tocsr()
    matrix += sparse.diags(diag, format="csr")
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix, tuple(operator.diag.shape)


def build_tpfa_ilu(operator, xp):
    """Build a fixed callable ``r -> ILU(TPFA)**-1 r`` on NumPy or CuPy.

    ``operator`` must be the two-point operator (or a block of such operators),
    rather than the corrected operator. SciPy SuperLU uses ``drop_tol=1e-6``
    and ``fill_factor=20``. Only construction copies GPU data to the host.
    The callable requires the original state shape and matching array backend.
    """
    matrix, shape = _assemble_tpfa(operator)
    return _build_ilu(matrix, shape, xp)


def build_corrected_ilu(matrix, shape, xp):
    """Build fixed ILU of the exact corrected ``host_matrix`` without scaling.

    ``shape`` is the corrected operator's ``diag.shape``. Factorization,
    permutations, backend application, and tolerances match TPFA ILU.
    """
    return _build_ilu(matrix, shape, xp)


def build_corrected_block_ilu(matrix, shape, xp):
    """Fixed lower-block ILU for corrected order-1 moment-major U states.

    Slice the unscaled corrected matrix in the actual ``[U0, U1]`` layout
    (scalar flux is ``U0 - 2/3 U1``). Apply ``z0=B00(r0)`` followed by
    ``z1=B11(r1-A10 z0)``. The full sparse A10 includes off-cell Marshak
    coupling. Each diagonal block uses the scalar ILU tolerances below.
    """
    start = perf_counter()
    shape = tuple(shape)
    if len(shape) < 2 or shape[0] != 2 or any(n <= 0 for n in shape):
        raise ValueError("corrected block ILU requires a two-moment state shape")
    n = int(np.prod(shape[1:]))
    if matrix.shape != (2*n, 2*n):
        raise ValueError("ILU matrix dimensions must match the state shape")
    if xp is not np and getattr(xp, "__name__", None) != "cupy":
        raise TypeError("corrected block ILU backend must be NumPy or CuPy")
    matrix = matrix.tocsr()
    blocks = []
    for moment in range(2):
        sl = slice(moment*n, (moment+1)*n)
        try:
            blocks.append(_build_ilu(matrix[sl, sl], shape[1:], xp))
        except RuntimeError as exc:
            raise RuntimeError(f"corrected block ILU A{moment}{moment}: {exc}") from exc
    a10 = matrix[n:, :n].tocsr()
    if xp is not np:
        from cupyx.scipy import sparse
        a10 = sparse.csr_matrix(a10)

    def precondition(r):
        if not isinstance(r, xp.ndarray):
            raise TypeError("ILU residual must use the configured array backend")
        if r.shape != shape:
            raise ValueError(f"ILU residual shape {r.shape} != {shape}")
        z0 = blocks[0](r[0])
        z1 = blocks[1](r[1] - (a10 @ z0.reshape(-1)).reshape(shape[1:]))
        return xp.stack((z0, z1))

    precondition.setup_seconds = perf_counter() - start
    precondition.nnz = sum(block.nnz for block in blocks)
    precondition.nnz_kind = blocks[0].nnz_kind
    precondition.matrix_nnz = matrix.nnz
    precondition.coupling_nnz = a10.nnz
    precondition.shape = shape
    precondition.block_stats = tuple(
        dict(block=f"A{i}{i}", setup_seconds=block.setup_seconds,
             matrix_nnz=block.matrix_nnz, nnz=block.nnz, nnz_kind=block.nnz_kind)
        for i, block in enumerate(blocks))
    return precondition


def _build_ilu(matrix, shape, xp):
    """Factor a host sparse matrix once and preserve its state layout."""
    from scipy.sparse.linalg import spilu

    start = perf_counter()
    shape = tuple(shape)
    if matrix.shape != (int(np.prod(shape)),) * 2:
        raise ValueError("ILU matrix dimensions must match the state shape")
    lu = spilu(matrix.tocsc(), drop_tol=1e-6, fill_factor=20)
    if xp is np:
        def solve(r):
            return lu.solve(r.reshape(-1)).reshape(shape)
    else:
        from cupyx.scipy import sparse
        from cupyx.scipy.sparse.linalg import spsolve_triangular

        lower, upper = sparse.csr_matrix(lu.L), sparse.csr_matrix(lu.U)
        # Pr A Pc = L U: Pr b = b[argsort(perm_r)], x = y[perm_c].
        row_inverse = xp.asarray(np.argsort(lu.perm_r))
        columns = xp.asarray(lu.perm_c)

        def solve(r):
            rhs = r.reshape(-1)[row_inverse]
            y = spsolve_triangular(lower, rhs, lower=True, unit_diagonal=True)
            z = spsolve_triangular(upper, y, lower=False)
            return z[columns].reshape(shape)

        # Fail explicitly during setup when triangular solves are unsupported.
        solve(xp.zeros(shape, dtype=matrix.dtype))

    def precondition(r):
        if not isinstance(r, xp.ndarray):
            raise TypeError("ILU residual must use the configured array backend")
        if r.shape != shape:
            raise ValueError(f"ILU residual shape {r.shape} != {shape}")
        return solve(r)

    precondition.setup_seconds = perf_counter() - start
    # Accessing L/U materializes explicit CSC copies, which CPU lu.solve does
    # not need. On large problems even statistics alone can exhaust memory.
    # SuperLU's reported count can differ from the extracted CSC entry count.
    precondition.nnz = int(lu.nnz)
    precondition.nnz_kind = "superlu_reported_not_explicit_csc_count"
    precondition.matrix_nnz = matrix.nnz
    precondition.shape = shape
    return precondition
