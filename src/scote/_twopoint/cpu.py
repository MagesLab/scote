"""CPU backend of the shell products P_k = X^T S_k X of one lattice level.

Strategies (:class:`scote.CPUOptions` ``mode``), all exact for integer counts (every partial sum is an
integer below 2^53, so the results are bit-identical across strategies, thread counts, and the GPU backend):

* ``"numba"``: a fused multi-threaded kernel on the CSR matrices X and S_k with one dense (G x G) float64 accumulator
  per thread (n_threads * G^2 * 8 bytes; above ``CPUOptions.numba_acc_bytes`` the level falls back to ``"sparse"``);
* ``"dense"``: X densified, Y = S_k X as a parallel sparse x dense product, P_k = X^T Y by BLAS;
* ``"sparse"``: two SciPy sparse x sparse products in row blocks (the reference implementation);
* ``"auto"`` (default): ``"dense"`` where the density nnz(X) / (n_cells G) exceeds ``dense_threshold`` (5 %),
  else ``"numba"``.

The numba kernels run on numba's current thread count (set per call by the public functions of ``scote``).
"""
from __future__ import annotations

import time
import warnings
from typing import Callable

import numba
import numpy as np
import scipy.sparse as sp
from numba import njit, prange

from ._common import CPUOptions, LevelTiming
from .lattice import HOST_MIN_PARALLEL_WORK, Lattice, shell_offsets, shell_operator

__all__ = ["NUMBA_ACC_BYTES_LIMIT", "numba_mode_available", "pair_counts_level"]

# Per-thread dense accumulators of mode "numba" cost n_threads * G^2 * 8 bytes; above this limit a level falls back to
# mode "sparse" with a warning (G = 18,000 at 4 threads needs 10.4 GB; G = 10,000 needs 3.2 GB).
NUMBA_ACC_BYTES_LIMIT = 8 * 2**30         # default of CPUOptions.numba_acc_bytes

# Bytes of the product S_k X of mode "dense" formed at once for consecutive row groups (at least one group)
DENSE_BATCH_BYTES = 64 * 2**20

# Multiply-adds below which the serial kernel runs (about 0.1 s of serial work; a parallel-region launch costs
# 0.1-100 ms depending on the load of the machine).
NUMBA_MIN_PARALLEL_WORK = 2e7


# ----------------------------------------------------------------------------- numba kernels
@njit(cache=True, nogil=True)
def _shell_pairs_worker(X_indptr, X_indices, X_data, S_indptr, S_indices, rows, P_t, y_t, mark_t, touched_t,
                        t, n_threads, chunk):
    """Rows of worker t: P_t += x_i^T (sum_{j in S[i]} x_j) for its rows i.

    The rows are split into chunks of ``chunk`` rows and dealt round-robin (chunk c -> worker c % n_threads: a
    static, interleaved schedule that is balanced for spatially ordered rows and needs neither thread ids nor
    atomics). For each row i, y = sum_{j in S[i]} x_j is formed in the dense scratch y_t (touched indices tracked for
    a cheap reset), then P_t[a, b] += x_ia y_b over the non-zeros a of x_i and the touched b. Exact for
    integer-valued X (every partial sum and product is an integer below 2^53), hence bit-identical to mode "sparse"
    for any thread count and summation order.

    Parameters
    ----------
    X_indptr, X_indices, X_data : numpy.ndarray
        CSR arrays of X (n_cells x G).
    S_indptr, S_indices : numpy.ndarray
        CSR arrays of the shell operator S (n_cells x n_cells; its values are 1).
    rows : numpy.ndarray
        Rows i of the product (int64).
    P_t : numpy.ndarray
        (G, G) float64 accumulator of the worker, updated in place.
    y_t : numpy.ndarray
        (G,) float64 scratch, zero on entry and exit.
    mark_t : numpy.ndarray
        (G,) uint8 scratch, zero on entry and exit.
    touched_t : numpy.ndarray
        (G,) int64 scratch.
    t : int
        Worker index.
    n_threads : int
        Number of workers.
    chunk : int
        Rows per chunk.

    Returns
    -------
    float
        Number of multiply-adds, sum_i nnz(x_i) |touched(y_i)|.
    """
    n = len(rows)
    G = y_t.shape[0]
    n_chunks = (n + chunk - 1) // chunk
    n_upd = 0.0
    for c in range(t, n_chunks, n_threads):
        for ii in range(c * chunk, min(n, (c + 1) * chunk)):
            i = rows[ii]
            nt = 0
            for p in range(S_indptr[i], S_indptr[i + 1]):
                j = S_indices[p]
                for q in range(X_indptr[j], X_indptr[j + 1]):
                    b = X_indices[q]
                    if mark_t[b] == 0:
                        mark_t[b] = 1
                        touched_t[nt] = b
                        nt += 1
                    y_t[b] += X_data[q]
            if nt == 0:
                continue
            nx = X_indptr[i + 1] - X_indptr[i]
            if 2 * nt > G:                          # y mostly full: contiguous axpy over all b (x * 0 == 0 exactly)
                for q in range(X_indptr[i], X_indptr[i + 1]):
                    row = P_t[X_indices[q]]
                    xa = X_data[q]
                    for b in range(G):
                        row[b] += xa * y_t[b]
            else:                                   # gather over the touched b only
                for q in range(X_indptr[i], X_indptr[i + 1]):
                    row = P_t[X_indices[q]]
                    xa = X_data[q]
                    for s in range(nt):
                        b = touched_t[s]
                        row[b] += xa * y_t[b]
            n_upd += nx * nt
            for s in range(nt):                     # reset only what was touched
                b = touched_t[s]
                y_t[b] = 0.0
                mark_t[b] = 0
    return n_upd

@njit(parallel=True, cache=True, nogil=True)
def _shell_pairs_parallel(X_indptr, X_indices, X_data, S_indptr, S_indices, rows, G, n_threads, chunk):
    """P = X[rows]^T (S[rows, :] X) with n_threads workers, each with its own (G, G) accumulator
    (n_threads G^2 8 bytes), reduced in parallel at the end.

    Parameters
    ----------
    X_indptr, X_indices, X_data : numpy.ndarray
        CSR arrays of X.
    S_indptr, S_indices : numpy.ndarray
        CSR arrays of the shell operator.
    rows : numpy.ndarray
        Rows of the product.
    G : int
        Number of features.
    n_threads : int
        Number of workers.
    chunk : int
        Rows per chunk of the round-robin schedule.

    Returns
    -------
    P : numpy.ndarray
        (G, G) float64 counts.
    updates : float
        Number of multiply-adds.
    """
    Pt = np.zeros((n_threads, G, G))
    y = np.zeros((n_threads, G))
    mark = np.zeros((n_threads, G), dtype=np.uint8)
    touched = np.zeros((n_threads, G), dtype=np.int64)
    upd = np.zeros(n_threads)
    for t in prange(n_threads):
        upd[t] = _shell_pairs_worker(X_indptr, X_indices, X_data, S_indptr, S_indices, rows,
                                     Pt[t], y[t], mark[t], touched[t], t, n_threads, chunk)
    P = np.empty((G, G))
    for a in prange(G):
        for b in range(G):
            acc = 0.0
            for t in range(n_threads):
                acc += Pt[t, a, b]
            P[a, b] = acc
    return P, upd.sum()

@njit(cache=True, nogil=True)
def _shell_pairs_serial(X_indptr, X_indices, X_data, S_indptr, S_indices, rows, G):
    """Serial variant of ``_shell_pairs_parallel`` for small problems (a parallel-region launch costs 0.1-100 ms
    depending on the load of the machine; the per-tile path issues thousands of small calls). Same arithmetic.

    Parameters
    ----------
    X_indptr, X_indices, X_data : numpy.ndarray
        CSR arrays of X.
    S_indptr, S_indices : numpy.ndarray
        CSR arrays of the shell operator.
    rows : numpy.ndarray
        Rows of the product.
    G : int
        Number of features.

    Returns
    -------
    P : numpy.ndarray
        (G, G) float64 counts.
    updates : float
        Number of multiply-adds.
    """
    P = np.zeros((G, G))
    y = np.zeros(G)
    mark = np.zeros(G, dtype=np.uint8)
    touched = np.zeros(G, dtype=np.int64)
    upd = _shell_pairs_worker(X_indptr, X_indices, X_data, S_indptr, S_indices, rows, P, y, mark, touched,
                              0, 1, len(rows) + 1)
    return P, upd

@njit(cache=True, nogil=True)
def _spmm_rows(S_indptr, S_indices, S_data, Xd, out, i0, i1):
    """out[i0:i1] = S[i0:i1] @ Xd; each row sums its entries in CSR order like SciPy's csr_matvecs, so the result is
    bit-identical to ``S @ Xd`` for any float data.

    Parameters
    ----------
    S_indptr, S_indices, S_data : numpy.ndarray
        CSR arrays of S.
    Xd : numpy.ndarray
        Dense C-contiguous float64 matrix.
    out : numpy.ndarray
        Output, rows i0 .. i1 - 1 overwritten.
    i0, i1 : int
        Row range.
    """
    G = out.shape[1]
    for i in range(i0, i1):
        row = out[i]
        for b in range(G):
            row[b] = 0.0
        for p in range(S_indptr[i], S_indptr[i + 1]):
            xr = Xd[S_indices[p]]
            v = S_data[p]
            for b in range(G):
                row[b] += v * xr[b]

@njit(parallel=True, cache=True, nogil=True)
def _spmm_par(S_indptr, S_indices, S_data, Xd, out, chunk):
    """out = S @ Xd in parallel row chunks.

    Parameters
    ----------
    S_indptr, S_indices, S_data : numpy.ndarray
        CSR arrays of S.
    Xd : numpy.ndarray
        Dense C-contiguous float64 matrix.
    out : numpy.ndarray
        Output (rows of S, columns of Xd), overwritten.
    chunk : int
        Rows per chunk.
    """
    n = out.shape[0]
    for c in prange((n + chunk - 1) // chunk):
        _spmm_rows(S_indptr, S_indices, S_data, Xd, out, c * chunk, min(n, (c + 1) * chunk))

@njit(cache=True, nogil=True)
def _spmm_ser(S_indptr, S_indices, S_data, Xd, out):
    """Serial variant of ``_spmm_par`` (same arguments without ``chunk``)."""
    _spmm_rows(S_indptr, S_indices, S_data, Xd, out, 0, out.shape[0])

def _spmm(S: sp.csr_matrix, Xd: np.ndarray, host: str = "numba") -> np.ndarray:
    """Y = S @ Xd with the parallel numba kernel or SciPy.

    Both sum each row in CSR order, so the results are bit-identical.

    Parameters
    ----------
    S : scipy.sparse.csr_matrix
        Sparse left factor.
    Xd : numpy.ndarray
        Dense right factor.
    host : {"numba", "numpy"}
        numba kernel or SciPy.

    Returns
    -------
    numpy.ndarray
        Dense float64 product.
    """
    if host != "numba" or S.shape[0] == 0:
        return S @ Xd
    S = S.tocsr()
    Xd = np.ascontiguousarray(Xd, dtype=np.float64)
    out = np.empty((S.shape[0], Xd.shape[1]), dtype=np.float64)
    data = S.data if S.data.dtype == np.float64 else S.data.astype(np.float64)
    n_threads = numba.get_num_threads()
    if n_threads > 1 and S.nnz * Xd.shape[1] >= HOST_MIN_PARALLEL_WORK:
        _spmm_par(S.indptr, S.indices, data, Xd, out, int(max(256, S.shape[0] // (16 * n_threads) + 1)))
    else:
        _spmm_ser(S.indptr, S.indices, data, Xd, out)
    return out


def numba_mode_available(n_features: int, n_threads: int | None = None, limit: int | None = None) -> bool:
    """Whether the per-thread accumulators of mode "numba" fit the memory limit.

    Parameters
    ----------
    n_features : int
        Number of features G.
    n_threads : int or None
        Thread count; ``None`` uses numba's current thread count.
    limit : int or None
        Limit in bytes (``CPUOptions.numba_acc_bytes``); ``None``: NUMBA_ACC_BYTES_LIMIT.

    Returns
    -------
    bool
        ``n_threads * G^2 * 8 <= limit``.
    """
    n_threads = numba.get_num_threads() if n_threads is None else n_threads
    limit = NUMBA_ACC_BYTES_LIMIT if limit is None else limit
    return n_threads * n_features * n_features * 8 <= limit


def _numba_chunk(n_rows: int, n_threads: int, target_chunks_per_thread: int = 64) -> int:
    """Rows per chunk of the round-robin schedule: small enough for load balance, large enough to amortize overhead.

    Parameters
    ----------
    n_rows : int
        Rows of the product.
    n_threads : int
        Number of workers.
    target_chunks_per_thread : int
        Chunks per worker aimed at.

    Returns
    -------
    int
        Rows per chunk, between 16 and 1024.
    """
    return int(max(16, min(1024, n_rows // max(1, n_threads * target_chunks_per_thread) + 1)))


def _numba_shell_product(X: sp.csr_matrix, S: sp.csr_matrix, rows: np.ndarray | None):
    """P = X[rows]^T S[rows, :] X with the numba kernels, without materializing the row slices.

    Small problems (estimated multiply-adds below NUMBA_MIN_PARALLEL_WORK) or one thread run the serial kernel.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix
        (n_cells, G) counts of all cells.
    S : scipy.sparse.csr_matrix
        (n_cells, n_cells) shell operator.
    rows : numpy.ndarray or None
        Rows of the product; ``None`` for all rows.

    Returns
    -------
    P : numpy.ndarray
        (G, G) float64 counts.
    updates : float
        Number of multiply-adds.
    """
    G = X.shape[1]
    n_threads = numba.get_num_threads()
    Xd = X.data if X.data.dtype == np.float64 else X.data.astype(np.float64)
    if rows is None:
        rows = np.arange(X.shape[0], dtype=np.int64)
        nnz_S_rows = S.nnz
    else:
        rows = np.asarray(rows, dtype=np.int64)
        nnz_S_rows = int((S.indptr[rows + 1] - S.indptr[rows]).sum())
    mean_nnz_x = X.nnz / max(1, X.shape[0])
    touched = min(G, mean_nnz_x * nnz_S_rows / max(1, len(rows)))            # about |touched(y_i)| per row
    work = nnz_S_rows * mean_nnz_x + len(rows) * mean_nnz_x * touched       # gathers into y + P updates
    if n_threads == 1 or work < NUMBA_MIN_PARALLEL_WORK:
        P, upd = _shell_pairs_serial(X.indptr, X.indices, Xd, S.indptr, S.indices, rows, G)
    else:
        chunk = _numba_chunk(len(rows), n_threads)
        P, upd = _shell_pairs_parallel(X.indptr, X.indices, Xd, S.indptr, S.indices, rows, G, n_threads, chunk)
    return P, float(upd)


def pair_counts_level(X: sp.csr_matrix, lat: Lattice, n_shells: int, shells: list[int] | None = None, *,
                      include_intra: bool = False, row_groups: np.ndarray | None = None, sink: Callable | None = None,
                      totals_only: bool = False, timing: LevelTiming | None = None,
                      options: CPUOptions | None = None) -> np.ndarray | None:
    """Ordered pair counts of the requested shells of one lattice level on the CPU.

    The GPU backend, ``scote._twopoint.gpu.pair_counts_level``, has the same arguments (with
    :class:`scote.GPUOptions`) and returns bit-identical counts.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix
        (n_cells, G) counts of the occupied cells, rows in lattice order.
    lat : Lattice
        The lattice of X.
    n_shells : int
        Number of shells of the level (offsets from ``scote._twopoint.lattice.shell_offsets``).
    shells : list of int or None
        Shells to count; ``None`` counts all shells 0 .. n_shells - 1.
    include_intra : bool
        Count shell 0 as the pairs of distinct molecules within one cell, x_a x_b - delta_ab x_a; otherwise shell 0
        stays zero.
    row_groups : numpy.ndarray or None
        (n_cells,) group id per cell (the spatial tile). The pairs are then attributed to the group of their first
        cell and returned per group.
    sink : callable or None
        Receives every (G, G) float64 host array instead of storing it: ``sink(i, slab)`` without row groups,
        ``sink(i, g, slab)`` with row groups (``i`` = position in ``shells``, ``g`` = group index in the order of
        ``np.unique(row_groups)``). Slabs arrive shells outer, groups inner, each (i, g) once.
    totals_only : bool
        Without row groups: return only the per-shell totals sum_ab P_k.
    timing : LevelTiming or None
        Receives the time spent in the shell operators and the two products.
    options : CPUOptions or None
        Options of the CPU backend (:class:`scote.CPUOptions`); ``None`` uses the defaults.

    Returns
    -------
    numpy.ndarray or None
        Without row groups: (len(shells), G, G) float64 counts, or the (len(shells),) totals with ``sink`` or
        ``totals_only``. With row groups: (n_groups, len(shells), G, G), or ``None`` with ``sink``.

    Raises
    ------
    ValueError
        For ``totals_only`` with row groups.
    """
    opt = CPUOptions() if options is None else options
    G = X.shape[1]
    shells = list(range(n_shells)) if shells is None else list(shells)
    if row_groups is None:
        P = _pair_counts_rows(X, lat, n_shells, opt, shells, include_intra, timing, None)
        if sink is None and not totals_only:
            return P
        if sink is not None:
            for i in range(P.shape[0]):
                sink(i, P[i])
        return P.sum(axis=(1, 2))
    if totals_only:
        raise ValueError("totals_only applies only without row_groups")
    # shells outer, groups inner: only the current shell operator S_k is alive (all shells of a level would need
    # 12 GB at 5.3 M cells); the products are the same calls in the same row order as a group-outer loop
    row_groups = np.asarray(row_groups)
    groups, inv = np.unique(row_groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(groups) + 1))
    rows_of = [order[bounds[g]:bounds[g + 1]] for g in range(len(groups))]     # ascending row indices per group
    out = None if sink is not None else np.zeros((len(groups), len(shells), G, G), dtype=np.float64)
    S_cache: dict = {}                                       # shell 0 (identity) stays; S_k only while shell k runs
    density = X.nnz / max(1, X.shape[0] * G)
    dense = opt.mode == "dense" or (opt.mode == "auto" and density > opt.dense_threshold)
    X_dense = X.toarray().astype(np.float64) if dense else None
    for i, k in enumerate(shells):
        if dense and k != 0:
            slabs = _dense_group_slabs(X_dense, lat, n_shells, opt, k, rows_of, timing)
        else:
            slabs = (_pair_counts_rows(X, lat, n_shells, opt, [k], include_intra, timing, rows_of[gi],
                                       S_cache=S_cache, X_dense=X_dense)[0] for gi in range(len(groups)))
        for gi, slab in enumerate(slabs):
            if sink is None:
                out[gi, i] = slab
            else:
                sink(i, gi, slab)
            del slab
        if k != 0:
            S_cache.pop(k, None)
    if dense and timing is not None:
        timing.mode = "dense"
        timing.n_cells, timing.nnz_X, timing.density_X = X.shape[0], int(X.nnz), density
    return out


def _dense_group_slabs(X_dense: np.ndarray, lat: Lattice, n_shells: int, opt: CPUOptions, k: int,
                       rows_of: list[np.ndarray], timing: LevelTiming | None):
    """Slabs X[rows]^T S_k[rows, :] X of shell k >= 1 for every row group, mode "dense".

    The product Y = S_k[rows, :] X is formed for consecutive groups together, at most ``DENSE_BATCH_BYTES`` of Y (or
    one group) at a time, so that the parallel regions do not multiply with the number of groups. Every row of Y is
    computed on its own (``_spmm``), so the slabs equal those of the groups computed one by one.

    Parameters
    ----------
    X_dense : numpy.ndarray
        (n_cells, G) dense float64 counts.
    lat : Lattice
        The lattice of X.
    n_shells : int
        Number of shells of the level.
    opt : CPUOptions
        Backend options.
    k : int
        The shell, at least 1.
    rows_of : list of numpy.ndarray
        Ascending rows of every group.
    timing : LevelTiming or None
        Receives the work of the shell, one record per group.

    Yields
    ------
    numpy.ndarray
        (G, G) float64 slab of each group, in the order of ``rows_of``.
    """
    G = X_dense.shape[1]
    t0 = time.perf_counter()
    S = shell_operator(lat, shell_offsets(n_shells, lat.cells.shape[1])[k], host=opt.host)
    t_S = time.perf_counter() - t0
    per_batch_rows = max(1, int(DENSE_BATCH_BYTES // (8 * max(1, G))))
    g0 = 0
    while g0 < len(rows_of):
        g1 = g0 + 1
        n_rows = len(rows_of[g0])
        while g1 < len(rows_of) and n_rows + len(rows_of[g1]) <= per_batch_rows:
            n_rows += len(rows_of[g1])
            g1 += 1
        rows = np.concatenate(rows_of[g0:g1])
        t1 = time.perf_counter()
        S_rows = S[rows]
        Y = _spmm(S_rows, X_dense, host=opt.host)
        t_SX = time.perf_counter() - t1
        nnz = np.diff(S_rows.indptr)
        start = 0
        for g in range(g0, g1):
            n = len(rows_of[g])
            t2 = time.perf_counter()
            slab = X_dense[rows_of[g]].T @ Y[start:start + n]
            t3 = time.perf_counter()
            if timing is not None:
                nnz_S = int(nnz[start:start + n].sum())
                share = n / max(1, len(rows))
                timing.add_shell(k, nnz_S, t_S if g == 0 else 0.0, t_SX * share, t3 - t2, float(nnz_S) * G * G)
            start += n
            yield slab
        g0 = g1


def _pair_counts_rows(X: sp.csr_matrix, lat: Lattice, n_shells: int, opt: CPUOptions, shells: list[int],
                      include_intra: bool, timing: LevelTiming | None, rows: np.ndarray | None,
                      S_cache: dict | None = None, X_dense: np.ndarray | None = None) -> np.ndarray:
    """Pair counts P[i] = X[rows]^T S_k[rows, :] X of the requested shells k = shells[i].

    Parameters
    ----------
    X : scipy.sparse.csr_matrix
        (n_cells, G) counts.
    lat : Lattice
        The lattice of X.
    n_shells : int
        Number of shells of the level.
    opt : CPUOptions
        Backend options.
    shells : list of int
        Shells to count.
    include_intra : bool
        Count shell 0 as the pairs of distinct molecules within one cell.
    timing : LevelTiming or None
        Receives the work of every shell.
    rows : numpy.ndarray or None
        Row subset (the cells of one group, ascending); ``None`` for all rows.
    S_cache : dict or None
        Shell index -> S_k, filled on first use and shared across row groups (shell 0 holds the identity).
    X_dense : numpy.ndarray or None
        Dense float64 X, if already formed.

    Returns
    -------
    numpy.ndarray
        (len(shells), G, G) float64 counts.
    """
    G = X.shape[1]
    offsets = shell_offsets(n_shells, lat.cells.shape[1])
    density = X.nnz / max(1, X.shape[0] * G)
    mode = opt.mode
    if mode == "auto":                       # numba kernel for the sparse fine levels, BLAS where X is dense
        mode = "dense" if density > opt.dense_threshold else "numba"
    limit = opt.numba_acc_bytes
    if mode == "numba" and not numba_mode_available(G, limit=limit):
        warnings.warn(f"mode 'numba': per-thread accumulators {numba.get_num_threads()} x {G}^2 x 8 B exceed "
                      f"CPUOptions.numba_acc_bytes = {limit / 2**30:.1f} GiB; falling back to mode 'sparse'",
                      RuntimeWarning, stacklevel=3)
        mode = "sparse"
    P = np.zeros((len(shells), G, G), dtype=np.float64)      # one slab per requested shell
    slot = {k: i for i, k in enumerate(shells)}
    if mode == "dense" and X_dense is None:
        X_dense = X.toarray().astype(np.float64)
    elif mode != "dense":
        X_dense = None
    X_rows = X if (rows is None or mode == "numba") else X[rows]   # numba: the kernel indexes X by `rows` (no copy)
    X_dense_rows = X_dense if (X_dense is None or rows is None) else X_dense[rows]
    for k in shells:
        if k == 0:
            if include_intra:  # intra-cell molecule pairs: x_a x_b - delta_ab x_a, kept sparse (never densify X here)
                if X_dense is not None:
                    P0 = X_dense_rows.T @ X_dense_rows
                elif mode == "numba":   # S = identity (cached under shell 0, which never holds a shell operator)
                    if S_cache is not None and 0 in S_cache:
                        identity = S_cache[0]
                    else:
                        identity = sp.identity(X.shape[0], dtype=np.float64, format="csr")
                        if S_cache is not None:
                            S_cache[0] = identity
                    P0, _ = _numba_shell_product(X, identity, rows)
                else:
                    P0 = (X_rows.T @ X_rows)
                    P0 = P0.toarray() if sp.issparse(P0) else P0
                col_sums = X_rows.sum(0) if (rows is None or mode != "numba") else X[rows].sum(0)
                P[slot[0]] = P0 - np.diag(np.asarray(col_sums).ravel())
            continue
        t0 = time.perf_counter()
        if S_cache is not None and k in S_cache:
            S = S_cache[k]
        else:
            S = shell_operator(lat, offsets[k], host=opt.host)
            if S_cache is not None:
                S_cache[k] = S
        t1 = time.perf_counter()
        if mode == "numba":                 # fused kernel on S and X directly (row subset passed as indices)
            t2 = t1
            Pk, updates = _numba_shell_product(X, S, rows)
            P[slot[k]] += Pk
            t3 = time.perf_counter()
        elif mode == "dense":
            S_rows = S if rows is None else S[rows]
            Y = _spmm(S_rows, X_dense, host=opt.host)
            t2 = time.perf_counter()
            P[slot[k]] = X_dense_rows.T @ Y
            t3 = time.perf_counter()
            updates = float(S_rows.nnz) * G * G
        else:
            S_rows = S if rows is None else S[rows]
            n = X_rows.shape[0]
            updates = 0.0
            t_sx = 0.0
            t_xtsx = 0.0
            XT = X_rows.T.tocsr()
            for b0 in range(0, n, opt.row_block):
                b1 = min(n, b0 + opt.row_block)
                ta = time.perf_counter()
                Y = (S_rows[b0:b1] @ X).tocsr()
                tb = time.perf_counter()
                Pk = XT[:, b0:b1] @ Y
                P[slot[k]] += Pk.toarray() if sp.issparse(Pk) else Pk
                tc = time.perf_counter()
                t_sx += tb - ta
                t_xtsx += tc - tb
                nx = np.diff(X_rows.indptr[b0:b1 + 1]).astype(np.float64)
                ny = np.diff(Y.indptr).astype(np.float64)
                updates += float((nx * ny).sum())
            t2 = t1 + t_sx
            t3 = t2 + t_xtsx
        if timing is not None:
            nnz_S = S.nnz if rows is None else int((S.indptr[rows + 1] - S.indptr[rows]).sum())
            timing.add_shell(k, nnz_S, t1 - t0, t2 - t1, t3 - t2, updates)
    if timing is not None:
        timing.mode = mode
        timing.n_cells, timing.nnz_X, timing.density_X = X.shape[0], int(X.nnz), density
    return P
