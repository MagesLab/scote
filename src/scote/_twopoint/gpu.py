"""GPU backend (PyTorch) of the shell products P_k = X^T S_k X of one lattice level.

All products run in float64 and are exact for integer counts, hence bit-identical to the CPU backend
(``scote._twopoint.cpu``), whose ``scote._twopoint.cpu.pair_counts_level`` has the same arguments. The shell
operators S_k are built on the device from the sorted cell hashes of the lattice. The same code runs with
``GPUOptions(device="cpu")`` (used by the tests).

Strategy per level (X: n_cells x G CSR counts):

* semidense (fine levels): per row block b, Y_b = S_k[b, :] X (sparse x sparse, densified) and
  P_k += X[b]^T Y_b (sparse x dense), i.e. n_offsets(shell) nnz(X) + nnz(X) G multiply-adds per shell; a block
  that runs out of device memory is split (global counts) or the row blocks are halved and the shell restarts
  (per-group counts);
* dense (coarse levels, where 2 n_cells G 8 bytes fits ``dense_bytes_limit`` and 30 % of the device memory budget):
  Y = S_k X (sparse x dense), P_k = X^T Y (GEMM).

The environment variable ``SCOTE_CUDA_MEM_FRACTION`` (e.g. ``0.6`` on a card shared with another job) limits the
memory this process uses on the device it computes on; it is applied once per process and device.
"""
from __future__ import annotations

import os
import time
import warnings
from typing import Callable

import numpy as np
import scipy.sparse as sp

from ._common import GPUOptions, LevelTiming
from .lattice import Lattice, shell_offsets

__all__ = ["pair_counts_level", "shell_operator_rows"]


def _torch():
    """The torch module, imported on first use.

    Returns
    -------
    module
        ``torch``.

    Raises
    ------
    ImportError
        If torch is not installed (install the ``gpu`` extra, ``pip install "scote[gpu]"``).
    """
    try:
        import torch
    except ImportError as err:
        raise ImportError('backend "gpu" needs PyTorch: pip install "scote[gpu]"') from err
    return torch


def _csr_tensor(crow, col, values, size, device):
    """``torch.sparse_csr_tensor`` without PyTorch's warnings that sparse CSR support is in beta and that the
    invariant checks are off (the indices are built sorted and in range).

    Parameters
    ----------
    crow, col, values : torch.Tensor
        Row pointers, column indices, and values.
    size : tuple of int
        Shape of the matrix.
    device : str or torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        The sparse CSR tensor.
    """
    torch = _torch()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state", category=UserWarning)
        return torch.sparse_csr_tensor(crow, col, values, size=size, device=device, check_invariants=False)


def _to_torch_csr(M: sp.spmatrix, device, dtype):
    """A SciPy matrix as a torch sparse CSR tensor.

    Parameters
    ----------
    M : scipy.sparse matrix
        The matrix (converted to CSR with sorted indices).
    device : str or torch.device
        Target device.
    dtype : torch.dtype
        Value dtype.

    Returns
    -------
    torch.Tensor
        Sparse CSR tensor with int64 indices.
    """
    torch = _torch()
    M = sp.csr_matrix(M)
    M.sort_indices()
    return _csr_tensor(torch.from_numpy(M.indptr.astype(np.int64)), torch.from_numpy(M.indices.astype(np.int64)),
                       torch.from_numpy(np.ascontiguousarray(M.data)).to(dtype), M.shape, device)


def _is_cuda(device) -> bool:
    """Whether a device names a CUDA device.

    Parameters
    ----------
    device : str or torch.device
        The device.

    Returns
    -------
    bool
        ``True`` for "cuda" and "cuda:N".
    """
    return str(device).startswith("cuda")


def _sync(device) -> None:
    """Wait for the queued work of a CUDA device, for attributable timings (no-op on other devices).

    Parameters
    ----------
    device : str or torch.device
        The device.
    """
    if _is_cuda(device):
        _torch().cuda.synchronize(device)


def _empty_cache(device) -> None:
    """Release the cached blocks of the CUDA allocator (no-op on other devices).

    Parameters
    ----------
    device : str or torch.device
        The device.
    """
    if _is_cuda(device):
        _torch().cuda.empty_cache()


_CONFIGURED_DEVICES: set = set()


def _configure_cuda(device) -> None:
    """Apply ``SCOTE_CUDA_MEM_FRACTION`` to a CUDA device, once per process and device.

    Parameters
    ----------
    device : str or torch.device
        The device the products run on ("cuda" means the current CUDA device).
    """
    if not _is_cuda(device):
        return
    torch = _torch()
    dev = torch.device(device)
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    if dev in _CONFIGURED_DEVICES:
        return
    _CONFIGURED_DEVICES.add(dev)
    frac = os.environ.get("SCOTE_CUDA_MEM_FRACTION")
    if frac:
        torch.cuda.set_per_process_memory_fraction(float(frac), device=dev)


def _device_budget_bytes(device) -> int | None:
    """Device memory this process may use: the total memory times ``SCOTE_CUDA_MEM_FRACTION``.

    Parameters
    ----------
    device : str or torch.device
        The device.

    Returns
    -------
    int or None
        Bytes, or ``None`` off CUDA.
    """
    if not _is_cuda(device):
        return None
    torch = _torch()
    frac = float(os.environ.get("SCOTE_CUDA_MEM_FRACTION") or 1.0)
    return int(torch.cuda.get_device_properties(torch.device(device)).total_memory * frac)


def _peak_mb(device) -> float | None:
    """Peak allocated memory of a CUDA device.

    Parameters
    ----------
    device : str or torch.device
        The device.

    Returns
    -------
    float or None
        MB, or ``None`` off CUDA.
    """
    return _torch().cuda.max_memory_allocated(device) / 2**20 if _is_cuda(device) else None


def _effective_dense_limit(device, dense_bytes_limit: int, share: float = 0.3) -> int:
    """Byte limit of the dense path of a level.

    The dense path holds X, Y (2 n G 8 bytes), a permuted copy of X, and cuSPARSE temporaries and has no
    out-of-memory fallback: it is allowed only when 2 n G 8 bytes fits ``share`` of the device memory budget.

    Parameters
    ----------
    device : str or torch.device
        The device.
    dense_bytes_limit : int
        The limit of the options.
    share : float
        Share of the device memory budget.

    Returns
    -------
    int
        The limit in bytes.
    """
    budget = _device_budget_bytes(device)
    return dense_bytes_limit if budget is None else min(dense_bytes_limit, int(share * budget))


def shell_operator_rows(lat: Lattice, offsets: np.ndarray, device, rows=None, cache: dict | None = None):
    """Rows of the shell operator S of a set of offsets, built on the device.

    For every offset the neighbor row is found by a binary search in the sorted cell hashes (the semantics of
    ``scote._twopoint.lattice.shell_operator``).

    Parameters
    ----------
    lat : Lattice
        The lattice.
    offsets : numpy.ndarray
        (m, ndim) integer offsets.
    device : str or torch.device
        Torch device.
    rows : None, tuple of int, or torch.Tensor
        ``None`` (all rows), a contiguous block ``(b0, b1)``, or a device int64 index tensor (rows in that order).
    cache : dict or None
        Keeps the device copies of the lattice arrays across shells and blocks of one level.

    Returns
    -------
    torch.Tensor
        Sparse CSR float64 tensor (len(rows), n_cells) with sorted column indices per row.
    """
    torch = _torch()
    n = lat.n_cells
    if cache is None or "hashes" not in cache:
        cache = {} if cache is None else cache
        cache["hashes"] = torch.from_numpy(lat.hashes).to(device)
        cache["cells"] = torch.from_numpy(lat.cells).to(device)
        cache["origin"] = torch.from_numpy(np.asarray(lat.origin, dtype=np.int64)).to(device)
        cache["extent"] = torch.from_numpy(np.asarray(lat.extent, dtype=np.int64)).to(device)
    hashes, cells, origin, extent = cache["hashes"], cache["cells"], cache["origin"], cache["extent"]
    if rows is None:
        c = cells
    elif isinstance(rows, tuple):
        c = cells[rows[0]:rows[1]]
    else:
        c = cells[rows]
    nb = c.shape[0]
    offs = torch.from_numpy(np.array(offsets, dtype=np.int64)).to(device)             # (m, ndim)
    ndim = c.shape[1]
    tgt = c[None, :, :] + offs[:, None, :]                                            # (m, nb, ndim)
    rel = tgt - origin
    h = torch.zeros(tgt.shape[:2], dtype=torch.int64, device=device)
    for d in range(ndim):
        h = h * extent[d] + rel[..., d]
    del tgt, rel
    pos = torch.searchsorted(hashes, h.reshape(-1)).clamp_(0, n - 1)
    ok = hashes[pos] == h.reshape(-1)
    del h
    row = torch.arange(nb, device=device).repeat(offs.shape[0])                      # matches reshape order (m, nb)
    row = row[ok]
    col = pos[ok]
    del pos, ok
    order = torch.argsort(row * n + col)
    row, col = row[order], col[order]
    crow = torch.zeros(nb + 1, dtype=torch.int64, device=device)
    crow[1:] = torch.cumsum(torch.bincount(row, minlength=nb), 0)
    return _csr_tensor(crow, col, torch.ones(len(col), dtype=torch.float64, device=device), (nb, n), device)


def _spmm_shell_blocks(lat: Lattice, offsets: np.ndarray, device, rows, cache: dict, X_dense,
                       block_bytes: int = 512 * 2**20):
    """Y = S[rows, :] X_dense for the dense path, in row blocks.

    The shell-operator construction (offsets x rows x ndim int64 targets, 3-5 such arrays alive) needs at most about
    ``block_bytes`` at a time; the result is row-wise identical to the unblocked product.

    Parameters
    ----------
    lat : Lattice
        The lattice.
    offsets : numpy.ndarray
        (m, ndim) offsets of the shell.
    device : str or torch.device
        Torch device.
    rows : torch.Tensor or None
        Device int64 row index (rows of Y in that order), or ``None`` for all rows.
    cache : dict
        Device copies of the lattice arrays (``shell_operator_rows``).
    X_dense : torch.Tensor
        (n_cells, G) dense float64 counts on the device.
    block_bytes : int
        Memory bound of one block.

    Returns
    -------
    Y : torch.Tensor
        (len(rows), G) dense product.
    nnz : int
        Stored entries of the shell-operator rows.
    seconds_S : float
        Time to build the shell-operator rows.
    seconds_SX : float
        Time of the products.
    """
    torch = _torch()
    n = lat.n_cells if rows is None else int(rows.shape[0])
    m = max(1, len(offsets))
    block_rows = max(1 << 15, min(int(block_bytes // (m * lat.cells.shape[1] * 8)),
                                  int(block_bytes // (X_dense.shape[1] * X_dense.element_size()))))
    Y = torch.empty((n, X_dense.shape[1]), dtype=X_dense.dtype, device=device)
    nnz = 0
    seconds_S = seconds_SX = 0.0
    for b0 in range(0, n, block_rows):
        b1 = min(n, b0 + block_rows)
        ta = time.perf_counter()
        S_block = shell_operator_rows(lat, offsets, device, (b0, b1) if rows is None else rows[b0:b1], cache)
        _sync(device)
        tb = time.perf_counter()
        nnz += int(S_block._nnz())
        Y[b0:b1] = torch.sparse.mm(S_block, X_dense)
        _sync(device)
        seconds_S += tb - ta
        seconds_SX += time.perf_counter() - tb
        del S_block
    return Y, nnz, seconds_S, seconds_SX


def pair_counts_level(X: sp.csr_matrix, lat: Lattice, n_shells: int, shells: list[int] | None = None, *,
                      include_intra: bool = False, row_groups: np.ndarray | None = None, sink: Callable | None = None,
                      totals_only: bool = False, timing: LevelTiming | None = None,
                      options: GPUOptions | None = None) -> np.ndarray | None:
    """Ordered pair counts of the requested shells of one lattice level on a torch device.

    The arguments are those of the CPU backend, ``scote._twopoint.cpu.pair_counts_level``, with
    :class:`scote.GPUOptions`; the counts are bit-identical.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix
        (n_cells, G) counts of the occupied cells, rows in lattice order.
    lat : Lattice
        The lattice of X.
    n_shells : int
        Number of shells of the level.
    shells : list of int or None
        Shells to count; ``None`` counts all shells 0 .. n_shells - 1.
    include_intra : bool
        Count shell 0 as the pairs of distinct molecules within one cell, x_a x_b - delta_ab x_a.
    row_groups : numpy.ndarray or None
        (n_cells,) group id per cell (the spatial tile). The pairs are then attributed to the group of their first
        cell and returned per group.
    sink : callable or None
        Receives every (G, G) float64 slab instead of storing it: ``sink(i, slab)`` without row groups,
        ``sink(i, g, slab)`` with row groups (``i`` = position in ``shells``, ``g`` = group index in the order of
        ``np.unique(row_groups)``). The slab is a host array, or a device tensor that is valid only during the call
        with ``options.device_slabs``. Slabs arrive shells outer, groups inner. When an out-of-memory retry
        restarts a shell after some of its slabs were passed on, ``sink.discard(i)`` is called (the error is raised
        if the sink has no such method).
    totals_only : bool
        Without row groups: return only the per-shell totals sum_ab P_k (avoids the G^2 device-to-host copy).
    timing : LevelTiming or None
        Receives times, work, and peak device memory.
    options : GPUOptions or None
        Options of the GPU backend (:class:`scote.GPUOptions`); ``None`` uses the defaults.

    Returns
    -------
    numpy.ndarray or None
        Without row groups: (len(shells), G, G) float64 counts in host memory, or the (len(shells),) totals with
        ``sink`` or ``totals_only``. With row groups: (n_groups, len(shells), G, G), or ``None`` with ``sink``.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    ValueError
        For ``totals_only`` with row groups.
    """
    opt = GPUOptions() if options is None else options
    _torch()
    shells = list(range(n_shells)) if shells is None else list(shells)
    if row_groups is not None:
        if totals_only:
            raise ValueError("totals_only applies only without row_groups")
        return _pair_counts_groups(X, lat, n_shells, shells, row_groups, opt, include_intra, timing, sink)
    return _pair_counts_global(X, lat, n_shells, shells, opt, include_intra, timing, sink, totals_only)


def _slab_out(slab, device_slabs: bool):
    """A slab as passed to a sink.

    Parameters
    ----------
    slab : torch.Tensor
        (G, G) float64 device tensor.
    device_slabs : bool
        Pass the device tensor itself instead of a host copy.

    Returns
    -------
    torch.Tensor or numpy.ndarray
        The tensor, or its host copy.
    """
    return slab if device_slabs else slab.cpu().numpy()


def _row_blocks(X: sp.csr_matrix, offsets: dict, shells: list[int], row_block_bytes: int) -> int:
    """Rows per block of the semidense path.

    Bounds both the dense Y_b (rows x G x 8 bytes) and the SpGEMM output S_b X, whose nnz per row approaches
    min(G, |shell| nnz-per-row(X)) at intermediate densities (cuSPARSE needs about 3x that as workspace).

    Parameters
    ----------
    X : scipy.sparse.csr_matrix
        (n_cells, G) counts.
    offsets : dict
        Shell index -> offsets.
    shells : list of int
        Shells counted.
    row_block_bytes : int
        Memory bound of one block.

    Returns
    -------
    int
        Rows per block, at least 1024.
    """
    n, G = X.shape
    max_shell = max(len(offsets[k]) for k in shells) if shells else 1
    nnz_row = X.nnz / max(1, n)
    y_nnz_row = min(G, max_shell * nnz_row)
    rows_per_block = int(min(row_block_bytes // (8 * G), row_block_bytes // (4 * (8 + 8) * max(1.0, y_nnz_row))))
    return max(1024, rows_per_block)


def _pair_counts_global(X, lat, n_shells, shells, opt: GPUOptions, include_intra, timing, sink, totals_only):
    """Global (ungrouped) counts of one level.

    Parameters
    ----------
    X, lat, n_shells, shells, include_intra, timing, sink, totals_only
        As in ``pair_counts_level``.
    opt : GPUOptions
        Backend options.

    Returns
    -------
    numpy.ndarray
        As ``pair_counts_level`` without row groups.
    """
    torch = _torch()
    device = opt.device
    _configure_cuda(device)
    f64 = torch.float64
    G = X.shape[1]
    n = X.shape[0]
    X = sp.csr_matrix(X)
    X.sort_indices()
    offsets = shell_offsets(n_shells, lat.cells.shape[1])
    use_dense = 2 * n * G * 8 <= _effective_dense_limit(device, opt.dense_bytes_limit)
    P_out = np.zeros((len(shells),) if (totals_only or sink is not None) else (len(shells), G, G), dtype=np.float64)
    slot = {k: i for i, k in enumerate(shells)}

    def _emit(i, slab):
        """Pass the slab of one shell slot on: to the sink (and its total), as a total, or as a host copy.

        Parameters
        ----------
        i : int
            Shell slot.
        slab : torch.Tensor
            (G, G) float64 device tensor.
        """
        if sink is not None:
            sink(i, _slab_out(slab, opt.device_slabs))
            P_out[i] = float(slab.sum().item())
        elif totals_only:
            P_out[i] = float(slab.sum().item())
        else:
            P_out[i] = slab.cpu().numpy()

    t_up = time.perf_counter()
    X_dev = _to_torch_csr(X, device, f64)
    if use_dense:
        X_dense = X_dev.to_dense()
        blocks = None
    else:
        rows_per_block = _row_blocks(X, offsets, shells, opt.row_block_bytes)
        blocks = [(b0, min(n, b0 + rows_per_block)) for b0 in range(0, n, rows_per_block)]
        XT_blocks = [_to_torch_csr(X[b0:b1].T.tocsr(), device, f64) for b0, b1 in blocks]
    _sync(device)
    upload_s = time.perf_counter() - t_up
    P_dev = torch.zeros((G, G), dtype=f64, device=device)
    lat_cache: dict = {}
    for k in shells:
        if k == 0:
            if include_intra:
                P_dev.zero_()
                if use_dense:
                    P_dev += X_dense.transpose(0, 1) @ X_dense
                else:
                    for (b0, b1), XT in zip(blocks, XT_blocks):
                        X_block = _to_torch_csr(X[b0:b1], device, f64)
                        P_dev += torch.sparse.mm(XT, X_block.to_dense())
                _sync(device)
                P_dev -= torch.diag(torch.from_numpy(np.asarray(X.sum(0)).ravel().astype(np.float64)).to(device))
                _emit(slot[0], P_dev)
            continue
        P_dev.zero_()
        nnz_S = 0
        updates = 0.0
        if use_dense:
            Y, nnz_S, s_S, s_SX = _spmm_shell_blocks(lat, offsets[k], device, None, lat_cache, X_dense)
            tb = time.perf_counter()
            P_dev += X_dense.transpose(0, 1) @ Y
            _sync(device)
            s_XtSX = time.perf_counter() - tb
            updates = float(nnz_S) * G * G
            del Y
        else:
            s_S = s_SX = s_XtSX = 0.0

            def _block(b0, b1, XT, depth=0):
                """P += X[b0:b1]^T S_k[b0:b1] X; on device out-of-memory the block is split in two (recursively).

                Parameters
                ----------
                b0, b1 : int
                    Row block.
                XT : torch.Tensor
                    X[b0:b1]^T on the device (sparse CSR).
                depth : int
                    Recursion depth of the splits.
                """
                nonlocal nnz_S, s_S, s_SX, s_XtSX, updates
                try:
                    ta = time.perf_counter()
                    S_block = shell_operator_rows(lat, offsets[k], device, (b0, b1), lat_cache)
                    _sync(device)
                    tb = time.perf_counter()
                    Y_block = torch.sparse.mm(S_block, X_dev).to_dense()           # rows x G (integers)
                    _sync(device)
                    tc = time.perf_counter()
                    P_dev.add_(torch.sparse.mm(XT, Y_block))                     # (G x rows) x (rows x G) dense
                    _sync(device)
                    td = time.perf_counter()
                    nnz_S += int(S_block._nnz())
                    s_S += tb - ta
                    s_SX += tc - tb
                    s_XtSX += td - tc
                    updates += float(X.indptr[b1] - X.indptr[b0]) * G
                    del S_block, Y_block
                except torch.OutOfMemoryError:
                    if b1 - b0 < 2048 or depth > 8:
                        raise
                    _empty_cache(device)
                    mid = (b0 + b1) // 2
                    if timing is not None:
                        timing.per_shell.append(dict(k=k, oom_split=(b0, b1)))
                    _block(b0, mid, _to_torch_csr(X[b0:mid].T.tocsr(), device, f64), depth + 1)
                    _block(mid, b1, _to_torch_csr(X[mid:b1].T.tocsr(), device, f64), depth + 1)

            for (b0, b1), XT in zip(blocks, XT_blocks):
                _block(b0, b1, XT)
        _emit(slot[k], P_dev)
        _empty_cache(device)
        if timing is not None:
            timing.add_shell(k, nnz_S, s_S, s_SX, s_XtSX, updates)
    if timing is not None:
        timing.mode = f"gpu-{'dense' if use_dense else 'semidense'}"
        timing.n_cells, timing.nnz_X, timing.density_X = n, int(X.nnz), X.nnz / max(1, n * G)
        timing.per_shell.append(dict(k=-1, upload_s=upload_s, device=str(device), max_mem_MB=_peak_mb(device)))
    X_dev = P_dev = None                                   # free the device memory before emptying the cache
    _empty_cache(device)
    return P_out


def _pair_counts_groups(X, lat, n_shells, shells, row_groups, opt: GPUOptions, include_intra, timing, sink):
    """Per-group counts P[g, i] = X[rows_g]^T S_k[rows_g, :] X of one level.

    Rows are permuted so that every group is contiguous; the shell operators are built on the device for the permuted
    rows, so Y = S_k[rows_g, :] X is one product per row block (a block packs whole groups up to ``row_block_bytes`` of
    dense Y; a larger group is split into segments whose partial sums are accumulated on the device), and every group
    costs one product X[rows_g]^T Y_g.

    Parameters
    ----------
    X, lat, n_shells, shells, row_groups, include_intra, timing, sink
        As in ``pair_counts_level``.
    opt : GPUOptions
        Backend options.

    Returns
    -------
    numpy.ndarray or None
        As ``pair_counts_level`` with row groups.
    """
    torch = _torch()
    device = opt.device
    _configure_cuda(device)
    f64 = torch.float64
    X = sp.csr_matrix(X)
    X.sort_indices()
    n, G = X.shape
    offsets = shell_offsets(n_shells, lat.cells.shape[1])
    row_groups = np.asarray(row_groups)
    groups, inv = np.unique(row_groups, return_inverse=True)
    n_groups = len(groups)
    perm = np.argsort(inv, kind="stable")                          # rows sorted by group
    starts = np.searchsorted(inv[perm], np.arange(n_groups + 1))   # group g = perm[starts[g]:starts[g+1]]
    P_out = None if sink is not None else np.zeros((n_groups, len(shells), G, G), dtype=np.float64)
    slot = {k: i for i, k in enumerate(shells)}
    use_dense = 2 * n * G * 8 <= _effective_dense_limit(device, opt.dense_bytes_limit)
    lat_cache: dict = {}
    emitted = {}                                                   # slot -> number of slabs passed to the sink

    def _emit(g, i, slab):
        """Store the slab of one group and shell slot, or pass it to the sink.

        Parameters
        ----------
        g : int
            Group index.
        i : int
            Shell slot.
        slab : torch.Tensor
            (G, G) float64 device tensor.
        """
        if sink is None:
            P_out[g, i] = slab.cpu().numpy()
        else:
            sink(i, g, _slab_out(slab, opt.device_slabs))
            emitted[i] = emitted.get(i, 0) + 1

    t_up = time.perf_counter()
    X_dev = _to_torch_csr(X, device, f64)                          # original row order: columns of S_k index it
    perm_dev = torch.from_numpy(perm).to(device)
    col_sums = None
    if use_dense:
        X_dense = X_dev.to_dense()
        X_dense_perm = X_dense[perm_dev]                           # rows in group order
        seg_state = None
    else:
        seg_state = {}

        def _segments(rows_per_block):
            """Split the groups into segments of at most ``rows_per_block`` rows (in permuted row order) and pack whole
            segments into blocks of at most that many rows; uploads X^T of every segment.

            Parameters
            ----------
            rows_per_block : int
                Rows per block.
            """
            segs = []
            for g in range(n_groups):
                for s0 in range(starts[g], starts[g + 1], rows_per_block):
                    segs.append((g, s0, min(starts[g + 1], s0 + rows_per_block)))
            blocks, cur = [], []
            for seg in segs:
                if cur and seg[2] - cur[0][1] > rows_per_block:
                    blocks.append(cur)
                    cur = []
                cur.append(seg)
            if cur:
                blocks.append(cur)
            XT = [_to_torch_csr(X[perm[s0:s1]].T.tocsr(), device, f64) if s1 > s0 else None for (_, s0, s1) in segs]
            seg_state.update(rows_per_block=rows_per_block, segs=segs, blocks=blocks, XT=XT)
        _segments(_row_blocks(X, offsets, shells, opt.row_block_bytes))
    _sync(device)
    upload_s = time.perf_counter() - t_up
    acc = torch.zeros((G, G), dtype=f64, device=device)

    for k in shells:
        if k == 0:
            if include_intra:
                if col_sums is None:
                    col_sums = torch.from_numpy(np.asarray(X.sum(0)).ravel().astype(np.float64)).to(device)
                for g in range(n_groups):
                    s0, s1 = starts[g], starts[g + 1]
                    if s1 <= s0:
                        continue
                    if use_dense:
                        X_group = X_dense_perm[s0:s1]
                        P_group = X_group.transpose(0, 1) @ X_group
                        group_sums = X_group.sum(0)
                    else:
                        X_group = _to_torch_csr(X[perm[s0:s1]], device, f64)
                        P_group = torch.sparse.mm(_to_torch_csr(X[perm[s0:s1]].T.tocsr(), device, f64),
                                                  X_group.to_dense())
                        group_sums = X_group.to_dense().sum(0)
                    P_group -= torch.diag(group_sums)
                    _emit(g, slot[0], P_group)
                    del P_group, X_group
                _sync(device)
            continue
        nnz_S = 0
        updates = 0.0
        s_S = s_SX = s_XtSX = 0.0
        if use_dense:
            Y, nnz_S, s_S, s_SX = _spmm_shell_blocks(lat, offsets[k], device, perm_dev, lat_cache, X_dense)
            tc = time.perf_counter()
            for g in range(n_groups):
                s0, s1 = starts[g], starts[g + 1]
                if s1 <= s0:
                    continue
                _emit(g, slot[k], X_dense_perm[s0:s1].transpose(0, 1) @ Y[s0:s1])
            _sync(device)
            s_XtSX = time.perf_counter() - tc
            updates = float(nnz_S) * G * G
            del Y
        else:
            attempt = 0
            while True:
                try:
                    blocks, XT = seg_state["blocks"], seg_state["XT"]
                    nnz_S = 0
                    updates = 0.0
                    s_S = s_SX = s_XtSX = 0.0
                    si = 0
                    for block in blocks:
                        b0, b1 = block[0][1], block[-1][2]
                        ta = time.perf_counter()
                        S_block = shell_operator_rows(lat, offsets[k], device, perm_dev[b0:b1], lat_cache)
                        _sync(device)
                        tb = time.perf_counter()
                        Y_block = torch.sparse.mm(S_block, X_dev).to_dense()               # (b1 - b0, G)
                        _sync(device)
                        tc = time.perf_counter()
                        nnz_S += int(S_block._nnz())
                        del S_block
                        for (g, s0, s1) in block:
                            if s1 <= s0:
                                si += 1
                                continue
                            Pk = torch.sparse.mm(XT[si], Y_block[s0 - b0:s1 - b0])
                            updates += float(XT[si]._nnz()) * G
                            whole = (s0 == starts[g] and s1 == starts[g + 1])
                            if whole:
                                _emit(g, slot[k], Pk)
                            else:
                                acc.add_(Pk)
                                if s1 == starts[g + 1]:                                   # last segment of the group
                                    _emit(g, slot[k], acc)
                                    acc.zero_()
                            del Pk
                            si += 1
                        _sync(device)
                        td = time.perf_counter()
                        del Y_block
                        s_S += tb - ta
                        s_SX += tc - tb
                        s_XtSX += td - tc
                    break
                except torch.OutOfMemoryError:
                    attempt += 1
                    acc.zero_()
                    if sink is not None and emitted.get(slot[k]):
                        if not hasattr(sink, "discard"):
                            raise
                        sink.discard(slot[k])
                        emitted[slot[k]] = 0
                    if attempt > 6 or seg_state["rows_per_block"] <= 1024:
                        raise
                    _empty_cache(device)
                    if timing is not None:
                        timing.per_shell.append(dict(k=k, oom_halve_rows_per_block=seg_state["rows_per_block"] // 2))
                    _segments(max(1024, seg_state["rows_per_block"] // 2))
        _empty_cache(device)
        if timing is not None:
            timing.add_shell(k, nnz_S, s_S, s_SX, s_XtSX, updates)
    if timing is not None:
        timing.mode = f"gpu-tiles-{'dense' if use_dense else 'semidense'}"
        timing.n_cells, timing.nnz_X, timing.density_X = n, int(X.nnz), X.nnz / max(1, n * G)
        timing.per_shell.append(dict(k=-1, upload_s=upload_s, device=str(device), n_groups=int(n_groups),
                                     rows_per_block=(None if use_dense else seg_state["rows_per_block"]),
                                     max_mem_MB=_peak_mb(device)))
    del X_dev, acc
    _empty_cache(device)
    return P_out
