"""Integer lattices of occupied cells, their coarse-graining, and the shell operators S_k.

A lattice level holds the occupied cells of an integer grid of pitch h (one row per occupied cell, sorted by a linear
hash of the cell coordinates) and the feature-count matrix X (cells x features, CSR). Shell k contains the integer cell
offsets o != 0 with floor(norm(o)) == k; the shell operator S_k is the 0/1 (cells x cells) adjacency of cell pairs whose
offset lies in shell k, so that X^T S_k X are the ordered pair counts of shell k. Both CPU and GPU products
(``scote._twopoint.cpu``, ``scote._twopoint.gpu``) use these lattices; they are built on the host with parallel numba
kernels (``host="numba"``, default) or with the serial NumPy/SciPy reference code (``host="numpy"``), which give
identical lattices and shell operators.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numba
import numpy as np
import scipy.sparse as sp
from numba import njit, prange

__all__ = ["Lattice", "build_lattice", "coarse_grain", "onehot_csr",
           "shell_central_distances", "shell_offsets", "shell_operator"]

# Rows below which `build_lattice` uses the NumPy/SciPy construction even with host="numba" (np.unique of 1e5 integers
# takes about 5 ms; the numba construction launches about 8 parallel regions).
LATTICE_MIN_NUMBA_N = 100_000

# Lookups below which the serial variants of the host kernels run (a parallel-region launch costs 0.1-100 ms; the
# serial rate is about 10 ns per lookup, so 4e6 is about 40 ms of work).
HOST_MIN_PARALLEL_WORK = 4e6


def shell_offsets(n_shells: int, ndim: int = 2) -> dict[int, np.ndarray]:
    """Integer cell offsets of every lattice shell.

    Parameters
    ----------
    n_shells : int
        Number of shells; shell k holds the offsets o != 0 with floor(norm(o)) == k, for k = 0 .. n_shells - 1.
    ndim : int
        Dimension of the lattice.

    Returns
    -------
    dict[int, numpy.ndarray]
        Shell index -> (n_offsets, ndim) read-only int array of the offsets in that shell (shell 0 is empty: the only
        offset with norm(o) < 1 is o = 0, which is excluded). The arrays are computed once per (n_shells, ndim).
    """
    return dict(_shell_offsets_cached(int(n_shells), int(ndim)))


@functools.lru_cache(maxsize=32)
def _shell_offsets_cached(n_shells: int, ndim: int) -> dict[int, np.ndarray]:
    """The offsets of ``shell_offsets``, computed once per argument pair.

    Parameters
    ----------
    n_shells : int
        Number of shells.
    ndim : int
        Dimension of the lattice.

    Returns
    -------
    dict[int, numpy.ndarray]
        Shell index -> read-only (n_offsets, ndim) offsets.
    """
    r = np.arange(-n_shells + 1, n_shells)
    grids = np.meshgrid(*([r] * ndim), indexing="ij")
    offs = np.stack([g.ravel() for g in grids], axis=1)
    dist = np.sqrt((offs.astype(np.float64) ** 2).sum(1))
    k = np.floor(dist).astype(int)
    out = {}
    for kk in range(n_shells):
        sel = (k == kk) & ~(offs == 0).all(1)
        out[kk] = offs[sel]
        out[kk].flags.writeable = False
    return out


@functools.lru_cache(maxsize=32)
def _shell_central_distances_cached(n_shells: int, ndim: int) -> np.ndarray:
    """The values of ``shell_central_distances``, computed once per argument pair.

    Parameters
    ----------
    n_shells : int
        Number of shells.
    ndim : int
        Dimension of the lattice.

    Returns
    -------
    numpy.ndarray
        Read-only (n_shells,) float64 array.
    """
    offs = shell_offsets(n_shells, ndim)
    central = np.array([np.sqrt((o.astype(float) ** 2).sum(1)).mean() if len(o) else np.nan
                        for k, o in sorted(offs.items())])
    central[0] = 0.5
    central.flags.writeable = False
    return central


def shell_central_distances(n_shells: int, ndim: int = 2) -> np.ndarray:
    """Central distance of each lattice shell, in units of the lattice pitch.

    Shell :math:`k \\ge 1` pairs the cells at the integer offsets :math:`o` with :math:`\\lfloor |o| \\rfloor = k`;
    its central distance is the mean offset length :math:`|o|`. Shell 0 holds the pairs within one cell (offset 0)
    and is placed at half the pitch, the middle of its interval :math:`[0, 1)`.

    Parameters
    ----------
    n_shells : int
        Number of shells.
    ndim : int
        Dimension of the lattice.

    Returns
    -------
    numpy.ndarray
        (n_shells,) float64 array (a copy of a value computed once per argument pair).
    """
    return _shell_central_distances_cached(int(n_shells), int(ndim)).copy()


@dataclass
class Lattice:
    """Occupied cells of an integer lattice, sorted by a linear hash for neighbor lookup.

    Attributes
    ----------
    cells : numpy.ndarray
        (n_cells, ndim) int64 cell coordinates, unique rows, sorted by hash.
    hashes : numpy.ndarray
        (n_cells,) int64 sorted hashes, ``sum_d (cells[:, d] - origin[d]) * prod(extent[d + 1:])``.
    origin : numpy.ndarray
        (ndim,) int64 origin of the hash (the minimum cell coordinate minus a margin).
    extent : numpy.ndarray
        (ndim,) int64 extent of the hash (larger than the coordinate range plus the margin on both sides, so that
        every offset of a shell maps to a valid hash).
    """
    cells: np.ndarray
    hashes: np.ndarray
    origin: np.ndarray
    extent: np.ndarray

    @property
    def n_cells(self) -> int:
        """Number of occupied cells."""
        return len(self.hashes)

    def hash_of(self, cells: np.ndarray) -> np.ndarray:
        """Hash of arbitrary cells.

        Parameters
        ----------
        cells : numpy.ndarray
            (n, ndim) int cell coordinates.

        Returns
        -------
        numpy.ndarray
            (n,) int64 hashes.
        """
        rel = cells - self.origin
        h = np.zeros(len(cells), dtype=np.int64)
        for d in range(cells.shape[1]):
            h = h * self.extent[d] + rel[:, d]
        return h

    def lookup(self, cells: np.ndarray) -> np.ndarray:
        """Row index of each cell in the lattice.

        Parameters
        ----------
        cells : numpy.ndarray
            (n, ndim) int cell coordinates.

        Returns
        -------
        numpy.ndarray
            (n,) int row indices, -1 where the cell is not occupied.
        """
        h = self.hash_of(cells)
        pos = np.searchsorted(self.hashes, h)
        pos = np.clip(pos, 0, len(self.hashes) - 1)
        ok = self.hashes[pos] == h
        return np.where(ok, pos, -1)


def onehot_csr(codes: np.ndarray, n_features: int | None = None, dtype=np.float64) -> sp.csr_matrix:
    """One-hot CSR matrix of feature codes (one molecule per row).

    Parameters
    ----------
    codes : numpy.ndarray
        (n,) integer feature codes in 0 .. n_features - 1.
    n_features : int or None
        Number of columns G; ``None`` uses ``codes.max() + 1``.
    dtype : numpy.dtype
        Dtype of the stored ones.

    Returns
    -------
    scipy.sparse.csr_matrix
        (n, G) matrix with a single 1 per row in column ``codes[i]``; built directly (indptr = arange,
        indices = codes), the same matrix as ``sp.csr_matrix((ones, (arange(n), codes)))`` without its sort.

    Raises
    ------
    ValueError
        If a code lies outside 0 .. n_features - 1.
    """
    codes = np.asarray(codes)
    n = len(codes)
    if n_features is None:
        n_features = int(codes.max()) + 1 if n else 0
    if n and (codes.min() < 0 or codes.max() >= n_features):
        raise ValueError(f"codes must lie in 0 .. {n_features - 1}")
    idx_dtype = np.int32 if max(n, n_features) < np.iinfo(np.int32).max else np.int64
    return sp.csr_matrix((np.ones(n, dtype=dtype), codes.astype(idx_dtype, copy=False),
                          np.arange(n + 1, dtype=idx_dtype)), shape=(n, n_features))


def build_lattice(cells_int: np.ndarray, X: sp.csr_matrix, margin: int,
                  host: str = "numba") -> tuple[Lattice, sp.csr_matrix]:
    """Aggregate the rows of X that fall into the same lattice cell.

    Parameters
    ----------
    cells_int : numpy.ndarray
        (n, ndim) integer cell coordinate of every row of X.
    X : scipy.sparse.csr_matrix
        (n, G) feature counts per row.
    margin : int
        Cells added to the hash range on both sides; at least the number of shells, so that shifted cells keep valid
        hashes.
    host : {"numba", "numpy"}
        "numba": parallel numba construction (from LATTICE_MIN_NUMBA_N rows on); "numpy": np.unique and a SciPy
        product. Both give the same lattice and bit-identical counts.

    Returns
    -------
    lattice : Lattice
        The occupied cells, sorted by hash.
    X : scipy.sparse.csr_matrix
        (n_cells, G) float64 summed counts per occupied cell, rows in lattice order.
    """
    if host not in ("numba", "numpy"):
        raise ValueError(f"host must be 'numba' or 'numpy', got {host!r}")
    if host == "numba" and len(cells_int) >= LATTICE_MIN_NUMBA_N:
        return _build_lattice_numba(cells_int, X, margin)
    cells_int = np.asarray(cells_int, dtype=np.int64)
    origin = cells_int.min(0) - margin
    extent = cells_int.max(0) - origin + margin + 1
    h = np.zeros(len(cells_int), dtype=np.int64)
    for d in range(cells_int.shape[1]):
        h = h * extent[d] + (cells_int[:, d] - origin[d])
    uh, inv = np.unique(h, return_inverse=True)
    if len(uh) != len(cells_int):
        A = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(uh), len(inv)))
        X = (A @ X).tocsr()
    else:
        X = X[np.argsort(h)].tocsr()
    ucells = np.empty((len(uh), cells_int.shape[1]), dtype=np.int64)
    rem = uh.copy()
    for d in reversed(range(cells_int.shape[1])):
        ucells[:, d] = rem % extent[d] + origin[d]
        rem //= extent[d]
    return Lattice(ucells, uh, origin, extent), X


def coarse_grain(lat: Lattice, X: sp.csr_matrix, factor: int = 2, margin: int = 8,
                 host: str = "numba") -> tuple[Lattice, sp.csr_matrix]:
    """Merge blocks of factor x factor cells (floor division of the cell coordinates).

    Parameters
    ----------
    lat : Lattice
        The finer lattice.
    X : scipy.sparse.csr_matrix
        (n_cells, G) counts of the finer lattice.
    factor : int
        Coarse-graining factor per dimension.
    margin : int
        Hash margin of the coarse lattice (see ``build_lattice``).
    host : {"numba", "numpy"}
        Construction, see ``build_lattice``.

    Returns
    -------
    lattice : Lattice
        The coarse lattice.
    X : scipy.sparse.csr_matrix
        Its summed counts.
    """
    return build_lattice(np.floor_divide(lat.cells, factor), X, margin, host=host)


def shell_operator(lat: Lattice, offsets: np.ndarray, host: str = "numba") -> sp.csr_matrix:
    """Shell operator S of a set of offsets on the host.

    Parameters
    ----------
    lat : Lattice
        The lattice.
    offsets : numpy.ndarray
        (m, ndim) integer offsets (one shell, see ``shell_offsets``).
    host : {"numba", "numpy"}
        "numba": one parallel merge of the sorted hashes per offset; "numpy": one ``searchsorted`` per offset. Both
        return the same matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        (n_cells, n_cells) canonical CSR (sorted, unique column indices per row; float64 ones) with S[i, j] = 1 if
        cell j = cell i + o for some offset o.
    """
    if host not in ("numba", "numpy"):
        raise ValueError(f"host must be 'numba' or 'numpy', got {host!r}")
    if host == "numba" and lat.n_cells > 0 and len(offsets):
        return _shell_operator_numba(lat, offsets)
    rows, cols = [], []
    n = lat.n_cells
    for off in offsets:
        j = lat.lookup(lat.cells + off)
        ok = j >= 0
        rows.append(np.nonzero(ok)[0])
        cols.append(j[ok])
    if not rows:
        return sp.csr_matrix((n, n))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(n, n))


# ----------------------------------------------------------------------------- host kernels

@njit(cache=True, nogil=True)
def _lower_bound(a, lo, hi, v):
    """First index in [lo, hi) with a[index] >= v.

    Parameters
    ----------
    a : numpy.ndarray
        Array sorted in ascending order.
    lo, hi : int
        Search range.
    v : int
        Value searched.

    Returns
    -------
    int
        The index, or ``hi`` if no element of the range reaches v.
    """
    while lo < hi:
        mid = (lo + hi) >> 1
        if a[mid] < v:
            lo = mid + 1
        else:
            hi = mid
    return lo

@njit(cache=True, nogil=True)
def _shell_csr_count_chunk(hashes, deltas, i0, i1, counts):
    """Pass 1 of ``_shell_operator_numba`` for rows [i0, i1): counts[i] = #{m : hashes[i] + deltas[m] in hashes}.

    For a fixed delta the targets hashes[i] + delta are sorted in i, so each (row chunk, delta) is one merge walk
    after a binary search (O(chunk) per delta, no search per row).

    Parameters
    ----------
    hashes : numpy.ndarray
        (n,) sorted int64 cell hashes.
    deltas : numpy.ndarray
        (m,) ascending hash differences of the shell's offsets.
    i0, i1 : int
        Row range.
    counts : numpy.ndarray
        (n,) int64 neighbor counts, incremented in place.
    """
    n = hashes.shape[0]
    M = deltas.shape[0]
    for m in range(M):
        d = deltas[m]
        j = _lower_bound(hashes, 0, n, hashes[i0] + d)
        for i in range(i0, i1):
            v = hashes[i] + d
            while j < n and hashes[j] < v:
                j += 1
            if j >= n:
                break
            if hashes[j] == v:
                counts[i] += 1

@njit(cache=True, nogil=True)
def _shell_csr_fill_chunk(hashes, deltas, i0, i1, indptr, indices, fill):
    """Pass 2 of ``_shell_operator_numba`` for rows [i0, i1): the column indices of every row.

    The deltas are ascending, so hashes[i] + delta is ascending in m and the matched columns are ascending within the
    row: the CSR is canonical (sorted, unique) without a sort.

    Parameters
    ----------
    hashes : numpy.ndarray
        (n,) sorted int64 cell hashes.
    deltas : numpy.ndarray
        (m,) ascending hash differences.
    i0, i1 : int
        Row range.
    indptr : numpy.ndarray
        (n + 1,) CSR row pointers from pass 1.
    indices : numpy.ndarray
        CSR column indices, filled in place.
    fill : numpy.ndarray
        Scratch of length at least i1 - i0 (zeroed here).
    """
    n = hashes.shape[0]
    M = deltas.shape[0]
    for i in range(i1 - i0):
        fill[i] = 0
    for m in range(M):
        d = deltas[m]
        j = _lower_bound(hashes, 0, n, hashes[i0] + d)
        for i in range(i0, i1):
            v = hashes[i] + d
            while j < n and hashes[j] < v:
                j += 1
            if j >= n:
                break
            if hashes[j] == v:
                indices[indptr[i] + fill[i - i0]] = j
                fill[i - i0] += 1

@njit(parallel=True, cache=True, nogil=True)
def _shell_csr_count_par(hashes, deltas, chunk):
    """Pass 1 of ``_shell_operator_numba`` over all rows in parallel chunks.

    Parameters
    ----------
    hashes : numpy.ndarray
        (n,) sorted int64 cell hashes.
    deltas : numpy.ndarray
        (m,) ascending hash differences.
    chunk : int
        Rows per chunk.

    Returns
    -------
    numpy.ndarray
        (n,) int64 number of neighbors of every row.
    """
    n = hashes.shape[0]
    counts = np.zeros(n, dtype=np.int64)
    for c in prange((n + chunk - 1) // chunk):
        _shell_csr_count_chunk(hashes, deltas, c * chunk, min(n, (c + 1) * chunk), counts)
    return counts

@njit(cache=True, nogil=True)
def _shell_csr_count_ser(hashes, deltas, chunk):
    """Serial variant of ``_shell_csr_count_par`` (same arguments and result)."""
    n = hashes.shape[0]
    counts = np.zeros(n, dtype=np.int64)
    for c in range((n + chunk - 1) // chunk):
        _shell_csr_count_chunk(hashes, deltas, c * chunk, min(n, (c + 1) * chunk), counts)
    return counts

@njit(parallel=True, cache=True, nogil=True)
def _shell_csr_fill_par(hashes, deltas, chunk, indptr, indices):
    """Pass 2 of ``_shell_operator_numba`` over all rows in parallel chunks.

    Parameters
    ----------
    hashes : numpy.ndarray
        (n,) sorted int64 cell hashes.
    deltas : numpy.ndarray
        (m,) ascending hash differences.
    chunk : int
        Rows per chunk.
    indptr : numpy.ndarray
        (n + 1,) CSR row pointers.
    indices : numpy.ndarray
        CSR column indices, filled in place.
    """
    n = hashes.shape[0]
    for c in prange((n + chunk - 1) // chunk):
        fill = np.empty(chunk, dtype=np.int64)
        _shell_csr_fill_chunk(hashes, deltas, c * chunk, min(n, (c + 1) * chunk), indptr, indices, fill)

@njit(cache=True, nogil=True)
def _shell_csr_fill_ser(hashes, deltas, chunk, indptr, indices):
    """Serial variant of ``_shell_csr_fill_par`` (same arguments)."""
    n = hashes.shape[0]
    fill = np.empty(chunk, dtype=np.int64)
    for c in range((n + chunk - 1) // chunk):
        _shell_csr_fill_chunk(hashes, deltas, c * chunk, min(n, (c + 1) * chunk), indptr, indices, fill)

@njit(parallel=True, cache=True, nogil=True)
def _fill_ones(out):
    """out[:] = 1.0 with a parallel first touch (np.ones page-faults 8 bytes per entry on one thread, about 3 s per
    GB-scale shell operator).

    Parameters
    ----------
    out : numpy.ndarray
        (n,) float64 array, overwritten.
    """
    n = out.shape[0]
    blk = 1 << 20
    for b in prange((n + blk - 1) // blk):
        for i in range(b * blk, min(n, (b + 1) * blk)):
            out[i] = 1.0

# ------------------------------------------------------------------ lattice construction
@njit(cache=True, nogil=True)
def _minmax_chunk(cells, i0, i1, lo_out, hi_out):
    """Per-dimension minimum and maximum of the rows [i0, i1) of the cells.

    Parameters
    ----------
    cells : numpy.ndarray
        (n, ndim) int64 cell coordinates.
    i0, i1 : int
        Row range (non-empty).
    lo_out, hi_out : numpy.ndarray
        (ndim,) outputs.
    """
    nd = cells.shape[1]
    for d in range(nd):
        lo = cells[i0, d]
        hi = lo
        for i in range(i0 + 1, i1):
            v = cells[i, d]
            if v < lo:
                lo = v
            if v > hi:
                hi = v
        lo_out[d] = lo
        hi_out[d] = hi

@njit(parallel=True, cache=True, nogil=True)
def _cells_minmax(cells, nt):
    """Per-dimension minimum and maximum of the cells (``cells.min(0)``, ``cells.max(0)``) on nt threads.

    Parameters
    ----------
    cells : numpy.ndarray
        (n, ndim) int64 cell coordinates, n >= 1.
    nt : int
        Number of threads.

    Returns
    -------
    lo, hi : numpy.ndarray
        (ndim,) int64 minimum and maximum.
    """
    n = cells.shape[0]
    nd = cells.shape[1]
    chunk = (n + nt - 1) // nt
    los = np.empty((nt, nd), dtype=np.int64)
    his = np.empty((nt, nd), dtype=np.int64)
    for t in prange(nt):
        i0 = t * chunk
        i1 = min(n, i0 + chunk)
        if i0 < i1:
            _minmax_chunk(cells, i0, i1, los[t], his[t])
        else:
            for d in range(nd):
                los[t, d] = cells[0, d]
                his[t, d] = cells[0, d]
    lo = los[0].copy()
    hi = his[0].copy()
    for t in range(1, nt):
        for d in range(nd):
            if los[t, d] < lo[d]:
                lo[d] = los[t, d]
            if his[t, d] > hi[d]:
                hi[d] = his[t, d]
    return lo, hi

@njit(parallel=True, cache=True, nogil=True)
def _hash_cells(cells, origin, strides, nt):
    """Linear hashes h[i] = sum_d (cells[i, d] - origin[d]) strides[d] (the Horner form of ``build_lattice`` and
    ``Lattice.hash_of``).

    Parameters
    ----------
    cells : numpy.ndarray
        (n, ndim) int64 cell coordinates.
    origin : numpy.ndarray
        (ndim,) int64 hash origin.
    strides : numpy.ndarray
        (ndim,) int64 strides (``_hash_strides``).
    nt : int
        Number of threads.

    Returns
    -------
    numpy.ndarray
        (n,) int64 hashes.
    """
    n = cells.shape[0]
    nd = cells.shape[1]
    out = np.empty(n, dtype=np.int64)
    chunk = (n + nt - 1) // nt
    for t in prange(nt):
        for i in range(t * chunk, min(n, (t + 1) * chunk)):
            h = 0
            for d in range(nd):
                h += (cells[i, d] - origin[d]) * strides[d]
            out[i] = h
    return out

@njit(cache=True, nogil=True)
def _radix_hist_chunk(keys, shift, mask, i0, i1, hist_t):
    """Histogram of the digit (keys >> shift) & mask over the rows [i0, i1).

    Parameters
    ----------
    keys : numpy.ndarray
        (n,) int64 keys.
    shift : int
        Bit position of the digit.
    mask : int
        Digit mask.
    i0, i1 : int
        Row range.
    hist_t : numpy.ndarray
        (2^bits,) int64 histogram, overwritten.
    """
    for b in range(hist_t.shape[0]):
        hist_t[b] = 0
    for i in range(i0, i1):
        hist_t[(keys[i] >> shift) & mask] += 1

@njit(cache=True, nogil=True)
def _radix_scatter_chunk(ks, ps, kd, pd, shift, mask, i0, i1, hist_t):
    """Stable scatter of the rows [i0, i1) of (ks, ps) to (kd, pd) by digit.

    Parameters
    ----------
    ks, ps : numpy.ndarray
        Source keys and carried indices.
    kd, pd : numpy.ndarray
        Destination keys and carried indices.
    shift : int
        Bit position of the digit.
    mask : int
        Digit mask.
    i0, i1 : int
        Row range.
    hist_t : numpy.ndarray
        Write offsets of the chunk per digit, advanced in place.
    """
    for i in range(i0, i1):
        d = (ks[i] >> shift) & mask
        pos = hist_t[d]
        hist_t[d] = pos + 1
        kd[pos] = ks[i]
        pd[pos] = ps[i]

@njit(parallel=True, cache=True, nogil=True)
def _radix_argsort(keys, nbits, nt):
    """Stable LSD radix sort of non-negative int64 keys below 2^nbits that carries the original index.

    ceil(nbits / 8) passes of at most 8-bit digits, with per-thread histograms over row chunks; offsets in
    (digit, chunk) order keep it stable.

    Parameters
    ----------
    keys : numpy.ndarray
        (n,) non-negative int64 keys; used as one of the two buffers, i.e. overwritten.
    nbits : int
        Number of significant bits of the keys.
    nt : int
        Number of threads.

    Returns
    -------
    sorted_keys : numpy.ndarray
        (n,) the keys in ascending order.
    perm : numpy.ndarray
        (n,) int64 permutation, ``np.argsort(keys, kind="stable")``.
    """
    n = keys.shape[0]
    npass = max(1, (nbits + 7) // 8)
    B = max(1, (nbits + npass - 1) // npass)
    nb = 1 << B
    mask = nb - 1
    kA = keys
    kB = np.empty(n, dtype=np.int64)
    pA = np.arange(n, dtype=np.int64)
    pB = np.empty(n, dtype=np.int64)
    chunk = (n + nt - 1) // nt
    hist = np.zeros((nt, nb), dtype=np.int64)
    for p in range(npass):
        shift = p * B
        if p % 2 == 0:
            ks = kA
            ps = pA
            kd = kB
            pd = pB
        else:
            ks = kB
            ps = pB
            kd = kA
            pd = pA
        for t in prange(nt):
            _radix_hist_chunk(ks, shift, mask, t * chunk, min(n, (t + 1) * chunk), hist[t])
        s = 0
        for b in range(nb):
            for t in range(nt):
                c = hist[t, b]
                hist[t, b] = s
                s += c
        for t in prange(nt):
            _radix_scatter_chunk(ks, ps, kd, pd, shift, mask, t * chunk, min(n, (t + 1) * chunk), hist[t])
    if npass % 2 == 1:
        return kB, pB
    return kA, pA

@njit(parallel=True, cache=True, nogil=True)
def _run_starts(sk, nt):
    """Start positions of the runs of equal values in a sorted array.

    Parameters
    ----------
    sk : numpy.ndarray
        (n,) sorted int64 values.
    nt : int
        Number of threads.

    Returns
    -------
    numpy.ndarray
        (n_runs + 1,) int64 start of every run, then n.
    """
    n = sk.shape[0]
    chunk = (n + nt - 1) // nt
    cnt = np.zeros(nt + 1, dtype=np.int64)
    for t in prange(nt):
        c = 0
        for i in range(t * chunk, min(n, (t + 1) * chunk)):
            if i == 0 or sk[i] != sk[i - 1]:
                c += 1
        cnt[t + 1] = c
    for t in range(nt):
        cnt[t + 1] += cnt[t]
    starts = np.empty(cnt[nt] + 1, dtype=np.int64)
    for t in prange(nt):
        o = cnt[t]
        for i in range(t * chunk, min(n, (t + 1) * chunk)):
            if i == 0 or sk[i] != sk[i - 1]:
                starts[o] = i
                o += 1
    starts[cnt[nt]] = n
    return starts

@njit(cache=True, nogil=True)
def _agg_count_chunk(starts, perm, X_indptr, X_indices, r0, r1, counts, mark, touched):
    """Pass 1 of the row aggregation for the runs [r0, r1): number of distinct columns of every run.

    Parameters
    ----------
    starts : numpy.ndarray
        Run boundaries in ``perm`` (``_run_starts``).
    perm : numpy.ndarray
        Rows of X in hash order.
    X_indptr, X_indices : numpy.ndarray
        CSR arrays of X.
    r0, r1 : int
        Run range.
    counts : numpy.ndarray
        Distinct columns per run, written for the range.
    mark : numpy.ndarray
        (G,) uint8 scratch, zero on entry and exit.
    touched : numpy.ndarray
        (G,) int64 scratch.
    """
    for r in range(r0, r1):
        nt = 0
        for q in range(starts[r], starts[r + 1]):
            i = perm[q]
            for p in range(X_indptr[i], X_indptr[i + 1]):
                b = X_indices[p]
                if mark[b] == 0:
                    mark[b] = 1
                    touched[nt] = b
                    nt += 1
        counts[r] = nt
        for s in range(nt):
            mark[touched[s]] = 0

@njit(cache=True, nogil=True)
def _agg_fill_chunk(starts, perm, X_indptr, X_indices, X_data, r0, r1, out_indptr, out_indices, out_data,
                    acc, mark, touched):
    """Pass 2 of the row aggregation for the runs [r0, r1): row r of the aggregate is the sum of the rows
    perm[starts[r]:starts[r + 1]] of X.

    The rows are added in that order (ascending original row, stable sort) and in each row's stored column order: the
    same sequence of additions as SciPy's csr_matmat for A @ X with the 0/1 aggregation matrix A, hence bit-identical
    data for any float X. Columns are written sorted (canonical CSR).

    Parameters
    ----------
    starts : numpy.ndarray
        Run boundaries in ``perm``.
    perm : numpy.ndarray
        Rows of X in hash order.
    X_indptr, X_indices, X_data : numpy.ndarray
        CSR arrays of X.
    r0, r1 : int
        Run range.
    out_indptr, out_indices, out_data : numpy.ndarray
        CSR arrays of the aggregate; indices and data are filled for the range.
    acc : numpy.ndarray
        (G,) float64 scratch, zero on entry and exit.
    mark : numpy.ndarray
        (G,) uint8 scratch, zero on entry and exit.
    touched : numpy.ndarray
        (G,) int64 scratch.
    """
    G = mark.shape[0]
    for r in range(r0, r1):
        nt = 0
        for q in range(starts[r], starts[r + 1]):
            i = perm[q]
            for p in range(X_indptr[i], X_indptr[i + 1]):
                b = X_indices[p]
                if mark[b] == 0:
                    mark[b] = 1
                    touched[nt] = b
                    nt += 1
                acc[b] += X_data[p]
        base = out_indptr[r]
        if nt > 32:                         # many columns: ordered scan over all G (insertion sort is O(nt^2))
            o = 0
            for b in range(G):
                if mark[b] == 1:
                    out_indices[base + o] = b
                    out_data[base + o] = acc[b]
                    o += 1
                    acc[b] = 0.0
                    mark[b] = 0
        else:                               # few columns: insertion sort of the touched list
            for s in range(1, nt):
                b = touched[s]
                j = s - 1
                while j >= 0 and touched[j] > b:
                    touched[j + 1] = touched[j]
                    j -= 1
                touched[j + 1] = b
            for s in range(nt):
                b = touched[s]
                out_indices[base + s] = b
                out_data[base + s] = acc[b]
                acc[b] = 0.0
                mark[b] = 0

@njit(parallel=True, cache=True, nogil=True)
def _agg_count(starts, perm, X_indptr, X_indices, G, nt):
    """Pass 1 of the row aggregation on nt threads.

    Parameters
    ----------
    starts : numpy.ndarray
        Run boundaries in ``perm``.
    perm : numpy.ndarray
        Rows of X in hash order.
    X_indptr, X_indices : numpy.ndarray
        CSR arrays of X.
    G : int
        Number of columns.
    nt : int
        Number of threads.

    Returns
    -------
    numpy.ndarray
        (n_runs,) int64 number of distinct columns per run.
    """
    nr = starts.shape[0] - 1
    counts = np.zeros(nr, dtype=np.int64)
    chunk = (nr + nt - 1) // nt
    for t in prange(nt):
        mark = np.zeros(G, dtype=np.uint8)
        touched = np.empty(G, dtype=np.int64)
        _agg_count_chunk(starts, perm, X_indptr, X_indices, t * chunk, min(nr, (t + 1) * chunk), counts, mark, touched)
    return counts

@njit(parallel=True, cache=True, nogil=True)
def _agg_fill(starts, perm, X_indptr, X_indices, X_data, out_indptr, out_indices, out_data, G, nt):
    """Pass 2 of the row aggregation on nt threads: the summed rows of every run into the output CSR.

    Parameters
    ----------
    starts : numpy.ndarray
        Run boundaries in ``perm``.
    perm : numpy.ndarray
        Rows of X in hash order.
    X_indptr, X_indices, X_data : numpy.ndarray
        CSR arrays of X.
    out_indptr, out_indices, out_data : numpy.ndarray
        CSR arrays of the aggregate; indices and data are filled.
    G : int
        Number of columns.
    nt : int
        Number of threads.
    """
    nr = starts.shape[0] - 1
    chunk = (nr + nt - 1) // nt
    for t in prange(nt):
        mark = np.zeros(G, dtype=np.uint8)
        touched = np.empty(G, dtype=np.int64)
        acc = np.zeros(G, dtype=np.float64)
        _agg_fill_chunk(starts, perm, X_indptr, X_indices, X_data, t * chunk, min(nr, (t + 1) * chunk),
                        out_indptr, out_indices, out_data, acc, mark, touched)

@njit(parallel=True, cache=True, nogil=True)
def _cells_from_hashes(uh, origin, extent, nt):
    """Cell coordinates of hashes, the inverse of ``_hash_cells``:
    cells[i, d] = uh[i] // prod(extent[d + 1:]) % extent[d] + origin[d].

    Parameters
    ----------
    uh : numpy.ndarray
        (n,) int64 hashes.
    origin : numpy.ndarray
        (ndim,) int64 hash origin.
    extent : numpy.ndarray
        (ndim,) int64 hash extent.
    nt : int
        Number of threads.

    Returns
    -------
    numpy.ndarray
        (n, ndim) int64 cell coordinates.
    """
    n = uh.shape[0]
    nd = origin.shape[0]
    out = np.empty((n, nd), dtype=np.int64)
    chunk = (n + nt - 1) // nt
    for t in prange(nt):
        for i in range(t * chunk, min(n, (t + 1) * chunk)):
            rem = uh[i]
            for d in range(nd - 1, -1, -1):
                out[i, d] = rem % extent[d] + origin[d]
                rem //= extent[d]
    return out


def _hash_strides(lat: Lattice) -> np.ndarray:
    """Strides of the linear hash, hash(cell) = sum_d (cell_d - origin_d) stride_d, as in ``Lattice.hash_of``.

    Parameters
    ----------
    lat : Lattice
        The lattice (only ``extent`` is used).

    Returns
    -------
    numpy.ndarray
        (ndim,) int64 strides.
    """
    ext = np.asarray(lat.extent, dtype=np.int64)
    strides = np.ones(len(ext), dtype=np.int64)
    for d in range(len(ext) - 2, -1, -1):
        strides[d] = strides[d + 1] * ext[d + 1]
    return strides


def _shell_operator_numba(lat: Lattice, offsets: np.ndarray, chunk: int | None = None) -> sp.csr_matrix:
    """``shell_operator`` on the host with numba kernels.

    The hash is linear in the cell coordinates, so the neighbor of row i under an offset has the hash hashes[i] + delta;
    for each offset the lookup is a merge of the sorted hashes with a shifted copy of themselves (O(n_cells) per offset,
    no search per row), in parallel over row chunks. Two passes (count, fill) give a canonical CSR with the same indptr
    and indices as the NumPy construction.

    Parameters
    ----------
    lat : Lattice
        The lattice.
    offsets : numpy.ndarray
        (m, ndim) integer offsets.
    chunk : int or None
        Rows per chunk; ``None`` chooses from the number of cells and threads.

    Returns
    -------
    scipy.sparse.csr_matrix
        (n_cells, n_cells) shell operator with float64 ones.
    """
    n = lat.n_cells
    offsets = np.asarray(offsets, dtype=np.int64).reshape(-1, lat.cells.shape[1])
    deltas = np.unique(offsets @ _hash_strides(lat))                  # ascending; distinct offsets -> distinct deltas
    hashes = np.ascontiguousarray(lat.hashes, dtype=np.int64)
    if chunk is None:
        chunk = int(max(1024, min(1 << 17, n // max(1, 16 * numba.get_num_threads()) + 1)))
    par = numba.get_num_threads() > 1 and n * len(deltas) >= HOST_MIN_PARALLEL_WORK
    counts = (_shell_csr_count_par if par else _shell_csr_count_ser)(hashes, deltas, chunk)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    nnz = int(indptr[-1])
    idx_dtype = np.int32 if max(nnz, n) < np.iinfo(np.int32).max else np.int64   # what scipy would pick
    indices = np.empty(nnz, dtype=idx_dtype)
    (_shell_csr_fill_par if par else _shell_csr_fill_ser)(hashes, deltas, chunk, indptr, indices)
    data = np.empty(nnz, dtype=np.float64)
    if par:
        _fill_ones(data)
    else:
        data.fill(1.0)
    S = sp.csr_matrix((data, indices, indptr.astype(idx_dtype, copy=False)), shape=(n, n))
    S.has_canonical_format = True                                      # sorted + unique by construction
    return S


def _build_lattice_numba(cells_int: np.ndarray, X: sp.csr_matrix, margin: int) -> tuple[Lattice, sp.csr_matrix]:
    """``build_lattice`` with numba kernels, all parallel over row chunks.

    Per-dimension minimum and maximum, the linear hash, a stable LSD radix argsort of the hashes, run compaction (the
    sorted unique hashes), a two-pass aggregation of X over the runs (the same additions in the same order as SciPy's
    ``A @ X``, so the data are bit-identical; sorted columns give a canonical CSR), and the recovery of the cells. The
    lattice equals that of the NumPy construction; X comes back float64 with int32 indices where they fit.

    Parameters
    ----------
    cells_int : numpy.ndarray
        (n, ndim) integer cell coordinate of every row of X.
    X : scipy.sparse.csr_matrix
        (n, G) counts per row.
    margin : int
        Hash margin (``build_lattice``).

    Returns
    -------
    lattice : Lattice
        The occupied cells, sorted by hash.
    X : scipy.sparse.csr_matrix
        (n_cells, G) float64 summed counts.
    """
    cells_int = np.ascontiguousarray(cells_int, dtype=np.int64)
    X = sp.csr_matrix(X)
    nt = max(1, numba.get_num_threads())
    lo, hi = _cells_minmax(cells_int, nt)
    origin = lo - margin
    extent = hi - origin + margin + 1
    strides = _hash_strides(Lattice(None, None, origin, extent))
    h = _hash_cells(cells_int, origin, strides, nt)
    nbits = int(int(extent.prod()).bit_length())
    sk, perm = _radix_argsort(h, nbits, nt)                              # h is consumed as a work buffer
    starts = _run_starts(sk, nt)
    uh = sk[starts[:-1]]
    G = X.shape[1]
    Xd = X.data if X.data.dtype == np.float64 else X.data.astype(np.float64)
    counts = _agg_count(starts, perm, X.indptr, X.indices, G, nt)
    nr = len(uh)
    indptr = np.zeros(nr + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    nnz = int(indptr[-1])
    idx_dtype = np.int32 if max(nnz, nr, G) < np.iinfo(np.int32).max else np.int64
    indices = np.empty(nnz, dtype=idx_dtype)
    data = np.empty(nnz, dtype=np.float64)
    _agg_fill(starts, perm, X.indptr, X.indices, Xd, indptr, indices, data, G, nt)
    Xn = sp.csr_matrix((data, indices, indptr.astype(idx_dtype, copy=False)), shape=(nr, G))
    Xn.has_sorted_indices = True
    Xn.has_canonical_format = True
    ucells = _cells_from_hashes(uh, origin, extent, nt)
    return Lattice(ucells, uh, origin, extent), Xn
