"""Exact enumeration of molecule pairs at short distances (linear bins [0, Δ) .. [(n_distance_bins - 1) Δ,
n_distance_bins Δ)).

The points of every group (a tile of the pipeline) are sorted into square blocks of at least the largest distance;
every block is paired with itself and with its neighboring blocks of larger index in the same group, and the distance
of every candidate pair is binned. The counts are ordered pairs: each unordered pair of features (a, b) is counted in
both (a, b) and (b, a). The blocks of many groups are counted in one parallel region, distributed over numba's
current thread count with a cost-balanced plan, so that the number of parallel regions does not grow with the number
of groups; small problems run on one thread. The integer counts do not depend on the plan, the block size, the
grouping into batches, or the thread count.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from numba import get_num_threads, njit, prange

from .._tacco_compat import _get_indptr, _getNN, _guess_blocksize, bin, hash, min_dtype

__all__ = ["bin_intervals", "enumeration_block_size", "pair_counts_enumerated", "pair_counts_enumerated_groups"]

# candidate pairs (block cost estimate) below which a batch is counted on one thread, without a parallel region
ENUM_MIN_PARALLEL_WORK = 4e6
# bytes of the per-thread accumulators of one batch of groups (at least one group per batch)
ENUM_BATCH_BYTES = 64 * 2**20


def bin_intervals(delta: float, n_distance_bins: int) -> pd.DataFrame:
    r"""Distance bins of the enumeration.

    Parameters
    ----------
    delta : float
        Bin width (µm).
    n_distance_bins : int
        Number of bins.

    Returns
    -------
    pandas.DataFrame
        One row per bin :math:`[a, b)`: columns ``from`` (:math:`a`), ``to`` (:math:`b`), and ``central_distance``,
        the central distance of the bin, :math:`\tfrac{3}{4} (b^4 - a^4) / (b^3 - a^3)` (µm).
    """
    intervals = np.array([[i, i + 1] for i in range(n_distance_bins)]) * float(delta)
    a, b = intervals[:, 0], intervals[:, 1]
    central = (b ** 4 - a ** 4) / 4 / ((b ** 3 - a ** 3) / 3)
    return pd.DataFrame(np.concatenate([intervals, central[:, None]], axis=1),
                        columns=["from", "to", "central_distance"])


def _positions(xy: np.ndarray) -> pd.DataFrame:
    """The positions as the float32 table of the block construction.

    Parameters
    ----------
    xy : numpy.ndarray
        (n, 2) positions (µm).

    Returns
    -------
    pandas.DataFrame
        Columns x, y (float32).
    """
    return pd.DataFrame({"x": xy[:, 0].astype(np.float32), "y": xy[:, 1].astype(np.float32)})


def enumeration_block_size(xy: np.ndarray, delta: float, n_distance_bins: int = 8) -> float:
    """Block size of the enumeration chosen for a set of points.

    The smallest size of the form ``n_distance_bins * delta * 2^(i/2)`` (rounded to two significant digits) for which
    the fullest block holds at least 200 points, so that the dense block-pair problems are efficient while the memory
    stays small.

    Parameters
    ----------
    xy : numpy.ndarray
        (n, 2) positions (µm).
    delta : float
        Bin width (µm).
    n_distance_bins : int
        Number of linear distance bins.

    Returns
    -------
    float
        Block side (µm), at least ``n_distance_bins * delta``.
    """
    xy = np.asarray(xy)
    return _guess_blocksize(_positions(xy), float(delta) * n_distance_bins, verbose=0)


def _check_points(xy: np.ndarray, codes: np.ndarray, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Validate the positions and codes of the enumeration.

    Parameters
    ----------
    xy : numpy.ndarray
        (n, 2) positions (µm).
    codes : numpy.ndarray
        (n,) integer feature codes.
    n_features : int
        Number of features G.

    Returns
    -------
    xy, codes : numpy.ndarray
        The inputs as arrays.

    Raises
    ------
    ValueError
        If xy is not (n, 2), the lengths differ, or a code is outside 0 .. n_features - 1.
    """
    xy = np.asarray(xy)
    codes = np.asarray(codes)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must have shape (n, 2), got {xy.shape}")
    if len(codes) != len(xy):
        raise ValueError(f"xy has {len(xy)} rows but codes has {len(codes)} entries")
    if len(codes) and (codes.min() < 0 or codes.max() >= n_features):
        raise ValueError(f"codes must lie in 0 .. {n_features - 1}")
    return xy, codes


def _blocks(xy: np.ndarray, block_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Square blocks of one group of points.

    Parameters
    ----------
    xy : numpy.ndarray
        (n, 2) positions (µm), n >= 1.
    block_size : float
        Side of the blocks (µm).

    Returns
    -------
    order : numpy.ndarray
        (n,) the points in block order.
    positions : numpy.ndarray
        (n, 2) float32 positions in block order.
    indptr : numpy.ndarray
        (n_blocks + 1,) int64 block boundaries in ``order``.
    neighbors : numpy.ndarray
        (n_blocks, 9) int64 neighbor table: entry 0 = number of neighbor blocks of larger index, then their indices.
    """
    positions = _positions(xy)
    blocks = bin(positions, position_keys=positions.columns, bin_size=block_size)
    hashes = hash(blocks)
    hashes.reset_index(drop=True, inplace=True)
    hashes.sort_values(inplace=True)
    # hashes is sorted, so the points of every block are contiguous: indptr delimits them
    indptr = _get_indptr(hashes.to_numpy(), hashes.max(), min_dtype(len(hashes)))
    # neighbor blocks with a larger block index than each block
    sorted_blocks = hashes.drop_duplicates().sort_values()
    neighbors = _getNN(blocks.iloc[sorted_blocks.index.to_numpy()].to_numpy(), sorted_blocks.to_numpy())
    order = hashes.index.to_numpy()
    return (order, positions.iloc[order].to_numpy(), indptr.astype(np.int64),
            neighbors.astype(np.int64))


def _batch_structure(xy: np.ndarray, codes: np.ndarray, bounds: np.ndarray, groups: range, block_size: float,
                     code_dtype) -> tuple:
    """Blocks of the groups of one batch, concatenated with group-local block indices shifted to batch indices.

    Parameters
    ----------
    xy, codes : numpy.ndarray
        All points, sorted by group.
    bounds : numpy.ndarray
        Group boundaries in the points.
    groups : range
        The groups of the batch.
    block_size : float
        Side of the blocks (µm).
    code_dtype : numpy.dtype
        Integer dtype of the codes in the kernel.

    Returns
    -------
    codes, positions, indptr, neighbors, block_group : numpy.ndarray
        Sorted codes and float32 positions, block boundaries, neighbor table, and the group (index within the batch)
        of every block.
    """
    parts_codes, parts_pos, parts_ptr, parts_nb, parts_group = [], [], [np.zeros(1, np.int64)], [], []
    n_points = n_blocks = 0
    for g, t in enumerate(groups):
        lo, hi = int(bounds[t]), int(bounds[t + 1])
        if hi - lo < 2:
            continue
        order, positions, indptr, neighbors = _blocks(xy[lo:hi], block_size)
        nb = len(indptr) - 1
        columns = np.arange(neighbors.shape[1])[None, :]
        shift = (columns >= 1) & (columns <= neighbors[:, :1])
        parts_codes.append(codes[lo:hi][order])
        parts_pos.append(positions)
        parts_ptr.append(indptr[1:] + n_points)
        parts_nb.append(neighbors + shift * n_blocks)
        parts_group.append(np.full(nb, g, dtype=np.int64))
        n_points += hi - lo
        n_blocks += nb
    if not parts_nb:
        return None
    return (np.concatenate(parts_codes).astype(code_dtype), np.concatenate(parts_pos), np.concatenate(parts_ptr),
            np.concatenate(parts_nb), np.concatenate(parts_group))


def pair_counts_enumerated_groups(xy: np.ndarray, codes: np.ndarray, bounds: np.ndarray, n_features: int,
                                  delta: float, n_distance_bins: int = 8,
                                  block_size: float | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Ordered pair counts within each group of points (the tiles of the pipeline), by exact enumeration.

    Consecutive groups are counted together in batches, one parallel region per batch; the per-thread accumulators of
    a batch take at most ``ENUM_BATCH_BYTES`` (or one group).

    Parameters
    ----------
    xy : numpy.ndarray
        (n, 2) positions (µm), sorted by group. They are rounded to float32 before the distances are computed.
    codes : numpy.ndarray
        (n,) integer feature codes in 0 .. n_features - 1, in the order of ``xy``.
    bounds : numpy.ndarray
        (T + 1,) group boundaries: group t holds the points ``bounds[t]:bounds[t + 1]``.
    n_features : int
        Number of features G.
    delta : float
        Bin width (µm).
    n_distance_bins : int
        Number of linear distance bins.
    block_size : float or None
        Side of the blocks (µm), at least ``n_distance_bins * delta``; ``None`` chooses it for all points
        (``enumeration_block_size``). The counts do not depend on it.

    Yields
    ------
    t : int
        The group, in ascending order.
    counts : numpy.ndarray
        (n_distance_bins, G, G) int64 counts of group t; entry [k, a, b] counts the ordered pairs (molecule of feature
        a, molecule of feature b) at a distance in [k delta, (k + 1) delta).

    Raises
    ------
    ValueError
        If xy is not (n, 2), the lengths differ, a code is outside 0 .. n_features - 1, or the block size is smaller
        than the largest distance.
    """
    xy, codes = _check_points(xy, codes, n_features)
    bounds = np.asarray(bounds, dtype=np.int64)
    delta = float(delta)
    max_distance = delta * n_distance_bins
    if block_size is None:
        block_size = _guess_blocksize(_positions(xy), max_distance, verbose=0) if len(xy) >= 2 else max_distance
    if block_size - max_distance < -1e-15 * max_distance:
        raise ValueError(f"the block size {block_size} is smaller than the largest distance {max_distance}")
    T = len(bounds) - 1
    G = int(n_features)
    code_dtype = np.int8 if G <= 127 else np.int16 if G <= 32767 else np.int32
    n_threads = max(1, get_num_threads())
    group_bytes = n_threads * n_distance_bins * G * G * 8
    per_batch = max(1, int(ENUM_BATCH_BYTES // group_bytes))
    for start in range(0, T, per_batch):
        groups = range(start, min(T, start + per_batch))
        counts = np.zeros((len(groups), n_distance_bins, G, G), dtype=np.int64)
        batch = _batch_structure(xy, codes, bounds, groups, block_size, code_dtype)
        if batch is not None:
            b_codes, b_pos, indptr, neighbors, block_group = batch
            cost = _block_costs(indptr, neighbors)
            nt = min(n_threads, len(cost))
            if nt == 1 or cost.sum() < ENUM_MIN_PARALLEL_WORK:
                _count_blocks_serial(b_codes, indptr, b_pos, delta, neighbors, block_group, counts)
            else:
                plan_ptr, plan_blocks = _make_plan(cost, nt)
                per_thread = np.zeros((nt, *counts.shape), dtype=np.int64)
                _count_blocks_parallel(b_codes, indptr, b_pos, delta, neighbors, block_group, plan_ptr, plan_blocks,
                                       per_thread)
                counts = per_thread.sum(axis=0)
            counts += counts.transpose((0, 1, 3, 2)).copy()   # the kernel counts every unordered pair once
        for g, t in enumerate(groups):
            yield t, counts[g]


def pair_counts_enumerated(xy: np.ndarray, codes: np.ndarray, n_features: int, delta: float,
                           n_distance_bins: int = 8, block_size: float | None = None) -> np.ndarray:
    """Ordered pair counts of all molecule pairs closer than n_distance_bins * delta, by exact enumeration.

    Parameters
    ----------
    xy : numpy.ndarray
        (n, 2) positions (µm). They are rounded to float32 before the distances are computed.
    codes : numpy.ndarray
        (n,) integer feature codes in 0 .. n_features - 1.
    n_features : int
        Number of features G.
    delta : float
        Bin width (µm).
    n_distance_bins : int
        Number of linear distance bins.
    block_size : float or None
        Side of the blocks (µm), at least ``n_distance_bins * delta``; ``None`` chooses it for these points
        (``enumeration_block_size``). The counts do not depend on it.

    Returns
    -------
    numpy.ndarray
        (n_distance_bins, G, G) int64 counts; entry [k, a, b] counts the ordered pairs (molecule of feature a,
        molecule of feature b) at a distance in [k delta, (k + 1) delta). The matrix of every bin is symmetric.

    Raises
    ------
    ValueError
        If xy is not (n, 2), the lengths differ, a code is outside 0 .. n_features - 1, or the block size is smaller
        than the largest distance.
    """
    xy, codes = _check_points(xy, codes, n_features)
    groups = pair_counts_enumerated_groups(xy, codes, np.array([0, len(xy)]), n_features, delta, n_distance_bins,
                                           block_size=block_size)
    return next(groups)[1]


# ----------------------------------------------------------------------------- numba kernels
@njit(cache=True)
def _block_costs(indptr, neighbors):
    """Cost estimate of every block i, n_i (n_i + sum of the sizes of its neighbor blocks).

    Parameters
    ----------
    indptr : numpy.ndarray
        Block boundaries in the sorted points.
    neighbors : numpy.ndarray
        Neighbor table (one row per block: entry 0 = number of neighbors, then their block indices).

    Returns
    -------
    numpy.ndarray
        (n_blocks,) float64 costs.
    """
    n_blocks = len(indptr) - 1
    cost = np.empty(n_blocks, dtype=np.float64)
    for i in range(n_blocks):
        n = indptr[i + 1] - indptr[i]
        c = n
        for ib in range(1, neighbors[i, 0] + 1):
            j = neighbors[i, ib]
            c += indptr[j + 1] - indptr[j]
        cost[i] = float(c) * float(n)
    return cost


@njit(cache=True)
def _make_plan(cost, n_threads):
    """Cost-balanced assignment of the blocks to threads (greedy: each block goes to the thread with the lowest total
    cost so far).

    Parameters
    ----------
    cost : numpy.ndarray
        (n_blocks,) cost of every block.
    n_threads : int
        Number of threads, at most the number of blocks.

    Returns
    -------
    plan_ptr : numpy.ndarray
        (n_threads + 1,) int64: the blocks of thread t are ``plan_blocks[plan_ptr[t]:plan_ptr[t + 1]]``.
    plan_blocks : numpy.ndarray
        (n_blocks,) int64 block indices, grouped by thread, ascending within a thread.
    """
    n = len(cost)
    total = np.zeros(n_threads, dtype=np.float64)
    thread_of = np.empty(n, dtype=np.int64)
    for i in range(n):
        t = np.argmin(total)
        total[t] += cost[i]
        thread_of[i] = t
    plan_ptr = np.zeros(n_threads + 1, dtype=np.int64)
    for i in range(n):
        plan_ptr[thread_of[i] + 1] += 1
    for t in range(n_threads):
        plan_ptr[t + 1] += plan_ptr[t]
    fill = plan_ptr[:-1].copy()
    plan_blocks = np.empty(n, dtype=np.int64)
    for i in range(n):
        t = thread_of[i]
        plan_blocks[fill[t]] = i
        fill[t] += 1
    return plan_ptr, plan_blocks


@njit(cache=True, fastmath=True)
def _count_pairs_within_block(points, codes, counts):
    """Add the pairs within one block to the counts (each unordered pair once, a before b).

    Parameters
    ----------
    points : numpy.ndarray
        (n, 2) positions in units of the bin width.
    codes : numpy.ndarray
        (n,) feature codes.
    counts : numpy.ndarray
        (n_distance_bins, G, G) counts, updated in place at [bin, code of a, code of b].
    """
    n_distance_bins = counts.shape[0]
    n_points = points.shape[0]
    max_distance2 = float(n_distance_bins**2)
    for a in range(n_points):
        for b in range(a + 1, n_points):
            dist2 = 0.0
            for c in range(2):
                diff = points[a, c] - points[b, c]
                diff *= diff
                dist2 += diff
            if dist2 < max_distance2:
                k = int(np.floor(np.sqrt(dist2)))
                counts[k, codes[a], codes[b]] += 1


@njit(cache=True, fastmath=True)
def _count_pairs_point_block(point, block_T, counts_row, block_codes):
    """Add the pairs of one point with all points of a neighbor block to the counts of the point's feature.

    Parameters
    ----------
    point : numpy.ndarray
        (2,) position in units of the bin width.
    block_T : numpy.ndarray
        (2, m) positions of the neighbor block, transposed (contiguous coordinates for vectorization).
    counts_row : numpy.ndarray
        (n_distance_bins, G) counts of the point's feature, updated in place at [bin, code of the neighbor].
    block_codes : numpy.ndarray
        (m,) feature codes of the neighbor block.
    """
    n_distance_bins = counts_row.shape[0]
    n_block = block_T.shape[1]
    max_distance2 = float(n_distance_bins**2)
    for b in range(n_block):
        dist2 = 0.0
        for c in range(2):
            diff = point[c] - block_T[c, b]
            diff *= diff
            dist2 += diff
        if dist2 < max_distance2:
            k = int(np.floor(np.sqrt(dist2)))
            counts_row[k, block_codes[b]] += 1


@njit(cache=True, fastmath=True)
def _count_pairs_between_blocks(points_a, points_b, codes_a, codes_b, counts):
    """Add the pairs between block a and a neighbor block b to the counts.

    Parameters
    ----------
    points_a : numpy.ndarray
        (n_a, 2) positions of block a in units of the bin width.
    points_b : numpy.ndarray
        (n_b, 2) positions of block b.
    codes_a : numpy.ndarray
        (n_a,) feature codes of block a.
    codes_b : numpy.ndarray
        (n_b,) feature codes of block b.
    counts : numpy.ndarray
        (n_distance_bins, G, G) counts, updated in place at [bin, code in a, code in b].
    """
    block_T = np.ascontiguousarray(points_b.T)  # contiguous memory accesses are faster and enable vectorization
    for a in range(points_a.shape[0]):
        _count_pairs_point_block(points_a[a], block_T, counts[:, codes_a[a]], codes_b)


@njit(cache=True)
def _count_block(i, codes, indptr, positions, bin_width, neighbors_i, counts):
    """All pairs of block i: within the block and with its neighbor blocks of larger index.

    Parameters
    ----------
    i : int
        Block index.
    codes : numpy.ndarray
        (n,) feature codes of the sorted points.
    indptr : numpy.ndarray
        Block boundaries.
    positions : numpy.ndarray
        (n, 2) sorted positions (µm).
    bin_width : float
        Width of the distance bins (µm); the positions are divided by it.
    neighbors_i : numpy.ndarray
        Row i of the neighbor table.
    counts : numpy.ndarray
        (n_distance_bins, G, G) counts, updated in place.
    """
    codes_i = codes[indptr[i]:indptr[i + 1]]
    points_i = positions[indptr[i]:indptr[i + 1]] / bin_width
    _count_pairs_within_block(points_i, codes_i, counts)        # self-pairs excluded
    for ib in range(1, neighbors_i[0] + 1):
        j = neighbors_i[ib]
        codes_j = codes[indptr[j]:indptr[j + 1]]
        points_j = positions[indptr[j]:indptr[j + 1]] / bin_width
        _count_pairs_between_blocks(points_i, points_j, codes_i, codes_j, counts)


@njit(cache=True)
def _count_blocks_serial(codes, indptr, positions, bin_width, neighbors, block_group, counts):
    """Pair counts of all blocks on the calling thread.

    Parameters
    ----------
    codes : numpy.ndarray
        (n,) feature codes of the sorted points.
    indptr : numpy.ndarray
        Block boundaries.
    positions : numpy.ndarray
        (n, 2) sorted positions (µm).
    bin_width : float
        Width of the distance bins (µm).
    neighbors : numpy.ndarray
        Neighbor table.
    block_group : numpy.ndarray
        (n_blocks,) group of every block.
    counts : numpy.ndarray
        (n_groups, n_distance_bins, G, G) counts, each unordered pair counted once, updated in place.
    """
    for i in range(len(indptr) - 1):
        _count_block(i, codes, indptr, positions, bin_width, neighbors[i], counts[block_group[i]])


@njit(cache=True, parallel=True)
def _count_blocks_parallel(codes, indptr, positions, bin_width, neighbors, block_group, plan_ptr, plan_blocks,
                           counts_per_thread):
    """Pair counts of all blocks in one parallel region, thread t counting its planned blocks into its own
    accumulator.

    Parameters
    ----------
    codes, indptr, positions, bin_width, neighbors, block_group
        As in ``_count_blocks_serial``.
    plan_ptr, plan_blocks : numpy.ndarray
        The plan of ``_make_plan``.
    counts_per_thread : numpy.ndarray
        (n_threads, n_groups, n_distance_bins, G, G) zeros, one accumulator per plan row, updated in place.
    """
    for t in prange(counts_per_thread.shape[0]):
        acc = counts_per_thread[t]
        for p in range(plan_ptr[t], plan_ptr[t + 1]):
            i = plan_blocks[p]
            _count_block(i, codes, indptr, positions, bin_width, neighbors[i], acc[block_group[i]])
