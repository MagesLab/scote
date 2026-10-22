"""Lattice levels (scote._twopoint.lattice) and the CPU shell products (scote._twopoint.cpu): brute-force counts, the
pair enumeration, and bit-identical results across strategies, host kernels, and thread counts."""
import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.spatial import cKDTree

import scote
from scote import random_label_reference, thread_limit
from scote import CPUOptions
from scote._twopoint import LevelTiming, cpu, lattice as LAT
from scote._twopoint._common import CPU_MODES
from scote._twopoint.cpu import pair_counts_level as _pair_counts_level
from scote._twopoint.enumeration import pair_counts_enumerated
from scote._twopoint.lattice import build_lattice, coarse_grain, onehot_csr, shell_offsets, shell_operator
from _helpers import max_threads


def pair_counts_level(X, lat, n_shells, shells=None, mode="auto", host="numba", **kw):
    """The CPU backend with the options given as keywords."""
    return _pair_counts_level(X, lat, n_shells, shells, options=CPUOptions(mode=mode, host=host), **kw)


def _molecules(n=40_000, G=6, side=300.0, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, side, (n, 2))
    p = 1 / np.arange(1, G + 1)
    p /= p.sum()
    codes = rng.choice(G, n, p=p)
    return xy, codes


def _lattice(n_mol, G, side, pitch=2.0, seed=0, max_count=1):
    """Synthetic low-density lattice: n_mol molecules (Zipf genes) binned to cells of `pitch`."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, side, (n_mol, 2))
    p = 1 / np.arange(1, G + 1)
    p /= p.sum()
    codes = rng.choice(G, n_mol, p=p)
    vals = rng.integers(1, max_count + 1, n_mol).astype(np.float64)
    cells = np.floor(xy / pitch).astype(np.int64)
    X0 = sp.csr_matrix((vals, (np.arange(n_mol), codes)), shape=(n_mol, G))
    return xy, codes, cells, X0


def _hierarchy(X0, cells, n_levels, mode, host="numba"):
    """Global counts of the lattice hierarchy level by level with an explicit host-kernel choice."""
    lat, X = build_lattice(cells, X0, 8, host=host)
    out = []
    for level in range(n_levels + 1):
        keep = list(range(8)) if level == 0 else [4, 5, 6, 7]
        out.append(pair_counts_level(X, lat, 8, keep, mode=mode, include_intra=level == 0, host=host))
        lat, X = coarse_grain(lat, X, 2, 8, host=host)
    return np.concatenate(out)


@pytest.fixture
def four_threads():
    with thread_limit(max_threads(4)) as n:
        yield n


@pytest.fixture
def force_numba_build(monkeypatch):
    monkeypatch.setattr(LAT, "LATTICE_MIN_NUMBA_N", 0)


# ----------------------------------------------------------------------------- brute force
def test_shell_offsets_partition():
    offs = shell_offsets(8)
    allo = np.concatenate([offs[k] for k in range(8)])
    assert len(np.unique(allo, axis=0)) == len(allo)
    for k in range(1, 8):
        d = np.floor(np.hypot(*offs[k].T.astype(float)))
        assert (d == k).all()
    assert len(offs[0]) == 0  # (0,0) excluded


def test_level_counts_match_bruteforce():
    """One lattice level: X^T S_k X equals brute-force counting of cell pairs by floor(lattice distance)."""
    xy, codes = _molecules(n=20_000, G=4, side=120.0)
    G, h, n_bins = 4, 2.0, 8
    cells = np.floor((xy - xy.min(0)) / h).astype(np.int64)
    lat, X = build_lattice(cells, onehot_csr(codes, G), margin=n_bins)
    P = pair_counts_level(X, lat, n_bins, mode="sparse", include_intra=True)
    Pd = pair_counts_level(X, lat, n_bins, mode="dense", include_intra=True)
    assert np.array_equal(P, Pd)
    C = lat.cells.astype(float)
    tree = cKDTree(C)
    pairs = tree.query_pairs(r=n_bins - 1e-9, output_type="ndarray")
    d = np.floor(np.hypot(*(C[pairs[:, 0]] - C[pairs[:, 1]]).T)).astype(int)
    Xd = X.toarray()
    B = np.zeros_like(P)
    for k in range(1, n_bins):
        sel = d == k
        if sel.any():
            M = Xd[pairs[sel, 0]].T @ Xd[pairs[sel, 1]]
            B[k] = M + M.T
    B[0] = Xd.T @ Xd - np.diag(Xd.sum(0))
    assert np.array_equal(P.astype(np.int64), B.astype(np.int64))
    ref = random_label_reference(P, np.asarray(X.sum(0)).ravel())
    assert np.allclose(ref.sum(axis=(1, 2)), P.sum(axis=(1, 2)))


@pytest.mark.parametrize("threads", [1, 4])
def test_enumeration_matches_kdtree(threads):
    """Pair enumeration (level 0 of the molecule pipeline) equals KD-tree brute force, ordered pairs."""
    xy, codes = _molecules(n=30_000, G=5, side=200.0, seed=1)
    G, delta, n_bins = 5, 0.5, 8
    with thread_limit(max_threads(threads)):
        got = pair_counts_enumerated(xy, codes, G, delta, n_bins)
    # the enumeration rounds the positions to float32 and bins in float64 units of delta: do the same
    u = xy.astype(np.float32).astype(np.float64) / delta
    pairs = cKDTree(u).query_pairs(r=n_bins + 1e-6, output_type="ndarray")
    dd = u[pairs[:, 0]] - u[pairs[:, 1]]
    d2 = dd[:, 0] * dd[:, 0] + dd[:, 1] * dd[:, 1]
    ok = d2 < float(n_bins ** 2)
    k = np.floor(np.sqrt(d2[ok])).astype(int)
    B = np.zeros((n_bins, G, G), dtype=np.int64)
    np.add.at(B, (k, codes[pairs[ok, 0]], codes[pairs[ok, 1]]), 1)
    B = B + B.transpose(0, 2, 1)
    assert got.dtype == np.int64 and np.array_equal(B, got)


@pytest.mark.parametrize("threads", [1, 4])
def test_grouped_enumeration_equals_the_groups_one_by_one(threads, monkeypatch):
    """Groups counted together (one batch, several batches, on one thread or in a parallel region) give the counts
    of each group counted on its own; groups with fewer than two points give zeros."""
    from scote._twopoint import enumeration as E
    xy, codes = _molecules(n=20_000, G=5, side=200.0, seed=2)
    group = np.floor(xy[:, 0] / 50.0).astype(int) * 4 + np.floor(xy[:, 1] / 50.0).astype(int)
    group[group == 5] = 6                                           # an empty group
    group[np.flatnonzero(group == 6)[0]] = 5                        # ... with one point
    order = np.argsort(group, kind="stable")
    bounds = np.searchsorted(group[order], np.arange(17))
    xs, cs = xy[order], codes[order]
    expected = [E.pair_counts_enumerated(xs[bounds[t]:bounds[t + 1]], cs[bounds[t]:bounds[t + 1]], 5, 0.5,
                                         block_size=8.0) for t in range(16)]
    for batch_bytes, min_work in ((E.ENUM_BATCH_BYTES, E.ENUM_MIN_PARALLEL_WORK), (1, 0.0), (2**30, 0.0)):
        monkeypatch.setattr(E, "ENUM_BATCH_BYTES", batch_bytes)
        monkeypatch.setattr(E, "ENUM_MIN_PARALLEL_WORK", min_work)
        with thread_limit(max_threads(threads)):
            got = list(E.pair_counts_enumerated_groups(xs, cs, bounds, 5, 0.5, block_size=8.0))
        assert [t for t, _ in got] == list(range(16))
        for t, c in got:
            assert c.dtype == np.int64 and np.array_equal(c, expected[t]), (batch_bytes, t)
    assert not expected[5].any() and expected[6].sum() > 0


def test_enumeration_rejects_bad_input():
    with pytest.raises(ValueError):
        pair_counts_enumerated(np.zeros((3, 3)), np.zeros(3, int), 1, 0.5)
    with pytest.raises(ValueError):
        pair_counts_enumerated(np.zeros((3, 2)), np.array([0, 1, 2]), 2, 0.5)
    assert not pair_counts_enumerated(np.zeros((1, 2)), np.zeros(1, int), 2, 0.5).any()


# ----------------------------------------------------------------------------- numba lattice construction
def _same(la, Xa, lb, Xb):
    assert np.array_equal(la.hashes, lb.hashes) and np.array_equal(la.cells, lb.cells)
    assert np.array_equal(la.origin, lb.origin) and np.array_equal(la.extent, lb.extent)
    Xa = sp.csr_matrix(Xa, dtype=np.float64)
    Xa.sort_indices()
    Xa.sum_duplicates()
    assert Xb.has_sorted_indices and Xb.has_canonical_format
    assert np.array_equal(Xa.indptr, Xb.indptr) and np.array_equal(Xa.indices, Xb.indices)
    assert np.array_equal(Xa.data, Xb.data)          # bitwise: same summation order as scipy's csr_matmat
    assert Xb.data.dtype == np.float64


@pytest.mark.parametrize("nt", [1, 4])
@pytest.mark.parametrize("ndim,G,N", [(2, 20, 5_000), (2, 200, 60_000), (2, 3000, 20_000), (3, 50, 30_000)])
def test_build_lattice_numba_equals_numpy(force_numba_build, nt, ndim, G, N):
    with thread_limit(max_threads(nt)):
        rng = np.random.default_rng(N + ndim)
        cells = np.floor(rng.uniform(0, 60, (N, ndim))).astype(np.int64) - 7
        X0 = onehot_csr(rng.integers(0, G, N), G)
        la, Xa = build_lattice(cells, X0, 8, host="numpy")
        lb, Xb = build_lattice(cells, X0, 8)
        _same(la, Xa, lb, Xb)
        for _ in range(3):                           # coarse-graining: aggregation of multi-column rows
            la, Xa = build_lattice(np.floor_divide(la.cells, 2), Xa, 8, host="numpy")
            lb, Xb = coarse_grain(lb, Xb, 2, 8)
            _same(la, Xa, lb, Xb)


def test_build_lattice_numba_float_data_and_unique_rows(force_numba_build):
    rng = np.random.default_rng(3)
    N, G = 20_000, 40
    cells = rng.integers(-5, 25, (N, 3))
    Xr = sp.random(N, G, density=0.08, format="csr", random_state=1, dtype=np.float64)   # non-integer data
    _same(*build_lattice(cells, Xr, 4, host="numpy"), *build_lattice(cells, Xr, 4))
    cells = np.stack(np.unravel_index(np.arange(N), (200, 100)), 1)                      # all cells distinct
    Xi = sp.random(N, G, density=0.1, format="csr", random_state=2)
    Xi.data = np.round(Xi.data * 5)
    _same(*build_lattice(cells, Xi, 8, host="numpy"), *build_lattice(cells, Xi, 8))
    with pytest.raises(ValueError):
        build_lattice(cells, Xi, 8, host="fortran")


def test_onehot_csr_equals_coo_construction():
    codes = np.random.default_rng(0).integers(0, 7, 1000)
    A = sp.csr_matrix((np.ones(1000), (np.arange(1000), codes)), shape=(1000, 7))
    B = onehot_csr(codes, 7)
    assert np.array_equal(A.indptr, B.indptr) and np.array_equal(A.indices, B.indices)
    assert np.array_equal(A.data, B.data)


@pytest.mark.parametrize("nt", [1, 4])
def test_hierarchy_counts_identical_numba_build(force_numba_build, nt):
    with thread_limit(max_threads(nt)):
        rng = np.random.default_rng(11)
        N, G = 150_000, 30
        xy = rng.uniform(0, 800, (N, 2))
        codes = rng.integers(0, G, N)
        cells = np.floor(xy / 2.0).astype(np.int64)
        X0 = onehot_csr(codes, G)
        Pa = _hierarchy(X0, cells, 4, "auto", host="numpy")
        Pb = _hierarchy(X0, cells, 4, "auto", host="numba")
        assert np.array_equal(Pa, Pb) and Pa.sum() > 0


# ----------------------------------------------------------------------------- CPU strategies
@pytest.mark.parametrize("G,n_mol,side", [(5, 50_000, 500.0), (50, 300_000, 1500.0), (400, 120_000, 1200.0)])
def test_one_level_all_shells_numba_equals_sparse(G, n_mol, side, four_threads):
    _, _, cells, X0 = _lattice(n_mol, G, side, max_count=3)
    lat, X = build_lattice(cells, X0, margin=8)
    assert G < 50 or X.nnz / (X.shape[0] * G) < 0.05            # low density: the regime the kernel targets
    tm_n = LevelTiming(0, 2.0, 0, 0, 0.0, "numba")
    tm_s = LevelTiming(0, 2.0, 0, 0, 0.0, "sparse")
    Pn = pair_counts_level(X, lat, 8, mode="numba", include_intra=True, timing=tm_n)
    Ps = pair_counts_level(X, lat, 8, mode="sparse", include_intra=True, timing=tm_s)
    assert Pn.shape == (8, G, G) and Pn.dtype == Ps.dtype
    assert np.array_equal(Pn, Ps)                                # bit-identical
    assert tm_n.mode == "numba" and tm_n.pair_updates == tm_s.pair_updates and tm_n.cell_pairs == tm_s.cell_pairs
    Pn2 = pair_counts_level(X, lat, 8, shells=[0, 3, 7], mode="numba", include_intra=False)
    assert np.array_equal(Pn2[1:], Ps[[3, 7]]) and not Pn2[0].any()


def test_rows_per_tile_path_numba_equals_sparse(four_threads):
    """row_groups (per-tile slabs, shared shell-operator cache, row subsets) incl. the intra shell."""
    G = 30
    _, _, cells, X0 = _lattice(150_000, G, 1000.0, max_count=2)
    lat, X = build_lattice(cells, X0, margin=8)
    groups = (lat.cells[:, 0] // 60) * 40 + lat.cells[:, 1] // 60
    shells = [0, 4, 5, 6, 7]
    Pn = pair_counts_level(X, lat, 8, shells, mode="numba", include_intra=True, row_groups=groups)
    Ps = pair_counts_level(X, lat, 8, shells, mode="sparse", include_intra=True, row_groups=groups)
    assert Pn.shape == (len(np.unique(groups)), 5, G, G)
    assert np.array_equal(Pn, Ps)
    assert np.array_equal(Pn.sum(0), pair_counts_level(X, lat, 8, shells, mode="numba", include_intra=True))


@pytest.mark.parametrize("mode", ["numba", "dense", "auto"])
def test_pair_counts_modes_equal_sparse(mode, four_threads):
    G = 40
    xy, codes, _, X0 = _lattice(200_000, G, 1200.0)
    m = scote.pair_counts(xy, X0, delta=2.0, bin_size=2.0, n_levels=5, backend=CPUOptions(mode=mode))
    s = scote.pair_counts(xy, X0, delta=2.0, bin_size=2.0, n_levels=5, backend=CPUOptions(mode="sparse"))
    assert m.counts.shape == (8 + 4 * 5, G, G) and np.array_equal(m.counts, s.counts) and m.bins.equals(s.bins)
    assert m.params["backend"] == "cpu" and m.params["options"]["mode"] == mode and m.n_distance_bins == 28
    if mode == "auto":         # numba at the sparse fine levels, BLAS where the density exceeds 5 %
        modes = [t["mode"] for t in m.timings["levels"]]
        assert modes[0] == "numba" and modes[-1] == "dense" and set(modes) == {"numba", "dense"}


def test_twopoint_bins_cpu_modes_identical(four_threads):
    rng = np.random.default_rng(5)
    n, G, pitch = 60_000, 6, 2.0
    xy = rng.uniform(0, 400, (n, 2))
    codes = rng.choice(G, n, p=np.array([6, 5, 4, 3, 2, 1]) / 21)
    X0 = onehot_csr(codes, G)
    with pytest.warns(UserWarning, match="fewer than 30 tiles"):
        res = {m: scote.twopoint(xy, X0, delta=pitch, bin_size=pitch, tile_um=100.0, n_levels=5,
                                 backend=CPUOptions(mode=m), keep_tiles=True)
               for m in ("numba", "sparse", "auto")}
    assert res["numba"].counts_tiles.shape == (16, 8 + 4 * 5, G, G)
    for m in ("sparse", "auto"):
        assert np.array_equal(res["numba"].counts_tiles, res[m].counts_tiles)
        assert np.array_equal(res["numba"].sd, res[m].sd, equal_nan=True)


@pytest.mark.skipif(max_threads(4) < 2, reason="numba has a single thread (NUMBA_NUM_THREADS=1): nothing to compare")
def test_one_vs_four_threads_identical():
    G = 60
    _, _, cells, X0 = _lattice(200_000, G, 1200.0, max_count=3)
    lat, X = build_lattice(cells, X0, margin=8)
    with thread_limit(1):
        P1 = pair_counts_level(X, lat, 8, mode="numba", include_intra=True)
    with thread_limit(max_threads(4)):
        P4 = pair_counts_level(X, lat, 8, mode="numba", include_intra=True)
    assert np.array_equal(P1, P4)
    assert np.array_equal(P1, pair_counts_level(X, lat, 8, mode="sparse", include_intra=True))


def test_memory_limit_falls_back_to_sparse(monkeypatch, four_threads):
    G = 20
    _, _, cells, X0 = _lattice(20_000, G, 300.0)
    lat, X = build_lattice(cells, X0, margin=8)
    Ps = pair_counts_level(X, lat, 8, mode="sparse", include_intra=True)
    limit = four_threads * G * G * 8 - 1
    tm = LevelTiming(0, 2.0, 0, 0, 0.0, "numba")
    with pytest.warns(RuntimeWarning, match="falling back to mode 'sparse'"):
        Pn = _pair_counts_level(X, lat, 8, include_intra=True, timing=tm,
                                options=CPUOptions(mode="numba", numba_acc_bytes=limit))
    assert tm.mode == "sparse" and np.array_equal(Pn, Ps)
    assert not cpu.numba_mode_available(G, four_threads, limit) and cpu.numba_mode_available(G - 1, four_threads,
                                                                                             limit)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.array_equal(pair_counts_level(X, lat, 8, mode="numba", include_intra=True), Ps)


def test_unknown_mode_or_backend_raises():
    xy, _, cells, X0 = _lattice(1_000, 3, 50.0)
    lat, X = build_lattice(cells, X0, margin=8)
    with pytest.raises(ValueError, match="CPU mode must be one of"):
        pair_counts_level(X, lat, 8, mode="fast")
    with pytest.raises(ValueError, match="backend must be 'cpu', 'gpu', a scote.CPUOptions, .* got 'tpu'"):
        scote.twopoint(xy, X0, delta=2.0, bin_size=2.0, backend="tpu")
    with pytest.raises(TypeError, match=r"got dict; pass the fields as scote.CPUOptions\(\*\*fields\)"):
        scote.pair_counts(xy, X0, delta=2.0, bin_size=2.0, backend={"mode": "sparse"})
    with pytest.raises(TypeError, match=r"got the class GPUOptions; pass an instance, e.g. scote.GPUOptions\(\)"):
        scote.twopoint(xy, X0, delta=2.0, bin_size=2.0, backend=scote.GPUOptions)
    with pytest.raises(TypeError, match="got NoneType"):
        scote.pair_counts(xy, X0, delta=2.0, bin_size=2.0, backend=None)


@pytest.mark.parametrize("nt", [1, 4])
def test_host_kernels_numba_equal_numpy(nt):
    """The parallel host kernels (shell-operator merge build, sparse x dense product of mode 'dense') reproduce the
    serial NumPy/SciPy construction exactly, and the counts are bit-identical in every CPU mode, incl. row groups."""
    with thread_limit(max_threads(nt)):
        rng = np.random.default_rng(3)
        for N, side in [(400, 40.0), (60_000, 600.0)]:                      # below / above HOST_MIN_PARALLEL_WORK
            G = 6
            xy = rng.uniform(0, side, (N, 2))
            cells = np.floor(xy / 2.0).astype(np.int64)
            X0 = sp.csr_matrix((rng.integers(1, 3, N).astype(np.float64), (np.arange(N), rng.integers(0, G, N))),
                               shape=(N, G))
            lat, X = build_lattice(cells, X0, margin=8)
            offs = shell_offsets(8)
            for k in (1, 4, 7):
                S0 = shell_operator(lat, offs[k], host="numpy")
                S1 = shell_operator(lat, offs[k], host="numba")
                assert S0.indices.dtype == S1.indices.dtype and S0.indptr.dtype == S1.indptr.dtype
                assert np.array_equal(S0.indptr, S1.indptr) and np.array_equal(S0.indices, S1.indices)
                assert np.array_equal(S0.data, S1.data) and S1.has_canonical_format
                Xd = X.toarray()
                assert np.array_equal(S0 @ Xd, cpu._spmm(S1, Xd))
                rows = np.sort(rng.choice(lat.n_cells, lat.n_cells // 3, replace=False))
                assert np.array_equal(S0[rows] @ Xd, cpu._spmm(S1[rows], Xd))
            res = {}
            for host in ("numpy", "numba"):
                res[host] = [_hierarchy(X0, cells, 3, m, host=host) for m in ("auto", "dense", "sparse")]
                groups = (lat.cells[:, 0] // 50) % 4
                res[host].append(pair_counts_level(X, lat, 8, mode="dense", row_groups=groups, host=host))
            for a, b in zip(res["numpy"], res["numba"]):
                assert np.array_equal(a, b)
        with pytest.raises(ValueError):
            shell_operator(lat, offs[1], host="fortran")


@pytest.mark.parametrize("n_mol,side,holes", [(2_000, 100.0, False), (60_000, 600.0, True), (400_000, 2000.0, False)])
def test_shell_operator_numba_identical_to_numpy_sparse_lattices(n_mol, side, holes, four_threads):
    """The merge-based numba build of S_k equals the NumPy construction entry for entry, also on lattices with empty
    stripes and an empty block (hash gaps)."""
    xy, codes, cells, X0 = _lattice(n_mol, 3, side)
    if holes:
        keep = ((cells[:, 0] // 7) % 3 != 1) & ~((cells[:, 0] > 60) & (cells[:, 0] < 120) & (cells[:, 1] < 150))
        cells, X0 = cells[keep], X0[keep]
    lat, X = build_lattice(cells, X0, margin=8)
    offs = shell_offsets(8, 2)
    for k in range(8):
        A = shell_operator(lat, offs[k], host="numpy")
        A.sort_indices()
        B = shell_operator(lat, offs[k], host="numba")
        assert A.shape == B.shape and A.nnz == B.nnz
        assert np.array_equal(A.indptr, B.indptr) and np.array_equal(A.indices, B.indices)
        assert np.array_equal(A.data, B.data) and B.has_canonical_format


def test_hierarchy_identical_host_kernels_and_threads(four_threads):
    """Host kernels 'numpy' vs 'numba' (4 threads) vs 'numba' (1 thread): bit-identical counts."""
    xy, codes, cells, X0 = _lattice(150_000, 20, 1000.0, max_count=3)
    Pa = _hierarchy(X0, cells, 4, "auto", host="numpy")
    Pb = _hierarchy(X0, cells, 4, "auto", host="numba")
    with thread_limit(1):
        Pc = _hierarchy(X0, cells, 4, "auto", host="numba")
    assert np.array_equal(Pa, Pb) and np.array_equal(Pa, Pc)


def test_intra_shell_central_distance_label():
    """The same-cell shell (level 0, k = 0) is labeled with half the pitch, the middle of its interval."""
    xy, _, _, X0 = _lattice(5_000, 4, 200.0, pitch=0.5)
    bins = scote.pair_counts(xy, X0, delta=0.5, bin_size=0.5, n_levels=2, backend=CPUOptions(mode="dense")).bins
    first = bins[(bins.level == 0) & (bins.shell == 0)].iloc[0]
    assert first["r_central"] == 0.25


@pytest.mark.parametrize("batch_bytes", [None, 8 * 6 * 6_000])
def test_row_groups_shell_outer_matches_per_group_and_whole(batch_bytes, monkeypatch):
    """Per-group slabs equal the whole level and the groups computed one by one; mode "dense" also when its
    products of consecutive groups are split into several batches."""
    if batch_bytes is not None:
        monkeypatch.setattr(cpu, "DENSE_BATCH_BYTES", batch_bytes)
    rng = np.random.default_rng(4)
    n, G, h = 30_000, 6, 2.0
    xy = rng.uniform(0, 300, (n, 2))
    codes = rng.choice(G, n)
    lat, X = build_lattice(np.floor(xy / h).astype(np.int64), onehot_csr(codes, G), margin=8)
    center = (lat.cells + 0.5) * h
    tiles = np.floor(center[:, 0] / 100).astype(np.int64) * 10 + np.floor(center[:, 1] / 100).astype(np.int64)
    whole = pair_counts_level(X, lat, 8, mode="sparse", include_intra=True)
    for mode in CPU_MODES:
        Pg = pair_counts_level(X, lat, 8, mode=mode, include_intra=True, row_groups=tiles)
        assert np.array_equal(Pg.sum(0), whole), mode
        for gi, g in enumerate(np.unique(tiles)):
            rows = np.nonzero(tiles == g)[0]
            direct = cpu._pair_counts_rows(X, lat, 8, CPUOptions(mode=mode), list(range(8)), True, None, rows)
            assert np.array_equal(Pg[gi], direct), (mode, g)


def test_bin_intervals_and_shell_central_distances():
    from scote._twopoint.enumeration import bin_intervals
    iv = bin_intervals(0.5, 4)
    assert list(iv.columns) == ["from", "to", "central_distance"]
    assert np.array_equal(iv["from"], [0.0, 0.5, 1.0, 1.5]) and np.array_equal(iv["to"], [0.5, 1.0, 1.5, 2.0])
    a, b = iv["from"].to_numpy(), iv["to"].to_numpy()
    np.testing.assert_allclose(iv["central_distance"], 3 / 4 * (b**4 - a**4) / (b**3 - a**3), rtol=1e-15)
    assert iv["central_distance"][0] == pytest.approx(3 / 8)
    cd = LAT.shell_central_distances(6)
    offs = shell_offsets(6)
    assert cd[0] == 0.5
    for k in range(1, 6):
        assert cd[k] == np.sqrt((offs[k].astype(float) ** 2).sum(1)).mean()      # the mean offset length
    cd3 = LAT.shell_central_distances(4, ndim=3)
    assert cd3[0] == 0.5 and cd3[1] == pytest.approx(np.linalg.norm(shell_offsets(4, 3)[1], axis=1).mean())


def test_shell_offsets_are_cached_and_read_only():
    a, b = shell_offsets(8), shell_offsets(8)
    assert a is not b and a[3] is b[3] and not a[3].flags.writeable
    a.pop(3)                                                          # the cached table is not affected
    assert 3 in shell_offsets(8)


def test_onehot_csr_width_from_the_codes():
    assert onehot_csr(np.array([0, 2, 1])).shape == (3, 3)
    assert onehot_csr(np.array([0, 2, 1]), 5).shape == (3, 5)
    with pytest.raises(ValueError):
        onehot_csr(np.array([0, 3]), 3)


def test_enumeration_counts_do_not_depend_on_the_block_size():
    from scote._twopoint.enumeration import enumeration_block_size
    xy, codes = _molecules(n=20_000, G=4, side=150.0, seed=3)
    auto = pair_counts_enumerated(xy, codes, 4, 0.5, 8)
    assert enumeration_block_size(xy, 0.5, 8) >= 4.0
    for size in (4.0, 5.7, 23.0):
        assert np.array_equal(pair_counts_enumerated(xy, codes, 4, 0.5, 8, block_size=size), auto)
    with pytest.raises(ValueError, match="block size"):
        pair_counts_enumerated(xy, codes, 4, 0.5, 8, block_size=3.0)
