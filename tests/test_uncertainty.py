"""Closed-form tile block bootstrap (scote.uncertainty) and the streaming of per-tile slabs in the pipelines."""
import json
import warnings

import numpy as np
import pytest
import scipy.sparse as sp

import scote
from scote._twopoint import cpu
from scote._twopoint.lattice import build_lattice
from scote.uncertainty import TileMomentAccumulator, closed_form_log_g, reliability_flag
from _helpers import direct_moments, loop_bootstrap_log_g


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

NO_TORCH = pytest.mark.skipif(torch is None, reason="torch not installed")
TORCH_CPU = scote.GPUOptions(device="cpu")      # the GPU backend on the PyTorch CPU device
BACKENDS = [("cpu", dict(kernel="numba")), ("cpu", dict(kernel="numpy"))] + \
           ([(TORCH_CPU, {})] if torch is not None else [])
# the same as parameters: the torch case is collected everywhere and skipped without torch (constant test totals)
BACKEND_PARAMS = [("cpu", dict(kernel="numba")), ("cpu", dict(kernel="numpy")),
                  pytest.param(TORCH_CPU, {}, marks=NO_TORCH)]


def _toy(T_=40, K=3, G=12, seed=0, empty_tile=3, spread=0.5):
    """Poisson tile counts with pair- and tile-dependent means (tile factor 1 -+ spread); one tile without slabs."""
    rng = np.random.default_rng(seed)
    N = rng.poisson(rng.uniform(20, 200, (1, G)) * rng.uniform(1 - spread, 1 + spread, (T_, 1))).astype(np.float64)
    lam = rng.uniform(2, 300, (1, K, G, G)) * rng.uniform(1 - spread, 1 + spread, (T_, 1, 1, 1))
    C = rng.poisson(lam).astype(np.float64)
    if empty_tile is not None:
        C[empty_tile] = 0.0
    return C, N


def _accumulate(C, N, backend, kw, skip_zero=True, **extra):
    acc = TileMomentAccumulator(N, C.shape[1], backend=backend, **kw, **extra)
    res = []
    for k in range(C.shape[1]):
        for t in range(C.shape[0]):
            if skip_zero and not C[t, k].any():
                continue                                   # missing tile: no update at all
            acc.update(k, t, C[t, k] if backend == "cpu" else torch.from_numpy(C[t, k]))
        res.append(acc.finish(k))
    return acc, res


def _direct(C, N, sym=False):
    """Two-pass reference written from the definitions: Var(P) = sum_t (P_t - P/T)^2, Var(log g) = sum_t l_t^2;
    sym: on the per-tile symmetrized slabs (P_t + P_t^T) / 2."""
    if sym:
        C = 0.5 * (C + np.swapaxes(C, 2, 3))
    T_, K, G, _ = C.shape
    P = C.sum(0)
    n_a = N.sum(0)
    n = n_a.sum()
    m = N.sum(1)
    var_P = np.zeros_like(P)
    var_l = np.zeros_like(P)
    for k in range(K):
        Q = P[k].sum()
        for t in range(T_):
            var_P[k] += (C[t, k] - P[k] / T_) ** 2
            with np.errstate(divide="ignore", invalid="ignore"):
                infl = C[t, k] / P[k] - C[t, k].sum() / Q - (N[t] / n_a)[:, None] - (N[t] / n_a)[None, :] + 2 * m[t] / n
            var_l[k] += infl ** 2
    return P, var_P, np.sqrt(var_l)


# --------------------------------------------------------------------------------------------- (a) exact identities
@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("backend,kw", BACKEND_PARAMS)
def test_closed_form_equals_two_pass(backend, kw, sym):
    C, N = _toy()
    P, var_P, sd = _direct(C, N, sym)
    _, res = _accumulate(C, N, backend, kw, symmetrize=sym)
    P_raw = C.sum(0)
    for k, r in enumerate(res):
        assert np.array_equal(r["P"], P_raw[k])                     # point estimate: raw slabs either way
        Ck = 0.5 * (C[:, k] + np.swapaxes(C[:, k], 1, 2)) if sym else C[:, k]
        assert r["t_eff"].dtype == np.float32
        np.testing.assert_allclose(r["t_eff"], P[k] ** 2 / (Ck ** 2).sum(0), rtol=1e-6)
        assert np.array_equal(r["flag"], np.where(r["t_eff"] >= 20, 0, np.where(r["t_eff"] >= 5, 1, 2)))
        np.testing.assert_allclose(r["var_P"], var_P[k], rtol=1e-9, atol=0)
        np.testing.assert_allclose(r["sd"], sd[k], rtol=1e-9, atol=0)
        with np.errstate(divide="ignore"):
            np.testing.assert_allclose(r["n_eff_pairs"], P[k] ** 2 / var_P[k], rtol=1e-9)


def test_linearized_contributions_sum_to_zero():
    C, N = _toy(empty_tile=None)
    P = C.sum(0)
    n_a = N.sum(0)
    n = n_a.sum()
    infl = (C[:, 0] / P[0] - C[:, 0].sum((1, 2))[:, None, None] / P[0].sum() - (N / n_a)[:, :, None]
         - (N / n_a)[:, None, :] + 2 * (N.sum(1) / n)[:, None, None])
    assert np.abs(infl.sum(0)).max() < 1e-12


def test_numba_and_numpy_kernels_agree_and_zero_slabs_are_neutral():
    C, N = _toy()
    _, a = _accumulate(C, N, "cpu", dict(kernel="numba"))
    _, b = _accumulate(C, N, "cpu", dict(kernel="numpy"))
    _, c = _accumulate(C, N, "cpu", dict(kernel="numba"), skip_zero=False)   # zero slab fed explicitly
    for x, y, z in zip(a, b, c):
        for key in ("P", "var_P", "sd", "log_g", "t_eff"):
            assert np.array_equal(x[key], y[key], equal_nan=True)
            np.testing.assert_allclose(x[key], z[key], rtol=1e-12)


@pytest.mark.parametrize("backend,kw", BACKEND_PARAMS)
def test_float32_accumulators_close(backend, kw):
    C, N = _toy()
    _, ref = _accumulate(C, N, backend, kw)
    acc, res = _accumulate(C, N, backend, kw, dtype=np.float32)
    for r, q in zip(res, ref):
        assert np.array_equal(r["P"], q["P"])                       # A0 stays float64
        np.testing.assert_allclose(r["sd"], q["sd"], rtol=1e-3)


def test_duplicate_tile_discard_and_finished_slot():
    C, N = _toy(T_=40, K=1)
    acc = TileMomentAccumulator(N, 1)
    acc.update(0, 0, C[0, 0])
    with pytest.raises(ValueError):
        acc.update(0, 0, C[0, 0])
    acc.discard(0)                                                    # restart of the shell (OOM retry)
    for t in range(40):
        acc.update(0, t, C[t, 0])
    r = acc.finish(0)
    assert np.array_equal(r["P"], C[:, 0].sum(0)) and acc.live_slots() == []
    with pytest.raises(ValueError):
        acc.update(0, 1, C[1, 0])


# ------------------------------------------------------------------------------------ (b) Monte Carlo B = 20,000
def test_against_loop_bootstrap_20000():
    # 8 tiles of moderately heterogeneous intensity (factor 0.8-1.2): Monte Carlo error of the SD is 0.5 % at
    # B = 20,000; the remaining difference is the delta method's second-order term (closed form lower by ~0.5 % in
    # the median here, up to ~1.5 % at a tile spread of 0.5, worst single pairs -2 to -5 % at P ~ 100-200).
    # the Monte Carlo bootstrap resamples the raw (first-member) slabs: compare with symmetrize=False
    C, N = _toy(T_=8, K=2, G=6, seed=3, empty_tile=None, spread=0.2)
    boot = loop_bootstrap_log_g(C, N, n_boot=20_000, seed=1)
    with pytest.warns(UserWarning, match="fewer than 30 tiles"):
        _, res = _accumulate(C, N, "cpu", dict(kernel="numba"), symmetrize=False)
    var_mc = boot["P"] ** 2 / boot["n_eff_pairs"]                   # Monte Carlo Var(P) of the 20,000 resamples
    for k, r in enumerate(res):
        sel = r["P"] >= 100
        assert sel.sum() > 20
        assert np.array_equal(r["log_g"], boot["log_g"][k])
        assert np.abs(r["var_P"][sel] / var_mc[k][sel] - 1).max() < 0.03
        ratio = r["sd"][sel] / boot["sd"][k][sel]
        assert np.abs(ratio - 1).max() < 0.03 and abs(np.median(ratio) - 1) < 0.01


# --------------------------------------------------------------------------------------------- (c) sink hooks
def _case(n=30_000, G=5, side=300.0, h=2.0, tile=100.0, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, side, (n, 2))
    codes = rng.choice(G, n, p=np.arange(G, 0, -1) / np.arange(G, 0, -1).sum())
    X0 = sp.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, G))
    lat, X = build_lattice(np.floor(xy / h).astype(np.int64), X0, margin=8)
    center = (lat.cells + 0.5) * h
    tiles = np.floor(center[:, 0] / tile).astype(np.int64) * 100 + np.floor(center[:, 1] / tile).astype(np.int64)
    return lat, X, tiles


def _host(slab):
    return np.array(slab.cpu().numpy() if hasattr(slab, "cpu") else slab, copy=True)


class _Collect:
    def __init__(self):
        self.slabs = {}

    def __call__(self, i, g, slab):
        assert (i, g) not in self.slabs
        self.slabs[(i, g)] = slab.copy() if isinstance(slab, np.ndarray) else slab


def _check_sink(ref, collect, n_groups):
    _, K, G, _ = ref.shape
    N = np.ones((n_groups, G))
    acc = TileMomentAccumulator(N, K, min_tiles_warn=0)
    for (i, g), slab in sorted(collect.slabs.items()):
        assert np.array_equal(slab, ref[g, i])
        acc.update(i, g, slab)
    for i in range(K):
        assert np.array_equal(acc.finish(i)["P"], ref[:, i].sum(0))


@pytest.mark.parametrize("mode", ["sparse", "dense", "numba", "auto"])
def test_cpu_row_groups_sink_bit_identical(mode):
    lat, X, tiles = _case()
    opt = scote.CPUOptions(mode=mode)
    ref = cpu.pair_counts_level(X, lat, 8, include_intra=True, row_groups=tiles, options=opt)
    col = _Collect()
    assert cpu.pair_counts_level(X, lat, 8, include_intra=True, row_groups=tiles, sink=col, options=opt) is None
    assert len(col.slabs) == ref.shape[0] * 8
    _check_sink(ref, col, ref.shape[0])


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize("dense_bytes_limit", [0, 4 * 2**30])
def test_gpu_groups_sink_bit_identical(dense_bytes_limit):
    from scote._twopoint import gpu
    lat, X, tiles = _case()
    kw = dict(include_intra=True, row_groups=tiles,
              options=scote.GPUOptions(device="cpu", dense_bytes_limit=dense_bytes_limit, row_block_bytes=48 * 1024))
    ref = gpu.pair_counts_level(X, lat, 8, range(8), **kw)
    assert np.array_equal(ref, cpu.pair_counts_level(X, lat, 8, include_intra=True, row_groups=tiles,
                                                     options=scote.CPUOptions(mode="sparse")))
    col = _Collect()
    assert gpu.pair_counts_level(X, lat, 8, range(8), sink=col, **kw) is None
    assert all(isinstance(slab, np.ndarray) for slab in col.slabs.values())
    _check_sink(ref, col, ref.shape[0])


def _bins_toy(n=30_000, G=4, side=420.0, seed=5):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, side, (n, 2))
    codes = np.where(xy[:, 0] > side / 2, rng.choice(G, n, p=[0.5, 0.3, 0.1, 0.1]), rng.choice(G, n))
    X0 = sp.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, G))
    return xy, X0


BACKENDS_RUN = [dict(backend=scote.CPUOptions(mode="sparse")), dict(backend="cpu"),
                pytest.param(dict(backend=scote.GPUOptions(device="cpu")), marks=NO_TORCH)]


@pytest.mark.parametrize("kw", BACKENDS_RUN)
def test_streamed_bins_equal_closed_form_of_stored_tiles(kw):
    xy, X0 = _bins_toy()
    res = scote.twopoint(xy, X0, delta=2.0, bin_size=2.0, tile_um=60.0, n_levels=3, keep_tiles=True, **kw)
    ref = closed_form_log_g(res.counts_tiles, res.totals_tiles, **kw)
    lo, hi = res.interval()
    for key, got in (("log_g", res.log_g), ("sd", res.sd), ("n_eff_pairs", res.n_eff_pairs), ("P", res.P),
                     ("t_eff", res.t_eff), ("flag", res.flag), ("lo", lo), ("hi", hi)):
        assert np.array_equal(got, getattr(ref, key), equal_nan=True), key
    assert np.array_equal(res.totals, ref.totals) and np.array_equal(res.P_tot, ref.P_tot)
    assert np.array_equal(res.P_tot, res.P.sum(axis=(1, 2)))
    assert res.counts_tiles.shape[0] >= 49 and res.n_tiles == res.counts_tiles.shape[0]
    # without keeping the tiles the results are the same
    res2 = scote.twopoint(xy, X0, delta=2.0, bin_size=2.0, tile_um=60.0, n_levels=3, **kw)
    assert res2.counts_tiles is None
    for key in ("log_g", "sd", "n_eff_pairs", "P", "t_eff", "flag"):
        assert np.array_equal(getattr(res2, key), getattr(res, key), equal_nan=True), key


def test_streamed_directory_matches_memory(tmp_path):
    xy, X0 = _bins_toy()
    sparse = scote.CPUOptions(mode="sparse")
    mem = scote.twopoint(xy, X0, delta=2.0, bin_size=2.0, tile_um=60.0, n_levels=3, backend=sparse,
                         keep_tiles=True)
    out = tmp_path / "cf"
    scote.twopoint(xy, X0, names=["a", "b", "c", "d"], delta=2.0, bin_size=2.0, tile_um=60.0, n_levels=3,
                   backend=sparse, out=out, dtype="float32")
    K, G = mem.n_distance_bins, mem.n_features
    d = {k: np.load(out / f"{k}.npy", mmap_mode="r") for k in ("log_g", "sd", "n_eff_pairs", "P")}
    assert d["log_g"].shape == (K, G, G) and d["log_g"].dtype == np.float32 and d["P"].dtype == np.float64
    assert np.array_equal(d["P"], mem.P)
    assert np.array_equal(np.asarray(d["log_g"]).view(np.uint32), mem.log_g.astype(np.float32).view(np.uint32))
    ref = direct_moments(mem.counts_tiles, mem.totals_tiles)
    np.testing.assert_allclose(d["sd"], ref["sd"].astype(np.float32), rtol=1e-5)
    np.testing.assert_allclose(np.load(out / "totals_tiles.npy"), mem.totals_tiles)
    bins = (out / "bins.csv").read_text().splitlines()
    assert bins[0] == "level,shell,r_from,r_to,r_central" and len(bins) == K + 1
    assert (out / "tiles.csv").read_text().splitlines()[0] == "column,row,x_from,y_from"
    assert (out / "features.txt").read_text().split() == ["a", "b", "c", "d"]
    meta = json.load(open(out / "meta.json"))
    assert meta["n_tiles"] == mem.n_tiles and meta["tile_um"] == 60.0 and meta["uncertainty"] == "closed"
    assert {"lattice", "accumulate", "write"} <= set(meta["timings"])
    loaded = scote.TwoPointResult.load(out)
    assert loaded.features == ["a", "b", "c", "d"] and np.array_equal(loaded.P, mem.P)
    assert loaded.params["tile_um"] == 60.0


def test_streamed_molecules_directory_matches_memory(tmp_path):
    rng = np.random.default_rng(2)
    n, G = 6_000, 3
    xy = rng.uniform(0, 300, (n, 2))
    codes = rng.choice(G, n, p=[0.5, 0.3, 0.2]).astype(np.int64)
    sparse = scote.CPUOptions(mode="sparse")
    mem = scote.twopoint(xy, codes, delta=0.5, n_levels=3, tile_um=50.0, backend=sparse, keep_tiles=True)
    res = scote.twopoint(xy, codes, delta=0.5, n_levels=3, tile_um=50.0, backend=sparse, out=tmp_path / "m",
                         dtype="float32")
    assert np.array_equal(np.asarray(res.P), mem.P)
    assert np.array_equal(np.asarray(res.log_g), mem.log_g.astype(np.float32), equal_nan=True)
    np.testing.assert_allclose(res.sd, direct_moments(mem.counts_tiles, mem.totals_tiles)["sd"]
                               .astype(np.float32), rtol=1e-5)
    assert list(res.bins["level"])[:8] == [0] * 8 and len(res.bins) == mem.counts_tiles.shape[1]
    ref = closed_form_log_g(mem.counts_tiles, mem.totals_tiles)
    assert np.array_equal(mem.sd, ref.sd, equal_nan=True)


# ------------------------------------------------------------------------------------------------ (e) warning
def test_warns_below_30_tiles():
    C, N = _toy(T_=29)
    with pytest.warns(UserWarning, match="smaller than the correlation length"):
        TileMomentAccumulator(N, 3)
    C, N = _toy(T_=30)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        TileMomentAccumulator(N, 3)


def test_accumulator_backend_argument():
    """One backend argument, as in scote.twopoint: names or options objects; the same errors."""
    C, N = _toy(T_=30, K=1)
    for b in ("cpu", scote.CPUOptions(mode="sparse")):
        acc = TileMomentAccumulator(N, 1, backend=b)
        assert acc.backend == "cpu" and acc.device is None
    with pytest.raises(ValueError, match="backend must be 'cpu', 'gpu', a scote.CPUOptions, .* got 'numpy'"):
        TileMomentAccumulator(N, 1, backend="numpy")
    with pytest.raises(TypeError, match="got dict"):
        closed_form_log_g(C, N, backend={"device": "cpu"})
    with pytest.raises(TypeError, match="device"):                  # no separate device argument
        closed_form_log_g(C, N, device="cpu")
    r = closed_form_log_g(C, N)
    assert r.params["backend"] == "cpu" and r.params["options"]["mode"] == "auto"


# ----------------------------------------------------------------------------- closed form of a stored tile tensor
def test_closed_form_log_g_result():
    from scipy.stats import norm
    C, N = _toy()
    C[:, 1, 0, 1] = 0.0                                              # a pair without counts in one shell
    P, var_P, sd = _direct(C, N)
    boot = loop_bootstrap_log_g(C, N, n_boot=10)
    cf = closed_form_log_g(C, N, symmetrize=False, features=list("abcdefghijkl"))
    assert isinstance(cf, scote.TwoPointResult) and cf.features[1] == "b" and cf.counts_tiles is C
    assert list(cf.bins.columns) == ["level", "shell", "r_from", "r_to", "r_central"]
    assert list(cf.bins.shell) == [0, 1, 2]
    assert np.array_equal(cf.log_g, boot["log_g"], equal_nan=True) and np.array_equal(cf.P, boot["P"])
    assert np.array_equal(cf.totals, boot["totals"]) and cf.n_tiles == 40 and np.array_equal(cf.totals_tiles, N)
    assert np.array_equal(cf.P_tot, C.sum(0).sum(axis=(1, 2)))
    ok = P > 0
    assert cf.sd.dtype == np.float64 and np.isnan(cf.sd[~ok]).all()
    np.testing.assert_allclose(cf.sd[ok], sd[ok], rtol=1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.testing.assert_allclose(cf.n_eff_pairs[ok], (P ** 2 / var_P)[ok], rtol=1e-9)
    z = norm.ppf(0.975)
    np.testing.assert_allclose(cf.hi[ok] - cf.log_g[ok], z * sd[ok], rtol=1e-9)
    np.testing.assert_allclose(cf.log_g[ok] - cf.lo[ok], z * sd[ok], rtol=1e-9)
    assert np.isnan(cf.lo[~ok]).all() and np.isnan(cf.hi[~ok]).all()
    lo90, hi90 = cf.interval((5, 95))
    np.testing.assert_allclose(hi90[ok] - cf.log_g[ok], norm.ppf(0.95) * sd[ok], rtol=1e-9)
    with pytest.raises(ValueError):
        closed_form_log_g(C, N[:, :3])
    with pytest.raises(ValueError):
        closed_form_log_g(C, N, bins=cf.bins.iloc[:2])


# ------------------------------------------------------------------- symmetrization, T_eff flag, four accumulators
def test_symmetrize_gives_symmetric_sd_and_matches_symmetrized_tensor():
    C, N = _toy()
    C = C + np.swapaxes(np.roll(C, 1, axis=0), 2, 3)              # total P symmetric, single-tile slabs are not
    sym = closed_form_log_g(C, N)
    raw = closed_form_log_g(C, N, symmetrize=False)
    def tr(A):
        return np.swapaxes(A, 1, 2)
    assert np.array_equal(sym.log_g, raw.log_g, equal_nan=True)
    for key in ("sd", "n_eff_pairs", "t_eff"):
        np.testing.assert_allclose(getattr(sym, key), tr(getattr(sym, key)), rtol=1e-12)
    assert np.nanmax(np.abs(raw.sd / tr(raw.sd) - 1)) > 0.05            # the raw convention is not symmetric
    Cs = 0.5 * (C + np.swapaxes(C, 2, 3))
    via = closed_form_log_g(Cs, N, symmetrize=False)
    for key in ("sd", "n_eff_pairs", "t_eff"):
        np.testing.assert_allclose(getattr(sym, key), getattr(via, key), rtol=1e-12)
    # and against the Monte Carlo bootstrap of the symmetrized tensor (P >= 100, B = 20,000)
    boot = loop_bootstrap_log_g(Cs[:, :1], N, n_boot=20_000, seed=3)
    ok = sym.P[0] >= 100
    ratio = sym.sd[0][ok] / boot["sd"][0][ok]
    assert abs(np.median(ratio) - 1) < 0.01 and np.abs(ratio - 1).max() < 0.05


def test_symmetrize_false_reproduces_unsymmetrized_values():
    C, N = _toy(seed=4)
    P, var_P, sd = _direct(C, N, sym=False)
    for backend, kw in BACKENDS:
        _, res = _accumulate(C, N, backend, kw, symmetrize=False)
        for k, r in enumerate(res):
            np.testing.assert_allclose(r["sd"], sd[k], rtol=1e-9)
            np.testing.assert_allclose(r["var_P"], var_P[k], rtol=1e-9)


def test_four_accumulators_per_live_shell():
    C, N = _toy(K=1)
    for backend, kw in BACKENDS:
        acc = TileMomentAccumulator(N, 1, backend=backend, **kw)
        acc.update(0, 0, C[0, 0])
        st = acc._live[0]
        gg = [v for v in st.values() if getattr(v, "shape", None) == (12, 12)]
        assert len(gg) == 4 + (2 if kw.get("kernel") == "numpy" else 0)   # A0, A1, A2, Av (+ numpy scratch)
        assert {"A0", "A1", "A2", "Av"} <= set(st)


def test_teff_flag_rule_and_nan_intervals():
    t_eff = np.array([np.nan, 0, 4.99, 5, 19.99, 20, 300], dtype=np.float32)
    assert list(reliability_flag(t_eff)) == [2, 2, 2, 1, 1, 0, 0]
    C, N = _toy(T_=40, K=1, G=6, seed=9, empty_tile=None)
    C[:3, 0, 0, 1] *= 50.0                                           # pair (0, 1) carried by 3 tiles: T_eff < 5
    C[3:, 0, 0, 1] = 0.0
    C[:12, 0, 2, 3] += 400.0
    C[12:, 0, 2, 3] = 0.0                    # pair (2, 3) carried by 12 tiles: 5 <= T_eff < 20
    r = closed_form_log_g(C, N, symmetrize=False)
    assert r.t_eff[0, 0, 1] < 5 and r.flag[0, 0, 1] == 2
    assert np.isnan(r.lo[0, 0, 1]) and np.isnan(r.hi[0, 0, 1]) and np.isfinite(r.sd[0, 0, 1])
    assert 5 <= r.t_eff[0, 2, 3] < 20 and r.flag[0, 2, 3] == 1 and np.isfinite(r.lo[0, 2, 3])
    assert (r.flag[0][r.t_eff[0] >= 20] == 0).all() and (r.flag[0] == 0).sum() > 20


@pytest.mark.parametrize("kw", [dict(backend=scote.CPUOptions(mode="sparse")), dict(),
                                pytest.param(dict(backend=scote.GPUOptions(device="cpu")), marks=NO_TORCH)])
def test_pair_counts_sink_matches_stored_counts(kw):
    xy, X0 = _bins_toy(n=8_000)
    res = scote.pair_counts(xy, X0, delta=2.0, bin_size=2.0, n_levels=3, **kw)
    assert isinstance(res, scote.PairCounts)
    ref = res.counts
    got = {}

    def sink(k, slab):
        assert isinstance(slab, np.ndarray)                          # host arrays on both backends
        got[k] = slab.copy()

    streamed = scote.pair_counts(xy, X0, delta=2.0, bin_size=2.0, n_levels=3, sink=sink, **kw)
    assert streamed.bins.equals(res.bins) and sorted(got) == list(range(ref.shape[0]))
    for k in range(ref.shape[0]):
        assert np.array_equal(got[k], ref[k])
    assert np.array_equal(streamed.counts, ref.sum(axis=(1, 2)))
    stored = scote.twopoint(xy, X0, delta=2.0, bin_size=2.0, tile_um=60.0, n_levels=3, **kw)
    assert np.array_equal(stored.P, ref)          # the per-tile path sums to the global counts
