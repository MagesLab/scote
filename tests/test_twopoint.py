"""End-to-end runs of scote.twopoint (molecules with enumeration, lattices only, count matrices), its inputs, the
result structure, and the random-label reference."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import scote
from scote._twopoint import pipeline
from scote._twopoint.lattice import onehot_csr


def test_random_labels_give_g_of_one():
    rng = np.random.default_rng(3)
    n, G = 60_000, 4
    xy = rng.uniform(0, 400, (n, 2))
    codes = rng.choice(G, n, p=[0.5, 0.3, 0.15, 0.05])
    with pytest.warns(UserWarning, match="fewer than 30 tiles"):
        res = scote.twopoint(xy, codes, delta=0.5, n_levels=5, tile_um=100.0, keep_tiles=True)
    assert res.counts_tiles.shape[0] == res.n_tiles == 16                      # 4x4 tiles
    assert res.n_distance_bins == 8 + 4 * 5 and res.n_features == G and res.features == ["0", "1", "2", "3"]
    assert np.array_equal(res.totals_tiles.sum(0), np.bincount(codes, minlength=G))
    lo, hi = res.interval()
    finite = np.isfinite(res.log_g) & np.isfinite(lo) & np.isfinite(hi)
    assert ((lo <= 0) & (hi >= 0))[finite].mean() > 0.85
    assert np.abs(np.nanmedian(res.log_g[finite])) < 0.02
    ref = scote.random_label_reference(res.P, res.totals)
    assert np.allclose(ref.sum(axis=(1, 2)), res.P.sum(axis=(1, 2)))
    assert list(res.bins.columns) == ["level", "shell", "r_from", "r_to", "r_central"]
    assert (res.bins.level[:8] == 0).all() and (np.diff(res.bins.r_from) > 0).all()
    assert list(res.bins.shell) == list(range(8)) + [4, 5, 6, 7] * 5


def test_bins_pipeline_random_labels_and_tile_consistency():
    """Lattice-only hierarchy on a bin table: random labels give g = 1; per-tile counts sum to the global counts."""
    rng = np.random.default_rng(5)
    n, G, pitch = 40_000, 3, 2.0
    xy = rng.uniform(0, 400, (n, 2))
    codes = rng.choice(G, n, p=[0.6, 0.3, 0.1])
    X0 = onehot_csr(codes, G)
    with pytest.warns(UserWarning, match="fewer than 30 tiles"):
        res = scote.twopoint(xy, X0, names=["a", "b", "c"], delta=pitch, bin_size=pitch, tile_um=100.0, n_levels=4,
                             keep_tiles=True)
    assert res.counts_tiles.shape[:2] == (16, 8 + 4 * 4)
    assert np.array_equal(res.totals_tiles.sum(0), np.bincount(codes, minlength=G))
    ref = scote.pair_counts(xy, X0, delta=pitch, bin_size=pitch, n_levels=4, backend=scote.CPUOptions(mode="sparse"))
    assert np.array_equal(res.counts_tiles.sum(0), ref.counts)              # tiles partition all pairs
    assert res.bins.equals(ref.bins)                                        # the same bins table
    fin = np.isfinite(res.log_g)
    assert np.abs(np.nanmedian(res.log_g[fin])) < 0.02
    assert (((res.lo <= 0) & (res.hi >= 0))[fin]).mean() > 0.85
    tab = res.pair("a", "c")
    assert len(tab) == res.n_distance_bins and np.array_equal(tab["P"], res.P[:, 0, 2])
    assert np.array_equal(tab["log_g"], res.log_g[:, 0, 2])
    with pytest.raises(KeyError):
        res.pair("a", "z")


def test_number_of_features_from_the_data():
    rng = np.random.default_rng(6)
    xy = rng.uniform(0, 300, (5_000, 2))
    codes = rng.integers(0, 3, 5_000)
    a = scote.twopoint(xy, codes, n_levels=2, tile_um=50.0)
    assert a.n_features == 3 and a.features == ["0", "1", "2"]
    # names fix G when the highest codes have no molecules
    b = scote.twopoint(xy, codes, names=["x", "y", "z", "absent"], n_levels=2, tile_um=50.0)
    assert b.n_features == 4 and b.features[-1] == "absent" and not b.P[:, 3].any()
    assert np.array_equal(b.P[:, :3, :3], a.P)
    with pytest.raises(ValueError, match="codes must lie in 0 .. 1"):
        scote.twopoint(xy, codes, names=["x", "y"], n_levels=2, tile_um=50.0)
    with pytest.raises(ValueError, match="feature names"):
        scote.twopoint(xy, onehot_csr(codes, 3), names=["x"], n_levels=2, tile_um=50.0)
    with pytest.raises(TypeError, match="dtype float64"):
        scote.twopoint(xy, codes.astype(float), n_levels=2, tile_um=50.0)


def _labeled(n=6_000, seed=7):
    """Random molecules with labels "a" .. "d" (integer codes 0 .. 3 in that order)."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 300, (n, 2))
    codes = rng.choice(4, n, p=[0.4, 0.3, 0.2, 0.1])
    return xy, codes, np.array(["a", "b", "c", "d"])[codes]


def test_label_types_give_the_same_result():
    """Strings, a Series, a Categorical, and integer codes with names are the same input; the order of the
    features is the category order of a Categorical, otherwise the sorted unique labels."""
    xy, codes, labels = _labeled()
    kw = dict(n_levels=2, tile_um=100.0)
    with pytest.warns(UserWarning):
        ref = scote.twopoint(xy, codes, names=["a", "b", "c", "d"], **kw)
    for features in (labels, labels.astype(object), pd.Series(labels), pd.Series(labels, dtype="category"),
                     pd.Categorical(labels), list(labels)):
        with pytest.warns(UserWarning):
            res = scote.twopoint(xy, features, **kw)
        assert res.features == ["a", "b", "c", "d"] and np.array_equal(res.P, ref.P)
        assert np.array_equal(res.sd, ref.sd, equal_nan=True) and res.params["input"] == "labels"
    order = ["d", "b", "a", "c"]
    with pytest.warns(UserWarning):
        res = scote.twopoint(xy, pd.Categorical(labels, categories=order), **kw)
    idx = [3, 1, 0, 2]
    assert res.features == order and np.array_equal(res.P, ref.P[:, idx][:, :, idx])
    with pytest.raises(TypeError, match="names apply"):
        scote.twopoint(xy, labels, names=["a", "b", "c", "d"], **kw)
    with pytest.raises(TypeError, match="names apply"):
        scote.twopoint(xy, pd.Categorical(labels), names=["a", "b", "c", "d"], **kw)
    missing = labels.astype(object)
    missing[5] = None
    with pytest.raises(ValueError, match="missing labels"):
        scote.twopoint(xy, missing, **kw)
    with pytest.raises(ValueError, match="labels for"):
        scote.twopoint(xy, labels[:-1], **kw)


def test_select_chooses_and_orders_a_subset():
    xy, codes, labels = _labeled()
    kw = dict(n_levels=2, tile_um=100.0)
    keep = np.isin(labels, ["c", "a"])
    with pytest.warns(UserWarning):
        sub = scote.twopoint(xy[keep], pd.Categorical(labels[keep], categories=["c", "a"]), **kw)
    with pytest.warns(UserWarning):
        res = scote.twopoint(xy, labels, select=["c", "a"], **kw)
    assert res.features == ["c", "a"] and res.params["n_points"] == keep.sum()
    assert np.array_equal(res.P, sub.P) and np.array_equal(res.sd, sub.sd, equal_nan=True)
    with pytest.warns(UserWarning):
        top = scote.twopoint(xy, labels, select="top:2", **kw)
    assert top.features == ["a", "b"]
    X = onehot_csr(codes, 4)
    with pytest.warns(UserWarning):
        cols = scote.twopoint(xy, X, names=list("abcd"), select="c,a", delta=2.0, bin_size=2.0, **kw)
    with pytest.warns(UserWarning):
        lat = scote.twopoint(xy, labels, select=["c", "a"], lattice_only=True, delta=2.0, **kw)
    assert cols.features == ["c", "a"] and np.array_equal(cols.P, lat.P)
    with pytest.raises(KeyError, match="TYPO"):
        scote.twopoint(xy, labels, select=["a", "TYPO"], **kw)
    with pytest.raises(ValueError, match="more than once"):
        scote.twopoint(xy, labels, select=["a", "a"], **kw)


def test_count_matrix_types_and_lattice_only_labels_agree():
    """A one-hot count matrix (sparse, dense, DataFrame) gives the lattice-only result of the labels."""
    xy, codes, labels = _labeled()
    kw = dict(n_levels=3, tile_um=100.0, delta=2.0)
    with pytest.warns(UserWarning):
        ref = scote.twopoint(xy, labels, lattice_only=True, **kw)
    assert ref.params["lattice_only"] and ref.params["input"] == "labels" and "bin_size" not in ref.params
    X = onehot_csr(codes, 4)
    for features, names in ((X, list("abcd")), (X.toarray(), list("abcd")), (sp.csc_array(X), list("abcd")),
                            (pd.DataFrame(X.toarray(), columns=list("abcd")), None),
                            (pd.DataFrame.sparse.from_spmatrix(X, columns=list("abcd")), None)):
        with pytest.warns(UserWarning):
            res = scote.twopoint(xy, features, names=names, **kw)
        assert res.features == list("abcd") and res.params["input"] == "counts"
        for key in ("P", "log_g", "sd", "t_eff", "flag", "totals_tiles"):
            assert np.array_equal(getattr(res, key), getattr(ref, key), equal_nan=True), key
        assert res.bins.equals(ref.bins)
    with pytest.raises(ValueError, match="always counted on lattices"):
        scote.twopoint(xy, X, names=list("abcd"), lattice_only=False, **kw)
    with pytest.raises(ValueError, match="rows for"):
        scote.twopoint(xy[:-1], X, **kw)
    with pytest.raises(TypeError, match="columns"):
        scote.twopoint(xy, pd.DataFrame(X.toarray()), names=list("abcd"), **kw)
    with pytest.raises(ValueError, match="bin_size applies"):
        scote.twopoint(xy, labels, bin_size=2.0, **kw)


def test_bin_size_default_and_delta_limits():
    """A square bin array: delta defaults to the inferred bin size, a larger delta coarse-grains, a smaller one
    raises; the inference ignores bins without counts left out of the table and a slight rotation."""
    rng = np.random.default_rng(12)
    ij = np.stack(np.meshgrid(np.arange(150), np.arange(150), indexing="ij"), -1).reshape(-1, 2)
    ij = ij[rng.random(len(ij)) < 0.7]                                   # 30 % of the bins left out
    angle = 2e-4
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xy = (ij + 0.5) @ rot.T * 2.0 * 0.9999 + 100.0                       # 2 um bins, registered to an image
    X = sp.csr_matrix(rng.poisson(0.5, (len(xy), 3)).astype(np.float64))
    assert pipeline._infer_bin_size(xy) == 2.0
    with pytest.warns(UserWarning):
        res = scote.twopoint(xy, X, n_levels=2, tile_um=100.0)
    assert res.params["delta"] == res.params["bin_size"] == 2.0 and res.features == ["0", "1", "2"]
    assert res.params["bin_size_source"].startswith("inferred")
    with pytest.warns(UserWarning):
        coarse = scote.twopoint(xy, X, delta=8.0, n_levels=0, tile_um=100.0)
    assert coarse.bins.r_to.iloc[-1] == 64.0
    with pytest.raises(ValueError, match="smaller than the bin size"):
        scote.twopoint(xy, X, delta=1.0, n_levels=2, tile_um=100.0)
    with pytest.warns(UserWarning):                                      # spots smaller than their spacing
        spots = scote.twopoint(xy, X, delta=1.5, bin_size=1.5, n_levels=1, tile_um=100.0)
    assert spots.params["bin_size_source"] == "given"
    hexagonal = np.array([[100.0 * i + 50.0 * (j % 2), 86.6 * j] for i in range(30) for j in range(30)])
    assert pipeline._infer_bin_size(hexagonal) == 100.0
    big = rng.uniform(0, 1000, (300_000, 2))                              # windows on large inputs
    assert 0 < pipeline._infer_bin_size(big) < 1.0
    with pytest.raises(ValueError, match="bin size"):
        pipeline._infer_bin_size(np.zeros((5, 2)))


def test_central_distances_of_the_bins():
    """Enumerated bins [a, b): (3/4)(b^4 - a^4)/(b^3 - a^3); lattice shells: the mean offset length of the cell pairs
    (shell 0: half the pitch)."""
    xy, codes, labels = _labeled(n=3_000)
    with pytest.warns(UserWarning):
        mol = scote.twopoint(xy, labels, delta=0.5, n_levels=1, tile_um=100.0)
    a, b = mol.bins.r_from.to_numpy()[:8], mol.bins.r_to.to_numpy()[:8]
    np.testing.assert_allclose(mol.r[:8], 3 / 4 * (b**4 - a**4) / (b**3 - a**3), rtol=1e-15)
    assert mol.r[0] == pytest.approx(3 / 8) and mol.r[1] == pytest.approx(45 / 56)
    with pytest.warns(UserWarning):
        lat = scote.twopoint(xy, labels, delta=2.0, lattice_only=True, n_levels=1, tile_um=100.0)
    assert lat.r[0] == 1.0
    assert lat.r[1] == pytest.approx(2.0 * (1 + np.sqrt(2)) / 2, rel=1e-15)          # offsets (1, 0) and (1, 1)
    np.testing.assert_allclose(mol.r[8:12] * 4.0, lat.r[8:12], rtol=1e-15)   # level 1: pitch 1 and 4 um, same rule


# ---------------------------------------------------------------------------------------------------------
# Integer overflow of the random-label reference, zero-level runs, and the size guard of the enumerated level.
# ---------------------------------------------------------------------------------------------------------

def _random_label_tiles_int64(n_tiles=4, n_bins=2):
    """An exactly random-label configuration at a scale that overflows int64: 2 features, 1.44e8 molecules
    (7.2e7 each), 4e12 ordered pairs per bin, so P_tot n_a n_b = 4e12 * 5.18e15 = 2e28 >> 2^63 = 9.2e18.
    Per tile: 1.8e7 molecules of each feature and 2.5e11 pairs in each of the 4 (a, b) cells, so that
    P_ab = P_tot n_a n_b / n_tot^2 = 4e12 * 0.25 = 1e12 holds exactly and the true log g is 0 everywhere."""
    counts = np.full((n_tiles, n_bins, 2, 2), 250_000_000_000, dtype=np.int64)
    totals = np.full((n_tiles, 2), 18_000_000, dtype=np.int64)
    return counts, totals


def test_random_label_reference_does_not_overflow_int64():
    counts, totals = _random_label_tiles_int64()
    P = counts.sum(0)
    n = totals.sum(0)                       # int64, as a caller summing int tiles would pass
    Ptot = float(P[0].sum())
    na, nb = (float(v) for v in n)
    assert Ptot * na * nb > 2.0**63                            # the product really would overflow int64
    assert float(n.sum()) == 1.44e8
    ref = scote.random_label_reference(P, n)
    ref64 = scote.random_label_reference(P.astype(np.float64), n.astype(np.float64))
    assert ref.dtype == np.float64
    assert np.all(ref > 0)
    assert np.array_equal(ref, ref64)                          # int input == float64 input
    assert np.allclose(ref, P, rtol=1e-12)                     # and it is the random-label configuration
    with np.errstate(divide="ignore", invalid="ignore"):
        log_g = np.log(P / ref)
    assert np.all(np.isfinite(log_g)) and np.allclose(log_g, 0.0, atol=1e-12)
    with np.errstate(over="ignore"):
        wrapped = (P[0].sum() * np.outer(n, n))                # int64 * int64 wraps silently
    assert wrapped.dtype == np.int64
    assert not np.allclose(wrapped.astype(np.float64), Ptot * np.outer(n, n).astype(np.float64), rtol=1e-6)


def test_closed_form_int64_tiles_give_zero_log_g():
    """closed_form_log_g takes int64 tile tensors at overflow scale; the random-label configuration gives log g = 0
    with SD 0."""
    import warnings
    counts, totals = _random_label_tiles_int64()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)                     # no overflow warning (Var(P) = 0)
        res = scote.uncertainty.closed_form_log_g(counts, totals, min_tiles_warn=0)
    assert isinstance(res, scote.TwoPointResult)
    assert np.allclose(res.log_g, 0.0, atol=1e-12)
    assert np.allclose(res.sd, 0.0, atol=1e-6)


def test_enumerated_counts_are_exact_integers_in_float64():
    rng = np.random.default_rng(11)
    n, G = 20_000, 3
    xy = rng.uniform(0, 200, (n, 2))
    codes = rng.choice(G, n, p=[0.5, 0.3, 0.2])
    with pytest.warns(UserWarning):
        res = scote.twopoint(xy, codes, n_levels=0, tile_um=100.0, keep_tiles=True)
    c0 = res.counts_tiles
    assert c0.dtype == np.float64 and c0.shape == (4, 8, G, G) and np.array_equal(c0, np.round(c0))
    from scote._twopoint.enumeration import pair_counts_enumerated
    tiles = pipeline._TileGrid(xy, 100.0, xy.min(0)).index(xy)
    for t in range(4):
        m = tiles == t
        assert np.array_equal(c0[t], pair_counts_enumerated(xy[m], codes[m], G, 0.5))


def test_zero_lattice_levels():
    """n_levels = 0: the enumerated level alone."""
    rng = np.random.default_rng(2)
    n, G = 5_000, 3
    xy = rng.uniform(0, 200, (n, 2))
    codes = rng.choice(G, n)
    with pytest.warns(UserWarning):
        res = scote.twopoint(xy, codes, n_levels=0, tile_um=100.0, keep_tiles=True)
    assert res.counts_tiles.shape == (4, 8, G, G) and res.n_distance_bins == 8 and (res.bins.level == 0).all()
    assert np.array_equal(res.totals_tiles.sum(0), np.bincount(codes, minlength=G))
    fin = np.isfinite(res.log_g)
    assert fin.any() and np.abs(np.nanmedian(res.log_g[fin])) < 0.1


def test_memory_warning_before_allocating(monkeypatch):
    """A run whose arrays exceed MEMORY_WARN_BYTES warns before allocating and names the remedies; the enumerated
    level streams into the accumulators, so its tile tensor is not formed."""
    rng = np.random.default_rng(4)
    xy = rng.uniform(0, 200, (1_000, 2))
    codes = rng.integers(0, 3, 1_000)
    monkeypatch.setattr(pipeline, "MEMORY_WARN_BYTES", 1000)
    with pytest.warns(UserWarning) as rec:
        scote.twopoint(xy, codes, n_levels=1, tile_um=100.0)
    text = " ".join(str(w.message) for w in rec)
    assert "GiB of host memory" in text and "select fewer features" in text and "lattice_only=True" in text
    assert "out=DIR" in text
    monkeypatch.setattr(pipeline, "MEMORY_WARN_BYTES", 4 * 2**30)
    with pytest.warns(UserWarning, match=r"55\.7 GiB .* G = 5000"):                    # a 5K panel, 12 levels
        assert pipeline._check_memory(T=100, K=56, G=5000, live=8, out=None, keep_tiles=False, dtype="float64",
                                      acc_dtype="float64", enumeration=True) > 4 * 2**30


def test_result_roundtrip_npz_and_directory(tmp_path):
    rng = np.random.default_rng(8)
    xy = rng.uniform(0, 300, (8_000, 2))
    codes = rng.integers(0, 3, 8_000)
    with pytest.warns(UserWarning):
        res = scote.twopoint(xy, codes, names=["a", "b", "c"], lattice_only=True, delta=2.0, n_levels=3,
                             tile_um=60.0, keep_tiles=True)
    for target in (tmp_path / "r.npz", tmp_path / "rdir"):
        res.save(target)
        back = scote.TwoPointResult.load(target)
        for key in ("log_g", "sd", "n_eff_pairs", "P", "t_eff", "flag", "P_tot", "totals", "totals_tiles",
                    "counts_tiles"):
            assert np.array_equal(np.asarray(getattr(back, key)), getattr(res, key), equal_nan=True), key
        assert back.features == res.features and back.bins.equals(res.bins) and back.tiles.equals(res.tiles)
        assert back.params["tile_um"] == 60.0 and back.params["lattice_only"] is True
    z = np.load(tmp_path / "r.npz")                                         # no pickle needed
    lo, hi = res.interval()
    assert np.array_equal(z["lo"], lo, equal_nan=True) and np.array_equal(z["hi"], hi, equal_nan=True)
    assert z["features"].dtype.kind == "U" and list(z["features"]) == ["a", "b", "c"]
    assert res.lo is res.lo and np.array_equal(res.lo, lo, equal_nan=True)  # computed once


def test_load_needs_no_pickle(tmp_path):
    """An archive with the feature names as an object array (which needs pickle) is not read."""
    rng = np.random.default_rng(9)
    xy = rng.uniform(0, 300, (4_000, 2))
    codes = rng.integers(0, 2, 4_000)
    with pytest.warns(UserWarning):
        res = scote.twopoint(xy, codes, names=["a", "b"], lattice_only=True, delta=2.0, n_levels=2, tile_um=60.0)
    res.save(tmp_path / "new.npz")
    with np.load(tmp_path / "new.npz") as z:
        arrays = {k: z[k] for k in z.files}
    arrays["features"] = np.array(["a", "b"], dtype=object)
    np.savez(tmp_path / "object.npz", **arrays)
    with pytest.raises(ValueError, match="allow_pickle"):
        scote.TwoPointResult.load(tmp_path / "object.npz")
    with pytest.raises(FileNotFoundError):
        scote.TwoPointResult.load(tmp_path / "missing.npz")


# ---------------------------------------------------------------------------------------------------------
# Tiles, validation, and the package namespace.
# ---------------------------------------------------------------------------------------------------------

def test_tile_keys_are_collision_free_beyond_the_data():
    """Tile keys of positions far beyond the data (coarse cell centers) and below the origin are unique and keep
    the (column, row) order; the index of a position without data is the next tile in that order."""
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 300, (3_000, 2))
    grid = pipeline._TileGrid(xy, 100.0, np.zeros(2))
    assert grid.n_tiles == 9
    cols, rows = np.meshgrid(np.arange(-3, 12), np.arange(-3, 12), indexing="ij")
    centers = np.stack([cols.ravel(), rows.ravel()], 1) * 100.0 + 50.0
    keys = grid.key(centers)
    assert len(np.unique(keys)) == len(keys) and (np.diff(keys) > 0).all()     # unique, (column, row) order
    idx = grid.index(centers)
    tab = grid.table()
    assert list(tab.column) == [0, 0, 0, 1, 1, 1, 2, 2, 2] and list(tab.row) == [0, 1, 2] * 3
    inside = (cols.ravel() >= 0) & (cols.ravel() < 3) & (rows.ravel() >= 0) & (rows.ravel() < 3)
    assert np.array_equal(idx[inside], cols.ravel()[inside] * 3 + rows.ravel()[inside])
    above = (cols.ravel() == 0) & (rows.ravel() == 5)            # beyond the last row of column 0:
    assert idx[above][0] == 3                                     # the first tile of column 1


def test_origin_above_the_data_raises():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 300, (3_000, 2))
    codes = rng.integers(0, 2, 3_000)
    with pytest.raises(ValueError, match="below the origin"):
        scote.twopoint(xy, codes, lattice_only=True, delta=2.0, origin=(150.0, 150.0), n_levels=1, tile_um=100.0)
    with pytest.raises(ValueError, match="below the origin"):
        scote.twopoint(xy, codes, origin=(150.0, 150.0), n_levels=1, tile_um=100.0)
    with pytest.raises(ValueError, match="below the origin"):
        scote.pair_counts(xy, codes, delta=2.0, origin=(0.0, 10.0), n_levels=1)
    for lattice_only in (True, False):
        with pytest.warns(UserWarning):
            res = scote.twopoint(xy, codes, lattice_only=lattice_only, delta=2.0, origin=(-50.0, -50.0), n_levels=1,
                                 tile_um=100.0)
        assert res.n_tiles == 16 and res.params["origin"] == [-50.0, -50.0]


@pytest.mark.parametrize("kwargs,match", [
    (dict(shells_per_level=7), "even"), (dict(shells_per_level=0), "even"), (dict(n_levels=-1), "n_levels"),
    (dict(n_levels=1.5), "n_levels"), (dict(tile_um=0.0), "tile_um"), (dict(tile_um=np.nan), "tile_um"),
    (dict(delta=0.0), "delta"), (dict(delta=-1.0), "delta"),
])
def test_invalid_parameters_raise(kwargs, match):
    rng = np.random.default_rng(2)
    xy = rng.uniform(0, 100, (500, 2))
    codes = rng.integers(0, 2, 500)
    base = dict(n_levels=1, tile_um=50.0)
    with pytest.raises(ValueError, match=match):
        scote.twopoint(xy, codes, **{**base, **kwargs})
    with pytest.raises(ValueError, match=match):
        scote.twopoint(xy, codes, lattice_only=True, **{**base, **kwargs})
    with pytest.raises(ValueError, match=match):
        scote.twopoint(xy, onehot_csr(codes, 2), bin_size=0.5, **{**base, **kwargs})


def test_nonfinite_coordinates_and_delta_raise():
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 100, (500, 2))
    codes = rng.integers(0, 2, 500)
    bad = xy.copy()
    bad[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        scote.twopoint(bad, codes, n_levels=1, tile_um=50.0)
    bad[3, 1] = np.inf
    with pytest.raises(ValueError, match="NaN or infinite"):
        scote.twopoint(bad, onehot_csr(codes, 2), delta=2.0, n_levels=1, tile_um=50.0)
    with pytest.raises(ValueError, match="delta"):
        scote.pair_counts(xy, codes, delta=0.0, n_levels=1)
    with pytest.raises(ValueError, match="bin_size"):
        scote.pair_counts(xy, onehot_csr(codes, 2), bin_size=-1.0, n_levels=1)


def test_gpu_backend_without_torch_names_the_extra(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)              # import torch -> ImportError
    rng = np.random.default_rng(4)
    xy = rng.uniform(0, 100, (500, 2))
    codes = rng.integers(0, 2, 500)
    with pytest.raises(ImportError, match=r"scote\[gpu\]"):
        scote.twopoint(xy, codes, n_levels=1, tile_um=50.0, backend="gpu")
    with pytest.raises(ImportError, match=r"scote\[gpu\]"):
        scote.pair_counts(xy, codes, delta=2.0, n_levels=1, backend="gpu")


def test_import_scote_exposes_the_submodules():
    import subprocess
    import sys
    code = "import scote; print(scote.io.read_bins.__name__, scote.uncertainty.closed_form_log_g.__name__)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["read_bins", "closed_form_log_g"]


def test_every_public_object_is_documented_once():
    """The public modules are ``scote``, ``scote.io``, and ``scote.uncertainty``; the objects of their ``__all__`` name
    that module as theirs, and no public module re-exports an object of another."""
    import inspect
    modules = {"scote": scote, "scote.io": scote.io, "scote.uncertainty": scote.uncertainty}
    seen = {}
    for module_name, module in modules.items():
        for name in module.__all__:
            obj = getattr(module, name)
            if inspect.ismodule(obj) or not callable(obj):
                continue
            assert obj.__module__ == module_name, f"{module_name}.{name}: {obj.__module__}"
            assert id(obj) not in seen, f"{module_name}.{name} is also {seen.get(id(obj))}"
            seen[id(obj)] = f"{module_name}.{name}"
    assert sorted(n for n in scote.__all__ if inspect.ismodule(getattr(scote, n))) == ["io", "uncertainty"]
    assert not hasattr(scote, "twopoint_molecules") and not hasattr(scote, "twopoint_bins")
    import pkgutil
    public = sorted(m.name for m in pkgutil.iter_modules(scote.__path__) if not m.name.startswith("_"))
    assert public == ["io", "uncertainty"]
    assert [m.name for m in pkgutil.iter_modules(scote.io.__path__) if not m.name.startswith("_")] == []
    assert {"visium_um_per_pixel", "cell_id_to_int", "aviti24_run_parameters"} <= set(scote.io.__all__)


def test_backend_and_names():
    """One backend argument (a name or an options object); the names of the shell argument and the result property."""
    rng = np.random.default_rng(5)
    xy = rng.uniform(0, 200, (3_000, 2))
    codes = rng.integers(0, 3, 3_000)
    with pytest.warns(UserWarning, match="fewer than 30 tiles"):
        res = scote.twopoint(xy, codes, shells_per_level=6, n_levels=2, tile_um=40.0,
                             backend=scote.CPUOptions(mode="sparse"))
    assert res.n_distance_bins == 6 + 3 * 2 and res.params["shells_per_level"] == 6
    assert res.params["backend"] == "cpu" and res.params["options"] == dict(
        mode="sparse", dense_threshold=0.05, row_block=200_000, host="numba", numba_acc_bytes=8 * 2**30)
    with pytest.raises(TypeError):
        scote.twopoint(xy, codes, n_bins=8)
    with pytest.raises(TypeError):
        scote.twopoint(xy, codes, cpu_mode="auto")
