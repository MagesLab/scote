"""Input checks of scote.twopoint: wrong shapes and types raise with the fix, duplicate names and negative counts
raise, non-integer counts warn, unused categories are dropped, and the default number of levels."""
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import scote
from scote._twopoint import pipeline
from scote._twopoint.lattice import onehot_csr

KW = dict(n_levels=1, tile_um=1000.0)


def _few(func, *args, **kwargs):
    """Call ``func`` with the few-tiles warning silenced."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="closed-form tile bootstrap with")
        return func(*args, **kwargs)


def _data(n=2_000, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 200, (n, 2))
    codes = rng.integers(0, 3, n)
    return xy, codes


def test_wrong_shapes_and_types_raise_with_the_fix():
    xy, codes = _data()
    labels = np.array(["a", "b", "c"])[codes]
    with pytest.raises(ValueError, match="must be 1-D"):
        scote.twopoint(xy, codes[:, None], **KW)                                 # a column of codes
    with pytest.raises(ValueError, match=r"df\['gene'\], not df\[\['gene'\]\]"):
        scote.twopoint(xy, pd.DataFrame({"gene": labels}), **KW)               # df[["gene"]]
    with pytest.raises(ValueError, match="one column"):
        scote.twopoint(xy, pd.DataFrame({"code": codes}), **KW)
    with pytest.raises(TypeError, match="1-D array or Series"):
        scote.twopoint(xy, labels[:, None], **KW)                              # 2-D strings
    with pytest.raises(TypeError, match="non-numeric columns"):
        scote.twopoint(xy, pd.DataFrame({"gene": labels, "other": labels}), **KW)
    with pytest.raises(TypeError, match="holds numbers"):
        scote.twopoint(xy, np.stack([labels, labels], 1).astype(object), **KW)

    class AnnData:                                                             # the attributes of an AnnData
        X, obs, var, var_names = onehot_csr(codes, 3), None, None, ["a", "b", "c"]

    with pytest.raises(TypeError, match=r"adata\.X, names=adata\.var_names"):
        scote.twopoint(xy, AnnData(), **KW)
    with pytest.raises(TypeError, match="micrometers"):
        scote.twopoint(AnnData(), codes, **KW)
    with pytest.warns(UserWarning, match="single feature"):                    # one column with names: allowed
        res = _few(scote.twopoint, xy, (codes == 0).astype(float)[:, None], names=["a"], delta=2.0, bin_size=2.0,
                     **KW)
    assert res.features == ["a"] and np.nanmax(np.abs(res.log_g[np.isfinite(res.log_g)])) < 1e-12


def test_duplicate_names_raise():
    xy, codes = _data()
    X = onehot_csr(codes, 3)
    with pytest.raises(ValueError, match="not unique: 'a'"):
        scote.twopoint(xy, codes, names=["a", "b", "a"], **KW)
    with pytest.raises(ValueError, match="not unique"):
        scote.twopoint(xy, X, names=["a", "a", "b"], bin_size=2.0, **KW)
    with pytest.raises(ValueError, match="not unique"):
        scote.twopoint(xy, pd.DataFrame(X.toarray(), columns=["a", "b", "a"]), bin_size=2.0, **KW)
    mixed = np.array([1, "1", 2], dtype=object)[codes]                         # 1 and "1" are both named "1"
    with pytest.raises(ValueError, match="not unique"):
        scote.twopoint(xy, mixed, **KW)


def test_count_values():
    xy, codes = _data()
    X = onehot_csr(codes, 3).astype(np.float64)
    bad = X.copy()
    bad.data[5] = -1.0
    with pytest.raises(ValueError, match="negative"):
        scote.twopoint(xy, bad, bin_size=2.0, **KW)
    bad.data[5] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        scote.twopoint(xy, bad, bin_size=2.0, **KW)
    with pytest.raises(ValueError, match="negative"):
        scote.pair_counts(xy, -X.toarray(), bin_size=2.0, n_levels=1)
    frac = X.copy()
    frac.data = frac.data * 0.5
    with pytest.warns(UserWarning, match="not integers"):
        _few(scote.twopoint, xy, frac, bin_size=2.0, **KW)


def test_unused_categories_are_dropped_in_their_order():
    xy, codes = _data()
    labels = np.array(["a", "b", "c"])[codes]
    keep = labels != "c"
    full = pd.Categorical(labels[keep], categories=["e", "b", "c", "a", "d"])  # c, d, e: no points
    res = _few(scote.twopoint, xy[keep], full, **KW)
    ref = _few(scote.twopoint, xy[keep], pd.Categorical(labels[keep], categories=["b", "a"]), **KW)
    assert res.features == ["b", "a"] and np.array_equal(res.P, ref.P)
    assert np.array_equal(res.sd, ref.sd, equal_nan=True)
    series = _few(scote.twopoint, xy[keep], pd.Series(full), select=["a", "b"], **KW)
    assert series.features == ["a", "b"]
    with pytest.raises(KeyError, match="'c'"):
        scote.twopoint(xy[keep], full, select=["a", "c"], **KW)
    # integer codes with names keep every name (a fixed feature axis)
    named = _few(scote.twopoint, xy[keep], codes[keep], names=["a", "b", "c"], **KW)
    assert named.features == ["a", "b", "c"] and not named.P[:, 2].any()


def test_default_number_of_levels_reaches_16_mm():
    assert [pipeline._default_n_levels(d, 8) for d in (0.5, 2.0, 64.0, 8.0)] == [12, 10, 5, 8]
    xy, codes = _data(n=500)
    for kwargs in (dict(), dict(lattice_only=True), dict(lattice_only=True, delta=2.0)):
        res = _few(scote.twopoint, xy, codes, tile_um=1000.0, **kwargs)
        assert res.bins.r_to.iloc[-1] == pytest.approx(16384.0)


def test_micrometer_coordinates_give_the_integer_lattice_indices():
    """Bin centers in micrometers (float32, as the converters store them, with an offset) and the integer bin
    indices give identical pair counts: floor((x - origin) / delta) reproduces the indices exactly."""
    rng = np.random.default_rng(2)
    ij = np.stack(np.meshgrid(np.arange(300), np.arange(200), indexing="ij"), -1).reshape(-1, 2)
    ij = ij[rng.random(len(ij)) < 0.6]
    X = sp.csr_matrix(rng.poisson(0.4, (len(ij), 3)).astype(np.float64))
    um = ((ij + 0.5) * 2.0 + 1234.5).astype(np.float32)
    assert np.array_equal(np.floor((um.astype(np.float64) - um.min(0)) / 2.0), ij - ij.min(0))
    a = scote.pair_counts(um, X, delta=2.0, bin_size=2.0, n_levels=3)
    b = scote.pair_counts(ij.astype(np.float64), X, delta=1.0, bin_size=1.0, n_levels=3)
    assert np.array_equal(a.counts, b.counts)
    c = scote.pair_counts(um, X, delta=8.0, bin_size=2.0, n_levels=1)        # coarse-grained by 4
    d = scote.pair_counts(ij.astype(np.float64), X, delta=4.0, bin_size=1.0, n_levels=1)
    assert np.array_equal(c.counts, d.counts)
