"""The helpers vendored from TACCO (scote._tacco_compat) against the TACCO originals. The comparisons with
TACCO skip when TACCO is not installed."""
import numpy as np
import pandas as pd
import pytest

from scote import _tacco_compat as C


def _points(n=5_000, side=80.0, seed=0, dim=2, dtype=np.float64):
    rng = np.random.default_rng(seed)
    cols = ["x", "y", "z"][:dim]
    return pd.DataFrame(rng.uniform(0, side, (n, dim)).astype(dtype), columns=cols,
                        index=pd.Index([f"c{i}" for i in range(n)]))


@pytest.fixture(scope="module")
def tacco():
    return pytest.importorskip("tacco")


@pytest.fixture(scope="module")
def tdist(tacco):
    from tacco.utils import _dist
    return _dist


def test_min_dtype(tdist):
    for m in [0, 1, 126, 127, 2**15 - 2, 2**15 - 1, 2**31 - 2, 2**31 - 1, 2**63 - 2]:
        assert C.min_dtype(m) is tdist.min_dtype(m)
    for f in (C.min_dtype, tdist.min_dtype):
        with pytest.raises(ValueError):
            f(2**63)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_bin_and_hash(tdist, dim, dtype):
    df = _points(dim=dim, dtype=dtype)
    keys = list(df.columns)
    for size in [0.7, 3.0, 50.0]:
        for shift in [None, -np.full(dim, size / 3)]:
            b0 = tdist.bin(df, size, position_keys=keys, shift=None if shift is None else shift.copy())
            b1 = C.bin(df, size, position_keys=keys, shift=None if shift is None else shift.copy())
            pd.testing.assert_frame_equal(b0, b1)
            for compress in (True, False):
                h0 = tdist.hash(b0, compress=compress)
                h1 = C.hash(b1, compress=compress)
                if compress:
                    pd.testing.assert_series_equal(h0, h1)
                else:
                    assert h0.dtype == h1.dtype and np.array_equal(h0, h1)
    d0 = tdist.bin(df.copy(), 2.0, position_keys=keys, bin_keys=[f"b{k}" for k in keys])
    d1 = C.bin(df.copy(), 2.0, position_keys=keys, bin_keys=[f"b{k}" for k in keys])
    pd.testing.assert_frame_equal(d0, d1)
    other = _points(n=700, seed=5, dim=dim, dtype=dtype)
    bo0 = tdist.bin(other, 2.0, position_keys=keys)
    bo1 = C.bin(other, 2.0, position_keys=keys)
    b0 = tdist.bin(df, 2.0, position_keys=keys)
    b1 = C.bin(df, 2.0, position_keys=keys)
    for r0, r1 in zip(tdist.hash(b0, other=bo0), C.hash(b1, other=bo1)):
        pd.testing.assert_series_equal(r0, r1)


def test_enum_shifts(tacco, tdist):
    from tacco.tools._annotate import _enum_shifts
    for n_shifts in (1, 2, 3):
        for n_dim in (1, 2, 3):
            a = C.enum_shifts(n_shifts=n_shifts, n_dim=n_dim)
            assert np.array_equal(a, _enum_shifts(n_shifts=n_shifts, n_dim=n_dim))
            assert np.array_equal(a, tdist.enum_shifts(n_shifts, n_dim))


@pytest.mark.parametrize("mode", ["auto", 0, 1, 2])
@pytest.mark.filterwarnings("ignore:overflow encountered in scalar multiply:RuntimeWarning")   # inside TACCO
def test_getNN_indptr(tdist, mode):
    df = _points(n=8_000, side=120.0, seed=2)
    bins = C.bin(df, 6.0, position_keys=df.columns)
    hashes = C.hash(bins).reset_index(drop=True).sort_values()
    sorted_bins = hashes.drop_duplicates().sort_values()
    coords = bins.iloc[sorted_bins.index.to_numpy()].to_numpy()
    nn0 = tdist._getNN(coords, sorted_bins.to_numpy(), mode)
    nn1 = C._getNN(coords, sorted_bins.to_numpy())
    assert nn0.dtype == nn1.dtype and np.array_equal(nn0, nn1)
    dt = C.min_dtype(len(hashes))
    i0 = tdist._get_indptr(hashes.to_numpy(), hashes.max(), dt)
    i1 = C._get_indptr(hashes.to_numpy(), hashes.max(), dt)
    assert i0.dtype == i1.dtype and np.array_equal(i0, i1)


def test_guess_blocksize(tdist):
    for n, side in [(500, 50.0), (20_000, 200.0), (50_000, 30.0)]:
        df = _points(n=n, side=side, seed=n)
        for max_distance in (0.5, 4.0, 16.0):
            mine = C._guess_blocksize(df, max_distance, verbose=0)
            assert mine == tdist._guess_blocksize(df, max_distance, verbose=0)


@pytest.mark.filterwarnings("ignore:overflow encountered in scalar multiply:RuntimeWarning")   # inside TACCO
def test_enumeration_with_tacco_helpers_is_identical(tacco, tdist, monkeypatch):
    """The pair enumeration gives the same counts with the TACCO originals patched in."""
    from scote._twopoint import enumeration as E
    rng = np.random.default_rng(4)
    n, G = 20_000, 5
    xy = rng.uniform(0, 150, (n, 2))
    codes = rng.choice(G, n)
    mine = E.pair_counts_enumerated(xy, codes, G, 0.5, 8)
    for name, f in [("bin", tdist.bin), ("hash", tdist.hash), ("_getNN", lambda c, b: tdist._getNN(c, b, "auto")),
                    ("_get_indptr", tdist._get_indptr), ("min_dtype", tdist.min_dtype),
                    ("_guess_blocksize", tdist._guess_blocksize)]:
        monkeypatch.setattr(E, name, f)
    theirs = E.pair_counts_enumerated(xy, codes, G, 0.5, 8)
    assert mine.dtype == theirs.dtype and np.array_equal(mine, theirs)
