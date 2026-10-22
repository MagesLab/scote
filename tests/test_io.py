"""Readers of the common data model and the feature selection."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scote import io

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402


def test_select_features(tmp_path):
    counts = pd.Series([50, 30, 30, 5], index=["a", "b", "c", "d"])
    assert io.select_features("top:2", counts) == ["a", "b"]
    assert io.select_features("min:30", counts) == ["a", "b", "c"]
    f = tmp_path / "genes.txt"
    f.write_text("# panel\nd\nz\na\n")
    assert io.select_features(f"file:{f}", counts) == ["d", "a"]
    assert io.select_features("c,a", counts) == ["c", "a"]


def test_read_molecules_subsample_and_codes(tmp_path):
    rng = np.random.default_rng(0)
    n = 5_000
    feat = rng.choice(["A", "B", "C"], n, p=[0.6, 0.3, 0.1])
    pq.write_table(pa.table(dict(x=rng.uniform(0, 100, n).astype(np.float32),
                                 y=rng.uniform(0, 100, n).astype(np.float32),
                                 feature=feat)), tmp_path / "m.parquet")
    df = io.read_molecules(tmp_path / "m.parquet", "top:2")
    assert list(df.columns) == ["x", "y", "feature"] and list(df["feature"].cat.categories) == ["A", "B"]
    assert len(df) == np.isin(feat, ["A", "B"]).sum() and df["x"].dtype == np.float32
    assert (df["feature"].astype(str).to_numpy() == feat[np.isin(feat, ["A", "B"])]).all()
    sub = io.read_molecules(tmp_path / "m.parquet", "top:2", max_molecules=100)
    idx = np.random.default_rng(0).choice(len(df), 100, replace=False)
    assert sub.equals(df.iloc[idx].reset_index(drop=True))
    every = io.read_molecules(tmp_path / "m.parquet")
    assert len(every) == n and list(every["feature"].cat.categories) == ["A", "B", "C"]


def test_read_bins_in_tissue_and_selection(tmp_path):
    X = sp.csr_matrix(np.array([[1, 0, 2], [0, 5, 0], [3, 0, 0], [0, 0, 9]], dtype=np.float32))
    np.savez(tmp_path / "b.npz", data=X.data, indices=X.indices, indptr=X.indptr, shape=np.array(X.shape),
             coords_um=np.arange(8, dtype=np.float32).reshape(4, 2), features=np.array(["g0", "g1", "g2"]),
             bin_um=2.0, in_tissue=np.array([True, True, True, False]))
    coords, Xs, names, bin_um = io.read_bins(tmp_path / "b.npz", "top:2")
    assert bin_um == 2.0 and coords.shape == (3, 2) and names == ["g1", "g0"]
    assert np.array_equal(Xs.toarray(), [[0, 1], [5, 0], [0, 3]])
    table = io.read_bins(tmp_path / "b.npz")
    assert isinstance(table, io.BinTable) and table.names == ["g0", "g1", "g2"] and table.bin_size == 2.0
    assert table.counts.shape == (3, 3) and table.xy.dtype == np.float64


def test_unknown_features_and_missing_files_raise(tmp_path):
    counts = pd.Series([5, 3], index=["a", "b"])
    with pytest.raises(KeyError, match="TYPO"):
        io.select_features("a,TYPO", counts)
    X = sp.csr_matrix(np.array([[1, 0], [0, 5]], dtype=np.float32))
    np.savez(tmp_path / "b.npz", data=X.data, indices=X.indices, indptr=X.indptr, shape=np.array(X.shape),
             coords_um=np.zeros((2, 2), np.float32), features=np.array(["g0", "g1"]), bin_um=2.0)
    with pytest.raises(KeyError, match="TYPO"):
        io.read_bins(tmp_path / "b.npz", "g0,TYPO")
    pq.write_table(pa.table(dict(x=np.zeros(3, np.float32), y=np.zeros(3, np.float32), feature=["A", "B", "A"])),
                   tmp_path / "m.parquet")
    with pytest.raises(KeyError, match="TYPO"):
        io.read_molecules(tmp_path / "m.parquet", "A,TYPO")
    for reader, name in ((io.read_bins, "none.npz"), (io.read_molecules, "none.parquet")):
        with pytest.raises(FileNotFoundError, match=name):
            reader(tmp_path / name)
    with pytest.raises(FileNotFoundError):
        io.read_bins(tmp_path / "b.npz", f"file:{tmp_path / 'none.txt'}")


def test_read_bins_needs_no_pickle(tmp_path):
    """Feature names stored as an object array (which needs pickle) raise with the fix; nothing is unpickled."""
    X = sp.csr_matrix(np.array([[1, 0], [0, 5]], dtype=np.float32))
    np.savez(tmp_path / "b.npz", data=X.data, indices=X.indices, indptr=X.indptr, shape=np.array(X.shape),
             coords_um=np.zeros((2, 2), np.float32), features=np.array(["g0", "g1"], dtype=object), bin_um=2.0)
    with pytest.raises(ValueError, match="unicode array"):
        io.read_bins(tmp_path / "b.npz")


def test_select_and_read_molecules_defaults(tmp_path):
    counts = pd.Series([5, 3, 1], index=["a", "b", "c"])
    assert io.select_features("c, a", counts) == ["c", "a"] and io.select_features(" b ,a,", counts) == ["b", "a"]
    pq.write_table(pa.table(dict(x=np.zeros(3, np.float32), y=np.zeros(3, np.float32), feature=["A", "B", "A"],
                                 cell_id=np.arange(3))), tmp_path / "m.parquet")
    assert list(io.read_molecules(tmp_path / "m.parquet").columns) == ["x", "y", "feature", "cell_id"]
    assert list(io.read_molecules(tmp_path / "m.parquet", columns=[]).columns) == ["x", "y", "feature"]
