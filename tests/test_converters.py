"""Converters of the vendor formats on tiny synthetic inputs (MERSCOPE, Visium, Visium HD) and their helpers. The
Space Ranger matrices are read through ``scanpy.read_10x_h5``, which is replaced by a stub here, so the tests run
without the ``io`` extra."""
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scote import io
from scote.io import cell_id_to_int, visium_um_per_pixel

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402


class _AnnData:
    """The part of an AnnData object the converters use."""

    def __init__(self, X, barcodes, genes):
        self.X, self.obs_names, self.var_names = X, pd.Index(barcodes), pd.Index(genes)


@pytest.fixture
def fake_scanpy(monkeypatch):
    """A stub ``scanpy`` whose read_10x_h5 returns the matrix registered for a path, warning about duplicate gene
    names as AnnData does."""
    import warnings
    matrices = {}

    def read_10x_h5(path):
        ad = matrices[str(path)]
        if not ad.var_names.is_unique:
            warnings.warn("Variable names are not unique. To make them unique, call `.var_names_make_unique`.",
                          UserWarning)
        return ad

    module = types.ModuleType("scanpy")
    module.read_10x_h5 = read_10x_h5
    monkeypatch.setitem(sys.modules, "scanpy", module)
    return matrices


def test_cell_id_to_int():
    assert cell_id_to_int("aaaaaaaa-1") == 0
    assert cell_id_to_int("aaaaaaab-1") == 1
    assert cell_id_to_int("ffkpbaba-1") == int("55af1010", 16)
    assert cell_id_to_int("ba") == 16


def test_convert_merscope(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    genes = rng.choice(["Cd3e", "Ms4a1", "Blank-0001"], n, p=[0.5, 0.4, 0.1])
    cell = rng.choice([np.nan, 0, 12, 40], n)
    raw = tmp_path / "raw" / "region_0"
    raw.mkdir(parents=True)
    pd.DataFrame(dict(barcode_id=rng.integers(0, 9, n), global_x=rng.uniform(0, 50, n), global_y=rng.uniform(0, 80, n),
                      global_z=rng.integers(0, 7, n).astype(float), gene=genes, cell_id=cell,
                      transcript_id=np.arange(n))).to_csv(raw / "detected_transcripts.csv", index=False)
    qc = io.convert_merscope(tmp_path / "raw", tmp_path / "out", "M1", chunk_rows=100)
    t = pq.read_table(tmp_path / "out" / "molecules.parquet").to_pandas()
    keep = genes != "Blank-0001"
    assert qc["molecules"] == len(t) == keep.sum() and qc["dropped_control"] == (~keep).sum()
    assert set(t.feature.astype(str)) == {"Cd3e", "Ms4a1"} and not t.nucleus.any()
    expected_cell = np.where(np.isnan(cell[keep]) | (cell[keep] == 0), -1, np.nan_to_num(cell[keep])).astype(int)
    assert np.array_equal(t.cell_id.to_numpy(), expected_cell)
    assert json.load(open(tmp_path / "out" / "feature_counts.json")) == {g: int((genes == g).sum())
                                                                         for g in ("Cd3e", "Ms4a1")}
    with pytest.raises(FileNotFoundError, match="detected_transcripts"):
        io.convert_merscope(tmp_path / "out", tmp_path / "out2", "M1")


def _hex_positions(n_rows=10, n_cols=12, um_per_px=0.5):
    """A Visium-like hexagonal spot array: adjacent spots of an array row are 2 array columns and 100 um apart."""
    rows = []
    pitch_px = 100.0 / um_per_px
    for r in range(n_rows):
        for j in range(n_cols):
            c = 2 * j + (r % 2)
            rows.append((f"S{r}_{c}-1", int(r < n_rows - 1), r, c, r * pitch_px * np.sqrt(3) / 2, c * pitch_px / 2))
    return pd.DataFrame(rows, columns=["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres",
                                       "pxl_col_in_fullres"])


def test_visium_um_per_pixel():
    pos = _hex_positions(um_per_px=0.37)
    upp, pitch_px = visium_um_per_pixel(pos)
    assert upp == pytest.approx(0.37) and pitch_px == pytest.approx(100 / 0.37)
    with pytest.raises(ValueError, match="too few"):
        visium_um_per_pixel(pos.iloc[:20])


def test_convert_visium(tmp_path, fake_scanpy):
    pos = _hex_positions()
    spatial = tmp_path / "raw" / "outs" / "spatial"
    spatial.mkdir(parents=True)
    pos.to_csv(spatial / "tissue_positions_list.csv", header=False, index=False)
    (spatial / "scalefactors_json.json").write_text(json.dumps(dict(spot_diameter_fullres=130.0)))
    h5 = tmp_path / "raw" / "outs" / "filtered_feature_bc_matrix.h5"
    h5.write_bytes(b"")
    barcodes = list(pos.barcode[::-1])                                  # matrix order differs from the positions
    X = sp.random(len(barcodes), 5, density=0.4, format="csr", random_state=1)
    fake_scanpy[str(h5)] = _AnnData(X, barcodes, [f"g{i}" for i in range(5)])
    qc = io.convert_visium(tmp_path / "raw", tmp_path / "out", "V1")
    assert qc["microns_per_pixel"] == pytest.approx(0.5) and qc["n_spots"] == len(barcodes)
    coords, Xr, names, bin_um = io.read_bins(tmp_path / "out" / "bins_spots55um.npz", "top:5")
    p = pos.set_index("barcode").loc[barcodes]
    np.testing.assert_allclose(coords, np.stack([p.pxl_col_in_fullres * 0.5, p.pxl_row_in_fullres * 0.5], 1),
                               rtol=1e-6)
    assert bin_um == 55.0 and Xr.sum() == pytest.approx(X.sum(), rel=1e-6)
    assert np.load(tmp_path / "out" / "bins_spots55um.npz")["features"].dtype.kind == "U"
    with pytest.raises(FileNotFoundError, match="filtered_feature_bc_matrix.h5"):
        io.convert_visium(tmp_path / "out", tmp_path / "out2", "V1")
    # the reader returns what the converter writes (all spots of the matrix; float32 positions as float64)
    table = io.read_visium(tmp_path / "raw")
    disk = io.read_bins(tmp_path / "out" / "bins_spots55um.npz")
    assert table.names == disk.names and table.bin_size == disk.bin_size == 55.0 and table.xy.dtype == np.float64
    assert np.array_equal(table.xy, disk.xy)
    np.testing.assert_allclose(table.counts.toarray(), disk.counts.toarray(), rtol=1e-6)   # float32 on disk


def test_duplicate_gene_names_are_made_unique(tmp_path, fake_scanpy):
    """Reader and converter make duplicate gene names unique the same way (AnnData's rule), without the warning."""
    import warnings
    pos = _hex_positions()
    spatial = tmp_path / "raw" / "spatial"
    spatial.mkdir(parents=True)
    pos.to_csv(spatial / "tissue_positions.csv", index=False)
    (spatial / "scalefactors_json.json").write_text(json.dumps(dict(spot_diameter_fullres=130.0)))
    h5 = tmp_path / "raw" / "filtered_feature_bc_matrix.h5"
    h5.write_bytes(b"")
    X = sp.random(len(pos), 5, density=0.4, format="csr", random_state=4)
    fake_scanpy[str(h5)] = _AnnData(X, list(pos.barcode), ["A", "B", "A", "A-1", "A"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        table = io.read_visium(tmp_path / "raw")
        io.convert_visium(tmp_path / "raw", tmp_path / "out", "V")
    assert table.names == ["A", "B", "A-2", "A-1", "A-3"]
    assert io.read_bins(tmp_path / "out" / "bins_spots55um.npz").names == table.names


def test_convert_visium_hd(tmp_path, fake_scanpy):
    rng = np.random.default_rng(2)
    bdir = tmp_path / "raw" / "outs" / "binned_outputs" / "square_002um"
    (bdir / "spatial").mkdir(parents=True)
    n = 50
    barcodes = [f"s_002um_{i:05d}-1" for i in range(n)]
    pd.DataFrame(dict(barcode=barcodes, in_tissue=(np.arange(n) % 5 != 0).astype(int),
                      pxl_row_in_fullres=rng.uniform(0, 1000, n), pxl_col_in_fullres=rng.uniform(0, 1000, n))
                 ).to_parquet(bdir / "spatial" / "tissue_positions.parquet")
    (bdir / "spatial" / "scalefactors_json.json").write_text(json.dumps(dict(microns_per_pixel=0.27)))
    (bdir / "filtered_feature_bc_matrix.h5").write_bytes(b"")
    X = sp.random(n, 4, density=0.3, format="csr", random_state=3)
    fake_scanpy[str(bdir / "filtered_feature_bc_matrix.h5")] = _AnnData(X, barcodes, ["a", "b", "c", "d"])
    qc = io.convert_visium_hd(tmp_path / "raw", tmp_path / "out", "HD1")
    assert list(qc["bins"]) == ["square_002um"] and qc["bins"]["square_002um"]["n_bins"] == n
    coords, Xr, names, bin_um = io.read_bins(tmp_path / "out" / "bins_square_002um.npz", "a,c")
    assert bin_um == 2.0 and names == ["a", "c"] and Xr.shape == (40, 2)              # in_tissue drops 10 bins
    with pytest.raises(FileNotFoundError, match="square_002um"):
        io.convert_visium_hd(tmp_path / "out", tmp_path / "out2", "HD1")
    with pytest.raises(FileNotFoundError):
        io.convert_visium_hd(tmp_path / "missing", tmp_path / "out2", "HD1")
    # the reader returns what the converter writes, with and without the in-tissue filter
    disk = io.read_bins(tmp_path / "out" / "bins_square_002um.npz")
    table = io.read_visium_hd(tmp_path / "raw")
    assert table.names == disk.names == ["a", "b", "c", "d"] and table.bin_size == disk.bin_size == 2.0
    assert table.xy.dtype == np.float64 and np.array_equal(table.xy, disk.xy) and len(table.xy) == 40
    np.testing.assert_allclose(table.counts.toarray(), disk.counts.toarray(), rtol=1e-6)    # float32 on disk
    every = io.read_visium_hd(tmp_path / "raw", in_tissue_only=False)
    assert len(every.xy) == n and np.array_equal(every.counts.toarray(), X.toarray())
    with pytest.raises(FileNotFoundError, match="square_008um"):
        io.read_visium_hd(tmp_path / "raw", bin="square_008um")


def test_convert_xenium_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError, match="transcripts.parquet"):
        io.convert_xenium(tmp_path, tmp_path / "out", "X")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        io.convert_xenium(tmp_path / "missing", tmp_path / "out", "X")
