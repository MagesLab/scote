"""In-memory readers and converters of every platform on tiny synthetic vendor outputs: the reader returns what the
converter writes, in the form scote.twopoint takes."""
import json
import warnings

import numpy as np
import pandas as pd
import pytest

import scote
from scote import io

import _fixtures as F

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402


def _same_molecules(df: pd.DataFrame, path) -> None:
    """The DataFrame of a reader equals the parquet file of the converter (values, column by column)."""
    written = pq.read_table(path).to_pandas()
    assert list(df.columns) == list(written.columns) and len(df) == len(written)
    for c in df.columns:
        a, b = df[c], written[c]
        if isinstance(a.dtype, pd.CategoricalDtype) or isinstance(b.dtype, pd.CategoricalDtype):
            assert (a.astype(str).to_numpy() == b.astype(str).to_numpy()).all(), c
        else:
            assert np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=a.dtype.kind == "f"), c
    assert list(df["feature"].cat.categories) == sorted(df["feature"].cat.categories)


def _twopoint_runs(df: pd.DataFrame) -> None:
    """The reader's frame is a valid input of scote.twopoint."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)                 # few tiles
        res = scote.twopoint(df[["x", "y"]].to_numpy(), df["feature"], n_levels=1, tile_um=1_000.0)
    assert res.features == list(df["feature"].cat.categories)
    assert np.array_equal(res.totals, df["feature"].value_counts(sort=False).to_numpy())


def test_read_xenium(tmp_path):
    raw = F.xenium(tmp_path / "raw")
    qc = io.convert_xenium(raw, tmp_path / "out", "S", batch_rows=250)
    df = io.read_xenium(raw, sample="S", batch_rows=250)
    _same_molecules(df, tmp_path / "out" / "molecules.parquet")
    assert qc["molecules"] == len(df) and set(df["feature"].cat.categories) == {"GENE1", "GENE2"}
    small = io.read_xenium(raw, columns=["cell_id"])
    assert list(small.columns) == ["x", "y", "feature", "cell_id"]
    _twopoint_runs(df)
    with pytest.raises(FileNotFoundError, match="transcripts.parquet"):
        io.read_xenium(tmp_path / "out")


def test_read_merscope(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(dict(global_x=rng.uniform(0, 50, n), global_y=rng.uniform(0, 80, n), global_z=np.zeros(n),
                      gene=rng.choice(["Cd3e", "Ms4a1", "Blank-0001"], n), cell_id=rng.choice([np.nan, 0, 12], n))
                 ).to_csv(raw / "detected_transcripts.csv", index=False)
    io.convert_merscope(raw, tmp_path / "out", "M", chunk_rows=150)
    df = io.read_merscope(raw, sample="M", chunk_rows=150)
    _same_molecules(df, tmp_path / "out" / "molecules.parquet")
    _twopoint_runs(df)


def test_read_and_convert_cosmx(tmp_path):
    src = F.cosmx(tmp_path / "raw")
    raw = pd.read_csv(src)
    qc = io.convert_cosmx(tmp_path / "raw", tmp_path / "out", "C", z_step_um=0.8, block_mb=1, drop_controls=False)
    df = io.read_cosmx(src, sample="C", z_step_um=0.8, drop_controls=False)
    _same_molecules(df, tmp_path / "out" / "molecules.parquet")
    assert qc["molecules"] == len(raw) == len(df) and qc["n_fov"] == raw.fov.nunique()
    np.testing.assert_allclose(df["x"], (raw.x_global_px * 0.120280945).astype(np.float32))
    assert np.array_equal(df["z"], (raw.z * 0.8).astype(np.float32))
    assert np.array_equal(df["cell_id"], np.where(raw.cell_ID == 0, -1, raw.fov * 100000 + raw.cell_ID))
    assert np.array_equal(df["nucleus"], raw.CellComp == "Nuclear")
    assert dict(df["codeclass"].value_counts()) == {k: v for k, v in qc["codeclass"].items() if v}
    genes = io.read_cosmx(src)                                        # controls dropped by default
    assert set(genes["feature"].cat.categories) == {"GENEA", "GENEB"} and genes["z"].isna().all()
    added = io.convert_cosmx(src, tmp_path / "out2", "C", transform=lambda t: t.append_column(
        "flag", pa.array(np.ones(t.num_rows, bool))))
    assert added["molecules"] == len(genes) == added["codeclass"]["Endogenous"]
    assert pq.read_table(tmp_path / "out2" / "molecules.parquet").column("flag")
    _twopoint_runs(genes)


def test_read_and_convert_atera(tmp_path):
    pytest.importorskip("zarr")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)                 # zip members written twice by zarr
        raw = F.atera(tmp_path / "raw")
    qc = io.convert_atera(raw, tmp_path / "out", "A", batch_rows=100)
    df = io.read_atera(raw, sample="A", batch_rows=100)
    _same_molecules(df, tmp_path / "out" / "molecules.parquet")
    assert qc["molecules"] == len(df) == qc["molecules_in"] - qc["dropped_qv"] - qc["dropped_control"]
    assert "NegControlProbe_1" not in set(df["feature"]) and (df["qv"] >= 20).all()
    # labels: pixel = 2 um; cells 1 (x < 20, y < 15), 2 (x < 20, y >= 15), 3 (x >= 20) -> ids 11, 12, 13
    expected = np.where(df["x"] >= 20, 13, np.where(df["y"] < 15, 11, 12))
    assert np.array_equal(df["cell_id"], expected)
    assert qc["atera"]["matrix_check"]["matrix_cells"] == 3
    with pytest.raises(FileNotFoundError, match="transcripts.zarr.zip"):
        io.read_atera(tmp_path / "out")
    _twopoint_runs(df)


def test_read_and_convert_aviti24(tmp_path):
    archive = F.aviti24(tmp_path / "run.zarr.zip")
    wells = io.aviti24_wells(archive)
    assert wells == {"A1": dict(condition="ctrl", batches=["B01", "B02"]),
                     "B1": dict(condition="drug", batches=["B01", "B02"])}
    assert io.aviti24_run_parameters(archive) == dict(pixel_size_um=0.48, z_positions_um=[2, 1, 0, -1, -2,
                                                                                                   -3, -4])
    qc = io.convert_aviti24(archive, tmp_path / "out", "A1", flush_rows=150)
    df = io.read_aviti24(archive, "A1")
    _same_molecules(df, tmp_path / "out" / "molecules.parquet")
    assert qc["per_batch"]["B01"]["molecules_in"] == 200 and qc["per_batch"]["B02"]["molecules_in"] == 400
    b01 = df[df["batch"] == "B01"]
    assert "Unassigned" not in set(b01["feature"]) and len(b01) == qc["per_batch"]["B01"]["molecules"]
    assert set(df["z"]) <= set(np.float32([2, 1, 0, -1, -2, -3, -4])) and (df["condition"] == "ctrl").all()
    groups = pq.ParquetFile(tmp_path / "out" / "molecules.parquet")
    for i in range(groups.num_row_groups):                           # no row group spans two batches
        assert len(set(groups.read_row_group(i, columns=["batch"]).column("batch").to_pylist())) == 1
    only = io.read_aviti24(archive, "B1", batches=["B02"], pixel_um=1.0)
    assert set(only["batch"]) == {"B02"} and only["x"].max() > 1000
    with pytest.raises(ValueError, match="not in"):
        io.read_aviti24(archive, "C9")
    _twopoint_runs(df)


def test_read_and_convert_stereoseq(tmp_path):
    path = F.gem(tmp_path / "s.gem.gz")
    raw = pd.read_csv(path, sep="\t", comment="#")
    table = io.read_stereoseq(path, bin_dnb=50)
    assert table.bin_size == 25.0 and set(table.names) == {"Actb", "Gapdh", "Mt1"}
    assert table.counts.sum() == raw.MIDCount.sum()
    bx, by = raw.x // 50, raw.y // 50
    assert table.counts.shape[0] == len(set(zip(bx, by)))
    centers = np.c_[bx * 50 + 24.5, by * 50 + 24.5] * 0.5
    assert set(map(tuple, table.xy)) == set(map(tuple, centers))
    one = io.read_stereoseq(path, bin_dnb=1, gene_column="geneID")
    assert one.bin_size == 0.5 and set(one.names) == {"ENSG1", "ENSG2", "ENSG3"}
    qc = io.convert_stereoseq(path, tmp_path / "out", "S", bin_dnb=50)
    back = io.read_bins(qc["output"])
    assert back.bin_size == 25.0 and back.names == table.names and (back.counts != table.counts).nnz == 0
    np.testing.assert_allclose(back.xy, table.xy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = scote.twopoint(*table[:2], names=table.names, bin_size=table.bin_size, n_levels=1, tile_um=1000.0)
    assert res.params["delta"] == 25.0
    with pytest.raises(ValueError, match="bin_dnb"):
        io.read_stereoseq(path, bin_dnb=0)
    with pytest.raises(FileNotFoundError):
        io.read_stereoseq(tmp_path / "none.gem")
    small = io.read_stereoseq(path, bin_dnb=50, chunk_rows=97)       # chunks reduced to (bin, gene) sums
    assert small.names == table.names and np.array_equal(small.xy, table.xy)
    assert (small.counts != table.counts).nnz == 0


def test_read_and_convert_stereoseq_gef(tmp_path):
    pytest.importorskip("h5py")
    gem = F.gem(tmp_path / "s.gem.gz")
    path = F.gef(tmp_path / "s.gef", gem)
    ref = io.read_stereoseq(gem, bin_dnb=20)
    for chunk_rows in (10_000_000, 37):
        table = io.read_stereoseq(path, bin_dnb=20, chunk_rows=chunk_rows)
        assert table.names == ["Mt1", "Actb", "Gapdh"] and table.bin_size == 10.0      # resolution 500 nm
        assert np.array_equal(table.xy, ref.xy)
        order = [ref.names.index(n) for n in table.names]
        assert np.array_equal(table.counts.toarray(), ref.counts.toarray()[:, order])
    qc = io.convert_stereoseq(path, tmp_path / "out", "S", bin_dnb=20)
    back = io.read_bins(qc["output"])
    assert back.names == table.names and (back.counts != table.counts).nnz == 0
    assert io.read_stereoseq(path, bin_dnb=20, dnb_pitch_um=1.0).bin_size == 20.0
    with pytest.raises(ValueError, match="gene name field"):
        io.read_stereoseq(path, gene_column="geneSymbol")
    import h5py
    with h5py.File(tmp_path / "cell.gef", "w") as f:
        f.create_group("cellBin")
    with pytest.raises(ValueError, match="geneExp/bin1"):
        io.read_stereoseq(tmp_path / "cell.gef")


def test_cli_converts_every_platform(tmp_path):
    from scote import _cli as cli
    src = F.cosmx(tmp_path / "cosmx")
    assert cli.main(["convert", "--platform", "cosmx", "--rawdir", str(src), "--outdir", str(tmp_path / "c"),
                     "--sample", "C", "--qc", str(tmp_path / "c.json"), "--z-step-um", "0.8"]) == 0
    qc = json.load(open(tmp_path / "c.json"))
    assert qc["z_step_um"] == 0.8 and qc["codeclass"]["Negative"] == 0
    assert cli.main(["convert", "--platform", "cosmx", "--rawdir", str(src), "--outdir", str(tmp_path / "kidney_1"),
                     "--qc", str(tmp_path / "k.json"), "--keep-controls"]) == 0
    qc = json.load(open(tmp_path / "k.json"))
    assert qc["sample"] == "kidney_1" and qc["codeclass"]["Negative"] > 0             # sample: the output directory
    assert set(pq.read_table(tmp_path / "kidney_1" / "molecules.parquet").column("sample").to_pylist()) == {"kidney_1"}
    archive = F.aviti24(tmp_path / "run.zarr.zip")
    assert cli.main(["convert", "--platform", "aviti24", "--rawdir", str(archive), "--list-wells"]) == 0
    assert cli.main(["convert", "--platform", "aviti24", "--rawdir", str(archive), "--well", "A1", "--batches", "B02",
                     "--outdir", str(tmp_path / "a"), "--qc", str(tmp_path / "a.json")]) == 0
    assert list(json.load(open(tmp_path / "a.json"))["per_batch"]) == ["B02"]
    gem = F.gem(tmp_path / "s.gem.gz")
    assert cli.main(["convert", "--platform", "stereoseq", "--rawdir", str(gem), "--bin-dnb", "20", "--outdir",
                     str(tmp_path / "s"), "--qc", str(tmp_path / "s.json")]) == 0
    assert (tmp_path / "s" / "bins_bin20.npz").exists()
    with pytest.raises(SystemExit):
        cli.main(["convert", "--platform", "aviti24", "--rawdir", str(archive), "--outdir", str(tmp_path / "x"),
                  "--qc", str(tmp_path / "x.json")])
