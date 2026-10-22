"""The command line (python -m scote): twopoint runs to npz and directory outputs, and conversion."""
import argparse
import json
import subprocess
import sys
import zipfile

import numpy as np
import pytest

import scote
from scote import _cli as cli
from scote.uncertainty import closed_form_log_g

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

NPZ_KEYS = {"log_g", "lo", "hi", "sd", "n_eff_pairs", "P", "t_eff", "flag", "P_tot", "totals", "totals_tiles",
            "features", "level", "shell", "r_from", "r_to", "r_central", "tile_column", "tile_row", "tile_x_from",
            "tile_y_from"}


@pytest.fixture
def molecules(tmp_path):
    rng = np.random.default_rng(1)
    n = 20_000
    tbl = pa.table(dict(x=rng.uniform(0, 400, n), y=rng.uniform(0, 400, n),
                        feature=rng.choice(["A", "B", "C"], n, p=[0.6, 0.3, 0.1])))
    path = tmp_path / "m.parquet"
    pq.write_table(tbl, path)
    return path


def test_twopoint_npz_layout_and_values(tmp_path, molecules):
    args = ["twopoint", "--molecules", str(molecules), "--select", "top:3", "--lattice-only", "--delta", "2",
            "--n-levels", "3", "--tile-um", "100", "--backend", "cpu,mode=sparse"]
    with pytest.warns(UserWarning, match="fewer than 30 tiles"):
        assert cli.main(args + ["--out", str(tmp_path / "run.npz")]) == 0
    out = tmp_path / "run.npz"
    assert all(zi.compress_type == zipfile.ZIP_STORED for zi in zipfile.ZipFile(out).infolist())
    d = np.load(out)
    assert set(d.files) == NPZ_KEYS | {"counts_tiles"} and list(map(str, d["features"])) == ["A", "B", "C"]
    ref = closed_form_log_g(d["counts_tiles"], d["totals_tiles"], min_tiles_warn=0)
    for key in ("log_g", "lo", "hi", "sd", "n_eff_pairs", "P", "t_eff", "flag"):
        assert np.array_equal(d[key], getattr(ref, key), equal_nan=True), key
    j = json.load(open(tmp_path / "run.json"))
    assert j["uncertainty"] == "closed" and j["tile_um"] == 100.0 and len(j["flag_fractions"]) == 3
    assert {"save", "total_cli"} <= set(j["timings"]) and j["n_units"] == 20_000
    # the same run as a directory: the same arrays (the dtype does not depend on the output type)
    with pytest.warns(UserWarning):
        cli.main(args + ["--out", str(tmp_path / "dir")])
    assert np.array_equal(np.load(tmp_path / "dir" / "P.npy"), d["P"])
    assert np.array_equal(np.load(tmp_path / "dir" / "sd.npy"), d["sd"], equal_nan=True)
    with pytest.warns(UserWarning):
        cli.main(args + ["--dtype", "float32", "--out", str(tmp_path / "dir32")])
    assert np.load(tmp_path / "dir32" / "sd.npy").dtype == np.float32
    np.testing.assert_allclose(np.load(tmp_path / "dir32" / "sd.npy"), d["sd"].astype(np.float32), rtol=1e-6)
    meta = json.load(open(tmp_path / "dir" / "meta.json"))
    assert meta["n_tiles"] == 16 and "total_cli" in meta["timings"]
    # --no-store-tiles drops the tile tensor only
    with pytest.warns(UserWarning):
        cli.main(args + ["--no-store-tiles", "--out", str(tmp_path / "small.npz")])
    small = np.load(tmp_path / "small.npz")
    assert set(small.files) == NPZ_KEYS and np.array_equal(small["sd"], d["sd"], equal_nan=True)


def test_twopoint_molecule_level_matches_library(tmp_path, molecules):
    with pytest.warns(UserWarning):
        cli.main(["twopoint", "--molecules", str(molecules), "--select", "B,A", "--n-levels", "2", "--tile-um",
                  "100", "--threads", "1", "--out", str(tmp_path / "mol.npz")])
    df = scote.io.read_molecules(molecules, "B,A")
    assert list(df["feature"].cat.categories) == ["B", "A"]
    with pytest.warns(UserWarning):
        ref = scote.twopoint(df[["x", "y"]].to_numpy(), df["feature"], n_levels=2, tile_um=100.0)
    back = scote.TwoPointResult.load(tmp_path / "mol.npz")
    assert back.features == ["B", "A"] and np.array_equal(back.P, ref.P)
    assert np.array_equal(back.sd, ref.sd, equal_nan=True) and back.params["threads"] == 1


def test_cli_defaults_are_the_library_defaults(molecules, tmp_path):
    import inspect
    a = cli._parser().parse_args(["twopoint", "--molecules", "m.parquet", "--out", "x.npz"])
    p = inspect.signature(scote.twopoint).parameters
    assert a.shells_per_level == p["shells_per_level"].default and a.tile_um == p["tile_um"].default
    assert a.dtype == p["dtype"].default and a.acc_dtype == p["acc_dtype"].default
    assert a.backend == p["backend"].default
    assert a.delta is None and p["delta"].default is None and a.n_levels is None and p["n_levels"].default is None
    assert a.select is None and p["select"].default is None
    assert cli._backend("cpu") == "cpu" and cli._backend("gpu,") == "gpu"
    assert cli._backend("gpu, device = cuda:1 ") == scote.GPUOptions(device="cuda:1")
    assert cli._backend("cpu,mode=sparse,numba_acc_bytes=1024") == scote.CPUOptions(mode="sparse", numba_acc_bytes=1024)
    assert cli._backend("gpu,device_slabs=true") == scote.GPUOptions(device_slabs=True)
    for bad in ("tpu", "mode=sparse", "cpu,mode", "cpu,speed=3", "cpu,row_block=many", "cpu,mode=fast",
                "gpu,mode=sparse"):
        with pytest.raises(argparse.ArgumentTypeError):
            cli._backend(bad)
    with pytest.raises(SystemExit):
        cli._parser().parse_args(["twopoint", "--molecules", "m.parquet", "--out", "x.npz", "--backend", "cpu,mode"])
    with pytest.warns(UserWarning):
        cli.main(["twopoint", "--bins", str(_bins_file(tmp_path)), "--tile-um", "40", "--out", str(tmp_path / "b.npz")])
    meta = json.load(open(tmp_path / "b.json"))
    assert meta["n_levels"] == 10 and meta["delta"] == meta["bin_size"] == 2.0 and meta["bin_size_source"] == "given"
    with pytest.warns(UserWarning):
        cli.main(["twopoint", "--molecules", str(molecules), "--tile-um", "100", "--out", str(tmp_path / "m.npz")])
    meta = json.load(open(tmp_path / "m.json"))
    assert meta["n_levels"] == 12 and meta["delta"] == 0.5 and not meta["lattice_only"]
    with pytest.raises(SystemExit):
        cli.main(["twopoint", "--bins", str(_bins_file(tmp_path)), "--lattice-only", "--out", str(tmp_path / "c.npz")])


def _bins_file(tmp_path):
    import scipy.sparse as sp
    rng = np.random.default_rng(3)
    X = sp.random(400, 3, density=0.5, format="csr", random_state=3)
    X.data = np.ceil(X.data * 4)
    path = tmp_path / "bins.npz"
    np.savez(path, data=X.data.astype(np.float32), indices=X.indices, indptr=X.indptr, shape=np.array(X.shape),
             coords_um=rng.uniform(0, 200, (400, 2)).astype(np.float32), features=np.array(["a", "b", "c"]),
             bin_um=2.0)
    return path


def test_python_m_scote_help():
    out = subprocess.run([sys.executable, "-m", "scote", "--help"], capture_output=True, text=True, check=True)
    assert "twopoint" in out.stdout and "convert" in out.stdout


def test_convert_xenium(tmp_path):
    rng = np.random.default_rng(0)
    n = 1_000
    raw = tmp_path / "raw" / "outs"
    raw.mkdir(parents=True)
    feats = rng.choice(["GENE1", "GENE2", "NegControlProbe_1", "BLANK_0001"], n, p=[0.5, 0.3, 0.1, 0.1])
    cell = rng.choice(["UNASSIGNED", "aaaaaaab-1", "aaaaaaac-1"], n)
    qv = rng.uniform(10, 40, n)
    pq.write_table(pa.table(dict(x_location=rng.uniform(0, 100, n), y_location=rng.uniform(0, 100, n),
                                 z_location=rng.uniform(0, 10, n), feature_name=feats, cell_id=cell,
                                 overlaps_nucleus=rng.integers(0, 2, n), qv=qv)),
                   raw / "transcripts.parquet")
    qc_path = tmp_path / "qc.json"
    assert cli.main(["convert", "--platform", "xenium", "--rawdir", str(tmp_path / "raw"), "--outdir",
                     str(tmp_path / "out"), "--sample", "S", "--qc", str(qc_path)]) == 0
    qc = json.load(open(qc_path))
    t = pq.read_table(tmp_path / "out" / "molecules.parquet").to_pandas()
    keep = np.isin(feats, ["GENE1", "GENE2"]) & (qv >= 20)
    assert qc["molecules"] == len(t) == keep.sum() and set(t.feature.astype(str)) == {"GENE1", "GENE2"}
    assert np.array_equal(t.cell_id.to_numpy() == -1, cell[keep] == "UNASSIGNED")
    assert set(t.cell_id[t.cell_id >= 0]) == {1, 2}
    assert set(t.columns) >= {"x", "y", "z", "feature", "sample", "cell_id", "nucleus", "qv"}
