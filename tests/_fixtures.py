"""Tiny synthetic vendor outputs for the reader and converter tests."""
import gzip
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def xenium(root: Path, n: int = 1_000, seed: int = 0) -> Path:
    """``outs/transcripts.parquet`` of a Xenium run; returns the run directory."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng = np.random.default_rng(seed)
    raw = root / "outs"
    raw.mkdir(parents=True, exist_ok=True)
    feats = rng.choice(["GENE1", "GENE2", "NegControlProbe_1", "BLANK_0001"], n, p=[0.5, 0.3, 0.1, 0.1])
    pq.write_table(pa.table(dict(x_location=rng.uniform(0, 100, n), y_location=rng.uniform(0, 100, n),
                                 z_location=rng.uniform(0, 10, n), feature_name=feats,
                                 cell_id=rng.choice(["UNASSIGNED", "aaaaaaab-1", "aaaaaaac-1"], n),
                                 overlaps_nucleus=rng.integers(0, 2, n), qv=rng.uniform(10, 40, n))),
                   raw / "transcripts.parquet", row_group_size=300)
    return root


def cosmx(root: Path, n: int = 3_000, seed: int = 0) -> Path:
    """An AtoMx transcript file (two gzip members); returns its path."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    targets = np.array(["GENEA", "GENEB", "Negative1", "SystemControl2", "FalseCode3"])
    tx = pd.DataFrame({"fov": rng.integers(1, 5, n), "cell_ID": rng.integers(0, 20, n), "cell": "c",
                       "x_local_px": rng.integers(0, 4000, n), "y_local_px": rng.integers(0, 4000, n),
                       "x_global_px": rng.uniform(0, 9000, n), "y_global_px": rng.uniform(0, 9000, n),
                       "z": rng.integers(0, 8, n), "target": rng.choice(targets, n, p=[0.5, 0.3, 0.1, 0.05, 0.05]),
                       "CellComp": rng.choice(["Nuclear", "Cytoplasm", "Membrane", "None"], n)})
    path = root / "run_tx_file.csv.gz"
    with gzip.open(path, "wt") as f:
        tx.iloc[: n // 2].to_csv(f, index=False)
    with gzip.open(path, "at") as f:
        tx.iloc[n // 2:].to_csv(f, index=False, header=False)
    return path


def atera(root: Path, seed: int = 0) -> Path:
    """A zarr-only 10x bundle: ``transcripts.zarr.zip`` (two level-0 grid cells), ``cells.zarr.zip`` (label masks),
    ``cell_feature_matrix.zarr.zip``, and ``experiment.xenium``; returns the bundle directory."""
    import zarr
    from zarr.storage import ZipStore
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    genes = ["GENE1", "GENE2", "NegControlProbe_1", "GENE3"]
    with ZipStore(str(root / "transcripts.zarr.zip"), mode="w") as store:
        g = zarr.open_group(store, mode="w", zarr_format=2)
        g.attrs.update(gene_names=genes, number_rnas=600)
        cat = g.create_array("gene_category", shape=(4, 2), dtype="u1")
        cat[:] = np.array([[1, 0], [1, 0], [0, 1], [1, 0]], np.uint8)
        cat.attrs["column_names"] = ["gene", "negative_control_probe"]
        grids = g.create_group("grids")
        grids.attrs.update(grid_keys=[["0,0", "0,1", "1,0"]], grid_number_objects=[[350, 250, 0]], grid_size=[250.0],
                           number_levels=1)
        for key, n in (("0,0", 350), ("0,1", 250)):
            cell = grids.create_group(f"0/{key}")
            loc = np.c_[rng.uniform(0, 40, n), rng.uniform(0, 30, n), rng.uniform(0, 8, n)].astype(np.float32)
            for name, arr in (("location", loc), ("gene_identity", rng.integers(0, 4, (n, 1)).astype(np.uint16)),
                              ("quality_score", rng.uniform(10, 40, (n, 1)).astype(np.float32)),
                              ("valid", rng.integers(0, 2, (n, 1)).astype(np.uint8)),
                              ("status", np.zeros((n, 1), np.uint8)), ("id", np.arange(2 * n).reshape(n, 2)),
                              ("codeword_identity", rng.integers(0, 9, (n, 2)).astype(np.uint16))):
                a = cell.create_array(name, shape=arr.shape, dtype=arr.dtype, chunks=(128,) + arr.shape[1:])
                a[:] = arr
    labels = np.zeros((60, 80), np.uint32)                       # pixel = 2 x um
    labels[0:30, 0:40], labels[30:60, 0:40], labels[0:60, 40:80] = 1, 2, 3
    nuclei = np.zeros_like(labels)
    nuclei[5:20, 5:30], nuclei[35:50, 5:30] = 1, 2
    with ZipStore(str(root / "cells.zarr.zip"), mode="w") as store:
        g = zarr.open_group(store, mode="w", zarr_format=2)
        for name, arr in (("masks/0", nuclei), ("masks/1", labels),
                          ("masks/homogeneous_transform", np.array([[2.0, 0, 0, 0], [0, 2.0, 0, 0], [0, 0, 1, 0],
                                                                    [0, 0, 0, 1]], np.float32)),
                          ("cell_id", np.array([[11, 1], [12, 1], [13, 1]], np.uint32)),
                          ("cell_summary", np.array([[10.0, 7.0], [10.0, 22.0], [30.0, 15.0]]))):
            a = g.create_array(name, shape=arr.shape, dtype=arr.dtype)
            a[:] = arr
        g["cell_summary"].attrs["column_names"] = ["cell_centroid_x", "cell_centroid_y"]
    with ZipStore(str(root / "cell_feature_matrix.zarr.zip"), mode="w") as store:
        g = zarr.open_group(store, mode="w", zarr_format=2)
        cf = g.create_group("cell_features")
        cf.attrs.update(feature_types=["gene", "gene", "negative_control_probe", "gene"],
                        feature_ids=genes, feature_keys=genes)
        for name, arr in (("csc/indptr", np.array([0, 2, 3, 5])), ("csc/data", np.array([3, 1, 2, 4, 1], np.uint32)),
                          ("csc/indices", np.array([0, 2, 1, 0, 3])),
                          ("cell_id", np.array([[11, 1], [12, 1], [13, 1]], np.uint32))):
            a = cf.create_array(name, shape=arr.shape, dtype=arr.dtype)
            a[:] = arr
    (root / "experiment.xenium").write_text(json.dumps(dict(num_cells=3, panel_name="test", pixel_size=0.2125)))
    return root


def aviti24(path: Path, seed: int = 0) -> Path:
    """A SpatialData run archive with wells A1 and B1, a targeted batch B01 (no IsCounted) and a 3' batch B02 in two
    parts; returns the archive path."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = "run.zarr"
    meta = dict(run_parameters=dict(ImageInfo=dict(PixelSizeUm=0.48), ZPositions=[2, 1, 0, -1, -2, -3, -4]),
                well_metadata=[dict(well_name="A1", well_label="ctrl"), dict(well_name="B1", well_label="drug")])
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(f"{root}/zarr.json", json.dumps(dict(zarr_format=3, node_type="group",
                                                         attributes=dict(run_metadata=json.dumps(meta)))))
        for well in ("A1", "B1"):
            for batch, parts in (("B01", 1), ("B02", 2)):
                element = f"{well}-points-{batch}"
                zf.writestr(f"{root}/points/{element}/zarr.json", json.dumps(dict(zarr_format=3, node_type="group")))
                for p in range(parts):
                    n = 200
                    cols = dict(x=rng.uniform(0, 2000, n), y=rng.uniform(0, 2000, n), z=rng.integers(0, 7, n),
                                target_name=rng.choice(["Unassigned", "GAPDH", "ACTB"], n),
                                label_Cell_global=rng.integers(0, 5, n), is_nuclear=rng.integers(0, 2, n).astype(bool),
                                MapQ=pa.array(np.where(rng.random(n) < 0.1, None, rng.uniform(0, 60, n)).tolist(),
                                              type=pa.float64()),
                                tile_id=rng.choice(["L1R01C01", "L1R01C02"], n))
                    if batch != "B01":
                        cols["IsCounted"] = pa.array(np.where(cols["target_name"] == "Unassigned", False,
                                                              rng.random(n) < 0.8).tolist())
                        cols["Extra"] = np.zeros(n)
                    sink = pa.BufferOutputStream()
                    pq.write_table(pa.table(cols), sink)
                    member = f"{root}/points/{element}/points.parquet/part.{p}.parquet"
                    zf.writestr(member, sink.getvalue().to_pybytes())
    return path


def gem(path: Path, n: int = 2_000, seed: int = 0) -> Path:
    """A gzip-compressed Stereo-seq GEM file with comment header lines; returns its path."""
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"geneID": rng.choice(["ENSG1", "ENSG2", "ENSG3"], n), "x": rng.integers(1000, 1300, n),
                       "y": rng.integers(2000, 2200, n), "MIDCount": rng.integers(1, 4, n),
                       "ExonCount": rng.integers(0, 2, n)})
    df["geneName"] = df["geneID"].map({"ENSG1": "Actb", "ENSG2": "Gapdh", "ENSG3": "Mt1"})
    with gzip.open(path, "wt") as f:
        f.write("#FileFormat=GEMv0.1\n#SortedBy=None\n#BinSize=1\n")
        df[["geneID", "geneName", "x", "y", "MIDCount", "ExonCount"]].to_csv(f, sep="\t", index=False)
    return path


def gef(path: Path, gem_path: Path) -> Path:
    """A square-bin Stereo-seq GEF file (HDF5, the layout of the SAW file-format manual) with the counts of a GEM
    file: ``geneExp/bin1/expression`` (x, y, count; rows grouped by gene, attribute ``resolution`` in nm) and
    ``geneExp/bin1/gene`` (gene, offset, count), plus a ``bin50`` group that the reader does not use; returns its
    path."""
    import h5py
    df = pd.read_csv(gem_path, sep="\t", comment="#")
    df = df.groupby(["geneName", "x", "y"], as_index=False)["MIDCount"].sum()
    names = ["Mt1", "Actb", "Gapdh"]                                  # gene table order differs from GEM order
    expr, genes, offset = [], [], 0
    for g in names:
        rows = df[df.geneName == g]
        expr.append(rows)
        genes.append((g.encode(), offset, len(rows)))
        offset += len(rows)
    expr = pd.concat(expr)
    e = np.zeros(len(expr), dtype=[("x", "<i4"), ("y", "<i4"), ("count", "u1")])
    e["x"], e["y"], e["count"] = expr.x, expr.y, expr.MIDCount
    gt = np.array(genes, dtype=[("gene", "S32"), ("offset", "<u4"), ("count", "<u4")])
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["version"] = 2
        b1 = f.create_group("geneExp/bin1")
        ds = b1.create_dataset("expression", data=e)
        ds.attrs["resolution"] = np.uint32(500)
        b1.create_dataset("gene", data=gt)
        b50 = f.create_group("geneExp/bin50")                         # coarser group: not read
        b50.create_dataset("expression", data=e[:3])
        b50.create_dataset("gene", data=gt)
    return path
