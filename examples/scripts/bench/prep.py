#!/usr/bin/env python
"""Write the benchmark input caches data/bench_scaling/<dataset>.npz from the processed molecule parquets
(x, y float32 um; code int16 = abundance rank; features in rank order; counts) and the cell-level table for
Xenium CRC P2 (centroids + 20 k-means clusters on log1p-normalized counts, for the TACCO/Squidpy cell-level points).
Usage: prep.py dataset [dataset ...] [--cells xenium_crc_p2 [--cells-out PATH]]"""
import argparse, json, time
import numpy as np, pandas as pd
from common import REPO, CACHE

ap = argparse.ArgumentParser(); ap.add_argument("datasets", nargs="*"); ap.add_argument("--cells", default=None)
ap.add_argument("--cells-out", default=None, help="cell table path (default data/bench_scaling/<cells>_cells.parquet)")
a = ap.parse_args()
CACHE.mkdir(parents=True, exist_ok=True)
import pyarrow.parquet as pq
for ds in a.datasets:
    out = CACHE / f"{ds}.npz"
    if out.exists():
        print("exists", out); continue
    t0 = time.time()
    tbl = pq.read_table(REPO / f"data/processed/{ds}/molecules.parquet", columns=["x", "y", "feature"])
    feat = tbl["feature"].to_pandas().astype("category")
    x = tbl["x"].to_numpy().astype(np.float32); y = tbl["y"].to_numpy().astype(np.float32); del tbl
    vc = feat.value_counts()                     # descending
    rank = pd.Series(np.arange(len(vc)), index=vc.index)
    code = rank.reindex(feat.cat.categories).to_numpy()[feat.cat.codes.to_numpy()].astype(np.int16)
    np.savez(out, x=x, y=y, code=code, features=np.array(vc.index.astype(str), dtype=object), counts=vc.to_numpy())
    print(f"{ds}: {len(code):,} molecules, {len(vc)} genes, {time.time()-t0:.0f}s", flush=True)
if a.cells:
    import h5py, scipy.sparse as sp
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import MiniBatchKMeans
    raw = REPO / f"data/raw/{a.cells}"
    with h5py.File(raw / "cell_feature_matrix.h5") as f:
        m = f["matrix"]; ftype = m["features/feature_type"][:].astype(str)
        X = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:])).T.tocsr()
        bc = m["barcodes"][:].astype(str)
    X = X[:, ftype == "Gene Expression"].astype(np.float64)
    cells = pq.read_table(raw / "cells.parquet", columns=["cell_id", "x_centroid", "y_centroid"]).to_pandas().set_index("cell_id").loc[bc]
    tot = np.asarray(X.sum(1)).ravel(); keep = tot >= 10
    Xn = sp.diags(1e2 / np.maximum(tot, 1)) @ X; Xn.data = np.log1p(Xn.data)
    Z = TruncatedSVD(30, random_state=0).fit_transform(Xn[keep])
    lab = np.full(len(tot), -1); lab[keep] = MiniBatchKMeans(20, random_state=0, n_init=3).fit_predict(Z)
    df = pd.DataFrame(dict(x=cells["x_centroid"].to_numpy(), y=cells["y_centroid"].to_numpy(), cluster=lab))
    df = df[df.cluster >= 0]
    df.to_parquet(a.cells_out or CACHE / f"{a.cells}_cells.parquet")
    print(f"cells: {len(df):,} cells with >= 10 counts, 20 k-means clusters on 30 SVD comps")
