#!/usr/bin/env python
"""Preprint Fig. 6 comparator: cell-level co-occurrence with TACCO (tacco.tools.co_occurrence), the "what people do today"
cell-level point, on the Xenium CRC P2 segmented cells (centroids; 20 k-means clusters from prep.py).
Settings: annotation = cluster, max_distance 300 um, delta_distance 10 um, min_distance 0, reads=False (unweighted),
sparse distance matrix (tacco default for finite max_distance), n_permutation 0. --thin-n subsamples cells."""
import argparse, os, time
import numpy as np, pandas as pd
from common import OUTDIR, CACHE, env_info, peak_rss_mb, append_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--cells", default=str(CACHE / "xenium_crc_p2_cells.parquet")); ap.add_argument("--thin-n", type=int, default=0)
ap.add_argument("--threads", type=int, default=1); ap.add_argument("--max-distance", type=float, default=300)
ap.add_argument("--delta", type=float, default=10); ap.add_argument("--sweep", default="cells")
ap.add_argument("--out", default=str(OUTDIR / "tacco.jsonl"))
a = ap.parse_args()
import anndata as ad, tacco as tc
from importlib.metadata import version
df = pd.read_parquet(a.cells)
if a.thin_n and a.thin_n < len(df):
    df = df.iloc[np.sort(np.random.default_rng(0).choice(len(df), a.thin_n, replace=False))]
A = ad.AnnData(obs=pd.DataFrame(dict(x=df["x"].to_numpy(), y=df["y"].to_numpy(), cluster=pd.Categorical(df["cluster"].astype(str)))))
A.obs.index = A.obs.index.astype(str)
t0 = time.perf_counter()
res = tc.tools.co_occurrence(A, "cluster", position_key=("x", "y"), max_distance=a.max_distance, delta_distance=a.delta,
                             min_distance=0, reads=False, n_permutation=0, verbose=0)
wall = time.perf_counter() - t0
append_jsonl(a.out, dict(method="tacco_co_occurrence_cells", device="cpu", threads=a.threads, dataset="xenium_crc_p2_cells",
                         sweep=a.sweep, level="cell", n_cells=int(len(df)), n_molecules=0, G=int(df["cluster"].nunique()),
                         r_max_um=a.max_distance, wall_s=wall, peak_rss_MB=peak_rss_mb(),
                         settings=dict(tacco=version("tacco"), max_distance=a.max_distance, delta_distance=a.delta, min_distance=0,
                                       reads=False, n_permutation=0, sparse="default (True for finite max_distance)"),
                         occ_shape=list(np.shape(res["occ"])), env=env_info()))
