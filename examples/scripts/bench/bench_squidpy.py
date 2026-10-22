#!/usr/bin/env python
"""Preprint Fig. 6 comparator: Squidpy co_occurrence at molecule level, gene as the cluster key, on the identical inputs
(common.select) and the same 49 radii as the SCOTE hierarchy (interval = shell edges 0 .. 16,384 um -> 48 cumulative
thresholds edges[1:]). Squidpy 1.8.3: `n_splits`/`n_jobs` are deprecated and ignored; the numba kernel
`_occur_count` loops over all N^2 ordered point pairs x all thresholds (parallel over points, numba threads) and
allocates an int32 buffer of N x L x k^2 (k = clusters present) -> memory 4 N L k^2 bytes. Runs whose predicted
buffer exceeds --mem-gb are skipped and recorded as such (predicted memory). Warm-up call (JIT) on 500 points first."""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from common import OUTDIR, load_cache, select, shell_edges, env_info, peak_rss_mb, append_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="xenium_crc_p2"); ap.add_argument("--G", type=int, default=200)
ap.add_argument("--thin-n", type=int, default=0); ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--mem-gb", type=float, default=200); ap.add_argument("--cells", default=None, help="cell table parquet (x, y, cluster)")
ap.add_argument("--interval", default="shells", help="'shells' (SCOTE shell edges) or 'cell300' (31 edges 0..300 um)")
ap.add_argument("--sweep", default=""); ap.add_argument("--out", default=str(OUTDIR / "squidpy.jsonl"))
a = ap.parse_args()
import numba
numba.set_num_threads(a.threads)
import anndata as ad, squidpy as sq

if a.cells:
    df = pd.read_parquet(a.cells)
    if a.thin_n and a.thin_n < len(df):
        df = df.iloc[np.sort(np.random.default_rng(0).choice(len(df), a.thin_n, replace=False))]
    xy = df[["x", "y"]].to_numpy(np.float64); lab = df["cluster"].astype(str).to_numpy()
    info = dict(G=int(df["cluster"].nunique()), n_molecules=0, n_cells=int(len(df)), level="cell")
    dataset = Path(a.cells).stem
else:
    c = load_cache(a.dataset)
    xy, codes, info = select(c, G=a.G, thin_n=a.thin_n); del c
    lab = codes.astype(str); info["level"] = "molecule"; dataset = a.dataset
edges = shell_edges(2.0, 8, 10) if a.interval == "shells" else np.linspace(0, 300, 31)
N = len(xy); k = len(np.unique(lab)); L = len(edges) - 1
pred_GB = 4.0 * N * L * k * k / 2**30
base = dict(method=("squidpy_co_occurrence" + ("_cells" if a.cells else "")), device="cpu", threads=a.threads,
            dataset=dataset, sweep=a.sweep, **info, k_present=int(k), n_points=int(N), n_thresholds=int(L),
            interval=a.interval, r_max_um=float(edges[-1]), predicted_buffer_GB=pred_GB,
            settings=dict(squidpy=sq.__version__, call="sq.gr.co_occurrence(adata, cluster_key, interval=edges, copy=True)",
                          n_splits="deprecated/ignored in 1.8.3", numba_threads=a.threads), env=env_info())
if pred_GB > a.mem_gb:
    append_jsonl(a.out, dict(base, skipped=True, reason=f"predicted N*L*k^2*4 B buffer {pred_GB:.0f} GB > {a.mem_gb:.0f} GB"))
    sys.exit(0)


def mk(xy, lab):
    A = ad.AnnData(obs=pd.DataFrame(dict(cl=pd.Categorical(lab))))
    A.obs["cl"] = A.obs["cl"].cat.remove_unused_categories()
    A.obsm["spatial"] = xy
    return A


w = mk(np.random.default_rng(1).uniform(0, 100, (500, 2)), np.repeat(np.array(["a", "b"]), 250))
tw = time.perf_counter(); sq.gr.co_occurrence(w, "cl", interval=edges, copy=True); jit_s = time.perf_counter() - tw
A = mk(xy, lab)
t0 = time.perf_counter()
occ, iv = sq.gr.co_occurrence(A, "cl", interval=edges, copy=True)
wall = time.perf_counter() - t0
append_jsonl(a.out, dict(base, skipped=False, wall_s=wall, jit_s=jit_s, peak_rss_MB=peak_rss_mb(), occ_shape=list(occ.shape)))
