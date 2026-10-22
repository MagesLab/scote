#!/usr/bin/env python
"""Preprint Fig. 6 comparator: KD-tree cross-K / pair-correlation reference implementation (the estimator behind
spatstat Kcross/pcfcross and pointpats cross-K without edge correction): for every unordered gene pair a <= b,
scipy cKDTree(a).count_neighbors(cKDTree(b), r=edges[1:], cumulative=True) on the exact molecule coordinates; shell
counts = differences over the same 49 edges (0 .. 16,384 um) as the SCOTE hierarchy; self pairs removed for a == b.
Cost ~ O(G^2 x per-pair dual-tree traversal). Parallel over pairs with a fork process pool (--workers).
--budget-s: stop after this many seconds of pair work and extrapolate linearly in the number of pairs (pairs are
processed in random order, so the done fraction is representative); flagged partial=True with the pair counts."""
import argparse, os, time
import numpy as np
from common import OUTDIR, load_cache, select, shell_edges, env_info, peak_rss_mb, append_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="xenium_crc_p2"); ap.add_argument("--G", type=int, default=200)
ap.add_argument("--thin-n", type=int, default=0); ap.add_argument("--thin-frac", type=float, default=1.0)
ap.add_argument("--crop", type=float, default=1.0)
ap.add_argument("--workers", type=int, default=1); ap.add_argument("--budget-s", type=float, default=1800)
ap.add_argument("--chunk", type=int, default=0, help="pairs per task (0 = min(64, pairs / (16 workers)))"); ap.add_argument("--sweep", default=""); ap.add_argument("--out", default=str(OUTDIR / "kdtree.jsonl"))
a = ap.parse_args()
from scipy.spatial import cKDTree
import scipy

c = load_cache(a.dataset)
xy, codes, info = select(c, G=a.G, crop=a.crop, thin_frac=a.thin_frac, thin_n=a.thin_n); del c
edges = shell_edges(2.0, 8, 10); R = edges[1:]

t0 = time.perf_counter()
present = np.unique(codes)
TREES = {g: cKDTree(xy[codes == g]) for g in present}
NS = {g: TREES[g].n for g in present}
t_build = time.perf_counter() - t0
pairs = [(i, j) for ii, i in enumerate(present) for j in present[ii:]]
order = np.random.default_rng(0).permutation(len(pairs))
pairs = [pairs[k] for k in order]


def work(chunk):
    out = []
    for i, j in chunk:
        cum = TREES[i].count_neighbors(TREES[j], R, cumulative=True).astype(np.float64)
        if i == j:
            cum -= NS[i]
        out.append((i, j, np.diff(np.concatenate([[0.0], cum]))))
    return out


G = info["G"]; P = np.zeros((len(R), G, G))
CH = a.chunk or max(1, min(64, len(pairs) // (16 * max(1, a.workers))))
chunks = [pairs[k:k + CH] for k in range(0, len(pairs), CH)]
done = 0; partial = False
t1 = time.perf_counter()
if a.workers > 1:
    import multiprocessing as mp
    pool = mp.get_context("fork").Pool(a.workers)
    it = pool.imap_unordered(work, chunks)
else:
    pool = None; it = map(work, chunks)
for res in it:
    for i, j, cnt in res:
        P[:, i, j] = cnt; P[:, j, i] = cnt; done += 1
    if time.perf_counter() - t1 > a.budget_s and done < len(pairs):
        partial = True; break
if pool is not None:
    pool.terminate()
t_pairs = time.perf_counter() - t1
wall_meas = t_build + t_pairs
wall_est = t_build + t_pairs * len(pairs) / max(done, 1)
rec = dict(method="kdtree_crossK", device="cpu", threads=a.workers, dataset=a.dataset, sweep=a.sweep, **info,
           n_genes_present=int(len(present)), n_pairs=len(pairs), n_pairs_done=done, partial=partial,
           n_edges=len(edges), r_max_um=float(edges[-1]),
           settings=dict(impl="scipy.spatial.cKDTree.count_neighbors(cumulative=True), per unordered gene pair",
                         radii="SCOTE shell edges p=2um, shells_per_level 8, n_levels 10", parallel=f"fork Pool over pairs, chunks of {CH}"),
           wall_s=wall_est, wall_measured_s=wall_meas, build_s=t_build, pairs_s=t_pairs,
           peak_rss_MB=peak_rss_mb(), total_pairs=float(P.sum()) if not partial else None,
           scipy_version=scipy.__version__, env=env_info())
append_jsonl(a.out, rec)
