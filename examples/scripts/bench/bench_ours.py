#!/usr/bin/env python
"""Preprint Fig. 6: time the SCOTE shell-product estimator (global ordered pair counts P_k[a, b] for all shells of the
hierarchy) on a real-data input selected by common.select(); one run per process (clean peak RSS).

Pipeline timed ("compute" wall): molecules (x, y, gene code) in memory -> scote.pair_counts(shells_per_level=8,
n_levels=10) on the 2 um lattice (the call builds the one-hot rows, molecules x G) -> P (49 edges, 48 shells, 0 ..
16.4 mm).
  --device cpu : backend 'cpu', CPU mode 'auto' (numba kernel at sparse levels, BLAS GEMM at dense levels), --threads
                 passed to the call (numba) and to threadpoolctl (BLAS), with the *_NUM_THREADS environment variables set
                 by the workflow. scote.CPUOptions(numba_acc_bytes=64 GiB) so that large G keeps the numba kernel
                 instead of falling back to SciPy (documented in the record).
  --device cuda: backend 'gpu' (exact fp64 shell products on the GPU, shell operators built on the device).
A warm-up on 100,000 molecules (numba JIT, including the lattice kernels / CUDA context) runs first and is reported
separately (jit_s).
--hd: Visium HD bins (data/processed/hd_crc_p2/bins_square_002um.npz) instead of molecules; G = top-G genes by UMI.
--molecules: a molecule parquet (x, y, feature) instead of the bench cache; genes top-G by count (or --genes).
--genes FILE: genes by name (one per line, '#' comments) instead of the top-G (molecule inputs).

Uncertainty and output (the recorded wall time `wall_s` includes the output write; compute_s and write_s split it):
  --uncertainty none   : the global counts path above. Without --write the (K, G, G) counts stay in host memory
                         (stored = True); with --write DIR every finished shell slab is copied to the host and written
                         into the memory map DIR/P.npy (float64, (K, G, G)) as it completes (write_s = device-to-host
                         copy + memory-map store + final flush; stored = False: no host copy of the counts is kept).
  --uncertainty closed : the per-tile path with the closed-form tile block bootstrap (scote.twopoint on the same
                         lattice, streamed to DIR; tiles of --tile-um, default 500 um): slabs per tile
                         stream into four G x G accumulators, every finished shell is turned into log g, sd, n_eff_pairs,
                         T_eff and its flag and written with P to DIR (memory maps, float32 except P float64 and flag
                         int8; in memory when --write is not given). GPU: lattice products and accumulators on the
                         device (fp64); CPU: CPU mode 'auto' with the numba accumulator kernel.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, scipy.sparse as sp
import common
from common import REPO, OUTDIR, load_cache, select, env_info, peak_rss_mb, append_jsonl
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="xenium_crc_p2"); ap.add_argument("--datasets", default="", help="comma list: concatenate these sections side by side (x offset 40 mm, beyond r_max -> no cross-section pairs) into ONE input; genes = top-G by summed counts"); ap.add_argument("--hd", default=None, help="bins npz (Visium HD)")
ap.add_argument("--G", type=int, default=200); ap.add_argument("--crop", type=float, default=1.0)
ap.add_argument("--thin-frac", type=float, default=1.0); ap.add_argument("--thin-n", type=int, default=0)
ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--pitch", type=float, default=2.0); ap.add_argument("--n-levels", type=int, default=10)
ap.add_argument("--sweep", default="")
ap.add_argument("--uncertainty", default="none", choices=["none", "closed"],
                help="none: global counts (no intervals); closed: per-tile closed-form tile bootstrap (see docstring)")
ap.add_argument("--write", default="", help="output directory (P.npy for none; log_g/sd/n_eff_pairs/P/t_eff/flag .npy for closed)")
ap.add_argument("--tile-um", type=float, default=500.0, help="closed: tile side (um)")
ap.add_argument("--acc-dtype", default="float64", choices=["float64", "float32"], help="closed: accumulator dtype of A1, A2, Av")
ap.add_argument("--genes", default="", help="file with gene names (one per line) instead of the top-G genes")
ap.add_argument("--molecules", default="", help="molecule parquet (x, y, feature) instead of the bench cache")
ap.add_argument("--cache-dir", default="", help="directory of the bench caches (default: <repo>/data/bench_scaling)")
ap.add_argument("--out", default=str(OUTDIR / "ours.jsonl"))
ap.add_argument("--save-counts", default="", help="write the (n_shells, G, G) counts to this .npy")
ap.add_argument("--check-counts", default="", help="assert bit-identical counts against this .npy (exit 3 if not)")
ap.add_argument("--profile", default="", help="run the compute step under cProfile and write pstats text here")
a = ap.parse_args()

from threadpoolctl import threadpool_limits
import scote
if a.device == "cpu":
    threadpool_limits(a.threads)
else:
    import torch
    torch.set_num_threads(a.threads)
if a.cache_dir:
    common.CACHE = Path(a.cache_dir)
closed = a.uncertainty == "closed"
RUN = dict(shells_per_level=8,
           backend=scote.CPUOptions(numba_acc_bytes=64 * 2**30) if a.device == "cpu" else scote.GPUOptions(device="cuda"),
           threads=a.threads, verbose=False)


def read_genes(path):
    return [l.strip() for l in open(path) if l.strip() and not l.lstrip().startswith("#")]


def restrict_genes(c, names):
    """Cache dict restricted to the genes `names` (in the given order; genes absent from the input are dropped),
    codes renumbered 0..G'-1 so that select(c, G=G') keeps all of them."""
    feats = list(c["features"]); idx = {f: i for i, f in enumerate(feats)}
    want = [n for n in names if n in idx]
    lut = np.full(len(feats), -1, dtype=np.int64); lut[[idx[n] for n in want]] = np.arange(len(want))
    new = lut[c["code"].astype(np.int64)]; m = new >= 0
    return dict(x=c["x"][m], y=c["y"][m], code=new[m], features=want), len(names) - len(want)


def cache_from_parquet(path):
    """Bench-cache dict (codes = abundance rank, ties by name order of the dictionary) from a molecule parquet; the
    feature column is read dictionary-encoded (no per-molecule strings)."""
    import pyarrow as pa, pyarrow.parquet as pq
    tbl = pq.read_table(path, columns=["x", "y", "feature"], read_dictionary=["feature"])
    col = tbl["feature"]
    col = col.unify_dictionaries() if isinstance(col, pa.ChunkedArray) else col
    names = [str(v) for v in col.chunk(0).dictionary.to_pylist()]
    idx = np.concatenate([np.asarray(ch.indices) for ch in col.chunks]).astype(np.int64)
    n_per = np.bincount(idx, minlength=len(names))
    order = np.argsort(-n_per, kind="stable")
    rank = np.empty(len(names), dtype=np.int64); rank[order] = np.arange(len(names))
    return dict(x=tbl["x"].to_numpy().astype(np.float32), y=tbl["y"].to_numpy().astype(np.float32), code=rank[idx],
                features=[names[i] for i in order])


rss_start = peak_rss_mb()
t0 = time.perf_counter()
features = None
if a.hd:
    import pandas as pd
    z = np.load(a.hd, allow_pickle=True)
    X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    coords = z["coords_um"].astype(np.float64)
    if "in_tissue" in z:
        keep = z["in_tissue"].astype(bool); X = X[keep]; coords = coords[keep]
    tot = np.asarray(X.sum(0)).ravel(); order = np.argsort(-tot, kind="stable")
    G = len(order) if a.G <= 0 else min(a.G, len(order))
    X = X[:, order[:G]].tocsr().astype(np.float64)
    pitch = float(z["bin_um"])
    info = dict(G=int(G), crop=1.0, thin_frac=1.0, thin_n=0, n_molecules=int(X.sum()), n_bins_hd=int(X.shape[0]),
                region_area_mm2=float(np.ptp(coords[:, 0]) * np.ptp(coords[:, 1]) / 1e6),
                occupied_area_mm2_50um=X.shape[0] * pitch * pitch / 1e6)
    info["density_per_um2"] = info["n_molecules"] / (info["occupied_area_mm2_50um"] * 1e6)
    dataset = Path(a.hd).parent.name
    features = [str(f) for f in np.asarray(z["features"]).astype(str)[order[:G]]]
elif a.datasets:
    import pandas as pd
    names = a.datasets.split(",")
    cs = [load_cache(n) for n in names]
    tot = pd.concat([pd.Series(c["counts"], index=c["features"]) for c in cs], axis=1).fillna(0).sum(1).sort_values(ascending=False)
    G = min(a.G, len(tot)); keep_genes = pd.Index(tot.index[:G])
    xs, cs_, x0 = [], [], 0.0
    per = []
    for n, c in zip(names, cs):
        rank_to_new = keep_genes.get_indexer(pd.Index(c["features"]))      # -1 if not in the common top-G
        new = rank_to_new[c["code"].astype(np.int64)]
        m = new >= 0
        x = c["x"][m].astype(np.float64); y = c["y"][m].astype(np.float64)
        x = x - x.min() + x0; x0 = x.max() + 40000.0
        xs.append(np.stack([x, y - y.min()], 1)); cs_.append(new[m]); per.append(dict(dataset=n, n_molecules=int(m.sum())))
        c.clear()
    xy = np.concatenate(xs); codes = np.concatenate(cs_); del xs, cs_, cs
    features = [str(f) for f in keep_genes]
    info = dict(G=int(G), crop=1.0, thin_frac=1.0, thin_n=0, n_molecules=int(len(codes)), sections=per,
                region_area_mm2=None, occupied_area_mm2_50um=None, density_per_um2=None)
    dataset = "+".join(names)
else:
    c = cache_from_parquet(a.molecules) if a.molecules else load_cache(a.dataset)
    G_sel, genes_missing = a.G, 0
    if a.genes:
        c, genes_missing = restrict_genes(c, read_genes(a.genes)); G_sel = len(c["features"])
    xy, codes, info = select(c, G=G_sel, crop=a.crop, thin_frac=a.thin_frac, thin_n=a.thin_n)
    G = info["G"]; features = [str(f) for f in list(c["features"])[:G]]; del c
    dataset = Path(a.molecules).parent.name if a.molecules else a.dataset
    if a.genes:
        info.update(gene_list=str(a.genes), genes_missing=genes_missing)
t_load = time.perf_counter() - t0
rss_loaded = peak_rss_mb()

write_dir = Path(a.write) if a.write else None
if write_dir is not None:
    write_dir.mkdir(parents=True, exist_ok=True)


# warm-up (JIT compile / CUDA context), excluded from the compute wall
tw = time.perf_counter()
rng = np.random.default_rng(1)
# 100,000 molecules: enough for the numba lattice constructor, which smaller inputs do not use
wxy = rng.uniform(0, 200, (100_000, 2)); wc = rng.integers(0, min(G, 20), 100_000); wnames = list(range(min(G, 20)))
if closed:   # per-tile path (row groups) and the accumulator kernels
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scote.twopoint(wxy, wc, names=wnames, lattice_only=True, delta=2.0, tile_um=50.0, n_levels=2,
                       acc_dtype=a.acc_dtype, **RUN)
else:
    scote.pair_counts(wxy, wc, names=wnames, delta=2.0, n_levels=2, **RUN)
jit_s = time.perf_counter() - tw
if a.device != "cpu":
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()

t1 = time.perf_counter()
if a.profile:   # the profile also covers the molecule -> lattice binning below
    import cProfile, pstats, io
    prof = cProfile.Profile(); prof.enable()
if a.hd:
    origin = coords.min(0); F0, kind = X, dict(bin_size=pitch)
else:
    pitch = a.pitch
    coords, origin = xy, np.zeros(2)
    F0, kind = codes, {}        # one label per molecule; the call builds the one-hot rows, cells floor(xy / pitch)
t_bin = time.perf_counter() - t1          # host binning before the call (the one-hot rows are built inside the call)
K = 8 + 4 * a.n_levels
extra = dict(uncertainty=a.uncertainty, write_dir=str(write_dir) if write_dir else None)
if closed:
    # cell centers of the same lattice with origin 0: the per-tile path re-derives exactly the cells of the global path,
    # so every level is coarse-grained as there and P is identical (tiles of --tile-um on the same origin)
    coords_c = (np.floor((coords - origin) / pitch) + 0.5) * pitch
    kind_c = kind if a.hd else dict(lattice_only=True)
    res = scote.twopoint(coords_c, F0, names=features, delta=pitch, n_levels=a.n_levels, tile_um=a.tile_um,
                         origin=np.zeros(2), out=write_dir, dtype="float32", acc_dtype=a.acc_dtype, **kind_c, **RUN)
    if a.device != "cpu":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t1
    tm = res.timings
    write_s = tm["write"]
    intervals = res.bins.to_dict("records")
    timings = tm["levels"]
    counts = None
    total_pairs = float(np.sum(res.P_tot))
    extra.update(n_tiles=res.n_tiles, tile_um=a.tile_um, acc_dtype=a.acc_dtype, symmetrize=True,
                 accumulate_s=tm["accumulate"], lattice_s=tm["lattice"], n_slab_updates=int(tm["n_updates"]),
                 flag_fractions=[float((np.asarray(res.flag) == f).mean()) for f in (0, 1, 2)])
    if write_dir is None:
        extra["output_bytes_in_memory"] = int(sum(np.asarray(getattr(res, k)).nbytes for k in ("log_g", "sd", "n_eff_pairs", "P", "t_eff", "flag")))
else:
    sink = None
    run = RUN
    write_s = 0.0
    if write_dir is not None:
        from numpy.lib.format import open_memmap
        tw0 = time.perf_counter()
        P_mm = open_memmap(write_dir / "P.npy", mode="w+", dtype=np.float64, shape=(K, G, G))
        write_s += time.perf_counter() - tw0
        wt = dict(s=0.0, d2h=0.0)

        if a.device != "cpu":   # device tensors to the sink, so that the device-to-host copy is timed as write
            run = dict(run, backend=scote.GPUOptions(device="cuda", device_slabs=True))

        def sink(k, slab):
            if a.device != "cpu":
                torch.cuda.synchronize()            # the shell's device work counts as compute, not as write
            ta = time.perf_counter()
            host = slab.cpu().numpy() if hasattr(slab, "cpu") else slab
            tb = time.perf_counter()
            P_mm[k] = host
            tc = time.perf_counter()
            wt["d2h"] += tb - ta; wt["s"] += tc - ta
    pc = scote.pair_counts(coords, F0, names=features, delta=pitch, n_levels=a.n_levels, origin=origin, sink=sink,
                           **kind, **run)
    counts, timings = pc.counts, pc.timings["levels"]
    intervals = pc.bins.to_dict("records")
    if write_dir is not None:
        tf = time.perf_counter(); P_mm.flush(); del P_mm
        write_s += wt["s"] + time.perf_counter() - tf
        extra.update(d2h_s=wt["d2h"])
    if a.device != "cpu":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t1
    total_pairs = float(counts.sum())
    if write_dir is not None:
        counts = np.load(write_dir / "P.npy", mmap_mode="r")    # for --save-counts / --check-counts (outside wall)
extra.update(compute_s=wall - write_s, write_s=write_s)
if write_dir is not None:
    extra["output_bytes"] = int(sum(f.stat().st_size for f in write_dir.iterdir() if f.is_file()))
if a.profile:
    prof.disable()
    buf = io.StringIO()
    for key in ("cumulative", "tottime"):
        buf.write(f"\n===== sorted by {key} (wall {wall:.1f} s, threads {a.threads}) =====\n")
        pstats.Stats(prof, stream=buf).sort_stats(key).print_stats(35)
    Path(a.profile).write_text(buf.getvalue())
if a.save_counts:
    np.save(a.save_counts, counts if counts is not None else np.asarray(res.P))
counts_identical = None
if a.check_counts:
    ref = np.load(a.check_counts)
    got = counts if counts is not None else np.asarray(res.P)
    counts_identical = bool(ref.shape == got.shape and np.array_equal(ref, got))
    print(f"counts bit-identical to {a.check_counts}: {counts_identical}", flush=True)

rec = dict(method="scote", device=("cpu" if a.device == "cpu" else "gpu"), threads=a.threads, dataset=dataset,
           sweep=a.sweep, **info, pitch_um=pitch, n_levels=a.n_levels, n_shells=len(intervals),
           r_max_um=float(intervals[-1]["r_to"]), stored=(not closed and write_dir is None),
           settings=(dict(backend="cpu", mode="auto", numba_acc_limit_GB=64) if a.device == "cpu" else
                     dict(backend="gpu", dtype="float64", dense_gb=4, shell_operator="device")),
           counts_identical_to_reference=counts_identical,
           n_lattice_cells_level0=int(timings[0]["n_cells"]), nnz_X_level0=int(timings[0]["nnz_X"]),
           wall_s=wall, load_s=t_load, jit_s=jit_s, bin_s=t_bin,
           level_s=[round(t["seconds_total"], 3) for t in timings], level_modes=[t["mode"] for t in timings],
           level_S_SX_XtSX_s=[[round(t["seconds_S"], 3), round(t["seconds_SX"], 3), round(t["seconds_XtSX"], 3)] for t in timings],
           peak_rss_MB=peak_rss_mb(), rss_after_load_MB=rss_loaded, rss_start_MB=rss_start,
           total_pairs=total_pairs, env=env_info(), **extra)
if a.device != "cpu":
    rec.update(gpu=torch.cuda.get_device_name(0), gpu_total_MB=torch.cuda.get_device_properties(0).total_memory / 2**20,
               torch=torch.__version__, cuda=torch.version.cuda,
               gpu_peak_alloc_MB=torch.cuda.max_memory_allocated() / 2**20,
               gpu_peak_reserved_MB=torch.cuda.max_memory_reserved() / 2**20)
append_jsonl(a.out, rec)
if counts_identical is False:
    sys.exit(3)
