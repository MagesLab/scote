#!/usr/bin/env python
"""Registered, matched-tile cross-platform two-point functions on the 10x FF ovarian serial sections (Xenium 5K =
reference frame, Xenium v1, Visium HD probe deep / min, 3' Visium HD when it exists). Every section is mapped into the 5K
frame (scripts/registration.py, TPS; transforms from scripts/register_ovff.py), one tile grid with a fixed origin is laid over
all of them, and the comparison runs on the tiles that carry tissue in every compared section. Preprint Supplementary
Fig. S3 uses the `genes`, `twopoint` and `tiles` stages (read by scripts/ovff_depth.py); `compare` compares platforms.

Stages
  genes     gene sets (--genesets): G_all = v1 ∩ 5K ∩ HD probe, G_5k = top-N of (5K ∩ HD probe) by geometric-mean count
            fraction (per-platform counts stored) -> results/ovff/crossplatform/genes_{all,5k}.txt, genes.json
  twopoint  one section x gene set x frame: units (molecules / 2 µm bins) -> reference frame (nonrigid) -> 2 µm lattice
            with the fixed origin -> per-tile pair-count tensor (scote.twopoint with keep_tiles=True) + tile
            bootstrap over ALL tiles (= the unregistered whole-section baseline) -> data/twopoint/ovff/<name>.npz
            (`--thin p`: binomial thinning of the bin counts, used by ovff_depth.py)
  tiles     tile table on the fixed grid: mask tissue / region fractions (5K-frame mask) and per-section counts from the
            run tensors -> matched tiles (tissue in every compared section), tumor / stroma tile sets
            -> results/ovff/crossplatform/tiles_<geneset>_<tile_um>.json
  compare   levels 1-3 (global / region-matched / local per tile) on matched tiles vs the unregistered whole-section
            baseline, same pair set; intervals of the agreement statistics from the closed-form tile block bootstrap
            with the delta method (per-tile influence of every log g; joint tiles for the registered levels)
            -> results/ovff/crossplatform/*.json|md, quick-looks figures/quicklook/ovff_crossplatform_{levels,pairs,regions}.png
"""
from __future__ import annotations

import argparse, json, resource, time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "results/ovff/crossplatform"
TP = DATA / "twopoint/ovff"
QL = REPO / "figures/quicklook"
ORIGIN = np.array([-1000.0, -1000.0])      # fixed tile/lattice origin in the 5K frame (µm); tiles align at multiples of 250/500
PITCH = 2.0                                # 2 µm lattice as in Figs. A/C
N_LEVELS = 10                              # r up to 2*8*2^10 = 16 mm
SECTIONS = {
    "ovff_xenium_5k": dict(kind="molecules", path=DATA / "processed/ovff_xenium_5k/molecules.parquet", label="Xenium 5K", short="5k"),
    "ovff_xenium_v1": dict(kind="molecules", path=DATA / "processed/ovff_xenium_v1/molecules.parquet", label="Xenium v1", short="v1"),
    "ovff_hd_probe_deep": dict(kind="bins", path=DATA / "processed/ovff_hd_probe_deep/bins_square_002um.npz", label="Visium HD probe, deep", short="hdd"),
    "ovff_hd_probe_min": dict(kind="bins", path=DATA / "processed/ovff_hd_probe_min/bins_square_002um.npz", label="Visium HD probe, min depth", short="hdm"),
    # 3' Visium HD section (aligned from the FASTQs with Space Ranger; not used by the preprint figures; skipped when absent)
    "ovff_hd3p_min": dict(kind="bins", path=DATA / "processed/ovff_hd3p_min/bins_square_002um.npz", label="Visium HD 3', min depth", short="hd3"),
}
GENESETS = {"all": ["ovff_xenium_v1", "ovff_xenium_5k", "ovff_hd_probe_deep", "ovff_hd_probe_min"],
            "5k": ["ovff_xenium_5k", "ovff_hd_probe_deep", "ovff_hd_probe_min"]}
R_QUOTE = [4.0, 10.0, 20.0, 48.0, 96.0]    # r_from of the bins quoted (≈ 5, 10, 20, 50, 100 µm)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def maxrss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


# ----------------------------------------------------------------------------- loading
def gene_totals(section):
    """Total counts per gene (Series) of a section; molecules from feature_counts.json, bins from the column sums."""
    s = SECTIONS[section]
    if s["kind"] == "molecules":
        return pd.Series(json.load(open(s["path"].parent / "feature_counts.json")), dtype=float)
    z = np.load(s["path"], allow_pickle=True)
    X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    return pd.Series(np.asarray(X.sum(0)).ravel(), index=z["features"].astype(str), dtype=float)


def load_units(section, genes, thin: float = 0.0, seed: int = 0):
    """coords (n, 2) own µm, X (n, G) CSR counts on `genes` (one-hot for molecules), n_units_total."""
    s = SECTIONS[section]
    if s["kind"] == "molecules":
        import pyarrow.parquet as pq
        tbl = pq.read_table(s["path"], columns=["x", "y", "feature"])
        feat = tbl["feature"].to_pandas().astype("category")
        m = feat.isin(genes).to_numpy()
        xy = np.stack([tbl["x"].to_numpy()[m], tbl["y"].to_numpy()[m]], 1).astype(np.float64)
        codes = pd.Categorical(np.asarray(feat[m]).astype(str), categories=genes).codes.astype(np.int64)
        X = sp.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)), shape=(len(codes), len(genes)))
        return xy, X, int(len(feat))
    z = np.load(s["path"], allow_pickle=True)
    X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    feats = pd.Index(z["features"].astype(str)); coords = z["coords_um"].astype(np.float64)
    if "in_tissue" in z:
        keep = z["in_tissue"].astype(bool); X = X[keep]; coords = coords[keep]
    cols = feats.get_indexer(genes); assert (cols >= 0).all(), "gene missing in bin table"
    X = X[:, cols].tocsr()
    if thin > 0:
        rng = np.random.default_rng(seed)
        X.data = rng.binomial(np.rint(X.data).astype(np.int64), thin).astype(np.float64)
        X.eliminate_zeros()
    return coords, X.astype(np.float64), int(X.shape[0])


def to_ref(coords, section, frame):
    if frame == "own" or section == "ovff_xenium_5k":
        return coords
    from registration import to_reference
    return to_reference(coords, section, nonrigid=True)


def run_name(section, geneset, frame, tile_um, thin=0.0):
    return f"{SECTIONS[section]['short']}_{geneset}_{frame}_t{int(tile_um)}" + (f"_thin{thin:g}" if thin > 0 else "")


# ----------------------------------------------------------------------------- stage genes
def stage_genes(a):
    OUT.mkdir(parents=True, exist_ok=True)
    gsets = a.genesets.split(",")
    tot = {s: gene_totals(s) for s in sorted({s for gs in gsets for s in GENESETS[gs]})}
    if SECTIONS["ovff_hd3p_min"]["path"].exists():
        tot["ovff_hd3p_min"] = gene_totals("ovff_hd3p_min")
    info = {}
    for gs in gsets:
        secs = GENESETS[gs]
        common = set.intersection(*[set(tot[s].index[tot[s] > 0]) for s in secs if s in tot])
        frac = pd.DataFrame({s: tot[s].reindex(sorted(common)) / tot[s].sum() for s in secs if s in tot})
        rank = np.exp(np.log(frac.clip(lower=1e-12)).mean(1)).sort_values(ascending=False)
        genes = list(rank.index[: a.top_5k]) if gs == "5k" else list(rank.index)
        (OUT / f"genes_{gs}.txt").write_text("\n".join(genes) + "\n")
        info[gs] = dict(n_common=len(common), n_used=len(genes), sections=secs, rank="geometric mean of per-platform count fractions",
                        counts={s: {g: float(tot[s][g]) for g in genes} for s in secs if s in tot})
        if "ovff_hd3p_min" in tot:   # fifth column: the lists above are NOT re-intersected (existing runs use them); record the 3' overlap
            t3 = tot["ovff_hd3p_min"]; present = [g for g in genes if g in t3.index and t3[g] > 0]
            info[gs]["hd3p"] = dict(n_in_3p_reference=int(sum(g in t3.index for g in genes)), n_with_counts=len(present),
                                    n_common_all_sections=len(common & set(t3.index[t3 > 0])), counts={g: float(t3.get(g, 0.0)) for g in genes})
            log(f"geneset {gs}: 3' HD has {len(present)} of {len(genes)} genes with counts ({t3.reindex(genes).fillna(0).sum():.3g} UMI on the set)")
        log(f"geneset {gs}: {len(common)} common, {len(genes)} used; counts per platform on the set: "
            + ", ".join(f"{s} {tot[s][genes].sum():.3g}" for s in secs if s in tot))
    json.dump(info, open(OUT / "genes.json", "w"), indent=1)


# ----------------------------------------------------------------------------- stage twopoint
def stage_twopoint(a):
    from scote import CPUOptions, twopoint
    TP.mkdir(parents=True, exist_ok=True); (REPO / "results/twopoint").mkdir(exist_ok=True)
    genes = [l.strip() for l in open(OUT / f"genes_{a.geneset}.txt") if l.strip()]
    for section in a.sections.split(","):
        name = run_name(section, a.geneset, a.frame, a.tile_um, a.thin)
        out = TP / f"{name}.npz"
        if out.exists() and not a.force:
            log("exists", out); continue
        t0 = time.time()
        coords, X, n_total = load_units(section, genes, thin=a.thin, seed=a.seed)
        t_load = time.time() - t0
        coords = to_ref(coords, section, a.frame)
        ok = np.all((coords > ORIGIN + 1) & (coords < ORIGIN + 40000), axis=1)   # drop units mapped far outside the reference frame
        coords, X = coords[ok], X[ok]
        origin = ORIGIN if a.frame == "ref" else coords.min(0)
        log(f"{name}: {X.shape[0]:,} units x {X.shape[1]} genes, {X.sum():.4g} counts (of {n_total:,} units; load {t_load:.0f}s, "
            f"transform {time.time()-t0-t_load:.0f}s), bbox {coords.min(0).round()} - {coords.max(0).round()}")
        t1 = time.time()
        # lattice hierarchy with the per-tile counts kept and the closed-form tile block bootstrap (symmetrized, T_eff flag)
        res = twopoint(coords, X, names=genes, delta=PITCH, bin_size=PITCH, n_levels=N_LEVELS, tile_um=a.tile_um,
                       origin=origin, backend=CPUOptions(mode=a.cpu_mode) if a.backend == "cpu" else a.backend,
                       keep_tiles=True, verbose=True)
        t2 = time.time()
        tiles = res.tiles      # column, row of every tile index on the grid of `origin`
        lo, hi = res.interval()
        np.savez(out, log_g=res.log_g, lo=lo, hi=hi, sd=res.sd, n_eff_pairs=res.n_eff_pairs, P=res.P, totals=res.totals,
                 features=np.array(genes, dtype=str), r_from=res.bins["r_from"].to_numpy(), r_to=res.bins["r_to"].to_numpy(), r_central=res.r,
                 t_eff=res.t_eff, flag=res.flag,
                 counts_tiles=res.counts_tiles.astype(np.float32), totals_tiles=res.totals_tiles, tile_ix=tiles["column"].to_numpy(), tile_iy=tiles["row"].to_numpy(), tile_um=a.tile_um,
                 origin=origin, frame=a.frame, section=section, thin=a.thin)
        summ = dict(section=section, geneset=a.geneset, frame=a.frame, tile_um=a.tile_um, thin=a.thin, seed=a.seed, n_units=int(X.shape[0]), n_units_total=n_total,
                    total_counts=float(X.sum()), n_genes=len(genes), n_tiles=res.n_tiles, n_levels=N_LEVELS, pitch=PITCH, backend=a.backend,
                    threads=res.params["threads"], uncertainty="closed", origin=list(map(float, origin)),
                    timings=dict(load=t_load, twopoint=t2 - t1, save=time.time() - t2, twopoint_detail=res.timings),
                    median_ci_width_log_g=float(np.nanmedian(hi - lo)), maxrss_gb=maxrss_gb(), out=str(out.relative_to(REPO)))
        json.dump(summ, open(REPO / f"results/twopoint/ovff_{name}.json", "w"), indent=1, default=float)
        log(f"{name}: tiles {res.n_tiles}, two-point run {t2-t1:.0f}s, CI width {summ['median_ci_width_log_g']:.3f}, maxRSS {maxrss_gb():.1f} GB")
        del res, X, coords


# ----------------------------------------------------------------------------- stage tiles
def mask_tile_fractions(tile_um, origin=ORIGIN):
    """Per tile on the fixed grid: tissue fraction and region fractions (of tissue pixels) from the 5K-frame 10 µm mask."""
    from registration import load_region_mask
    m = load_region_mask(None)
    lab = m["labels"]; px = m["pixel_um"]; x0, y0 = m["origin_um"]
    rows, cols = np.indices(lab.shape)
    ix = np.floor((cols * px + px / 2 + x0 - origin[0]) / tile_um).astype(int); iy = np.floor((rows * px + px / 2 + y0 - origin[1]) / tile_um).astype(int)
    df = pd.DataFrame(dict(ix=ix.ravel(), iy=iy.ravel(), lab=lab.ravel()))
    n_px = (tile_um / px) ** 2
    g = df.groupby(["ix", "iy"])
    out = pd.DataFrame(dict(tissue=g["lab"].apply(lambda v: (v > 0).sum() / n_px)))
    for k, nm in enumerate(m["names"]):
        if k == 0: continue
        out[nm] = g["lab"].apply(lambda v, k=k: (v == k).sum() / max((v > 0).sum(), 1))
    return out.reset_index()


def load_run(name, arrays=("totals_tiles", "tile_ix", "tile_iy")):
    d = np.load(TP / f"{name}.npz", allow_pickle=True)
    return {k: d[k] for k in arrays}


def stage_tiles(a):
    factor = int(round(a.tile_um / a.run_tile_um))
    secs = [s for s in GENESETS[a.geneset] if (TP / f"{run_name(s, a.geneset, 'ref', a.run_tile_um)}.npz").exists()]
    if not a.no_hd3 and SECTIONS["ovff_hd3p_min"]["path"].exists() and (TP / f"{run_name('ovff_hd3p_min', a.geneset, 'ref', a.run_tile_um)}.npz").exists():
        secs.append("ovff_hd3p_min")
    frac = mask_tile_fractions(a.tile_um).set_index(["ix", "iy"])
    tab = frac.copy()
    for s in secs:
        r = load_run(run_name(s, a.geneset, "ref", a.run_tile_um))
        cnt = pd.Series(r["totals_tiles"].sum(1), index=pd.MultiIndex.from_arrays([r["tile_ix"] // factor, r["tile_iy"] // factor])).groupby(level=[0, 1]).sum()
        tab[f"counts_{SECTIONS[s]['short']}"] = cnt.reindex(tab.index).fillna(0.0)
    tab = tab.fillna(0.0)
    # occupancy per section: counts relative to the median of tissue tiles (mask tissue fraction >= min_tissue; this holey
    # papillary tumor has no tile with > 85 % tissue)
    full = tab["tissue"] >= a.min_tissue
    occ = {}
    for s in secs:
        c = tab[f"counts_{SECTIONS[s]['short']}"]
        med = float(np.median(c[full & (c > 0)]))
        occ[s] = (c >= a.occ_frac * med) & (tab["tissue"] >= a.min_tissue)
        tab[f"in_{SECTIONS[s]['short']}"] = occ[s]
    matched = np.logical_and.reduce([occ[s].to_numpy() for s in secs]) if secs else np.zeros(len(tab), bool)
    tab["matched"] = matched
    tab["tumor_tile"] = matched & (tab["tumor"] >= a.tumor_frac)
    tab["stroma_tile"] = matched & (tab["stroma"] >= a.stroma_frac)
    area = (a.tile_um / 1000) ** 2
    summ = dict(geneset=a.geneset, tile_um=a.tile_um, run_tile_um=a.run_tile_um, origin=ORIGIN.tolist(), sections=secs, occ_frac=a.occ_frac, min_tissue=a.min_tissue,
                tumor_frac=a.tumor_frac, stroma_frac=a.stroma_frac,
                n_tiles_mask_tissue=int((tab["tissue"] >= a.min_tissue).sum()), n_tiles_per_section={s: int(occ[s].sum()) for s in secs},
                area_mm2_per_section={s: float(occ[s].sum() * area) for s in secs},
                n_matched=int(matched.sum()), area_matched_mm2=float(matched.sum() * area),
                n_tumor_tiles=int(tab["tumor_tile"].sum()), n_stroma_tiles=int(tab["stroma_tile"].sum()),
                matched_tissue_area_weighted_mm2=float((tab["tissue"][matched]).sum() * area),
                matched_region_area_mm2={k: float((tab["tissue"][matched] * tab[k][matched]).sum() * area) for k in ["tumor", "stroma", "vessel", "immune", "necrosis"]},
                unmatched_by_section={s: int((occ[s] & ~matched).sum()) for s in secs})
    tab.reset_index().to_csv(OUT / f"tiles_{a.geneset}_t{int(a.tile_um)}.csv", index=False)
    json.dump(summ, open(OUT / f"tiles_{a.geneset}_t{int(a.tile_um)}.json", "w"), indent=1)
    log(json.dumps(summ, indent=1))
    # quick-look map
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = len(secs) + 1
    fig, axs = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False); axs = axs[0]
    for k, s in enumerate(secs):
        c = tab[f"counts_{SECTIONS[s]['short']}"]
        axs[k].scatter(tab.index.get_level_values(0) * a.tile_um + ORIGIN[0], tab.index.get_level_values(1) * a.tile_um + ORIGIN[1], c=np.log10(c + 1), s=12, marker="s", cmap="viridis")
        axs[k].scatter(tab.index.get_level_values(0)[~occ[s]] * a.tile_um + ORIGIN[0], tab.index.get_level_values(1)[~occ[s]] * a.tile_um + ORIGIN[1], s=12, marker="s", facecolors="none", edgecolors="r", lw=0.3)
        axs[k].set_title(f"{SECTIONS[s]['label']}: log10 counts, {int(occ[s].sum())} tiles", fontsize=8); axs[k].set_aspect("equal"); axs[k].invert_yaxis()
    cls = np.where(tab["tumor_tile"], 2, np.where(tab["stroma_tile"], 3, np.where(matched, 1, 0)))
    axs[-1].scatter(tab.index.get_level_values(0) * a.tile_um + ORIGIN[0], tab.index.get_level_values(1) * a.tile_um + ORIGIN[1], c=cls, s=12, marker="s", cmap=matplotlib.colors.ListedColormap(["#dddddd", "#1f77b4", "#d62728", "#2ca02c"]), vmin=-0.5, vmax=3.5)
    axs[-1].set_title(f"matched {int(matched.sum())} (blue), tumor {int(tab['tumor_tile'].sum())} (red), stroma {int(tab['stroma_tile'].sum())} (green)", fontsize=8)
    axs[-1].set_aspect("equal"); axs[-1].invert_yaxis()
    fig.suptitle(f"{a.tile_um:g} µm tile grid in the Xenium 5K frame (gene set {a.geneset})", fontsize=9); fig.tight_layout()
    QL.mkdir(parents=True, exist_ok=True); fig.savefig(QL / f"ovff_tiles_{a.geneset}_t{int(a.tile_um)}.png", dpi=130)


# ----------------------------------------------------------------------------- stage compare
def load_tensor(name):
    d = np.load(TP / f"{name}.npz", allow_pickle=True)
    return dict(ct=d["counts_tiles"], tt=d["totals_tiles"], ix=d["tile_ix"], iy=d["tile_iy"], feats=list(map(str, d["features"])),
                r_from=d["r_from"], r_to=d["r_to"], r=d["r_central"], log_g=d["log_g"], sd=d["sd"], tile_um=float(d["tile_um"]))


def aggregate_tiles(run, factor):
    """Sum 2x2 (factor) sub-tiles into coarser tiles (fixed origin -> exact)."""
    if factor == 1:
        run["aggregated"] = False
        return run
    ix, iy = run["ix"] // factor, run["iy"] // factor
    key = ix * 100000 + iy
    u, inv = np.unique(key, return_inverse=True)
    T = len(u)
    ct = np.zeros((T,) + run["ct"].shape[1:], dtype=np.float32); tt = np.zeros((T, run["tt"].shape[1]))
    np.add.at(ct, inv, run["ct"]); np.add.at(tt, inv, run["tt"])
    out = dict(run); out.update(ct=ct, tt=tt, ix=u // 100000, iy=u % 100000, tile_um=run["tile_um"] * factor, aggregated=True)
    return out


def tile_index(run, tab_keys):
    """Positions of the tiles (ix, iy) in tab_keys within the run (or -1)."""
    key = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(run["ix"], run["iy"]))}
    return np.array([key.get(k, -1) for k in tab_keys], dtype=int)


def logg_from_weights(ct, tt, W, k):
    """log g (n_boot, G, G) at r-bin k for tile weights W (n_boot, T) (W = ones row -> point estimate)."""
    from scote import random_label_reference
    T = ct.shape[0]
    Pb = (W @ ct[:, k].reshape(T, -1).astype(np.float64)).reshape(W.shape[0], ct.shape[2], ct.shape[3])
    nb = W @ tt
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(Pb / random_label_reference(Pb[:, None], nb)[:, 0])   # reference wants (..., n_bins, G, G)


def metrics(x, y, sx=None, sy=None):
    from scipy.stats import spearmanr
    ok = np.isfinite(x) & np.isfinite(y)
    if sx is not None:
        ok &= np.isfinite(sx) & np.isfinite(sy) & (sx < 0.5) & (sy < 0.5)
    x, y = x[ok], y[ok]
    if len(x) < 10:
        return dict(n=int(len(x)))
    slope, icpt = np.polyfit(x, y, 1)
    out = dict(n=int(len(x)), pearson=float(np.corrcoef(x, y)[0, 1]), spearman=float(spearmanr(x, y)[0]), rms=float(np.sqrt(np.mean((x - y) ** 2))),
               mean_diff=float(np.mean(y - x)), slope=float(slope), intercept=float(icpt))
    if sx is not None:
        z = (y - x) / np.sqrt(sx[ok] ** 2 + sy[ok] ** 2 + 1e-12)
        out.update(within_1sd=float(np.mean(np.abs(z) < 1)), within_2sd=float(np.mean(np.abs(z) < 2)), median_abs_z=float(np.median(np.abs(z))))
    return out


def _slab(run, idx, genes, k):
    """(T, G, G) float64 pair counts of r bin k on the tiles idx and the genes `genes` (one shell; no copy of the tensor)."""
    C = np.asarray(run["ct"][np.asarray(idx), k])
    return C[:, genes][:, :, genes].astype(np.float64)


def closed_form_bin(run, idx, genes, k):
    """Point log g (upper triangle a <= b), its closed-form tile-bootstrap SD, reliability flag and per-tile influence
    (T, n_pairs) at r bin k for run `run` restricted to the tiles idx and the genes `genes` (random-label reference of
    that restriction, as the point estimate). Point log g by the arithmetic of logg_from_weights (unit weights)."""
    from scote import uncertainty as TV
    C = _slab(run, idx, genes, k); tt = run["tt"][np.asarray(idx)][:, genes]
    T, G = C.shape[0], C.shape[1]; iu = np.triu_indices(G)
    lg = logg_from_weights(C[:, None], tt, np.ones((1, T)), 0)[0][iu]
    l = TV.influence_log_g(C[:, None], tt, shells=[0])[:, 0]            # (T, G, G), symmetrized slabs
    Cs = 0.5 * (C + np.swapaxes(C, 1, 2)); Ps = Cs.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_eff = (Ps ** 2 / np.einsum("tab,tab->ab", Cs, Cs))[iu]
    del C, Cs
    li = l[:, iu[0], iu[1]]; del l
    sd = np.sqrt(np.einsum("tp,tp->p", li, li))
    return lg, sd, TV.reliability_flag(t_eff), li


def _bin_key(run, idx, genes, k):
    return (id(run), hash(np.asarray(idx).tobytes()), hash(np.asarray(genes).tobytes()), int(k))


def cached_bin(cache, run, idx, genes, k):
    """closed_form_bin through an optional dict cache (keyed by run, tiles, genes and bin)."""
    if cache is None:
        return closed_form_bin(run, idx, genes, k)
    key = _bin_key(run, idx, genes, k)
    if key not in cache:
        cache[key] = closed_form_bin(run, idx, genes, k)
    return cache[key]


AGREEMENT_GRADS = ("pearson", "rms", "mean_diff", "slope", "intercept")


def compare_pair(A, B, idxA, idxB, ia, ib, ks, n_boot=None, seed=None, joint=False, pairs_mask=None, cache=None):
    """Agreement of log g between run A (tiles idxA) and run B (tiles idxB) at r bins ks. Intervals: closed-form tile block
    bootstrap with the delta method. Per run and bin, the per-tile influence l_t(ab) of every log g on the selected tiles
    and genes (scote.uncertainty.influence_log_g, symmetrized slabs) gives its SD (sqrt(sum_t l_t^2); pair filter: finite,
    SD < 0.5 and an interval, reliability flag != 2, on both sides) and, combined with the gradient of Pearson, RMS, mean
    difference, OLS slope and intercept over the pairs (scote.uncertainty.grad_*), the variance of the statistic
    (scote.uncertainty.delta_var). joint=True: matched tiles, idxA[i] and idxB[i] are the same registered tile, so one tile
    resample acts on both runs and the per-tile contributions are summed before squaring; joint=False: independent tile
    resamples (unregistered whole sections), the variances add. Interval = estimate +- 1.96 SD (centered by construction).
    Spearman is reported as a point value only (rank statistic, no gradient). n_boot and seed are unused (Monte Carlo
    version). cache: optional dict reused across calls on the same runs (per-bin results keyed by run, tiles and bin)."""
    from scote import uncertainty as TV
    rows = []; valid = {}
    if joint:
        assert len(idxA) == len(idxB), "joint tiles need the same tiles in both runs"
    for k in ks:
        la, sa, fa, lA = cached_bin(cache, A, idxA, ia, k)
        lb, sb, fb, lB = cached_bin(cache, B, idxB, ib, k)
        ok_noflag = np.isfinite(la) & np.isfinite(lb) & (sa < 0.5) & (sb < 0.5)
        ok = ok_noflag & (fa < 2) & (fb < 2)
        if pairs_mask is not None:
            ok &= pairs_mask[k]; ok_noflag &= pairs_mask[k]
        valid[k] = ok
        m = metrics(la[ok], lb[ok], sa[ok], sb[ok])
        if "pearson" in m:
            x, y = la[ok], lb[ok]; lA_, lB_ = lA[:, ok], lB[:, ok]
            gr = dict(pearson=TV.grad_pearson(x, y), rms=TV.grad_rms(x, y), mean_diff=TV.grad_mean_diff(x, y),
                      slope=TV.grad_ols_slope(x, y), intercept=TV.grad_ols_intercept(x, y))
            for key, (val, (gx, gy)) in gr.items():
                assert np.isclose(val, m[key], rtol=1e-9, atol=1e-12), (key, val, m[key])
                var = TV.delta_var([gx, gy], [lA_, lB_], joint=joint)
                m[key + "_sd"] = float(np.sqrt(var)); m[key + "_ci"] = [float(v) for v in TV.stat_ci(m[key], var)]
        m.update(k=int(k), r_from=float(A["r_from"][k]), r_to=float(A["r_to"][k]), r_um=float(A["r"][k]),
                 n_pairs_excluded_flag2=int((ok_noflag & ~ok).sum()), n_pairs_flag1=int(((np.maximum(fa, fb) == 1) & ok).sum()),
                 sd_A_median=float(np.median(sa[ok])) if ok.any() else float("nan"), sd_B_median=float(np.median(sb[ok])) if ok.any() else float("nan"))
        rows.append(m)
    return rows, valid


def check_whole_section(run, name, ia, ks, cache=None):
    """Identity checks for a whole-section run on its stored tile grid: the point log g of closed_form_bin equals the stored
    log_g of the npz (to float32 rounding of the stored tile tensor), and (if the tiles were not aggregated) the influence SD
    equals the stored closed-form sd."""
    G = len(ia); iu = np.triu_indices(G); out = dict(run=name, max_abs_dlogg=0.0, max_rel_dsd=None)
    d = np.load(TP / f"{name}.npz", allow_pickle=True); lg_st, sd_st = d["log_g"], d["sd"]
    for k in ks:
        lg, sd, _, _ = cached_bin(cache, run, np.arange(run["ct"].shape[0]), ia, k)
        ref = lg_st[k][np.ix_(ia, ia)][iu]; fin = np.isfinite(ref) & np.isfinite(lg)
        assert np.array_equal(np.isfinite(ref), np.isfinite(lg)), f"{name}: finite pattern of log g differs at bin {k}"
        out["max_abs_dlogg"] = max(out["max_abs_dlogg"], float(np.max(np.abs(lg[fin] - ref[fin]))))
        if run.get("aggregated") is False:
            s0 = sd_st[k][np.ix_(ia, ia)][iu]; mm = fin & np.isfinite(s0) & (s0 > 0)
            out["max_rel_dsd"] = max(out["max_rel_dsd"] or 0.0, float(np.max(np.abs(sd[mm] / s0[mm] - 1))))
    # the stored tile tensors are float32 (entries >= 2^24 lose their last bits), the stored log_g was computed from the
    # float64 counts before the cast: agreement to float32 rounding (the point estimates of this stage, as before the
    # closed form, come from the float32 tensors)
    assert out["max_abs_dlogg"] < 1e-6, out
    if out["max_rel_dsd"] is not None and out["max_rel_dsd"] > 1e-6:
        log(f"WARNING {name}: influence SD differs from the stored closed-form sd (max rel {out['max_rel_dsd']:.2e})")
    return out


def per_tile_log_g(ct, tt, min_tot, min_pairs=10):
    """Per-tile log g (T, K, G, G) float32; NaN where the tile has < min_tot counts or the pair < min_pairs pairs."""
    T = ct.shape[0]; out = np.full(ct.shape, np.nan, dtype=np.float32)
    for t in range(T):
        n = tt[t]; P = ct[t].astype(np.float64)
        if n.sum() < min_tot: continue
        ref = P.sum(axis=(-1, -2))[:, None, None] * np.outer(n, n)[None] / max(n.sum(), 1) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.log(P / ref)
        lg[P < min_pairs] = np.nan
        out[t] = lg
    return out


def stage_compare(a):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    gs = a.geneset
    tab = pd.read_csv(OUT / f"tiles_{gs}_t{int(a.tile_um)}.csv")
    tinfo = json.load(open(OUT / f"tiles_{gs}_t{int(a.tile_um)}.json"))
    secs = tinfo["sections"]
    if a.pairs_with:   # only comparisons involving this section (e.g. the 3' HD column added later); output tag gets a suffix
        assert a.pairs_with in secs, (a.pairs_with, secs)
    factor = int(round(a.tile_um / a.run_tile_um))
    runs = {s: aggregate_tiles(load_tensor(run_name(s, gs, "ref", a.run_tile_um)), factor) for s in secs}
    log(f"loaded {len(runs)} runs ({gs}, run tiles {a.run_tile_um} -> {a.tile_um} µm), maxRSS {maxrss_gb():.1f} GB")
    genes = sorted(set.intersection(*[set(r["feats"]) for r in runs.values()]))
    sets = {"matched": tab[tab.matched], "tumor": tab[tab.tumor_tile], "stroma": tab[tab.stroma_tile]}
    keys = {k: list(zip(v.ix, v.iy)) for k, v in sets.items()}
    ks = [int(np.argmin(np.abs(runs[secs[0]]["r_from"] - rf))) for rf in R_QUOTE] + \
         [k for k in range(len(runs[secs[0]]["r_from"])) if runs[secs[0]]["r_from"][k] >= a.r_min and runs[secs[0]]["r_from"][k] <= a.r_max_curve]
    ks = sorted(set(ks))
    res = dict(geneset=gs, tile_um=a.tile_um, run_tile_um=a.run_tile_um, n_genes=len(genes), sections=secs, tiles=tinfo, n_boot="closed",
               uncertainty="closed", comparisons=[])
    # identity checks: whole-section point log g = stored npz log_g (and SD = stored closed-form sd on the stored tile grid)
    # whole-section bins are cached for the whole stage (each section's all-tile influence is computed once and reused by
    # every comparison it takes part in); other tile sets are dropped after each comparison
    cache = {}
    all_keys = lambda: {_bin_key(r, np.arange(r["ct"].shape[0]), [r["feats"].index(g) for g in genes], k) for r in runs.values() for k in ks}
    res["checks"] = {s: check_whole_section(runs[s], run_name(s, gs, "ref", a.run_tile_um), [runs[s]["feats"].index(g) for g in genes], ks, cache)
                     for s in secs}
    log("identity checks " + json.dumps(res["checks"]))
    for x in range(len(secs)):
        for y in range(x + 1, len(secs)):
            if a.pairs_with and a.pairs_with not in (secs[x], secs[y]): continue
            A, B = runs[secs[x]], runs[secs[y]]
            ia = [A["feats"].index(g) for g in genes]; ib = [B["feats"].index(g) for g in genes]
            comp = dict(A=secs[x], B=secs[y], levels={})
            t0 = time.time()
            # level 1a: unregistered baseline = all tiles of each section, independent resamples
            allA, allB = np.arange(A["ct"].shape[0]), np.arange(B["ct"].shape[0])
            keep = all_keys()
            for key in [key for key in cache if key not in keep]:
                del cache[key]
            rows_u, valid_u = compare_pair(A, B, allA, allB, ia, ib, ks, joint=False, cache=cache)
            # level 1b: registered, matched tiles, joint resamples, same pair set as the unregistered comparison
            iA, iB = tile_index(A, keys["matched"]), tile_index(B, keys["matched"])
            okt = (iA >= 0) & (iB >= 0)
            rows_m, valid_m = compare_pair(A, B, iA[okt], iB[okt], ia, ib, ks, joint=True, cache=cache)
            common_pairs = {k: valid_u[k] & valid_m[k] for k in ks}
            rows_u2, _ = compare_pair(A, B, allA, allB, ia, ib, ks, joint=False, pairs_mask=common_pairs, cache=cache)
            rows_m2, _ = compare_pair(A, B, iA[okt], iB[okt], ia, ib, ks, joint=True, pairs_mask=common_pairs, cache=cache)
            comp["levels"]["1_unregistered_whole"] = rows_u2; comp["levels"]["1_registered_matched"] = rows_m2
            comp["levels"]["1_unregistered_whole_allpairs"] = rows_u; comp["levels"]["1_registered_matched_allpairs"] = rows_m
            comp["n_tiles"] = dict(A_all=int(len(allA)), B_all=int(len(allB)), matched=int(okt.sum()))
            # level 2: region tiles (tumor-only, stroma-only), matched, joint
            for reg in ("tumor", "stroma"):
                jA, jB = tile_index(A, keys[reg]), tile_index(B, keys[reg]); okr = (jA >= 0) & (jB >= 0)
                comp["n_tiles"][reg] = int(okr.sum())
                if okr.sum() >= a.min_region_tiles:
                    rows_r, _ = compare_pair(A, B, jA[okr], jB[okr], ia, ib, ks, joint=True, pairs_mask=common_pairs)
                    comp["levels"][f"2_{reg}_tiles"] = rows_r
            # level 3: per-tile functions on matched tiles: tile medians and the paired per-tile agreement
            lgA = per_tile_log_g(A["ct"][iA[okt]][:, :, ia][:, :, :, ia], A["tt"][iA[okt]][:, ia], a.min_tile_counts)
            lgB = per_tile_log_g(B["ct"][iB[okt]][:, :, ib][:, :, :, ib], B["tt"][iB[okt]][:, ib], a.min_tile_counts)
            iu = np.triu_indices(len(genes)); rows3 = []
            rng = np.random.default_rng(a.seed)
            for k in ks:
                if A["r_from"][k] > 0.2 * a.tile_um: continue
                la, lb = lgA[:, k][:, iu[0], iu[1]], lgB[:, k][:, iu[0], iu[1]]           # (T, n_pairs)
                nA, nB = np.isfinite(la).sum(0), np.isfinite(lb).sum(0)
                okp = (nA >= a.min_tiles) & (nB >= a.min_tiles) & common_pairs[k]
                if okp.sum() < 10: continue
                with np.errstate(all="ignore"):
                    medA, medB = np.nanmedian(la[:, okp], 0), np.nanmedian(lb[:, okp], 0)
                    iqrA = np.nanpercentile(la[:, okp], 75, 0) - np.nanpercentile(la[:, okp], 25, 0); iqrB = np.nanpercentile(lb[:, okp], 75, 0) - np.nanpercentile(lb[:, okp], 25, 0)
                m_med = metrics(medA, medB)
                # paired per-tile agreement: for each pair, correlation over matched tiles of log g_A(t) vs log g_B(t); floor = shuffled tile assignment
                def tile_corr(lb_):
                    xa, xb = la[:, okp], lb_[:, okp]; both = np.isfinite(xa) & np.isfinite(xb)
                    xa = np.where(both, xa, np.nan); xb = np.where(both, xb, np.nan)
                    ma, mb = np.nanmean(xa, 0), np.nanmean(xb, 0)
                    cov = np.nansum((xa - ma) * (xb - mb), 0); va = np.nansum((xa - ma) ** 2, 0); vb = np.nansum((xb - mb) ** 2, 0)
                    with np.errstate(all="ignore"):
                        return cov / np.sqrt(va * vb), np.nanmean(xa - xb, 0), both.sum(0)
                rho, dmean, nboth = tile_corr(lb)
                floor = np.stack([tile_corr(lb[rng.permutation(lb.shape[0])])[0] for _ in range(a.n_perm)])
                # per tile: correlation over pairs of the two platforms' log g in the same tile vs in a shuffled tile
                def pair_corr(lb_):
                    out = []
                    for t in range(la.shape[0]):
                        xa, xb = la[t, okp], lb_[t, okp]; both = np.isfinite(xa) & np.isfinite(xb)
                        out.append(np.corrcoef(xa[both], xb[both])[0, 1] if both.sum() >= 20 else np.nan)
                    return np.array(out)
                pc = pair_corr(lb); pc_floor = np.stack([pair_corr(lb[rng.permutation(lb.shape[0])]) for _ in range(a.n_perm)])
                rows3.append(dict(k=int(k), r_from=float(A["r_from"][k]), r_um=float(A["r"][k]), n_pairs=int(okp.sum()),
                                  tile_median=dict(pearson=m_med.get("pearson"), rms=m_med.get("rms"), slope=m_med.get("slope"),
                                                   iqr_A=float(np.nanmedian(iqrA)), iqr_B=float(np.nanmedian(iqrB))),
                                  paired_tiles=dict(median_rho_over_pairs=float(np.nanmedian(rho)), rho_q25_q75=[float(np.nanpercentile(rho, 25)), float(np.nanpercentile(rho, 75))],
                                                    frac_rho_pos=float(np.nanmean(rho > 0)), median_tiles_per_pair=float(np.median(nboth)),
                                                    floor_median_rho=[float(np.nanmedian(floor[i])) for i in range(a.n_perm)],
                                                    floor_median_rho_mean=float(np.mean(np.nanmedian(floor, 1))), floor_median_rho_sd=float(np.std(np.nanmedian(floor, 1), ddof=1)),
                                                    per_tile_pearson_over_pairs_median=float(np.nanmedian(pc)), per_tile_pearson_q25_q75=[float(np.nanpercentile(pc, 25)), float(np.nanpercentile(pc, 75))],
                                                    per_tile_pearson_floor_mean=float(np.nanmean(np.nanmedian(pc_floor, 1))), per_tile_pearson_floor_sd=float(np.nanstd(np.nanmedian(pc_floor, 1), ddof=1)),
                                                    rms_tile_diff_median=float(np.nanmedian(np.sqrt(np.nanmean((la[:, okp] - lb[:, okp]) ** 2, 0)))))))
            comp["levels"]["3_local"] = rows3
            comp["seconds"] = time.time() - t0
            res["comparisons"].append(comp)
            def fmt(rows, key):
                return " ".join(f"{r['r_from']:.0f}:{r.get(key, np.nan):.3f}" for r in rows if r["r_from"] in R_QUOTE)
            log(f"{secs[x]} vs {secs[y]} ({comp['n_tiles']}): Pearson unreg {fmt(rows_u2, 'pearson')} | matched {fmt(rows_m2, 'pearson')} | "
                f"RMS unreg {fmt(rows_u2, 'rms')} | matched {fmt(rows_m2, 'rms')} | within1sd unreg {fmt(rows_u2, 'within_1sd')} | matched {fmt(rows_m2, 'within_1sd')} ({comp['seconds']:.0f}s)")
            if rows3:
                log("   local: " + " ".join(f"{r['r_from']:.0f}: rho {r['paired_tiles']['median_rho_over_pairs']:.3f} (floor {r['paired_tiles']['floor_median_rho_mean']:.3f}±{r['paired_tiles']['floor_median_rho_sd']:.3f}), "
                                             f"tile-pearson {r['paired_tiles']['per_tile_pearson_over_pairs_median']:.3f} (floor {r['paired_tiles']['per_tile_pearson_floor_mean']:.3f})" for r in rows3))
    tag = f"{gs}_t{int(a.tile_um)}" + (f"_{SECTIONS[a.pairs_with]['short']}" if a.pairs_with else "")
    json.dump(res, open(OUT / f"compare_{tag}.json", "w"), indent=1, default=float)
    write_md(res, OUT / f"compare_{tag}.md")
    quicklooks(res, runs, genes, tag, a)
    log("wrote", OUT / f"compare_{tag}.json", f"maxRSS {maxrss_gb():.1f} GB")


def write_md(res, path):
    L = [f"# Cross-platform agreement — gene set {res['geneset']} ({res['n_genes']} genes), {res['tile_um']:g} µm tiles, "
         f"{res['tiles']['n_matched']} matched tiles ({res['tiles']['area_matched_mm2']:.1f} mm²), tumor tiles {res['tiles']['n_tumor_tiles']}, stroma tiles {res['tiles']['n_stroma_tiles']}",
         "", "Intervals: 95 % closed-form tile block bootstrap, estimate ± 1.96 SD by the delta method over the per-tile influence of every log g (joint tiles "
         "for the registered levels, independent sections for the unregistered baseline). Pair set: identical for the unregistered and the registered rows (pairs "
         "finite with closed-form SD < 0.5 and an interval, T_eff ≥ 5, in both). r ≥ 4 µm only (3× the 2 µm resolution is 6 µm; the 4–6 µm bin is shown grayed in "
         "the figures).", ""]
    for c in res["comparisons"]:
        L.append(f"## {SECTIONS[c['A']]['label']} vs {SECTIONS[c['B']]['label']} (tiles: {c['n_tiles']})")
        L.append("| r_from [µm] | level | pairs | Pearson [95 % CI] | Spearman | RMS [CI] | slope [CI] | intercept | within 1 SD | within 2 SD |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for lev, rows in c["levels"].items():
            if lev.endswith("allpairs") or lev == "3_local": continue
            for r in rows:
                if r["r_from"] not in R_QUOTE or "pearson" not in r: continue
                pc, rc, sc = r["pearson_ci"], r["rms_ci"], r["slope_ci"]
                L.append(f"| {r['r_from']:.0f} | {lev} | {r['n']} | {r['pearson']:.3f} [{pc[0]:.3f}, {pc[1]:.3f}] | {r['spearman']:.3f} | "
                         f"{r['rms']:.3f} [{rc[0]:.3f}, {rc[1]:.3f}] | {r['slope']:.3f} [{sc[0]:.3f}, {sc[1]:.3f}] | {r['intercept']:.3f} | "
                         f"{r['within_1sd']:.2f} | {r['within_2sd']:.2f} |")
        if c["levels"].get("3_local"):
            L.append(""); L.append("| r_from [µm] | pairs | tile-median Pearson / RMS | per-pair ρ over matched tiles (median; floor mean ± SD over shuffles) | per-tile Pearson over pairs (median; floor) | median IQR over tiles A / B |")
            L.append("|---|---|---|---|---|---|")
            for r in c["levels"]["3_local"]:
                p = r["paired_tiles"]; m = r["tile_median"]
                L.append(f"| {r['r_from']:.0f} | {r['n_pairs']} | {m['pearson']:.3f} / {m['rms']:.3f} | {p['median_rho_over_pairs']:.3f} ({p['floor_median_rho_mean']:.3f} ± {p['floor_median_rho_sd']:.3f}) | "
                         f"{p['per_tile_pearson_over_pairs_median']:.3f} ({p['per_tile_pearson_floor_mean']:.3f} ± {p['per_tile_pearson_floor_sd']:.3f}) | {m['iqr_A']:.2f} / {m['iqr_B']:.2f} |")
        L.append("")
    path.write_text("\n".join(L))


def quicklooks(res, runs, genes, tag, a):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    comps = res["comparisons"]
    fig, axs = plt.subplots(2, len(comps), figsize=(3.4 * len(comps), 6), squeeze=False)
    for j, c in enumerate(comps):
        for lev, st in (("1_unregistered_whole", "--"), ("1_registered_matched", "-"), ("2_tumor_tiles", ":"), ("2_stroma_tiles", "-.")):
            rows = [r for r in c["levels"].get(lev, []) if "pearson" in r]
            if not rows: continue
            r_ = [r["r_um"] for r in rows]
            axs[0, j].plot(r_, [r["pearson"] for r in rows], st, color="C0", label=lev)
            axs[0, j].fill_between(r_, [r["pearson_ci"][0] for r in rows], [r["pearson_ci"][1] for r in rows], color="C0", alpha=0.12)
            axs[1, j].plot(r_, [r["rms"] for r in rows], st, color="C3", label=lev)
            axs[1, j].fill_between(r_, [r["rms_ci"][0] for r in rows], [r["rms_ci"][1] for r in rows], color="C3", alpha=0.12)
        axs[0, j].set_title(f"{SECTIONS[c['A']]['short']} vs {SECTIONS[c['B']]['short']}", fontsize=9)
        for ax in axs[:, j]: ax.set_xscale("log"); ax.set_xlabel("r [µm]")
        axs[0, j].set_ylim(0, 1); axs[0, j].set_ylabel("Pearson of log g over pairs"); axs[1, j].set_ylabel("RMS log-difference")
    axs[0, 0].legend(fontsize=6, frameon=False); fig.suptitle(f"Cross-platform agreement, gene set {res['geneset']} ({res['n_genes']} genes), {res['tile_um']:g} µm tiles", fontsize=9)
    fig.tight_layout(); fig.savefig(QL / f"ovff_crossplatform_levels_{tag}.png", dpi=130); plt.close(fig)
    # regions: matched vs tumor vs stroma Pearson/RMS at the quoted r (bar chart)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    levs = ["1_unregistered_whole", "1_registered_matched", "2_tumor_tiles", "2_stroma_tiles"]
    x = np.arange(len(R_QUOTE)); w = 0.8 / (len(comps) * len(levs))
    for j, c in enumerate(comps):
        for l, lev in enumerate(levs):
            rows = {r["r_from"]: r for r in c["levels"].get(lev, []) if "pearson" in r}
            for ax, key in zip(axs, ("pearson", "rms")):
                vals = [rows[rf][key] if rf in rows else np.nan for rf in R_QUOTE]
                err = np.clip([[rows[rf][key] - rows[rf][key + "_ci"][0] if rf in rows else 0 for rf in R_QUOTE], [rows[rf][key + "_ci"][1] - rows[rf][key] if rf in rows else 0 for rf in R_QUOTE]], 0, None)
                ax.bar(x + (j * len(levs) + l) * w, vals, w, yerr=err, label=f"{SECTIONS[c['A']]['short']}-{SECTIONS[c['B']]['short']} {lev[2:]}", alpha=0.85, capsize=1.5)
    for ax, key in zip(axs, ("Pearson", "RMS")):
        ax.set_xticks(x + 0.4); ax.set_xticklabels([f"{rf:g}" for rf in R_QUOTE]); ax.set_xlabel("r_from [µm]"); ax.set_ylabel(key)
    axs[0].legend(fontsize=5, frameon=False, ncol=2); fig.tight_layout(); fig.savefig(QL / f"ovff_crossplatform_regions_{tag}.png", dpi=130); plt.close(fig)
    # pairs: scatter of log g at r ≈ 10 µm and 50 µm, unregistered vs matched, first comparison
    c = comps[0]; A, B = runs[c["A"]], runs[c["B"]]
    ia = [A["feats"].index(g) for g in genes]; ib = [B["feats"].index(g) for g in genes]; iu = np.triu_indices(len(genes))
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4))
    for m, rf in enumerate((10.0, 48.0)):
        k = int(np.argmin(np.abs(A["r_from"] - rf)))
        la, lb = A["log_g"][k][ia][:, ia][iu], B["log_g"][k][ib][:, ib][iu]
        ok = np.isfinite(la) & np.isfinite(lb) & (A["sd"][k][ia][:, ia][iu] < 0.5) & (B["sd"][k][ib][:, ib][iu] < 0.5)
        axs[2 * m].scatter(la[ok], lb[ok], s=2, alpha=0.3); axs[2 * m].set_title(f"unregistered, whole sections, r_from {rf:g} µm", fontsize=8)
        tab = pd.read_csv(OUT / f"tiles_{res['geneset']}_t{int(res['tile_um'])}.csv"); keys = list(zip(tab[tab.matched].ix, tab[tab.matched].iy))
        iA, iB = tile_index(A, keys), tile_index(B, keys); okt = (iA >= 0) & (iB >= 0)
        Wa = np.ones((1, okt.sum()))
        ga = logg_from_weights(A["ct"][iA[okt]][:, :, ia][:, :, :, ia], A["tt"][iA[okt]][:, ia], Wa, k)[0][iu]
        gb = logg_from_weights(B["ct"][iB[okt]][:, :, ib][:, :, :, ib], B["tt"][iB[okt]][:, ib], Wa, k)[0][iu]
        ok2 = ok & np.isfinite(ga) & np.isfinite(gb)
        axs[2 * m + 1].scatter(ga[ok2], gb[ok2], s=2, alpha=0.3); axs[2 * m + 1].set_title(f"registered, {int(okt.sum())} matched tiles, r_from {rf:g} µm", fontsize=8)
    for ax in axs:
        lim = [-3, 3]; ax.plot(lim, lim, "k-", lw=0.5); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel(f"log g {SECTIONS[c['A']]['short']}"); ax.set_ylabel(f"log g {SECTIONS[c['B']]['short']}")
    fig.suptitle(f"Per-pair log g, {SECTIONS[c['A']]['label']} vs {SECTIONS[c['B']]['label']} ({res['n_genes']} genes)", fontsize=9)
    fig.tight_layout(); fig.savefig(QL / f"ovff_crossplatform_pairs_{tag}.png", dpi=130); plt.close(fig)


def stage_md(a):
    tag = f"{a.geneset}_t{int(a.tile_um)}" + (f"_{SECTIONS[a.pairs_with]['short']}" if a.pairs_with else "")
    write_md(json.load(open(OUT / f"compare_{tag}.json")), OUT / f"compare_{tag}.md"); log("rewrote", OUT / f"compare_{tag}.md")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["genes", "twopoint", "tiles", "compare", "md"])
    ap.add_argument("--geneset", default="all", choices=list(GENESETS)); ap.add_argument("--top-5k", type=int, default=1000)
    ap.add_argument("--genesets", default=",".join(GENESETS), help="genes: the gene sets to build (comma list; Fig. S3 needs 5k only)")
    ap.add_argument("--sections", default="ovff_xenium_5k"); ap.add_argument("--frame", default="ref", choices=["ref", "own"])
    ap.add_argument("--tile-um", type=float, default=500.0); ap.add_argument("--run-tile-um", type=float, default=None, help="tile size of the stored runs (compare aggregates to --tile-um)")
    ap.add_argument("--backend", default="gpu", choices=["cpu", "gpu"], help="twopoint: lattice products and accumulators")
    ap.add_argument("--cpu-mode", default="auto", help="twopoint: product strategy of the CPU backend"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs-with", default=None, help="compare/md: only comparisons involving this section (output tag suffixed with its short name)")
    ap.add_argument("--thin", type=float, default=0.0, help="binomial thinning probability of bin counts (depth ablation)"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--occ-frac", type=float, default=0.3, help="tile occupied if counts >= this x the median of full-tissue tiles")
    ap.add_argument("--min-tissue", type=float, default=0.5); ap.add_argument("--tumor-frac", type=float, default=0.9); ap.add_argument("--stroma-frac", type=float, default=0.5)
    ap.add_argument("--min-region-tiles", type=int, default=6); ap.add_argument("--r-min", type=float, default=4.0); ap.add_argument("--r-max-curve", type=float, default=3000.0)
    ap.add_argument("--min-tile-counts", type=float, default=2e4); ap.add_argument("--min-tiles", type=int, default=20); ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--no-hd3", action="store_true", help="tiles: matched tiles of the four probe-panel sections only (the 3' HD column is "
                    "compared separately with --pairs-with ovff_hd3p_min on the five-section matched tiles)")
    a = ap.parse_args()
    if a.run_tile_um is None: a.run_tile_um = a.tile_um
    {"genes": stage_genes, "twopoint": stage_twopoint, "tiles": stage_tiles, "compare": stage_compare, "md": stage_md}[a.stage](a)


if __name__ == "__main__":
    main()
