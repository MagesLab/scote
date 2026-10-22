#!/usr/bin/env python
"""Second platform set of preprint Supplementary Fig. S2, SpatialBench spleen: WT709 on three platforms (Xenium `sb_xenium_WT709_rep1`, Visium HD `sb_hd_WT709`, MERSCOPE
`sb_merscope_WT709`) and KO168 on two (Xenium `sb_xenium_KO168`, HD `sb_hd_KO168`), all on the 90 genes common to the
three platforms (results/platforms/common_genes_spleen.txt) and the same 2 µm lattice hierarchy (11 levels, 500 µm tiles,
closed-form tile bootstrap; global runs `data/twopoint/<dataset>_bin2um_common.npz` of workflows/twopoint.smk).

Stages (the CRC implementation in scripts/crc_region_local.py is reused: region operator, tile layout, per-tile log g):
  regions    : white-pulp / red-pulp operator on all five sections (marker fractions of the common genes on a 10 µm grid,
               Gaussian σ = 50 µm, z-scored over tissue pixels, argmax) -> data/platforms/regions_spleen_<key>.npz,
               results/platforms/regions_spleen.json, figures/quicklook/spleen_regions.png
  twopoint   : region-matched 2 µm hierarchies for WT709 on the three platforms x {white, red} (--backend gpu on a GPU)
               -> data/twopoint/regions/spleen_<key>_<region>.npz, results/twopoint/regions_spleen_<key>_<region>.json
  compare    : level 1 (global) and level 2 (region-matched) agreement of log g over all gene pairs vs r for
               Xenium-vs-HD, Xenium-vs-MERSCOPE and HD-vs-MERSCOPE (Pearson, RMS, within-2σ; additionally on the pair set
               that passes on ALL three platforms in that bin -> `*_common`) -> results/platforms/spleen_summary.json,
               spleen_regions_table.{json,md}, quick-looks spleen_platforms.png, spleen_region_<r>.png
  local      : level 3 from the per-tile tensors of the global runs (tile medians over 500 µm tiles; global reference on the
               *same* pair set), per-tile log g vs tile white-pulp fraction -> results/platforms/spleen_local.json,
               figures/quicklook/spleen_local.png
  replicates : between-animal comparison WT709 vs KO168 within Xenium and within HD (Pearson between sections, between-
               section SD / within-section bootstrap SD) next to the cross-platform agreement per animal, on every
               2 µm lattice bin from the intra-bin shell [0, 2) µm on (REPLICATE_R_MIN; rows carry r_from and r_central)
               -> results/platforms/spleen_replicates.{json,md}
  all        : compare + local + replicates (the CPU stages after the runs exist)
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
import crc_region_local as RL  # noqa: E402  (CRC implementation: region_operator, label_points, tile_layout, per_tile_log_g)

DATA = REPO / "data"; TP = DATA / "twopoint"; RES = REPO / "results/platforms"; QL = REPO / "figures/quicklook"; RTP = REPO / "results/twopoint"
GENES = RES / "common_genes_spleen.txt"
REGIONS = ["white", "red"]
# Marker sets restricted to the 90 common genes (the panel has no Ms4a1/Cd79a/Hba/Hbb/Klf1): white pulp = B-cell follicle
# and T-cell zone genes; red pulp = red-pulp macrophages (F4/80 = Adgre1, Hmox1 = heme oxygenase of iron-recycling
# macrophages, Cd68, Csf1r), granulocytes (S100a9, Ngp), plasma cells (Jchain), NK cells (Gzma, Ncr1), sinusoidal endothelium (Egfl7).
MARKERS = {"white": ["Cd19", "Ighd", "Cd22", "Cr2", "Fcer2a", "Cd3e", "Cd3d", "Trac", "Cd4", "Cd8a", "Ccl19"],
           "red": ["Adgre1", "Hmox1", "Cd68", "Csf1r", "S100a9", "Ngp", "Jchain", "Gzma", "Ncr1", "Egfl7"]}
SECTIONS = {
    "xenium_WT709": dict(ds="sb_xenium_WT709_rep1", kind="molecules", platform="xenium", animal="WT709", label="Xenium 2 µm bins", res=2.0),
    "hd_WT709": dict(ds="sb_hd_WT709", kind="bins", platform="hd", animal="WT709", label="Visium HD 2 µm", res=2.0),
    "merscope_WT709": dict(ds="sb_merscope_WT709", kind="molecules", platform="merscope", animal="WT709", label="MERSCOPE 2 µm bins", res=2.0),
    "xenium_KO168": dict(ds="sb_xenium_KO168", kind="molecules", platform="xenium", animal="KO168", label="Xenium 2 µm bins", res=2.0),
    "hd_KO168": dict(ds="sb_hd_KO168", kind="bins", platform="hd", animal="KO168", label="Visium HD 2 µm", res=2.0),
}
WT = ["xenium_WT709", "hd_WT709", "merscope_WT709"]
PLATFORM_PAIRS = [("xenium_WT709", "hd_WT709"), ("xenium_WT709", "merscope_WT709"), ("hd_WT709", "merscope_WT709")]
PAIRS_SHOW = "Cd19:Cd3e,Adgre1:Cd19,Cd3d:Cd3e,Cd19:Cd19,Cr2:Cd19,Hmox1:Adgre1,Cd4:Cd8a,Ighd:Cd19"
COLS = {"xenium": "#E69F00", "hd": "#009E73", "merscope": "#0072B2"}
SD_MAX = 0.5


def sd_filter(d):
    """Stored closed-form SD for the pair filters (SD < 0.5 and the like), with the pairs without an interval (reliability
    flag 2, T_eff < 5) set to +inf so that every SD filter excludes them."""
    sd = d["sd"]
    return np.where(d["flag"] == 2, np.inf, sd) if "flag" in d.files else sd


def common_genes():
    return [l.strip() for l in open(GENES) if l.strip() and not l.startswith("#")]


def raw_path(key):
    c = SECTIONS[key]
    return DATA / "processed" / c["ds"] / ("molecules.parquet" if c["kind"] == "molecules" else "bins_square_002um.npz")


def run_path(key):
    return TP / f"{SECTIONS[key]['ds']}_bin2um_common.npz"


def load_units(key):
    """coords (n, 2) µm and X (n, 90) CSR counts of the common genes, exactly as `python -m scote twopoint` selects them
    (molecules of the common genes / in-tissue bins), so that the tile enumeration matches the stored tensors."""
    genes = common_genes(); c = SECTIONS[key]; p = raw_path(key)
    if c["kind"] == "molecules":
        import pyarrow.parquet as pq
        tbl = pq.read_table(p, columns=["x", "y", "feature"])
        feat = tbl["feature"].to_pandas().astype("category")
        m = feat.isin(genes).to_numpy()
        xy = np.stack([tbl["x"].to_numpy()[m], tbl["y"].to_numpy()[m]], 1).astype(np.float64)
        codes = pd.Categorical(np.asarray(feat[m]).astype(str), categories=genes).codes.astype(np.int64)
        return xy, sp.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)), shape=(len(codes), len(genes))), genes
    z = np.load(p, allow_pickle=True)
    X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    feats = pd.Index(z["features"].astype(str)); coords = z["coords_um"].astype(np.float64)
    if "in_tissue" in z:
        keep = z["in_tissue"].astype(bool); X = X[keep]; coords = coords[keep]
    cols = feats.get_indexer(genes); assert (cols >= 0).all()
    return coords, X[:, cols].tocsr().astype(np.float64), genes


def load_run(path, res=2.0):
    d = np.load(path, allow_pickle=True)
    return dict(feats=list(map(str, d["features"])), r=d["r_central"], r_from=d["r_from"], r_to=d["r_to"], lg=d["log_g"],
                lo=d["lo"], hi=d["hi"], sd=sd_filter(d), res=res)


# ----------------------------------------------------------------------------- agreement statistics
def _grid(v, feats):
    idx = [v["feats"].index(f) for f in feats]
    return v["lg"][:, idx][:, :, idx], v["sd"][:, idx][:, :, idx]


def _bin_key(f, t):
    return (round(float(f), 6), round(float(t), 6))


def pass_masks(runs: dict, common):
    """Per run: {bin key: upper-triangle mask of pairs with finite log g and bootstrap SD < SD_MAX}."""
    iu = np.triu_indices(len(common)); out = {}
    for k, v in runs.items():
        lg, sd = _grid(v, common); m = {}
        for i, (f, t) in enumerate(zip(v["r_from"], v["r_to"])):
            m[_bin_key(f, t)] = (np.isfinite(lg[i]) & (sd[i] < SD_MAX))[iu]
        out[k] = m
    return out


def agreement(A, B, common, rmin, common_masks=None):
    """Pairwise agreement of log g over all gene pairs on identical [from, to) bins with r_from >= rmin
    (3 x the coarser resolution). Rows: Pearson, RMS, median |z|, within-2σ fraction on the pairs passing on both runs, and
    (`*_common`) on the pairs passing on every run in `common_masks` (same pair set for all platform pairs)."""
    lgA, sdA = _grid(A, common); lgB, sdB = _grid(B, common)
    iu = np.triu_indices(len(common))
    keyA = {_bin_key(f, t): i for i, (f, t) in enumerate(zip(A["r_from"], A["r_to"]))}
    rows = []
    for j, (f, t) in enumerate(zip(B["r_from"], B["r_to"])):
        kb = _bin_key(f, t)
        if kb not in keyA or f < rmin:
            continue
        i = keyA[kb]
        da, db = lgA[i][iu], lgB[j][iu]; sa, sb = sdA[i][iu], sdB[j][iu]
        ok = np.isfinite(da) & np.isfinite(db) & (sa < SD_MAX) & (sb < SD_MAX)
        if ok.sum() <= 10:
            continue
        z = (da - db) / np.sqrt(sa ** 2 + sb ** 2 + 1e-12)
        row = dict(r_from=float(f), r_to=float(t), n_pairs=int(ok.sum()), pearson=float(np.corrcoef(da[ok], db[ok])[0, 1]),
                   rms_diff=float(np.sqrt(np.mean((da[ok] - db[ok]) ** 2))), median_abs_z=float(np.median(np.abs(z[ok]))),
                   frac_within_2sigma=float(np.mean(np.abs(z[ok]) < 2)), sd_A_median=float(np.median(sa[ok])), sd_B_median=float(np.median(sb[ok])))
        if common_masks is not None:
            okc = ok.copy()
            for m in common_masks.values():
                okc &= m.get(kb, np.zeros_like(okc))
            row.update(n_pairs_common=int(okc.sum()))
            if okc.sum() > 10:
                row.update(pearson_common=float(np.corrcoef(da[okc], db[okc])[0, 1]), rms_common=float(np.sqrt(np.mean((da[okc] - db[okc]) ** 2))),
                           frac_within_2sigma_common=float(np.mean(np.abs(z[okc]) < 2)))
        rows.append(row)
    return rows


def overlay(runs: dict, pairs, out_png, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    common = sorted(set.intersection(*[set(v["feats"]) for v in runs.values()]))
    pairs = [p for p in pairs if all(g in common for g in p)]
    n = len(pairs); nc = 4; nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.4 * nc, 2.9 * nr), sharex=True, squeeze=False)
    for ax, (ga, gb) in zip(axs.ravel(), pairs):
        for key, v in runs.items():
            i, j = v["feats"].index(ga), v["feats"].index(gb); c = COLS[SECTIONS[key]["platform"]]
            ok = (v["r"] >= 3 * v["res"]) & (v["r"] <= 3000) & np.isfinite(v["lg"][:, i, j])
            ax.fill_between(v["r"][ok], v["lo"][ok, i, j], v["hi"][ok, i, j], alpha=0.2, color=c)
            ax.plot(v["r"][ok], v["lg"][ok, i, j], color=c, lw=1.2, label=f"{SECTIONS[key]['label']} ({SECTIONS[key]['animal']})")
        ax.axhline(0, c="k", lw=0.5); ax.set_xscale("log"); ax.set_title(f"{ga} – {gb}", fontsize=9); ax.set_xlabel("r [µm]")
    for ax in axs.ravel()[n:]: ax.axis("off")
    axs[0, 0].set_ylabel("log g_ab(r)"); axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


# ----------------------------------------------------------------------------- stages
def stage_regions(a):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    keys = a.sections.split(",") if a.sections else list(SECTIONS)
    summary = {}
    fig, axs = plt.subplots(len(keys), 3, figsize=(12, 3.6 * len(keys)), squeeze=False)
    for row, key in enumerate(keys):
        t0 = time.time()
        coords, X, genes = load_units(key)
        reg = RL.region_operator(coords, X, genes, markers=MARKERS, regions=REGIONS)
        lab = RL.label_points(coords, reg); tot = np.asarray(X.sum(1)).ravel()
        out = DATA / f"platforms/regions_spleen_{key}.npz"; out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **reg)
        area = {r: float((reg["label"] == i).sum() * (RL.GRID_UM / 1000) ** 2) for i, r in enumerate(REGIONS)}
        summary[key] = dict(dataset=SECTIONS[key]["ds"], n_units=int(len(coords)), total_counts=float(tot.sum()),
                            tissue_area_mm2=float(reg["tissue"].sum() * (RL.GRID_UM / 1000) ** 2), area_mm2=area,
                            frac_counts={r: float(tot[lab == i].sum() / tot.sum()) for i, r in enumerate(REGIONS)},
                            frac_counts_background=float(tot[lab < 0].sum() / tot.sum()), seconds=time.time() - t0)
        print(key, json.dumps(summary[key]), flush=True)
        ext = [reg["origin"][0] / 1000, (reg["origin"][0] + reg["shape"][0] * RL.GRID_UM) / 1000, reg["origin"][1] / 1000, (reg["origin"][1] + reg["shape"][1] * RL.GRID_UM) / 1000]
        for k, r in enumerate(REGIONS):
            axs[row, k].imshow(reg["z"][k].T, origin="lower", extent=ext, cmap="RdBu_r", vmin=-3, vmax=3)
            axs[row, k].set_title(f"{SECTIONS[key]['label']} {SECTIONS[key]['animal']}: z({r} pulp)", fontsize=9)
        cmap = matplotlib.colors.ListedColormap(["white", "#d62728", "#1f77b4"])
        axs[row, 2].imshow((reg["label"].T + 1), origin="lower", extent=ext, cmap=cmap, vmin=-0.5, vmax=2.5, interpolation="nearest")
        axs[row, 2].set_title(f"label (red = white pulp {area['white']:.1f} mm², blue = red pulp {area['red']:.1f} mm²)", fontsize=8)
        for ax in axs[row]: ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    fig.suptitle(f"Spleen region operator: marker fractions smoothed with σ = {RL.SIGMA_UM:g} µm on a {RL.GRID_UM:g} µm grid, z-scored, argmax", fontsize=10)
    fig.tight_layout(); QL.mkdir(parents=True, exist_ok=True); fig.savefig(QL / "spleen_regions.png", dpi=120)
    prev = json.load(open(RES / "regions_spleen.json")) if (RES / "regions_spleen.json").exists() else {}
    prev.update(summary); prev.update(markers=MARKERS, regions=REGIONS, grid_um=RL.GRID_UM, sigma_um=RL.SIGMA_UM, genes_common=len(common_genes()))
    json.dump(prev, open(RES / "regions_spleen.json", "w"), indent=1)


def stage_twopoint(a):
    from scote import CPUOptions, twopoint
    outdir = TP / "regions"; outdir.mkdir(parents=True, exist_ok=True)
    for key in (a.sections.split(",") if a.sections else WT):
        coords, X, genes = load_units(key)
        reg = np.load(DATA / f"platforms/regions_spleen_{key}.npz", allow_pickle=True)
        lab = RL.label_points(coords, reg)
        for ri, r in enumerate(REGIONS):
            if a.regions and r not in a.regions.split(","):
                continue
            out = outdir / f"spleen_{key}_{r}.npz"
            if out.exists() and not a.force:
                print("exists", out); continue
            m = lab == ri; t0 = time.time()
            res = twopoint(coords[m], X[m], names=genes, delta=2.0, bin_size=2.0, tile_um=a.tile_um, n_levels=a.n_levels,
                           backend=CPUOptions(mode=a.cpu_mode) if a.backend == "cpu" else a.backend,
                           keep_tiles=True, verbose=True)
            lo, hi = res.interval()
            np.savez(out, log_g=res.log_g, lo=lo, hi=hi, sd=res.sd, n_eff_pairs=res.n_eff_pairs, P=res.P, totals=res.totals,
                     features=np.array(genes, dtype=str), r_from=res.bins["r_from"].to_numpy(), r_to=res.bins["r_to"].to_numpy(),
                     r_central=res.r, counts_tiles=res.counts_tiles.astype(np.float32), totals_tiles=res.totals_tiles)
            summ = dict(section=key, dataset=SECTIONS[key]["ds"], region=r, n_units=int(m.sum()), total_counts=float(X[m].sum()), n_tiles=res.n_tiles,
                        tile_um=a.tile_um, n_levels=a.n_levels, backend=a.backend, threads=res.params["threads"], timings=res.timings,
                        seconds=time.time() - t0, median_ci_width_log_g=float(np.nanmedian(hi - lo)))
            RTP.mkdir(parents=True, exist_ok=True)
            json.dump(summ, open(RTP / f"regions_spleen_{key}_{r}.json", "w"), indent=1, default=float)
            print(json.dumps({k: v for k, v in summ.items() if k != "timings"}), flush=True)


def stage_compare(a):
    common = common_genes(); pairs = [p.split(":") for p in PAIRS_SHOW.split(",")]
    runs = {k: load_run(run_path(k)) for k in WT if run_path(k).exists()}
    out = dict(common_genes=len(common), sd_max=SD_MAX, sections={k: SECTIONS[k] for k in runs}, levels={})
    masks = pass_masks(runs, common)
    out["levels"]["global"] = {f"{x}|{y}": agreement(runs[x], runs[y], common, 3 * max(runs[x]["res"], runs[y]["res"]), masks)
                               for x, y in PLATFORM_PAIRS if x in runs and y in runs}
    overlay(runs, pairs, QL / "spleen_platforms.png", "SpatialBench spleen WT709: normalized 2-pt functions across platforms (global normalization)")
    for r in REGIONS:
        rr = {k: load_run(TP / f"regions/spleen_{k}_{r}.npz") for k in WT if (TP / f"regions/spleen_{k}_{r}.npz").exists()}
        if len(rr) < 2:
            print("region runs missing for", r); continue
        mk = pass_masks(rr, common)
        out["levels"][r] = {f"{x}|{y}": agreement(rr[x], rr[y], common, 6.0, mk) for x, y in PLATFORM_PAIRS if x in rr and y in rr}
        overlay(rr, pairs, QL / f"spleen_region_{r}.png", f"SpatialBench spleen WT709, {r} pulp only (region-matched normalization)")
    json.dump(out, open(RES / "spleen_summary.json", "w"), indent=1)
    # table: Pearson / RMS / within-2σ (n) per level and platform pair at picked r
    picks = [6, 10, 24, 48, 96, 256, 512, 1024, 2048]
    cols = [(lv, pk) for lv in out["levels"] for pk in out["levels"][lv]]
    md = ["| r_from [µm] | " + " | ".join(f"{lv}: {pk.replace('_WT709', '').replace('|', ' vs ')} Pearson / RMS / within-2σ (n) [common-pair Pearson / RMS (n)]" for lv, pk in cols) + " |",
          "|---|" + "---|" * len(cols)]
    table = []
    for rf in picks:
        rec = dict(r_from_um=rf); cells = [f"{rf}"]
        for lv, pk in cols:
            bins = out["levels"][lv][pk]
            if not bins: cells.append("—"); continue
            b = min(bins, key=lambda b: abs(b["r_from"] - rf))
            if abs(b["r_from"] - rf) > 0.5 * rf: cells.append("—"); continue
            rec[f"{lv}|{pk}"] = {k: b.get(k) for k in ("r_from", "n_pairs", "pearson", "rms_diff", "frac_within_2sigma", "n_pairs_common", "pearson_common", "rms_common")}
            cc = f" [{b['pearson_common']:.3f} / {b['rms_common']:.2f} ({b['n_pairs_common']})]" if "pearson_common" in b else ""
            cells.append(f"{b['pearson']:.3f} / {b['rms_diff']:.2f} / {b['frac_within_2sigma']:.2f} ({b['n_pairs']}){cc}")
        table.append(rec); md.append("| " + " | ".join(cells) + " |")
    json.dump(dict(table=table), open(RES / "spleen_regions_table.json", "w"), indent=1)
    open(RES / "spleen_regions_table.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))


def stage_local(a):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    pairs = [p.split(":") for p in a.pairs.split(",")]
    res = {}
    for key in WT:
        if not run_path(key).exists():
            print("missing run", key); continue
        t0 = time.time()
        d = np.load(run_path(key), allow_pickle=True)
        feats = list(map(str, d["features"])); ct = d["counts_tiles"]; tt = d["totals_tiles"]
        coords, X, genes = load_units(key)
        tiles, ny, origin, tkey = RL.tile_layout(coords, a.tile_um)
        assert len(tiles) == ct.shape[0], (key, len(tiles), ct.shape)
        reg = np.load(DATA / f"platforms/regions_spleen_{key}.npz", allow_pickle=True)
        lab = RL.label_points(coords, reg); tot = np.asarray(X.sum(1)).ravel()
        pos = np.searchsorted(tiles, tkey)
        comp = np.stack([np.bincount(pos, weights=tot * (lab == i), minlength=len(tiles)) for i in range(len(REGIONS))], 1)
        comp = comp / np.maximum(np.bincount(pos, weights=tot, minlength=len(tiles)), 1)[:, None]
        lg = RL.per_tile_log_g(ct, tt, min_tot=a.min_tile_counts)
        res[key] = dict(feats=feats, r=d["r_central"], r_from=d["r_from"], lg=lg, glg=d["log_g"], gsd=d["sd"], comp=comp, n_tiles=len(tiles),
                        tile_counts=tt.sum(1), label=SECTIONS[key]["label"], res=SECTIONS[key]["res"], platform=SECTIONS[key]["platform"])
        print(f"{key}: {len(tiles)} tiles, {int((tt.sum(1) >= a.min_tile_counts).sum())} with >= {a.min_tile_counts:g} counts; per-tile log g in {time.time()-t0:.0f}s", flush=True)
        del ct, d
    common = sorted(set.intersection(*[set(v["feats"]) for v in res.values()]))
    iu = np.triu_indices(len(common))
    dist = {}
    for x, y in PLATFORM_PAIRS:
        if x not in res or y not in res: continue
        A, B = res[x], res[y]
        ia = [A["feats"].index(g) for g in common]; ib = [B["feats"].index(g) for g in common]
        keyA = {round(float(f), 6): i for i, f in enumerate(A["r_from"])}
        rows = []
        for j, f in enumerate(B["r_from"]):
            if round(float(f), 6) not in keyA or f < 3 * max(A["res"], B["res"]): continue
            i = keyA[round(float(f), 6)]
            la = A["lg"][:, i][:, ia][:, :, ia][:, iu[0], iu[1]]; lb = B["lg"][:, j][:, ib][:, :, ib][:, iu[0], iu[1]]
            with np.errstate(all="ignore"):
                medA, medB = np.nanmedian(la, 0), np.nanmedian(lb, 0)
                iqrA = np.nanpercentile(la, 75, 0) - np.nanpercentile(la, 25, 0); iqrB = np.nanpercentile(lb, 75, 0) - np.nanpercentile(lb, 25, 0)
                nA = np.isfinite(la).sum(0); nB = np.isfinite(lb).sum(0)
            ok = (nA >= a.min_tiles) & (nB >= a.min_tiles) & np.isfinite(medA) & np.isfinite(medB)
            if ok.sum() < 10: continue
            gA = A["glg"][i][ia][:, ia][iu]; gB = B["glg"][j][ib][:, ib][iu]
            okg = ok & np.isfinite(gA) & np.isfinite(gB)          # the global reference on the SAME pairs (level-3 comparison)
            rows.append(dict(r_from=float(f), n_pairs=int(ok.sum()), pearson_tile_median=float(np.corrcoef(medA[ok], medB[ok])[0, 1]),
                             rms_tile_median=float(np.sqrt(np.mean((medA[ok] - medB[ok]) ** 2))),
                             pearson_global_same_pairs=float(np.corrcoef(gA[okg], gB[okg])[0, 1]), rms_global_same_pairs=float(np.sqrt(np.mean((gA[okg] - gB[okg]) ** 2))),
                             median_iqr_A=float(np.nanmedian(iqrA[ok])), median_iqr_B=float(np.nanmedian(iqrB[ok])),
                             median_tiles_per_pair_A=float(np.median(nA[ok])), median_tiles_per_pair_B=float(np.median(nB[ok]))))
        dist[f"{x}|{y}"] = rows
    # showcase pairs: quantile bands over tiles and per-tile log g at r ~ r_comp vs the tile's white-pulp fraction
    n = len(pairs); fig, axs = plt.subplots(2, n, figsize=(3.3 * n, 6), squeeze=False)
    pair_rows = {}
    for k, (ga, gb) in enumerate(pairs):
        for key, v in res.items():
            if ga not in v["feats"] or gb not in v["feats"]: continue
            i, j = v["feats"].index(ga), v["feats"].index(gb); c = COLS[v["platform"]]
            ok = v["r"] >= 3 * v["res"]; x = v["lg"][:, :, i, j]
            with np.errstate(all="ignore"):
                q = np.nanpercentile(x, [10, 25, 50, 75, 90], axis=0)
            axs[0, k].fill_between(v["r"][ok], q[0, ok], q[4, ok], color=c, alpha=0.15); axs[0, k].fill_between(v["r"][ok], q[1, ok], q[3, ok], color=c, alpha=0.3)
            axs[0, k].plot(v["r"][ok], q[2, ok], color=c, lw=1.3, label=v["label"])
            kk = int(np.argmin(np.abs(v["r"] - a.r_comp))); y = x[:, kk]; cw = v["comp"][:, 0]; good = np.isfinite(y)
            axs[1, k].scatter(cw[good], y[good], s=9, color=c, alpha=0.6, label=v["label"])
            if good.sum() >= 8:
                rho, pval = spearmanr(cw[good], y[good])
                pair_rows.setdefault(f"{ga}:{gb}", {})[key] = dict(r_um=float(v["r"][kk]), n_tiles=int(good.sum()), spearman_vs_white_frac=float(rho), p=float(pval),
                                                                  tile_median=float(np.nanmedian(y)), tile_iqr=float(np.nanpercentile(y, 75) - np.nanpercentile(y, 25)),
                                                                  tiles=dict(white_frac=np.round(cw[good], 4).tolist(), log_g=np.round(y[good], 4).tolist()),
                                                                  bands=dict(r_um=np.round(v["r"][ok], 3).tolist(), **{f"q{qq}": np.round(q[m, ok], 4).tolist() for m, qq in enumerate((10, 25, 50, 75, 90))}))
        axs[0, k].set_xscale("log"); axs[0, k].axhline(0, c="k", lw=0.5); axs[0, k].set_title(f"{ga} – {gb}", fontsize=9); axs[0, k].set_xlabel("r [µm]")
        axs[1, k].set_xlabel("tile white-pulp fraction"); axs[1, k].set_title(f"per-tile log g at r ≈ {a.r_comp:g} µm", fontsize=8); axs[1, k].axhline(0, c="k", lw=0.5)
    axs[0, 0].set_ylabel("per-tile log g_ab(r): median, 25–75 %, 10–90 % over 500 µm tiles"); axs[0, 0].legend(fontsize=7, frameon=False)
    axs[1, 0].set_ylabel("per-tile log g_ab"); axs[1, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("SpatialBench spleen WT709: local (per-tile) normalized 2-pt functions on three platforms", fontsize=10)
    fig.tight_layout(); fig.savefig(QL / "spleen_local.png", dpi=140)
    out = dict(tile_um=a.tile_um, min_tile_counts=a.min_tile_counts, min_tiles=a.min_tiles, r_comp_um=a.r_comp,
               n_tiles={k: int(v["n_tiles"]) for k, v in res.items()}, tiles_used={k: int((v["tile_counts"] >= a.min_tile_counts).sum()) for k, v in res.items()},
               distribution_agreement=dist, showcase_pairs=pair_rows)
    json.dump(out, open(RES / "spleen_local.json", "w"), indent=1)
    for pk, rows in dist.items():
        print(pk); [print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}) for d in rows if d["r_from"] in (6, 14, 28, 56, 112)]


REPLICATE_R_MIN = 0.0   # µm, lower bound on r_from in the replicates stage (0: from the intra-bin shell on)


def with_center(rows, run):
    """Add r_central (lattice-shell distance of `run`, matched on r_from) to agreement rows."""
    rc = {round(float(f), 6): float(c) for f, c in zip(run["r_from"], run["r"])}
    return [row | dict(r_central=rc[round(row["r_from"], 6)]) for row in rows]


def stage_replicates(a):
    common = common_genes(); iu = np.triu_indices(len(common))
    runs = {k: load_run(run_path(k)) for k in SECTIONS if run_path(k).exists()}
    out = dict(per_animal={}, between_animals={}, sections={k: SECTIONS[k]["ds"] for k in runs})
    for animal in ("WT709", "KO168"):
        x, y = f"xenium_{animal}", f"hd_{animal}"
        if x in runs and y in runs:
            out["per_animal"][f"{x}|{y}"] = with_center(agreement(runs[x], runs[y], common, REPLICATE_R_MIN), runs[x])
    if "xenium_WT709" in runs and "merscope_WT709" in runs:
        out["per_animal"]["xenium_WT709|merscope_WT709"] = with_center(agreement(runs["xenium_WT709"], runs["merscope_WT709"], common, REPLICATE_R_MIN),
                                                                       runs["xenium_WT709"])
    for plat in ("xenium", "hd"):
        ks = [k for k in (f"{plat}_WT709", f"{plat}_KO168") if k in runs]
        if len(ks) < 2: continue
        ref = runs[ks[0]]; rows = []
        for i, rf in enumerate(ref["r_from"]):
            if rf < REPLICATE_R_MIN: continue
            L, S = [], []
            for k in ks:
                v = runs[k]; j = int(np.argmin(np.abs(v["r_from"] - rf)))
                if abs(v["r_from"][j] - rf) > 1e-6: break
                lg, sd = _grid(v, common); L.append(lg[j][iu]); S.append(sd[j][iu])
            if len(L) != len(ks): continue
            L, S = np.array(L), np.array(S)
            ok = np.isfinite(L).all(0) & (S < SD_MAX).all(0)
            if ok.sum() < 50: continue
            between = L[:, ok].std(0, ddof=1); within = np.sqrt((S[:, ok] ** 2).mean(0))
            rows.append(dict(r_from=float(rf), r_central=float(ref["r"][i]), n_pairs=int(ok.sum()), median_between_section_sd=float(np.median(between)),
                             median_within_section_sd=float(np.median(within)), ratio_between_within=float(np.median(between / np.maximum(within, 1e-9))),
                             pearson_between_sections=[float(np.corrcoef(L[0, ok], L[1, ok])[0, 1])],
                             rms_between_sections=float(np.sqrt(np.mean((L[0, ok] - L[1, ok]) ** 2)))))
        out["between_animals"][plat] = dict(sections=ks, rows=rows)
    json.dump(out, open(RES / "spleen_replicates.json", "w"), indent=1)
    md = ["| r_from [µm] | " + " | ".join(f"{k.replace('|', ' vs ')}: Pearson / RMS" for k in out["per_animal"]) + " | " +
          " | ".join(f"{p}: WT709 vs KO168 between-animal SD / within SD (ratio), Pearson, RMS" for p in out["between_animals"]) + " |",
          "|---|" + "---|" * (len(out["per_animal"]) + len(out["between_animals"]))]
    for rf in (0, 2, 4, 6, 12, 24, 48, 96, 256, 512, 1024, 2048):
        cells = [f"{rf}"]
        for k, bins in out["per_animal"].items():
            b = min(bins, key=lambda b: abs(b["r_from"] - rf)) if bins else None
            cells.append(f"{b['pearson']:.3f} / {b['rms_diff']:.2f}" if b and abs(b["r_from"] - rf) <= 0.5 * rf else "—")
        for p, v in out["between_animals"].items():
            b = min(v["rows"], key=lambda b: abs(b["r_from"] - rf)) if v["rows"] else None
            cells.append(f"{b['median_between_section_sd']:.3f} / {b['median_within_section_sd']:.3f} ({b['ratio_between_within']:.1f}), {b['pearson_between_sections'][0]:.2f}, {b['rms_between_sections']:.2f}"
                         if b and abs(b["r_from"] - rf) <= 0.5 * rf else "—")
        md.append("| " + " | ".join(cells) + " |")
    open(RES / "spleen_replicates.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["regions", "twopoint", "compare", "local", "replicates", "all"])
    ap.add_argument("--sections", default=None, help="comma list of section keys (regions/twopoint stages)")
    ap.add_argument("--regions", default=None); ap.add_argument("--force", action="store_true")
    ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"]); ap.add_argument("--cpu-mode", default="auto")
    ap.add_argument("--n-levels", type=int, default=11)
    ap.add_argument("--tile-um", type=float, default=500.0); ap.add_argument("--min-tile-counts", type=float, default=2e4); ap.add_argument("--min-tiles", type=int, default=20)
    ap.add_argument("--r-comp", type=float, default=50.0); ap.add_argument("--pairs", default="Cd19:Cd3e,Adgre1:Cd19,Cd3d:Cd3e,Cd19:Cd19,Cr2:Cd19,Hmox1:Adgre1")
    a = ap.parse_args()
    stages = ["compare", "local", "replicates"] if a.stage == "all" else [a.stage]
    for s in stages:
        t0 = time.time()
        {"regions": stage_regions, "twopoint": stage_twopoint, "compare": stage_compare, "local": stage_local, "replicates": stage_replicates}[s](a)
        print(f"stage {s}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
