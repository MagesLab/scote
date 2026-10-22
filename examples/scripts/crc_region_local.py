#!/usr/bin/env python
"""Oliveira CRC P2 serial sections: normalization levels 2 (region-matched) and 3 (local / per-tile), on top of the global
runs in data/twopoint/{xenium_crc_p2_bin2um,hd_crc_p2_bin2um}_common.npz. scripts/spleen_platforms.py reuses its region operator,
tile layout and per-tile log g.

Stages (run in this order; `twopoint` is the heavy one):
  regions  : platform-independent region operator. Marker-set counts (from the 408 common genes) and total counts of the
             408 genes are binned to a 10 µm grid per platform, smoothed with a Gaussian of σ = 50 µm, turned into
             marker fractions, z-scored over tissue pixels and assigned by argmax to tumor / normal mucosa / stroma+immune
             (background where the smoothed density is < 5 % of the tissue median). Identical operator, parameters and
             marker sets on Xenium molecules and Visium HD 2 µm bins. Output data/platforms/regions_{platform}.npz
             (+ quick-look map, results/platforms/regions_crc_p2.json with area / count fractions).
  twopoint : per platform and region, the lattice-only 2 µm hierarchy (scote.twopoint: 11 levels, 500 µm tiles,
             closed-form tile bootstrap) on the molecules / bins inside that region -> data/twopoint/regions/{platform}_{region}.npz
  compare  : agreement statistics per region (scripts/platform_overlay.py on the two platforms) and a table that puts
             the region-matched numbers next to the global ones -> results/platforms/crc_p2_regions_table.{json,md}
  local    : per-tile log g (500 µm tiles) from the global runs' per-tile count tensors: quantile bands over tiles per
             platform, distribution agreement over all gene pairs per r bin, and per-tile log g vs tile composition
             (tumor fraction from the region grid) for a few pairs -> results/platforms/crc_p2_local.json,
             figures/quicklook/crc_p2_local.png
"""
from __future__ import annotations

import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
GENES = REPO / "results/platforms/common_genes_crc.txt"
REGIONS = ["tumor", "normal", "stroma"]
# marker sets restricted to the 408 genes common to the three platforms (results/platforms/common_genes_crc.txt)
MARKERS = {
    "tumor": ["CEACAM6", "LCN2", "MMP7", "SOX9", "ASCL2", "RNF43", "AXIN2", "LGR5"],                    # CRC epithelium (Wnt targets, tumor-enriched)
    "normal": ["CA1", "CA2", "CA4", "SLC26A3", "SLC26A2", "GUCA2A", "GUCA2B", "AQP8", "SELENBP1", "CLCA1", "ITLN1",
               "PIGR", "BEST4", "OTOP2", "FABP1", "MS4A12", "CEACAM7", "ADH1C", "CA7"],                  # differentiated colonocytes / goblet
    "stroma": ["ACTA2", "DES", "TAGLN", "PDGFRA", "FAP", "CTHRC1", "COL11A1", "IGFBP7", "VWF", "PECAM1", "PLVAP",
               "C1QA", "C1QB", "C1QC", "CD68", "CD3E", "MS4A1", "CD79A", "LTB", "IL7R"],                # fibroblasts, muscle, vessels, immune
}
GRID_UM = 10.0
SIGMA_UM = 50.0
PLATFORMS = {"xenium": dict(kind="molecules", path=DATA / "processed/xenium_crc_p2/molecules.parquet", res=2.0,
                            twopoint=DATA / "twopoint/xenium_crc_p2_bin2um_common.npz", label="Xenium 2 µm bins"),
             "hd": dict(kind="bins", path=DATA / "processed/hd_crc_p2/bins_square_002um.npz", res=2.0,
                        twopoint=DATA / "twopoint/hd_crc_p2_bin2um_common.npz", label="Visium HD 2 µm")}


def common_genes():
    return [l.strip() for l in open(GENES) if l.strip() and not l.startswith("#")]


def load_platform(name):
    """Returns coords (n, 2) µm, X (n, 408) CSR counts (one-hot for molecules), features list."""
    p = PLATFORMS[name]; genes = common_genes()
    if p["kind"] == "molecules":
        import pyarrow.parquet as pq
        tbl = pq.read_table(p["path"], columns=["x", "y", "feature"])
        feat = tbl["feature"].to_pandas().astype("category")
        m = feat.isin(genes).to_numpy()
        xy = np.stack([tbl["x"].to_numpy()[m], tbl["y"].to_numpy()[m]], 1).astype(np.float64)
        codes = pd.Categorical(np.asarray(feat[m]).astype(str), categories=genes).codes.astype(np.int64)
        X = sp.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)), shape=(len(codes), len(genes)))
        return xy, X, genes
    z = np.load(p["path"], allow_pickle=True)
    X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    feats = pd.Index(z["features"].astype(str)); coords = z["coords_um"].astype(np.float64)
    if "in_tissue" in z:
        keep = z["in_tissue"].astype(bool); X = X[keep]; coords = coords[keep]
    cols = feats.get_indexer(genes); assert (cols >= 0).all()
    return coords, X[:, cols].tocsr().astype(np.float64), genes


# ----------------------------------------------------------------------------- region operator
def grid_counts(coords, weights, origin, shape):
    ij = np.floor((coords - origin) / GRID_UM).astype(np.int64)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[1])
    flat = ij[ok, 0] * shape[1] + ij[ok, 1]
    return np.bincount(flat, weights=weights[ok], minlength=shape[0] * shape[1]).reshape(shape)


def region_operator(coords, X, genes, origin=None, shape=None, markers=None, regions=None):
    """Platform-independent region label on a GRID_UM grid (see module docstring). markers/regions default to the CRC
    sets (MARKERS/REGIONS); scripts/spleen_platforms.py passes the white-pulp / red-pulp sets of the SpatialBench spleen."""
    markers = MARKERS if markers is None else markers
    regions = REGIONS if regions is None else regions
    origin = coords.min(0) - GRID_UM if origin is None else origin
    shape = tuple(np.floor((coords.max(0) - origin) / GRID_UM).astype(int) + 2) if shape is None else shape
    gi = {g: i for i, g in enumerate(genes)}
    tot = np.asarray(X.sum(1)).ravel()
    sig = SIGMA_UM / GRID_UM
    dens = gaussian_filter(grid_counts(coords, tot, origin, shape), sig)
    tissue = dens > 0.05 * np.median(dens[dens > 0.01 * dens.max()])
    scores = {}
    for r in regions:
        cols = [gi[g] for g in markers[r] if g in gi]
        w = np.asarray(X[:, cols].sum(1)).ravel()
        frac = gaussian_filter(grid_counts(coords, w, origin, shape), sig) / np.maximum(dens, 1e-9)
        z = (frac - frac[tissue].mean()) / frac[tissue].std()
        scores[r] = z.astype(np.float32)
    Z = np.stack([scores[r] for r in regions])
    label = np.where(tissue, Z.argmax(0), -1).astype(np.int8)
    return dict(origin=origin, shape=np.array(shape), grid_um=GRID_UM, sigma_um=SIGMA_UM, density=dens.astype(np.float32),
                tissue=tissue, label=label, z=Z, regions=np.array(regions, dtype=object),
                markers=np.array(json.dumps(markers), dtype=object))


def label_points(coords, reg):
    ij = np.floor((coords - reg["origin"]) / reg["grid_um"]).astype(np.int64)
    shape = tuple(reg["shape"])
    ij[:, 0] = np.clip(ij[:, 0], 0, shape[0] - 1); ij[:, 1] = np.clip(ij[:, 1], 0, shape[1] - 1)
    return reg["label"][ij[:, 0], ij[:, 1]]


def stage_regions(a):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    summary = {}
    fig, axs = plt.subplots(2, 4, figsize=(16, 7.5))
    for row, name in enumerate(PLATFORMS):
        t0 = time.time()
        coords, X, genes = load_platform(name)
        reg = region_operator(coords, X, genes)
        lab = label_points(coords, reg)
        tot = np.asarray(X.sum(1)).ravel()
        out = DATA / f"platforms/regions_{name}.npz"; out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **reg)
        area = {r: float((reg["label"] == i).sum() * (GRID_UM / 1000) ** 2) for i, r in enumerate(REGIONS)}
        summary[name] = dict(n_units=int(len(coords)), total_counts=float(tot.sum()), tissue_area_mm2=float(reg["tissue"].sum() * (GRID_UM / 1000) ** 2),
                             area_mm2=area, frac_counts={r: float(tot[lab == i].sum() / tot.sum()) for i, r in enumerate(REGIONS)},
                             frac_counts_background=float(tot[lab < 0].sum() / tot.sum()), seconds=time.time() - t0)
        print(name, json.dumps(summary[name]), flush=True)
        ext = [reg["origin"][0] / 1000, (reg["origin"][0] + reg["shape"][0] * GRID_UM) / 1000, reg["origin"][1] / 1000, (reg["origin"][1] + reg["shape"][1] * GRID_UM) / 1000]
        for k, r in enumerate(REGIONS):
            im = axs[row, k].imshow(reg["z"][k].T, origin="lower", extent=ext, cmap="RdBu_r", vmin=-3, vmax=3)
            axs[row, k].set_title(f"{PLATFORMS[name]['label']}: z({r})", fontsize=9)
        cmap = matplotlib.colors.ListedColormap(["white", "#d62728", "#2ca02c", "#1f77b4"])
        axs[row, 3].imshow((reg["label"].T + 1), origin="lower", extent=ext, cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
        axs[row, 3].set_title(f"label (red tumor {area['tumor']:.1f}, green normal {area['normal']:.1f}, blue stroma {area['stroma']:.1f} mm²)", fontsize=8)
        for ax in axs[row]: ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    fig.suptitle(f"Region operator: marker fractions smoothed with σ = {SIGMA_UM:g} µm on a {GRID_UM:g} µm grid, z-scored, argmax", fontsize=10)
    fig.tight_layout(); fig.savefig(REPO / "figures/quicklook/crc_p2_regions.png", dpi=130)
    summary["markers"] = MARKERS; summary["grid_um"] = GRID_UM; summary["sigma_um"] = SIGMA_UM
    json.dump(summary, open(REPO / "results/platforms/regions_crc_p2.json", "w"), indent=1)


# ----------------------------------------------------------------------------- region-matched two-point runs
def stage_twopoint(a):
    from scote import CPUOptions, twopoint
    outdir = DATA / "twopoint/regions"; outdir.mkdir(parents=True, exist_ok=True)
    for name in (a.platforms.split(",") if a.platforms else PLATFORMS):
        coords, X, genes = load_platform(name)
        reg = np.load(DATA / f"platforms/regions_{name}.npz", allow_pickle=True)
        lab = label_points(coords, reg)
        for ri, r in enumerate(REGIONS):
            if a.regions and r not in a.regions.split(","):
                continue
            out = outdir / f"{name}_{r}.npz"
            if out.exists() and not a.force:
                print("exists", out); continue
            m = lab == ri
            t0 = time.time()
            res = twopoint(coords[m], X[m], names=genes, delta=2.0, bin_size=2.0, tile_um=a.tile_um,
                           n_levels=a.n_levels, backend=CPUOptions(mode=a.cpu_mode), keep_tiles=True, verbose=True)
            lo, hi = res.interval()
            np.savez_compressed(out, log_g=res.log_g, lo=lo, hi=hi, sd=res.sd, n_eff_pairs=res.n_eff_pairs, P=res.P,
                                totals=res.totals, features=np.array(genes, dtype=str),
                                r_from=res.bins["r_from"].to_numpy(), r_to=res.bins["r_to"].to_numpy(), r_central=res.r,
                                counts_tiles=res.counts_tiles.astype(np.float32), totals_tiles=res.totals_tiles)
            summ = dict(platform=name, region=r, n_units=int(m.sum()), total_counts=float(X[m].sum()), n_tiles=res.n_tiles, tile_um=a.tile_um,
                        n_levels=a.n_levels, options=res.params["options"], threads=res.params["threads"], timings=res.timings,
                        seconds=time.time() - t0, median_ci_width_log_g=float(np.nanmedian(hi - lo)))
            json.dump(summ, open(REPO / f"results/twopoint/regions_{name}_{r}.json", "w"), indent=1, default=float)
            print(json.dumps({k: v for k, v in summ.items() if k != "timings"}), flush=True)


# ----------------------------------------------------------------------------- compare
def stage_compare(a):
    rows = {}
    for r in REGIONS:
        xb, hd = DATA / f"twopoint/regions/xenium_{r}.npz", DATA / f"twopoint/regions/hd_{r}.npz"
        if not (xb.exists() and hd.exists()):
            print("missing", r); continue
        summ = REPO / f"results/platforms/crc_p2_region_{r}_summary.json"
        subprocess.run([sys.executable, str(REPO / "scripts/platform_overlay.py"), "--xenium-bin", str(xb), "--hd", str(hd),
                        "--out", str(REPO / f"figures/quicklook/crc_p2_region_{r}.png"), "--summary", str(summ)], check=True)
        rows[r] = json.load(open(summ))["comparisons"][0]["bins"]
    rows["global"] = [c for c in json.load(open(REPO / "results/platforms/crc_p2_summary.json"))["comparisons"]
                      if c["A"] == "Xenium 2 µm bins" and c["B"] == "Visium HD 2 µm"][0]["bins"]
    picks = [6, 10, 24, 48, 96, 256, 512, 1024, 2048]
    table = []
    for rf in picks:
        rec = dict(r_from_um=rf)
        for k, bins in rows.items():
            b = min(bins, key=lambda b: abs(b["r_from"] - rf))
            if abs(b["r_from"] - rf) > 0.5 * rf: continue
            rec[k] = dict(r_from=b["r_from"], n_pairs=b["n_pairs"], pearson=round(b["pearson"], 3), rms=round(b["rms_diff"], 3), within2s=round(b["frac_within_2sigma"], 2))
        table.append(rec)
    json.dump(dict(rows=rows, table=table), open(REPO / "results/platforms/crc_p2_regions_table.json", "w"), indent=1)
    md = ["| r_from [µm] | " + " | ".join(f"{k}: Pearson / RMS / within-2σ (n pairs)" for k in rows) + " |", "|---|" + "---|" * len(rows)]
    for rec in table:
        md.append(f"| {rec['r_from_um']} | " + " | ".join((f"{rec[k]['pearson']:.3f} / {rec[k]['rms']:.2f} / {rec[k]['within2s']:.2f} ({rec[k]['n_pairs']})" if k in rec else "—") for k in rows) + " |")
    open(REPO / "results/platforms/crc_p2_regions_table.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


# ----------------------------------------------------------------------------- local (per-tile)
def tile_layout(coords, tile_um):
    """The tile enumeration of scote.twopoint (default origin) -> (tile keys sorted, ny_tiles, origin, key per point)."""
    origin = coords.min(0)
    ny = int(np.floor((coords[:, 1].max() - origin[1]) / tile_um)) + 2
    key = np.floor((coords[:, 0] - origin[0]) / tile_um).astype(np.int64) * ny + np.floor((coords[:, 1] - origin[1]) / tile_um).astype(np.int64)
    return np.unique(key), ny, origin, key


def per_tile_log_g(counts_tiles, totals_tiles, min_tot=1e4, min_pairs=10):
    T = counts_tiles.shape[0]
    out = np.full(counts_tiles.shape, np.nan, dtype=np.float32)
    for t in range(T):
        n = totals_tiles[t]; P = counts_tiles[t]
        if n.sum() < min_tot: continue
        ref = P.sum(axis=(-1, -2))[:, None, None] * np.outer(n, n)[None] / max(n.sum(), 1) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.log(P / ref)
        lg[P < min_pairs] = np.nan
        out[t] = lg
    return out


def stage_local(a):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pairs = [p.split(":") for p in a.pairs.split(",")]
    res = {}
    for name in PLATFORMS:
        t0 = time.time()
        d = np.load(PLATFORMS[name]["twopoint"], allow_pickle=True)
        feats = list(map(str, d["features"])); r = d["r_central"]; r_from = d["r_from"]
        ct = d["counts_tiles"]; tt = d["totals_tiles"]
        coords, X, genes = load_platform(name)
        tiles, ny, origin, key = tile_layout(coords, a.tile_um)
        assert len(tiles) == ct.shape[0], (len(tiles), ct.shape)
        reg = np.load(DATA / f"platforms/regions_{name}.npz", allow_pickle=True)
        lab = label_points(coords, reg); tot = np.asarray(X.sum(1)).ravel()
        pos = np.searchsorted(tiles, key)
        comp = np.stack([np.bincount(pos, weights=tot * (lab == i), minlength=len(tiles)) for i in range(len(REGIONS))], 1)
        comp = comp / np.maximum(np.bincount(pos, weights=tot, minlength=len(tiles)), 1)[:, None]
        lg = per_tile_log_g(ct, tt, min_tot=a.min_tile_counts)
        res[name] = dict(feats=feats, r=r, r_from=r_from, lg=lg, glg=d["log_g"], comp=comp, tiles=tiles, ny=ny, n_tiles=len(tiles),
                         tile_counts=tt.sum(1), label=PLATFORMS[name]["label"], res=PLATFORMS[name]["res"])
        print(f"{name}: {len(tiles)} tiles, per-tile log g in {time.time()-t0:.0f}s; tiles with >= {a.min_tile_counts:g} counts: {int(np.isfinite(lg[:, 0, 0, 0]).sum() if False else (tt.sum(1) >= a.min_tile_counts).sum())}", flush=True)
        del ct, d
    common = sorted(set(res["xenium"]["feats"]) & set(res["hd"]["feats"]))
    # --- distribution agreement over all pairs per r bin: tile-median and tile-IQR of log g, per platform
    A, B = res["xenium"], res["hd"]
    ia = [A["feats"].index(g) for g in common]; ib = [B["feats"].index(g) for g in common]
    keyA = {round(f, 6): i for i, f in enumerate(A["r_from"])}
    iu = np.triu_indices(len(common))
    dist_rows = []
    for j, f in enumerate(B["r_from"]):
        if round(f, 6) not in keyA or f < 3 * max(A["res"], B["res"]): continue
        i = keyA[round(f, 6)]
        la = A["lg"][:, i][:, ia][:, :, ia][:, iu[0], iu[1]]; lb = B["lg"][:, j][:, ib][:, :, ib][:, iu[0], iu[1]]   # (T, n_pairs)
        with np.errstate(all="ignore"):
            medA, medB = np.nanmedian(la, 0), np.nanmedian(lb, 0)
            iqrA = np.nanpercentile(la, 75, 0) - np.nanpercentile(la, 25, 0); iqrB = np.nanpercentile(lb, 75, 0) - np.nanpercentile(lb, 25, 0)
            nA = np.isfinite(la).sum(0); nB = np.isfinite(lb).sum(0)
        ok = (nA >= a.min_tiles) & (nB >= a.min_tiles) & np.isfinite(medA) & np.isfinite(medB)
        if ok.sum() < 10: continue
        gA = A["glg"][i][ia][:, ia][iu]; gB = B["glg"][j][ib][:, ib][iu]   # global log g of the same pairs (fairness reference)
        okg = ok & np.isfinite(gA) & np.isfinite(gB)
        dist_rows.append(dict(r_from=float(f), n_pairs=int(ok.sum()), pearson_tile_median=float(np.corrcoef(medA[ok], medB[ok])[0, 1]),
                              rms_tile_median=float(np.sqrt(np.mean((medA[ok] - medB[ok]) ** 2))),
                              pearson_global_same_pairs=float(np.corrcoef(gA[okg], gB[okg])[0, 1]), rms_global_same_pairs=float(np.sqrt(np.mean((gA[okg] - gB[okg]) ** 2))),
                              median_iqr_xenium=float(np.nanmedian(iqrA[ok])), median_iqr_hd=float(np.nanmedian(iqrB[ok])),
                              median_tiles_per_pair_xenium=float(np.median(nA[ok])), median_tiles_per_pair_hd=float(np.median(nB[ok]))))
    # --- showcase pairs: quantile bands over tiles + per-tile log g vs tumor fraction
    n = len(pairs); fig, axs = plt.subplots(2, n, figsize=(3.3 * n, 6), squeeze=False)
    cols = {"xenium": "C1", "hd": "C2"}
    pair_rows = {}
    for k, (ga, gb) in enumerate(pairs):
        for name, v in res.items():
            if ga not in v["feats"] or gb not in v["feats"]: continue
            i, j = v["feats"].index(ga), v["feats"].index(gb)
            ok = v["r"] >= 3 * v["res"]
            x = v["lg"][:, :, i, j]   # (T, K)
            with np.errstate(all="ignore"):
                q = np.nanpercentile(x, [10, 25, 50, 75, 90], axis=0)
            axs[0, k].fill_between(v["r"][ok], q[0, ok], q[4, ok], color=cols[name], alpha=0.15)
            axs[0, k].fill_between(v["r"][ok], q[1, ok], q[3, ok], color=cols[name], alpha=0.3)
            axs[0, k].plot(v["r"][ok], q[2, ok], color=cols[name], lw=1.3, label=v["label"])
            # composition relation at one scale (closest r bin to --r-comp)
            kk = int(np.argmin(np.abs(v["r"] - a.r_comp)))
            y = x[:, kk]; c = v["comp"][:, 0]; good = np.isfinite(y)
            axs[1, k].scatter(c[good], y[good], s=9, color=cols[name], alpha=0.6, label=v["label"])
            if good.sum() >= 8:
                from scipy.stats import spearmanr
                rho, pval = spearmanr(c[good], y[good])
                pair_rows.setdefault(f"{ga}:{gb}", {})[name] = dict(r_um=float(v["r"][kk]), n_tiles=int(good.sum()), spearman_vs_tumor_frac=float(rho), p=float(pval),
                                                                   tile_median=float(np.nanmedian(y)), tile_iqr=float(np.nanpercentile(y, 75) - np.nanpercentile(y, 25)),
                                                                   # per-tile values (small) so that figures/make_fig2.py can draw the scatter and the quantile bands from results/
                                                                   tiles=dict(tumor_frac=np.round(c[good], 4).tolist(), log_g=np.round(y[good], 4).tolist()),
                                                                   bands=dict(r_um=np.round(v["r"][ok], 3).tolist(), **{f"q{qq}": np.round(q[m, ok], 4).tolist() for m, qq in enumerate((10, 25, 50, 75, 90))}))
        axs[0, k].set_xscale("log"); axs[0, k].axhline(0, c="k", lw=0.5); axs[0, k].set_title(f"{ga} – {gb}", fontsize=9); axs[0, k].set_xlabel("r [µm]")
        axs[1, k].set_xlabel("tile tumor fraction (region operator)"); axs[1, k].set_title(f"per-tile log g at r ≈ {a.r_comp:g} µm", fontsize=8); axs[1, k].axhline(0, c="k", lw=0.5)
    axs[0, 0].set_ylabel("per-tile log g_ab(r): median, 25–75 %, 10–90 % over 500 µm tiles"); axs[0, 0].legend(fontsize=7, frameon=False)
    axs[1, 0].set_ylabel("per-tile log g_ab"); axs[1, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("CRC P2: local (per-tile) normalized 2-pt functions, Xenium vs Visium HD", fontsize=10)
    fig.tight_layout(); fig.savefig(REPO / "figures/quicklook/crc_p2_local.png", dpi=140)
    out = dict(tile_um=a.tile_um, min_tile_counts=a.min_tile_counts, min_tiles=a.min_tiles, r_comp_um=a.r_comp,
               n_tiles={k: int(v["n_tiles"]) for k, v in res.items()}, tiles_used={k: int((v["tile_counts"] >= a.min_tile_counts).sum()) for k, v in res.items()},
               distribution_agreement=dist_rows, showcase_pairs=pair_rows)
    json.dump(out, open(REPO / "results/platforms/crc_p2_local.json", "w"), indent=1)
    for d in dist_rows: print(d)
    print(json.dumps(pair_rows, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["regions", "twopoint", "compare", "local"])
    ap.add_argument("--platforms", default=None); ap.add_argument("--regions", default=None); ap.add_argument("--force", action="store_true")
    ap.add_argument("--cpu-mode", default="auto"); ap.add_argument("--n-levels", type=int, default=11)
    ap.add_argument("--tile-um", type=float, default=500.0); ap.add_argument("--min-tile-counts", type=float, default=2e4); ap.add_argument("--min-tiles", type=int, default=20)
    ap.add_argument("--r-comp", type=float, default=50.0)
    ap.add_argument("--pairs", default="CEACAM5:CD3E,MS4A1:CD3E,ACTA2:CEACAM5,PECAM1:VWF,CD3E:CD3E,CEACAM5:CEACAM5")
    a = ap.parse_args()
    {"regions": stage_regions, "twopoint": stage_twopoint, "compare": stage_compare, "local": stage_local}[a.stage](a)


if __name__ == "__main__":
    main()
