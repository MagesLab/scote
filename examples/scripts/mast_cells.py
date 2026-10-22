#!/usr/bin/env python
"""Preprint Fig. 5e, f: mast cells as the "missing" cell type (the mast-cell genes are the Xenium-vs-HD tumor/stroma
density-ratio outliers of scripts/platform_artifacts.py).

Mast cells are rare, small, scattered, granule-rich cells that segmentation- and cluster-based pipelines often fail to recover
as a type. The molecule-level n-point functions see them without segmentation or clustering. Five stages on the Oliveira CRC
P2 serial sections (Xenium molecules / Visium HD 2 µm bins), P1/P5 Xenium+HD for replication where the stored tensors allow:

  twopoint   : from the stored per-tile two-point tensors (data/twopoint/*_crc_p2_bin2um_common.npz, 408 genes, 500 µm tiles):
               (i) gene-set inventory (presence / abundance of the mast candidates on each platform, XCL2 as the NK/T control),
               (ii) rank of every mast pair among all 408·407/2 gene pairs by log g at the contact scale (2–12 µm) on both
               platforms, (iii) the mast block's pooled log g ± tile-bootstrap error vs r (mast–mast cross, mast self,
               mast × tumor / smooth muscle / fibroblast / endothelium / T / macrophage / B, the NK block as the rare-type
               reference), Xenium/HD agreement for the mast pairs, (iv) per-gene block membership (mean log g with the four
               strongest mast genes), (v) the molecule-level (Δ = 0.5 µm) view below 2 µm and the P1/P5 replication.
  cells      : the Xenium segmentation view (data/raw/xenium_crc_p2/{transcripts,cells}.parquet, cell_feature_matrix.h5):
               fraction of mast transcripts inside cells vs the panel, cells with ≥ 1/2/3/5 mast transcripts, the share of
               in-cell mast transcripts sitting in cells whose mast fraction is < 10 % (mis-assignment to neighbors), a
               standard scanpy pipeline (filter ≥ 10 counts, normalize_total, log1p, PCA 30, kNN 15, leiden 0.5/1/2) and
               whether any cluster is mast-dominated (purity / recall / size), molecule-level DBSCAN spots (eps 5 µm, ≥ 3
               molecules) with density per mm² / transcripts per spot / diameter, and the 8 µm-bin analogue on Xenium.
  hd         : the Visium HD view: 2 µm-bin DBSCAN spots, 8 µm bins with ≥ 2/3/5 mast UMIs, mis-assignment share at 8 µm,
               leiden on 8 µm bins (2000 HVGs of the whole transcriptome, 200k-bin subsample).
  isolation  : molecule-level co-marker test for every candidate gene (fraction of its molecules without another CORE7 molecule
               within 5 / 10 µm, per region, Xenium P1/P2/P5 and HD P2) -> non-mast fraction b_iso relative to the CPA3/KIT floor.
  background : is the outlier status a position effect (mast cells at the tumor–stroma interface, where the serial-section
               region masks disagree) or a detection effect? (a) region-density ratios recomputed on region cores (≥ 50 µm
               from any boundary) for all 408 genes, (b) distance-to-boundary distribution of mast molecules vs all molecules,
               (c) the pixel co-marker intercept test of platform_artifacts.py applied to each mast gene with the four clean
               mast genes as co-markers (tumor + stroma, both platforms, 500 µm tile bootstrap), (d) the uniform-background
               compression prediction vs the observed Xenium−HD difference inside the tumor region.
  table      : collects the JSONs into results/artifacts/mastcells_table.md ("how mast cells appear in each representation").
Outputs: results/artifacts/mastcells_{twopoint,cells,hd,background,isolation}.json (small), results/artifacts/mastcells_table.md,
figures/quicklook/artifacts_mast_*.png; wall / peak RSS via scripts/timed_run.py (logs/artifacts/mastcells.timing.jsonl).
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "results/artifacts"; OUT.mkdir(exist_ok=True, parents=True)
QL = REPO / "figures/quicklook"; QL.mkdir(exist_ok=True, parents=True)
REGIONS = ["tumor", "normal", "stroma"]
TILE_UM = 500.0
# mast-cell candidates present in the 422-gene CRC panel and in the 408 common genes (TPSAB1 / TPSB2 are not on the panel)
CAND9 = ["CPA3", "KIT", "MS4A2", "HDC", "CMA1", "CTSG", "HPGDS", "IL1RL1", "GATA2"]
CORE7 = ["CPA3", "KIT", "MS4A2", "CTSG", "HPGDS", "IL1RL1", "GATA2"]   # tight block on Xenium (twopoint stage, membership table)
CORE4 = ["CPA3", "KIT", "MS4A2", "CTSG"]                              # co-marker set for the pixel intercept test
NOTMAST = ["XCL2"]                                                     # NK / T chemokine; on the density-ratio outlier list, tested here
MARKERS = {"tumor": ["CEACAM5", "CEACAM6"], "smooth_muscle": ["ACTA2", "TAGLN", "DES"], "fibroblast": ["PDGFRA", "FAP"],
           "endothelium": ["PECAM1", "VWF"], "T_cell": ["CD3E", "CD2"], "macrophage": ["CD68", "C1QA"], "B_cell": ["MS4A1", "CD79A"],
           "NK": ["CCL5", "KLRD1", "GNLY", "NKG7"]}
TWOPOINT = {"xenium": DATA / "twopoint/xenium_crc_p2_bin2um_common.npz", "hd": DATA / "twopoint/hd_crc_p2_bin2um_common.npz"}
TWOPOINT_MOL = DATA / "twopoint/xenium_crc_p2_mol_common.npz"
REPL = {("xenium", p): DATA / f"twopoint/xenium_crc_{p}_bin2um_common.npz" for p in ("p1", "p5")}
REPL.update({("hd", p): DATA / f"twopoint/hd_crc_{p}_bin2um_common.npz" for p in ("p1", "p5")})


def jdump(obj, path):
    json.dump(obj, open(path, "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))


def region_grid(platform):
    reg = {k: np.load(DATA / f"platforms/regions_{platform}.npz", allow_pickle=True)[k] for k in ("origin", "shape", "grid_um", "label")}
    reg["grid_um"] = float(reg["grid_um"]); reg["shape"] = tuple(int(s) for s in reg["shape"])
    return reg


def region_areas(reg):
    a = {r: float((reg["label"] == i).sum() * (reg["grid_um"] / 1000) ** 2) for i, r in enumerate(REGIONS)}
    a["tissue"] = float((reg["label"] >= 0).sum() * (reg["grid_um"] / 1000) ** 2)
    return a


def label_grid(coords, reg):
    """Region label per point (−1 tissue-free, −2 outside the grid) and the flat 10 µm pixel index (−1 outside)."""
    ij = np.floor((coords - reg["origin"]) / reg["grid_um"]).astype(np.int64)
    sh = reg["shape"]
    inside = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < sh[0]) & (ij[:, 1] < sh[1])
    lab = np.full(len(coords), -2, dtype=np.int8)
    lab[inside] = reg["label"][ij[inside, 0], ij[inside, 1]]
    pix = np.where(inside, ij[:, 0] * sh[1] + ij[:, 1], -1)
    return lab, pix


def tile_keys(xy, tile_um=TILE_UM):
    o = xy.min(0)
    k = np.floor((xy[:, 0] - o[0]) / tile_um).astype(np.int64) * 100000 + np.floor((xy[:, 1] - o[1]) / tile_um).astype(np.int64)
    u, inv = np.unique(k, return_inverse=True)
    return inv, len(u)


def tile_boot(stat, tinv, n_tiles, n_boot=None, seed=None):
    """Block bootstrap over tiles of a statistic stat(w) where w = per-point weight (tile multiplicity), in closed form
    (B -> infinity, linearized; platform_artifacts.closed_form_tile_stat). Returns (base, sd, lo, hi) with lo / hi =
    base -+ 1.96 sd. n_boot, seed: ignored (kept for the call signature)."""
    from platform_artifacts import closed_form_tile_stat
    return closed_form_tile_stat(stat, tinv, n_tiles)


def load_meta(path, keys=("features", "r_central", "r_from", "r_to", "log_g", "sd", "P", "totals", "totals_tiles")):
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in keys if k in z.files}
    d["features"] = list(map(str, d["features"]))
    return d, z


# ---------------------------------------------------------------------------------------------------------------- twopoint
def gene_inventory():
    inv = {}
    common = [l.strip() for l in open(REPO / "results/platforms/common_genes_crc.txt") if l.strip() and not l.startswith("#")]
    hd_tot = {}
    for p in ("p1", "p2", "p5"):
        f = DATA / f"twopoint/hd_crc_{p}_bin2um_common.npz"
        if f.exists():
            z = np.load(f, allow_pickle=True); ft = list(map(str, z["features"])); tot = z["totals"]
            hd_tot[p] = dict(zip(ft, tot.astype(int).tolist()))
    for g in CAND9 + NOTMAST + ["TPSAB1", "TPSB2"]:
        row = dict(in_common_408=g in common, xenium={}, hd={})
        for p in ("p1", "p2", "p5"):
            fc = json.load(open(DATA / f"processed/xenium_crc_{p}/feature_counts.json"))
            vals = np.array(sorted(fc.values()))
            row["xenium"][p] = dict(n=int(fc.get(g, 0)), in_panel=g in fc, panel_percentile=float((vals < fc[g]).mean()) if g in fc else None)
            if p in hd_tot and g in hd_tot[p]:
                v = np.array(sorted(hd_tot[p].values()))
                row["hd"][p] = dict(n=int(hd_tot[p][g]), percentile_408=float((v < hd_tot[p][g]).mean()))
        inv[g] = row
    return inv


def pair_ranks(meta, pairs_named, r_max=12.0, min_pairs=10, n_top=40):
    """Rank of the named pairs among all cross pairs (upper triangle, P >= min_pairs) by log g, per bin with r <= r_max."""
    f = meta["features"]; G = len(f); lg = meta["log_g"]; P = meta["P"]; r = meta["r_central"]
    iu = np.triu_indices(G, 1)
    flat = {(min(i, j), max(i, j)): n for n, (i, j) in enumerate(zip(iu[0], iu[1]))}
    out = []
    for k in range(len(r)):
        if r[k] > r_max: continue
        v = lg[k][iu]; ok = np.isfinite(v) & (P[k][iu] >= min_pairs)
        vv = np.where(ok, v, -np.inf); order = np.argsort(-vv); rank = np.empty(len(vv), int); rank[order] = np.arange(1, len(vv) + 1)
        n_ok = int(ok.sum())
        rows = {}
        for name, plist in pairs_named.items():
            rk = []
            for a, b in plist:
                if a in f and b in f:
                    ia, ib = f.index(a), f.index(b)
                    if ia == ib: continue
                    n = flat[(min(ia, ib), max(ia, ib))]
                    rk.append(dict(pair=f"{a}:{b}", rank=int(rank[n]) if ok[n] else None, log_g=float(v[n]) if ok[n] else None, percentile=float(1 - (rank[n] - 1) / n_ok) if ok[n] else None))
            valid = [x["rank"] for x in rk if x["rank"] is not None]
            rows[name] = dict(pairs=rk, n_valid=len(valid), median_rank=float(np.median(valid)) if valid else None, max_rank=int(max(valid)) if valid else None,
                              n_in_top100=int(sum(x <= 100 for x in valid)), n_in_top1pct=int(sum(x <= 0.01 * n_ok for x in valid)), n_in_top01pct=int(sum(x <= 0.001 * n_ok for x in valid)))
        top = [dict(pair=f"{f[iu[0][o]]}:{f[iu[1][o]]}", log_g=float(vv[o])) for o in order[:n_top]]
        mastset = set(CAND9)
        for t in top:
            a, b = t["pair"].split(":"); t["mast_pair"] = (a in mastset) and (b in mastset)
        out.append(dict(k=int(k), r_um=float(r[k]), n_pairs_ranked=n_ok, sets=rows, top=top, n_mast_in_top40=int(sum(t["mast_pair"] for t in top))))
    return out


def block_bootstrap(z, meta, sets, n_boot=None, seed=None):
    """Pooled and mean-of-log log g for named pair sets vs r, with the closed-form block bootstrap over the stored 500 µm
    tiles (B -> infinity, delta method). sets: name -> list of (gene_a, gene_b). Loads counts_tiles once (13 GB for Xenium),
    keeps the rows of every gene of a set. Uncertainty per shell from the
    per-tile influence: pooled log g as scote.uncertainty.pooled_log_g (symmetrized slabs; checked against it on one set),
    l_t = S_t / S - Q_t / Q - sum_p w_p (r_t(a_p) + r_t(b_p)) + 2 m_t / n; mean of log g over the pairs with counts: the
    mean of the per-pair influences l_t(ab) (scote.uncertainty.influence_log_g). lo / hi = estimate -+ 1.96 sd; no interval
    (NaN) where T_eff = S^2 / sum_t S_t^2 < 5 (reliability flag 2). n_boot, seed: ignored."""
    from scote import uncertainty as TV
    f = meta["features"]; G = len(f)
    rows_needed = sorted({f.index(g) for plist in sets.values() for a, b in plist if a in f and b in f for g in (a, b)})
    ridx = {g: i for i, g in enumerate(rows_needed)}
    t0 = time.time()
    ct = z["counts_tiles"]                        # (T, K, G, G)
    T, K = ct.shape[:2]
    Ptot_tiles = ct.sum(axis=(2, 3))              # (T, K)
    sub = np.ascontiguousarray(ct[:, :, rows_needed, :], dtype=np.float64)   # (T, K, M, G)
    tt = meta["totals_tiles"].astype(np.float64)  # (T, G)
    chk_set = "mast7_cross"
    pa_chk = [f.index(a) for a, b in sets[chk_set]]; pb_chk = [f.index(b) for a, b in sets[chk_set]]
    # reference on a few populated shells (pooled_log_g divides by the shell total, which is 0 for empty shells)
    k_chk = [k for k in np.nonzero(Ptot_tiles.sum(0) > 0)[0]][:: max(1, K // 8)][:8]
    ref_chk = TV.pooled_log_g(ct[:, k_chk], tt, pa_chk, pb_chk)
    del ct
    print(f"  counts_tiles loaded and reduced: T={T}, K={K}, rows={len(rows_needed)} in {time.time()-t0:.0f} s", flush=True)
    # sanity: full-tile sum must reproduce the stored log_g for one pair
    P = sub.sum(0); n = tt.sum(0); ref = Ptot_tiles.sum(0)[:, None, None] * n[rows_needed][None, :, None] * n[None, None, :] / n.sum() ** 2
    a0 = rows_needed[0]
    with np.errstate(all="ignore"):
        chk = np.log(P[:, ridx[a0], :] / ref[:, ridx[a0], :]) - meta["log_g"][:, a0, :]
    print(f"  reproduction of stored log_g (row {f[a0]}): max |diff| = {np.nanmax(np.abs(chk)):.2e}", flush=True)
    pair_idx = {name: ([ridx[f.index(a)] for a, b in plist if a in f and b in f], [f.index(b) for a, b in plist if a in f and b in f]) for name, plist in sets.items()}

    def stat(w):
        Pb = np.tensordot(w, sub, axes=(0, 0)); Pt = w @ Ptot_tiles; nb = w @ tt; ntot = nb.sum()
        refb = Pt[:, None, None] * nb[rows_needed][None, :, None] * nb[None, None, :] / ntot ** 2
        res = []
        for name, (ii, jj) in pair_idx.items():
            Pp = Pb[:, ii, jj]; Rp = refb[:, ii, jj]
            with np.errstate(all="ignore"):
                pooled = np.log(Pp.sum(1) / Rp.sum(1))
                lgp = np.log(Pp / Rp); lgp[Pp <= 0] = np.nan
                meanlog = np.nanmean(lgp, 1)
            res.append(np.stack([pooled, meanlog]))
        return np.stack(res)   # (n_sets, 2, K)

    base = stat(np.ones(T))
    # closed form: per-tile influence of the pooled and of the mean-of-log statistic
    n_a = tt.sum(0); ntot = float(n_a.sum()); m_t = tt.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Rt = np.where(n_a > 0, tt / n_a, 0.0)
    Qt = Ptot_tiles.astype(np.float64); Q = Qt.sum(0)                       # (T, K), (K,)
    z95 = 1.959963984540054
    out = {}
    for s_, name in enumerate(pair_idx):
        ii, jj = pair_idx[name]
        ga = np.array([rows_needed[i] for i in ii]); gb = np.array(jj)
        Pt_ab = sub[:, :, ii, jj]                                           # (T, K, np) raw slabs
        Pt_ba = sub[:, :, [ridx[b] for b in gb], ga]                        # P_t(b, a)
        Ps = 0.5 * (Pt_ab + Pt_ba)                                          # symmetrized
        St = Ps.sum(2); S = St.sum(0)
        nn = n_a[ga] * n_a[gb]; wgt = nn / nn.sum()
        v = -(Rt[:, ga] @ wgt + Rt[:, gb] @ wgt) + 2.0 * m_t / ntot           # (T,)
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = St / S - Qt / Q + v[:, None]                                # (T, K)
            sd_p = np.sqrt((lp ** 2).sum(0)); sd_p[S <= 0] = np.nan
            t_eff = S ** 2 / (St ** 2).sum(0)
            lpair = (Ps / Ps.sum(0) - (Qt / Q)[:, :, None] - Rt[:, ga][:, None, :] - Rt[:, gb][:, None, :]
                     + (2.0 * m_t / ntot)[:, None, None])                    # (T, K, np)
        valid = (Pt_ab.sum(0) > 0)                                           # pairs entering the mean (as lgp[Pp <= 0] = nan)
        lm = np.where(valid[None], lpair, 0.0).sum(2) / np.maximum(valid.sum(1), 1)[None, :]
        sd_m = np.sqrt((lm ** 2).sum(0)); sd_m[valid.sum(1) == 0] = np.nan
        if name == chk_set:
            okk = np.isfinite(sd_p[k_chk]) & np.isfinite(ref_chk["sd"])
            assert okk.any() and np.allclose(sd_p[k_chk][okk], ref_chk["sd"][okk], rtol=1e-8), "pooled SD differs from scote.uncertainty.pooled_log_g"
            assert np.allclose(base[s_, 0][k_chk][okk], ref_chk["log_g"][okk], rtol=1e-10, atol=1e-12), "pooled log g differs"
            print(f"  check against scote.uncertainty.pooled_log_g on shells {k_chk}: max rel SD diff {np.max(np.abs(sd_p[k_chk][okk] / ref_chk['sd'][okk] - 1)):.1e}", flush=True)
        flag = TV.reliability_flag(t_eff)
        lo = base[s_, 0] - z95 * sd_p; hi = base[s_, 0] + z95 * sd_p; lo[flag == 2] = np.nan; hi[flag == 2] = np.nan
        out[name] = dict(n_pairs=len(ii), pooled_log_g=base[s_, 0].tolist(), pooled_sd=sd_p.tolist(), pooled_lo=lo.tolist(), pooled_hi=hi.tolist(),
                         pooled_t_eff=t_eff.tolist(), pooled_flag=flag.astype(int).tolist(),
                         meanlog_log_g=base[s_, 1].tolist(), meanlog_sd=sd_m.tolist(),
                         pair_counts=float(P[:, ii, jj].sum()))
    print(f"  closed-form tile bootstrap in {time.time()-t0:.0f} s", flush=True)
    return out, dict(n_tiles=int(T), uncertainty="closed", seconds=time.time() - t0)


def pair_sets(f):
    sets = {"mast7_cross": [(a, b) for i, a in enumerate(CORE7) for b in CORE7[i + 1:]], "mast7_self": [(a, a) for a in CORE7],
            "mast9_cross": [(a, b) for i, a in enumerate(CAND9) for b in CAND9[i + 1:]], "mast9_self": [(a, a) for a in CAND9],
            "mast4_cross": [(a, b) for i, a in enumerate(CORE4) for b in CORE4[i + 1:]]}
    for name, ms in MARKERS.items():
        sets[f"mast7_x_{name}"] = [(a, b) for a in CORE7 for b in ms if b in f]
    sets["NK_cross"] = [(a, b) for i, a in enumerate(MARKERS["NK"]) for b in MARKERS["NK"][i + 1:]]
    sets["T_x_tumor"] = [(a, b) for a in MARKERS["T_cell"] for b in MARKERS["tumor"] if a in f and b in f]
    sets["tumor_self"] = [("CEACAM5", "CEACAM5")]
    for g in CAND9 + NOTMAST:            # membership: g against the four clean mast genes (excluding itself)
        sets[f"member_{g}"] = [(g, b) for b in CORE4 if b != g]
    sets["xcl2_x_NK"] = [("XCL2", b) for b in MARKERS["NK"]]
    sets["xcl2_self"] = [("XCL2", "XCL2")]
    return {k: [(a, b) for a, b in v if a in f and b in f] for k, v in sets.items()}


def stage_twopoint(a):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t_all = time.time()
    out = dict(gene_sets=dict(candidates=CAND9, core7=CORE7, core4=CORE4, not_mast=NOTMAST, absent_from_panel=["TPSAB1", "TPSB2"]), inventory=gene_inventory(), platforms={})
    metas = {}
    for plat, path in TWOPOINT.items():
        print(f"[{plat}] loading {path.name}", flush=True)
        meta, z = load_meta(path); metas[plat] = meta
        f = meta["features"]; sets = pair_sets(f)
        ranks = pair_ranks(meta, {k: sets[k] for k in ("mast7_cross", "mast9_cross", "mast4_cross", "NK_cross", "mast7_x_tumor")}, r_max=a.r_max_rank, min_pairs=a.min_pairs)
        res = dict(r_central=meta["r_central"].tolist(), r_from=meta["r_from"].tolist(), r_to=meta["r_to"].tolist(), ranks=ranks)
        # membership from the stored log g (no bootstrap needed): mean log g with CORE4 \ {g} at r <= 8 µm, per gene
        memb = {}
        for g in CAND9 + NOTMAST:
            partners = [b for b in CORE4 if b != g and b in f]
            vals = {}
            for k in range(len(meta["r_central"])):
                if meta["r_central"][k] > 12: break
                v = [meta["log_g"][k, f.index(g), f.index(b)] for b in partners]; s = [meta["sd"][k, f.index(g), f.index(b)] for b in partners]
                v = np.array(v, float); s = np.array(s, float); okv = np.isfinite(v)
                vals[f"{meta['r_central'][k]:.1f}"] = dict(mean_log_g=float(np.mean(v[okv])) if okv.any() else None, sd_rms=float(np.sqrt(np.mean(s[okv] ** 2))) if okv.any() else None,
                                                          self_log_g=float(meta["log_g"][k, f.index(g), f.index(g)]), self_sd=float(meta["sd"][k, f.index(g), f.index(g)]), n=int(okv.sum()))
            memb[g] = vals
        res["membership"] = memb
        if not a.no_tiles:
            boot, info = block_bootstrap(z, meta, sets, n_boot=a.n_boot)
            res["block"] = boot; res["block_info"] = info
        out["platforms"][plat] = res
        del z
    # Xenium/HD agreement for the mast pairs (per pair, r >= 6 µm) and pooled difference
    X, H = metas["xenium"], metas["hd"]; fx, fh = X["features"], H["features"]
    agree = []
    for k in range(len(X["r_central"])):
        r = X["r_central"][k]
        if r < 6 or r > 200: continue
        kh = int(np.argmin(np.abs(H["r_central"] - r)))
        vx, vh, sx, sh = [], [], [], []
        for i, ga in enumerate(CORE7):
            for gb in CORE7[i + 1:]:
                gx = X["log_g"][k, fx.index(ga), fx.index(gb)]; gh = H["log_g"][kh, fh.index(ga), fh.index(gb)]
                if np.isfinite(gx) and np.isfinite(gh): vx.append(gx); vh.append(gh); sx.append(X["sd"][k, fx.index(ga), fx.index(gb)]); sh.append(H["sd"][kh, fh.index(ga), fh.index(gb)])
        vx, vh, sx, sh = map(np.array, (vx, vh, sx, sh))
        row = dict(r_um=float(r), n_pairs=int(len(vx)), pearson=float(np.corrcoef(vx, vh)[0, 1]) if len(vx) > 2 else None, rms=float(np.sqrt(np.mean((vx - vh) ** 2))),
                   mean_diff=float(np.mean(vx - vh)), within_2sd=float(np.mean(np.abs(vx - vh) < 2 * np.hypot(sx, sh))))
        if "block" in out["platforms"]["xenium"]:
            bx, bh = out["platforms"]["xenium"]["block"]["mast7_cross"], out["platforms"]["hd"]["block"]["mast7_cross"]
            row["pooled_diff"] = float(bx["pooled_log_g"][k] - bh["pooled_log_g"][kh]); row["pooled_diff_sd"] = float(np.hypot(bx["pooled_sd"][k], bh["pooled_sd"][kh]))
        agree.append(row)
    out["xen_vs_hd_mast7"] = agree
    # molecule level (Δ = 0.5 µm) below 2 µm and the P1/P5 replication (stored log g only)
    if TWOPOINT_MOL.exists():
        m, _ = load_meta(TWOPOINT_MOL, keys=("features", "r_central", "r_from", "r_to", "log_g", "sd", "P")); fm = m["features"]
        rows = []
        for k in range(len(m["r_central"])):
            if m["r_central"][k] > 12: break
            v = np.array([m["log_g"][k, fm.index(ga), fm.index(gb)] for i, ga in enumerate(CORE7) for gb in CORE7[i + 1:]])
            s = np.array([m["log_g"][k, fm.index(g), fm.index(g)] for g in CORE7])
            rows.append(dict(r_from=float(m["r_from"][k]), r_to=float(m["r_to"][k]), mean_log_g_cross=float(np.nanmean(v)), mean_log_g_self=float(np.nanmean(s)),
                             cpa3_self=float(m["log_g"][k, fm.index("CPA3"), fm.index("CPA3")]), hdc_self=float(m["log_g"][k, fm.index("HDC"), fm.index("HDC")])))
        out["molecule_level_xenium"] = rows
    repl = {}
    for (plat, p), path in REPL.items():
        if not path.exists(): continue
        m, _ = load_meta(path, keys=("features", "r_central", "log_g", "sd", "P", "totals")); f = m["features"]
        rk = pair_ranks(m, {"mast7_cross": pair_sets(f)["mast7_cross"]}, r_max=a.r_max_rank, min_pairs=a.min_pairs)
        rows = []
        for k in range(len(m["r_central"])):
            if m["r_central"][k] > 12: break
            v = np.array([m["log_g"][k, f.index(ga), f.index(gb)] for i, ga in enumerate(CORE7) for gb in CORE7[i + 1:]]); okv = np.isfinite(v)
            sd = np.array([m["sd"][k, f.index(ga), f.index(gb)] for i, ga in enumerate(CORE7) for gb in CORE7[i + 1:]])
            rows.append(dict(r_um=float(m["r_central"][k]), mean_log_g_cross=float(np.mean(v[okv])), median_pair_sd=float(np.nanmedian(sd[okv])), n_pairs=int(okv.sum()),
                             hdc_member=float(np.nanmean([m["log_g"][k, f.index("HDC"), f.index(b)] for b in CORE4])), cma1_member=float(np.nanmean([m["log_g"][k, f.index("CMA1"), f.index(b)] for b in CORE4])),
                             xcl2_member=float(np.nanmean([m["log_g"][k, f.index("XCL2"), f.index(b)] for b in CORE4]))))
        repl[f"{plat}_{p}"] = dict(curve=rows, ranks=[dict(r_um=x["r_um"], median_rank=x["sets"]["mast7_cross"]["median_rank"], max_rank=x["sets"]["mast7_cross"]["max_rank"], n_top100=x["sets"]["mast7_cross"]["n_in_top100"], n_pairs_ranked=x["n_pairs_ranked"]) for x in rk],
                                   totals={g: int(m["totals"][f.index(g)]) for g in CAND9 + NOTMAST})
    out["replication"] = repl
    out["seconds"] = time.time() - t_all
    jdump(out, OUT / "mastcells_twopoint.json")
    # ---- quick-look
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = {"xenium": "C1", "hd": "C2"}
    for plat, res in out["platforms"].items():
        r = np.array(res["r_central"])
        if "block" in res:
            for name, ls in (("mast7_cross", "-"), ("NK_cross", "--"), ("mast7_x_tumor", ":"), ("mast7_x_smooth_muscle", "-.")):
                b = res["block"][name]; y = np.array(b["pooled_log_g"]); lo = np.array(b["pooled_lo"]); hi = np.array(b["pooled_hi"])
                ok = (r >= (6 if plat == "hd" else 2)) & (r <= 3000) & np.isfinite(y)
                axs[0].fill_between(r[ok], lo[ok], hi[ok], color=cols[plat], alpha=0.15, lw=0); axs[0].plot(r[ok], y[ok], ls, color=cols[plat], label=f"{plat} {name}")
    axs[0].axhline(0, c="k", lw=0.5); axs[0].set_xscale("log"); axs[0].set_xlabel("r [µm]"); axs[0].set_ylabel("pooled log g of the pair set"); axs[0].legend(fontsize=6); axs[0].set_title("mast block vs r (closed-form tile bootstrap 95 %)")
    for plat, res in out["platforms"].items():
        rk = [x for x in res["ranks"] if 6 <= x["r_um"] <= 7][0] if plat == "hd" else [x for x in res["ranks"] if 2 <= x["r_um"] <= 3][0]
        ranks = [p["rank"] for p in rk["sets"]["mast7_cross"]["pairs"] if p["rank"]]
        axs[1].scatter(ranks, [p["log_g"] for p in rk["sets"]["mast7_cross"]["pairs"] if p["rank"]], color=cols[plat], s=14, label=f"{plat} mast7 pairs at r = {rk['r_um']:.1f} µm (of {rk['n_pairs_ranked']:,})")
        nk = [p["rank"] for p in rk["sets"]["NK_cross"]["pairs"] if p["rank"]]
        axs[1].scatter(nk, [p["log_g"] for p in rk["sets"]["NK_cross"]["pairs"] if p["rank"]], color=cols[plat], marker="x", s=14, label=f"{plat} NK pairs")
    axs[1].set_xscale("log"); axs[1].set_xlabel("rank among all 83,028 gene pairs"); axs[1].set_ylabel("log g"); axs[1].legend(fontsize=6); axs[1].set_title("contact-scale rank of the mast pairs")
    genes = CAND9 + NOTMAST; M = np.full((len(genes), 2), np.nan)
    for j, plat in enumerate(("xenium", "hd")):
        memb = out["platforms"][plat]["membership"]
        for i, g in enumerate(genes):
            key = [k for k in memb[g] if 6 <= float(k) <= 7][0]
            M[i, j] = memb[g][key]["mean_log_g"] if memb[g][key]["mean_log_g"] is not None else np.nan
    im = axs[2].imshow(M, cmap="viridis", aspect="auto"); axs[2].set_yticks(range(len(genes))); axs[2].set_yticklabels(genes); axs[2].set_xticks([0, 1]); axs[2].set_xticklabels(["Xenium", "Visium HD"])
    for i in range(len(genes)):
        for j in range(2): axs[2].text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", color="w", fontsize=8)
    plt.colorbar(im, ax=axs[2], label="mean log g with CPA3/KIT/MS4A2/CTSG at r ≈ 6.6 µm"); axs[2].set_title("block membership")
    fig.tight_layout(); fig.savefig(QL / "artifacts_mast_twopoint.png", dpi=130)
    # ---- console summary
    for plat, res in out["platforms"].items():
        for x in res["ranks"]:
            s = x["sets"]["mast7_cross"]
            print(f"[{plat}] r={x['r_um']:.1f}: mast7 median rank {s['median_rank']}, max {s['max_rank']}, in top100 {s['n_in_top100']}/{s['n_valid']}, in top 1% {s['n_in_top1pct']}; mast pairs in top40: {x['n_mast_in_top40']}")
        if "block" in res:
            b = res["block"]["mast7_cross"]; r = np.array(res["r_central"])
            print(f"[{plat}] mast7 pooled log g:", [(round(float(r[k]), 1), round(b["pooled_log_g"][k], 2), round(b["pooled_sd"][k], 2)) for k in range(0, 12)])
    print("membership at 6.6 µm:", {g: (round(M[i, 0], 2), round(M[i, 1], 2)) for i, g in enumerate(genes)})
    print("agreement:", [(round(x["r_um"], 1), x["pearson"] and round(x["pearson"], 2), round(x["rms"], 2), round(x["mean_diff"], 2)) for x in agree[:6]])
    print(f"done in {time.time()-t_all:.0f} s")


# ------------------------------------------------------------------------------------------------------------------- cells
def read_cell_matrix():
    import h5py, scipy.sparse as sp
    with h5py.File(DATA / "raw/xenium_crc_p2/cell_feature_matrix.h5") as h:
        shape = h["matrix/shape"][:]; data = h["matrix/data"][:]; indices = h["matrix/indices"][:]; indptr = h["matrix/indptr"][:]
        names = [x.decode() for x in h["matrix/features/name"][:]]; ftype = [x.decode() for x in h["matrix/features/feature_type"][:]]
        barcodes = [x.decode() for x in h["matrix/barcodes"][:]]
    X = sp.csc_matrix((data, indices, indptr), shape=tuple(shape)).T.tocsr()   # cells x features
    keep = np.array([t == "Gene Expression" for t in ftype])
    return X[:, keep], [n for n, k in zip(names, keep) if k], barcodes


def dbscan_spots(xy, eps=5.0, min_samples=3, weights=None):
    from sklearn.cluster import DBSCAN
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit(xy, sample_weight=weights).labels_
    ok = lab >= 0
    if ok.sum() == 0: return lab, dict(n_spots=0)
    w = np.ones(len(xy)) if weights is None else weights
    n_spots = lab.max() + 1
    cnt = np.bincount(lab[ok], weights=w[ok], minlength=n_spots)
    cx = np.bincount(lab[ok], weights=w[ok] * xy[ok, 0], minlength=n_spots) / cnt; cy = np.bincount(lab[ok], weights=w[ok] * xy[ok, 1], minlength=n_spots) / cnt
    d2 = (xy[ok, 0] - cx[lab[ok]]) ** 2 + (xy[ok, 1] - cy[lab[ok]]) ** 2
    rms = np.sqrt(np.bincount(lab[ok], weights=w[ok] * d2, minlength=n_spots) / cnt)
    diam = 2 * np.sqrt(2) * rms      # uniform disk: RMS radius = R / sqrt(2)
    return lab, dict(n_spots=int(n_spots), frac_points_in_spots=float(w[ok].sum() / w.sum()), molecules_per_spot=dict(median=float(np.median(cnt)), q25=float(np.percentile(cnt, 25)), q75=float(np.percentile(cnt, 75)), mean=float(cnt.mean())),
                     diameter_um=dict(median=float(np.median(diam)), q25=float(np.percentile(diam, 25)), q75=float(np.percentile(diam, 75))), centroids=np.stack([cx, cy], 1), counts=cnt)


def leiden_pipeline(X, mast_counts, total_counts, resolutions=(0.5, 1.0, 2.0, 4.0, 8.0), min_counts=10, n_pcs=30, n_neighbors=15, hvg=None, seed=0, label="cells", subsample=None, var_names=None):
    """Standard scanpy pipeline; returns per-resolution cluster tables with mast purity / recall and timings."""
    import scanpy as sc, anndata as ad, scipy.sparse as sp
    t0 = time.time(); timings = {}
    keep = np.asarray(total_counts >= min_counts).ravel()
    idx = np.where(keep)[0]
    if subsample and len(idx) > subsample:
        idx = np.sort(np.random.default_rng(seed).choice(idx, subsample, replace=False))
    A = ad.AnnData(sp.csr_matrix(X[idx]).astype(np.float32))
    if var_names is not None: A.var_names = [str(v) for v in var_names]
    A.obs["mast"] = np.asarray(mast_counts[idx]).ravel(); A.obs["total"] = np.asarray(total_counts[idx]).ravel()
    A.obs["mast_frac"] = A.obs["mast"] / np.maximum(A.obs["total"], 1); A.obs["is_mast3"] = A.obs["mast"] >= 3
    sc.pp.normalize_total(A); sc.pp.log1p(A)
    if hvg:
        sc.pp.highly_variable_genes(A, n_top_genes=hvg); A = A[:, A.var.highly_variable].copy()
    timings["normalize"] = time.time() - t0; t1 = time.time()
    sc.tl.pca(A, n_comps=n_pcs, svd_solver="arpack", random_state=seed); timings["pca"] = time.time() - t1; t1 = time.time()
    sc.pp.neighbors(A, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=seed); timings["neighbors"] = time.time() - t1
    res_out = {}
    n_mast3 = int(A.obs["is_mast3"].sum())
    for res in resolutions:
        t1 = time.time(); key = f"leiden_{res}"
        sc.tl.leiden(A, resolution=res, key_added=key, flavor="igraph", n_iterations=2, directed=False, random_state=seed)
        timings[key] = time.time() - t1
        cl = A.obs[key].astype(str).values; ucl = np.unique(cl)
        rows = []
        for c in ucl:
            m = cl == c
            rows.append(dict(cluster=c, size=int(m.sum()), mean_mast_frac=float(A.obs["mast_frac"].values[m].mean()), purity_mast3=float(A.obs["is_mast3"].values[m].mean()),
                             recall_mast3=float(A.obs["is_mast3"].values[m].sum() / max(n_mast3, 1)), mean_mast_counts=float(A.obs["mast"].values[m].mean())))
        rows.sort(key=lambda d: -d["purity_mast3"])
        best = rows[0]
        # marker genes of the best cluster (mean log1p-normalized expression in vs out), top 8
        m = cl == best["cluster"]
        mu_in = np.asarray(A.X[m].mean(0)).ravel(); mu_out = np.asarray(A.X[~m].mean(0)).ravel()
        top = np.argsort(-(mu_in - mu_out))[:8]
        best_markers = [dict(gene=str(A.var_names[t]), in_=float(mu_in[t]), out=float(mu_out[t])) for t in top]
        rec = max(rows, key=lambda d: d["recall_mast3"])
        res_out[str(res)] = dict(n_clusters=len(ucl), best_cluster=best, highest_recall_cluster=rec, best_cluster_markers=best_markers, recovered=bool(best["purity_mast3"] >= 0.5 and best["recall_mast3"] >= 0.5),
                                 n_clusters_purity_ge_0p5=int(sum(r["purity_mast3"] >= 0.5 for r in rows)), clusters=rows[:5], seconds=timings[key])
        print(f"  [{label}] leiden {res}: {len(ucl)} clusters; best purity {best['purity_mast3']:.2f}, recall {best['recall_mast3']:.2f}, size {best['size']}, mean mast frac {best['mean_mast_frac']:.2f}; markers {[d['gene'] for d in best_markers[:5]]}; {timings[key]:.0f} s", flush=True)
    return dict(n_input=int(len(idx)), n_mast3=n_mast3, resolutions=res_out, timings=timings, seconds=time.time() - t0)


def stage_cells(a):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import pyarrow.parquet as pq, pandas as pd
    t_all = time.time()
    reg = region_grid("xenium"); areas = region_areas(reg)
    mast_set = CORE7 if a.mast_set == "core7" else CAND9
    # (1) mast molecules from the raw transcript table (string cell ids match cells.parquet / the h5 barcodes)
    t0 = time.time()
    tb = pq.read_table(DATA / "raw/xenium_crc_p2/transcripts.parquet", columns=["cell_id", "overlaps_nucleus", "feature_name", "x_location", "y_location", "qv"],
                       filters=[("feature_name", "in", CAND9 + NOTMAST), ("qv", ">=", 20)])
    tr = tb.to_pandas(); tr["in_cell"] = tr["cell_id"] != "UNASSIGNED"
    print(f"mast transcripts (qv >= 20): {len(tr):,} in {time.time()-t0:.0f} s", flush=True)
    per_gene = {}
    for g, d in tr.groupby("feature_name", observed=True):
        per_gene[g] = dict(n=int(len(d)), frac_in_cell=float(d.in_cell.mean()), frac_nucleus=float(d.overlaps_nucleus.astype(bool).mean()), median_qv=float(d.qv.median()))
    # panel reference: per-gene in-cell fraction over all 422 genes from the processed molecule table (cell_id -1 = unassigned)
    mol = pq.read_table(DATA / "processed/xenium_crc_p2/molecules.parquet", columns=["feature", "cell_id", "nucleus"])
    fcol = mol.column("feature").combine_chunks(); codes = fcol.indices.to_numpy(zero_copy_only=False).astype(np.int64); dic = fcol.dictionary.to_pylist()
    incell = mol.column("cell_id").to_numpy() >= 0; nuc = mol.column("nucleus").to_numpy(zero_copy_only=False).astype(bool)
    n_g = np.bincount(codes, minlength=len(dic)); n_in = np.bincount(codes[incell], minlength=len(dic)); n_nu = np.bincount(codes[nuc], minlength=len(dic))
    frac_in = n_in / np.maximum(n_g, 1); frac_nu = n_nu / np.maximum(n_g, 1)
    okg = n_g >= 500
    panel = dict(n_genes=int(okg.sum()), frac_in_cell=dict(median=float(np.median(frac_in[okg])), q05=float(np.percentile(frac_in[okg], 5)), q25=float(np.percentile(frac_in[okg], 25)), q75=float(np.percentile(frac_in[okg], 75)), q95=float(np.percentile(frac_in[okg], 95))),
                 frac_nucleus=dict(median=float(np.median(frac_nu[okg])), q05=float(np.percentile(frac_nu[okg], 5)), q95=float(np.percentile(frac_nu[okg], 95))), all_molecules_frac_in_cell=float(incell.mean()))
    for g in per_gene:
        if g in dic:
            per_gene[g]["percentile_in_cell_vs_panel"] = float((frac_in[okg] < frac_in[dic.index(g)]).mean())
    mast_mol = tr[tr.feature_name.isin(mast_set)]
    in_cell_mast = float(mast_mol.in_cell.mean())
    del mol, codes, incell, nuc
    # (2) cell matrix
    t0 = time.time()
    X, gnames, barcodes = read_cell_matrix()
    cells = pd.read_parquet(DATA / "raw/xenium_crc_p2/cells.parquet").set_index("cell_id").loc[barcodes]
    total = np.asarray(X.sum(1)).ravel()
    gi = [gnames.index(g) for g in mast_set if g in gnames]
    mast = np.asarray(X[:, gi].sum(1)).ravel()
    mast9 = np.asarray(X[:, [gnames.index(g) for g in CAND9 if g in gnames]].sum(1)).ravel()
    xy_cells = cells[["x_centroid", "y_centroid"]].to_numpy()
    print(f"cell matrix {X.shape}, mast transcripts in matrix {int(mast.sum()):,} (table: {int(mast_mol.in_cell.sum()):,}) in {time.time()-t0:.0f} s", flush=True)
    frac = mast / np.maximum(total, 1)
    tinv, nt = tile_keys(xy_cells)

    def cell_stats(w):
        n_tot = w.sum()
        res = [np.sum(w * (mast >= 1)), np.sum(w * (mast >= 2)), np.sum(w * (mast >= 3)), np.sum(w * (mast >= 5)), np.sum(w * ((mast >= 3) & (frac >= 0.1))), np.sum(w * ((mast >= 5) & (frac >= 0.2)))]
        wm = w * mast; tot = wm.sum()
        res += [np.sum(wm[frac < 0.1]) / tot, np.sum(wm[frac < 0.05]) / tot, np.sum(wm[mast >= 3]) / tot, np.sum(wm[mast < 2]) / tot, n_tot]
        return np.array(res)
    base, sd, lo, hi = tile_boot(cell_stats, tinv, nt, a.n_boot)
    names = ["cells_ge1", "cells_ge2", "cells_ge3", "cells_ge5", "cells_ge3_frac_ge10pct", "cells_ge5_frac_ge20pct", "frac_mast_in_cells_mastfrac_lt10pct", "frac_mast_in_cells_mastfrac_lt5pct", "frac_mast_in_cells_ge3", "frac_mast_in_cells_lt2", "n_cells"]
    cellview = {n: dict(value=float(base[i]), sd=float(sd[i]), lo=float(lo[i]), hi=float(hi[i])) for i, n in enumerate(names)}
    for n in names[:6]:
        cellview[n]["per_mm2"] = cellview[n]["value"] / areas["tissue"]; cellview[n]["per_mm2_sd"] = cellview[n]["sd"] / areas["tissue"]
    m3 = mast >= 3
    cellview["mast_frac_in_cells_ge3"] = dict(median=float(np.median(frac[m3])), q25=float(np.percentile(frac[m3], 25)), q75=float(np.percentile(frac[m3], 75)))
    cellview["mast_counts_in_cells_ge3"] = dict(median=float(np.median(mast[m3])), q25=float(np.percentile(mast[m3], 25)), q75=float(np.percentile(mast[m3], 75)), max=int(mast.max()))
    cellview["total_counts_in_cells_ge3"] = dict(median=float(np.median(total[m3])))
    srt = np.sort(mast)[::-1]; cum = np.cumsum(srt) / srt.sum()
    cellview["cells_holding_50pct_of_mast_transcripts"] = int(np.searchsorted(cum, 0.5) + 1); cellview["cells_holding_90pct_of_mast_transcripts"] = int(np.searchsorted(cum, 0.9) + 1)
    cellview["cells_with_any_mast_transcript_frac"] = float((mast >= 1).mean())
    cellview["mast9_cells_ge3"] = int((mast9 >= 3).sum())
    cellview["in_cell_fraction_mast_set"] = in_cell_mast
    # (3) standard scanpy pipeline
    print("scanpy pipeline on cells ...", flush=True)
    if a.keep_leiden:   # the clustering does not depend on the uncertainty estimator: reuse the stored block
        leiden = json.load(open(OUT / "mastcells_cells.json"))["leiden"]; print("leiden: kept from the stored mastcells_cells.json", flush=True)
    else:
        leiden = leiden_pipeline(X, mast, total, resolutions=tuple(a.resolutions), min_counts=10, subsample=a.subsample, label="cells", var_names=gnames)
    # (4) molecule-level DBSCAN spots on the mast molecules
    t0 = time.time()
    xy = mast_mol[["x_location", "y_location"]].to_numpy().astype(np.float64)
    lab, spots = dbscan_spots(xy, eps=a.eps, min_samples=a.min_samples)
    cen = spots.pop("centroids"); cnt = spots.pop("counts")
    tinv_s, nt_s = tile_keys(cen)
    dens, dens_sd, dens_lo, dens_hi = tile_boot(lambda w: np.array([w.sum() / areas["tissue"], np.sum(w * (cnt >= 5)) / areas["tissue"], np.sum(w * (cnt >= 10)) / areas["tissue"]]), tinv_s, nt_s, a.n_boot)
    spots.update(eps_um=a.eps, min_samples=a.min_samples, spots_per_mm2=float(dens[0]), spots_per_mm2_sd=float(dens_sd[0]), spots_ge5_per_mm2=float(dens[1]), spots_ge5_per_mm2_sd=float(dens_sd[1]),
                 spots_ge10_per_mm2=float(dens[2]), spots_ge10_per_mm2_sd=float(dens_sd[2]), tissue_area_mm2=areas["tissue"], seconds=time.time() - t0)
    # spot ↔ segmented cell match: spot centroid within 10 µm of a cell with >= 3 mast transcripts, and the reverse
    from scipy.spatial import cKDTree
    c3 = xy_cells[m3]
    if len(c3):
        d_sc, _ = cKDTree(c3).query(cen, k=1); spots["frac_spots_within_10um_of_mast3_cell"] = float((d_sc <= 10).mean())
        d_cs, _ = cKDTree(cen).query(c3, k=1); spots["frac_mast3_cells_within_10um_of_spot"] = float((d_cs <= 10).mean())
    # region of the spots + spot molecule composition: in-cell fraction of spot molecules vs isolated molecules
    slab, _ = label_grid(cen, reg)
    spots["spots_per_region_per_mm2"] = {r: float((slab == i).sum() / areas[r]) for i, r in enumerate(REGIONS)}
    spots["frac_spot_molecules_in_cells"] = float(mast_mol.in_cell.to_numpy()[lab >= 0].mean()); spots["frac_isolated_molecules_in_cells"] = float(mast_mol.in_cell.to_numpy()[lab < 0].mean())
    # cells per spot: how many distinct segmented cells share one spot's molecules
    cid = mast_mol.cell_id.to_numpy()
    sp_df = pd.DataFrame(dict(spot=lab[lab >= 0], cid=cid[lab >= 0])); sp_df = sp_df[sp_df.cid != "UNASSIGNED"]
    ncells = sp_df.groupby("spot").cid.nunique(); spots["distinct_cells_per_spot"] = dict(median=float(ncells.median()), frac_gt1=float((ncells > 1).mean()), frac_ge3=float((ncells >= 3).mean()))
    # sensitivity: eps and gene set
    sens = {}
    for eps in (3.0, 5.0, 8.0):
        _, s = dbscan_spots(xy, eps=eps, min_samples=3); s.pop("centroids", None); s.pop("counts", None); sens[f"core7_eps{eps:g}"] = dict(n_spots=s["n_spots"], per_mm2=s["n_spots"] / areas["tissue"], frac_points_in_spots=s["frac_points_in_spots"], molecules_per_spot_median=s["molecules_per_spot"]["median"])
    xy9 = tr[tr.feature_name.isin(CAND9)][["x_location", "y_location"]].to_numpy().astype(np.float64)
    _, s = dbscan_spots(xy9, eps=a.eps, min_samples=3); s.pop("centroids", None); s.pop("counts", None); sens["cand9_eps5"] = dict(n_spots=s["n_spots"], per_mm2=s["n_spots"] / areas["tissue"], frac_points_in_spots=s["frac_points_in_spots"], molecules_per_spot_median=s["molecules_per_spot"]["median"])
    spots["sensitivity"] = sens
    # (5) 8 µm-bin analogue on Xenium (fair comparison with the HD 8 µm bins): all molecules binned to 8 µm
    t0 = time.time()
    molxy = pq.read_table(DATA / "processed/xenium_crc_p2/molecules.parquet", columns=["x", "y", "feature"])
    fcol = molxy.column("feature").combine_chunks(); codes = fcol.indices.to_numpy(zero_copy_only=False).astype(np.int64); dic = fcol.dictionary.to_pylist()
    x = molxy.column("x").to_numpy(); y = molxy.column("y").to_numpy(); o = np.array([x.min(), y.min()])
    bx = np.floor((x - o[0]) / 8).astype(np.int64); by = np.floor((y - o[1]) / 8).astype(np.int64); nby = by.max() + 1; bkey = bx * nby + by
    ub, binv = np.unique(bkey, return_inverse=True)
    tot8 = np.bincount(binv, minlength=len(ub)); ismast = np.isin(codes, [dic.index(g) for g in mast_set if g in dic]); m8 = np.bincount(binv[ismast], minlength=len(ub))
    f8 = m8 / np.maximum(tot8, 1)
    bins8 = dict(n_bins_with_molecules=int(len(ub)), bins_ge2=int((m8 >= 2).sum()), bins_ge3=int((m8 >= 3).sum()), bins_ge5=int((m8 >= 5).sum()), bins_ge3_per_mm2=float((m8 >= 3).sum() / areas["tissue"]),
                 frac_mast_in_bins_mastfrac_lt10pct=float(m8[f8 < 0.1].sum() / m8.sum()), frac_mast_in_bins_ge3=float(m8[m8 >= 3].sum() / m8.sum()), median_mast_frac_bins_ge3=float(np.median(f8[m8 >= 3])), seconds=time.time() - t0)
    out = dict(seconds=time.time() - t_all, mast_set=mast_set, areas_mm2=areas, per_gene=per_gene, panel_reference=panel, n_mast_transcripts=int(len(mast_mol)), cell_view=cellview, leiden=leiden, spots=spots, xenium_bins_8um=bins8,
               n_cells=int(X.shape[0]), n_cells_ge10_counts=int((total >= 10).sum()))
    jdump(out, OUT / "mastcells_cells.json")
    # ---- quick-look
    fig, axs = plt.subplots(1, 4, figsize=(17, 4))
    axs[0].hist(mast[mast >= 1], bins=np.arange(1, 60), log=True); axs[0].set_xlabel(f"mast transcripts per cell ({'+'.join(mast_set)})"); axs[0].set_ylabel("cells"); axs[0].set_title(f"{int((mast>=3).sum()):,} cells ≥ 3, {int((mast>=1).sum()):,} ≥ 1")
    axs[1].hist(frac[mast >= 3], bins=40); axs[1].set_xlabel("mast fraction of the cell's transcripts (cells ≥ 3)"); axs[1].set_title(f"{cellview['frac_mast_in_cells_mastfrac_lt10pct']['value']*100:.0f} % of in-cell mast transcripts in cells with mast fraction < 10 %")
    for res, d in leiden["resolutions"].items():
        rows = d["clusters"]; axs[2].scatter([r["size"] for r in rows], [r["purity_mast3"] for r in rows], label=f"res {res}: best recall {d['best_cluster']['recall_mast3']:.2f}")
    axs[2].set_xscale("log"); axs[2].set_xlabel("cluster size"); axs[2].set_ylabel("purity (fraction of cells with ≥ 3 mast transcripts)"); axs[2].legend(fontsize=7); axs[2].set_title("leiden clusters (top 5 by purity)")
    sel = (cen[:, 0] > cen[:, 0].mean() - 1000) & (cen[:, 0] < cen[:, 0].mean() + 1000) & (cen[:, 1] > cen[:, 1].mean() - 1000) & (cen[:, 1] < cen[:, 1].mean() + 1000)
    axs[3].scatter(cen[sel, 0], cen[sel, 1], s=cnt[sel] / 3, alpha=0.6, label=f"DBSCAN spots ({spots['spots_per_mm2']:.0f} ± {spots['spots_per_mm2_sd']:.0f} / mm²)")
    selc = (xy_cells[:, 0] > cen[:, 0].mean() - 1000) & (xy_cells[:, 0] < cen[:, 0].mean() + 1000) & (xy_cells[:, 1] > cen[:, 1].mean() - 1000) & (xy_cells[:, 1] < cen[:, 1].mean() + 1000) & m3
    axs[3].scatter(xy_cells[selc, 0], xy_cells[selc, 1], s=6, marker="x", c="C3", label="cells ≥ 3 mast transcripts"); axs[3].set_aspect("equal"); axs[3].legend(fontsize=7); axs[3].set_title("2 × 2 mm window")
    fig.tight_layout(); fig.savefig(QL / "artifacts_mast_cells.png", dpi=130)
    print(json.dumps(dict(per_gene=per_gene, panel=panel, cell_view={k: v for k, v in cellview.items()}, spots={k: v for k, v in spots.items() if k != "sensitivity"}, bins8=bins8), indent=1, default=float))
    print("leiden:", json.dumps({r: dict(best=d["best_cluster"], recovered=d["recovered"], n_clusters=d["n_clusters"], markers=[m["gene"] for m in d["best_cluster_markers"]]) for r, d in leiden["resolutions"].items()}, indent=1, default=float), leiden["timings"])
    print(f"done in {time.time()-t_all:.0f} s")


# ---------------------------------------------------------------------------------------------------------------------- hd
def stage_hd(a):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import scipy.sparse as sp
    t_all = time.time()
    reg = region_grid("hd"); areas = region_areas(reg)
    mast_set = CORE7 if a.mast_set == "core7" else CAND9
    out = dict(mast_set=mast_set, areas_mm2=areas)
    # 2 µm bins: mast UMIs per bin, DBSCAN on bins with mast UMIs
    t0 = time.time()
    z = np.load(DATA / "processed/hd_crc_p2/bins_square_002um.npz", allow_pickle=True)
    feats = list(map(str, z["features"])); X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])); coords = z["coords_um"].astype(np.float64)
    gi = [feats.index(g) for g in mast_set if g in feats]
    m2 = np.asarray(X[:, gi].sum(1)).ravel(); tot2 = np.asarray(X.sum(1)).ravel()
    print(f"HD 2 µm: {X.shape[0]:,} bins, {int(m2.sum()):,} mast UMIs in {time.time()-t0:.0f} s", flush=True)
    has = m2 > 0
    lab, spots = dbscan_spots(coords[has], eps=a.eps, min_samples=a.min_samples, weights=m2[has].astype(float))
    cen = spots.pop("centroids"); cnt = spots.pop("counts")
    tinv_s, nt_s = tile_keys(cen)
    dens, dens_sd, _, _ = tile_boot(lambda w: np.array([w.sum() / areas["tissue"], np.sum(w * (cnt >= 5)) / areas["tissue"], np.sum(w * (cnt >= 10)) / areas["tissue"]]), tinv_s, nt_s, a.n_boot)
    slab, _ = label_grid(cen, reg)
    spots.update(eps_um=a.eps, min_samples=a.min_samples, spots_per_mm2=float(dens[0]), spots_per_mm2_sd=float(dens_sd[0]), spots_ge5_per_mm2=float(dens[1]), spots_ge5_per_mm2_sd=float(dens_sd[1]),
                 spots_ge10_per_mm2=float(dens[2]), spots_ge10_per_mm2_sd=float(dens_sd[2]), tissue_area_mm2=areas["tissue"], spots_per_region_per_mm2={r: float((slab == i).sum() / areas[r]) for i, r in enumerate(REGIONS)},
                 n_bins_with_mast=int(has.sum()), bins_2um_ge2=int((m2 >= 2).sum()), bins_2um_ge3=int((m2 >= 3).sum()), n_mast_umis=int(m2.sum()))
    sens = {}
    for eps in (3.0, 5.0, 8.0):
        _, s = dbscan_spots(coords[has], eps=eps, min_samples=3, weights=m2[has].astype(float)); sens[f"eps{eps:g}"] = dict(n_spots=s["n_spots"], per_mm2=s["n_spots"] / areas["tissue"], frac_umis_in_spots=s["frac_points_in_spots"], umis_per_spot_median=s["molecules_per_spot"]["median"])
    spots["sensitivity"] = sens
    out["spots_2um"] = spots
    # 8 µm bins
    t0 = time.time()
    z8 = np.load(DATA / "processed/hd_crc_p2/bins_square_008um.npz", allow_pickle=True)
    f8 = list(map(str, z8["features"])); X8 = sp.csr_matrix((z8["data"], z8["indices"], z8["indptr"]), shape=tuple(z8["shape"])); c8 = z8["coords_um"].astype(np.float64)
    intis = z8["in_tissue"].astype(bool) if "in_tissue" in z8.files else np.ones(X8.shape[0], bool)
    gi8 = [f8.index(g) for g in mast_set if g in f8]
    m8 = np.asarray(X8[:, gi8].sum(1)).ravel(); tot8 = np.asarray(X8.sum(1)).ravel(); fr8 = m8 / np.maximum(tot8, 1)
    lab8, _ = label_grid(c8, reg)
    bins8 = dict(n_bins=int(X8.shape[0]), n_bins_in_tissue=int(intis.sum()), median_umis_per_bin=float(np.median(tot8[intis])), n_mast_umis=int(m8.sum()),
                 bins_ge2=int((m8 >= 2).sum()), bins_ge3=int((m8 >= 3).sum()), bins_ge5=int((m8 >= 5).sum()), bins_ge3_per_mm2=float((m8 >= 3).sum() / areas["tissue"]),
                 frac_mast_in_bins_mastfrac_lt10pct=float(m8[fr8 < 0.1].sum() / max(m8.sum(), 1)), frac_mast_in_bins_ge3=float(m8[m8 >= 3].sum() / max(m8.sum(), 1)),
                 median_mast_frac_bins_ge3=float(np.median(fr8[m8 >= 3])) if (m8 >= 3).any() else None, frac_bins_with_any_mast=float((m8 >= 1).mean()),
                 bins_ge3_per_region_per_mm2={r: float(((m8 >= 3) & (lab8 == i)).sum() / areas[r]) for i, r in enumerate(REGIONS)}, seconds=time.time() - t0)
    out["bins_8um"] = bins8
    print("8 µm bins:", json.dumps(bins8, indent=1), flush=True)
    # leiden on the 8 µm bins (whole transcriptome, 2000 HVGs, subsample)
    print("scanpy pipeline on 8 µm bins ...", flush=True)
    keep = intis & (tot8 >= a.min_counts_hd)
    if a.keep_leiden:   # the clustering does not depend on the uncertainty estimator: reuse the stored block
        leiden = json.load(open(OUT / "mastcells_hd.json"))["leiden_8um"]; print("leiden: kept from the stored mastcells_hd.json", flush=True)
    else:
        Xk = X8[keep]; leiden = leiden_pipeline(Xk, m8[keep], tot8[keep], resolutions=tuple(a.resolutions), min_counts=a.min_counts_hd, hvg=2000, subsample=a.subsample, label="hd8um", var_names=f8)
        leiden["min_counts"] = a.min_counts_hd; leiden["hvg"] = 2000
    out["leiden_8um"] = leiden
    out["seconds"] = time.time() - t_all
    jdump(out, OUT / "mastcells_hd.json")
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    axs[0].hist(m8[m8 >= 1], bins=np.arange(1, 40), log=True); axs[0].set_xlabel("mast UMIs per 8 µm bin"); axs[0].set_title(f"{bins8['bins_ge3']:,} bins ≥ 3 ({bins8['bins_ge3_per_mm2']:.0f} / mm²)")
    for res, d in leiden["resolutions"].items():
        rows = d["clusters"]; axs[1].scatter([r["size"] for r in rows], [r["purity_mast3"] for r in rows], label=f"res {res}: best recall {d['best_cluster']['recall_mast3']:.2f}")
    axs[1].set_xscale("log"); axs[1].set_xlabel("cluster size (8 µm bins)"); axs[1].set_ylabel("purity (bins with ≥ 3 mast UMIs)"); axs[1].legend(fontsize=7); axs[1].set_title("leiden on HD 8 µm bins")
    sel = (np.abs(cen[:, 0] - cen[:, 0].mean()) < 1000) & (np.abs(cen[:, 1] - cen[:, 1].mean()) < 1000)
    axs[2].scatter(cen[sel, 0], cen[sel, 1], s=cnt[sel] / 2, alpha=0.6); axs[2].set_aspect("equal"); axs[2].set_title(f"HD 2 µm DBSCAN spots: {spots['spots_per_mm2']:.0f} ± {spots['spots_per_mm2_sd']:.0f} / mm², {spots['molecules_per_spot']['median']:.0f} UMIs / spot")
    fig.tight_layout(); fig.savefig(QL / "artifacts_mast_hd.png", dpi=130)
    print("spots:", json.dumps({k: v for k, v in spots.items() if k != "sensitivity"}, indent=1, default=float))
    print("leiden:", json.dumps({r: dict(best=d["best_cluster"], recovered=d["recovered"], n_clusters=d["n_clusters"], markers=[m["gene"] for m in d["best_cluster_markers"]]) for r, d in leiden["resolutions"].items()}, indent=1, default=float), leiden["timings"])
    print(f"done in {time.time()-t_all:.0f} s")


# -------------------------------------------------------------------------------------------------------------- background
def core_labels(reg, core_um=50.0):
    """Region label restricted to pixels >= core_um from any other label (incl. tissue-free); −1 elsewhere."""
    from scipy.ndimage import distance_transform_edt
    lab = reg["label"]; core = np.full(lab.shape, -1, dtype=np.int8); dist = np.zeros(lab.shape, float)
    for i in range(len(REGIONS)):
        d = distance_transform_edt(lab == i) * reg["grid_um"]; dist[lab == i] = d[lab == i]
        core[(lab == i) & (d >= core_um)] = i
    return core, dist


def stage_background(a):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import pyarrow.parquet as pq, scipy.sparse as sp
    from platform_artifacts import pixel_intercept, predicted_compression
    t_all = time.time()
    genes408 = [l.strip() for l in open(REPO / "results/platforms/common_genes_crc.txt") if l.strip() and not l.startswith("#")]
    regX, regH = region_grid("xenium"), region_grid("hd"); areaX, areaH = region_areas(regX), region_areas(regH)
    coreX, distX = core_labels(regX, a.core_um); coreH, distH = core_labels(regH, a.core_um)
    core_area = {"xenium": {r: float((coreX == i).sum() * (regX["grid_um"] / 1000) ** 2) for i, r in enumerate(REGIONS)}, "hd": {r: float((coreH == i).sum() * (regH["grid_um"] / 1000) ** 2) for i, r in enumerate(REGIONS)}}
    targets = CAND9 + NOTMAST
    # ---- Xenium: all molecules (feature, x, y) -> per-gene region / core counts, pixel counts of the targets, distance to boundary
    t0 = time.time()
    mol = pq.read_table(DATA / "processed/xenium_crc_p2/molecules.parquet", columns=["x", "y", "feature"])
    fcol = mol.column("feature").combine_chunks(); codes = fcol.indices.to_numpy(zero_copy_only=False).astype(np.int64); dic = fcol.dictionary.to_pylist()
    xy = np.stack([mol.column("x").to_numpy(), mol.column("y").to_numpy()], 1).astype(np.float64); del mol
    labX, pixX = label_grid(xy, regX); inside = pixX >= 0
    coreLab = np.full(len(xy), -1, np.int8); coreLab[inside] = coreX.ravel()[pixX[inside]]
    dist_mol = np.full(len(xy), np.nan); dist_mol[inside] = distX.ravel()[pixX[inside]]
    nG = len(dic)
    regc = np.bincount(codes * 5 + (labX.astype(np.int64) + 2), minlength=nG * 5).reshape(nG, 5)[:, 2:]     # tumor, normal, stroma
    corec = np.bincount(codes[coreLab >= 0] * 3 + coreLab[coreLab >= 0].astype(np.int64), minlength=nG * 3).reshape(nG, 3)
    npixX = int(np.prod(regX["shape"])); pix_allX = np.bincount(pixX[inside], minlength=npixX)
    pixX_gene = {g: np.bincount(pixX[inside & (codes == dic.index(g))], minlength=npixX) for g in targets if g in dic}
    # distance to the region boundary: mast molecules (CORE7) vs all molecules, per region
    is_core7 = np.isin(codes, [dic.index(g) for g in CORE7])
    dist_stats = {}
    for i, r in enumerate(REGIONS):
        m_all = (labX == i) & np.isfinite(dist_mol); m_m = m_all & is_core7
        dist_stats[r] = dict(xenium=dict(n_mast=int(m_m.sum()), median_dist_mast=float(np.median(dist_mol[m_m])), median_dist_all=float(np.median(dist_mol[m_all])),
                                         frac_within_50um_mast=float((dist_mol[m_m] < 50).mean()), frac_within_50um_all=float((dist_mol[m_all] < 50).mean()),
                                         frac_within_100um_mast=float((dist_mol[m_m] < 100).mean()), frac_within_100um_all=float((dist_mol[m_all] < 100).mean())))
    print(f"Xenium molecules scanned: {len(xy)/1e6:.1f} M in {time.time()-t0:.0f} s", flush=True)
    del xy, codes, labX, pixX, coreLab, dist_mol, is_core7
    # ---- HD: bins
    t0 = time.time()
    z = np.load(DATA / "processed/hd_crc_p2/bins_square_002um.npz", allow_pickle=True)
    feats = list(map(str, z["features"])); X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])); coords = z["coords_um"].astype(np.float64)
    labH, pixH = label_grid(coords, regH); insideH = pixH >= 0
    coreLabH = np.full(len(coords), -1, np.int8); coreLabH[insideH] = coreH.ravel()[pixH[insideH]]
    distH_bin = np.full(len(coords), np.nan); distH_bin[insideH] = distH.ravel()[pixH[insideH]]
    gidx = [feats.index(g) for g in genes408]; Xg = X[:, gidx].tocsc()
    regcH = np.zeros((len(genes408), 3)); corecH = np.zeros((len(genes408), 3))
    for i in range(3):
        rows = np.where(labH == i)[0]; regcH[:, i] = np.asarray(Xg[rows].sum(0)).ravel()
        rows = np.where(coreLabH == i)[0]; corecH[:, i] = np.asarray(Xg[rows].sum(0)).ravel()
    tot = np.asarray(X.sum(1)).ravel(); npixH = int(np.prod(regH["shape"])); pix_allH = np.bincount(pixH[insideH], weights=tot[insideH], minlength=npixH)
    pixH_gene = {g: np.bincount(pixH[insideH], weights=np.asarray(X[:, feats.index(g)].todense()).ravel()[insideH], minlength=npixH) for g in targets if g in feats}
    m7 = np.asarray(X[:, [feats.index(g) for g in CORE7]].sum(1)).ravel()
    for i, r in enumerate(REGIONS):
        m_all = (labH == i) & np.isfinite(distH_bin)
        w_all = tot[m_all]; w_m = m7[m_all]; d = distH_bin[m_all]
        def wmed(v, w):
            o = np.argsort(v); c = np.cumsum(w[o]); return float(v[o][np.searchsorted(c, 0.5 * c[-1])])
        dist_stats[r]["hd"] = dict(n_mast=int(w_m.sum()), median_dist_mast=wmed(d, w_m), median_dist_all=wmed(d, w_all), frac_within_50um_mast=float(w_m[d < 50].sum() / w_m.sum()), frac_within_50um_all=float(w_all[d < 50].sum() / w_all.sum()),
                                   frac_within_100um_mast=float(w_m[d < 100].sum() / w_m.sum()), frac_within_100um_all=float(w_all[d < 100].sum() / w_all.sum()))
    print(f"HD bins scanned in {time.time()-t0:.0f} s", flush=True)
    # ---- (a) region-density ratio test: whole regions (as in platform_artifacts.py) and region cores
    common = [g for g in genes408 if g in dic]
    def ratio_test(cx, ch, ax, ah, label):
        lrx = np.log(cx[:, 0] / ax["tumor"]) - np.log(cx[:, 2] / ax["stroma"]); lrh = np.log(ch[:, 0] / ah["tumor"]) - np.log(ch[:, 2] / ah["stroma"])
        ok = np.isfinite(lrx) & np.isfinite(lrh) & (cx[:, 0] > 0) & (cx[:, 2] > 0) & (ch[:, 0] >= 5) & (ch[:, 2] >= 5)
        slope, icpt = np.polyfit(lrh[ok], lrx[ok], 1); res = lrx - (slope * lrh + icpt); mad = 1.4826 * np.median(np.abs(res[ok] - np.median(res[ok]))); zr = (res - np.median(res[ok])) / mad
        order = np.argsort(-np.where(ok, zr, -99))
        return dict(label=label, slope=float(slope), intercept=float(icpt), robust_sd=float(mad), n_genes=int(ok.sum()),
                    genes={g: dict(z=float(zr[common.index(g)]), residual=float(res[common.index(g)]), lr_xenium=float(lrx[common.index(g)]), lr_hd=float(lrh[common.index(g)]), ok=bool(ok[common.index(g)]),
                                   n_xen_tumor=int(cx[common.index(g), 0]), n_xen_stroma=int(cx[common.index(g), 2]), n_hd_tumor=int(ch[common.index(g), 0]), n_hd_stroma=int(ch[common.index(g), 2])) for g in targets + ["ACTA2"]},
                    top10=[dict(gene=common[i], z=float(zr[i])) for i in order[:10]])
    cx = np.array([regc[dic.index(g)] for g in common], float); ch = np.array([regcH[genes408.index(g)] for g in common], float)
    cxc = np.array([corec[dic.index(g)] for g in common], float); chc = np.array([corecH[genes408.index(g)] for g in common], float)
    rt_full = ratio_test(cx, ch, areaX, areaH, "whole regions"); rt_core = ratio_test(cxc, chc, core_area["xenium"], core_area["hd"], f"region cores (>= {a.core_um:g} µm from any boundary)")
    # mast set coherence: SD of z over CORE7 vs over the outlier genes
    # ---- (c) pixel co-marker intercept tests
    pix_tests = {}
    for g in targets:
        co = [c for c in CORE4 if c != g]
        pix_tests[g] = {}
        for region in ("tumor", "stroma"):
            pix_tests[g][region] = dict(comarkers=co,
                                        xenium=pixel_intercept(regX, pixX_gene[g], sum(pixX_gene[c] for c in co), pix_allX, region, a.n_boot),
                                        hd=pixel_intercept(regH, pixH_gene[g], sum(pixH_gene[c] for c in co), pix_allH, region, a.n_boot))
            for plat in ("xenium", "hd"): pix_tests[g][region][plat].pop("binned_means", None)
        print(f"  intercept {g}: tumor b_xen = {pix_tests[g]['tumor']['xenium']['b_zero_comarker']:.2f} ± {pix_tests[g]['tumor']['xenium']['b_zero_comarker_sd']:.2f}, b_hd = {pix_tests[g]['tumor']['hd']['b_zero_comarker']:.2f}; stroma b_xen = {pix_tests[g]['stroma']['xenium']['b_zero_comarker']:.2f}", flush=True)
    # ---- (d) compression prediction inside the tumor region (region tensors) and whole section (global tensors)
    comp = {}
    for scope, paths in (("tumor", {p: DATA / f"twopoint/regions/{p}_tumor.npz" for p in ("xenium", "hd")}), ("global", TWOPOINT)):
        zt = {p: np.load(paths[p], allow_pickle=True) for p in paths}
        fx = list(map(str, zt["xenium"]["features"])); fh = list(map(str, zt["hd"]["features"]))
        lgx, sdx, lgh, sdh = zt["xenium"]["log_g"], zt["xenium"]["sd"], zt["hd"]["log_g"], zt["hd"]["sd"]; rx = zt["xenium"]["r_central"]; rh = zt["hd"]["r_central"]
        pairs = [("HDC", "CPA3", "cross", "HDC"), ("CMA1", "CPA3", "cross", "CMA1"), ("MS4A2", "CPA3", "cross", "MS4A2"), ("CTSG", "CPA3", "cross", "CTSG"), ("HDC", "HDC", "self", "HDC"), ("CMA1", "CMA1", "self", "CMA1"),
                 ("XCL2", "GNLY", "cross", "XCL2"), ("XCL2", "XCL2", "self", "XCL2"), ("KIT", "CPA3", "cross", None), ("CPA3", "CPA3", "self", None), ("HDC", "CEACAM5", "cross", "HDC"), ("CPA3", "CEACAM5", "cross", None)]
        rows_all = {}
        for ga, gb, kind, btarget in pairs:
            b = pix_tests[btarget]["tumor" if scope == "tumor" else "tumor"]["xenium"]["b_zero_comarker"] if btarget else 0.0
            bsd = pix_tests[btarget]["tumor"]["xenium"]["b_zero_comarker_sd"] if btarget else 0.0
            rows = []
            for k in range(len(rx)):
                r = float(rx[k])
                if r < 6 or r > 60: continue
                kh = int(np.argmin(np.abs(rh - r)))
                gx = float(lgx[k, fx.index(ga), fx.index(gb)]); gh = float(lgh[kh, fh.index(ga), fh.index(gb)]); sx = float(sdx[k, fx.index(ga), fx.index(gb)]); sh = float(sdh[kh, fh.index(ga), fh.index(gb)])
                if not (np.isfinite(gx) and np.isfinite(gh)): continue
                gtrue = np.exp(gh)
                pred = float(predicted_compression(b, gtrue, kind)) if btarget else 0.0
                q = (np.exp(gx) - 1) / (gtrue - 1)
                b_exact = (float(1 - np.sqrt(q)) if q > 0 else None) if kind == "self" else (float(1 - q) if np.isfinite(q) else None)
                rows.append(dict(r_um=r, log_g_xenium=gx, sd_xenium=sx, log_g_hd=gh, sd_hd=sh, observed_diff=gx - gh, diff_sd=float(np.hypot(sx, sh)), predicted_diff=pred,
                                 predicted_range=[float(predicted_compression(min(b + bsd, 0.999), gtrue, kind)) if btarget else 0.0, float(predicted_compression(max(b - bsd, 0), gtrue, kind)) if btarget else 0.0], b_exact=b_exact))
            rows_all[f"{ga}:{gb}"] = dict(kind=kind, b_used=float(b), b_sd=float(bsd), rows=rows)
        comp[scope] = rows_all
    out = dict(seconds=time.time() - t_all, core_um=a.core_um, areas_mm2=dict(xenium=areaX, hd=areaH), core_areas_mm2=core_area, ratio_test=dict(whole=rt_full, core=rt_core), distance_to_boundary=dist_stats,
               pixel_test=dict(grid_um=regX["grid_um"], tile_um=TILE_UM, n_boot="closed", uncertainty="closed", comarkers_base=CORE4, targets=pix_tests), compression=comp,
               xenium_counts={g: int(regc[dic.index(g)].sum()) for g in targets}, hd_counts={g: int(regcH[genes408.index(g)].sum()) for g in targets})
    jdump(out, OUT / "mastcells_background.json")
    # ---- quick-look
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, rt in zip(axs[:2], (rt_full, rt_core)):
        gz = rt["genes"]
        ax.bar(range(len(targets)), [gz[g]["z"] for g in targets], color=["C3" if g in ("HDC", "CMA1", "XCL2") else "C0" for g in targets]); ax.set_xticks(range(len(targets))); ax.set_xticklabels(targets, rotation=60, fontsize=7)
        ax.axhline(0, c="k", lw=0.5); ax.axhline(2, c="gray", lw=0.5, ls="--"); ax.set_ylabel("z of Xenium log(tumor/stroma) residual vs the 408-gene line"); ax.set_title(rt["label"] + f" (slope {rt['slope']:.2f}, scatter {rt['robust_sd']:.2f})", fontsize=9)
    bx = [pix_tests[g]["tumor"]["xenium"]["b_zero_comarker"] for g in targets]; bxs = [pix_tests[g]["tumor"]["xenium"]["b_zero_comarker_sd"] for g in targets]
    bh = [pix_tests[g]["tumor"]["hd"]["b_zero_comarker"] for g in targets]; bs = [pix_tests[g]["stroma"]["xenium"]["b_zero_comarker"] for g in targets]
    xx = np.arange(len(targets)); axs[2].bar(xx - 0.25, bx, 0.25, yerr=bxs, label="Xenium, tumor"); axs[2].bar(xx, bs, 0.25, label="Xenium, stroma"); axs[2].bar(xx + 0.25, bh, 0.25, label="HD, tumor (estimator floor)")
    axs[2].set_xticks(xx); axs[2].set_xticklabels(targets, rotation=60, fontsize=7); axs[2].set_ylabel("b = fraction of calls in pixels without co-markers"); axs[2].legend(fontsize=7); axs[2].set_title("pixel co-marker intercept (co-markers CPA3/KIT/MS4A2/CTSG)", fontsize=9)
    fig.tight_layout(); fig.savefig(QL / "artifacts_mast_background.png", dpi=130)
    print("ratio test whole:", json.dumps({g: round(v["z"], 1) for g, v in rt_full["genes"].items()}), "core:", json.dumps({g: round(v["z"], 1) for g, v in rt_core["genes"].items()}))
    print("core top10:", [(t["gene"], round(t["z"], 1)) for t in rt_core["top10"]])
    print("distance to boundary:", json.dumps(dist_stats, indent=1, default=float))
    for scope in comp:
        for k, v in comp[scope].items():
            print(scope, k, f"b={v['b_used']:.2f}", [(round(d["r_um"], 1), round(d["observed_diff"], 2), round(d["diff_sd"], 2), round(d["predicted_diff"], 2), None if d["b_exact"] is None else round(d["b_exact"], 2)) for d in v["rows"][:4]])
    print(f"done in {time.time()-t_all:.0f} s")


# --------------------------------------------------------------------------------------------------------------- isolation
def isolation_fractions(xy_t, w_t, xy_c, w_c, radii=(5.0, 10.0)):
    """Fraction (UMI-weighted) of target points without any co-marker point within r; co-marker weights > 0 only."""
    from scipy.spatial import cKDTree
    if len(xy_c) == 0 or len(xy_t) == 0: return {f"{r:g}": None for r in radii}, np.ones(len(xy_t), bool)
    tree = cKDTree(xy_c); out = {}; iso5 = None
    for r in radii:
        n = tree.query_ball_point(xy_t, r, return_length=True)
        iso = n == 0; out[f"{r:g}"] = float(np.sum(w_t[iso]) / np.sum(w_t))
        if iso5 is None: iso5 = iso
    return out, iso5


def stage_isolation(a):
    """Molecule-level co-marker test (replaces the pixel intercept for sparse genes): for every candidate gene, the fraction
    of its molecules / UMIs with no *other* CORE7 molecule within 5 / 10 µm, per region and platform, with a 500 µm tile
    bootstrap on Xenium. For a gene expressed only in mast cells this fraction is the co-marker detection floor (set by the
    CPA3 / KIT reference); the excess over the floor is the fraction of calls that are not in mast cells:
    b_iso = (iso_g − iso_ref) / (1 − iso_ref). The same for XCL2 against the NK block (CCL5/KLRD1/GNLY/NKG7)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import pyarrow.parquet as pq, scipy.sparse as sp
    from platform_artifacts import predicted_compression
    t_all = time.time()
    NK = MARKERS["NK"]; targets = CAND9 + NOTMAST
    out = dict(radii_um=[5.0, 10.0], targets=targets, comarkers="CORE7 minus the target (XCL2: CORE7); NK test: NK block minus the target", platforms={})
    # ---- Xenium P2 (+ P1 / P5 replication, all-tissue only)
    for sec in ("p2", "p1", "p5"):
        t0 = time.time()
        tb = pq.read_table(DATA / f"processed/xenium_crc_{sec}/molecules.parquet", columns=["x", "y", "feature"], filters=[("feature", "in", targets + NK)])
        feat = np.array(tb.column("feature").to_numpy(zero_copy_only=False).astype(str)); xy = np.stack([tb.column("x").to_numpy(), tb.column("y").to_numpy()], 1).astype(np.float64)
        reg = region_grid("xenium") if sec == "p2" else None
        lab = label_grid(xy, reg)[0] if reg is not None else np.zeros(len(xy), np.int8)
        tinv, nt = tile_keys(xy)
        res = {}
        for g in targets + ["GNLY", "NKG7"]:
            co = [c for c in (CORE7 if g in targets else NK) if c != g]
            mt = feat == g; mc = np.isin(feat, co)
            fr, iso5 = isolation_fractions(xy[mt], np.ones(mt.sum()), xy[mc], np.ones(mc.sum()))
            row = dict(n=int(mt.sum()), all=fr)
            # tile bootstrap of the 5 µm isolated fraction (all tissue)
            ti = tinv[mt]
            def st(w): return np.array([np.sum(w[iso5]) / max(np.sum(w), 1)])
            base, sd, lo, hi = tile_boot(st, ti, nt, a.n_boot); row["iso5_sd"] = float(sd[0])
            if reg is not None:
                for i, r in enumerate(REGIONS):
                    m = lab[mt] == i
                    row[r] = dict(n=int(m.sum()), iso5=float(iso5[m].mean()) if m.any() else None)
            if g == "XCL2":   # XCL2 against the NK block as well
                mc2 = np.isin(feat, NK); fr2, _ = isolation_fractions(xy[mt], np.ones(mt.sum()), xy[mc2], np.ones(mc2.sum())); row["vs_NK"] = fr2
            res[g] = row
        out["platforms"][f"xenium_{sec}"] = dict(genes=res, n_tiles=int(nt), seconds=time.time() - t0)
        print(f"[xenium {sec}] iso5:", {g: round(v["all"]["5"], 3) for g, v in res.items()}, f"{time.time()-t0:.0f} s", flush=True)
    # ---- HD P2 (2 µm bins, UMI-weighted)
    t0 = time.time()
    z = np.load(DATA / "processed/hd_crc_p2/bins_square_002um.npz", allow_pickle=True)
    feats = list(map(str, z["features"])); X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])); coords = z["coords_um"].astype(np.float64)
    regH = region_grid("hd"); labH = label_grid(coords, regH)[0]
    cols = {g: np.asarray(X[:, feats.index(g)].todense()).ravel() for g in targets + NK}
    res = {}
    for g in targets + ["GNLY", "NKG7"]:
        co = [c for c in (CORE7 if g in targets else NK) if c != g]
        wt = cols[g]; mt = wt > 0; wc = sum(cols[c] for c in co); mc = wc > 0
        fr, iso5 = isolation_fractions(coords[mt], wt[mt], coords[mc], wc[mc])
        row = dict(n=int(wt.sum()), all=fr)
        for i, r in enumerate(REGIONS):
            m = labH[mt] == i; row[r] = dict(n=int(wt[mt][m].sum()), iso5=float(np.sum(wt[mt][m][iso5[m]]) / max(np.sum(wt[mt][m]), 1)) if m.any() else None)
        if g == "XCL2":
            wc2 = sum(cols[c] for c in NK); fr2, _ = isolation_fractions(coords[mt], wt[mt], coords[wc2 > 0], wc2[wc2 > 0]); row["vs_NK"] = fr2
        res[g] = row
    out["platforms"]["hd_p2"] = dict(genes=res, seconds=time.time() - t0)
    print("[hd p2] iso5:", {g: round(v["all"]["5"], 3) for g, v in res.items()}, flush=True)
    # ---- b_iso relative to the CPA3/KIT reference, per platform and region; compression prediction with b_iso (tumor)
    b_iso = {}
    for plat, d in out["platforms"].items():
        ref_all = np.mean([d["genes"]["CPA3"]["all"]["5"], d["genes"]["KIT"]["all"]["5"]])
        b_iso[plat] = {}
        for g in targets:
            row = dict(all=float((d["genes"][g]["all"]["5"] - ref_all) / (1 - ref_all)))
            if "tumor" in d["genes"][g]:
                for r in REGIONS:
                    ref_r = np.mean([d["genes"]["CPA3"][r]["iso5"], d["genes"]["KIT"][r]["iso5"]]); v = d["genes"][g][r]["iso5"]
                    row[r] = float((v - ref_r) / (1 - ref_r)) if v is not None else None
            b_iso[plat][g] = row
    out["b_iso"] = b_iso
    bgf = OUT / "mastcells_background.json"
    if bgf.exists():
        comp = json.load(open(bgf))["compression"]; pred = {}
        for scope in comp:
            pred[scope] = {}
            for pair, v in comp[scope].items():
                ga = pair.split(":")[0]; kind = v["kind"]
                bt = b_iso["xenium_p2"].get(ga, {}).get("tumor" if scope == "tumor" else "all")
                if ga in ("KIT", "CPA3") or bt is None: bt = 0.0
                bt = float(np.clip(bt, 0, 0.999))
                rows = [dict(r_um=d["r_um"], observed_diff=d["observed_diff"], diff_sd=d["diff_sd"], predicted_iso=float(predicted_compression(bt, np.exp(d["log_g_hd"]), kind)), b_exact=d["b_exact"]) for d in v["rows"]]
                pred[scope][pair] = dict(kind=kind, b_iso=bt, rows=rows)
        out["compression_with_b_iso"] = pred
    out["seconds"] = time.time() - t_all
    jdump(out, OUT / "mastcells_isolation.json")
    fig, ax = plt.subplots(figsize=(8, 4)); xx = np.arange(len(targets))
    for k, (plat, col) in enumerate((("xenium_p2", "C1"), ("xenium_p1", "peru"), ("xenium_p5", "goldenrod"), ("hd_p2", "C2"))):
        d = out["platforms"][plat]["genes"]; ax.bar(xx + (k - 1.5) * 0.2, [d[g]["all"]["5"] for g in targets], 0.2, color=col, label=plat, yerr=[d[g].get("iso5_sd", 0) for g in targets])
    ax.set_xticks(xx); ax.set_xticklabels(targets, rotation=45); ax.set_ylabel("fraction of molecules with no other mast gene within 5 µm"); ax.legend(fontsize=8); ax.set_title("molecule-level co-marker isolation (CORE7 minus target)")
    fig.tight_layout(); fig.savefig(QL / "artifacts_mast_isolation.png", dpi=130)
    print("b_iso:", json.dumps(b_iso, indent=1, default=float))
    if "compression_with_b_iso" in out:
        for scope in out["compression_with_b_iso"]:
            for pair, v in out["compression_with_b_iso"][scope].items():
                print(scope, pair, f"b_iso={v['b_iso']:.2f}", [(round(d["r_um"], 1), round(d["observed_diff"], 2), round(d["predicted_iso"], 2)) for d in v["rows"][:3]])
    print(f"done in {time.time()-t_all:.0f} s")


# ------------------------------------------------------------------------------------------------------------------- table
def stage_table(a):
    tp = json.load(open(OUT / "mastcells_twopoint.json")); ce = json.load(open(OUT / "mastcells_cells.json")); hd = json.load(open(OUT / "mastcells_hd.json")); bg = json.load(open(OUT / "mastcells_background.json"))
    lines = ["# How mast cells appear in each representation (CRC P2; Xenium vs Visium HD serial sections)", "",
             "| representation | platform | what is measured | number ± error |", "|---|---|---|---|"]
    for plat in ("xenium", "hd"):
        res = tp["platforms"][plat]; r = np.array(res["r_central"]); b = res["block"]["mast7_cross"]; k = int(np.argmin(np.abs(r - (6.6 if plat == "hd" else 2.4))))
        rk = [x for x in res["ranks"] if abs(x["r_um"] - r[k]) < 0.1][0]["sets"]["mast7_cross"]
        lines.append(f"| molecule 2-point block (21 CORE7 pairs) | {plat} | pooled log g at r = {r[k]:.1f} µm; rank among {[x for x in res['ranks'] if abs(x['r_um'] - r[k]) < 0.1][0]['n_pairs_ranked']:,} pairs | {b['pooled_log_g'][k]:.2f} ± {b['pooled_sd'][k]:.2f}; median rank {rk['median_rank']:.0f}, max {rk['max_rank']}, {rk['n_in_top100']}/{rk['n_valid']} in the top 100 |")
    s = ce["spots"]; lines.append(f"| molecule DBSCAN spots (eps {s['eps_um']:g} µm, ≥ {s['min_samples']} molecules) | xenium | spots per mm²; molecules per spot; diameter | {s['spots_per_mm2']:.0f} ± {s['spots_per_mm2_sd']:.0f} / mm² ({s['n_spots']:,} spots); {s['molecules_per_spot']['median']:.0f} (IQR {s['molecules_per_spot']['q25']:.0f}–{s['molecules_per_spot']['q75']:.0f}); {s['diameter_um']['median']:.1f} µm (IQR {s['diameter_um']['q25']:.1f}–{s['diameter_um']['q75']:.1f}); {s['frac_points_in_spots']*100:.0f} % of mast molecules in spots |")
    s = hd["spots_2um"]; lines.append(f"| 2 µm-bin DBSCAN spots (eps {s['eps_um']:g} µm, ≥ {s['min_samples']} UMIs) | hd | spots per mm²; UMIs per spot | {s['spots_per_mm2']:.0f} ± {s['spots_per_mm2_sd']:.0f} / mm² ({s['n_spots']:,}); {s['molecules_per_spot']['median']:.0f} (IQR {s['molecules_per_spot']['q25']:.0f}–{s['molecules_per_spot']['q75']:.0f}); {s['frac_points_in_spots']*100:.0f} % of mast UMIs in spots |")
    cv = ce["cell_view"]
    lines.append(f"| segmented cells (Xenium cell_feature_matrix) | xenium | cells with ≥ 1 / 2 / 3 / 5 mast transcripts (of {cv['n_cells']['value']:.0f}) | {cv['cells_ge1']['value']:.0f} / {cv['cells_ge2']['value']:.0f} / **{cv['cells_ge3']['value']:.0f} ± {cv['cells_ge3']['sd']:.0f}** ({cv['cells_ge3']['per_mm2']:.0f} ± {cv['cells_ge3']['per_mm2_sd']:.0f} / mm²) / {cv['cells_ge5']['value']:.0f}; median mast fraction in ≥ 3 cells {cv['mast_frac_in_cells_ge3']['median']*100:.0f} % |")
    lines.append(f"| transcript assignment | xenium | mast transcripts inside cells (panel median); in cells with mast fraction < 10 % | {cv['in_cell_fraction_mast_set']*100:.0f} % (panel {ce['panel_reference']['frac_in_cell']['median']*100:.0f} %); {cv['frac_mast_in_cells_mastfrac_lt10pct']['value']*100:.0f} ± {cv['frac_mast_in_cells_mastfrac_lt10pct']['sd']*100:.0f} % mis-assigned / ambient |")
    for res, d in ce["leiden"]["resolutions"].items():
        bc = d["best_cluster"]; lines.append(f"| leiden cluster (scanpy standard, res {res}) | xenium | best cluster: size, purity (≥ 3 mast), recall | {bc['size']:,} cells, purity {bc['purity_mast3']:.2f}, recall {bc['recall_mast3']:.2f} → {'recovered' if d['recovered'] else 'not recovered'} ({d['n_clusters']} clusters) |")
    b8 = hd["bins_8um"]; lines.append(f"| 8 µm bins | hd | bins with ≥ 2 / 3 / 5 mast UMIs; share of mast UMIs in bins with mast fraction < 10 % | {b8['bins_ge2']:,} / {b8['bins_ge3']:,} ({b8['bins_ge3_per_mm2']:.0f} / mm²) / {b8['bins_ge5']:,}; {b8['frac_mast_in_bins_mastfrac_lt10pct']*100:.0f} % |")
    x8 = ce["xenium_bins_8um"]; lines.append(f"| 8 µm bins (Xenium molecules binned) | xenium | same | {x8['bins_ge2']:,} / {x8['bins_ge3']:,} ({x8['bins_ge3_per_mm2']:.0f} / mm²) / {x8['bins_ge5']:,}; {x8['frac_mast_in_bins_mastfrac_lt10pct']*100:.0f} % |")
    for res, d in hd["leiden_8um"]["resolutions"].items():
        bc = d["best_cluster"]; lines.append(f"| leiden cluster on 8 µm bins (2000 HVGs, res {res}) | hd | best cluster: size, purity (≥ 3 mast UMIs), recall | {bc['size']:,} bins, purity {bc['purity_mast3']:.2f}, recall {bc['recall_mast3']:.2f} → {'recovered' if d['recovered'] else 'not recovered'} |")
    lines += ["", "## Outlier status (Xenium vs HD tumor/stroma density ratio), z relative to the 408-gene line", "", "| gene | z whole regions | z region cores | b tumor (Xenium) | b tumor (HD floor) | Xenium calls | HD UMIs |", "|---|---|---|---|---|---|---|"]
    for g in CAND9 + NOTMAST:
        pt = bg["pixel_test"]["targets"][g]["tumor"]
        lines.append(f"| {g} | {bg['ratio_test']['whole']['genes'][g]['z']:.1f} | {bg['ratio_test']['core']['genes'][g]['z']:.1f} | {pt['xenium']['b_zero_comarker']:.2f} ± {pt['xenium']['b_zero_comarker_sd']:.2f} | {pt['hd']['b_zero_comarker']:.2f} | {bg['xenium_counts'][g]:,} | {bg['hd_counts'][g]:,} |")
    open(OUT / "mastcells_table.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["twopoint", "cells", "hd", "background", "isolation", "table"])
    ap.add_argument("--n-boot", type=int, default=None, help="ignored: tile-bootstrap uncertainties are closed-form");
    ap.add_argument("--keep-leiden", action="store_true", help="cells, hd: reuse the leiden block of the stored JSON (no reclustering)"); ap.add_argument("--min-pairs", type=float, default=10); ap.add_argument("--r-max-rank", type=float, default=12.0)
    ap.add_argument("--no-tiles", action="store_true", help="twopoint: skip the counts_tiles bootstrap (quick test)")
    ap.add_argument("--mast-set", default="core7", choices=["core7", "cand9"]); ap.add_argument("--eps", type=float, default=5.0); ap.add_argument("--min-samples", type=int, default=3)
    ap.add_argument("--resolutions", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0]); ap.add_argument("--subsample", type=int, default=None); ap.add_argument("--min-counts-hd", type=int, default=50)
    ap.add_argument("--core-um", type=float, default=50.0)
    a = ap.parse_args()
    {"twopoint": stage_twopoint, "cells": stage_cells, "hd": stage_hd, "background": stage_background, "isolation": stage_isolation, "table": stage_table}[a.stage](a)


if __name__ == "__main__":
    main()
