#!/usr/bin/env python
"""Preprint Fig. 3 (cell level): baselines on the segmented cells of the SpatialBench spleen Xenium sections, with the same
marker sets as the molecule-level operators (config/operators_spleen.yaml).

  cells     : cell typing from data/raw/<section>/cell_feature_matrix.h5 + cells.parquet (marker-fraction rule, see `call_cell_types`)
  squidpy   : sq.gr.spatial_neighbors (generic coordinates, Delaunay graph, the recommended setting for imaging-based data) +
              sq.gr.nhood_enrichment (z-scores, 1000 permutations, seed 0) + sq.gr.co_occurrence (interval = 30 bins to 300 µm;
              on a random subsample of --co-occ-cells cells because the implementation is O(N²) in memory and time)
  centroids : cross pair-correlation g_ab(r) on the cell centroids with the same scote.twopoint pipeline (level 0 at
              delta = 0.5 µm + lattice levels, 500 µm tiles, random-label reference, closed-form tile block bootstrap) and
              feature = cell type ("other" kept so that the reference uses all cells): molecule-level and cell-level functions
              are compared with an identical estimator and error model.
  fit       : three-parameter sigmoid g(r) = 1 + A / (1 + exp((r - r0)/w)) per pair function (A = contact-scale excess or
              deficit, r0 = transition scale, w = width), generalized least squares on 5-300 µm with the closed-form cross-shell
              covariance of log g, correlation shrunk by Ledoit-Wolf (`sigmoid`; scripts/gls_tiles.py; unregularized GLS as
              `sigmoid_gls_raw`, the SD-only fit as `sigmoid_diag`), for molecule-level and centroid-level functions of every section
              -> results/spleen_operators/baseline_functions.json
  compare   : CI widths at molecule vs cell level at matched r, Squidpy effect sizes and z-scores, group contrasts
              -> results/spleen_operators/baselines_compare.json

  python scripts/spleen_cell_baselines.py cells --section sb_xenium_WT713       # then squidpy, centroids; fit and compare once
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]
from spleen_operators import OPS  # noqa: E402
RES = REPO / "results/spleen_operators"; OUT = REPO / "data/twopoint/spleen_operators"
TYPES = ["B", "T", "GCB", "FDC", "Tfh", "other"]
MARKERS = dict(B=["Cd19", "Ighd", "Ighm"], T=["Cd3e", "Cd3d", "Trac", "Cd8a", "Cd4"], GCB=["Aicda", "Rgs13"], FDC=["Cr2", "Fcer2a"], Tfh_score=["Bcl6"], Tfh_gate=["Cd4"])
PAIRS = [("B", "T"), ("T", "FDC"), ("Tfh", "FDC"), ("B", "FDC"), ("GCB", "FDC"), ("GCB", "Tfh"), ("B", "B"), ("GCB", "GCB")]


def load_cells(section):
    import h5py, scipy.sparse as sp, pyarrow.parquet as pq
    raw = REPO / f"data/raw/{section}"
    with h5py.File(raw / "cell_feature_matrix.h5") as f:
        m = f["matrix"]; names = m["features/name"][:].astype(str); ftype = m["features/feature_type"][:].astype(str)
        X = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:])).T.tocsr()   # cells x features
        barcodes = m["barcodes"][:].astype(str)
    keep = ftype == "Gene Expression"; X = X[:, keep]; genes = names[keep]
    cells = pq.read_table(raw / "cells.parquet", columns=["cell_id", "x_centroid", "y_centroid", "transcript_counts", "cell_area"]).to_pandas().set_index("cell_id").loc[barcodes]
    return X, list(genes), cells


def call_cell_types(X, genes, min_counts=10, quantile=0.8, min_frac=0.1, gcb_frac=0.05):
    """Marker-fraction rule mirroring the molecule operators (config/operators_spleen.yaml):
    f_k = counts of marker set k / total panel counts of the cell. Priority: (1) FDC: cells with >= 1 Cr2/Fcer2a count whose
    f_FDC is in the top (1 - quantile) of those cells (the molecule rule: Cr2/Fcer2a molecule in the top 20 % of the local
    Cr2+Fcer2a fraction); (2) Tfh: Cd4-expressing cells whose Bcl6 fraction is in the top 20 % of Cd4-expressing cells;
    (3) GCB: f_GCB >= gcb_frac (Aicda/Rgs13 molecules are GCB by gene at molecule level); (4) B or T by the larger of f_B, f_T
    if >= min_frac; else 'other'. Cells with < min_counts transcripts are 'other'."""
    gi = {g: i for i, g in enumerate(genes)}
    tot = np.asarray(X.sum(1)).ravel().astype(float)
    frac = lambda gs: np.asarray(X[:, [gi[g] for g in gs if g in gi]].sum(1)).ravel() / np.maximum(tot, 1)
    fB, fT, fG, fF, fBcl6, fCd4 = (frac(MARKERS[k]) for k in ("B", "T", "GCB", "FDC", "Tfh_score", "Tfh_gate"))
    lab = np.full(X.shape[0], "other", dtype=object); ok = tot >= min_counts
    bt = np.where(fB >= fT, "B", "T"); lab[ok & (np.maximum(fB, fT) >= min_frac)] = bt[ok & (np.maximum(fB, fT) >= min_frac)]
    lab[ok & (fG >= gcb_frac)] = "GCB"
    cd4 = ok & (fCd4 > 0); thr_tfh = np.quantile(fBcl6[cd4], quantile); lab[cd4 & (fBcl6 >= thr_tfh) & (fBcl6 > 0)] = "Tfh"
    fdc = ok & (fF > 0); thr_fdc = np.quantile(fF[fdc], quantile); lab[fdc & (fF >= thr_fdc)] = "FDC"
    info = dict(rule="see call_cell_types docstring", min_counts=min_counts, quantile=quantile, min_frac=min_frac, gcb_frac=gcb_frac,
                threshold_fdc_fraction=float(thr_fdc), threshold_tfh_bcl6_fraction=float(thr_tfh), n_cells=int(len(lab)),
                counts={t: int((lab == t).sum()) for t in TYPES}, frac_assigned=float((lab != "other").mean()))
    return lab, info


def stage_cells(a):
    t0 = time.time(); X, genes, cells = load_cells(a.section)
    lab, info = call_cell_types(X, genes)
    df = pd.DataFrame(dict(x=cells["x_centroid"].to_numpy(), y=cells["y_centroid"].to_numpy(), cell_type=lab, counts=cells["transcript_counts"].to_numpy()))
    OUT.mkdir(parents=True, exist_ok=True); df.to_parquet(OUT / f"cells_{a.section}.parquet")
    info.update(section=a.section, seconds=time.time() - t0); RES.mkdir(parents=True, exist_ok=True)
    json.dump(info, open(RES / f"cells_{a.section}.json", "w"), indent=1); print(json.dumps(info))


def stage_squidpy(a):
    import anndata as ad, squidpy as sq, scipy.sparse as sp
    t0 = time.time(); df = pd.read_parquet(OUT / f"cells_{a.section}.parquet")
    adata = ad.AnnData(X=sp.csr_matrix((len(df), 1)), obs=pd.DataFrame(dict(cell_type=pd.Categorical(df["cell_type"], categories=TYPES)), index=df.index.astype(str)))
    adata.obsm["spatial"] = df[["x", "y"]].to_numpy()
    sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True); t1 = time.time()
    sq.gr.nhood_enrichment(adata, cluster_key="cell_type", n_perms=1000, seed=0, n_jobs=1, numba_parallel=True, show_progress_bar=False); t2 = time.time()   # n_jobs > 1 hands read-only memmaps to the numba kernel (TypeError at 360k cells)
    ne = adata.uns["cell_type_nhood_enrichment"]; z = ne["zscore"]; cnt = ne["count"]
    rng = np.random.default_rng(0); sub = adata[rng.choice(adata.n_obs, min(a.co_occ_cells, adata.n_obs), replace=False)].copy()
    interval = np.linspace(0, 300, 31)
    sq.gr.co_occurrence(sub, cluster_key="cell_type", interval=interval); t3 = time.time()
    co = sub.uns["cell_type_co_occurrence"]
    res = dict(section=a.section, types=TYPES, n_cells=int(adata.n_obs), graph="generic coordinates, Delaunay triangulation (sq.gr.spatial_neighbors coord_type='generic', delaunay=True)",
               n_edges=int(adata.obsp["spatial_connectivities"].nnz // 2), nhood_zscore=z.tolist(), nhood_count=cnt.tolist(), nhood_settings=dict(n_perms=1000, seed=0),
               co_occurrence=dict(n_cells=int(sub.n_obs), interval_um=co["interval"].tolist(), occ=co["occ"].tolist()),
               seconds=dict(graph=t1 - t0, nhood_enrichment=t2 - t1, co_occurrence=t3 - t2, total=t3 - t0))
    json.dump(res, open(RES / f"squidpy_{a.section}.json", "w"), indent=1)
    print(f"{a.section}: nhood z B-T {z[0,1]:+.1f}, T-FDC {z[1,3]:+.1f}, Tfh-FDC {z[4,3]:+.1f}, B-FDC {z[0,3]:+.1f}; graph {t1-t0:.0f}s nhood {t2-t1:.0f}s co_occ {t3-t2:.0f}s")


def stage_centroids(a):
    from scote import CPUOptions, twopoint
    t0 = time.time(); df = pd.read_parquet(OUT / f"cells_{a.section}.parquet")
    xy = df[["x", "y"]].to_numpy(dtype=np.float64); codes = pd.Categorical(df["cell_type"], categories=TYPES).codes.astype(np.int64)
    res = twopoint(xy, codes, names=TYPES, delta=0.5, shells_per_level=8, n_levels=a.n_levels, tile_um=500.0,
                   backend=CPUOptions(mode=a.cpu_mode), threads=a.threads or None, keep_tiles=True)
    res.save(OUT / f"centroids_{a.section}.npz", compress=True)
    json.dump(dict(section=a.section, n_cells=int(len(xy)), n_tiles=res.n_tiles, timings=res.timings, seconds=time.time() - t0), open(RES / f"centroids_{a.section}.json", "w"), indent=1, default=float)
    print(f"{a.section}: centroid run {time.time()-t0:.0f}s, {res.timings}")


def merged_log_g(d, ga, gb, with_cov=False):
    """log g_ab from a stored run with the closed-form tile-bootstrap SD, interval and reliability flag
    (spleen_operators.merged_closed_form; dict with log_g, sd, lo, hi, t_eff, flag[, cov])."""
    from spleen_operators import merged_log_g as _mlg
    return _mlg(d, ga, gb, with_cov=with_cov)


def sigmoid(r, A, r0, w):
    return 1 + A / (1 + np.exp((r - r0) / w))


def fit_sigmoid(r, lg, sd, rmin=5.0, rmax=300.0, influence=None, shrink=None, flag=None):
    """Three-parameter sigmoid g(r) = 1 + A / (1 + exp((r - r0) / w)) fitted to g = exp(log g) on rmin..rmax.
    influence=None: weighted least squares with the closed-form SDs (sigma = g sd; the diagonal version).
    influence = (T, K) per-tile influence of log g: generalized least squares with the joint covariance of the fitted bins,
    Cov(g_k, g_k') = g_k g_k' Cov(log g_k, log g_k') (delta method), Cov(log g) = sum_t l_t l_t^T with the correlation
    matrix shrunk toward the identity (scripts/gls_tiles.py; shrink=None: Ledoit-Wolf intensity, 0: unregularized);
    curve_fit with a 2-D sigma minimizes r^T C^-1 r. Bins with reliability flag 2 (T_eff < 5) are left out. Parameter
    errors from the curve_fit covariance scaled by chi2_red (absolute_sigma=False)."""
    from scipy.optimize import curve_fit
    from gls_tiles import shrunk_cov
    m = np.isfinite(lg) & np.isfinite(sd) & (sd > 0) & (r >= rmin) & (r <= rmax)
    if flag is not None: m &= np.asarray(flag) < 2
    if m.sum() < 5: return None
    g = np.exp(lg[m]); sg = g * sd[m]; extra = {}
    if influence is None:
        sigma = sg
    else:
        cov_lg, s_used, cond = shrunk_cov(np.asarray(influence)[:, m], shrink)
        sigma = cov_lg * np.outer(g, g)
        extra = dict(shrinkage=s_used, cond_corr=cond)
    p0 = (g[0] - 1, 30.0, 10.0)
    try:
        p, pcov = curve_fit(sigmoid, r[m], g, p0=p0, sigma=sigma, bounds=([-1, 0.5, 0.2], [100, 1000, 500]), maxfev=20000)
    except Exception as e:
        return dict(error=str(e))
    res_ = sigmoid(r[m], *p) - g
    chi2 = float(res_ @ np.linalg.solve(sigma, res_)) if influence is not None else float(np.sum((res_ / sg) ** 2))
    err = np.sqrt(np.diag(pcov))
    return dict(A=float(p[0]), r0_um=float(p[1]), w_um=float(p[2]), A_err=float(err[0]), r0_err=float(err[1]), w_err=float(err[2]), chi2=chi2,
                n_bins=int(m.sum()), chi2_red=chi2 / max(1, m.sum() - 3), method="diagonal" if influence is None else "gls", **extra)


def stage_fit(a):
    sections = [s for g in OPS["sections"] for s in OPS["sections"][g]]   # all groups of the operator YAML (WT, KO, Ctrl)
    res = {}
    for s in sections:
        for level, path in (("molecules", OUT / f"{s}.npz"), ("centroids", OUT / f"centroids_{s}.npz")):
            if not path.exists(): continue
            d = np.load(path, allow_pickle=True); r = d["r_central"]
            for ga, gb in PAIRS:
                m = merged_log_g(d, ga, gb, with_cov=True)
                lg, sd = m["log_g"], m["sd"]
                res.setdefault(f"{ga}:{gb}", {}).setdefault(s, {})[level] = dict(
                    r_um=r.tolist(), log_g=lg.tolist(), sd=sd.tolist(), lo=m["lo"].tolist(), hi=m["hi"].tolist(), t_eff=m["t_eff"].tolist(),
                    flag=m["flag"].tolist(), uncertainty="closed",
                    sigmoid=fit_sigmoid(r, lg, sd, influence=m["influence"], flag=m["flag"]),                       # GLS, Ledoit-Wolf shrinkage
                    sigmoid_gls_raw=fit_sigmoid(r, lg, sd, influence=m["influence"], shrink=0.0, flag=m["flag"]),   # GLS, no shrinkage
                    sigmoid_diag=fit_sigmoid(r, lg, sd, flag=m["flag"]))                                            # SDs only
    json.dump(res, open(RES / "baseline_functions.json", "w"), indent=1, default=float)
    for pair, per in res.items():
        for s, lv in per.items():
            print(pair, s, {k: (None if v["sigmoid"] is None else {kk: round(vv, 2) for kk, vv in v["sigmoid"].items() if kk in ("A", "r0_um", "w_um", "chi2_red")}) for k, v in lv.items()})


def stage_compare(a):
    groups = {s: g for g in OPS["sections"] for s in OPS["sections"][g]}   # all groups (WT, KO, Ctrl)
    fn = json.load(open(RES / "baseline_functions.json"))
    out = dict(uncertainty="closed", ci_width={}, squidpy={}, cells={}, timing={}, group_contrasts={})
    targets = (5, 10, 20, 50, 100, 200)
    for pair in ("B:T", "T:FDC", "Tfh:FDC", "B:FDC", "GCB:FDC", "GCB:Tfh"):
        per = fn.get(pair, {}); rows = {}
        for s, lv in per.items():
            if "centroids" not in lv or "molecules" not in lv: continue
            r = np.array(lv["molecules"]["r_um"])
            for t in targets:
                k = int(np.argmin(np.abs(r - t)))
                mo, ce = lv["molecules"], lv["centroids"]
                rows.setdefault(t, []).append(dict(section=s, group=groups[s], r_um=float(r[k]), mol_log_g=mo["log_g"][k], mol_ci=mo["hi"][k] - mo["lo"][k], cen_log_g=ce["log_g"][k], cen_ci=ce["hi"][k] - ce["lo"][k]))
        out["ci_width"][pair] = {str(t): v for t, v in rows.items()}
    for s in groups:
        p = RES / f"squidpy_{s}.json"
        if p.exists():
            q = json.load(open(p)); z = np.array(q["nhood_zscore"]); ti = {t: i for i, t in enumerate(q["types"])}
            occ = np.array(q["co_occurrence"]["occ"]); iv = np.array(q["co_occurrence"]["interval_um"])
            cnt = np.array(q["nhood_count"], dtype=float); cells = json.load(open(RES / f"cells_{s}.json"))["counts"]; n = np.array([cells[t] for t in q["types"]], dtype=float)
            E = cnt.sum() * np.outer(n, n) / n.sum() ** 2        # expected edge counts under a label permutation (Squidpy's null) -> effect size log(O/E)
            with np.errstate(all="ignore"):
                loe = np.log(cnt / E)
            out["squidpy"][s] = dict(group=groups[s], nhood_log_OE={f"{a_}:{b_}": float(loe[ti[a_], ti[b_]]) for a_, b_ in [("B", "T"), ("T", "FDC"), ("Tfh", "FDC"), ("B", "FDC"), ("GCB", "FDC"), ("GCB", "Tfh"), ("B", "B"), ("GCB", "GCB")]}, nhood_z={f"{a_}:{b_}": float(z[ti[a_], ti[b_]]) for a_, b_ in [("B", "T"), ("T", "FDC"), ("Tfh", "FDC"), ("B", "FDC"), ("GCB", "FDC"), ("GCB", "Tfh"), ("B", "B"), ("GCB", "GCB")]},
                                     co_occ_log={f"{a_}:{b_}": np.log(np.maximum(occ[ti[a_], ti[b_]], 1e-9)).tolist() for a_, b_ in [("B", "T"), ("T", "FDC"), ("Tfh", "FDC"), ("B", "FDC")]}, co_occ_r_um=((iv[1:] + iv[:-1]) / 2).tolist(),
                                     seconds=q["seconds"], n_cells=q["n_cells"])
        p = RES / f"cells_{s}.json"
        if p.exists():
            c = json.load(open(p)); out["cells"][s] = dict(counts=c["counts"], frac_assigned=c["frac_assigned"], n_cells=c["n_cells"])
        for kind in ("centroids",):
            p = RES / f"{kind}_{s}.json"
            if p.exists(): out["timing"].setdefault(s, {})[kind] = json.load(open(p))["seconds"]
    tl = RES / "baselines_timing.jsonl"
    if tl.exists():
        out["timing"]["timed_run"] = [json.loads(l) for l in tl.read_text().splitlines() if l.strip()]
    # group contrasts (WT - KO, Ctrl - WT, Ctrl - KO, WT+Ctrl - KO) on the baselines' effect sizes: Squidpy log(O/E) (replicate spread only,
    # no per-section error model) and the centroid / molecule log g at r ~ 10 µm (tile bootstrap + replicate spread), as in compare3
    from spleen_operators import CONTRASTS3, contrast, fmt_c
    gnames = list(OPS["sections"])
    def add_contrast(key, vals, sds):
        pooled = lambda name: (sum((vals.get(x, []) for x in name.split("+")), []), sum((sds.get(x, []) for x in name.split("+")), []))
        e = dict(values=vals, contrasts={})
        for g1, g2 in CONTRASTS3:
            v1, s1 = pooled(g1); v2, s2 = pooled(g2)
            if v1 and v2: e["contrasts"][f"{g1}-{g2}"] = contrast(v1, s1, v2, s2)
        out["group_contrasts"][key] = e
    for pair in ("B:T", "T:FDC", "Tfh:FDC", "B:FDC", "GCB:GCB"):
        vals = {g: [q["nhood_log_OE"][pair] for s, q in out["squidpy"].items() if q["group"] == g and np.isfinite(q["nhood_log_OE"][pair])] for g in gnames}
        add_contrast(f"squidpy_logOE {pair}", vals, {g: [0.0] * len(v) for g, v in vals.items()})
        for level in ("centroids", "molecules"):
            vals, sds = {g: [] for g in gnames}, {g: [] for g in gnames}
            for s, lv in fn.get(pair, {}).items():
                if level not in lv or s not in groups: continue
                r = np.array(lv[level]["r_um"]); k = int(np.argmin(np.abs(r - 10)))
                if np.isfinite(lv[level]["log_g"][k]): vals[groups[s]].append(lv[level]["log_g"][k]); sds[groups[s]].append(lv[level]["sd"][k])
            add_contrast(f"{level} log g(10 µm) {pair}", vals, sds)
    # sigmoid amplitude A (GLS fit with the cross-shell covariance; diagonal fit for comparison) with its fit error as the per-section SD
    for pair in ("B:T", "T:FDC", "Tfh:FDC", "B:FDC"):
        for level in ("molecules", "centroids"):
            for fkey in ("sigmoid", "sigmoid_gls_raw", "sigmoid_diag"):
                vals, sds = {g: [] for g in gnames}, {g: [] for g in gnames}
                for s, lv in fn.get(pair, {}).items():
                    f = (lv.get(level) or {}).get(fkey)
                    if s in groups and f and "A" in f: vals[groups[s]].append(f["A"]); sds[groups[s]].append(f["A_err"])
                add_contrast(f"{level} {fkey} A {pair}", vals, sds)
    json.dump(out, open(RES / "baselines_compare.json", "w"), indent=1, default=float)
    for key, e in out["group_contrasts"].items():
        print(f"{key:32s} " + "  ".join(f"{k} {fmt_c(c)}" for k, c in e["contrasts"].items()))
    # print the headline table
    for pair, per in out["ci_width"].items():
        for t in ("10", "20", "50"):
            rows = per.get(t, [])
            if not rows: continue
            mc = np.array([x["mol_ci"] for x in rows]); cc = np.array([x["cen_ci"] for x in rows])
            print(f"{pair:8s} r≈{t:>3s} µm ({len(rows)} sections): 95% CI width molecule {np.nanmedian(mc):.3f} vs centroid {np.nanmedian(cc):.3f} (ratio {np.nanmedian(cc/mc):.1f}); log g mol {np.nanmean([x['mol_log_g'] for x in rows]):+.2f} cen {np.nanmean([x['cen_log_g'] for x in rows]):+.2f}")
    for s, q in out["squidpy"].items():
        print(s, {k: round(v, 1) for k, v in q["nhood_z"].items()})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["cells", "squidpy", "centroids", "fit", "compare"])
    ap.add_argument("--section"); ap.add_argument("--threads", type=int, default=3, help="numba threads (0 = scote's default)")
    ap.add_argument("--cpu-mode", default="auto", help="CPU product strategy of the lattice levels")
    ap.add_argument("--n-levels", type=int, default=8); ap.add_argument("--co-occ-cells", type=int, default=50000)
    a = ap.parse_args()
    if a.threads and a.stage != "centroids":   # Squidpy's numba kernels; the scote run takes the count per call
        import numba; numba.set_num_threads(a.threads)
    dict(cells=stage_cells, squidpy=stage_squidpy, centroids=stage_centroids, fit=stage_fit, compare=stage_compare)[a.stage](a)


if __name__ == "__main__":
    main()
