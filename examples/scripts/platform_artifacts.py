#!/usr/bin/env python
"""Preprint Fig. 5a-d: platform artifacts as identifiable, scale-localized features.

Three stages on the Oliveira CRC P2 serial sections (Xenium molecules vs Visium HD 2 µm bins):

  acta2    : the background test for the Xenium ACTA2 case (inside the tumor region Xenium ACTA2 correlations are
             compressed 3–4x relative to HD). Reads the *raw* Xenium `transcripts.parquet` (all codeword
             classes, qv) and the HD 2 µm bin table, labels every molecule / bin with the region operator of
             scripts/crc_region_local.py (data/platforms/regions_{xenium,hd}.npz, 10 µm grid) and computes
               (i)   codeword-class inventory (genes, NegControlProbe, NegControlCodeword, UnassignedCodeword, Deprecated) and
                     their densities per region -> the platform's own background reference (counts / mm² / codeword);
               (ii)  qv distribution of ACTA2 vs the panel; fraction of molecules outside cells / in nuclei vs the panel;
               (iii) for all 408 common genes: density per region on both platforms and the fit x_R = s·h_R + β over the three
                     regions (x = Xenium density, h = HD density, β = uniform Xenium background) -> background fraction
                     b_R = β / x_R per region; ACTA2's rank among the 408 genes;
               (iv)  pixel-level intercept test on four domains (the tumor, normal, and stroma regions, and the whole section =
                     all tissue pixels regardless of region): occupied 10 µm pixels of the domain, mean ACTA2 count in the pixels
                     without smooth-muscle co-marker (TAGLN, DES) calls over the mean over all pixels = co-marker-independent
                     fraction b, on both platforms, SD from the closed-form tile bootstrap over 500 µm tiles; the same test for
                     TAGLN against DES as a control and the negative-control codeword rate per pixel of each domain;
               (v)   predicted compression of log g by a background making a fraction b of the gene's molecules —
                     Δlog g_aa ≈ log[(1−b)² + b(2−b)/g_true] (self) and Δlog g_ab ≈ log[(1−b) + b/g_true] (cross) — compared
                     with the observed Xenium−HD differences in the tumor-region runs (data/twopoint/regions/{xenium,hd}_tumor.npz)
                     at 6–60 µm for ACTA2–ACTA2, ACTA2–TAGLN, ACTA2–CEACAM5 (controls: TAGLN–TAGLN, TAGLN–CEACAM5).
  collapse : the sub-micrometer self-correlation collapse of the molecule-level run (data/twopoint/xenium_crc_p2_mol_common.npz):
             log g_aa(0–0.5 µm) − log g_aa(0.5–1 µm) for all 408 genes vs abundance, cross pairs as control, the missing-pair
             fraction 1 − g_aa(0)/g_aa(1), the steps between neighboring bins, and the deficit of bins 0 and 1 against a smooth
             extrapolation from 1–4 µm with the missing self-pair fraction below 1 µm (`extrapolation` block; threshold 1 µm).
  regions  : the ACTA2 background on the tumor, normal, and stroma regions and on the whole section: background density per
             domain from the pixel test (b · ACTA2 density), against a background of constant density per area and one
             proportional to the total transcript density (both anchored on the tumor pixel test); Xenium − HD differences of
             the ACTA2 pairs per domain against both models and against the domain's own pixel-test b; b fitted to the ACTA2 self
             pair; platform reference for the control pairs (stage_regions docstring). Reads acta2.json.
Outputs: results/artifacts/{acta2,collapse,acta2_regions}.json (small), figures/quicklook/artifacts_*.png; wall time / peak RSS via scripts/timed_run.py.
"""
from __future__ import annotations

import argparse, json, re, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "results/artifacts"; OUT.mkdir(exist_ok=True, parents=True)
QL = REPO / "figures/quicklook"
REGIONS = ["tumor", "normal", "stroma"]
PIXEL_DOMAINS = REGIONS + ["whole"]   # "whole": every tissue pixel (region label >= 0), regardless of region
CLASS_RE = [("NegControlProbe", re.compile(r"^NegControlProbe", re.I)), ("NegControlCodeword", re.compile(r"^NegControlCodeword", re.I)),
            ("UnassignedCodeword", re.compile(r"^Unassigned", re.I)), ("Deprecated", re.compile(r"^Deprecated", re.I)),
            ("BLANK", re.compile(r"^BLANK", re.I)), ("Intergenic", re.compile(r"^Intergenic", re.I))]
QV_EDGES = np.arange(0, 60.5, 0.5)
GRID_UM = 10.0
TILE_UM = 500.0


def classify(names):
    cls = np.array(["gene"] * len(names), dtype=object)
    for i, n in enumerate(names):
        for c, rx in CLASS_RE:
            if rx.match(n): cls[i] = c; break
    return cls


def label_grid(coords, reg):
    """Region label per point (−1 background tissue-free, −2 outside the grid) and the flat 10 µm pixel index."""
    ij = np.floor((coords - reg["origin"]) / reg["grid_um"]).astype(np.int64)
    sh = reg["shape"]
    inside = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < sh[0]) & (ij[:, 1] < sh[1])
    lab = np.full(len(coords), -2, dtype=np.int8)
    lab[inside] = reg["label"][ij[inside, 0], ij[inside, 1]]
    pix = np.where(inside, ij[:, 0] * sh[1] + ij[:, 1], -1)
    return lab, pix


def region_areas(reg):
    a = {r: float((reg["label"] == i).sum() * (reg["grid_um"] / 1000) ** 2) for i, r in enumerate(REGIONS)}
    a["tissue"] = float((reg["label"] >= 0).sum() * (reg["grid_um"] / 1000) ** 2)
    return a


# ------------------------------------------------------------------------------------------------------------------- acta2
def scan_xenium(a, reg, genes408):
    """Stream the raw transcripts.parquet once: per-feature class / qv / cell / nucleus / region counts + 10 µm pixel counts
    of the genes needed for the pixel-level test."""
    import pyarrow.parquet as pq, pyarrow.compute as pc
    pf = pq.ParquetFile(DATA / "raw/xenium_crc_p2/transcripts.parquet")
    feat_index = {}; names = []
    per_feat = dict(n=[], n_qv20=[], n_unassigned=[], n_nucleus=[], reg=[], qv_hist=[])
    pix_genes = a.pixel_genes.split(","); npix = int(np.prod(reg["shape"]))
    pix_counts = {g: np.zeros(npix, dtype=np.int32) for g in pix_genes}
    pix_ctrl = np.zeros(npix, dtype=np.int32)   # all NegControl* codewords per pixel
    pix_all = np.zeros(npix, dtype=np.int64)    # all gene molecules (qv >= 20) per pixel
    t0 = time.time(); n_rows = 0
    cols = ["feature_name", "x_location", "y_location", "qv", "cell_id", "overlaps_nucleus"]
    for b in pf.iter_batches(batch_size=4_000_000, columns=cols):
        n_rows += b.num_rows
        fn = pc.dictionary_encode(b.column("feature_name"))
        if hasattr(fn, "combine_chunks"): fn = fn.combine_chunks()
        dic = fn.dictionary.to_pylist(); idx = fn.indices.to_numpy(zero_copy_only=False).astype(np.int64)
        # map batch dictionary -> global feature index
        gmap = np.empty(len(dic), dtype=np.int64)
        for k, nm in enumerate(dic):
            if nm not in feat_index:
                feat_index[nm] = len(names); names.append(nm)
                for key in ("n", "n_qv20", "n_unassigned", "n_nucleus"): per_feat[key].append(0)
                per_feat["reg"].append(np.zeros(5, dtype=np.int64)); per_feat["qv_hist"].append(np.zeros(len(QV_EDGES) - 1, dtype=np.int64))
            gmap[k] = feat_index[nm]
        gi = gmap[idx]
        qv = b.column("qv").to_numpy(zero_copy_only=False).astype(np.float32)
        xy = np.stack([b.column("x_location").to_numpy(zero_copy_only=False), b.column("y_location").to_numpy(zero_copy_only=False)], 1).astype(np.float64)
        unassigned = pc.equal(b.column("cell_id"), "UNASSIGNED").to_numpy(zero_copy_only=False)
        nuc = b.column("overlaps_nucleus").to_numpy(zero_copy_only=False).astype(bool)
        lab, pix = label_grid(xy, reg)
        q20 = qv >= 20
        nf = len(names)
        cnt = np.bincount(gi, minlength=nf); c20 = np.bincount(gi[q20], minlength=nf)
        cun = np.bincount(gi[q20 & unassigned], minlength=nf); cnu = np.bincount(gi[q20 & nuc], minlength=nf)
        for f in range(nf):
            per_feat["n"][f] += int(cnt[f]); per_feat["n_qv20"][f] += int(c20[f]); per_feat["n_unassigned"][f] += int(cun[f]); per_feat["n_nucleus"][f] += int(cnu[f])
        # region counts (qv >= 20): labels -2..2 -> 0..4
        lr = (lab.astype(np.int64) + 2)
        rc = np.bincount(gi[q20] * 5 + lr[q20], minlength=nf * 5).reshape(nf, 5)
        for f in range(nf): per_feat["reg"][f] += rc[f]
        # qv histograms per feature (only for the features we report: all, cheap enough via 2-D bincount)
        qb = np.clip(np.searchsorted(QV_EDGES, qv, side="right") - 1, 0, len(QV_EDGES) - 2)
        qh = np.bincount(gi * (len(QV_EDGES) - 1) + qb, minlength=nf * (len(QV_EDGES) - 1)).reshape(nf, -1)
        for f in range(nf): per_feat["qv_hist"][f] += qh[f]
        # pixel counts
        ok = q20 & (pix >= 0)
        for g in pix_genes:
            if g in feat_index:
                m = ok & (gi == feat_index[g]); pix_counts[g] += np.bincount(pix[m], minlength=npix).astype(np.int32)
        cls_b = classify(dic); is_ctrl = np.isin(idx, np.where(np.char.startswith(cls_b.astype(str), "NegControl"))[0])
        pix_ctrl += np.bincount(pix[ok & is_ctrl], minlength=npix).astype(np.int32)
        is_gene = cls_b[idx] == "gene"
        pix_all += np.bincount(pix[ok & is_gene], minlength=npix)
        print(f"  {n_rows/1e6:.0f} M rows, {time.time()-t0:.0f} s", flush=True)
    cls = classify(names)
    return dict(names=names, cls=cls, n=np.array(per_feat["n"]), n_qv20=np.array(per_feat["n_qv20"]), n_unassigned=np.array(per_feat["n_unassigned"]),
                n_nucleus=np.array(per_feat["n_nucleus"]), reg=np.stack(per_feat["reg"]), qv_hist=np.stack(per_feat["qv_hist"]),
                pix_counts=pix_counts, pix_ctrl=pix_ctrl, pix_all=pix_all, n_rows=n_rows, seconds=time.time() - t0)


def scan_hd(a, reg, genes):
    """HD 2 µm bins -> per-gene region counts and 10 µm pixel counts of the pixel-test genes."""
    import scipy.sparse as sp
    z = np.load(DATA / "processed/hd_crc_p2/bins_square_002um.npz", allow_pickle=True)
    feats = list(map(str, z["features"])); X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    coords = z["coords_um"].astype(np.float64)
    lab, pix = label_grid(coords, reg)
    gi = [feats.index(g) for g in genes if g in feats]; gnames = [g for g in genes if g in feats]
    Xg = X[:, gi].tocsc()
    regc = np.zeros((len(gnames), 5), dtype=np.int64)
    lr = lab.astype(np.int64) + 2
    for k in range(5):
        rows = np.where(lr == k)[0]
        if len(rows): regc[:, k] = np.asarray(Xg[rows].sum(0)).ravel()
    npix = int(np.prod(reg["shape"])); ok = pix >= 0
    pix_counts = {}
    for g in a.pixel_genes.split(","):
        if g in gnames:
            col = np.asarray(Xg[:, gnames.index(g)].todense()).ravel()
            pix_counts[g] = np.bincount(pix[ok], weights=col[ok], minlength=npix)
    tot = np.asarray(X.sum(1)).ravel()
    pix_all = np.bincount(pix[ok], weights=tot[ok], minlength=npix)
    return dict(names=gnames, reg=regc, pix_counts=pix_counts, pix_all=pix_all)


def closed_form_tile_stat(stat, tinv, n_tiles, eps=1e-6):
    """Closed-form (B -> infinity) block bootstrap over tiles of a smooth statistic stat(w) of per-point weights w = W[tinv]
    (W = multinomial tile multiplicities with E W = 1). Seen as a function of the T tile
    weights, the statistic depends on tile-additive sums with one indicator per tile, so its closed-form variance is
    scote.uncertainty.tile_sum_var(I_T, grad) = sum_t (g_t - mean g)^2 with g_t = d stat / d W_t at W = 1 (central differences;
    exact up to O(eps^2) for the ratios of weighted sums used here): the infinitesimal jackknife over tiles.
    Returns (base, sd, lo, hi): base = stat(np.ones(len(tinv))) (the point estimate), lo / hi = base -+ 1.96 sd."""
    from scote import uncertainty as TV
    base = np.asarray(stat(np.ones(len(tinv))), float)
    W = np.ones(n_tiles); Gr = np.empty((base.size, n_tiles))
    for t in range(n_tiles):
        W[t] = 1.0 + eps; fp = np.asarray(stat(W[tinv]), float).ravel()
        W[t] = 1.0 - eps; fm = np.asarray(stat(W[tinv]), float).ravel()
        W[t] = 1.0
        Gr[:, t] = (fp - fm) / (2.0 * eps)
    sd = np.sqrt(np.clip(np.diag(np.atleast_2d(TV.tile_sum_var(np.eye(n_tiles), Gr))), 0, None)).reshape(base.shape)
    z = 1.959963984540054
    return base, sd, base - z * sd, base + z * sd


def domain_mask(reg, region):
    """Flat boolean mask of the 10 µm pixels of a domain: one region of REGIONS, or "whole" = all tissue pixels (label >= 0)."""
    lab = reg["label"].ravel()
    return lab >= 0 if region == "whole" else lab == REGIONS.index(region)


def pixel_intercept(reg, pix_target, pix_co, pix_all, region="tumor", n_boot=None, seed=None, n_bins=6):
    """Binned-mean regression of the target's pixel counts on the co-marker pixel counts inside `region` (a region of REGIONS or
    "whole"; occupied pixels only, i.e. at least one gene molecule); the intercept (mean target count in pixels with zero
    co-marker) relative to the mean over the domain's pixels is the background fraction estimate.
    Closed-form block bootstrap over 500 µm tiles of pixels: every statistic is a smooth function of seven tile-additive sums
    (pixels, target counts, zero-co-marker pixels and their target counts, co-marker counts, co-marker x target,
    co-marker^2), so its SD is scote.uncertainty.tile_sum_stat on the per-tile sums. n_boot, seed: unused."""
    from scote import uncertainty as TV
    sh = reg["shape"]
    sel = np.where(domain_mask(reg, region) & (pix_all > 0))[0]
    t = pix_target[sel].astype(float); c = pix_co[sel].astype(float)
    ii, jj = np.divmod(sel, sh[1]); tile = (ii * reg["grid_um"] // TILE_UM).astype(int) * 10000 + (jj * reg["grid_um"] // TILE_UM).astype(int)
    tiles, tinv = np.unique(tile, return_inverse=True)

    def stat(w):
        # w = weight per pixel (tile multiplicities)
        m_all = np.sum(w * t) / np.sum(w)
        zero = c == 0
        m0 = np.sum(w[zero] * t[zero]) / max(np.sum(w[zero]), 1)
        # linear fit t = α + β c (weighted) as the second intercept estimate
        W = np.sum(w); mc = np.sum(w * c) / W; mt = m_all
        beta = np.sum(w * (c - mc) * (t - mt)) / max(np.sum(w * (c - mc) ** 2), 1e-12); alpha = mt - beta * mc
        return np.array([m_all, m0, alpha, m0 / m_all, alpha / m_all])

    def f(S):
        # the same statistics from the tile-additive sums S = (n, sum t, n0, sum t [c = 0], sum c, sum c t, sum c^2)
        n, st, n0, st0, sc, sct, scc = S
        m_all = st / n; m0 = st0 / n0; mc = sc / n
        beta = (sct - n * mc * m_all) / (scc - n * mc * mc); alpha = m_all - beta * mc
        return np.array([m_all, m0, alpha, m0 / m_all, alpha / m_all])

    base = stat(np.ones(len(sel)))
    zero = (c == 0).astype(float)
    S_tiles = np.stack([np.bincount(tinv, weights=w, minlength=len(tiles)) for w in
                        (np.ones_like(t), t, zero, zero * t, c, c * t, c * c)], 1)
    val, sd = TV.tile_sum_stat(f, S_tiles)
    assert np.allclose(val, base, rtol=1e-9), (val, base)
    # binned means for the quick-look
    edges = np.unique(np.quantile(c[c > 0], np.linspace(0, 1, n_bins + 1))); edges = np.r_[0, edges]
    bins = np.clip(np.searchsorted(edges, c, side="right") - 1, 0, len(edges) - 2)
    bm = [(float(c[bins == k].mean()), float(t[bins == k].mean()), int((bins == k).sum())) for k in range(len(edges) - 1) if (bins == k).sum() > 0]
    return dict(region=region, n_pixels=int(len(sel)), n_tiles=int(len(tiles)), frac_pixels_zero_comarker=float(np.mean(c == 0)),
                mean_per_pixel=float(base[0]), mean_zero_comarker=float(base[1]), intercept_linear=float(base[2]),
                b_zero_comarker=float(base[3]), b_zero_comarker_sd=float(sd[3]), b_linear=float(base[4]), b_linear_sd=float(sd[4]),
                uncertainty="closed", binned_means=bm)


def predicted_compression(b, g_true, kind):
    """Observed g under a uniform background making a fraction b of the molecules of the gene: self g_obs = (1−b)² g + b(2−b);
    cross (background only on one side) g_obs = (1−b) g + b."""
    if kind == "self": return np.log((1 - b) ** 2 * g_true + b * (2 - b)) - np.log(g_true)
    return np.log((1 - b) * g_true + b) - np.log(g_true)


def pixel_prediction(bp, bsd, gh, kind):
    """Predicted Xenium − HD difference of log g for the pixel-test b = bp ± bsd and the HD log g values gh: (pred, [lo, hi] range)."""
    g = np.exp(np.asarray(gh, float))
    return (predicted_compression(bp, g, kind).tolist(),
            [predicted_compression(min(bp + bsd, 0.999), g, kind).tolist(), predicted_compression(max(bp - bsd, 0), g, kind).tolist()])


def pixel_density(bp, bsd, x):
    """Background density (calls per mm²) of a domain from its pixel-test b = bp ± bsd and its ACTA2 density x (calls per mm²)."""
    return dict(b_pixel=bp, b_pixel_sd=bsd, bkg_density_pixel=bp * x, bkg_density_pixel_sd=bsd * x)


def stage_acta2(a):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t_all = time.time()
    genes408 = [l.strip() for l in open(REPO / "results/platforms/common_genes_crc.txt") if l.strip() and not l.startswith("#")]
    regX = {k: np.load(DATA / "platforms/regions_xenium.npz", allow_pickle=True)[k] for k in ("origin", "shape", "grid_um", "label")}
    regH = {k: np.load(DATA / "platforms/regions_hd.npz", allow_pickle=True)[k] for k in ("origin", "shape", "grid_um", "label")}
    areaX, areaH = region_areas(regX), region_areas(regH)
    print("scanning Xenium transcripts.parquet ...", flush=True)
    X = scan_xenium(a, regX, genes408)
    print(f"Xenium scan: {X['n_rows']/1e6:.1f} M rows in {X['seconds']:.0f} s", flush=True)
    H = scan_hd(a, regH, genes408)
    names = X["names"]; cls = X["cls"]; nX = {n: i for i, n in enumerate(names)}
    # (i) codeword classes
    classes = {}
    for c in ["gene"] + [k for k, _ in CLASS_RE]:
        m = cls == c
        if m.sum() == 0: continue
        regc = X["reg"][m].sum(0)   # columns: outside grid, background(no tissue), tumor, normal, stroma
        classes[c] = dict(n_codewords=int(m.sum()), n_total=int(X["n"][m].sum()), n_qv20=int(X["n_qv20"][m].sum()),
                          per_region_qv20={r: int(regc[k + 2]) for k, r in enumerate(REGIONS)}, off_tissue_qv20=int(regc[0] + regc[1]),
                          density_per_mm2_per_codeword={r: float(regc[k + 2] / areaX[r] / m.sum()) for k, r in enumerate(REGIONS)},
                          frac_unassigned_cell=float(X["n_unassigned"][m].sum() / max(X["n_qv20"][m].sum(), 1)))
    gene_m = cls == "gene"
    # (ii) qv / cell assignment of ACTA2 vs the panel
    def qv_stats(h):
        cdf = np.cumsum(h) / max(h.sum(), 1); mid = 0.5 * (QV_EDGES[1:] + QV_EDGES[:-1])
        return dict(median=float(mid[np.searchsorted(cdf, 0.5)]), frac_below_20=float(h[QV_EDGES[:-1] < 20].sum() / max(h.sum(), 1)),
                    frac_below_30=float(h[QV_EDGES[:-1] < 30].sum() / max(h.sum(), 1)), mean=float((mid * h).sum() / max(h.sum(), 1)))
    per_gene = {}
    for g in names:
        if cls[nX[g]] != "gene": continue
        i = nX[g]
        per_gene[g] = dict(n=int(X["n"][i]), n_qv20=int(X["n_qv20"][i]), frac_unassigned=float(X["n_unassigned"][i] / max(X["n_qv20"][i], 1)),
                           frac_nucleus=float(X["n_nucleus"][i] / max(X["n_qv20"][i], 1)), **qv_stats(X["qv_hist"][i]))
    panel_med = {k: float(np.median([v[k] for v in per_gene.values()])) for k in ("frac_unassigned", "frac_nucleus", "median", "frac_below_20", "frac_below_30", "mean")}
    panel_q = {k: [float(q) for q in np.percentile([v[k] for v in per_gene.values()], [5, 25, 75, 95])] for k in panel_med}
    qv_all = X["qv_hist"][gene_m].sum(0)
    # (iii) per-region densities for the 408 genes on both platforms and the 3-region background fit
    common = [g for g in genes408 if g in nX and g in H["names"]]
    dX = np.array([[X["reg"][nX[g], k + 2] / areaX[r] for k, r in enumerate(REGIONS)] for g in common])   # counts / mm²
    dH = np.array([[H["reg"][H["names"].index(g), k + 2] / areaH[r] for k, r in enumerate(REGIONS)] for g in common])
    fits = []
    for k, g in enumerate(common):
        x, h = dX[k], dH[k]
        w = 1 / np.maximum(x, 1)          # Poisson-ish weights on the Xenium densities (per mm²; counts are large, so this is the shape only)
        A = np.stack([h, np.ones(3)], 1) * np.sqrt(w)[:, None]; y = x * np.sqrt(w)
        sol, *_ = np.linalg.lstsq(A, y, rcond=None); s, beta = sol
        resid = x - (s * h + beta)
        fits.append(dict(gene=g, s=float(s), beta=float(beta), b=[float(np.clip(beta / xx, -1, 1)) for xx in x], x=x.tolist(), h=h.tolist(),
                         rel_resid=float(np.sqrt(np.mean(resid ** 2)) / max(x.mean(), 1e-9)),
                         log_ratio_tumor_stroma=dict(xenium=float(np.log(x[0] / x[2])), hd=float(np.log(h[0] / h[2]))),
                         log_ratio_tumor_normal=dict(xenium=float(np.log(x[0] / x[1])), hd=float(np.log(h[0] / h[1])))))
    # the tumor/stroma log density ratio on Xenium vs HD over all genes: robust residual of ACTA2
    lrx = np.array([f["log_ratio_tumor_stroma"]["xenium"] for f in fits]); lrh = np.array([f["log_ratio_tumor_stroma"]["hd"] for f in fits])
    ok = np.isfinite(lrx) & np.isfinite(lrh)
    slope, icpt = np.polyfit(lrh[ok], lrx[ok], 1)
    res = lrx - (slope * lrh + icpt); mad = 1.4826 * np.nanmedian(np.abs(res[ok] - np.nanmedian(res[ok])))
    zres = (res - np.nanmedian(res[ok])) / mad
    order = np.argsort(-np.nan_to_num(zres, nan=-99))
    b_t = np.array([f["b"][0] for f in fits])
    rank_b = int(np.sum(b_t > b_t[common.index("ACTA2")]))
    ratio_test = dict(slope=float(slope), intercept=float(icpt), robust_sd=float(mad), n_genes=int(ok.sum()),
                      acta2=dict(z=float(zres[common.index("ACTA2")]), residual=float(res[common.index("ACTA2")]), xenium=float(lrx[common.index("ACTA2")]), hd=float(lrh[common.index("ACTA2")])),
                      top_positive_residuals=[dict(gene=common[i], z=float(zres[i]), residual=float(res[i]), n_xenium=int(X["n_qv20"][nX[common[i]]])) for i in order[:10]],
                      b_tumor_quantiles=[float(q) for q in np.nanpercentile(b_t, [5, 25, 50, 75, 95])], acta2_b_tumor_rank_from_top=rank_b)
    # (iv) pixel-level intercept, both platforms
    co = a.comarkers.split(",")
    pixX = dict(target=X["pix_counts"][a.target], co=sum(X["pix_counts"][g] for g in co), all=X["pix_all"], ctrl=X["pix_ctrl"])
    pixH = dict(target=H["pix_counts"][a.target], co=sum(H["pix_counts"][g] for g in co), all=H["pix_all"])
    pix_tests = {}
    for region in PIXEL_DOMAINS:
        pix_tests[region] = dict(xenium=pixel_intercept(regX, pixX["target"], pixX["co"], pixX["all"], region, a.n_boot),
                                 hd=pixel_intercept(regH, pixH["target"], pixH["co"], pixH["all"], region, a.n_boot))
        # the same intercept for a control gene (TAGLN vs DES) on Xenium — is the background ACTA2-specific?
        pix_tests[region]["xenium_control_TAGLN_vs_DES"] = pixel_intercept(regX, X["pix_counts"]["TAGLN"], X["pix_counts"]["DES"], pixX["all"], region, a.n_boot) if "DES" in X["pix_counts"] else None
        # negative-control codewords per pixel in the same region: background reference
        selp = domain_mask(regX, region) & (pixX["all"] > 0)
        n_ctrl_cw = classes.get("NegControlCodeword", {}).get("n_codewords", 0) + classes.get("NegControlProbe", {}).get("n_codewords", 0)
        pix_tests[region]["negcontrol_per_pixel_per_codeword"] = float(pixX["ctrl"][selp].mean() / max(n_ctrl_cw, 1))
        pix_tests[region]["acta2_per_pixel"] = float(pixX["target"][selp].mean())
    # (v) predicted vs observed compression in the tumor-region runs
    comp = {}
    zt = {p: np.load(DATA / f"twopoint/regions/{p}_tumor.npz", allow_pickle=True) for p in ("xenium", "hd")}
    fx = list(map(str, zt["xenium"]["features"])); fh = list(map(str, zt["hd"]["features"]))
    bT = pix_tests["tumor"]["xenium"]["b_zero_comarker"]; bT_sd = pix_tests["tumor"]["xenium"]["b_zero_comarker_sd"]
    b3 = fits[common.index("ACTA2")]["b"][0]
    for pair, kind in ((("ACTA2", "ACTA2"), "self"), (("ACTA2", "TAGLN"), "cross"), (("ACTA2", "CEACAM5"), "cross"), (("TAGLN", "TAGLN"), "self"), (("TAGLN", "CEACAM5"), "cross"), (("DES", "DES"), "self")):
        rows = []
        for k in range(len(zt["xenium"]["r_central"])):
            r = float(zt["xenium"]["r_central"][k])
            if r < 6 or r > 60: continue
            kh = int(np.argmin(np.abs(zt["hd"]["r_central"] - r)))
            gx = float(zt["xenium"]["log_g"][k, fx.index(pair[0]), fx.index(pair[1])]); gh = float(zt["hd"]["log_g"][kh, fh.index(pair[0]), fh.index(pair[1])])
            sx = float(zt["xenium"]["sd"][k, fx.index(pair[0]), fx.index(pair[1])]); sh_ = float(zt["hd"]["sd"][kh, fh.index(pair[0]), fh.index(pair[1])])
            gtrue = np.exp(gh)
            pred = float(predicted_compression(bT, gtrue, kind)) if "ACTA2" in pair else 0.0
            pred_lo = float(predicted_compression(min(bT + bT_sd, 0.999), gtrue, kind)) if "ACTA2" in pair else 0.0
            pred_hi = float(predicted_compression(max(bT - bT_sd, 0), gtrue, kind)) if "ACTA2" in pair else 0.0
            pred3 = float(predicted_compression(b3, gtrue, kind)) if "ACTA2" in pair else 0.0
            # b that would exactly explain the observed difference (assuming HD = truth)
            if "ACTA2" in pair:
                gobs = np.exp(gx)
                if kind == "self":
                    # solve (1-b)^2 g + b(2-b) = gobs -> (1-b)^2 (g - 1) = gobs - 1
                    q = (gobs - 1) / (gtrue - 1); b_exact = float(1 - np.sqrt(q)) if q > 0 else float("nan")
                else:
                    q = (gobs - 1) / (gtrue - 1); b_exact = float(1 - q) if np.isfinite(q) else float("nan")
            else: b_exact = None
            rows.append(dict(r_um=r, log_g_xenium=gx, sd_xenium=sx, log_g_hd=gh, sd_hd=sh_, observed_diff=gx - gh, diff_sd=float(np.hypot(sx, sh_)),
                             predicted_diff=pred, predicted_diff_range=[min(pred_lo, pred_hi), max(pred_lo, pred_hi)], predicted_diff_b3region=pred3, b_exact=b_exact))
        comp[f"{pair[0]}:{pair[1]}"] = dict(kind=kind, rows=rows)
    out = dict(seconds=time.time() - t_all, n_rows_raw=int(X["n_rows"]), areas_mm2=dict(xenium=areaX, hd=areaH), classes=classes,
               qv_panel=dict(all_genes=qv_stats(qv_all), acta2=per_gene["ACTA2"], panel_median=panel_med, panel_quantiles_5_25_75_95=panel_q,
                             qv_hist_edges=QV_EDGES.tolist(), qv_hist_acta2=X["qv_hist"][nX["ACTA2"]].tolist(), qv_hist_all_genes=qv_all.tolist(),
                             qv_hist_negcontrol=X["qv_hist"][np.char.startswith(cls.astype(str), "NegControl")].sum(0).tolist()),
               region_fit=dict(genes=common, acta2=fits[common.index("ACTA2")], tagln=fits[common.index("TAGLN")] if "TAGLN" in common else None,
                               des=fits[common.index("DES")] if "DES" in common else None, ratio_test=ratio_test,
                               all=[dict(gene=f["gene"], b_tumor=f["b"][0], b_stroma=f["b"][2], s=f["s"], lr_ts_x=f["log_ratio_tumor_stroma"]["xenium"], lr_ts_h=f["log_ratio_tumor_stroma"]["hd"]) for f in fits]),
               pixel_test=dict(target=a.target, comarkers=co, grid_um=GRID_UM, tile_um=TILE_UM, n_boot="closed", uncertainty="closed", **pix_tests),
               compression=comp)
    json.dump(out, open(OUT / "acta2.json", "w"), indent=1, default=float)
    # ---- quick-look
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    mid = 0.5 * (QV_EDGES[1:] + QV_EDGES[:-1])
    for h, lab_ in ((qv_all, "all genes"), (X["qv_hist"][nX["ACTA2"]], "ACTA2"), (out["qv_panel"]["qv_hist_negcontrol"], "NegControl*")):
        h = np.asarray(h, float); axs[0].step(mid, h / h.sum(), where="mid", label=lab_)
    axs[0].axvline(20, c="k", lw=0.5); axs[0].set_xlabel("qv"); axs[0].set_ylabel("fraction"); axs[0].legend(fontsize=8); axs[0].set_title("Xenium quality score")
    axs[1].scatter(lrh[ok], lrx[ok], s=6, alpha=0.5); i = common.index("ACTA2"); axs[1].scatter(lrh[i], lrx[i], s=40, c="C3", label="ACTA2")
    for g in ("TAGLN", "DES", "CEACAM5", "CD3E"):
        if g in common: j = common.index(g); axs[1].annotate(g, (lrh[j], lrx[j]), fontsize=7)
    xx = np.linspace(np.nanmin(lrh[ok]), np.nanmax(lrh[ok]), 2); axs[1].plot(xx, slope * xx + icpt, "k-", lw=0.8)
    axs[1].set_xlabel("HD: log(density tumor / stroma)"); axs[1].set_ylabel("Xenium: log(density tumor / stroma)"); axs[1].legend(fontsize=8); axs[1].set_title("408 genes, region densities")
    for name, pt, col in (("xenium", pix_tests["tumor"]["xenium"], "C1"), ("hd", pix_tests["tumor"]["hd"], "C2")):
        bm = np.array(pt["binned_means"]); axs[2].plot(bm[:, 0], bm[:, 1] / pt["mean_per_pixel"], "o-", c=col, label=f"{name}: b = {pt['b_zero_comarker']:.2f} ± {pt['b_zero_comarker_sd']:.2f}")
    axs[2].axhline(1, c="k", lw=0.5); axs[2].set_xscale("symlog"); axs[2].set_xlabel(f"{'+'.join(co)} counts per 10 µm pixel"); axs[2].set_ylabel(f"{a.target} counts per pixel / regional mean"); axs[2].legend(fontsize=8); axs[2].set_title("tumor region: intercept = background")
    for key, col in (("ACTA2:ACTA2", "C3"), ("ACTA2:TAGLN", "C1"), ("ACTA2:CEACAM5", "C0"), ("TAGLN:TAGLN", "gray"), ("TAGLN:CEACAM5", "silver")):
        rows = comp[key]["rows"]; r = [d["r_um"] for d in rows]
        axs[3].errorbar(r, [d["observed_diff"] for d in rows], yerr=[d["diff_sd"] for d in rows], fmt="o-", c=col, ms=3, label=f"{key} observed")
        if "ACTA2" in key: axs[3].plot(r, [d["predicted_diff"] for d in rows], "--", c=col, label=f"{key} predicted (b = {bT:.2f})")
    axs[3].axhline(0, c="k", lw=0.5); axs[3].set_xscale("log"); axs[3].set_xlabel("r [µm]"); axs[3].set_ylabel("log g Xenium − log g HD (tumor region)"); axs[3].legend(fontsize=6); axs[3].set_title("uniform-background prediction")
    fig.tight_layout(); fig.savefig(QL / "artifacts_acta2_background.png", dpi=130)
    print(json.dumps({k: out[k] for k in ("classes",)}, indent=1, default=float))
    print("qv:", json.dumps(dict(acta2=out["qv_panel"]["acta2"], all=out["qv_panel"]["all_genes"], panel_median=panel_med), indent=1))
    print("region fit ACTA2:", json.dumps(out["region_fit"]["acta2"], indent=1)); print("ratio test:", json.dumps(ratio_test, indent=1))
    print("pixel tests:", json.dumps({r: {k: ({kk: vv for kk, vv in v.items() if kk != "binned_means"} if isinstance(v, dict) else v) for k, v in pt.items()} for r, pt in pix_tests.items()}, indent=1))
    for k, v in comp.items():
        print(k, [(round(d["r_um"], 1), round(d["observed_diff"], 2), round(d["diff_sd"], 2), round(d["predicted_diff"], 2), None if d["b_exact"] is None else round(d["b_exact"], 2)) for d in v["rows"][:4]])
    print(f"done in {time.time()-t_all:.0f} s")


# ---------------------------------------------------------------------------------------------------------------- collapse
EXTRAP_FIT = (2, 8)        # bins 2..7 = 1.0–4.0 µm: the range the sub-micrometer bins are extrapolated from
EXTRAP_SHOW = 10           # residuals reported for bins 0..9 (bins 8, 9 = 4–6 µm lie outside the fit: out-of-sample check)
EXTRAP_FORMS = {"linear_r": (lambda r: r, 1), "quadratic_r": (lambda r: r, 2), "linear_logr": (np.log, 1), "quadratic_logr": (np.log, 2)}
EXTRAP_PRIMARY = "quadratic_r"   # predicts the held-out 4–6 µm bins within 0.005 (linear in r: +0.02/+0.07; linear in log r: −0.01/−0.03)


def collapse_extrapolation(lg, P, rc, r_from, r_to, feats, tot, min_pairs):
    """Deficit of the sub-micrometer bins relative to a smooth extrapolation from 1–4 µm (the 0.5–1 µm bin is depleted too).
    Per same-gene pair (all bins 0..7 with >= min_pairs pairs) and per different-gene pair, log g(r)
    over bins 2..7 is fitted by a polynomial in r or log r (four forms; primary: quadratic in r, the form that best predicts the held-out
    bins 8, 9 = 4–6 µm) and extrapolated
    to bins 0 and 1; the residual log g − extrapolation is the deficit. The missing self-pair fraction below 1 µm (the 1 µm
    threshold) is 1 − (P0 + P1) / (P0·e^{−d0} + P1·e^{−d1}), i.e. observed pairs over the pairs the extrapolation expects."""
    G = lg.shape[1]; di = np.arange(G); nshow = EXTRAP_SHOW; f0, f1 = EXTRAP_FIT
    S = lg[:nshow, di, di]; PS = P[:nshow, di, di]
    oks = np.all(PS[:f1] >= min_pairs, 0) & np.all(np.isfinite(S[:f1]), 0)
    iu = np.triu_indices(G, 1)
    C = lg[:nshow, iu[0], iu[1]]; PC = P[:nshow, iu[0], iu[1]]
    okc = np.all(PC[:f1] >= min_pairs, 0) & np.all(np.isfinite(C[:f1]), 0)
    rc = np.asarray(rc[:nshow], float)

    def resid(Y, xf, deg):
        X = xf(rc); A = np.vander(X[f0:f1], deg + 1); coef, *_ = np.linalg.lstsq(A, Y[f0:f1], rcond=None)
        return Y - np.vander(X, deg + 1) @ coef

    def q(v):
        v = v[np.isfinite(v)]; return dict(median=float(np.median(v)), q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)), frac_negative=float(np.mean(v < 0)), n=int(v.size))

    out = dict(fit_bins=[f0, f1 - 1], fit_r_um=[float(r_from[f0]), float(r_to[f1 - 1])], r_central=rc.tolist(), r_from=[float(x) for x in r_from[:nshow]], r_to=[float(x) for x in r_to[:nshow]],
               primary=EXTRAP_PRIMARY, n_self=int(oks.sum()), n_cross=int(okc.sum()), forms={})
    for name, (xf, deg) in EXTRAP_FORMS.items():
        Ds = resid(S[:, oks], xf, deg); Dc = resid(C[:, okc], xf, deg)
        P0, P1 = PS[0, oks], PS[1, oks]
        miss1 = 1 - (P0 + P1) / (P0 * np.exp(-Ds[0]) + P1 * np.exp(-Ds[1]))      # below 1 µm (bins 0 + 1)
        miss_b0 = 1 - np.exp(Ds[0]); miss_b1 = 1 - np.exp(Ds[1])
        out["forms"][name] = dict(self_bins=[q(Ds[k]) for k in range(nshow)], cross_bins=[q(Dc[k]) for k in range(nshow)],
                                  self_missing_below_1um=q(miss1), self_missing_bin0=q(miss_b0), self_missing_bin1=q(miss_b1))
        if name == EXTRAP_PRIMARY:
            gi = np.where(oks)[0]
            out["per_gene"] = dict(gene=[feats[i] for i in gi], n=[int(tot[i]) for i in gi], deficit_bin0=Ds[0].tolist(), deficit_bin1=Ds[1].tolist(),
                                   missing_below_1um=miss1.tolist())
    print("extrapolation from 1–4 µm (self: median deficit bin0 / bin1; missing below 1 µm):")
    for name, f in out["forms"].items():
        print(f"  {name:15s} {f['self_bins'][0]['median']:+.2f} / {f['self_bins'][1]['median']:+.2f} (bin1 IQR {f['self_bins'][1]['q25']:+.2f}..{f['self_bins'][1]['q75']:+.2f}, "
              f"{f['self_bins'][1]['frac_negative']*100:.0f} % < 0); missing <1 µm {f['self_missing_below_1um']['median']*100:.0f} %; cross bin0 / bin1 "
              f"{f['cross_bins'][0]['median']:+.2f} / {f['cross_bins'][1]['median']:+.2f}")
    return out

def pair_shell_influence(z, pair, ks):
    """Per-tile influence (T, len(ks)) of log g of one gene pair in the shells ks of a stored run, from the tensor collapsed
    to {a, b, rest} (scote.uncertainty.collapse_tensor: the rest group keeps every tile's full P_tot and n, so the influence of
    log g_ab is that of the full tensor; scote.uncertainty.influence_log_g). Its outer-product sum is the closed-form
    cross-shell covariance (scote.uncertainty.cross_shell_cov). Loads counts_tiles."""
    from scote import uncertainty as TV
    feats = list(map(str, z["features"])); i, j = feats.index(pair[0]), feats.index(pair[1])
    ct = z["counts_tiles"]
    labels = np.full(len(feats), -1, dtype=np.int64); labels[i] = 0; labels[j] = 0 if i == j else 1
    C, N = TV.collapse_tensor(ct[:, list(ks)], z["totals_tiles"], labels)
    del ct
    return np.ascontiguousarray(TV.influence_log_g(C, N)[:, :, 0, 0 if i == j else 1])


def gls_b_fit(obs, gh, kind, cov, n_grid=991):
    """b fitted to the observed Xenium - HD difference with chi2(b) = r^T Cov^-1 r (r = obs - predicted_compression(b));
    interval from Delta chi2 = max(1, chi2_red). cov a vector: diagonal weights 1/SD^2 (the SDs)."""
    from scipy.optimize import minimize_scalar
    if np.ndim(cov) == 1:
        f = lambda b: float(np.sum(((obs - predicted_compression(b, np.exp(gh), kind)) / cov) ** 2))
    else:
        Ci = np.linalg.inv(cov)
        def f(b):
            r = obs - predicted_compression(b, np.exp(gh), kind); return float(r @ Ci @ r)
    res = minimize_scalar(f, bounds=(0, 0.99), method="bounded"); bfit = float(res.x)
    chi2 = f(bfit); red = max(chi2 / max(len(obs) - 1, 1), 1.0); grid = np.linspace(0, 0.99, n_grid)
    inside = grid[np.array([f(b) for b in grid]) <= chi2 + red]
    return bfit, [float(inside.min()), float(inside.max())], chi2 / max(len(obs) - 1, 1)


def stage_regions(a):
    """Is the ACTA2 background confined to the tumor region, and is it uniform per area? Reads acta2.json (pixel tests, region densities) and only the log_g / sd arrays of the region runs
    (data/twopoint/regions/{xenium,hd}_{tumor,normal,stroma}.npz) and of the whole-section runs
    (data/twopoint/{xenium,hd}_crc_p2_bin2um_common.npz; same 2 µm lattice, same 408 genes); counts_tiles only for the cross-shell
    covariance of the ACTA2 self pair. Two background models, both anchored on
    the tumor pixel test (density β_T = b_T · x_T of co-marker-independent ACTA2 calls):
      constant    : background calls have the same density per area in every region, b_R = β_T / x_R;
      prop_total  : background density follows the total Xenium transcript density T_R (per mm² per codeword), b_R = β_T T_R / (T_T x_R)
                    — i.e. background calls are distributed like the random-label reference, for which the transformation of g
                    used here is exact on any domain;
    whole section: b = Σ background / Σ ACTA2 calls over the three regions. The measured background density of every domain
    (tumor, normal, stroma, whole) is the pixel test of that domain, b_pixel · x_domain (x = ACTA2 calls per mm², whole: all calls of
    the three regions over their area), SD = SD(b_pixel) · x_domain. For every domain the observed Xenium − HD difference of
    the ACTA2 pairs (6–60 µm) is compared with both predictions and with the prediction from the domain's pixel-test b, and b is also fitted to the ACTA2 self pair alone (generalized least
    squares with the closed-form cross-shell covariance of the difference, Cov_Xenium + Cov_HD, correlation shrunk by
    Ledoit-Wolf as in scripts/gls_tiles.py; the unregularized GLS fit is kept as b_fit_gls_raw, the 1/SD² fit as b_fit_diag). Pairs with reliability flag 2 (T_eff < 5) are left out of the reference sets. Control: in the tumor region, every pair's Xenium − HD difference is compared with the reference of all pairs
    with the same HD log g (self pairs within ±0.25, cross pairs within ±0.15): the platform-wide difference at that correlation level.
    Writes results/artifacts/acta2_regions.json."""
    from scipy.optimize import minimize_scalar
    t0 = time.time()
    ac = json.load(open(OUT / "acta2.json"))
    A = ac["areas_mm2"]["xenium"]; T = ac["classes"]["gene"]["density_per_mm2_per_codeword"]
    x = dict(zip(REGIONS, ac["region_fit"]["acta2"]["x"]))
    pt = ac["pixel_test"]; bT = pt["tumor"]["xenium"]["b_zero_comarker"]; beta = bT * x["tumor"]
    N = {R: x[R] * A[R] for R in REGIONS}
    bk = dict(constant={R: beta * A[R] for R in REGIONS}, prop_total={R: beta * T[R] / T["tumor"] * A[R] for R in REGIONS})
    models = {m: {**{R: v[R] / N[R] for R in REGIONS}, "whole": sum(v.values()) / sum(N.values())} for m, v in bk.items()}
    files = {**{R: (DATA / f"twopoint/regions/xenium_{R}.npz", DATA / f"twopoint/regions/hd_{R}.npz") for R in REGIONS},
             "whole": (DATA / "twopoint/xenium_crc_p2_bin2um_common.npz", DATA / "twopoint/hd_crc_p2_bin2um_common.npz")}
    pairs = ((("ACTA2", "ACTA2"), "self"), (("ACTA2", "CEACAM5"), "cross"), (("ACTA2", "TAGLN"), "cross"), (("TAGLN", "TAGLN"), "self"), (("TAGLN", "CEACAM5"), "cross"))
    bpix = {dom: (pt[dom]["xenium"]["b_zero_comarker"], pt[dom]["xenium"]["b_zero_comarker_sd"]) for dom in PIXEL_DOMAINS}
    out = dict(b_tumor_pixel=bT, b_tumor_pixel_sd=bpix["tumor"][1], b_pixel={dom: dict(b=b, sd=sd) for dom, (b, sd) in bpix.items()},
               total_density_per_mm2_per_codeword={R: T[R] for R in REGIONS}, acta2_calls=N, areas_mm2={R: A[R] for R in REGIONS},
               models=models, domains={}, reference={})
    for dom, (fx, fh) in files.items():
        zx, zh = np.load(fx, allow_pickle=True), np.load(fh, allow_pickle=True)
        feats = list(map(str, zx["features"])); assert feats == list(map(str, zh["features"]))
        r = np.asarray(zx["r_central"]); ks = [k for k in range(len(r)) if 6 <= r[k] <= 60]
        LX, LH, SX, SH = zx["log_g"], zh["log_g"], zx["sd"], zh["sd"]
        if "flag" not in zx.files or "flag" not in zh.files:
            raise SystemExit(f"{fx.name} / {fh.name}: no 'flag' key; re-run them with python -m scote twopoint")
        FX, FH = zx["flag"], zh["flag"]          # reliability flag (2: T_eff < 5, no interval): out of the reference sets
        dd = dict(r_um=[float(r[k]) for k in ks], pairs={})
        for (g1, g2), kind in pairs:
            i, j = feats.index(g1), feats.index(g2)
            gx = np.array([LX[k, i, j] for k in ks], float); gh = np.array([LH[k, i, j] for k in ks], float)
            sd = np.hypot([SX[k, i, j] for k in ks], [SH[k, i, j] for k in ks]); obs = gx - gh
            e = dict(kind=kind, log_g_xenium=gx.tolist(), log_g_hd=gh.tolist(), observed_diff=obs.tolist(), diff_sd=sd.tolist())
            if g1 == "ACTA2":
                for m in models:
                    p = predicted_compression(min(models[m][dom], 0.999), np.exp(gh), kind)
                    e[f"pred_{m}"] = p.tolist(); e[f"rms_{m}"] = float(np.sqrt(np.mean((obs - p) ** 2)))
                e["pred_pixel"], e["pred_pixel_range"] = pixel_prediction(*bpix[dom], gh, kind)
                if kind == "self":
                    # generalized least squares with the joint closed-form tile-bootstrap covariance of the difference over the
                    # fitted bins (independent sections: Cov = Cov_Xenium + Cov_HD); interval from Δχ² = max(1, χ²_red).
                    # The diagonal (1/SD² weights) fit is kept as b_fit_diag for comparison.
                    # tiles of both sections as rows (independent sections: Cov = Cov_Xenium + Cov_HD); correlation shrunk toward
                    # the identity with the Ledoit-Wolf intensity (scripts/gls_tiles.py), unregularized GLS and 1/SD² kept
                    from gls_tiles import shrunk_cov
                    Lxh = np.vstack([pair_shell_influence(zx, (g1, g2), ks), pair_shell_influence(zh, (g1, g2), ks)])
                    assert np.allclose(np.sqrt((Lxh ** 2).sum(0)), sd, rtol=1e-6), "cross-shell covariance does not match the stored SDs"
                    cov, s_lw, cond = shrunk_cov(Lxh); cov0, _, cond0 = shrunk_cov(Lxh, 0.0)
                    e["b_fit"], e["b_fit_interval"], e["b_fit_chi2_red"] = gls_b_fit(obs, gh, kind, cov)
                    e["b_fit_method"] = "gls_ledoit_wolf"; e["b_fit_shrinkage"] = s_lw; e["b_fit_cond_corr"] = cond; e["b_fit_cond_corr_raw"] = cond0
                    e["b_fit_gls_raw"], e["b_fit_gls_raw_interval"], e["b_fit_gls_raw_chi2_red"] = gls_b_fit(obs, gh, kind, cov0)
                    e["b_fit_diag"], e["b_fit_diag_interval"], e["b_fit_diag_chi2_red"] = gls_b_fit(obs, gh, kind, sd)
            dd["pairs"][f"{g1}:{g2}"] = e
        out["domains"][dom] = dd
        if dom == "tumor":       # platform reference at the same HD correlation level
            G = len(feats); di = np.arange(G); iu = np.triu_indices(G, 1); ref = dict(r_um=dd["r_um"], pairs={})
            for (g1, g2), kind in pairs:
                i, j = feats.index(g1), feats.index(g2); rows = []
                for k in ks:
                    X, H, sx, sh = LX[k], LH[k], SX[k], SH[k]
                    if kind == "self": D, Hh, ok, tol = (X - H)[di, di], H[di, di], (sx[di, di] < 0.3) & (sh[di, di] < 0.3) & (FX[k][di, di] < 2) & (FH[k][di, di] < 2), 0.25
                    else: D, Hh, ok, tol = (X - H)[iu], H[iu], (sx[iu] < 0.3) & (sh[iu] < 0.3) & (FX[k][iu] < 2) & (FH[k][iu] < 2), 0.15
                    m = ok & np.isfinite(D) & (np.abs(Hh - H[i, j]) < tol); v = D[m]; d0 = float(X[i, j] - H[i, j])
                    rows.append(dict(n=int(m.sum()), median=float(np.median(v)), q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)),
                                     q05=float(np.percentile(v, 5)), q95=float(np.percentile(v, 95)), percentile_of_pair=float(np.mean(v < d0) * 100)))
                ref["pairs"][f"{g1}:{g2}"] = rows
            out["reference"] = ref
        del zx, zh, LX, LH, SX, SH, FX, FH
    # background density (calls per mm²) per domain: pixel test of the domain; the b fitted to the self pair kept for reference
    xd = {**x, "whole": sum(N.values()) / sum(A[R] for R in REGIONS)}
    dens = {}
    for dom in ("tumor", "normal", "stroma", "whole"):
        e = out["domains"][dom]["pairs"]["ACTA2:ACTA2"]
        dens[dom] = dict(acta2_density=xd[dom], b_fit=e["b_fit"], b_fit_interval=e["b_fit_interval"],
                         bkg_density_fit=e["b_fit"] * xd[dom], bkg_density_fit_interval=[v * xd[dom] for v in e["b_fit_interval"]],
                         b_fit_diag=e["b_fit_diag"], b_fit_diag_interval=e["b_fit_diag_interval"],
                         b_fit_gls_raw=e["b_fit_gls_raw"], b_fit_gls_raw_interval=e["b_fit_gls_raw_interval"],
                         bkg_density_fit_diag_interval=[v * xd[dom] for v in e["b_fit_diag_interval"]],
                         pred_constant=models["constant"][dom] * xd[dom], pred_prop_total=models["prop_total"][dom] * xd[dom])
        dens[dom].update(pixel_density(*bpix[dom], xd[dom]))
    out["background_density"] = dens
    out["seconds"] = time.time() - t0
    json.dump(out, open(OUT / "acta2_regions.json", "w"), indent=1, default=float)
    for dom, v in dens.items(): print(dom, {k: (round(w) if isinstance(w, float) and w > 10 else w) for k, w in v.items()})
    print("models:", json.dumps(models, indent=None))
    for dom, dd in out["domains"].items():
        for pk, e in dd["pairs"].items():
            if "rms_constant" in e:
                print(f"{dom:7s} {pk:14s} obs(6.6) {e['observed_diff'][0]:+.2f}  rms constant {e['rms_constant']:.2f}  prop_total {e['rms_prop_total']:.2f}" + (f"  b_fit {e['b_fit']:.2f} {e['b_fit_interval']}" if "b_fit" in e else ""))
    for pk, rows in out["reference"]["pairs"].items():
        print(f"reference {pk:14s} percentiles", [round(x["percentile_of_pair"]) for x in rows])
    print(f"done in {out['seconds']:.0f} s")


def stage_collapse(a):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t0 = time.time()
    z = np.load(DATA / "twopoint/xenium_crc_p2_mol_common.npz", allow_pickle=True)
    feats = list(map(str, z["features"])); lg = z["log_g"]; sd = z["sd"]; P = z["P"]; tot = z["totals"]; r_from, r_to = z["r_from"], z["r_to"]
    G = len(feats); di = np.arange(G)
    nb = a.n_ref_bins
    self0 = lg[0, di, di]; self1 = lg[1, di, di]; self2 = lg[2, di, di]
    sd0 = sd[0, di, di]; sd1 = sd[1, di, di]
    delta = self0 - self1
    okd = np.isfinite(delta) & (P[0, di, di] >= a.min_pairs) & (P[1, di, di] >= a.min_pairs)
    missing = 1 - np.exp(self0 - self1)          # fraction of the self-pairs expected from the 0.5–1 µm bin that are missing below 0.5 µm
    # cross pairs as control
    iu = np.triu_indices(G, 1)
    cross0 = lg[0][iu]; cross1 = lg[1][iu]; dcross = cross0 - cross1
    okc = np.isfinite(dcross) & (P[0][iu] >= a.min_pairs) & (P[1][iu] >= a.min_pairs)
    # how far does the deficit extend? compare bin k to bin k+1 for the diagonal, k = 0..5
    steps = []
    for k in range(min(8, lg.shape[0] - 1)):
        d = lg[k, di, di] - lg[k + 1, di, di]; m = np.isfinite(d) & (P[k, di, di] >= a.min_pairs) & (P[k + 1, di, di] >= a.min_pairs)
        dc = lg[k][iu] - lg[k + 1][iu]; mc = np.isfinite(dc) & (P[k][iu] >= a.min_pairs) & (P[k + 1][iu] >= a.min_pairs)
        steps.append(dict(r_from=float(r_from[k]), r_to=float(r_to[k]), next_r_to=float(r_to[k + 1]), n_self=int(m.sum()), median_step_self=float(np.median(d[m])),
                          q25_self=float(np.percentile(d[m], 25)), q75_self=float(np.percentile(d[m], 75)), n_cross=int(mc.sum()), median_step_cross=float(np.median(dc[mc])),
                          q25_cross=float(np.percentile(dc[mc], 25)), q75_cross=float(np.percentile(dc[mc], 75))))
    # abundance dependence: Spearman of delta vs log totals; binned medians
    from scipy.stats import spearmanr
    rho, pval = spearmanr(np.log(tot[okd]), delta[okd])
    qe = np.quantile(np.log10(tot[okd]), np.linspace(0, 1, 6)); ab = []
    for i in range(5):
        m = okd & (np.log10(tot) >= qe[i]) & (np.log10(tot) <= qe[i + 1])
        ab.append(dict(log10_n_from=float(qe[i]), log10_n_to=float(qe[i + 1]), n_genes=int(m.sum()), median_delta=float(np.median(delta[m])), q25=float(np.percentile(delta[m], 25)), q75=float(np.percentile(delta[m], 75)),
                       median_missing_frac=float(np.median(missing[m]))))
    # genes with the strongest / weakest collapse
    order = np.argsort(delta[okd]); gsel = np.where(okd)[0]
    examples = {g: dict(n=int(tot[feats.index(g)]), log_g_0=float(self0[feats.index(g)]), sd_0=float(sd0[feats.index(g)]), log_g_1=float(self1[feats.index(g)]), sd_1=float(sd1[feats.index(g)]),
                        log_g_2=float(self2[feats.index(g)]), delta=float(delta[feats.index(g)]), missing_frac=float(missing[feats.index(g)])) for g in a.examples.split(",") if g in feats}
    # significance: |delta| / sqrt(sd0² + sd1²)
    zsc = delta / np.hypot(sd0, sd1)
    out = dict(seconds=time.time() - t0, n_genes=G, bins=dict(first=[float(r_from[0]), float(r_to[0])], second=[float(r_from[1]), float(r_to[1])]), min_pairs=a.min_pairs,
               self=dict(n=int(okd.sum()), median_delta=float(np.median(delta[okd])), q10=float(np.percentile(delta[okd], 10)), q25=float(np.percentile(delta[okd], 25)), q75=float(np.percentile(delta[okd], 75)), q90=float(np.percentile(delta[okd], 90)),
                         frac_negative=float(np.mean(delta[okd] < 0)), frac_z_below_minus3=float(np.mean(zsc[okd] < -3)), median_missing_frac=float(np.median(missing[okd])), q25_missing=float(np.percentile(missing[okd], 25)), q75_missing=float(np.percentile(missing[okd], 75)),
                         median_log_g_first=float(np.median(self0[okd])), median_log_g_second=float(np.median(self1[okd])), median_log_g_third=float(np.median(self2[okd]))),
               cross=dict(n=int(okc.sum()), median_delta=float(np.median(dcross[okc])), q10=float(np.percentile(dcross[okc], 10)), q25=float(np.percentile(dcross[okc], 25)), q75=float(np.percentile(dcross[okc], 75)), q90=float(np.percentile(dcross[okc], 90)),
                          frac_negative=float(np.mean(dcross[okc] < 0))),
               steps=steps, abundance=dict(spearman=float(rho), p=float(pval), bins=ab), examples=examples,
               strongest=[dict(gene=feats[i], delta=float(delta[i]), n=int(tot[i])) for i in gsel[order[:8]]], weakest=[dict(gene=feats[i], delta=float(delta[i]), n=int(tot[i])) for i in gsel[order[-8:]]],
               per_gene=dict(gene=[feats[i] for i in gsel], n=[int(tot[i]) for i in gsel], delta=[float(delta[i]) for i in gsel], missing_frac=[float(missing[i]) for i in gsel], z=[float(zsc[i]) for i in gsel]))
    out["extrapolation"] = collapse_extrapolation(lg, P, z["r_central"], r_from, r_to, feats, tot, a.min_pairs)
    json.dump(out, open(OUT / "collapse.json", "w"), indent=1, default=float)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    axs[0].scatter(tot[okd], delta[okd], s=8, alpha=0.6, label="self pairs a–a"); axs[0].set_xscale("log"); axs[0].axhline(0, c="k", lw=0.5)
    axs[0].set_xlabel("molecules of gene a"); axs[0].set_ylabel("log g_aa(0–0.5 µm) − log g_aa(0.5–1 µm)"); axs[0].set_title(f"self-correlation collapse below 0.5 µm ({okd.sum()} genes)")
    axs[1].hist(dcross[okc], bins=60, range=(-3, 3), alpha=0.6, density=True, label=f"cross pairs (n = {okc.sum()})"); axs[1].hist(delta[okd], bins=60, range=(-3, 3), alpha=0.6, density=True, label=f"self pairs (n = {okd.sum()})")
    axs[1].set_xlabel("log g(0–0.5 µm) − log g(0.5–1 µm)"); axs[1].legend(fontsize=8); axs[1].set_title("distribution")
    rr = [0.5 * (s["r_from"] + s["r_to"]) for s in steps]
    axs[2].errorbar(rr, [s["median_step_self"] for s in steps], yerr=[[s["median_step_self"] - s["q25_self"] for s in steps], [s["q75_self"] - s["median_step_self"] for s in steps]], fmt="o-", label="self: log g(k) − log g(k+1)")
    axs[2].errorbar(rr, [s["median_step_cross"] for s in steps], yerr=[[s["median_step_cross"] - s["q25_cross"] for s in steps], [s["q75_cross"] - s["median_step_cross"] for s in steps]], fmt="s-", label="cross")
    axs[2].axhline(0, c="k", lw=0.5); axs[2].set_xlabel("r of bin k [µm]"); axs[2].set_ylabel("median step to the next bin"); axs[2].legend(fontsize=8); axs[2].set_title("the deficit is confined to the first bin")
    fig.tight_layout(); fig.savefig(QL / "artifacts_collapse.png", dpi=130)
    print(json.dumps({k: out[k] for k in ("self", "cross", "abundance", "examples", "strongest", "weakest")}, indent=1, default=float))
    print("steps:", [(round(s["r_from"], 2), round(s["median_step_self"], 2), round(s["median_step_cross"], 2)) for s in steps])
    print(f"done in {time.time()-t0:.0f} s")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["acta2", "collapse", "regions"])
    ap.add_argument("--target", default="ACTA2"); ap.add_argument("--comarkers", default="TAGLN,DES")
    ap.add_argument("--pixel-genes", default="ACTA2,TAGLN,DES,MYH11,CEACAM5"); ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--min-pairs", type=float, default=20); ap.add_argument("--n-ref-bins", type=int, default=2)
    ap.add_argument("--examples", default="CD3E,CEACAM5,ACTA2,TAGLN,PECAM1,VWF,MS4A1,EPCAM,KRT8,IGKC")
    a = ap.parse_args()
    {"acta2": stage_acta2, "collapse": stage_collapse, "regions": stage_regions}[a.stage](a)


if __name__ == "__main__":
    main()
