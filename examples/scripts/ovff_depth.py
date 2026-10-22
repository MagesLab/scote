#!/usr/bin/env python
"""Preprint Supplementary Fig. S3: sequencing-depth ablation on the Visium HD probe library of the FF ovarian section.
The deep and the min-depth runs are the same library (5.2x reads, 3.0x UMIs); in addition the deep 2 µm matrix is
binomially thinned to the min-depth UMI total (p = 0.335) and to 1/2, 1/4, 1/8 so the real depth ablation can be compared
with the molecule subsampling of preprint Fig. 4c. All runs: scripts/ovff_crossplatform.py
twopoint --geneset 5k (top-1000 of 5K ∩ HD probe), registered frame, 500 µm tiles, 10 levels of the 2 µm lattice.

Stages
  power    error scaling with sampled area per depth (error_vs_area method on the matched tiles: tile fractions f,
           closed-form tile block bootstrap SD within the subset (error_vs_area.closed_sd), 8 subset draws shared across
           depths, 200 most abundant genes; pair set SD < 0.5 and an interval (flag != 2) at f = 1)
           -> results/ovff/depth/power_<run>.json
  scaling  sigma(log g) vs depth at fixed area: beta (sigma ∝ m^-beta) and floor share over the thinned series, true min vs
           thinned-to-min, and the same-UMI-budget comparison depth vs area -> results/ovff/depth/scaling.{json,md}
  bias     log g at min depth / thinned vs deep on the same matched tiles (closed-form tile block bootstrap, delta method with
           joint tiles, ovff_crossplatform.compare_pair): mean and RMS difference, slope, fraction within the combined
           SD, per r bin; dependence on abundance -> results/ovff/depth/bias.{json,md}
Quick-looks: figures/quicklook/ovff_depth_{power,scaling,bias}.png
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
import ovff_crossplatform as X  # noqa: E402
from error_vs_area import closed_sd  # noqa: E402

OUT = REPO / "results/ovff/depth"
QL = REPO / "figures/quicklook"
THIN = [0.335, 0.5, 0.25, 0.125]
DEPTHS = {"deep": X.run_name("ovff_hd_probe_deep", "5k", "ref", 500), "min": X.run_name("ovff_hd_probe_min", "5k", "ref", 500),
          **{f"thin{p:g}": X.run_name("ovff_hd_probe_deep", "5k", "ref", 500, thin=p) for p in THIN}}
UMI_FRAC = {"deep": 1.0, "min": 268606784 / 801955840, **{f"thin{p:g}": p for p in THIN}}   # results/qc/ovff_hd_probe_{deep,min}.json
FRACS = [0.05, 0.1, 0.125, 0.25, 0.335, 0.5, 1.0]


def matched_tiles(tile_um=500):
    tab = pd.read_csv(X.OUT / f"tiles_5k_t{tile_um}.csv")
    return list(zip(tab[tab.matched].ix, tab[tab.matched].iy))


def load_depth(name, genes=None, n_top=None, keys=None):
    r = X.load_tensor(name)
    if keys is not None:
        idx = X.tile_index(r, keys); assert (idx >= 0).all(), f"{name}: {int((idx < 0).sum())} matched tiles missing"
        r["ct"], r["tt"], r["ix"], r["iy"] = r["ct"][idx], r["tt"][idx], r["ix"][idx], r["iy"][idx]
    if genes is not None:
        ii = [r["feats"].index(g) for g in genes]
        r["ct"] = np.ascontiguousarray(r["ct"][:, :, ii][:, :, :, ii]); r["tt"] = r["tt"][:, ii]; r["feats"] = list(genes)
    elif n_top:
        top = np.argsort(-r["tt"].sum(0))[:n_top]; top.sort()
        r["ct"] = np.ascontiguousarray(r["ct"][:, :, top][:, :, :, top]); r["tt"] = r["tt"][:, top]; r["feats"] = [r["feats"][i] for i in top]
    return r


# ----------------------------------------------------------------------------- power (area scaling per depth)
def stage_power(a):
    OUT.mkdir(parents=True, exist_ok=True)
    keys = matched_tiles(); T = len(keys)
    genes = load_depth(DEPTHS["deep"], n_top=a.n_genes, keys=keys)["feats"]     # the same 200 genes (deep abundance) for every depth
    rng0 = np.random.default_rng(a.seed)
    subsets = {f: [rng0.choice(T, max(3, int(round(f * T))), replace=False) for _ in range(a.n_subsets if f < 1 else 1)] for f in FRACS}
    for dep, name in DEPTHS.items():
        if not (X.TP / f"{name}.npz").exists():
            X.log("missing", name); continue
        out = OUT / f"power_{dep}.json"
        if out.exists() and not a.force:
            X.log("exists", out); continue
        t0 = time.time()
        r = load_depth(name, genes=genes, keys=keys)
        ct = np.ascontiguousarray(r["ct"], dtype=np.float64); tt = r["tt"]; G = ct.shape[2]; iu = np.triu_indices(G)
        X.log(f"{dep}: {T} matched tiles, {ct.shape[1]} r bins, {G} genes, {tt.sum():.4g} counts")
        sig = {}; flag1 = None
        for f in FRACS:
            reps = []
            for s, idx in enumerate(subsets[f]):
                sd, flag = closed_sd(ct, tt, idx)
                reps.append(sd[:, iu[0], iu[1]])
                if f == 1:
                    flag1 = flag[:, iu[0], iu[1]]
            sig[f] = np.stack(reps)
        lf = np.log(np.array(FRACS)); lf_c = lf - lf.mean()
        S = np.stack([np.nanmean(sig[f], 0) for f in FRACS])          # (nf, K, n_pairs)
        with np.errstate(all="ignore"):
            ls = np.log(S)
        good_noflag = np.isfinite(ls).all(0) & (S[-1] < 0.5) & (S[-1] > 0)
        good = good_noflag & (flag1 < 2)                               # pairs with an interval at f = 1 (T_eff >= 5)
        alpha = -(np.tensordot(lf_c, ls - ls.mean(0), axes=(0, 0)) / (lf_c ** 2).sum())
        n_t = np.array([len(subsets[f][0]) for f in FRACS])
        corr = np.log(np.sqrt(n_t / (n_t - 1)))                        # divisor-1 bias of the SD of an n-tile subset
        alpha_corr = -(np.tensordot(lf_c, (ls + corr[:, None, None]) - (ls + corr[:, None, None]).mean(0), axes=(0, 0)) / (lf_c ** 2).sum())
        rows = []
        rng = np.random.default_rng(a.seed)
        for k in range(ct.shape[1]):
            ok = good[k]
            rec = dict(k=k, r_from=float(r["r_from"][k]), r_to=float(r["r_to"][k]), r_um=float(r["r"][k]), n_pairs=int(ok.sum()),
                       n_pairs_excluded_flag2=int((good_noflag[k] & ~ok).sum()))
            if ok.sum() >= 20:
                al = alpha[k, ok]
                al_sub = []
                for s in range(a.n_subsets):
                    Ss = np.stack([sig[f][min(s, sig[f].shape[0] - 1), k, ok] for f in FRACS])
                    with np.errstate(all="ignore"):
                        lss = np.log(Ss)
                    fin = np.isfinite(lss).all(0)
                    al_sub.append(np.median(-(lf_c @ (lss[:, fin] - lss[:, fin].mean(0))) / (lf_c ** 2).sum()))
                bs = [np.median(al[rng.integers(0, len(al), len(al))]) for _ in range(200)]
                rec.update(alpha_median=float(np.median(al)), alpha_median_subset_corrected=float(np.median(alpha_corr[k, ok])), alpha_ci_pairs=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                           alpha_sd_subsets=float(np.std(al_sub, ddof=1)), alpha_iqr_pairs=[float(np.percentile(al, 25)), float(np.percentile(al, 75))],
                           sigma_median_by_f={f"{f:g}": float(np.nanmedian(S[i, k, ok])) for i, f in enumerate(FRACS)},
                           sigma_sd_over_subsets_by_f={f"{f:g}": float(np.nanmedian(np.nanstd(sig[f][:, k, ok], 0, ddof=1))) if sig[f].shape[0] > 1 else 0.0 for f in FRACS})
            rows.append(rec)
        json.dump(dict(depth=dep, run=name, umi_fraction=UMI_FRAC[dep], n_tiles=T, n_genes=G, fractions=FRACS, n_subsets=a.n_subsets, n_boot="closed", uncertainty="closed",
                       total_counts=float(tt.sum()), rows=rows, seconds=time.time() - t0), open(out, "w"), indent=1)
        X.log(f"{dep}: " + " ".join(f"r{row['r_from']:.0f}: a={row['alpha_median']:.3f}±{row['alpha_sd_subsets']:.3f} s1={row['sigma_median_by_f']['1']:.3f}" for row in rows if "alpha_median" in row and row["r_from"] in X.R_QUOTE)
              + f" ({time.time()-t0:.0f}s, maxRSS {X.maxrss_gb():.1f} GB)")
        del r, ct


# ----------------------------------------------------------------------------- scaling with depth at fixed area
def stage_scaling(a):
    P = {dep: json.load(open(OUT / f"power_{dep}.json")) for dep in DEPTHS if (OUT / f"power_{dep}.json").exists()}
    thinned = ["thin0.125", "thin0.25", "thin0.335", "thin0.5", "deep"]
    ms = np.array([UMI_FRAC[d] for d in thinned]); lm = np.log(ms); lm_c = lm - lm.mean()
    rows = []
    for i, rec in enumerate(P["deep"]["rows"]):
        if "alpha_median" not in rec: continue
        row = dict(k=rec["k"], r_from=rec["r_from"], r_um=rec["r_um"])
        s1 = {d: P[d]["rows"][i]["sigma_median_by_f"]["1"] for d in P if "sigma_median_by_f" in P[d]["rows"][i]}
        row["sigma_full_area"] = s1
        row["sigma_sd_subsets"] = {d: P[d]["rows"][i]["sigma_sd_over_subsets_by_f"]["0.5"] for d in P if "sigma_median_by_f" in P[d]["rows"][i]}
        if all(d in s1 for d in thinned):
            ls = np.log([s1[d] for d in thinned])
            row["beta_thinned"] = float(-(lm_c @ (ls - ls.mean())) / (lm_c ** 2).sum())
            x = 1 / ms ** 2; xc = x - x.mean(); y = np.array([s1[d] for d in thinned]) ** 2
            A = max(float((xc @ (y - y.mean())) / (xc ** 2).sum()), 0.0); B = max(float(y.mean() - A * x.mean()), 0.0)
            row["floor_share_thinned"] = B / max(A + B, 1e-300)
        if "min" in s1 and "thin0.335" in s1:
            row["ratio_true_min_over_thinned"] = s1["min"] / s1["thin0.335"]
            row["ratio_min_over_deep"] = s1["min"] / s1["deep"]
        # same UMI budget: min depth on the full area (f = 1, 0.335 of the UMIs) vs deep on 0.335 of the area
        sd_area = P["deep"]["rows"][i]["sigma_median_by_f"]
        row["sigma_deep_area0.335"] = sd_area["0.335"]; row["sigma_deep_area0.5"] = sd_area["0.5"]
        row["alpha_by_depth"] = {d: [P[d]["rows"][i]["alpha_median"], P[d]["rows"][i]["alpha_sd_subsets"]] for d in P if "alpha_median" in P[d]["rows"][i]}
        rows.append(row)
    out = dict(depths={d: dict(run=DEPTHS[d], umi_fraction=UMI_FRAC[d], total_counts=P[d]["total_counts"]) for d in P}, n_tiles=P["deep"]["n_tiles"], n_genes=P["deep"]["n_genes"], rows=rows)
    json.dump(out, open(OUT / "scaling.json", "w"), indent=1)
    L = [f"# Depth ablation, Visium HD probe (ovarian FF), {P['deep']['n_tiles']} matched 500 µm tiles, {P['deep']['n_genes']} most abundant genes; σ = pair-median tile-bootstrap SD of log g at full area", "",
         "| r_from [µm] | σ deep | σ min (true, 0.335 UMI) | σ thin 0.335 | σ thin 0.5 / 0.25 / 0.125 | min / thin0.335 | min / deep | β (σ ∝ m^−β, thinned) | floor share | σ deep on 0.335 of the area (same UMI budget as min) | α deep / min (± SD subsets) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s = r["sigma_full_area"]
        if r["r_from"] not in X.R_QUOTE + [256.0, 1024.0, 2048.0]: continue
        L.append(f"| {r['r_from']:.0f} | {s.get('deep', np.nan):.3f} | {s.get('min', np.nan):.3f} | {s.get('thin0.335', np.nan):.3f} | {s.get('thin0.5', np.nan):.3f} / {s.get('thin0.25', np.nan):.3f} / {s.get('thin0.125', np.nan):.3f} | "
                 f"{r.get('ratio_true_min_over_thinned', np.nan):.2f} | {r.get('ratio_min_over_deep', np.nan):.2f} | {r.get('beta_thinned', np.nan):.2f} | {r.get('floor_share_thinned', np.nan):.2f} | {r['sigma_deep_area0.335']:.3f} | "
                 + " / ".join(f"{r['alpha_by_depth'][d][0]:.2f} ± {r['alpha_by_depth'][d][1]:.2f}" for d in ("deep", "min") if d in r["alpha_by_depth"]) + " |")
    (OUT / "scaling.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for d, st in (("deep", "-"), ("min", "-"), ("thin0.335", "--"), ("thin0.125", ":")):
        if d in P:
            rr = [r for r in P[d]["rows"] if "alpha_median" in r]
            axs[0].plot([r["r_um"] for r in rr], [r["sigma_median_by_f"]["1"] for r in rr], st, label=f"{d} (UMI ×{UMI_FRAC[d]:.2f})")
            axs[2].errorbar([r["r_um"] for r in rr], [r["alpha_median"] for r in rr], yerr=[r["alpha_sd_subsets"] for r in rr], fmt=st, label=d, capsize=2)
    axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("r [µm]"); axs[0].set_ylabel("median σ(log g), full area"); axs[0].legend(fontsize=7, frameon=False)
    for r in rows:
        if r["r_from"] in (4.0, 20.0, 96.0, 1024.0) and "beta_thinned" in r:
            axs[1].plot(ms, [r["sigma_full_area"][d] for d in thinned], "o-", ms=3, label=f"thinned, r ≈ {r['r_um']:.0f} µm (β = {r['beta_thinned']:.2f})")
            axs[1].plot([UMI_FRAC["min"]], [r["sigma_full_area"]["min"]], "k*", ms=8)
    axs[1].set_xscale("log"); axs[1].set_yscale("log"); axs[1].set_xlabel("UMI fraction m (stars: the real min-depth run)"); axs[1].set_ylabel("median σ(log g)"); axs[1].legend(fontsize=6, frameon=False)
    axs[2].axhline(0.5, c="k", lw=0.5, ls="--"); axs[2].set_xscale("log"); axs[2].set_xlabel("r [µm]"); axs[2].set_ylabel("α (σ ∝ area^−α)"); axs[2].legend(fontsize=7, frameon=False)
    fig.suptitle("Depth ablation: Visium HD probe deep vs min (same library) vs binomial thinning", fontsize=9); fig.tight_layout()
    fig.savefig(QL / "ovff_depth_scaling.png", dpi=130)
    # power quick-look: σ vs f per depth at r ≈ 20 µm and 100 µm
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.4))
    for ax, rf in zip(axs, (20.0, 96.0)):
        for d in P:
            rec = [r for r in P[d]["rows"] if r["r_from"] == rf and "alpha_median" in r]
            if rec:
                ax.plot(FRACS, [rec[0]["sigma_median_by_f"][f"{f:g}"] for f in FRACS], "o-", ms=3, label=f"{d}: α = {rec[0]['alpha_median']:.2f}")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("tile fraction f"); ax.set_ylabel("median σ(log g)"); ax.set_title(f"r_from = {rf:g} µm", fontsize=9); ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(QL / "ovff_depth_power.png", dpi=130)


# ----------------------------------------------------------------------------- bias with depth
def stage_bias(a):
    keys = matched_tiles()
    deep = load_depth(DEPTHS["deep"], keys=keys)
    ks = sorted(set([int(np.argmin(np.abs(deep["r_from"] - rf))) for rf in X.R_QUOTE] + [k for k in range(len(deep["r_from"])) if 4 <= deep["r_from"][k] <= 3000]))
    ia = list(range(len(deep["feats"])))
    res = dict(n_tiles=len(keys), n_genes=len(deep["feats"]), uncertainty="closed", n_boot="closed", comparisons={})
    abund = np.log10(deep["tt"].sum(0) + 1); iu = np.triu_indices(len(ia)); pair_ab = np.minimum(abund[iu[0]], abund[iu[1]])
    for dep in [d for d in DEPTHS if d != "deep"]:
        if not (X.TP / f"{DEPTHS[dep]}.npz").exists(): continue
        t0 = time.time()
        other = load_depth(DEPTHS[dep], genes=deep["feats"], keys=keys)
        idx = np.arange(len(keys))
        rows, valid = X.compare_pair(deep, other, idx, idx, ia, ia, ks, joint=True)
        # abundance dependence of the difference at the quoted r: mean (other - deep) in abundance terciles
        W1 = np.ones((1, len(keys)))
        for row in rows:
            k = row["k"]
            ga = X.logg_from_weights(deep["ct"], deep["tt"], W1, k)[0][iu]; gb = X.logg_from_weights(other["ct"], other["tt"], W1, k)[0][iu]
            ok = valid[k]
            q = np.nanpercentile(pair_ab[ok], [33.3, 66.7])
            terc = np.digitize(pair_ab[ok], q)
            row["mean_diff_by_abundance_tercile"] = [float(np.mean((gb - ga)[ok][terc == t])) for t in range(3)]
            row["rms_by_abundance_tercile"] = [float(np.sqrt(np.mean((gb - ga)[ok][terc == t] ** 2))) for t in range(3)]
            # compression: regression of (other - deep) on deep (negative slope = shrinkage of log g toward 0 at lower depth)
            row["shrinkage_slope"] = float(np.polyfit(ga[ok], (gb - ga)[ok], 1)[0])
        res["comparisons"][dep] = dict(run=DEPTHS[dep], umi_fraction=UMI_FRAC[dep], rows=rows, seconds=time.time() - t0)
        X.log(f"{dep} vs deep: " + " ".join(f"r{r['r_from']:.0f}: Δ={r['mean_diff']:+.3f} rms={r['rms']:.3f} slope={r['slope']:.3f} w1sd={r['within_1sd']:.2f}" for r in rows if r["r_from"] in X.R_QUOTE) + f" ({time.time()-t0:.0f}s)")
        del other
    json.dump(res, open(OUT / "bias.json", "w"), indent=1, default=float)
    write_bias_md(res)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for dep, c in res["comparisons"].items():
        rr = [r for r in c["rows"] if "pearson" in r]; r_ = [r["r_um"] for r in rr]
        axs[0].plot(r_, [r["mean_diff"] for r in rr], "-" if dep == "min" else "--", label=dep); axs[0].fill_between(r_, [r["mean_diff_ci"][0] for r in rr], [r["mean_diff_ci"][1] for r in rr], alpha=0.15)
        axs[1].plot(r_, [r["rms"] for r in rr], "-" if dep == "min" else "--", label=dep)
        axs[2].plot(r_, [r["slope"] for r in rr], "-" if dep == "min" else "--", label=dep)
    axs[0].axhline(0, c="k", lw=0.5); axs[2].axhline(1, c="k", lw=0.5)
    for ax, yl in zip(axs, ("mean log g(other) − log g(deep)", "RMS difference", "slope log g(other) vs log g(deep)")):
        ax.set_xscale("log"); ax.set_xlabel("r [µm]"); ax.set_ylabel(yl); ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Depth ablation: bias of log g with depth (same tiles; min = real min-depth run, thin = binomial thinning of deep)", fontsize=9)
    fig.tight_layout(); fig.savefig(QL / "ovff_depth_bias.png", dpi=130)


def write_bias_md(res):
    L = [f"# Depth bias: log g(other) vs log g(deep) on the same {res['n_tiles']} matched tiles, {res['n_genes']} genes; 95 % intervals of the "
         "closed-form tile block bootstrap (delta method, joint tiles), estimate ± 1.96 SD; pairs with SD < 0.5 and an interval (T_eff ≥ 5) at both depths", "",
         "| depth (UMI frac) | r_from [µm] | pairs (flag-2 excluded) | mean Δ [CI] | RMS Δ [CI] | slope [CI] | Pearson | within 1 SD / 2 SD | Δ by abundance tercile (low/mid/high) | shrinkage slope dΔ/d log g |", "|---|---|---|---|---|---|---|---|---|---|"]
    for dep, c in res["comparisons"].items():
        for r in c["rows"]:
            if r["r_from"] not in X.R_QUOTE + [256.0, 1024.0] or "pearson" not in r: continue
            mc, rc, sc = r["mean_diff_ci"], r["rms_ci"], r["slope_ci"]
            L.append(f"| {dep} ({c['umi_fraction']:.3f}) | {r['r_from']:.0f} | {r['n']} ({r.get('n_pairs_excluded_flag2', 0)}) | {r['mean_diff']:+.3f} [{mc[0]:+.3f}, {mc[1]:+.3f}] | {r['rms']:.3f} [{rc[0]:.3f}, {rc[1]:.3f}] | "
                     f"{r['slope']:.3f} [{sc[0]:.3f}, {sc[1]:.3f}] | {r['pearson']:.3f} | {r['within_1sd']:.2f} / {r['within_2sd']:.2f} | "
                     + " / ".join(f"{v:+.3f}" for v in r["mean_diff_by_abundance_tercile"]) + f" | {r['shrinkage_slope']:+.3f} |")
    (OUT / "bias.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


def stage_md(a):
    """Rewrite bias.md from bias.json."""
    write_bias_md(json.load(open(OUT / "bias.json")))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["power", "scaling", "bias", "md"])
    ap.add_argument("--n-genes", type=int, default=200); ap.add_argument("--n-subsets", type=int, default=8); ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"power": stage_power, "scaling": stage_scaling, "bias": stage_bias, "md": stage_md}[a.stage](a)


if __name__ == "__main__":
    main()
