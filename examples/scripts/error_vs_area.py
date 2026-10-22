#!/usr/bin/env python
"""Preprint Fig. 4a, b: scaling of the statistical error of log g_ab(r) with the
sampled area. From the per-tile count tensors of a scote two-point run (counts_tiles (T, K, G, G), totals_tiles (T, G);
square tiles of tile_um), for tile fractions f: draw a random subset of round(f T) tiles and compute the tile block
bootstrap SD of log g *within* the subset -> sigma_f(k, ab); repeat over `n_subsets` subsets and average sigma_f. Fit
log sigma_f = -alpha log f + c per r bin (median over gene pairs with sigma_1 < 0.5 and an interval at f = 1 (reliability
flag != 2, T_eff >= 5) of the per-pair least-squares alpha; error = SD over the subset draws). Independent tiles give
alpha = 0.5; alpha < 0.5 flags correlated tiles (structure on the tile scale), alpha > 0.5 pairs whose counts are dominated
by few tiles (rare / clustered genes: the SD is then driven by whether the rare tiles are drawn).

Within-subset SD: the closed-form (B -> infinity, linearized) tile block bootstrap of scote.uncertainty.closed_form_log_g on
the subset tensor (gene-subsetted to the top-N genes first, so the random-label reference is the one of the top-N panel;
symmetrized slabs; no few-tiles warning, the subsets have 3-20 tiles by design). Like the resampling bootstrap it has
divisor 1, so it underestimates the SD of an n-tile subset by sqrt((n - 1) / n); `alpha_median_subset_corrected` is
alpha with that factor removed, and the difference is the small-subset share of the deficit below 0.5.

Subsets: drawn from a random stream that interleaves the subset draws with within-subset resampling draws
(`subset_draws` replays both; seed --seed), so the tile subsets are reproducible.

Usage: python scripts/error_vs_area.py --npz data/twopoint/x5k_lymph_node_bin2um_top200.npz --out results/error_scaling/x5k_top200.json
"""
import argparse, json, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
from scote.uncertainty import closed_form_log_g  # noqa: E402


def subset_draws(T, fracs, n_subsets, n_boot, seed):
    """Tile subsets of the area-scaling design. From one generator default_rng(seed), per f and subset: first
    idx = rng.choice(T, n_t, replace=False), then n_boot within-subset resample draws (rng.integers(0, n_t, n_t) each),
    which are consumed but not used (the within-subset SD is the closed form); this fixes the subsets of the published
    tables. Returns ({f: [idx, ...]}, rng) with rng positioned after these draws (used for the pair bootstrap of the
    median alpha)."""
    rng = np.random.default_rng(seed)
    out = {}
    for f in fracs:
        n_t = max(3, int(round(f * T)))
        out[f] = []
        for _ in range(n_subsets if f < 1 else 1):
            out[f].append(rng.choice(T, n_t, replace=False))
            for _ in range(n_boot):
                rng.integers(0, n_t, n_t)
    return out, rng


def closed_sd(counts, totals, idx=None):
    """Closed-form tile block bootstrap of log g on the tiles in `idx` (all tiles if None): (sd (K, G, G), flag (K, G, G)).
    sd is NaN where the subset has no pair (P = 0); flag = reliability flag of the subset (2: T_eff < 5)."""
    C = counts if idx is None else counts[np.asarray(idx)]
    N = totals if idx is None else totals[np.asarray(idx)]
    r = closed_form_log_g(C, N, min_tiles_warn=0)
    return r.sd, r.flag


def boot_sd(counts, totals, n_boot=None, rng=None, idx=None):
    """Tile block bootstrap SD of log g for the tiles in `idx` (all tiles if None), in closed form (closed_sd); n_boot and
    rng are accepted for the callers of the Monte Carlo version and ignored."""
    return closed_sd(counts, totals, idx)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--fractions", default="0.05,0.1,0.25,0.5,1"); ap.add_argument("--n-subsets", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=100, help="resample draws consumed per subset (subset_draws; fixes "
                    "the random stream), the SD is the closed form"); ap.add_argument("--max-genes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n-pair-boot", type=int, default=200)
    ap.add_argument("--fig", default=None)
    a = ap.parse_args()
    t0 = time.time()
    d = np.load(a.npz, allow_pickle=True)
    ct = d["counts_tiles"]; tt = d["totals_tiles"]; feats = list(map(str, d["features"])); r = d["r_central"]; r_from = d["r_from"]; r_to = d["r_to"]
    G = ct.shape[2]
    if G > a.max_genes:   # keep the most abundant genes
        top = np.argsort(-tt.sum(0))[: a.max_genes]; top.sort()
        ct = np.ascontiguousarray(ct[:, :, top][:, :, :, top]); tt = tt[:, top]; feats = [feats[i] for i in top]; G = len(top)
    ct = np.ascontiguousarray(ct, dtype=np.float64)
    T = ct.shape[0]
    print(f"{a.npz}: {T} tiles, {ct.shape[1]} r bins, {G} genes ({time.time()-t0:.0f}s load)", flush=True)
    fracs = [float(f) for f in a.fractions.split(",")]
    subsets, rng = subset_draws(T, fracs, a.n_subsets, a.n_boot, a.seed)
    iu = np.triu_indices(G)
    sig = {}   # f -> (n_subsets, K, n_pairs)
    n_t_by_f = {}
    flag1 = None
    for f in fracs:
        n_t = max(3, int(round(f * T))); n_t_by_f[f] = n_t
        reps = []
        for idx in subsets[f]:
            sd, flag = closed_sd(ct, tt, idx)
            reps.append(sd[:, iu[0], iu[1]])
            if f == 1:
                flag1 = flag[:, iu[0], iu[1]]
        sig[f] = np.stack(reps)
        print(f"  f={f:g}: {n_t} tiles, median sigma over pairs/bins {np.nanmedian(sig[f]):.3f} ({time.time()-t0:.0f}s)", flush=True)
    sd_check = None
    if G == len(d["features"]) and "sd" in d.files:   # no gene subsetting: f = 1 must reproduce the stored closed-form sd
        sd_st = d["sd"][:, iu[0], iu[1]]; m = np.isfinite(sd_st) & np.isfinite(sig[1.0][0]) & (sd_st > 0)
        sd_check = float(np.max(np.abs(sig[1.0][0][m] / sd_st[m] - 1)))
        print(f"  f=1 sd vs stored npz sd: max rel diff {sd_check:.2e}", flush=True)
    # per-pair alpha per r bin from log sigma vs log f (mean sigma over subsets)
    lf = np.log(np.array(fracs))
    S = np.stack([np.nanmean(sig[f], 0) for f in fracs])       # (nf, K, n_pairs)
    with np.errstate(all="ignore"):
        ls = np.log(S)
    good_noflag = np.isfinite(ls).all(0) & (S[-1] < 0.5) & (S[-1] > 0)   # pairs with a finite, informative error at f = 1
    good = good_noflag & (flag1 < 2)                                      # ... and an interval at f = 1 (T_eff >= 5)
    lf_c = lf - lf.mean()
    alpha = -(np.tensordot(lf_c, ls - ls.mean(0), axes=(0, 0)) / (lf_c ** 2).sum())   # (K, n_pairs)
    # the divisor-1 (bootstrap) SD of an n-tile subset is sqrt((n - 1) / n) of the n-tile sampling SD: alpha without it
    corr = np.log(np.sqrt(np.array([n_t_by_f[f] / (n_t_by_f[f] - 1) for f in fracs])))
    alpha_corr = -(np.tensordot(lf_c, (ls + corr[:, None, None]) - (ls + corr[:, None, None]).mean(0), axes=(0, 0)) / (lf_c ** 2).sum())
    rows = []
    for k in range(ct.shape[1]):
        ok = good[k]
        if ok.sum() < 20:
            rows.append(dict(r_from=float(r_from[k]), r_to=float(r_to[k]), r_um=float(r[k]), n_pairs=int(ok.sum()),
                             n_pairs_excluded_flag2=int((good_noflag[k] & ~ok).sum()))); continue
        al = alpha[k, ok]
        bs = [np.median(al[rng.integers(0, len(al), len(al))]) for _ in range(a.n_pair_boot)]
        # subset-draw uncertainty: alpha from each subset replicate separately (f<1 replicates paired with the f=1 value)
        al_sub = []
        for s in range(a.n_subsets):
            Ss = np.stack([sig[f][min(s, sig[f].shape[0] - 1), k, ok] for f in fracs])
            with np.errstate(all="ignore"):
                lss = np.log(Ss)
            fin = np.isfinite(lss).all(0)
            al_sub.append(np.median(-(lf_c @ (lss[:, fin] - lss[:, fin].mean(0))) / (lf_c ** 2).sum()))
        rows.append(dict(r_from=float(r_from[k]), r_to=float(r_to[k]), r_um=float(r[k]), n_pairs=int(ok.sum()),
                         n_pairs_excluded_flag2=int((good_noflag[k] & ~ok).sum()),
                         alpha_median=float(np.median(al)), alpha_median_subset_corrected=float(np.median(alpha_corr[k, ok])), alpha_ci_pairs=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                         alpha_sd_subsets=float(np.std(al_sub, ddof=1)), alpha_iqr_pairs=[float(np.percentile(al, 25)), float(np.percentile(al, 75))],
                         sigma_median_by_f={f"{f:g}": float(np.nanmedian(S[i, k, ok])) for i, f in enumerate(fracs)}))
    js = Path(str(a.npz).replace(".npz", ".json"))
    out = dict(npz=a.npz, n_tiles=T, tile_um=(json.load(open(js)).get("tile_um") if js.exists() else None),
               n_genes=G, fractions=fracs, n_subsets=a.n_subsets, n_boot="closed", uncertainty="closed",
               n_tiles_by_f={f"{f:g}": n_t_by_f[f] for f in fracs}, subset_seed=a.seed,
               small_subset_alpha_shift=float(-(lf_c @ (corr - corr.mean())) / (lf_c ** 2).sum()),
               f1_sd_vs_npz_max_rel=sd_check, rows=rows, seconds=time.time() - t0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    for row in rows:
        if "alpha_median" in row:
            print(f"  r={row['r_um']:8.1f} um: alpha = {row['alpha_median']:.3f} [{row['alpha_ci_pairs'][0]:.3f}, {row['alpha_ci_pairs'][1]:.3f}] (pairs) ± {row['alpha_sd_subsets']:.3f} (subsets), "
                  f"sigma(f={fracs[-1]:g}) {row['sigma_median_by_f'][f'{fracs[-1]:g}']:.3f}, sigma(f={fracs[0]:g}) {row['sigma_median_by_f'][f'{fracs[0]:g}']:.3f}, n={row['n_pairs']}")
    if a.fig:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
        ks = [i for i, row in enumerate(rows) if "alpha_median" in row]
        axs[0].errorbar([rows[i]["r_um"] for i in ks], [rows[i]["alpha_median"] for i in ks],
                        yerr=[[rows[i]["alpha_median"] - rows[i]["alpha_ci_pairs"][0] for i in ks], [rows[i]["alpha_ci_pairs"][1] - rows[i]["alpha_median"] for i in ks]], fmt="o-", ms=3)
        axs[0].axhline(0.5, c="k", lw=0.6, ls="--"); axs[0].set_xscale("log"); axs[0].set_xlabel("r [µm]"); axs[0].set_ylabel("α in σ(log g) ∝ f^−α (median over pairs, 95 % CI)")
        for i in ks[::max(1, len(ks) // 6)]:
            axs[1].plot(fracs, [rows[i]["sigma_median_by_f"][f"{f:g}"] for f in fracs], "o-", ms=3, label=f"r ≈ {rows[i]['r_um']:.0f} µm")
        axs[1].set_xscale("log"); axs[1].set_yscale("log"); axs[1].set_xlabel("tile fraction f"); axs[1].set_ylabel("median σ(log g) over pairs"); axs[1].legend(fontsize=7, frameon=False)
        fig.suptitle(f"Error scaling with sampled area ({Path(a.npz).stem}, {T} tiles, {G} genes)", fontsize=9)
        fig.tight_layout(); fig.savefig(a.fig, dpi=140)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
