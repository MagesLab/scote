#!/usr/bin/env python
"""Method-method scatter statistics (preprint Fig. 2b, Supplementary Figs. S1 and S2).

For every comparison of two sections and every length scale it compares log g_ab(r) of all gene pairs a <= b (upper triangle
incl. the diagonal) that are finite with tile-bootstrap SD < 0.5 and an interval (reliability flag != 2, T_eff >= 5) on both
sides (the filter of figures/make_fig2.py): Pearson r, Deming slope of B on A (error-variance ratio lambda = median SD_B^2 /
median SD_A^2 of the pair set), OLS slope, major-axis (orthogonal) slope, RMS difference and mean difference B - A.
Intervals: closed-form tile block bootstrap over the 500 µm tiles with the delta method across pairs. Per section, the
per-tile influence l_t(ab) of every log g (scote.uncertainty.influence_log_g, symmetrized slabs; summed bin sets as one bin) is
combined with the gradient of the statistic over the pairs (scote.uncertainty.grad_*); the two sections are different tissue,
so their tile resamples are independent and the variances add (scote.uncertainty.delta_var, joint=False). The pair set and
lambda are fixed to those of the point estimate. Interval = estimate +- 1.96 SD (scote.uncertainty.stat_ci). SDs, flags and
n_eff of single bins are read from the npz (closed form); the influence-based SD of each bin is checked against
the stored one.

Scales. Lattice bins [10, 12), [96, 112) and [896, 1024) µm (the bins containing 10, 100 and 1000 µm). Visium v2 (55 µm spots
on a 64 µm lattice) has no [96, 112) bin; at '100 µm' it is compared on its first neighbor shell [64, 128) against the sum of
the four 2 µm-lattice bins [64, 80) ... [112, 128) of the other platform (pair counts and random-label references summed,
pairs required to pass the SD filter in all four sub-bins). From 256 µm on the v2 shells coincide with the 2 µm hierarchy.

Comparisons. CRC P2: Xenium (molecule-level run; identical integer counts to the 2 µm-binned run for r >= 8 µm) vs Visium HD
at the three scales; Xenium vs Visium v2 and Visium HD vs Visium v2 at 100 and 1000 µm. Spleen WT709: Xenium vs Visium HD and
Xenium vs MERSCOPE at the three scales.

Usage: python scripts/scatter_stats.py [--only crc|spleen]   (loads the per-tile tensors, 14.5 GB for the
Xenium CRC run) -> results/platforms/scatter_stats.json
"""
from __future__ import annotations

import argparse, json, time, resource
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
from scote import uncertainty as TV  # noqa: E402

TP = REPO / "data/twopoint"; RES = REPO / "results/platforms"
SD_MAX = 0.5
RUNS = {"crc_xenium": "xenium_crc_p2_mol_common.npz", "crc_hd": "hd_crc_p2_bin2um_common.npz", "crc_v2": "v2_crc_p2_spots_common.npz",
        "sp_xenium": "sb_xenium_WT709_rep1_bin2um_common.npz", "sp_hd": "sb_hd_WT709_bin2um_common.npz", "sp_merscope": "sb_merscope_WT709_bin2um_common.npz"}
# scale label -> list of lattice bins [from, to) summed for this scale, per run kind
FINE = {"10": [(10, 12)], "100": [(96, 112)], "1000": [(896, 1024)]}
FINE_V2 = {"100": [(64, 80), (80, 96), (96, 112), (112, 128)], "1000": [(896, 1024)]}
SPOTS = {"100": [(64, 128)], "1000": [(896, 1024)]}
COMPARISONS = {   # name: (A, B, {scale: (bins A, bins B)})
    "crc_xenium_hd": ("crc_xenium", "crc_hd", {s: (FINE[s], FINE[s]) for s in FINE}),
    "crc_xenium_v2": ("crc_xenium", "crc_v2", {s: (FINE_V2[s], SPOTS[s]) for s in SPOTS}),
    "crc_hd_v2": ("crc_hd", "crc_v2", {s: (FINE_V2[s], SPOTS[s]) for s in SPOTS}),
    "spleen_xenium_hd": ("sp_xenium", "sp_hd", {s: (FINE[s], FINE[s]) for s in FINE}),
    "spleen_xenium_merscope": ("sp_xenium", "sp_merscope", {s: (FINE[s], FINE[s]) for s in FINE}),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GB]", *a, flush=True)


def bin_index(r_from, r_to, lo, hi):
    k = np.where(np.isclose(r_from, lo) & np.isclose(r_to, hi))[0]
    assert len(k) == 1, (lo, hi)
    return int(k[0])


def stats(x, y, lam):
    sxx, syy = np.var(x), np.var(y); sxy = np.mean((x - x.mean()) * (y - y.mean()))
    dem = (syy - lam * sxx + np.sqrt((syy - lam * sxx) ** 2 + 4 * lam * sxy ** 2)) / (2 * sxy)
    ma = (syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy ** 2)) / (2 * sxy)
    return dict(pearson=float(sxy / np.sqrt(sxx * syy)), deming=float(dem), ols=float(sxy / sxx), major_axis=float(ma),
                rms=float(np.sqrt(np.mean((y - x) ** 2))), mean_diff=float(np.mean(y - x)))


def gradients(x, y, lam):
    """Value and [d/dx, d/dy] of every statistic of `stats` (scote.uncertainty gradients; Deming and major axis with lambda fixed)."""
    return dict(pearson=TV.grad_pearson(x, y), deming=TV.grad_deming(x, y, lam), ols=TV.grad_ols_slope(x, y),
                major_axis=TV.grad_deming(x, y, 1.0), rms=TV.grad_rms(x, y), mean_diff=TV.grad_mean_diff(x, y))


def prepare(run, binsets):
    """For one run and the needed bin sets: point log g, closed-form SD, per-tile influence l (T, G, G) and n_eff per bin set,
    plus the filter (finite, SD < 0.5 and reliability flag != 2 in every summed sub-bin; flag != 2 of the summed bin)."""
    t0 = time.time()
    d = np.load(TP / RUNS[run], allow_pickle=True)
    if "flag" not in d.files:
        raise SystemExit(f"{RUNS[run]} has no 'flag' key: re-run it with python -m scote twopoint")
    r_from, r_to = d["r_from"], d["r_to"]; G = len(d["features"])
    keys = sorted({b for bs in binsets for b in bs})
    ks = [bin_index(r_from, r_to, *b) for b in keys]
    lg_store, sd_store, flag_store, ne_store = d["log_g"], d["sd"], d["flag"], d["n_eff_pairs"]
    ok_sub = {b: np.isfinite(lg_store[k]) & (sd_store[k] < SD_MAX) & (flag_store[k] < 2) for b, k in zip(keys, ks)}
    ok_sub_noflag = {b: np.isfinite(lg_store[k]) & (sd_store[k] < SD_MAX) for b, k in zip(keys, ks)}
    log(run, "loading counts_tiles ...")
    ct = d["counts_tiles"]; T = ct.shape[0]
    sub = np.ascontiguousarray(ct[:, ks]); del ct
    tt = d["totals_tiles"].astype(np.float64)
    log(run, f"tiles {T}, bins {keys}, loaded in {time.time() - t0:.0f} s")
    out = {}
    for bs in binsets:
        idx = [keys.index(b) for b in bs]
        C = sub[:, idx].sum(1)                                   # (T, G, G) pair counts summed over the sub-bins
        # random-label reference summed over the sub-bins: sum_k P_tot(k) n_a n_b / n_tot^2 = (sum_k P_tot(k)) n_a n_b / n_tot^2
        Ptot_t = sub[:, idx].sum(axis=(1, 2, 3))                 # (T,) all ordered pairs of the bin set per tile
        n = tt.sum(0)
        E = Ptot_t.sum() * np.outer(n, n) / max(n.sum(), 1) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.log(C.sum(0) / E)
        l = TV.influence_log_g(sub, tt, shells=[idx])[:, 0]      # (T, G, G), summed bin set = one bin
        sd = np.sqrt(np.einsum("tab,tab->ab", l, l))
        Cs = 0.5 * (C + np.swapaxes(C, 1, 2)); Ps = Cs.sum(0)     # symmetrized slabs, as the closed form
        with np.errstate(divide="ignore", invalid="ignore"):
            t_eff = Ps ** 2 / np.einsum("tab,tab->ab", Cs, Cs)
        flag = TV.reliability_flag(t_eff)
        ok = np.logical_and.reduce([ok_sub[b] for b in bs]) & np.isfinite(lg) & (flag < 2)
        ok_noflag = np.logical_and.reduce([ok_sub_noflag[b] for b in bs]) & np.isfinite(lg)
        if len(bs) == 1:   # single bin: the stored arrays (closed form) must be reproduced
            k = ks[keys.index(bs[0])]; fin = np.isfinite(lg)
            assert np.allclose(lg[fin], lg_store[k][fin]), "point log g differs from the stored run"
            m = fin & np.isfinite(sd_store[k])
            dsd = float(np.max(np.abs(sd[m] / sd_store[k][m] - 1))) if m.any() else 0.0
            assert dsd < 1e-6, f"influence SD differs from the stored closed-form SD (max rel {dsd:.2e})"
            assert np.array_equal(flag[fin], flag_store[k][fin]), "reliability flag differs from the stored one"
            n_eff = ne_store[k]
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                n_eff = C.sum(0) ** 2 / np.maximum(((Cs - Ps / T) ** 2).sum(0), 1e-300)
            dsd = None
        out[tuple(bs)] = dict(lg=lg, sd=sd, l=l, ok=ok, ok_noflag=ok_noflag, flag=flag, n_eff=n_eff, T=T, sd_check=dsd)
        del C, Cs
    log(run, f"influence done, {time.time() - t0:.0f} s")
    return out, list(map(str, d["features"]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=str(RES / "scatter_stats.json"))
    a = ap.parse_args()
    comps = {k: v for k, v in COMPARISONS.items() if a.only is None or k.startswith(a.only)}
    need = {}
    for A, B, sc in comps.values():
        for bA, bB in sc.values():
            need.setdefault(A, set()).add(tuple(bA)); need.setdefault(B, set()).add(tuple(bB))
    prep, feats = {}, {}
    for run, bsets in need.items():
        prep[run], feats[run] = prepare(run, [list(b) for b in bsets])
    res = dict(rule=__doc__.split("Usage")[0].strip(), sd_max=SD_MAX, n_boot="closed", uncertainty="closed", comparisons={})
    for name, (A, B, sc) in comps.items():
        assert feats[A] == feats[B], "feature order differs"
        G = len(feats[A]); iu = np.triu_indices(G)
        rows = {}
        for s, (bA, bB) in sc.items():
            pa, pb = prep[A][tuple(bA)], prep[B][tuple(bB)]
            ok = (pa["ok"] & pb["ok"])[iu]
            ok_noflag = (pa["ok_noflag"] & pb["ok_noflag"])[iu]
            x, y = pa["lg"][iu][ok], pb["lg"][iu][ok]
            lam = float(np.median(pb["sd"][iu][ok] ** 2) / np.median(pa["sd"][iu][ok] ** 2))
            st = stats(x, y, lam)
            ia, ib = iu[0][ok], iu[1][ok]
            lA, lB = pa["l"][:, ia, ib], pb["l"][:, ia, ib]       # (T_A, n), (T_B, n)
            fl = np.maximum(pa["flag"], pb["flag"])[iu][ok]
            row = dict(scale_um=float(s), bins_A=bA, bins_B=bB, n_pairs=int(ok.sum()),
                       n_pairs_excluded_flag2=int((ok_noflag & ~ok).sum()), n_pairs_flag1=int((fl == 1).sum()),
                       n_tiles_A=int(pa["T"]), n_tiles_B=int(pb["T"]), lambda_deming=lam,
                       median_sd_A=float(np.median(pa["sd"][iu][ok])), median_sd_B=float(np.median(pb["sd"][iu][ok])),
                       sd_logg_A=float(np.std(x)), sd_logg_B=float(np.std(y)), **st)
            for key, (val, (gx, gy)) in gradients(x, y, lam).items():
                assert np.isclose(val, st[key], rtol=1e-9, atol=1e-12), (key, val, st[key])
                var = TV.delta_var([gx, gy], [lA, lB], joint=False)
                row[key + "_sd"] = float(np.sqrt(var)); row[key + "_ci"] = [float(v) for v in TV.stat_ci(st[key], var)]
            # n_eff summary of the limiting platform (for the color scale of the figure)
            ne = np.minimum(pa["n_eff"], pb["n_eff"])[iu][ok]
            row["n_eff_min_quantiles"] = np.percentile(ne, [5, 25, 50, 75, 95]).tolist()
            rows[s] = row
            log(name, s, f"n {row['n_pairs']} (flag-2 excluded {row['n_pairs_excluded_flag2']})",
                {k: round(row[k], 3) for k in ("pearson", "deming", "ols", "major_axis", "rms")}, "deming CI", np.round(row["deming_ci"], 3))
        res["comparisons"][name] = dict(A=A, B=B, run_A=RUNS[A], run_B=RUNS[B], rows=rows)
    res["sd_check_max_rel"] = {run: {"+".join(f"{lo}-{hi}" for lo, hi in bs): v["sd_check"] for bs, v in p.items()} for run, p in prep.items()}
    out = Path(a.out)
    if out.exists() and a.only is not None:   # merge partial runs
        old = json.load(open(out)); old["comparisons"].update(res["comparisons"]); res = old
    json.dump(res, open(out, "w"), indent=1)
    log("wrote", out)


if __name__ == "__main__":
    main()
