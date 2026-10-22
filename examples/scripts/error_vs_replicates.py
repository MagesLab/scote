#!/usr/bin/env python
"""Preprint Fig. 4d, replicate axis: how many sections are needed? Input: results/platforms/crc_replicates.json
(CRC P1/P2/P5, Xenium and Visium HD:
per r bin the median over ≈ 83,000 gene pairs of the between-patient SD of log g (3 sections) and of the within-section tile-bootstrap SD).
Output: results/error_scaling/replicates_sections.{json,md}.

Stage `liver`: the same statement for the MERFISH liver from the four Vizgen slices (Liver1Slice1/2 = animal 1,
Liver2Slice1/2 = animal 2; data/twopoint/error_scaling/vizgen_liver*_bin2um_top200.npz, same top-200 gene list as slice 1): per r bin the
pair-median between-slice SD of log g over the 4 slices (ν = 3), the pooled within-animal SD (two pairs of serial slices, ν = 2), the
between-animal SD of the two animal means (ν = 1) and the within-slice tile-bootstrap SD (closed form; pairs with SD < 0.5 and an
interval, reliability flag != 2, on every slice); sections needed for SEM 0.1 / 0.05 with χ² intervals. Output: results/error_scaling/replicates_liver.{json,md}.

  python scripts/error_vs_replicates.py [crc|liver]

Statement: with n sections the standard error of the cohort mean of log g_ab(r) is s_obs / sqrt(n), where s_obs is the observed
between-section SD (it already contains the within-section measurement noise: s_obs² ≈ s_bio² + s_within²). The number of sections
for a target SEM ε is n = (s_obs / ε)². With only ν = n_sections − 1 = 2 degrees of freedom the SD itself is uncertain: the 95 % χ²
interval of s is [s·sqrt(ν/χ²_{0.975,ν}), s·sqrt(ν/χ²_{0.025,ν})] = [0.52 s, 6.3 s] for ν = 2, so n is quoted with that interval too.
The pair-median SD is the typical gene pair; rare pairs need more.
"""
import json, sys
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.stats import chi2

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "results/platforms/crc_replicates.json"
OUT = REPO / "results/error_scaling/replicates_sections"
TARGETS_R = [6, 10, 24, 96, 224, 448, 1024]
EPS = [0.05, 0.1]
EPS_LIVER = [0.01, 0.02, 0.05, 0.1]   # the liver spread is 10x smaller than the CRC one; SEM 0.1 needs < 1 slice everywhere
LIVER = {"vizgen_liver1slice1": 1, "vizgen_liver1slice2": 1, "vizgen_liver2slice1": 2, "vizgen_liver2slice2": 2}   # slice -> animal
LIVER_NPZ = REPO / "data/twopoint/error_scaling/{}_bin2um_top200.npz"
OUT_LIVER = REPO / "results/error_scaling/replicates_liver"
TARGETS_R_LIVER = [1, 4.7, 10.7, 21, 51, 101, 213, 406, 1085, 2728]


def chi2_factors(nu):
    return float(np.sqrt(nu / chi2.ppf(0.975, nu))), float(np.sqrt(nu / chi2.ppf(0.025, nu)))


def n_sections(s, lo_f, hi_f, eps=EPS):
    return {f"{e}": dict(point=(s / e) ** 2, ci=[(s * lo_f / e) ** 2, (s * hi_f / e) ** 2]) for e in eps}


def liver():
    slices = [k for k in LIVER if LIVER_NPZ.with_name(LIVER_NPZ.name.format(k)).exists()]
    if len(slices) < 2:
        print("need >= 2 liver tensors, have", slices); return
    zs = {k: np.load(LIVER_NPZ.with_name(LIVER_NPZ.name.format(k)), allow_pickle=True) for k in slices}
    feats = [list(map(str, z["features"])) for z in zs.values()]
    assert all(f == feats[0] for f in feats), "gene lists differ between slices"
    G = len(feats[0]); iu = np.triu_indices(G)
    LG = np.stack([zs[k]["log_g"][:, iu[0], iu[1]].astype(np.float64) for k in slices])      # (n_slices, K, n_pairs)
    SD = np.stack([zs[k]["sd"][:, iu[0], iu[1]].astype(np.float64) for k in slices])
    FL = np.stack([zs[k]["flag"][:, iu[0], iu[1]] for k in slices]).max(0)                      # worst reliability flag over the slices
    r_from = zs[slices[0]]["r_from"]; r_c = zs[slices[0]]["r_central"]; K = LG.shape[1]
    animals = np.array([LIVER[k] for k in slices]); an_ids = sorted(set(animals))
    n_sl = len(slices); nu_all = n_sl - 1
    lo_all, hi_all = chi2_factors(nu_all)
    within_pairs = [[i for i in range(n_sl) if animals[i] == a] for a in an_ids if (animals == a).sum() >= 2]
    nu_w = sum(len(p) - 1 for p in within_pairs); lo_w, hi_w = chi2_factors(nu_w) if nu_w else (np.nan, np.nan)
    nu_b = len(an_ids) - 1; lo_b, hi_b = chi2_factors(nu_b) if nu_b else (np.nan, np.nan)
    rows = []
    for k in range(K):
        ok_noflag = np.isfinite(LG[:, k]).all(0) & (SD[:, k] < 0.5).all(0) & (SD[:, k] > 0).all(0)
        ok = ok_noflag & (FL[k] < 2)                                                          # pairs without an interval (T_eff < 5) left out
        if ok.sum() < 20:
            rows.append(dict(r_from=float(r_from[k]), r_um=float(r_c[k]), n_pairs=int(ok.sum()), n_pairs_excluded_flag2=int((ok_noflag & ~ok).sum()))); continue
        lg = LG[:, k, ok]; sd = SD[:, k, ok]
        s_all = float(np.median(lg.std(0, ddof=1)))
        s_w = float(np.median(sd))
        rec = dict(r_from=float(r_from[k]), r_um=float(r_c[k]), n_pairs=int(ok.sum()), n_pairs_excluded_flag2=int((ok_noflag & ~ok).sum()), s_between_slices=s_all, s_within_slice=s_w,
                   s_bio_slices=float(np.sqrt(max(s_all ** 2 - s_w ** 2, 0))), n_sections=n_sections(s_all, lo_all, hi_all, EPS_LIVER),
                   pearson_between_slices={f"{slices[i]}|{slices[j]}": float(np.corrcoef(lg[i], lg[j])[0, 1]) for i, j in combinations(range(n_sl), 2)},
                   mean_abs_log_g=float(np.median(np.abs(lg.mean(0)))))
        if within_pairs:
            var_w = np.mean([lg[p].var(0, ddof=1) for p in within_pairs], 0)
            rec["s_within_animal"] = float(np.median(np.sqrt(var_w)))
            rec["n_sections_within_animal"] = n_sections(rec["s_within_animal"], lo_w, hi_w, EPS_LIVER)
        if len(an_ids) >= 2:
            means = np.stack([lg[animals == a].mean(0) for a in an_ids])
            s_b = float(np.median(means.std(0, ddof=1)))
            rec["s_between_animals"] = s_b
            # the animal means carry within-animal noise var_w / n_slices_per_animal; the biological between-animal part is the remainder
            n_per = np.mean([(animals == a).sum() for a in an_ids])
            rec["s_bio_animals"] = float(np.sqrt(max(s_b ** 2 - (rec.get("s_within_animal", 0.0) ** 2) / n_per, 0)))
            rec["n_animals"] = n_sections(s_b, lo_b, hi_b, EPS_LIVER)
        rows.append(rec)
    out = dict(slices=slices, animals={k: LIVER[k] for k in slices}, n_genes=G, targets_eps=EPS_LIVER,
               chi2_factors=dict(between_slices=[lo_all, hi_all, nu_all], within_animal=[lo_w, hi_w, nu_w], between_animals=[lo_b, hi_b, nu_b]), rows=rows)
    json.dump(out, open(OUT_LIVER.with_suffix(".json"), "w"), indent=1)
    md = [f"# Replicate axis — MERFISH liver, {n_sl} Vizgen slices ({', '.join(slices)}; animals {[LIVER[k] for k in slices]}), top-200 genes of slice 1, pair medians", "",
          "s_slices = SD of log g over the slices (ν = %d), s_within = pooled SD between the two serial slices of one animal (ν = %d), s_animals = SD of the two animal means (ν = %d), "
          "s_w = within-slice tile-bootstrap SD. n(ε) = (s/ε)² sections for SEM ε with the 95 %% χ² interval of s propagated; Pearson = correlation of log g over pairs between slices (min / max over slice pairs)." % (nu_all, nu_w, nu_b), "",
          "| r [µm] | s_slices | s_within-animal | s_animals | s_w | Pearson | " + " | ".join(f"n slices SEM {e} [CI]" for e in EPS_LIVER[:2]) + " | " + " | ".join(f"n within-animal SEM {e}" for e in EPS_LIVER[:2]) + " | n animals SEM 0.01 [CI] |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for rec in rows:
        if "s_between_slices" not in rec or not any(abs(rec["r_um"] - t) / t < 0.15 for t in TARGETS_R_LIVER): continue
        pe = list(rec["pearson_between_slices"].values())
        md.append(f"| {rec['r_um']:.1f} | {rec['s_between_slices']:.4f} | {rec.get('s_within_animal', float('nan')):.4f} | {rec.get('s_between_animals', float('nan')):.4f} | {rec['s_within_slice']:.4f} | {min(pe):.3f}–{max(pe):.3f} | " +
                  " | ".join(f"{rec['n_sections'][str(e)]['point']:.1f} [{rec['n_sections'][str(e)]['ci'][0]:.1f}, {rec['n_sections'][str(e)]['ci'][1]:.0f}]" for e in EPS_LIVER[:2]) + " | " +
                  " | ".join(f"{rec['n_sections_within_animal'][str(e)]['point']:.1f}" if 'n_sections_within_animal' in rec else "—" for e in EPS_LIVER[:2]) +
                  (f" | {rec['n_animals']['0.01']['point']:.1f} [{rec['n_animals']['0.01']['ci'][0]:.1f}, {rec['n_animals']['0.01']['ci'][1]:.0f}] |" if "n_animals" in rec else " | — |"))
    OUT_LIVER.with_suffix(".md").write_text("\n".join(md)); print("\n".join(md))


def main():
    d = json.load(open(SRC))["between_sections"]
    out = dict(source=str(SRC.relative_to(REPO)), targets_eps=EPS, platforms={}, note=__doc__.strip())
    md = ["# Replicate axis — sections needed for a target SEM of log g (CRC P1/P2/P5; pair-median SDs)", "",
          "s_obs = between-patient SD over 3 sections (includes measurement noise), s_w = within-section tile-bootstrap SD, s_bio = sqrt(s_obs² − s_w²). "
          "n(ε) = (s_obs/ε)² sections for SEM ε; [lo, hi] = 95 % χ² interval of s_obs with ν = 2 propagated to n.", ""]
    for plat, v in d.items():
        n_sec = len(v["sections"]); nu = n_sec - 1
        lo_f, hi_f = np.sqrt(nu / chi2.ppf(0.975, nu)), np.sqrt(nu / chi2.ppf(0.025, nu))
        rows = []
        md += [f"## {plat} ({', '.join(v['sections'])})", "", "| r_from [µm] | s_obs | s_w | s_bio | ratio | Pearson between patients | " +
               " | ".join(f"n for SEM {e} [95 % CI]" for e in EPS) + " |", "|---|---|---|---|---|---|" + "---|" * len(EPS)]
        for row in v["rows"]:
            r = row["r_from"]
            s, sw = row["median_between_section_sd"], row["median_within_section_sd"]
            sbio = float(np.sqrt(max(s ** 2 - sw ** 2, 0)))
            rec = dict(r_from=r, s_obs=s, s_within=sw, s_bio=sbio, ratio=row["ratio_between_within"], pearson_between=row["pearson_between_sections"],
                       n_sections={f"{e}": dict(point=(s / e) ** 2, ci=[(s * lo_f / e) ** 2, (s * hi_f / e) ** 2]) for e in EPS})
            rows.append(rec)
            if any(abs(r - t) < 1e-6 for t in TARGETS_R):
                md.append(f"| {r:g} | {s:.3f} | {sw:.3f} | {sbio:.3f} | {s/sw:.1f} | {', '.join(f'{p:.2f}' for p in row['pearson_between_sections'])} | " +
                          " | ".join(f"{rec['n_sections'][str(e)]['point']:.1f} [{rec['n_sections'][str(e)]['ci'][0]:.1f}, {rec['n_sections'][str(e)]['ci'][1]:.0f}]" for e in EPS) + " |")
        md.append("")
        out["platforms"][plat] = dict(sections=v["sections"], chi2_factors=[float(lo_f), float(hi_f)], rows=rows)
    json.dump(out, open(OUT.with_suffix(".json"), "w"), indent=1)
    OUT.with_suffix(".md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    (liver if (len(sys.argv) > 1 and sys.argv[1] == "liver") else main)()
