#!/usr/bin/env python
"""Preprint Fig. 4c: molecules vs area. Runs of `python -m scote twopoint` on CRC P2 Xenium (408 common genes, 2 µm
lattice, 500 µm tiles, 200 tile-bootstrap resamples) with a seeded random subsample of a fraction m of the molecules at fixed
area (`--max-molecules`; data/twopoint/error_scaling/xenium_crc_p2_bin2um_common_mol{m}.npz) vs the full run
(data/twopoint/xenium_crc_p2_bin2um_common.npz). For the 200 most abundant genes and every r bin: median over pairs of the
tile-bootstrap SD of log g, σ_mol(m); per pair the decomposition σ² = A/m² + B (shot noise of the pair counts, which fall as m²,
plus a floor B from the finite number of tiles = structure sampling at fixed area); log-log slope β of σ vs m. Comparison with
the area route σ_area(f) (results/error_scaling/xenium_crc_p2_top200.json) at equal pair count: area fraction f gives f of the pairs,
molecule fraction m gives m² of them. Output results/error_scaling/xenium_crc_p2_molecules.{json,md}, quick-look figures/quicklook/molecules_vs_area.png.
`--dataset x5k_lymph_node` / `vizgen_liver1slice1` run the same analysis on the Xenium 5K lymph node and the MERFISH
liver slice 1 (top-200 runs; subsample runs data/twopoint/error_scaling/<dataset>_bin2um_top200_mol{m}.npz; whichever fractions exist are used).
SDs are the closed-form tile block bootstrap SDs stored in the npz (symmetrized); pairs without an interval (reliability
flag 2, T_eff < 5) in any of the runs are left out of the pair set (n_pairs_excluded_flag2 per bin).
"""
import argparse, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SUB = REPO / "data/twopoint/error_scaling"
FRACS = [0.05, 0.1, 0.25, 0.5]
N_TOP = 200
DATASETS = {
    "xenium_crc_p2": dict(full=REPO / "data/twopoint/xenium_crc_p2_bin2um_common.npz", sub="xenium_crc_p2_bin2um_common_mol{m:g}.npz",
                          area=REPO / "results/error_scaling/xenium_crc_p2_top200.json", out=REPO / "results/error_scaling/xenium_crc_p2_molecules",
                          fig=REPO / "figures/quicklook/molecules_vs_area.png", title="CRC P2 Xenium, 200 most abundant of the 408 common genes"),
    "x5k_lymph_node": dict(full=REPO / "data/twopoint/x5k_lymph_node_bin2um_top200.npz", sub="x5k_lymph_node_bin2um_top200_mol{m:g}.npz",
                           area=REPO / "results/error_scaling/x5k_lymph_node_top200.json", out=REPO / "results/error_scaling/x5k_lymph_node_molecules",
                           fig=REPO / "figures/quicklook/molecules_vs_area_x5k.png", title="Xenium 5K lymph node, 200 most abundant of 4,624 genes"),
    "vizgen_liver1slice1": dict(full=REPO / "data/twopoint/error_scaling/vizgen_liver1slice1_bin2um_top200.npz", sub="vizgen_liver1slice1_bin2um_top200_mol{m:g}.npz",
                                area=REPO / "results/error_scaling/vizgen_liver1slice1_top200.json", out=REPO / "results/error_scaling/vizgen_liver1slice1_molecules",
                                fig=REPO / "figures/quicklook/molecules_vs_area_liver.png", title="MERFISH liver slice 1, 200 most abundant of 347 genes"),
}


def load_sd(path, genes=None):
    d = np.load(path, allow_pickle=True)
    feats = list(map(str, d["features"])); sd = d["sd"].astype(np.float64); tot = d["totals"]
    if "flag" not in d.files:
        raise SystemExit(f"{path} has no 'flag' key: re-run it with python -m scote twopoint")
    flag = d["flag"]
    if genes is None:
        top = np.argsort(-tot)[:N_TOP]; top.sort(); genes = [feats[i] for i in top]
    idx = [feats.index(g) for g in genes]
    return dict(genes=genes, sd=sd[:, idx][:, :, idx], flag=flag[:, idx][:, :, idx], r=d["r_central"], r_from=d["r_from"], n_units=float(tot.sum()), n_tiles=int(d["totals_tiles"].shape[0]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="xenium_crc_p2", choices=sorted(DATASETS)); a = ap.parse_args()
    cfg = DATASETS[a.dataset]; FULL, AREA, OUT = cfg["full"], cfg["area"], cfg["out"]
    full = load_sd(FULL); genes = full["genes"]
    subs = {m: load_sd(SUB / cfg["sub"].format(m=m), genes) for m in FRACS if (SUB / cfg["sub"].format(m=m)).exists()}
    if not subs:
        print("no molecule-subsample runs yet"); return
    print(f"{a.dataset}: full {full['n_units']:.4g} molecules, {full['n_tiles']} tiles; subsamples m = {sorted(subs)}")
    ms = sorted(subs) + [1.0]; iu = np.triu_indices(len(genes))
    S = np.stack([(subs[m] if m < 1 else full)["sd"][:, iu[0], iu[1]] for m in ms])   # (n_m, K, n_pairs)
    FL = np.stack([(subs[m] if m < 1 else full)["flag"][:, iu[0], iu[1]] for m in ms]).max(0)   # (K, n_pairs): worst flag over the runs
    K = S.shape[1]; lm = np.log(ms); lm_c = lm - lm.mean()
    area = json.load(open(AREA)) if AREA.exists() else None
    rows = []
    for k in range(K):
        with np.errstate(all="ignore"):
            ls = np.log(S[:, k])
        ok_noflag = np.isfinite(ls).all(0) & (S[-1, k] < 0.5) & (S[-1, k] > 0)
        ok = ok_noflag & (FL[k] < 2)
        n_x = int((ok_noflag & ~ok).sum())
        if ok.sum() < 20:
            rows.append(dict(r_um=float(full["r"][k]), r_from=float(full["r_from"][k]), n_pairs=int(ok.sum()), n_pairs_excluded_flag2=n_x)); continue
        beta = -(lm_c @ (ls[:, ok] - ls[:, ok].mean(0))) / (lm_c ** 2).sum()             # sigma ∝ m^-beta per pair
        # per pair: sigma^2 = A / m^2 + B (least squares in x = 1/m^2, clipped at 0)
        x = 1 / np.array(ms) ** 2; xc = x - x.mean(); y = S[:, k, ok] ** 2
        A = (xc @ (y - y.mean(0))) / (xc ** 2).sum(); B = y.mean(0) - A * x.mean()
        A = np.clip(A, 0, None); B = np.clip(B, 0, None)
        floor_frac = B / np.maximum(A + B, 1e-300)                                          # share of the full-data variance that does not shrink with molecules
        rec = dict(r_um=float(full["r"][k]), r_from=float(full["r_from"][k]), n_pairs=int(ok.sum()), n_pairs_excluded_flag2=n_x,
                   sigma_median_by_m={f"{m:g}": float(np.median(S[i, k, ok])) for i, m in enumerate(ms)},
                   beta_median=float(np.median(beta)), beta_iqr=[float(np.percentile(beta, 25)), float(np.percentile(beta, 75))],
                   floor_fraction_median=float(np.median(floor_frac)), floor_fraction_iqr=[float(np.percentile(floor_frac, 25)), float(np.percentile(floor_frac, 75))])
        if area is not None:
            ar = [r for r in area["rows"] if abs(r["r_from"] - rec["r_from"]) < 1e-6 and "sigma_median_by_f" in r]
            if ar: rec["sigma_area_median_by_f"] = ar[0]["sigma_median_by_f"]; rec["alpha_area"] = ar[0]["alpha_median"]
        rows.append(rec)
    out = dict(full=str(FULL.relative_to(REPO)), uncertainty="closed", n_genes=len(genes), molecule_fractions=ms, n_molecules_full=full["n_units"],
               n_molecules_by_m={f"{m:g}": subs[m]["n_units"] for m in subs}, n_tiles=full["n_tiles"], rows=rows)
    json.dump(out, open(OUT.with_suffix(".json"), "w"), indent=1)
    md = [f"# Molecules vs area — {cfg['title']}, 2 µm lattice, {full['n_tiles']} tiles of 500 µm ({full['n_units']:.4g} molecules)", "",
          "σ_mol(m): median tile-bootstrap SD of log g with a fraction m of the molecules at the full area; β: σ ∝ m^−β (median over pairs); floor: share B/(A+B) of the "
          "full-data variance that does not shrink with more molecules (σ² = A/m² + B); σ_area(f): area subsampling at the full molecule density (f of the pairs; m gives m² of the pairs).", "",
          "| r [µm] | " + " | ".join(f"σ_mol({m:g})" for m in ms) + " | β | floor B/(A+B) | σ_area(0.05) | σ_area(0.25) | α_area | σ_mol(0.5) vs σ_area(0.25) (same pair count) |", "|---|" + "---|" * (len(ms) + 6)]
    for rec in rows:
        if "beta_median" not in rec: continue
        sa = rec.get("sigma_area_median_by_f", {})
        cmp = f"{rec['sigma_median_by_m']['0.5']:.3f} vs {sa['0.25']:.3f}" if "0.5" in rec["sigma_median_by_m"] and "0.25" in sa else "—"
        md.append(f"| {rec['r_um']:.1f} | " + " | ".join(f"{rec['sigma_median_by_m'][f'{m:g}']:.3f}" for m in ms) +
                  f" | {rec['beta_median']:.2f} [{rec['beta_iqr'][0]:.2f}, {rec['beta_iqr'][1]:.2f}] | {rec['floor_fraction_median']:.2f} [{rec['floor_fraction_iqr'][0]:.2f}, {rec['floor_fraction_iqr'][1]:.2f}] | "
                  f"{sa.get('0.05', float('nan')):.3f} | {sa.get('0.25', float('nan')):.3f} | {rec.get('alpha_area', float('nan')):.2f} | {cmp} |")
    OUT.with_suffix(".md").write_text("\n".join(md)); print("\n".join(md))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    ks = [i for i, rec in enumerate(rows) if "beta_median" in rec]
    for i in ks[::max(1, len(ks) // 6)]:
        rec = rows[i]; axs[0].plot(ms, [rec["sigma_median_by_m"][f"{m:g}"] for m in ms], "o-", ms=3, label=f"r ≈ {rec['r_um']:.0f} µm")
    axs[0].plot(ms, rows[ks[0]]["sigma_median_by_m"]["1"] / np.array(ms), "k:", lw=0.8, label="∝ 1/m (shot noise)"); axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel("molecule fraction m (full area)"); axs[0].set_ylabel("median σ(log g)"); axs[0].legend(fontsize=7, frameon=False)
    axs[1].plot([rows[i]["r_um"] for i in ks], [rows[i]["beta_median"] for i in ks], "o-", ms=3, label="β (σ ∝ m^−β)")
    axs[1].plot([rows[i]["r_um"] for i in ks], [rows[i]["floor_fraction_median"] for i in ks], "s-", ms=3, label="floor share B/(A+B)")
    if area is not None: axs[1].plot([rows[i]["r_um"] for i in ks if "alpha_area" in rows[i]], [rows[i]["alpha_area"] for i in ks if "alpha_area" in rows[i]], "^-", ms=3, label="α (σ ∝ f^−α, area)")
    axs[1].axhline(1, c="k", lw=0.5, ls="--"); axs[1].axhline(0.5, c="k", lw=0.5, ls=":"); axs[1].set_xscale("log"); axs[1].set_xlabel("r [µm]"); axs[1].legend(fontsize=7, frameon=False)
    for i in ks[::max(1, len(ks) // 3)]:
        rec = rows[i]
        l, = axs[2].plot(np.array(ms) ** 2, [rec["sigma_median_by_m"][f"{m:g}"] for m in ms], "o-", ms=3, label=f"molecules, r ≈ {rec['r_um']:.0f} µm")
        if "sigma_area_median_by_f" in rec:
            fs = sorted(float(f) for f in rec["sigma_area_median_by_f"]); axs[2].plot(fs, [rec["sigma_area_median_by_f"][f"{f:g}"] for f in fs], "s--", ms=3, color=l.get_color(), label=f"area, r ≈ {rec['r_um']:.0f} µm")
    axs[2].set_xscale("log"); axs[2].set_yscale("log"); axs[2].set_xlabel("fraction of the pairs (m² for molecules, f for area)"); axs[2].set_ylabel("median σ(log g)"); axs[2].legend(fontsize=6, frameon=False)
    fig.suptitle(f"Molecules vs area ({cfg['title']})", fontsize=9); fig.tight_layout(); fig.savefig(cfg["fig"], dpi=140)


if __name__ == "__main__":
    main()
