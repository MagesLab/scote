#!/usr/bin/env python
"""Quick-look of the platform comparison: overlay log g_ab(r) (random-label reference, tile-bootstrap 95 % bands) of the same
gene pairs across platforms (Xenium molecule level, Xenium binned to 2 um, Visium HD 2 um bins, Visium 55 um spots),
plus a summary JSON with pairwise agreement statistics over all common gene pairs on the matched [from, to) bins with
r_from >= --r-min (default 3 x the larger bin scale of the two runs; --r-min 0 keeps every matched bin down to the intra-bin
shell). Used by scripts/crc_replicates.py."""
import argparse, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def sd_filter(d):
    """Stored closed-form SD for the pair filters (SD < 0.5 and the like), with the pairs without an interval (reliability
    flag 2, T_eff < 5) set to +inf so that every SD filter excludes them."""
    sd = d["sd"]
    return np.where(d["flag"] == 2, np.inf, sd) if "flag" in d.files else sd


ap = argparse.ArgumentParser()
ap.add_argument("--xenium-mol"); ap.add_argument("--xenium-bin"); ap.add_argument("--hd"); ap.add_argument("--visium")
ap.add_argument("--pairs", default="CEACAM5:CEACAM5,CD3E:CD3E,CEACAM5:CD3E,ACTA2:ACTA2,MS4A1:CD3E,CEACAM5:ACTA2,PECAM1:VWF,MKI67:LGR5")
ap.add_argument("--out", required=True); ap.add_argument("--summary", required=True)
ap.add_argument("--r-min", type=float, default=None, help="lower bound on r_from [um] of the compared bins (default: 3 x the coarser resolution)")
a = ap.parse_args()

runs = {}
for name, path, res in [("Xenium molecules", a.xenium_mol, 0.5), ("Xenium 2 µm bins", a.xenium_bin, 2.0),
                        ("Visium HD 2 µm", a.hd, 2.0), ("Visium 55 µm", a.visium, 55.0)]:
    if path:
        d = np.load(path, allow_pickle=True)
        runs[name] = dict(feats=list(map(str, d["features"])), r=d["r_central"], lg=d["log_g"], lo=d["lo"], hi=d["hi"], sd=sd_filter(d), res=res,
                          r_from=d["r_from"], r_to=d["r_to"])
common = sorted(set.intersection(*[set(v["feats"]) for v in runs.values()]))
pairs = [p.split(":") for p in a.pairs.split(",") if all(g in common for g in p.split(":"))]
cols = {"Xenium molecules": "C0", "Xenium 2 µm bins": "C1", "Visium HD 2 µm": "C2", "Visium 55 µm": "C3"}
n = len(pairs); nc = 4; nr = int(np.ceil(n / nc))
fig, axs = plt.subplots(nr, nc, figsize=(3.4 * nc, 2.9 * nr), sharex=True, squeeze=False)
for ax, (ga, gb) in zip(axs.ravel(), pairs):
    for name, v in runs.items():
        i, j = v["feats"].index(ga), v["feats"].index(gb)
        ok = (v["r"] >= 3 * v["res"]) & np.isfinite(v["lg"][:, i, j])
        ax.fill_between(v["r"][ok], v["lo"][ok, i, j], v["hi"][ok, i, j], alpha=0.2, color=cols[name])
        ax.plot(v["r"][ok], v["lg"][ok, i, j], color=cols[name], lw=1.2, label=name)
        faint = (v["r"] < 3 * v["res"]) & np.isfinite(v["lg"][:, i, j])
        ax.plot(v["r"][faint], v["lg"][faint, i, j], color=cols[name], lw=0.8, ls=":")
    ax.axhline(0, c="k", lw=0.5); ax.set_xscale("log"); ax.set_title(f"{ga} – {gb}", fontsize=9); ax.set_xlabel("r [µm]")
for ax in axs.ravel()[n:]: ax.axis("off")
axs[0, 0].set_ylabel("log g_ab(r)"); axs[0, 0].legend(fontsize=7, frameon=False)
fig.suptitle("Oliveira CRC P2 serial sections: normalized 2-pt functions across platforms (dotted = below 3× resolution)", fontsize=9)
fig.tight_layout(); fig.savefig(a.out, dpi=150, bbox_inches="tight")

# summary: pairwise agreement over all common gene pairs on shared r bins
def on_common_grid(v, feats):
    idx = [v["feats"].index(f) for f in feats]
    return v["lg"][:, idx][:, :, idx], v["sd"][:, idx][:, :, idx], v["r_from"], v["r_to"]
summ = dict(common_genes=len(common), pairs_plotted=pairs, comparisons=[])
names = list(runs)
for x in range(len(names)):
    for y in range(x + 1, len(names)):
        A, B = runs[names[x]], runs[names[y]]
        lgA, sdA, fA, tA = on_common_grid(A, common); lgB, sdB, fB, tB = on_common_grid(B, common)
        rmin = 3 * max(A["res"], B["res"]) if a.r_min is None else a.r_min
        # match bins by identical [from, to) intervals
        keyA = {(round(f, 6), round(t, 6)): i for i, (f, t) in enumerate(zip(fA, tA))}
        matched = [(keyA[(round(f, 6), round(t, 6))], j) for j, (f, t) in enumerate(zip(fB, tB)) if (round(f, 6), round(t, 6)) in keyA and f >= rmin]
        rows = []
        for i, j in matched:
            da, db = lgA[i], lgB[j]; sa, sb = sdA[i], sdB[j]
            ok = np.isfinite(da) & np.isfinite(db) & (sa < 0.5) & (sb < 0.5)
            iu = np.triu_indices(len(common))
            ok = ok[iu]; da = da[iu][ok]; db = db[iu][ok]; z = (da - db) / np.sqrt(sa[iu][ok] ** 2 + sb[iu][ok] ** 2 + 1e-12)
            if ok.sum() > 10:
                rows.append(dict(r_from=float(fA[i]), r_to=float(tA[i]), n_pairs=int(ok.sum()), pearson=float(np.corrcoef(da, db)[0, 1]),
                                 rms_diff=float(np.sqrt(np.mean((da - db) ** 2))), median_abs_z=float(np.median(np.abs(z))),
                                 frac_within_2sigma=float(np.mean(np.abs(z) < 2)), sd_A_median=float(np.median(sa[iu][ok])), sd_B_median=float(np.median(sb[iu][ok]))))
        summ["comparisons"].append(dict(A=names[x], B=names[y], r_min_um=rmin, n_matched_bins=len(matched), bins=rows))
json.dump(summ, open(a.summary, "w"), indent=1)
print(json.dumps({k: v for k, v in summ.items() if k != "comparisons"}), flush=True)
for c in summ["comparisons"]:
    print(c["A"], "vs", c["B"], f"r>={c['r_min_um']:g}", [(b["r_from"], round(b["pearson"], 3), round(b["rms_diff"], 3), round(b["frac_within_2sigma"], 2)) for b in c["bins"]])
print("wrote", a.out)
