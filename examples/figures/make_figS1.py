#!/usr/bin/env python
"""Preprint Supplementary Fig. S1 (companion of Fig. 2b): method-method scatter of log g over all
gene pairs for the combinations with Visium v2 (55 µm spots), CRC P2, at '100 µm' and 1 mm. Writes figures/figS1_v2_scatter.{pdf,png}.
  (a) Visium v2 vs Xenium, (b) Visium v2 vs Visium HD; left r ≈ 100 µm, right r = 896–1024 µm.
The spot run counts pairs on a 64 µm lattice (2^5 x 2 µm, each spot in its own lattice cell; workflows/twopoint.smk,
rule twopoint_visium_spots). At '100 µm' the spots are compared on the lattice shell [64, 128) µm, which holds the nearest-neighbor
spots, against the sum of the four 2 µm-lattice bins [64, 80) ... [112, 128) of the other platform (scripts/scatter_stats.py);
from 256 µm on the spot shells coincide with the 2 µm hierarchy. Same axes and style as Fig. 2b (make_fig2.scatter_panel):
identity line dashed, line fit solid, color = relative local point density, Pearson r in the lower right corner. Statistics in
results/platforms/scatter_stats.json; per-panel values in results/platforms/figS1_v2_pearson.json; `--print` prints them.
"""
import argparse, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_fig2 as F  # noqa: E402  (loaders, scatter panel, style; same directory)

V2_BINS = {"100": ([(64, 80), (80, 96), (96, 112), (112, 128)], [(64, 128)]), "1000": ([(896, 1024)], [(896, 1024)])}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--print", action="store_true", dest="prt"); a = ap.parse_args()
    F.require(F.DATA / "xenium_crc_p2_mol_common.npz", F.DATA / "hd_crc_p2_bin2um_common.npz", F.DATA / "v2_crc_p2_spots_common.npz")
    runs = {"Xenium": F.load_run(F.DATA / "xenium_crc_p2_mol_common.npz", 0.5), "Visium HD": F.load_run(F.DATA / "hd_crc_p2_bin2um_common.npz", 2.0),
            "Visium v2": F.load_run(F.DATA / "v2_crc_p2_spots_common.npz", 64.0)}
    stats = json.load(open(F.RES / "scatter_stats.json"))["comparisons"]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.2), gridspec_kw=dict(wspace=0.42, left=0.07, right=0.99, top=0.86, bottom=0.17))
    rec = []
    for j, (A, key) in enumerate((("Xenium", "crc_xenium_v2"), ("Visium HD", "crc_hd_v2"))):
        for i, s in enumerate(("100", "1000")):
            ax = axs[2 * j + i]; st = stats[key]["rows"][s]; bA, bB = V2_BINS[s]
            n, out, sc, info = F.scatter_panel(ax, runs[A], runs["Visium v2"], bA, bB, st=st)
            rec.append(F.scatter_record(key, s, bA, bB, info) | dict(panel=f"{'ab'[j]}, {'left' if i == 0 else 'right'}", A=A, B="Visium v2"))
            ax.set_title("64–128 µm" if s == "100" else "896–1024 µm", pad=3); ax.set_xlabel(f"log g, {A}", labelpad=1)
            if i == 0:
                ax.set_ylabel("log g, Visium v2", labelpad=1); F.label(ax, "ab"[j], dx=-26)
            if a.prt:
                print(f"{key} {s} um: n={st['n_pairs']} (plotted {n}, outside {out}); Pearson {st['pearson']:.3f} [{st['pearson_ci'][0]:.3f}, {st['pearson_ci'][1]:.3f}]; "
                      f"line fit {st['deming']:.3f} [{st['deming_ci'][0]:.3f}, {st['deming_ci'][1]:.3f}]; OLS {st['ols']:.3f}; RMS {st['rms']:.3f}; mean diff {st['mean_diff']:+.3f}")
    json.dump(dict(figure="preprint Supplementary Fig. S1 (figures/figS1_v2_scatter)", script="figures/make_figS1.py",
                   note="pearson = point estimate of scatter_stats.json (quoted in the panel, two decimals); pearson_plotted_points = "
                        "Pearson r of the plotted points (check)", panels=rec), open(F.RES / "figS1_v2_pearson.json", "w"), indent=1)
    out = F.REPO / "figures/figS1_v2_scatter"
    fig.savefig(str(out) + ".pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(str(out) + ".png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    print("wrote", out)


if __name__ == "__main__":
    main()
