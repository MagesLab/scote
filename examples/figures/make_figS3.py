#!/usr/bin/env python
"""Preprint Supplementary Fig. S3: sequencing-depth ablation on the Visium HD probe library of the FF ovarian section.
Reads results/ovff/depth/{power_<depth>.json, scaling.json} (scripts/ovff_depth.py) and writes
figures/figS3_depth.{pdf,png}.

Panels: (a) pair-median tile-bootstrap SD of log g vs r at full area for the deep run, the real min-depth run and the deep run
thinned to the min-depth UMI total and to 1/8; (b) SD vs UMI fraction at three scales (thinned series; stars = the real
min-depth run; dotted = same UMI budget spent on area instead). The fitted depth exponents beta of (b) are printed here and
quoted in the caption (scaling.json, rows[*].beta_thinned), not drawn in the panel. The area-scaling exponents and the bias
with depth are reported in the Methods only.
All depths are the same library on the same section, so every tile is the same tissue at every depth (79 tissue tiles of
500 um).
"""
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter

REPO = Path(__file__).resolve().parents[1]
COL = {"deep": "#0072B2", "min": "#D55E00", "thin0.335": "#E69F00", "thin0.5": "#56B4E9", "thin0.25": "#009E73", "thin0.125": "#CC79A7"}
LAB = {"deep": "deep (802 M UMI)", "min": "minimum depth, real (269 M UMI)", "thin0.335": "deep thinned to 0.335", "thin0.125": "deep thinned to 0.125"}
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results-dir", default=str(REPO / "results/ovff/depth")); ap.add_argument("--out-dir", default=str(REPO / "figures"))
    a = ap.parse_args(); R = Path(a.results_dir)
    sc = json.load(open(R / "scaling.json"))
    P = {d: json.load(open(R / f"power_{d}.json")) for d in COL if (R / f"power_{d}.json").exists()}
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))

    # (a) error versus r at full area
    for d in ("deep", "min", "thin0.335", "thin0.125"):
        if d not in P: continue
        rr = [r for r in P[d]["rows"] if "alpha_median" in r and r["r_um"] <= 3000]
        axs[0].plot([r["r_um"] for r in rr], [r["sigma_median_by_f"]["1"] for r in rr],
                    "-" if d in ("deep", "min") else "--", color=COL[d], lw=1.0, label=LAB[d])
    axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlim(3, 3000); axs[0].set_ylim(0.004, 0.03)
    axs[0].set_xlabel("r [µm]", labelpad=1); axs[0].set_ylabel("median σ(log g), full area")
    axs[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.21), ncol=2, handlelength=2.2, columnspacing=1.2, borderaxespad=0)

    # (b) error versus UMI or area budget
    thinned = ["thin0.125", "thin0.25", "thin0.335", "thin0.5", "deep"]; ms = [sc["depths"][d]["umi_fraction"] for d in thinned if d in sc["depths"]]
    marks = ["o", "s", "^"]; scale_handles = []
    for m, rf in enumerate((4.0, 20.0, 96.0)):
        rows = [r for r in sc["rows"] if r["r_from"] == rf and "beta_thinned" in r]
        if not rows: continue
        r = rows[0]; s = r["sigma_full_area"]
        print(f"(b) {r['r_um']:.1f} um: beta (thinned, least squares over five depths) = {r['beta_thinned']:.3f}; "
              f"sigma deep {s['deep']:.5f}, min real {s['min']:.5f}, thinned 0.335 {s['thin0.335']:.5f}")
        axs[1].plot(ms, [s[d] for d in thinned if d in s], "-", marker=marks[m], ms=3, color="0.3", lw=0.8)
        axs[1].plot([sc["depths"]["min"]["umi_fraction"]], [s["min"]], marker="*", ms=7, color=COL["min"], ls="none")
        fr = P["deep"]["rows"][r["k"]]["sigma_median_by_f"]
        axs[1].plot([0.335, 0.5, 1.0], [fr["0.335"], fr["0.5"], fr["1"]], ":", marker=marks[m], ms=3, color=COL["deep"], lw=0.8, mfc="none")
        scale_handles.append(Line2D([], [], color="0.3", marker=marks[m], ms=3, ls="none", label=f"r = {r['r_um']:.0f} µm"))
    style_handles = [Line2D([], [], color="0.3", lw=0.8, ls="-", label="deep run thinned, UMI fraction"),
                     Line2D([], [], color=COL["deep"], lw=0.8, ls=":", label="deep run, area fraction"),
                     Line2D([], [], color=COL["min"], marker="*", ms=7, ls="none", label="minimum depth, real")]
    axs[1].set_xscale("log"); axs[1].set_yscale("log"); axs[1].set_xlabel("UMI or area fraction", labelpad=1); axs[1].set_ylabel("median σ(log g)")
    axs[1].set_xticks([0.125, 0.25, 0.5, 1.0]); axs[1].set_xticklabels(["1/8", "1/4", "1/2", "1"]); axs[1].minorticks_off()
    axs[1].legend(handles=scale_handles + style_handles, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.21), ncol=2,
                  handlelength=2.2, columnspacing=1.2, borderaxespad=0)

    for ax in axs:
        ax.yaxis.set_major_locator(FixedLocator([0.005, 0.01, 0.02, 0.05])); ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    for j, ax in enumerate(axs):
        ax.text(-0.16, 1.02, "ab"[j], transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.94, bottom=0.36, wspace=0.28)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figS3_depth.pdf"); fig.savefig(out / "figS3_depth.png", dpi=600)
    print("wrote", out / "figS3_depth.{pdf,png}")


if __name__ == "__main__":
    main()
