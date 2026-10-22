#!/usr/bin/env python
"""Preprint Supplementary Fig. S2: the second platform set, SpatialBench spleen (WT709 on Xenium /
Visium HD / MERSCOPE, KO168 on Xenium / HD; 90 common genes, all three platforms on the same 2 µm lattice). Same structure and
scatter style as figures/make_fig2.py:
  (a) log g_ab(r) with 95 % tile-bootstrap bands for three gene pairs (Cd3d-Cd3e, Adgre1-Cd19, Cd19-Cd3e) on the three
      platforms, from the 2 µm lattice pitch (first bin [2, 4) µm) to 3 mm. All spleen runs are 2 µm-lattice runs (no
      molecule-level run exists), so there is no binned duplicate of the Xenium series here;
  (b) method-method scatter of log g over all 4,095 gene pairs at r in [10, 12), [96, 112), [896, 1024) µm: Visium HD vs Xenium
      (top row) and MERSCOPE vs Xenium (bottom row); color = relative local point density, Pearson r in the lower right corner,
      line fit (errors-in-variables slope) drawn only where its 95 % interval excludes zero (make_fig2.fit_determined; not at
      1 mm, where the interval spans zero); statistics in results/platforms/scatter_stats.json (scripts/scatter_stats.py);
  (c) cross-platform agreement per animal against the between-animal agreement of one platform
      (results/platforms/spleen_replicates.json), on every 2 µm lattice bin from the intra-bin shell [0, 2) µm on,
      plotted at the lattice-shell distance r_central (1 µm for the intra-bin shell); legends below the panel.
Writes results/platforms/figS2_spleen_pearson.json (per scatter panel, as fig2_pearson.json). `--print` prints the numbers
quoted in the caption.
"""
import argparse, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_fig2 as F  # noqa: E402  (loaders, scatter panel, level panel, style; same directory)

REPO = F.REPO; DATA = F.DATA; RES = F.RES
COL = {"Xenium": F.PLATFORM["Xenium"], "Visium HD": F.PLATFORM["Visium HD"], "MERSCOPE": F.PLATFORM["MERSCOPE"],   # platform palette as in make_fig2.py
       "xenium": F.PLATFORM["Xenium"], "hd": F.PLATFORM["Visium HD"]}
LIM = (-2.0, 2.0)   # common axis range of the six scatter panels (90-gene panel: log g spans less than on CRC; points outside: --print)
RUN = {"Xenium": "sb_xenium_WT709_rep1_bin2um_common.npz", "Visium HD": "sb_hd_WT709_bin2um_common.npz", "MERSCOPE": "sb_merscope_WT709_bin2um_common.npz"}


def panel_replicates(ax):
    """Legends split: color = what is compared (black / gray, Xenium against another platform; platform color, one platform
    between animals), line style = animal (solid WT709, dotted KO168, dashed WT709 vs KO168); both below the panel."""
    from matplotlib.lines import Line2D
    d = json.load(open(RES / "spleen_replicates.json"))
    styles = {"xenium_WT709|hd_WT709": (F.PLATFORM["cross"], "-"), "xenium_KO168|hd_KO168": (F.PLATFORM["cross"], ":"),
              "xenium_WT709|merscope_WT709": ("#999999", "-")}
    for k, rows in d["per_animal"].items():
        if k not in styles or not rows: continue
        c, ls = styles[k]
        r = np.array([b["r_central"] for b in rows]); p = np.array([b["pearson"] for b in rows]); m = r <= 1100
        ax.plot(r[m], p[m], ls=ls, color=c, lw=1.0)
    for plat in ("xenium", "hd"):
        v = d["between_animals"].get(plat)
        if not v or not v["rows"]: continue
        r = np.array([b["r_central"] for b in v["rows"]]); pb = np.array([b["pearson_between_sections"][0] for b in v["rows"]]); m = r <= 1100
        ax.plot(r[m], pb[m], "--", color=COL[plat], lw=1.0)
    ax.set_xscale("log"); ax.set_xlim(0.8, 1200); ax.set_ylim(0.3, 1.0); ax.set_yticks([0.4, 0.6, 0.8, 1.0]); ax.set_xticks([1, 10, 100, 1000])
    ax.set_xlabel("r [µm]", labelpad=1); ax.set_ylabel("Pearson r of log g\n(all gene pairs)", labelpad=1)
    hc = [Line2D([], [], color=F.PLATFORM["cross"], lw=1.2, label="Xenium vs Visium HD"), Line2D([], [], color="#999999", lw=1.2, label="Xenium vs MERSCOPE"),
          Line2D([], [], color=COL["xenium"], lw=1.2, label="Xenium"), Line2D([], [], color=COL["hd"], lw=1.2, label="Visium HD")]
    hs = [Line2D([], [], color="0.3", lw=1.0, ls="-", label="WT709"), Line2D([], [], color="0.3", lw=1.0, ls=":", label="KO168"),
          Line2D([], [], color="0.3", lw=1.0, ls="--", label="WT709 vs KO168")]
    l1 = F.legend_below(ax, hc, y=-0.27); ax.add_artist(l1)
    F.legend_below(ax, hs, y=-0.70)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--print", action="store_true", dest="prt"); a = ap.parse_args()
    F.require(*(DATA / v for v in RUN.values()))
    runs = {k: F.load_run(DATA / v, 2.0) for k, v in RUN.items()}
    F.OKABE.update(COL)
    stats = json.load(open(RES / "scatter_stats.json"))["comparisons"]
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.15], hspace=0.32, left=0.075, right=0.985, top=0.96, bottom=0.07)
    gs_a = gs[0].subgridspec(1, 3, wspace=0.3); gs_b = gs[1].subgridspec(2, 4, width_ratios=[1, 1, 1, 1.1], wspace=0.6, hspace=0.42)
    axa = [fig.add_subplot(gs_a[0, i]) for i in range(3)]
    F.panel_a(axa, [("Cd3d", "Cd3e"), ("Adgre1", "Cd19"), ("Cd19", "Cd3e")], runs)
    for ax in axa:
        ax.set_xlim(1.5, F.R_MAX * 1.15)
    rec = []
    for row, (B, key) in enumerate((("Visium HD", "spleen_xenium_hd"), ("MERSCOPE", "spleen_xenium_merscope"))):
        axs = [fig.add_subplot(gs_b[row, i]) for i in range(3)]
        for ax, (s, bins) in zip(axs, F.SCALES.items()):
            st = stats[key]["rows"][s]
            n, out, sc, info = F.scatter_panel(ax, runs["Xenium"], runs[B], bins, bins, st=st, lim=LIM)
            rec.append(F.scatter_record(key, s, bins, bins, info) | dict(panel=f"b, row {row + 1}, column {len(rec) % 3 + 1}", A="Xenium", B=B))
            ax.set_title(f"{bins[0][0]}–{bins[0][1]} µm", pad=3); ax.set_xlabel("log g, Xenium", labelpad=1)
            if a.prt:
                print(f"{key} {s} um: n={st['n_pairs']} (plotted {n}, outside {out}); Pearson {st['pearson']:.3f} [{st['pearson_ci'][0]:.3f}, {st['pearson_ci'][1]:.3f}]; "
                      f"line fit {st['deming']:.3f} [{st['deming_ci'][0]:.3f}, {st['deming_ci'][1]:.3f}] (drawn: {info['line_fit_drawn']}); OLS {st['ols']:.3f}; RMS {st['rms']:.3f}")
        axs[0].set_ylabel(f"log g, {B}", labelpad=1)
        if row == 0:
            F.label(axs[0], "b", dx=-28)
    axc = fig.add_subplot(gs_b[0, 3])
    panel_replicates(axc)
    F.label(axa[0], "a", dx=-28); F.label(axc, "c", dx=-28)
    json.dump(dict(figure="preprint Supplementary Fig. S2 (figures/figS2_spleen_platform_invariance), panel b", script="figures/make_figS2.py",
                   note="pearson = point estimate of scatter_stats.json (quoted in the panel, two decimals); pearson_plotted_points = "
                        "Pearson r of the plotted points (check); line_fit_drawn = false where the 95 % interval of the slope includes zero",
                   panels=rec), open(RES / "figS2_spleen_pearson.json", "w"), indent=1)
    out = REPO / "figures/figS2_spleen_platform_invariance"
    fig.savefig(str(out) + ".pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(str(out) + ".png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    print("wrote", out)


if __name__ == "__main__":
    main()
