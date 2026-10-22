#!/usr/bin/env python
"""Preprint Supplementary Fig. S4: what a two-point function measures at the sub-micrometer end: composition pedestal,
z-projection and resolution floor. Writes figures/figS4_subcellular.{pdf,png} (7.2 x 2.45 in, the style of make_fig5.py) and
results/cosmx_kidney/figS4_numbers.json (every number quoted in the caption).

Panels and sources (file -> field):
  (a) composition pedestal on the genes shared by the CosMx 6k kidney panel and AVITI24 (184 genes = kidney top-1000 and
      AVITI24 top-1000, results/cosmx_kidney/common_genes_matched_aviti24.txt), both platforms counted directly in 0.5 um
      level-0 bins (6 bins, 0-3 um):
        results/cosmx_kidney/crossplatform_pedestal_bin05_direct.json
          cells["cosmx|all" / "aviti24|all"].observed      -> stacked same-gene log g_aa (random-label reference), solid
          cells[...].pedestal_mean, pedestal_sd             -> within-cell label permutation (3 permutations), dashed
          read_point, delta_um, n_bins, n_genes
  (b) z-projection, AVITI24 A549, 50 most abundant genes, six wells, 0.48 um pixels:
        results/cosmx/aviti24_zprojection_correction.json
          modes["per-slice" / "projected"].r_coarse_um, same, diff  -> stacked same-gene and different-gene log g
          modes[...].bin0_same, fold_above_random                    -> first bin per plane and projected
          pitch_um                                                   -> shaded first pixel (0.48 um)
          n_wells, n_genes                                           -> "6 wells, 50 genes"
  (c) resolution floor, CosMx 6k kidney (GEO GSE282059, slide 2B_FFPE, tissue piece 2), first 16 FOVs, 0.05 um bins,
      150 most abundant genes:
        results/cosmx_kidney/resolution.json
          classes.endogenous.r_um, log_g_same, log_g_diff  -> curves (bins without same-gene pairs have log g = -inf and are
                                                              shaded, not drawn)
          usable floor = lower edge of the bin holding the maximum of log_g_same beyond the minimum
          centroid_precision_um is null for these flat files (no per-molecule spread), so no precision box is drawn.
The platform is named "AVITI24 (Cytos)": the AVITI24 instrument with Element's Cytos multiomics workflow.
In-panel text: none beyond titles, axis labels and legends. The +-1 SD band of the permutation mean is not drawn: the SD over
the three permutations is 0.001-0.010 in log g, below the line width at this axis scale; its range is quoted in the caption.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results/cosmx"             # AVITI24 z-projection (panel b)
RES_K = REPO / "results/cosmx_kidney"     # public CosMx kidney, GEO GSE282059 (panels a, c)
YMAX_RES = 1.9                            # panel c y-limit without the precision box (kidney curve peaks at +1.31 on the whole piece)
COL = {"cosmx": "#999933", "aviti24": "#882255", "same": "#D55E00", "diff": "#009E73", "floor": "#555555"}   # cosmx, aviti24: platform palette as in make_fig2.py (no Xenium / Visium colors)
from matplotlib.lines import Line2D
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "legend.fontsize": 6.2, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})


def missing(ax, name):
    ax.text(0.5, 0.5, f"{name} missing", transform=ax.transAxes, ha="center")


def panel_pedestal(ax, num):
    f = RES_K / "crossplatform_pedestal_bin05_direct.json"
    if not f.exists():
        return missing(ax, f.name + " (run scripts/pedestal_direct.py)")
    d = json.load(open(f)); YMAX = 2.0; dl = d["delta_um"]
    ax.axvspan(0.5, 1.0, color="0.9", lw=0, zorder=0)
    plat_h = []; num["a"] = dict(source=str(f.relative_to(REPO)), n_genes=d["n_genes"], n_perm=d["n_perm"], delta_um=dl, platforms={})
    for key, lab in (("aviti24", "AVITI24 (Cytos)"), ("cosmx", "CosMx")):
        c = d["cells"][key]; col = COL[key]
        r = (np.arange(c["n_bins"]) + 0.5) * dl
        obs, ped, psd = (np.array(c[k]) for k in ("observed", "pedestal_mean", "pedestal_sd"))
        ax.plot(r, obs, "o-", color=col, ms=2.2, lw=1.0, clip_on=True)
        ax.plot(r, ped, "s", ls=(0, (3, 1.5)), color=col, ms=1.8, lw=0.9)
        rp = c["read_point"]
        num["a"]["platforms"][key] = dict(n_molecules=c["n_molecules"], n_cells=c["n_cells"], observed_first_bin=float(obs[0]),
                                          read_point=rp, pedestal_sd_min=float(psd.min()), pedestal_sd_max=float(psd.max()),
                                          pedestal_sd_per_bin=[float(v) for v in psd])
        plat_h.append((Line2D([], [], color=col, lw=2.0), lab))
    sty_h = [(Line2D([], [], color="0.3", marker="o", ms=2.2, lw=1.0), "observed"),
             (Line2D([], [], color="0.3", marker="s", ms=1.8, lw=0.9, ls=(0, (3, 1.5))), "within-cell pedestal")]
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlim(0, 3.0); ax.set_ylim(-0.05, YMAX); ax.set_xlabel("r [µm]", labelpad=1); ax.set_ylabel("stacked same-gene log g_aa")
    l1 = ax.legend(*zip(*plat_h), frameon=False, loc="upper right", handlelength=1.4, labelspacing=0.2, borderaxespad=0.1)
    ax.add_artist(l1)
    ax.legend(*zip(*sty_h), frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.80), handlelength=2.8, labelspacing=0.2, borderaxespad=0.1)
    ax.set_title("composition pedestal", pad=3)


def panel_zproj(ax, num):
    f = RES / "aviti24_zprojection_correction.json"
    if not f.exists():
        return missing(ax, f.name)
    d = json.load(open(f)); m = d["modes"]
    ax.axvspan(0, d["pitch_um"], color="0.9", lw=0, zorder=0)
    for mode, ls, mk, lab in (("per-slice", "-", "o", "per plane"), ("projected", (0, (3, 1.5)), "^", "projected")):
        r = np.array(m[mode]["r_coarse_um"])
        ax.plot(r, m[mode]["same"], ls=ls, marker=mk, color=COL["same"], ms=2.4, lw=1.0)
        ax.plot(r, m[mode]["diff"], ls=ls, marker=mk, color=COL["diff"], ms=2.4, lw=1.0)
    num["b"] = dict(source=str(f.relative_to(REPO)), pitch_um=d["pitch_um"], n_wells=d.get("n_wells"), n_genes=d.get("n_genes"),
                    **{mode: dict(bin0_same=m[mode]["bin0_same"], fold_above_random=m[mode]["fold_above_random"]) for mode in ("per-slice", "projected")})
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlim(0, 3.0); ax.set_ylim(-0.6, 2.45); ax.set_xlabel("r [µm]", labelpad=1); ax.set_ylabel("stacked log g")
    gene_h = [(Line2D([], [], color=COL["same"], lw=2.0), "same gene"), (Line2D([], [], color=COL["diff"], lw=2.0), "different genes")]
    sty_h = [(Line2D([], [], color="0.3", marker="o", ms=2.4, lw=1.0), "pairs per plane"),
             (Line2D([], [], color="0.3", marker="^", ms=2.4, lw=1.0, ls=(0, (3, 1.5))), "planes projected")]
    l1 = ax.legend(*zip(*gene_h), frameon=False, loc="upper right", handlelength=1.4, labelspacing=0.2, borderaxespad=0.1)
    ax.add_artist(l1)
    ax.legend(*zip(*sty_h), frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.82), handlelength=2.8, labelspacing=0.2, borderaxespad=0.1)
    ax.set_title("z-projection (AVITI24)", pad=3)


def panel_resolution(ax, num):
    f = RES_K / "resolution.json"
    if not f.exists():
        return missing(ax, f.name)
    d = json.load(open(f)); e = d["classes"]["endogenous"]; cp = d["centroid_precision_um"]; dl = d["delta_um"]
    r = np.array(e["r_um"]); ys = np.array(e["log_g_same"]); yd = np.array(e["log_g_diff"])
    ok = np.isfinite(ys)
    if cp is not None:   # shaded: r below the median localization precision
        ax.axvspan(0, cp["p50"], color="0.88", lw=0, zorder=0); shade = cp["p50"]
    else:                # no precision in the deposit: shade the bins without same-gene pairs (decoder hard core)
        shade = round(r[np.argmax(ok)] / dl - 0.5) * dl; ax.axvspan(0, shade, color="0.88", lw=0, zorder=0)
    ax.plot(r[ok], ys[ok], "o-", color=COL["same"], ms=1.6, lw=1.0, label="same gene")
    ax.plot(r, yd, "o-", color=COL["diff"], ms=1.6, lw=1.0, label="different genes")
    # peak and minimum below 0.5 um; floor = lower bin edge of the maximum beyond the minimum
    sub = np.where(ok & (r < 0.5))[0]; ipk = sub[np.argmax(ys[sub])]; imn = sub[np.argmin(ys[sub])]
    beyond = np.where(ok & (r > r[imn]))[0]; ifl = beyond[np.argmax(ys[beyond])]; floor = round(r[ifl] / dl - 0.5) * dl
    ax.axvline(floor, color=COL["floor"], lw=0.8, ls=":")
    num["c"] = dict(source=str(f.relative_to(REPO)), n_fov=d["n_fov"], n_molecules=d["n_molecules"], n_genes=e["n_features"], delta_um=dl,
                    shaded_below_um=float(shade), shaded_meaning="median localization precision" if cp is not None else "bins without same-gene pairs",
                    peak=dict(r_um=float(r[ipk]), log_g=float(ys[ipk])), minimum=dict(r_um=float(r[imn]), log_g=float(ys[imn])),
                    floor_um=float(floor), log_g_at_floor_bin=float(ys[ifl]))
    if cp is not None:
        panel_precision_box(ax, cp)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlim(0, 1.5); ax.set_ylim(*((-1.0, 2.6) if cp is not None else (-0.4, YMAX_RES))); ax.set_xlabel("r [µm]", labelpad=1); ax.set_ylabel("stacked log g")
    ax.legend(frameon=False, loc="upper right", handlelength=1.6, labelspacing=0.2, borderaxespad=0.1)
    ax.set_title("resolution floor (CosMx)", pad=3)


def panel_precision_box(ax, cp):
    yb = -0.75
    ax.plot([cp["p5"], cp["p95"]], [yb, yb], color="k", lw=0.7)
    ax.add_patch(plt.Rectangle((cp["p25"], yb - 0.09), cp["p75"] - cp["p25"], 0.18, fc="white", ec="k", lw=0.7, zorder=3))
    ax.plot([cp["p50"], cp["p50"]], [yb - 0.09, yb + 0.09], color="k", lw=1.0, zorder=4)
    ax.text(cp["p95"] + 0.04, yb, f"localization precision\nmedian {cp['p50'] * 1000:.0f} nm "
            f"({cp['p5'] * 1000:.0f}–{cp['p95'] * 1000:.0f} nm)", fontsize=6, va="center")


def main():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.35), gridspec_kw=dict(wspace=0.36, left=0.06, right=0.995, top=0.9, bottom=0.17))
    num = {}
    panel_pedestal(axs[0], num); panel_zproj(axs[1], num); panel_resolution(axs[2], num)
    for ax, lab in zip(axs, "abc"):
        ax.annotate(lab, xy=(0, 1), xycoords="axes fraction", xytext=(-28, 4), textcoords="offset points", fontweight="bold", fontsize=8, ha="left", va="bottom", annotation_clip=False)
    json.dump(dict(figure="preprint Supplementary Fig. S4 (figures/figS4_subcellular)", script="figures/make_figS4.py", panels=num),
              open(RES_K / "figS4_numbers.json", "w"), indent=1)
    out = REPO / "figures/figS4_subcellular"
    fig.savefig(str(out) + ".pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(str(out) + ".png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    print("wrote", out)


if __name__ == "__main__":
    main()
