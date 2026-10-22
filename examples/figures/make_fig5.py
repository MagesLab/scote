#!/usr/bin/env python
"""Preprint Fig. 5: platform artifacts are identifiable, scale-localized features of the molecule-level two-point
functions. Regenerates figures/fig5_artifacts.{pdf,png} from results/artifacts/{collapse,acta2_regions,mastcells_*}.json
(scripts/platform_artifacts.py, scripts/mast_cells.py).
Layout 2 x 3, main panel of each topic on top and its supporting panel below, letters in order of appearance in the text; legends
split color -> one property, style -> another.
  (a) sub-micrometer deficit: same-gene and different-gene log g minus a smooth extrapolation from 1-4 um (quadratic in r), median
      and IQR over pairs, bins 0-6 um; the 1 um threshold (two 0.5 um bins) marked;
  (b) missing same-gene pairs below 1 um per gene against abundance;
  (c) gene-specific background in the tumor region: Xenium - Visium HD difference of log g for the ACTA2 pairs (filled) and the
      TAGLN pairs (open, control), points without connecting lines, and the prediction from the pixel-test b (dashed);
  (d) ACTA2 background density from the pixel test (b * ACTA2 density, SD from the closed-form tile bootstrap) on the tumor,
      normal-mucosa, and stroma regions and on the whole section (all tissue pixels), against a constant-per-area background
      (dotted) and one proportional to the total transcript density (solid), both anchored on the tumor pixel test;
  (e) mast-cell genes vs NK-cell genes, pooled log g(r) on Xenium and Visium HD (color = platform, style = cell type);
  (f) share of the mast signal recovered by cell-level representations.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results/artifacts"
COLM = {"xenium": "#0072B2", "hd": "#009E73"}   # platform palette as in make_fig2.py (e, f: color = platform)
COLK = {"self": "#0072B2", "cross": "#CC79A7"}
COLP = {"ACTA2:ACTA2": "#D55E00", "ACTA2:TAGLN": "#CC79A7", "ACTA2:CEACAM5": "#0072B2", "TAGLN:TAGLN": "#D55E00", "TAGLN:CEACAM5": "#0072B2"}
YMAX_DENSITY = 12.0   # panel d: data up to 9.9; the legend sits in the empty band below 4 (lower left)
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "legend.fontsize": 6.3, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})


def jl(name):
    f = RES / name
    return json.load(open(f)) if f.exists() else None


def panel_deficit(ax):
    d = jl("collapse.json"); e = d["extrapolation"]; f = e["forms"][e["primary"]]
    r = np.array(e["r_central"])
    for kind, lab in (("self", "same gene (a–a)"), ("cross", "different genes (a–b)")):
        b = f[f"{kind}_bins"]; med = np.array([x["median"] for x in b]); lo = np.array([x["q25"] for x in b]); hi = np.array([x["q75"] for x in b])
        ax.fill_between(r, lo, hi, color=COLK[kind], alpha=0.25, lw=0)
        ax.plot(r, med, "o", color=COLK[kind], ms=2.6, label=lab)
        ax.plot(r, med, "-", color=COLK[kind], lw=0.6, alpha=0.6)
    ax.axvspan(e["fit_r_um"][0], e["fit_r_um"][1], color="0.85", lw=0, zorder=0)
    ax.axvline(1.0, color="k", lw=0.6, ls=":")
    ax.axhline(0, color="k", lw=0.4); ax.set_xlim(0, 6); ax.set_ylim(-2.75, 0.55)
    ax.set_xlabel("r [µm]", labelpad=1); ax.set_ylabel("log g − extrapolation from 1–4 µm", labelpad=1)
    ax.legend(frameon=False, loc="lower right", handlelength=1.0, borderaxespad=0.3)
    ax.set_title("same-gene pairs below 1 µm", pad=3)


def panel_acta2(ax):
    d = jl("acta2_regions.json"); dom = d["domains"]["tumor"]; r = np.array(dom["r_um"])
    for key in ("ACTA2:ACTA2", "ACTA2:TAGLN", "ACTA2:CEACAM5", "TAGLN:TAGLN", "TAGLN:CEACAM5"):
        e = dom["pairs"][key]; obs = np.array(e["observed_diff"]); sd = np.array(e["diff_sd"]); acta = key.startswith("ACTA2")
        ax.errorbar(r, obs, yerr=sd, fmt="o" if acta else "s", color=COLP[key], mfc=COLP[key] if acta else "white", ms=2.6 if acta else 2.4,
                    mew=0.7, elinewidth=0.6, capsize=0, lw=0)
        if acta: ax.plot(r, e["pred_pixel"], "--", color=COLP[key], lw=0.9)
    ax.axhline(0, color="k", lw=0.4); ax.set_xscale("log"); ax.set_xlabel("r [µm]", labelpad=1)
    ax.set_ylabel("log g Xenium − log g Visium HD", labelpad=1)
    ax.set_xticks([6, 10, 20, 40, 60]); ax.set_xticklabels(["6", "10", "20", "40", "60"]); ax.minorticks_off(); ax.set_ylim(-1.6, 2.1)
    h1 = [Line2D([], [], color=COLP["ACTA2:ACTA2"], lw=2, label="self pair"), Line2D([], [], color=COLP["ACTA2:CEACAM5"], lw=2, label="× CEACAM5"),
          Line2D([], [], color=COLP["ACTA2:TAGLN"], lw=2, label="× TAGLN")]
    h2 = [Line2D([], [], marker="o", ls="", color="k", ms=3, label="ACTA2"), Line2D([], [], marker="s", ls="", color="k", mfc="white", ms=3, label="TAGLN (control)"),
          Line2D([], [], ls="--", color="k", lw=0.9, label="prediction")]
    l1 = ax.legend(handles=h1, frameon=False, loc="upper left", handlelength=1.0, labelspacing=0.2, borderaxespad=0.1, fontsize=6.2); ax.add_artist(l1)
    ax.legend(handles=h2, frameon=False, loc="upper right", handlelength=1.3, labelspacing=0.2, borderaxespad=0.1, fontsize=6.2)
    ax.set_title("ACTA2 background, tumor region", pad=3)


def panel_mast(ax):
    d = jl("mastcells_twopoint.json"); plat = d["platforms"]
    for name, ls, lw in (("mast7_cross", "-", 1.1), ("NK_cross", "--", 0.9)):
        for p in ("xenium", "hd"):
            r = np.array(plat[p]["r_central"]); b = plat[p]["block"][name]; y = np.array(b["pooled_log_g"]); lo = np.array(b["pooled_lo"]); hi = np.array(b["pooled_hi"])
            ok = (r >= (6 if p == "hd" else 2)) & (r <= 1000) & np.isfinite(y)
            if name == "mast7_cross": ax.fill_between(r[ok], lo[ok], hi[ok], color=COLM[p], alpha=0.2, lw=0)
            ax.plot(r[ok], y[ok], ls, color=COLM[p], lw=lw)
    ax.axhline(0, color="k", lw=0.4); ax.set_xscale("log"); ax.set_xlabel("r [µm]", labelpad=1); ax.set_ylabel("pooled log g of the gene pairs", labelpad=1)
    ax.set_xticks([2, 5, 10, 30, 100, 300, 1000]); ax.set_xticklabels(["2", "5", "10", "30", "100", "300", "1000"]); ax.minorticks_off(); ax.set_ylim(-0.4, 3.7)
    h = [Line2D([], [], color=COLM["xenium"], lw=1.2, label="Xenium"), Line2D([], [], color=COLM["hd"], lw=1.2, label="Visium HD"),
         Line2D([], [], color="k", lw=1.1, label="mast-cell genes"), Line2D([], [], color="k", lw=0.9, ls="--", label="NK-cell genes")]
    ax.legend(handles=h, frameon=False, loc="upper right", ncol=1, handlelength=1.6, labelspacing=0.2, borderaxespad=0.1)
    ax.set_title("mast cells, missed by segmentation", pad=3)


def panel_missing(ax):
    e = jl("collapse.json")["extrapolation"]; pg = e["per_gene"]; n = np.array(pg["n"]); mf = np.array(pg["missing_below_1um"])
    med = e["forms"][e["primary"]]["self_missing_below_1um"]["median"]
    ax.scatter(n, mf * 100, s=3, color=COLK["self"], alpha=0.6, lw=0); ax.set_xscale("log")
    ax.axhline(med * 100, color="k", lw=0.5, ls="--")
    ax.set_ylim(0, 70); ax.set_xlabel("molecules per gene", labelpad=1); ax.set_ylabel("same-gene pairs missing\nbelow 1 µm [%]", labelpad=1)
    ax.set_title("per gene (270 genes)", pad=3)


def panel_density(ax):
    d = jl("acta2_regions.json"); bd = d["background_density"]
    doms = ["tumor", "normal", "stroma", "whole"]; labs = ["tumor", "normal\nmucosa", "stroma", "whole\nsection"]; x = np.arange(len(doms))
    for i, dom in enumerate(doms):
        v = bd[dom]
        ax.plot([i - 0.3, i + 0.3], [v["pred_prop_total"] / 1e3] * 2, color="0.45", lw=1.0, zorder=1)
        ax.errorbar(i, v["bkg_density_pixel"] / 1e3, yerr=v["bkg_density_pixel_sd"] / 1e3, fmt="o", color="k", ms=3.2, elinewidth=0.7, capsize=0, zorder=3)
    ax.axhline(bd["tumor"]["pred_constant"] / 1e3, color="0.45", lw=0.8, ls=":", zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_xlim(-0.5, len(doms) - 0.5); ax.set_ylim(0, YMAX_DENSITY)
    ax.set_ylabel("ACTA2 background\n[10³ calls per mm²]", labelpad=1)
    h = [Line2D([], [], marker="o", ls="", color="k", ms=3, label="pixel test"),
         Line2D([], [], color="0.45", ls=":", lw=0.8, label="constant per area"), Line2D([], [], color="0.45", lw=1.0, label="∝ transcript density")]
    ax.legend(handles=h, frameon=False, loc="lower left", ncol=1, handlelength=1.3, handletextpad=0.4,
              labelspacing=0.2, borderaxespad=0.1, fontsize=6.2)
    ax.set_title("ACTA2 background by region", pad=3)


def panel_recovery(ax):
    c = jl("mastcells_cells.json"); h = jl("mastcells_hd.json"); cv = c["cell_view"]; le = c["leiden"]["resolutions"]
    best_pur = max(v["best_cluster"]["purity_mast3"] for v in le.values())
    hd_pur = max(v["best_cluster"]["purity_mast3"] for v in h["leiden_8um"]["resolutions"].values())
    vals = [c["spots"]["frac_points_in_spots"], 1 - cv["frac_mast_in_cells_mastfrac_lt10pct"]["value"], best_pur, hd_pur]
    labs = ["molecule spots", "cells ≥ 10% mast", "leiden, cells", "leiden, HD bins"]
    y = np.arange(len(vals))[::-1]
    ax.barh(y, [v * 100 for v in vals], color=[COLM["xenium"]] * 3 + [COLM["hd"]], height=0.6)
    for yi, v, lab in zip(y, vals, labs): ax.text(v * 100 + 1.2, yi, f"{v*100:.0f}%, {lab}", va="center", fontsize=6.3)
    ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_xlim(0, 60); ax.set_xlabel("mast signal recovered [%]", labelpad=1)
    ax.set_title("cell-level representations", pad=3)


def main():
    fig = plt.figure(figsize=(7.2, 4.55))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.45, 1.0], hspace=0.55, wspace=0.42, left=0.075, right=0.995, top=0.95, bottom=0.075)
    axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    for ax, fn in zip(axs, (panel_deficit, panel_acta2, panel_mast, panel_missing, panel_density, panel_recovery)):
        fn(ax)
    # grid: main panel of each topic on top, its supporting panel below; letters follow the order of appearance in the text
    for ax, lab in zip(axs, "acebdf"):
        dx = -12 if lab == "f" else -34
        ax.annotate(lab, xy=(0, 1), xycoords="axes fraction", xytext=(dx, 4), textcoords="offset points", fontweight="bold", fontsize=9, ha="left", va="bottom", annotation_clip=False)
    out = REPO / "figures/fig5_artifacts"
    fig.savefig(str(out) + ".pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(str(out) + ".png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    print("wrote", out)


if __name__ == "__main__":
    main()
