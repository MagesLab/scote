#!/usr/bin/env python
"""Preprint Fig. 4: statistical power and error scaling. Regenerates figures/fig4_error_scaling.{pdf,png}
(2 x 2, 17.5 cm wide, text >= 7 pt) from results/error_scaling/ and writes results/error_scaling/fig4_panel_c_slopes.json.
  (a) sigma(log g) vs sampled area fraction f (median over gene pairs, one r bin of about 20 um) on three tissues (CRC P2 Xenium,
      Xenium 5K lymph node, MERFISH liver slice 1); points are the plotted pair medians and the lines are least-squares power-law
      fits log sigma = c - alpha log f to those points (alpha in the legend = slope of the drawn line);
  (b) the exponent alpha (sigma ~ f^-alpha) vs r for the same data (liver: all four slices; median of the per-pair fits);
      0.5 = independent tiles;
  (c) molecules vs area: sigma vs the fraction of the pairs, subsampling molecules at fixed area (m^2 of the pairs) vs subsampling
      area at fixed density (f of the pairs); dotted: pure shot noise ~ N_pairs^-1/2. Every exponent written to the slopes table
      is the least-squares slope of the plotted points against the plotted x: alpha for area, beta/2 for molecules, where
      sigma ~ m^-beta for a molecule fraction m. Legends are split into color -> data set and line style -> sampling;
  (d) replicates: between-section SD of log g vs within-section bootstrap SD vs r, CRC P1/P2/P5 on Xenium and Visium HD, and the
      four MERFISH liver slices (between slices, between animals).
Missing inputs are skipped (the panel says so).
"""
import argparse
import json
from pathlib import Path
import numpy as np
from matplotlib.lines import Line2D
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results/error_scaling"
OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#000000"]
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "legend.fontsize": 4.8, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
R_PANEL = 20.0     # µm: r bin shown in (a) and (c)
LEGFS = 7.0        # legend size [pt]
R_MAX = 4000.0     # beyond ≈ 4 mm the shells leave the sections and the tile bootstrap is degenerate
TISSUES = [("CRC Xenium", "xenium_crc_p2_top200.json"), ("lymph node", "x5k_lymph_node_top200.json"), ("liver slice 1", "vizgen_liver1slice1_top200.json")]
LIVER_OTHER = ["vizgen_liver1slice2_top200.json", "vizgen_liver2slice1_top200.json", "vizgen_liver2slice2_top200.json"]
C_CRC, C_LN, C_LIV, C_HD, C_K = OKABE[0], OKABE[1], OKABE[2], OKABE[3], OKABE[4]


def jload(name, base=RES):
    p = base / name
    return json.load(open(p)) if p.exists() else None


def nearest(rows, r, key="r_um"):
    rows = [x for x in rows if key in x]
    return min(rows, key=lambda x: abs(x[key] - r)) if rows else None


PANEL_C_SLOPES = []   # least-squares slopes of the plotted points in (c); written to results/error_scaling/fig4_panel_c_slopes.json (--preprint)


def slope(xs, ys):
    lx, ly = np.log(xs), np.log(ys); lx = lx - lx.mean()
    return -(lx @ (ly - ly.mean())) / (lx ** 2).sum()


def panel_a(ax):
    for (label, fn), col in zip(TISSUES, (C_CRC, C_LN, C_LIV)):
        d = jload(fn)
        if d is None: continue
        row = nearest([x for x in d["rows"] if "sigma_median_by_f" in x], R_PANEL)
        fs = sorted(float(f) for f in row["sigma_median_by_f"]); ys = [row["sigma_median_by_f"][f"{f:g}"] for f in fs]
        a = slope(fs, ys); c = np.mean(np.log(ys)) + a * np.mean(np.log(fs))           # least-squares line in log-log
        ff = np.geomspace(min(fs), max(fs), 50)
        ax.plot(fs, ys, "o", ms=3, color=col)
        ax.plot(ff, np.exp(c) * ff ** -a, "-", lw=1, color=col, label=f"{label}: α = {a:.2f}")
        rb = nearest([x for x in d["rows"] if "alpha_median" in x], R_PANEL)
        print(f"(a) {label}: fit to plotted medians alpha = {a:.3f}; panel b (median of per-pair fits) at {rb['r_um']:.1f} um: {rb['alpha_median']:.3f}")
    if not ax.lines:
        ax.text(0.5, 0.5, "results/error_scaling/ missing", ha="center", va="center", transform=ax.transAxes)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("sampled area fraction f"); ax.set_ylabel(f"σ at r ≈ {R_PANEL:.0f} µm (pair median)")
    ax.set_ylim(2e-4, 0.6)
    h, l = ax.get_legend_handles_labels()
    h = h + [Line2D([], [], ls="none", marker="o", ms=3, color="0.3"), Line2D([], [], ls="-", lw=1, color="0.3")]
    l = l + ["pair median", "fit σ ∝ f$^{-α}$"]
    ax.legend(h, l, frameon=False, loc="lower left", fontsize=LEGFS, handlelength=1.6)


def panel_b(ax):
    for (label, fn), col in zip(TISSUES, (C_CRC, C_LN, C_LIV)):
        d = jload(fn)
        if d is None: continue
        rows = [x for x in d["rows"] if "alpha_median" in x and x["r_um"] <= R_MAX]
        lab = label
        if fn.startswith("vizgen"):
            others = [jload(o) for o in LIVER_OTHER]; others = [o for o in others if o is not None]
            for o in others:
                ro = [x for x in o["rows"] if "alpha_median" in x and x["r_um"] <= R_MAX]
                ax.plot([x["r_um"] for x in ro], [x["alpha_median"] for x in ro], "-", lw=0.6, color=col, alpha=0.55)
            if others: lab = f"liver (slice 1; thin: slices 2–{1 + len(others)})"
            else: lab = f"{label}"
        else:
            lab = f"{label}"
        ax.errorbar([x["r_um"] for x in rows], [x["alpha_median"] for x in rows], yerr=[x["alpha_sd_subsets"] for x in rows], fmt="o-", ms=2, lw=0.9, capsize=0, color=col, label=lab)
    ax.axhline(0.5, color="k", lw=0.5, ls="--"); ax.set_xscale("log"); ax.set_xlabel("r [µm]"); ax.set_ylabel("α  (σ ∝ area$^{-α}$)"); ax.set_ylim(0.0, 0.9)
    ax.legend(frameon=False, loc="lower left", fontsize=LEGFS, handlelength=1.6)


def panel_c(ax):
    d = jload("xenium_crc_p2_molecules.json")
    if d is None:
        ax.text(0.5, 0.5, "molecule-subsample runs pending", ha="center", va="center", transform=ax.transAxes); ax.set_axis_off(); return
    color_handles = []
    crc, lnn = "CRC Xenium", "lymph node"
    sets = [(d, R_PANEL, C_CRC, crc), (d, 100.0, OKABE[5], crc)]
    ln = jload("x5k_lymph_node_molecules.json")
    if ln is not None:
        sets.append((ln, R_PANEL, C_LN, lnn))
    for src, r_show, col, name in sets:
        row = nearest([x for x in src["rows"] if "sigma_median_by_m" in x], r_show)
        ms = sorted(float(m) for m in row["sigma_median_by_m"]); ys = [row["sigma_median_by_m"][f"{m:g}"] for m in ms]
        xm = np.array(ms) ** 2                                                    # fraction of the pairs
        ax.plot(xm, ys, "o-", ms=2.5, lw=1, color=col)
        s_mol = slope(xm, ys)                                                     # = beta / 2 (sigma ~ m^-beta)
        lab = f"{name}, {row['r_um']:.0f} µm: β/2 = {s_mol:.2f}"
        rec = dict(data=name, r_um=row["r_um"], pair_fraction=list(map(float, xm)), sigma_molecules=ys, slope_molecules_beta_half=s_mol)
        if "sigma_area_median_by_f" in row:
            fs = sorted(float(f) for f in row["sigma_area_median_by_f"]); ya = [row["sigma_area_median_by_f"][f"{f:g}"] for f in fs]
            ax.plot(fs, ya, "s", ls=(0, (3, 1.5)), ms=2.5, lw=1, color=col)
            s_area = slope(fs, ya); lab += f", α = {s_area:.2f}"
            rec.update(area_fraction=fs, sigma_area=ya, slope_area_alpha=s_area)
            print(f"(c) {name} {row['r_um']:.1f} um: plotted slopes molecules {s_mol:.3f} (beta = {2 * s_mol:.3f}; median per-pair beta "
                  f"{row['beta_median']:.3f}), area {s_area:.3f} (median per-pair alpha {row['alpha_area']:.3f})")
        x = np.array([xm[0], 1.0]); ax.plot(x, ys[-1] / np.sqrt(x), ":", color=col, lw=0.7)
        PANEL_C_SLOPES.append(rec)
        lab = f"{name}, {row['r_um']:.0f} µm"                                     # fit results go to the caption, not the panel
        color_handles.append((Line2D([], [], color=col, lw=2.5), lab))
    style_handles = [(Line2D([], [], color="0.3", lw=1, marker="o", ms=2.5), "molecules, fraction m"),
                     (Line2D([], [], color="0.3", lw=1, marker="s", ms=2.5, ls=(0, (3, 1.5))), "area, fraction f"),
                     (Line2D([], [], color="0.3", lw=0.7, ls=":"), "shot noise ∝ N$_{pairs}^{-1/2}$")]
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("fraction of the pairs (m² molecules, f area)"); ax.set_ylabel("σ(log g) (pair median)")
    ax.set_ylim(1.5e-3, 40)
    l1 = ax.legend(*zip(*color_handles), frameon=False, loc="upper right", fontsize=LEGFS, handlelength=1.2)
    ax.add_artist(l1)
    ax.legend(*zip(*style_handles), frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.77), fontsize=LEGFS, handlelength=3.2)


def panel_d(ax):
    d = jload("replicates_sections.json"); lv = jload("replicates_liver.json")
    if d is None and lv is None:
        ax.text(0.5, 0.5, "replicate table pending", ha="center", va="center", transform=ax.transAxes); ax.set_axis_off(); return
    cols, has_animals = [], False
    if d is not None:
        for (plat, lab), col in zip((("xenium", "Xenium"), ("hd", "Visium HD")), (C_CRC, C_HD)):
            rows = [x for x in d["platforms"][plat]["rows"] if x["r_from"] <= R_MAX]
            r = [x["r_from"] for x in rows]
            ax.plot(r, [x["s_obs"] for x in rows], "-", lw=1.1, color=col)
            ax.plot(r, [x["s_within"] for x in rows], "--", lw=0.9, color=col)
            cols.append((Line2D([], [], color=col, lw=2.5), f"CRC {lab} (3)"))
    if lv is not None:
        rows = [x for x in lv["rows"] if "s_between_slices" in x and 5 <= x["r_um"] <= R_MAX]
        r = [x["r_um"] for x in rows]
        ax.plot(r, [x["s_between_slices"] for x in rows], "-", lw=1.1, color=C_LIV)
        if all("s_between_animals" in x for x in rows):
            ax.plot(r, [x["s_between_animals"] for x in rows], "-.", lw=0.9, color=C_LIV); has_animals = True
        ax.plot(r, [x["s_within_slice"] for x in rows], "--", lw=0.9, color=C_LIV)
        cols.append((Line2D([], [], color=C_LIV, lw=2.5), f"liver MERFISH ({len(lv['slices'])})"))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("r [µm]"); ax.set_ylabel("SD of log g (pair median)"); ax.set_ylim(1.5e-3, 6)
    styles = [(Line2D([], [], color="0.3", lw=1.1), "between sections"), (Line2D([], [], color="0.3", lw=0.9, ls="--"), "within section")]
    if has_animals:
        styles.append((Line2D([], [], color="0.3", lw=0.9, ls="-."), "between animals"))
    l1 = ax.legend(*zip(*cols), frameon=False, loc="upper left", fontsize=LEGFS, handlelength=1.2)
    ax.add_artist(l1)
    ax.legend(*zip(*styles), frameon=False, loc="upper right", fontsize=LEGFS, handlelength=3.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preprint", action="store_true", help="accepted for compatibility; the preprint cut is the only output")
    ap.parse_args()
    plt.rcParams.update({"axes.titlesize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
    fig, axs = plt.subplots(2, 2, figsize=(17.5 / 2.54, 12.5 / 2.54))
    panel_a(axs[0, 0]); panel_b(axs[0, 1]); panel_c(axs[1, 0]); panel_d(axs[1, 1])
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.085, top=0.965, wspace=0.27, hspace=0.3)
    name = "fig4_error_scaling"
    for ax, lab in zip(axs.ravel(), "abcd"):
        ax.text(-0.2, 1.02, lab, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    (REPO / "figures").mkdir(exist_ok=True)
    fig.savefig(REPO / f"figures/{name}.pdf"); fig.savefig(REPO / f"figures/{name}.png", dpi=600)
    out = RES / "fig4_panel_c_slopes.json"
    json.dump(dict(note="panel c: least-squares slope of log sigma against log(fraction of the pairs) of the plotted pair medians; "
                        "molecules: beta/2 (sigma ~ m^-beta, m^2 of the pairs); area: alpha (sigma ~ f^-alpha)", rows=PANEL_C_SLOPES),
              open(out, "w"), indent=1)
    print("wrote", out.relative_to(REPO))
    print(f"wrote figures/{name}.{{pdf,png}}")


if __name__ == "__main__":
    main()
