#!/usr/bin/env python
"""Preprint Fig. 2: cross-platform invariance of normalized two-point functions on CRC serial sections.
Regenerates figures/fig2_platform_invariance.{pdf,png} from data/twopoint/ (log g arrays of
the stored runs, no per-tile tensors) and results/platforms/ (scatter_stats.json, crc_replicates.json). Panels:
  (a) log g_ab(r) with 95 % tile-bootstrap bands for three gene pairs (PECAM1-VWF, MS4A1-CD3E, CEACAM5-CD3E) on Xenium
      (molecule level), Visium HD (2 µm bins) and Visium v2 (55 µm spots), each curve from the platform's own resolution (first
      bin at or beyond 0.5 µm, 2 µm and the 64 µm spot lattice) to 3 mm. The "Xenium binned to 2 µm" run is not shown: its
      integer pair counts are identical to the molecule-level run for r >= 8 µm by construction (same lattice levels);
  (b) method-method scatter of log g over all gene pairs, Visium HD vs Xenium, at r in [10, 12), [96, 112), [896, 1024) µm,
      same axes, identity line dashed, line fit solid (errors-in-variables (Deming) slope through the means); color = relative
      local point density (binned Gaussian KDE, Scott's-rule bandwidth; point_density), densest points on top; Pearson r in
      the lower right corner (point estimate of results/platforms/scatter_stats.json, scripts/scatter_stats.py);
  (c) replicates: Xenium-vs-HD agreement on three patients against the same-platform agreement between patients, on every bin
      of the common 2 µm lattice from the intra-bin shell [0, 2) µm on, plotted at the lattice-shell distance r_central
      (1 µm for the intra-bin shell).
Writes results/platforms/fig2_pearson.json (per scatter panel: comparison, bins, Pearson r, line-fit slope and interval,
whether the line is drawn, density bandwidths). `python figures/make_fig2.py --print` prints the numbers quoted in the caption.
The other scatter figures (make_figS1.py, make_figS2.py) import scatter_panel, panel_a and the style from here.
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data/twopoint"
RES = REPO / "results/platforms"
# Platform palette of every preprint figure (Okabe-Ito; color = platform wherever platforms are shown). The other figure scripts
# copy these hex values with the comment "platform palette as in make_fig2.py"; change them here and there together.
# AVITI24 and CosMx (Supplementary Fig. S4 only) take two colors outside the Okabe-Ito set so that they repeat no platform color.
PLATFORM = {"Xenium": "#0072B2", "Visium HD": "#009E73", "Visium v2": "#CC79A7", "MERSCOPE": "#E69F00", "Xenium 5K": "#56B4E9",
            "Atera": "#D55E00", "AVITI24": "#882255", "CosMx": "#999933", "cross": "#000000"}   # cross: two platforms compared
OKABE = {"Xenium": PLATFORM["Xenium"], "Visium HD": PLATFORM["Visium HD"], "Visium v2": PLATFORM["Visium v2"],
         "xenium": PLATFORM["Xenium"], "hd": PLATFORM["Visium HD"], "cross": PLATFORM["cross"]}
R_MAX = 3000.0   # µm: beyond ~3 mm (section size 6–7 mm) the shells leave the tissue and the tile bootstrap is degenerate
SD_MAX = 0.5
SCALES = {"10": [(10, 12)], "100": [(96, 112)], "1000": [(896, 1024)]}
DENS_MIN = 1e-3   # color scale of the point density: log, from 1/1000 of the maximum (lower values clipped) to the maximum
LIM = (-4.0, 4.0)   # common axis range of every scatter panel (points outside are counted in --print)
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "legend.fontsize": 6.2, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4})


def require(*paths):
    """Stop with a clear message if the two-point runs are absent (they are produced by the workflow, README.md)."""
    missing = [str(Path(p).relative_to(REPO)) for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit("this figure needs the two-point runs " + ", ".join(missing) +
                         " (several GB each). Produce them with the Snakemake workflow (README.md) and run this script again.")


def load_run(path, res, keys=("log_g", "lo", "hi", "sd", "n_eff_pairs", "P", "totals")):
    if not Path(path).exists():
        return None
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in keys}
    out["flag"] = d["flag"] if "flag" in d.files else None      # closed-form reliability flag (2: T_eff < 5, no interval)
    out.update(feats=list(map(str, d["features"])), r=d["r_central"], r_from=d["r_from"], r_to=d["r_to"], res=res)
    return out


def binned(run, bins):
    """log g, SD-filter mask and n_eff of a bin or of a sum of lattice bins [from, to) (pair counts and random-label references
    summed; the SD filter (SD < 0.5 and reliability flag != 2) must pass in every sub-bin; n_eff only for single bins, else
    None). The pair set of scripts/scatter_stats.py additionally requires flag != 2 of a summed bin."""
    ks = [int(np.where(np.isclose(run["r_from"], lo) & np.isclose(run["r_to"], hi))[0][0]) for lo, hi in bins]
    ok = np.logical_and.reduce([np.isfinite(run["log_g"][k]) & (run["sd"][k] < SD_MAX)
                                & (True if run.get("flag") is None else run["flag"][k] < 2) for k in ks])
    if len(ks) == 1:
        return run["log_g"][ks[0]], ok, run["n_eff_pairs"][ks[0]]
    P = sum(run["P"][k] for k in ks); Ptot = sum(run["P"][k].sum() for k in ks); n = run["totals"]
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.log(P / (Ptot * np.outer(n, n) / n.sum() ** 2))
    return lg, ok & np.isfinite(lg), None


def point_density(x, y, n_grid=400):
    """Relative local point density at every point: Gaussian kernel density estimate with Scott's-rule bandwidth per axis
    (h = SD * n^(-1/6)), evaluated by binning the points on an n_grid x n_grid grid over their range, smoothing the histogram
    with the Gaussian kernel and interpolating it bilinearly at the points (binned KDE); scaled to a maximum of 1."""
    from scipy.ndimage import gaussian_filter, map_coordinates
    n = len(x); hx, hy = np.std(x) * n ** (-1 / 6), np.std(y) * n ** (-1 / 6)
    ex = np.linspace(x.min() - 3 * hx, x.max() + 3 * hx, n_grid + 1); ey = np.linspace(y.min() - 3 * hy, y.max() + 3 * hy, n_grid + 1)
    H, _, _ = np.histogram2d(x, y, bins=(ex, ey))
    H = gaussian_filter(H, sigma=(hx / (ex[1] - ex[0]), hy / (ey[1] - ey[0])), mode="constant")
    ix = (x - ex[0]) / (ex[1] - ex[0]) - 0.5; iy = (y - ey[0]) / (ey[1] - ey[0]) - 0.5
    dens = map_coordinates(H, [ix, iy], order=1, mode="nearest")
    return dens / dens.max(), dict(method="binned Gaussian KDE", bandwidth_rule="Scott, per axis", h_x=float(hx), h_y=float(hy), n_grid=n_grid)


def fit_determined(st):
    """The line fit is drawn only where its 95 % interval excludes zero; otherwise the slope is undetermined (spleen at 1 mm)."""
    lo, hi = st["deming_ci"]
    return bool(lo > 0 or hi < 0)


def scatter_panel(ax, A, B, binsA, binsB, st=None, lim=LIM):
    """Scatter of log g_B vs log g_A over the upper triangle (a <= b) of pairs passing the filter on both; color = relative local
    point density (point_density), densest points drawn on top; identity line dashed; line fit (errors-in-variables (Deming)
    slope of scatter_stats.json, through the means, over the central 99 % of the x values) solid where determined;
    Pearson r of scatter_stats.json in the lower right corner. Returns (n_pairs, n outside the axis range, mappable, info)."""
    la, oka, _ = binned(A, binsA); lb, okb, _ = binned(B, binsB)
    iu = np.triu_indices(len(A["feats"]))
    ok = (oka & okb)[iu]; x, y = la[iu][ok], lb[iu][ok]
    dens, kde = point_density(x, y)
    o = np.argsort(dens)
    sc = ax.scatter(x[o], y[o], c=np.maximum(dens[o], DENS_MIN), s=0.8, lw=0, cmap="viridis", norm=LogNorm(DENS_MIN, 1), rasterized=True)
    ax.plot(lim, lim, color="0.35", lw=0.6, ls="--", zorder=3)
    drawn = st is not None and fit_determined(st)
    if drawn:
        xs = np.percentile(x, [0.5, 99.5]); ax.plot(xs, y.mean() + st["deming"] * (xs - x.mean()), color="k", lw=0.9, zorder=4)
    if st is not None:
        ax.text(0.98, 0.02, f"Pearson\nr = {st['pearson']:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2, linespacing=1.1)
    tk = np.arange(np.ceil(lim[0]), lim[1] + 0.01, 2 if lim[1] - lim[0] > 5 else 1)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal"); ax.set_xticks(tk); ax.set_yticks(tk)
    out = int(((x < lim[0]) | (x > lim[1]) | (y < lim[0]) | (y > lim[1])).sum())
    info = dict(n_pairs_plotted=int(ok.sum()), n_outside_axes=out, pearson_plotted_points=float(np.corrcoef(x, y)[0, 1]),
                line_fit_drawn=drawn, density=kde)
    if st is not None:
        info.update(pearson=float(st["pearson"]), pearson_rounded=round(float(st["pearson"]), 2), n_pairs=int(st["n_pairs"]),
                    line_fit_slope=float(st["deming"]), line_fit_slope_ci95=[float(v) for v in st["deming_ci"]])
    return int(ok.sum()), out, sc, info


def scatter_record(key, s, binsA, binsB, info, stats_file="results/platforms/scatter_stats.json"):
    """One row of the per-panel JSON (fig2_pearson.json and the supplementary counterparts)."""
    return dict(comparison=key, scale=s, bins_A_um=[list(b) for b in binsA], bins_B_um=[list(b) for b in binsB],
                source=f"{stats_file} comparisons.{key}.rows.{s}", **info)


def panel_a(ax_list, pairs, runs, legend_ax=2, legend_loc="lower right"):
    for ax, (ga, gb) in zip(ax_list, pairs):
        for name, v in runs.items():
            if v is None or ga not in v["feats"] or gb not in v["feats"]:
                continue
            i, j = v["feats"].index(ga), v["feats"].index(gb)
            # from the platform's own resolution: first bin starting at or beyond the localization bin (0.5 µm), the 2 µm bin pitch
            # or the 64 µm lattice of the 55 µm spots
            ok = (v["r_from"] >= v["res"] - 1e-9) & (v["r"] <= R_MAX) & np.isfinite(v["log_g"][:, i, j])
            ax.fill_between(v["r"][ok], v["lo"][ok, i, j], v["hi"][ok, i, j], color=OKABE[name], alpha=0.18, lw=0)
            ax.plot(v["r"][ok], v["log_g"][ok, i, j], color=OKABE[name], lw=1.0, label=name)
        ax.axhline(0, color="k", lw=0.4); ax.set_xscale("log"); ax.set_title(f"{ga} – {gb}", pad=3); ax.set_xlabel("r [µm]", labelpad=1)
        ax.set_xlim(0.5, R_MAX * 1.15); ax.set_xticks([1, 10, 100, 1000])
    ax_list[0].set_ylabel("log g_ab(r)")
    ax_list[legend_ax].legend(frameon=False, loc=legend_loc, handlelength=1.2, borderaxespad=0.1)


def legend_below(ax, handles, ncol=1, y=-0.30, **kw):
    """Legend centered below the axes (below its x label)."""
    return ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, handlelength=2.0,
                     borderaxespad=0.0, labelspacing=0.25, columnspacing=1.0, **kw)


def panel_replicates(ax):
    """Xenium against Visium HD on the same patient (black, solid; one line per patient) against one platform between patients
    (platform color, dashed: mean of the three patient pairs; band: their range); legend below the panel."""
    d = json.load(open(RES / "crc_replicates.json"))
    for k, sec in enumerate(d["per_section"]):
        r = np.array([b["r_central"] for b in sec["bins"]]); p = np.array([b["pearson"] for b in sec["bins"]]); m = r <= R_MAX
        ax.plot(r[m], p[m], "-", color=OKABE["cross"], lw=0.9, alpha=0.5 + 0.25 * k)
    for plat in ("xenium", "hd"):
        rows = d["between_sections"][plat]["rows"]
        r = np.array([b["r_central"] for b in rows]); pb = np.array([b["pearson_between_sections"] for b in rows]); m = r <= R_MAX
        ax.fill_between(r[m], pb[m].min(1), pb[m].max(1), color=OKABE[plat], alpha=0.2, lw=0)
        ax.plot(r[m], pb[m].mean(1), "--", color=OKABE[plat], lw=1.0)
    ax.set_xscale("log"); ax.set_xlim(0.8, R_MAX * 1.1); ax.set_ylim(-0.4, 1.02); ax.set_yticks([-0.4, 0, 0.4, 0.8]); ax.set_xticks([1, 10, 100, 1000])
    ax.set_xlabel("r [µm]", labelpad=1); ax.set_ylabel("Pearson r of log g\n(all gene pairs)", labelpad=1); ax.axhline(0, color="k", lw=0.4)
    h = [Line2D([], [], color=OKABE["cross"], lw=1.0, ls="-", label="Xenium vs Visium HD,\nsame patient"),
         Line2D([], [], color=OKABE["xenium"], lw=1.0, ls="--", label="Xenium, between patients"),
         Line2D([], [], color=OKABE["hd"], lw=1.0, ls="--", label="Visium HD, between patients")]
    legend_below(ax, h)


def label(ax, lab, dx=-30, dy=6):
    ax.annotate(lab, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points", fontweight="bold", fontsize=8, ha="left", va="bottom", annotation_clip=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--print", action="store_true", dest="prt"); a = ap.parse_args()
    require(DATA / "xenium_crc_p2_mol_common.npz", DATA / "hd_crc_p2_bin2um_common.npz", DATA / "v2_crc_p2_spots_common.npz")
    runs = {"Xenium": load_run(DATA / "xenium_crc_p2_mol_common.npz", 0.5), "Visium HD": load_run(DATA / "hd_crc_p2_bin2um_common.npz", 2.0),
            "Visium v2": load_run(DATA / "v2_crc_p2_spots_common.npz", 64.0)}
    stats = json.load(open(RES / "scatter_stats.json"))["comparisons"]
    fig = plt.figure(figsize=(7.2, 4.1))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1], hspace=0.5, left=0.075, right=0.985, top=0.95, bottom=0.2)
    gs_a = gs[0].subgridspec(1, 3, wspace=0.3); gs_b = gs[1].subgridspec(1, 4, width_ratios=[1, 1, 1, 1.1], wspace=0.6)
    axa = [fig.add_subplot(gs_a[0, i]) for i in range(3)]
    panel_a(axa, [("PECAM1", "VWF"), ("MS4A1", "CD3E"), ("CEACAM5", "CD3E")], runs)
    axb = [fig.add_subplot(gs_b[0, i]) for i in range(3)]
    rec = []
    for ax, (s, bins) in zip(axb, SCALES.items()):
        st = stats["crc_xenium_hd"]["rows"][s]
        n, out, sc, info = scatter_panel(ax, runs["Xenium"], runs["Visium HD"], bins, bins, st=st)
        rec.append(scatter_record("crc_xenium_hd", s, bins, bins, info) | dict(panel=f"b{len(rec) + 1}", A="Xenium", B="Visium HD"))
        ax.set_title(f"{bins[0][0]}–{bins[0][1]} µm", pad=3); ax.set_xlabel("log g, Xenium", labelpad=1)
        if a.prt:
            print(f"Fig2b {s} um: n={st['n_pairs']} (plotted {n}, outside axes {out}); Pearson {st['pearson']:.3f} [{st['pearson_ci'][0]:.3f}, {st['pearson_ci'][1]:.3f}]; "
                  f"line fit {st['deming']:.3f} [{st['deming_ci'][0]:.3f}, {st['deming_ci'][1]:.3f}] (lambda {st['lambda_deming']:.2f}); OLS {st['ols']:.3f}; MA {st['major_axis']:.3f}; "
                  f"RMS {st['rms']:.3f}; mean diff {st['mean_diff']:+.3f}; SD log g X {st['sd_logg_A']:.3f} HD {st['sd_logg_B']:.3f}")
    axb[0].set_ylabel("log g, Visium HD", labelpad=1)
    axc = fig.add_subplot(gs_b[0, 3])
    panel_replicates(axc)
    for ax, lab in ((axa[0], "a"), (axb[0], "b"), (axc, "c")):
        label(ax, lab, dx=-28)
    json.dump(dict(figure="preprint Fig. 2 (figures/fig2_platform_invariance), panel b", script="figures/make_fig2.py",
                   note="pearson = point estimate of scatter_stats.json (quoted in the panel, two decimals); pearson_plotted_points = "
                        "Pearson r of the plotted points (check); density = color of the points", panels=rec),
              open(RES / "fig2_pearson.json", "w"), indent=1)
    out = REPO / "figures/fig2_platform_invariance"
    fig.savefig(str(out) + ".pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(str(out) + ".png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    print("wrote", out)


if __name__ == "__main__":
    main()
