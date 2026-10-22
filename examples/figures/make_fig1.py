#!/usr/bin/env python
"""Preprint Fig. 1 — schematic of the estimator: molecules -> lattice -> shell products -> normalized g(r) per tile -> comparisons.

Regenerates figures/fig1_schematic.{pdf,png} (180 mm wide, vector PDF + 300 dpi PNG). Terminology follows the preprint
(Results, first section; Methods "Two-point estimator", "Distance binning", "Shell products on the
log-hierarchical lattice", "Block bootstrap over tiles", "Normalization levels"). Okabe-Ito colors as in make_fig2.py.
Data (all real except the matrix glyphs and the shell diagram in (c), which are drawn):
  (a) data/processed/xenium_crc_p2/molecules.parquet, a 256 x 256 um crop read with pyarrow filters (~1e5 rows);
  (b) the same crop, molecule counts per lattice cell at pitch h_L = 2^L * Delta (Delta = 0.5 um): levels 4, 5, 6 = 8, 16, 32 um;
  (d) data/twopoint/xenium_crc_p2_mol_common.npz (whole-section log g and its 95 % interval from the block bootstrap over
      500 um tiles, molecule level). Per-tile functions are not drawn: each tile has its own random-label reference, so
      their quantile bands are not centered on the whole-section function;
  (e) data/twopoint/{xenium,hd}_crc_p{1,2}_bin2um_common.npz (global log g with 95 % tile-bootstrap intervals), every bin of the
      2 um lattice from the intra-bin shell [0, 2) um (plotted at r_central = 1 um) to 3 mm.
      Xenium and Visium HD of P2 are serial sections; their CEACAM5-CD3E functions differ below ~50 um and agree above.
      Visium HD is the more segregated platform, so lateral diffusion of the 2 um bins (which would flatten log g) does
      not explain the difference.
Layout: all five axes share one top edge, so titles and panel letters align; the four arrows sit at one height; the matrix
glyphs in (c) are vector rectangles. The gene pair CEACAM5-CD3E is one of the three pairs of Fig. 2 (make_fig2.py). Only one
gene pair is read from each npz.
"""
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle, Patch
from matplotlib.collections import PatchCollection
import pyarrow.dataset as ds

REPO = Path(__file__).resolve().parents[1]
MOL = REPO / "data/processed/xenium_crc_p2/molecules.parquet"
TP = REPO / "data/twopoint"
OUT = REPO / "figures/fig1_schematic"

GA, GB = "CEACAM5", "CD3E"                      # pair shown in (d), (e); colored in (a)
X0, Y0, SIDE = 4250.0, 1792.0, 256.0            # crop (um) at a tumor/T-cell interface of CRC P2
DELTA = 0.5                                     # um, level-0 bin width (Methods, Distance binning)
R_MAX = 3000.0                                  # um, as in make_fig2.py
C = {"ga": "#0072B2", "gb": "#D55E00", "gray": "#BBBBBB", "glob": "#000000", "tile": "#56B4E9",
     "xenium": "#0072B2", "hd": "#009E73"}   # xenium, hd: platform palette as in make_fig2.py (e: color = platform, style = patient)
plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 7, "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5})


def read_crop():
    dset = ds.dataset(str(MOL), format="parquet")
    f = (ds.field("x") >= X0) & (ds.field("x") < X0 + SIDE) & (ds.field("y") >= Y0) & (ds.field("y") < Y0 + SIDE)
    t = dset.to_table(columns=["x", "y", "feature"], filter=f)
    return t.column("x").to_numpy() - X0, t.column("y").to_numpy() - Y0, t.column("feature").to_pandas().astype(str).to_numpy()


def load_pair(path, ga=GA, gb=GB):
    d = np.load(path, allow_pickle=True)
    feats = list(map(str, d["features"])); i, j = feats.index(ga), feats.index(gb)
    return dict(r=d["r_central"], lg=d["log_g"][:, i, j], lo=d["lo"][:, i, j], hi=d["hi"][:, i, j])


def label(ax, s, dx=-8, dy=3):
    ax.annotate(s, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points",
                fontweight="bold", fontsize=9, ha="left", va="bottom", annotation_clip=False)


def panel_a(ax, x, y, f):
    other = (f != GA) & (f != GB)
    ax.scatter(x[other], y[other], s=0.15, c=C["gray"], lw=0, rasterized=True)
    for g, s in ((GA, 0.5), (GB, 1.6)):
        m = f == g
        ax.scatter(x[m], y[m], s=s, c=C["ga" if g == GA else "gb"], lw=0, rasterized=True)
    ax.set_xlim(0, SIDE); ax.set_ylim(SIDE, 0); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), SIDE, SIDE, fill=False, lw=0.5, ec="0.3"))
    ax.plot([SIDE - 12 - 50, SIDE - 12], [SIDE - 10] * 2, color="k", lw=1.5, solid_capstyle="butt")
    ax.text(SIDE - 12 - 25, SIDE - 16, "50 µm", ha="center", va="bottom", fontsize=6,
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
    ax.set_title("Molecules", pad=3)
    ax.text(0.0, -0.04, "●", color=C["ga"], transform=ax.transAxes, va="top", fontsize=6)
    ax.text(0.07, -0.04, GA, transform=ax.transAxes, va="top", fontsize=6)
    ax.text(0.52, -0.04, "●", color=C["gb"], transform=ax.transAxes, va="top", fontsize=6)
    ax.text(0.59, -0.04, GB, transform=ax.transAxes, va="top", fontsize=6)
    ax.text(0.0, -0.13, f"Xenium CRC P2, {len(f) / 1e3:.0f}k\nmolecules (others gray)", transform=ax.transAxes, va="top", fontsize=6)


def panel_b(ax, x, y):
    h = 2 ** 4 * DELTA                                            # level 4: 8 um
    n = int(SIDE / h)
    H, _, _ = np.histogram2d(y, x, bins=n, range=[[0, SIDE], [0, SIDE]])
    ax.imshow(H, extent=(0, SIDE, SIDE, 0), cmap="Greys", vmin=0, vmax=1.6 * np.percentile(H, 99), interpolation="nearest")
    styles = {5: dict(color=C["tile"], lw=0.6), 6: dict(color="k", lw=1.0)}
    for L, st in styles.items():
        p = 2 ** L * DELTA
        for v in np.arange(p, SIDE, p):
            ax.axvline(v, **st); ax.axhline(v, **st)
    ax.set_xlim(0, SIDE); ax.set_ylim(SIDE, 0); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), SIDE, SIDE, fill=False, lw=0.5, ec="0.3"))
    ax.set_title("Lattice", pad=3)
    rows = [("■", "0.55", "level 4, h = 8 µm (counts)"), ("—", C["tile"], "level 5, h = 16 µm"), ("—", "k", "level 6, h = 32 µm")]
    for k, (sym, col, txt) in enumerate(rows):
        ax.text(0.0, -0.04 - 0.085 * k, sym, color=col, transform=ax.transAxes, va="top", fontsize=6, fontweight="bold")
        ax.text(0.09, -0.04 - 0.085 * k, txt, transform=ax.transAxes, va="top", fontsize=6)
    ax.text(0.0, -0.30, r"pitch $h_L = 2^L\Delta$, $\Delta$ = 0.5 µm",
            transform=ax.transAxes, va="top", fontsize=6)


def panel_c(ax_sh, ax_mx):
    # shells: integer offsets o of one lattice cell, shell k = floor(|o|), the four shells 4 <= |o| < 8 of one level
    R = 8
    o = np.arange(-R, R + 1)
    ox, oy = np.meshgrid(o, o)
    k = np.floor(np.hypot(ox, oy)).astype(int)
    img = np.full(k.shape, np.nan); sel = (k >= 4) & (k < 8); img[sel] = k[sel]
    shell_cols = ["#C6DBEF", "#6BAED6", "#2171B5", "#08306B"]
    ax_sh.imshow(img, cmap=ListedColormap(shell_cols), vmin=3.5, vmax=7.5, extent=(-R - .5, R + .5, R + .5, -R - .5), interpolation="nearest")
    for v in np.arange(-R - .5, R + 1, 1):
        ax_sh.plot([v, v], [-R - .5, R + .5], color="0.85", lw=0.25, zorder=0.5); ax_sh.plot([-R - .5, R + .5], [v, v], color="0.85", lw=0.25, zorder=0.5)
    ax_sh.add_patch(Rectangle((-.5, -.5), 1, 1, color="k"))
    ax_sh.set_xlim(-R - .6, R + .6); ax_sh.set_ylim(R + .6, -R - .6); ax_sh.set_aspect("equal"); ax_sh.axis("off")
    ax_sh.set_title("Shell products", pad=3)
    for j, kk in enumerate(range(4, 8)):
        ax_sh.text(1.04, 0.92 - 0.13 * j, "■", color=shell_cols[j], transform=ax_sh.transAxes, fontsize=7, va="center")
        ax_sh.text(1.16, 0.92 - 0.13 * j, f"k = {kk}", transform=ax_sh.transAxes, fontsize=6, va="center")
    ax_sh.text(1.04, 0.33, "lattice\nshells", transform=ax_sh.transAxes, fontsize=6, va="center")

    # matrix glyphs: P_k = X^T (S_k X)
    rng = np.random.default_rng(0)
    N, G = 24, 6
    X = (rng.random((N, G)) < 0.35) * rng.integers(1, 4, (N, G))
    S = np.zeros((N, N))
    for off in (3, 5, 9):                                          # offsets -> diagonals of the sorted-hash order
        idx = np.arange(N - off); keep = rng.random(len(idx)) < 0.75
        S[idx[keep], idx[keep] + off] = 1; S[idx[keep] + off, idx[keep]] = 1
    P = X.T @ S @ X
    gap, xx = 2.2, 0.0
    tiles = [(r"$X^{\mathsf{T}}$", X.T, "Blues"), (r"$S_k$", S, "Greys"), (r"$X$", X, "Blues"), (r"$P_k$", P, "Oranges")]
    for k, (lab, M, cm) in enumerate(tiles):
        h, w = M.shape
        top = (N - h) / 2
        cmap = matplotlib.colormaps[cm]; vmin, vmax = -0.5 * M.max(), M.max()
        ii, jj = np.nonzero(M)                                       # one vector square per nonzero element
        cells = [Rectangle((xx + j, top + i), 1, 1) for i, j in zip(ii, jj)]
        ax_mx.add_collection(PatchCollection(cells, facecolors=[cmap((M[i, j] - vmin) / (vmax - vmin)) for i, j in zip(ii, jj)],
                                             edgecolors="none", linewidths=0))
        ax_mx.add_patch(Rectangle((xx, top), w, h, fill=False, lw=0.5, ec="0.3", joinstyle="miter", zorder=3))
        ax_mx.text(xx + w / 2, -1.2, lab, ha="center", va="bottom", fontsize=7)
        xx += w + gap
        if k == 2:
            ax_mx.text(xx - gap / 2 + 0.9, N / 2, "=", ha="center", va="center", fontsize=8)
            xx += 2.2
    ax_mx.text(0, N + 2.0, "X: lattice cells × genes (counts)\n" r"$S_k$: shell adjacency (sparse)" "\n" r"$P_k$: genes × genes, one per shell",
               ha="left", va="top", fontsize=6, color="0.2", linespacing=1.1)
    ax_mx.set_xlim(-0.5, xx - gap + 0.5); ax_mx.set_ylim(N + 15, -5); ax_mx.set_aspect("equal"); ax_mx.axis("off")


def panel_d(ax):
    mol = load_pair(TP / "xenium_crc_p2_mol_common.npz")
    # lattice levels: level 0 = linear bins below 4 um, level L >= 1 = shells [4 h_L, 8 h_L), h_L = 2^L Delta
    edges = [4 * 2 ** L * DELTA for L in range(1, 14)]
    ax.axvspan(0.2, 4, color="0.80", lw=0, zorder=0)
    for L in range(1, 12, 2):
        ax.axvspan(edges[L - 1], edges[L], color="0.93", lw=0, zorder=0)
    r = mol["r"]; ok = (r <= R_MAX) & np.isfinite(mol["lg"])
    ax.fill_between(r[ok], mol["lo"][ok], mol["hi"][ok], color=C["glob"], alpha=0.25, lw=0)
    ax.plot(r[ok], mol["lg"][ok], color=C["glob"], lw=0.9, marker="o", ms=1.2, label="whole section")
    hband = Patch(color=C["glob"], alpha=0.25, lw=0, label="95% interval,\ntile bootstrap")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xscale("log"); ax.set_xlim(0.3, R_MAX); ax.set_xticks([1, 10, 100, 1000]); ax.set_ylim(-1.6, 0.45); ax.set_yticks([-1.5, -1.0, -0.5, 0.0])
    ax.set_xlabel("r [µm]", labelpad=1); ax.set_ylabel(r"$\log g_{ab}(r)$", labelpad=1)
    ax.set_title("Normalized g(r)", pad=3)
    ax.text(1.1, 0.97, "level\n0", fontsize=6, color="0.25", ha="center", va="top", linespacing=0.95, transform=ax.get_xaxis_transform())
    for L, xt in ((1, 5.6), (2, 11.3), (3, 22.6)):
        ax.text(xt, 0.97, f"{L}", fontsize=6, color="0.25", ha="center", va="top", transform=ax.get_xaxis_transform())
    ax.text(45, 0.97, "…", fontsize=6, color="0.25", ha="center", va="top", transform=ax.get_xaxis_transform())
    # legend and definition under the panel, so no text sits on the level shading
    h, _ = ax.get_legend_handles_labels()
    leg = ax.legend(handles=h + [hband], frameon=False, loc="upper left", bbox_to_anchor=(-0.02, -0.27), handlelength=1.0, borderaxespad=0.0,
                    labelspacing=0.25, fontsize=6, title=f"{GA} – {GB}", title_fontsize=6)
    leg._legend_box.align = "left"
    ax.text(0.0, -0.66, r"$g_{ab}(r_k)=P_{ab}(k)\,/\,P^{\mathrm{ref}}_{ab}(k)$", transform=ax.transAxes, ha="left", va="top", fontsize=6)


def panel_e(ax_t, ax_b):
    runs = {k: load_pair(TP / f"{k}_bin2um_common.npz") for k in ("xenium_crc_p2", "hd_crc_p2", "xenium_crc_p1")}
    def draw(ax, key, col, lab, ls="-"):
        v = runs[key]; ok = (v["r"] <= R_MAX) & np.isfinite(v["lg"])
        ax.fill_between(v["r"][ok], v["lo"][ok], v["hi"][ok], color=col, alpha=0.25, lw=0)
        ax.plot(v["r"][ok], v["lg"][ok], ls, color=col, lw=0.9, label=lab)
    draw(ax_t, "xenium_crc_p2", C["xenium"], "Xenium"); draw(ax_t, "hd_crc_p2", C["hd"], "Visium HD")
    draw(ax_b, "xenium_crc_p2", C["xenium"], "P2"); draw(ax_b, "xenium_crc_p1", C["xenium"], "P1", "--")
    ax_t.set_title("Comparisons", pad=3)
    for ax, txt in ((ax_t, "P2 serial sections,\ntwo platforms"), (ax_b, "Xenium,\ntwo patients")):
        ax.set_xscale("log"); ax.set_xlim(0.8, R_MAX); ax.set_xticks([1, 10, 100, 1000]); ax.set_ylim(-2.7, 1.05); ax.axhline(0, color="k", lw=0.4)
        ax.set_yticks([-2, -1, 0])
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=6, va="top", linespacing=0.95)
        ax.legend(frameon=False, loc="lower right", handlelength=1.1, handletextpad=0.4, borderaxespad=0.1, labelspacing=0.15, fontsize=6)
    ax_t.tick_params(labelbottom=False)
    ax_b.set_xlabel("r [µm]", labelpad=1)
    ax_b.set_ylabel(r"$\log g_{ab}(r)$", labelpad=1); ax_b.yaxis.set_label_coords(-0.31, 1.1)


def arrow(fig, ax1, ax2, y):
    b1, b2 = ax1.get_position(), ax2.get_position()
    x1, x2 = b1.x1 + 0.004, b2.x0 - 0.004
    fig.add_artist(FancyArrowPatch((x1, y), (x2, y), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7, lw=0.9, color="0.35"))


def main():
    need = [MOL] + [TP / f for f in ("xenium_crc_p2_mol_common.npz", "xenium_crc_p2_bin2um_common.npz", "hd_crc_p2_bin2um_common.npz",
                                      "xenium_crc_p1_bin2um_common.npz")]
    missing = [str(p.relative_to(REPO)) for p in need if not p.exists()]
    if missing:
        raise SystemExit("this figure needs the converted molecules and the two-point runs " + ", ".join(missing) +
                         "; produce them with the Snakemake workflow (snakemake fig1, see README.md).")
    W, H = 180 / 25.4, 62 / 25.4
    fig = plt.figure(figsize=(W, H))
    # columns (inches): a | b | c | d | e
    def box(x, y, w, h):
        return fig.add_axes((x / W, y / H, w / W, h / H))
    top = 2.15
    sq = 1.10
    axa = box(0.10, top - sq, sq, sq)
    axb = box(1.42, top - sq, sq, sq)
    axc1 = box(2.72, top - 0.95, 0.95, 0.95)
    axc2 = box(2.68, 0.15, 1.45, 0.95)
    axd = box(4.74, 1.00, 0.97, top - 1.00)                       # all axes share the top edge `top`; legend of (d) below it
    axe1 = box(6.30, 1.30, 0.76, top - 1.30)
    axe2 = box(6.30, 0.36, 0.76, 0.85)
    x, y, f = read_crop()
    panel_a(axa, x, y, f); panel_b(axb, x, y); panel_c(axc1, axc2); panel_d(axd); panel_e(axe1, axe2)
    for ax, s, dx in ((axa, "a", -6), (axb, "b", -6), (axc1, "c", -6), (axd, "d", -30), (axe1, "e", -30)):
        label(ax, s, dx=dx)
    ya = (top - sq / 2) / H
    arrow(fig, axa, axb, ya); arrow(fig, axb, axc1, ya)
    bc, bd = axc1.get_position(), axd.get_position()
    fig.add_artist(FancyArrowPatch((bc.x1 + 0.065, ya), (bd.x0 - 0.058, ya), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7, lw=0.9, color="0.35"))
    bdd, be = axd.get_position(), axe1.get_position()
    fig.add_artist(FancyArrowPatch((bdd.x1 + 0.01, ya), (be.x0 - 0.045, ya), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7, lw=0.9, color="0.35"))
    fig.savefig(str(OUT) + ".pdf", dpi=600); fig.savefig(str(OUT) + ".png", dpi=300)
    print("wrote", OUT, f"({len(f)} molecules in the crop)")


if __name__ == "__main__":
    main()
