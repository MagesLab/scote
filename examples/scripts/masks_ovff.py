#!/usr/bin/env python
"""Region masks of the FF ovarian sections (preprint Supplementary Fig. S3, matched tiles): masks in the reference frame
(Xenium 5K µm) at a 10 µm raster, from the 5K H&E and the 5K cell-by-gene matrix, and the same masks transferred into every
section's own frame (inverse affine).

Labels: 0 outside tissue, 1 tumor, 2 stroma, 3 immune aggregate, 4 necrosis, 5 vessel.
  * tissue      : H&E stain (hematoxylin + eosin optical density) above threshold, holes and gaps > 1 px stay "outside"
  * cell classes: per Xenium 5K cell, fraction of counts in each marker set (below), z-scored; class = argmax if z > 0.5;
                  class densities (cells/mm^2) rasterized at 10 µm and smoothed (Gaussian sigma 15 µm)
  * necrosis    : tissue with eosin but cell density < `necrosis_frac` x tissue median and low hematoxylin
  * vessel      : endothelial density >= `vessel_density` cells/mm^2 and endothelial fraction >= `vessel_frac`
  * immune      : immune density >= `immune_density` cells/mm^2 and immune fraction >= `immune_frac` (aggregates only;
                  scattered immune cells inherit the surrounding compartment)
  * tumor/stroma: larger of the two smoothed class densities (sigma 20 µm) on the remaining tissue
Marker sets are restricted to genes on the 5K panel (COL1A1/DCN/ACTA2/KRT8/PTPRC/VWF are not on it).

Usage: masks_ovff.py [--transforms results/ovff/registration/transforms.json] [--out data/ovff/masks]
Writes data/ovff/masks/regions_10um*.npz (read by registration.load_region_mask) and
results/ovff/registration/region_areas.csv.
"""
from __future__ import annotations

import argparse, json, time, resource
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tifffile
from scipy.ndimage import gaussian_filter, label as cc_label
from skimage import morphology
from skimage.color import rgb2hed

REPO = Path(__file__).resolve().parents[1]
from registration import REFERENCE, apply_affine, load_transforms, scale_matrix, to_reference

NAMES = ["outside", "tumor", "stroma", "immune", "necrosis", "vessel"]
COLORS = {"outside": "#e8e8e4", "tumor": "#2a78d6", "stroma": "#eb6834", "immune": "#1baf7a", "necrosis": "#eda100", "vessel": "#e87ba4"}
MARKERS = dict(
    tumor=["EPCAM", "PAX8", "WT1", "MUC16", "MSLN", "SOX17", "FOLR1"],
    stroma=["PDGFRB", "PDGFRA", "COL5A1", "FAP", "THY1", "POSTN"],
    immune=["CD3E", "CD2", "CD4", "CD8A", "MS4A1", "CD79A", "CD19", "CD68", "CD163", "CD14", "MZB1", "CXCR4", "CCR7"],
    vessel=["PECAM1", "CDH5", "KDR", "ENG", "CLDN5", "PLVAP", "TIE1", "EGFL7"],
)
RAW = REPO / "data/raw/ovff_xenium_5k"
XENIUM_HE_UM_PER_PX = 0.13687


def read_cells():
    with h5py.File(RAW / "cell_feature_matrix.h5") as f:
        g = f["matrix"]
        names = g["features/name"][:].astype(str); ftype = g["features/feature_type"][:].astype(str)
        X = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g["shape"][:]))  # features x cells
        barcodes = g["barcodes"][:].astype(str)
    X = X.T.tocsr()  # cells x features
    gene = ftype == "Gene Expression"
    cells = pd.read_parquet(RAW / "cells.parquet", columns=["cell_id", "x_centroid", "y_centroid", "transcript_counts"]).set_index("cell_id").reindex(barcodes)
    return X[:, gene], names[gene], cells


def class_scores(X, names, min_counts=20):
    tot = np.asarray(X.sum(1)).ravel()
    idx = {n: i for i, n in enumerate(names)}
    S = {}
    for k, genes in MARKERS.items():
        cols = [idx[g] for g in genes if g in idx]
        frac = np.asarray(X[:, cols].sum(1)).ravel() / np.maximum(tot, 1)
        ok = tot >= min_counts
        z = (frac - frac[ok].mean()) / (frac[ok].std() + 1e-12)
        S[k] = z
    Z = np.stack([S[k] for k in MARKERS], 1)
    cls = np.full(len(tot), -1)
    good = tot >= min_counts
    am = Z.argmax(1); cls[good & (Z.max(1) > 0.5)] = am[good & (Z.max(1) > 0.5)]
    return cls, Z, tot


def rasterize(xy, origin, px, shape, sigma_um):
    col = np.floor((xy[:, 0] - origin[0]) / px).astype(int); row = np.floor((xy[:, 1] - origin[1]) / px).astype(int)
    ok = (col >= 0) & (row >= 0) & (col < shape[1]) & (row < shape[0])
    im = np.zeros(shape, np.float32)
    np.add.at(im, (row[ok], col[ok]), 1.0)
    if sigma_um > 0:
        im = gaussian_filter(im, sigma_um / px)
    return im / (px / 1000.0) ** 2  # cells per mm^2


def he_features(A_ref_img2um, origin, px, shape, level_um=4.4):
    """Warp the reference H&E into the 10 µm raster; return (rgb, hematoxylin OD, eosin OD)."""
    with tifffile.TiffFile(RAW / "he_image.ome.tif") as t:
        ser = t.series[0]
        ums = [XENIUM_HE_UM_PER_PX * 2 ** k for k in range(len(ser.levels))]
        k = max(i for i, u in enumerate(ums) if u <= level_um * 1.01)
        img = ser.levels[k].asarray()
    f = 2 ** k
    M_work2full = scale_matrix(f, f, (f - 1) / 2, (f - 1) / 2)
    M_um2raster = scale_matrix(1 / px, 1 / px, -origin[0] / px, -origin[1] / px)
    M = M_um2raster @ A_ref_img2um @ M_work2full
    # pixel-center convention: raster index i covers [origin + i*px, origin + (i+1)*px); warpAffine maps px centers, shift by 0.5
    Mc = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1.0]]) @ M
    rgb = cv2.warpAffine(img, Mc[:2].astype(np.float64), (shape[1], shape[0]), flags=cv2.INTER_AREA, borderValue=(255, 255, 255))
    hed = rgb2hed(rgb)
    return rgb, hed[..., 0], hed[..., 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transforms", default=str(REPO / "results/ovff/registration/transforms.json"))
    ap.add_argument("--out", default=str(REPO / "data/ovff/masks"))
    ap.add_argument("--results", default=str(REPO / "results/ovff/registration"))
    ap.add_argument("--fig", default=str(REPO / "figures/quicklook/ovff_regions.png"))
    ap.add_argument("--px", type=float, default=10.0)
    ap.add_argument("--necrosis-frac", type=float, default=0.25)
    ap.add_argument("--vessel-density", type=float, default=1500.0)
    ap.add_argument("--vessel-frac", type=float, default=0.25)
    ap.add_argument("--immune-density", type=float, default=2000.0)
    ap.add_argument("--immune-frac", type=float, default=0.4)
    ap.add_argument("--stroma-tumor-frac", type=float, default=0.3, help="tumor-cell density below this x tissue median -> stroma")
    ap.add_argument("--nonrigid", action="store_true", help="transfer masks with the kept TPS instead of the affine")
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); res = Path(args.results); res.mkdir(parents=True, exist_ok=True)
    T = load_transforms(args.transforms)
    A_ref = np.asarray(T["reference_image_px_to_um"])
    px = args.px

    X, names, cells = read_cells()
    cls, Z, tot = class_scores(X, names)
    xy = cells[["x_centroid", "y_centroid"]].to_numpy(float)
    print(f"cells {len(xy)}, class counts {dict(zip(['other'] + list(MARKERS), np.bincount(cls + 1, minlength=5)))}", flush=True)

    # raster covering the reference H&E and cells (origin at 0,0; the 5K frame is positive)
    he_corners = apply_affine(A_ref, np.array([[0, 0], [51163, 0], [0, 54232], [51163, 54232]], float))
    xmax = max(he_corners[:, 0].max(), xy[:, 0].max()); ymax = max(he_corners[:, 1].max(), xy[:, 1].max())
    origin = (0.0, 0.0); shape = (int(np.ceil(ymax / px)) + 2, int(np.ceil(xmax / px)) + 2)
    print(f"raster {shape} at {px} um, origin {origin}", flush=True)

    rgb, hema, eos = he_features(A_ref, origin, px, shape)
    stain = hema + eos
    thr = max(0.04, float(np.percentile(stain, 60)) * 0.3)
    tissue = stain > thr
    tissue = morphology.remove_small_objects(tissue, 25)   # < 2500 um^2 specks
    tissue = morphology.remove_small_holes(tissue, 4)      # holes <= 400 um^2 are filled; larger holes stay outside
    dens_all = rasterize(xy, origin, px, shape, 15)
    dens = {k: rasterize(xy[cls == i], origin, px, shape, 15) for i, k in enumerate(MARKERS)}
    dens20 = {k: rasterize(xy[cls == i], origin, px, shape, 20) for i, k in enumerate(MARKERS)}
    med = float(np.median(dens_all[tissue]))
    hema_med = float(np.median(hema[tissue]))

    lab = np.zeros(shape, np.uint8)
    tum_med = float(np.median(dens20["tumor"][tissue & (dens20["tumor"] > 0)]))
    ts = tissue & (dens20["tumor"] >= dens20["stroma"]) & (dens20["tumor"] >= args.stroma_tumor_frac * tum_med); lab[ts] = 1
    lab[tissue & ~ts] = 2   # stroma: more stromal than tumor cells, or few tumor cells (the 5K stroma markers are weak)
    frac_imm = dens["immune"] / np.maximum(dens_all, 1e-6); frac_ves = dens["vessel"] / np.maximum(dens_all, 1e-6)
    imm = tissue & (dens["immune"] >= args.immune_density) & (frac_imm >= args.immune_frac)
    imm = morphology.remove_small_objects(imm, 4)
    lab[imm] = 3
    nec = tissue & (dens_all < args.necrosis_frac * med) & (hema < hema_med) & (eos > np.percentile(eos[tissue], 20))
    nec = morphology.binary_opening(nec, morphology.disk(1)); nec = morphology.remove_small_objects(nec, 9)
    lab[nec] = 4
    ves = tissue & (dens["vessel"] >= args.vessel_density) & (frac_ves >= args.vessel_frac)
    ves = morphology.remove_small_objects(ves, 2)
    lab[ves] = 5
    lab[~tissue] = 0

    areas = {n: float((lab == i).sum() * (px / 1000) ** 2) for i, n in enumerate(NAMES)}
    tissue_mm2 = sum(v for k, v in areas.items() if k != "outside")
    cells_lab = lab[np.clip(((xy[:, 1] - origin[1]) / px).astype(int), 0, shape[0] - 1), np.clip(((xy[:, 0] - origin[0]) / px).astype(int), 0, shape[1] - 1)]
    tab = pd.DataFrame(dict(label=range(len(NAMES)), region=NAMES, area_mm2=[areas[n] for n in NAMES],
                            frac_of_tissue=[areas[n] / tissue_mm2 if n != "outside" else np.nan for n in NAMES],
                            n_cells=[int((cells_lab == i).sum()) for i in range(len(NAMES))]))
    tab.to_csv(res / "region_areas.csv", index=False)
    print(tab.to_string(index=False), flush=True)

    meta = dict(frame=f"{REFERENCE} um", pixel_um=px, origin_um=list(origin), names=NAMES, markers=MARKERS,
                thresholds=dict(stain_od=thr, cell_density_median_per_mm2=med, hematoxylin_median=hema_med, tumor_density_median_per_mm2=tum_med, **{k: v for k, v in vars(args).items() if k not in ("transforms", "out", "results", "fig")}),
                cell_classes=dict(zip(["other"] + list(MARKERS), [int(v) for v in np.bincount(cls + 1, minlength=5)])))
    np.savez_compressed(out / "regions_10um.npz", labels=lab, origin_um=np.array(origin), pixel_um=px, names=np.array(NAMES, dtype=object),
                        frame=meta["frame"], cell_density_per_mm2=dens_all.astype(np.float32), hematoxylin_od=hema.astype(np.float32),
                        tissue=tissue, meta=json.dumps(meta))
    # per-cell class + region
    pd.DataFrame(dict(cell_id=cells.index, x=xy[:, 0], y=xy[:, 1], marker_class=np.array(["other"] + list(MARKERS))[cls + 1],
                      region=np.array(NAMES)[cells_lab])).to_parquet(out / "xenium_5k_cell_regions.parquet", index=False)

    # transfer into each section's own frame: raster over that section's bbox (mapped from the reference raster extent), nearest label
    for sec, t in T["sections"].items():
        if sec == REFERENCE:
            continue
        Minv = np.linalg.inv(np.asarray(t["um_to_reference_um"]))
        corners = apply_affine(Minv, np.array([[0, 0], [shape[1] * px, 0], [0, shape[0] * px], [shape[1] * px, shape[0] * px]], float))
        o = (float(np.floor(corners[:, 0].min() / px) * px), float(np.floor(corners[:, 1].min() / px) * px))
        sh = (int(np.ceil((corners[:, 1].max() - o[1]) / px)) + 1, int(np.ceil((corners[:, 0].max() - o[0]) / px)) + 1)
        yy, xx = np.mgrid[0:sh[0], 0:sh[1]]
        cen = np.stack([o[0] + (xx.ravel() + 0.5) * px, o[1] + (yy.ravel() + 0.5) * px], 1)
        ref_xy = to_reference(cen, sec, args.transforms, nonrigid=args.nonrigid) if args.nonrigid else apply_affine(np.asarray(t["um_to_reference_um"]), cen)
        c = np.floor((ref_xy[:, 0] - origin[0]) / px).astype(int); r = np.floor((ref_xy[:, 1] - origin[1]) / px).astype(int)
        ok = (c >= 0) & (r >= 0) & (c < shape[1]) & (r < shape[0])
        l2 = np.zeros(len(cen), np.uint8); l2[ok] = lab[r[ok], c[ok]]
        np.savez_compressed(out / f"regions_10um_{sec}.npz", labels=l2.reshape(sh), origin_um=np.array(o), pixel_um=px, names=np.array(NAMES, dtype=object),
                            frame=f"{sec} own um frame ({t['own_frame']['kind']}; provisional={t['own_frame'].get('provisional')}; "
                                  f"transfer={'tps' if args.nonrigid and t.get('nonrigid', {}).get('kept') else 'affine'})", meta=json.dumps(meta))
        print(f"  {sec}: mask {sh} origin {o}", flush=True)

    # quick-look
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    ax[0].imshow(rgb); ax[0].set_title(f"{REFERENCE} H&E in the reference frame ({px:.0f} um raster)", fontsize=9)
    cmap = ListedColormap([COLORS[n] for n in NAMES])
    ax[1].imshow(lab, cmap=cmap, vmin=-0.5, vmax=len(NAMES) - 0.5, interpolation="nearest")
    ax[1].legend(handles=[Patch(color=COLORS[n], label=f"{n} {areas[n]:.2f} mm2") for n in NAMES], loc="lower right", fontsize=7, frameon=True)
    ax[1].set_title("regions: H&E stain/nuclei + Xenium 5K marker classes", fontsize=9)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
        a.plot([20, 20 + 1000 / px], [shape[0] - 20, shape[0] - 20], color="k", lw=3); a.text(20, shape[0] - 30, "1 mm", fontsize=8)
    fig.savefig(args.fig, dpi=150, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(1, 5, figsize=(25, 5.5))
    for a, (k, d) in zip(ax, [("all cells", dens_all)] + [(k, dens[k]) for k in MARKERS]):
        im = a.imshow(np.where(tissue, d, np.nan), cmap="Blues", vmin=0, vmax=np.nanpercentile(d[tissue], 99)); a.set_title(f"{k} density (cells/mm2, sigma 15 um)", fontsize=9)
        a.set_xticks([]); a.set_yticks([]); plt.colorbar(im, ax=a, fraction=0.04)
    fig.savefig(str(args.fig).replace(".png", "_classes.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"done in {time.time() - t0:.0f}s, max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")


if __name__ == "__main__":
    main()
