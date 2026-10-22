#!/usr/bin/env python
"""Register the H&E images of the serial FF ovarian sections into the Xenium 5K frame (results/ovff/registration/).

Per moving section (ovff_xenium_v1, ovff_hd_probe_deep [= _min], ovff_hd3p_min):
  1. load H&E at a working resolution (~1.1 µm/px; pyramid level of the OME-TIFF, block-mean of the BigTIFF, or the
     Space Ranger tissue_hires png if the microscope image is not on disk); stain-density image d = 255 - gray, masked;
  2. coarse rigid at ~8.8 µm/px: exhaustive rotation search (+ mirror) with phase correlation for the translation, scored by
     the masked Pearson correlation of the two density images; scale fixed at 1 (both pixel sizes are known from metadata);
  3. affine refinement at ~4.4 µm/px with ECC (cv2.findTransformECC, MOTION_AFFINE);
  4. landmarks: NCC block matching, hierarchical: (A) 4.4 µm/px, 2x`block_um` blocks, +-4x`search_um` around the ECC affine;
     (B) working resolution, `block_um` blocks within +-`search_um` of a TPS pre-warp fitted on the (A) blocks, two passes; least-squares affine on the accepted blocks (RANSAC only for gross outliers > `gross_um`);
  5. landmark error: `n_rep` spatially stratified hold-outs (k-means on block positions, ~`n_hold` blocks each); affine and
     regularized thin-plate spline fitted on the remaining blocks; median / p90 hold-out residual in µm;
  6. compose with each assay's image -> µm mapping; write transforms.json, overlays and landmark inspection panels.

Usage: register_ovff.py [--work-um 1.1] [--coarse-um 8.8] [--mid-um 4.4] [--sections a,b,c] [--out DIR] [--fig DIR]
       [--cache DIR] [--tag proto]
"""
from __future__ import annotations

import argparse, json, time, resource
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image
from skimage import measure, morphology, transform as sktf
from skimage.filters import threshold_otsu
from skimage.registration import phase_cross_correlation

REPO = Path(__file__).resolve().parents[1]
from registration import (REFERENCE, TPS, XENIUM_MORPHOLOGY_UM_PER_PX, apply_affine, decompose_affine, scale_matrix,
                          visium_hd_px_to_um, xenium_he_to_um)

Image.MAX_IMAGE_PIXELS = None
XENIUM_HE_UM_PER_PX = 0.13687  # OME PhysicalSizeX of both Xenium H&E OME-TIFFs (0.13687470 / 0.13687519)

SECTIONS = {
    "ovff_xenium_5k": dict(kind="xenium", image=REPO / "data/raw/ovff_xenium_5k/he_image.ome.tif",
                           alignment=REPO / "data/raw/ovff_xenium_5k/he_imagealignment.csv"),
    "ovff_xenium_v1": dict(kind="xenium", image=REPO / "data/raw/ovff_xenium_v1/he_image.ome.tif",
                           alignment=REPO / "data/raw/ovff_xenium_v1/he_imagealignment.csv"),
    "ovff_hd_probe_deep": dict(kind="visium_hd", image=REPO / "data/archives/ovff_hd_probe_deep/tissue_image.btf",
                               hires=REPO / "data/raw/ovff_hd_probe_deep/spatial/tissue_hires_image.png",
                               scalefactors=REPO / "data/raw/ovff_hd_probe_deep/binned_outputs/square_002um/spatial/scalefactors_json.json"),
    "ovff_hd3p_min": dict(kind="visium_hd_pending", image=REPO / "data/archives/ovff_hd3p_min/tissue_image.btf",
                          ome_um_per_px=0.27381,  # OME PhysicalSizeX/Y of the btf; provisional until outs/spatial exists
                          spaceranger_outs=REPO / "data/spaceranger/ovff_hd3p_min/ovff_hd3p_min/outs/segmented_outputs/spatial"),  # SR 4.1: outs/spatial has no scalefactors
}


# ----------------------------------------------------------------------------- image loading
def block_mean(img, f):
    h, w = (img.shape[0] // f) * f, (img.shape[1] // f) * f
    im = img[:h, :w].reshape(h // f, f, w // f, f, -1).mean((1, 3))
    return im.astype(np.uint8)


def load_work_image(section: str, target_um: float, cache: Path | None = None):
    """Return (rgb uint8, um_per_px, M_work2full, source) where M_work2full maps work px -> full-res image px (pixel centers)."""
    s = SECTIONS[section]
    cf = (cache / f"{section}_{target_um:.3f}.npz") if cache else None
    if cf and cf.exists():
        z = np.load(cf, allow_pickle=True)
        print(f"  [{section}] cached {cf.name}: {z['img'].shape} at {float(z['um']):.4f} um/px", flush=True)
        return z["img"], float(z["um"]), z["M"], str(z["src"])
    t0 = time.time()
    if s["kind"] == "xenium":
        with tifffile.TiffFile(s["image"]) as t:
            ser = t.series[0]
            ums = [XENIUM_HE_UM_PER_PX * 2 ** k for k in range(len(ser.levels))]
            k = max(i for i, u in enumerate(ums) if u <= target_um * 1.01)
            img = ser.levels[k].asarray()
        f = 2 ** k
        um = XENIUM_HE_UM_PER_PX * f
        M = scale_matrix(f, f, (f - 1) / 2, (f - 1) / 2)
        src = f"{s['image'].name} level {k}"
    else:
        if s["image"].exists():
            img = tifffile.imread(s["image"])
            full_um = json.load(open(s["scalefactors"]))["microns_per_pixel"] if s["kind"] == "visium_hd" else s["ome_um_per_px"]
            f = max(1, int(round(target_um / full_um)))
            img = block_mean(img, f) if f > 1 else img
            um = full_um * f
            M = scale_matrix(f, f, (f - 1) / 2, (f - 1) / 2)
            src = f"{s['image'].name} block-mean {f}"
        else:
            sf = json.load(open(s["scalefactors"]))
            img = np.asarray(Image.open(s["hires"]).convert("RGB"))
            sc = sf["tissue_hires_scalef"]
            um = sf["microns_per_pixel"] / sc
            M = scale_matrix(1 / sc, 1 / sc, 0.5 / sc - 0.5, 0.5 / sc - 0.5)  # x_full = (x_h + 0.5)/sc - 0.5
            src = f"{s['hires'].name} (scale {sc})"
    print(f"  [{section}] {src}: {img.shape} at {um:.4f} um/px ({time.time() - t0:.1f}s)", flush=True)
    if cf:
        cf.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cf, img=img, um=um, M=M, src=src)
    return img, um, M, src


def image_px_to_um(section: str) -> tuple[np.ndarray, dict]:
    """Full-res image px -> section's own µm frame, and a description of that frame."""
    s = SECTIONS[section]
    if s["kind"] == "xenium":
        return xenium_he_to_um(s["alignment"]), dict(kind="xenium_transcript_um", alignment=str(s["alignment"].relative_to(REPO)),
                                                    morphology_um_per_px=XENIUM_MORPHOLOGY_UM_PER_PX, provisional=False)
    if s["kind"] == "visium_hd":
        mpp = json.load(open(s["scalefactors"]))["microns_per_pixel"]
        return visium_hd_px_to_um(mpp), dict(kind="fullres_image_px_times_mpp", microns_per_pixel=mpp,
                                             source=str(s["scalefactors"].relative_to(REPO)), provisional=False)
    outs = s["spaceranger_outs"] / "scalefactors_json.json"
    if outs.exists():
        mpp = json.load(open(outs))["microns_per_pixel"]
        return visium_hd_px_to_um(mpp), dict(kind="fullres_image_px_times_mpp", microns_per_pixel=mpp, source=str(outs.relative_to(REPO)), provisional=False)
    mpp = s["ome_um_per_px"]
    return visium_hd_px_to_um(mpp), dict(kind="fullres_image_px_times_mpp", microns_per_pixel=mpp, source="OME PhysicalSizeX of the btf",
                                         provisional=True, note="replace with Space Ranger outs/spatial/scalefactors_json.json microns_per_pixel "
                                                                "(registration.update_hd3p_transform) once the pipestance finishes")


# ----------------------------------------------------------------------------- masks, density images
def tissue_mask(rgb, um_per_px, min_area_um2=2e5):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(float); val = hsv[..., 2].astype(float)
    stained = (sat > min(40, max(15, threshold_otsu(sat) * 0.6))) & (val < 235)
    r = max(1, int(round(8 / um_per_px)))
    m = morphology.binary_closing(stained, morphology.disk(r))
    m = morphology.remove_small_objects(m, int(min_area_um2 / um_per_px ** 2))
    return m


def density(rgb, mask, um_per_px, sigma_um=0.0):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    d = (255.0 - g) * mask
    if sigma_um > 0:
        s = sigma_um / um_per_px
        d = cv2.GaussianBlur(d, (0, 0), s)
    return d


def warp(img, M, shape, nearest=False, border=0):
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(img, np.asarray(M[:2], np.float64), (shape[1], shape[0]), flags=flags, borderValue=border)


def masked_corr(a, b, m):
    a = a[m]; b = b[m]
    if len(a) < 100:
        return -1.0
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def score(M, dm, dr, mm, mr):
    """(Pearson correlation of the density images within the mask intersection, mask IoU)."""
    w = warp(dm, M, dr.shape); wm = warp(mm.astype(np.uint8), M, dr.shape, nearest=True) > 0
    return masked_corr(w, dr, wm & mr), float((wm & mr).sum() / max(1, (wm | mr).sum()))


# ----------------------------------------------------------------------------- coarse rigid, ECC affine
def rot_matrix(theta_deg, center, flip=False):
    t = np.radians(theta_deg); c, s = np.cos(t), np.sin(t)
    R = np.array([[c, -s], [s, c]])
    if flip:
        R = R @ np.array([[-1.0, 0.0], [0.0, 1.0]])
    M = np.eye(3); M[:2, :2] = R; M[:2, 2] = center - R @ center
    return M


def rigid_search(dm, dr, mm, mr, um, angles, flips=(False, True)):
    """Exhaustive rotation (+ mirror) with phase-correlation translation on a common padded canvas."""
    H = max(dm.shape[0], dr.shape[0]) * 2; W = max(dm.shape[1], dr.shape[1]) * 2
    canvas_r = np.zeros((H, W), np.float32); canvas_r[:dr.shape[0], :dr.shape[1]] = dr
    center = np.array([dm.shape[1] / 2, dm.shape[0] / 2])
    best = (None, -2, None)
    for flip in flips:
        for th in angles:
            R = rot_matrix(th, center, flip)
            T = np.eye(3); T[:2, 2] = [W / 2 - center[0], H / 2 - center[1]]
            rot = warp(dm, T @ R, (H, W))
            shift, _, _ = phase_cross_correlation(canvas_r, rot, normalization=None)
            S = np.eye(3); S[:2, 2] = [shift[1], shift[0]]
            M = S @ T @ R
            c, iou = score(M, dm, dr, mm, mr)
            if c * iou > best[1]:
                best = (M, c * iou, dict(theta=float(th), flip=bool(flip), corr=c, iou=iou, score=c * iou))
    return best


def coarse_rigid(mov_rgb, ref_rgb, mov_mask, ref_mask, um):
    dm = density(mov_rgb, mov_mask, um, 30); dr = density(ref_rgb, ref_mask, um, 30)
    M, c, info = rigid_search(dm, dr, mov_mask, ref_mask, um, np.arange(-180, 180, 2.0))
    M2, c2, info2 = rigid_search(dm, dr, mov_mask, ref_mask, um, info["theta"] + np.arange(-2, 2.01, 0.25), flips=(info["flip"],))
    if c2 > c:
        M, info = M2, info2
    info["method"] = "rotation sweep 2 deg (+mirror) + phase correlation, refined 0.25 deg; scale 1"
    return M, info


def ecc_affine(dm, dr, M0, um, n_iter=300, eps=1e-6, gauss_um=20):
    """Refine forward affine M0 (mov px -> ref px) by ECC on density images (float32 normalized)."""
    a = dm / max(1.0, dm.max()); b = dr / max(1.0, dr.max())
    W = np.linalg.inv(M0)[:2].astype(np.float32)    # ECC warp maps template(ref) coords -> input(mov) coords
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, eps)
    try:
        cc, W = cv2.findTransformECC(b.astype(np.float32), a.astype(np.float32), W, cv2.MOTION_AFFINE, crit, None, max(1, int(gauss_um / um)) | 1)
    except cv2.error as e:
        return M0, dict(ecc=None, error=str(e)[:200])
    M = np.linalg.inv(np.vstack([W, [0, 0, 1]]))
    return M, dict(ecc=float(cc))


# ----------------------------------------------------------------------------- block matching landmarks
def bilinear_map(mp, pts):
    """Evaluate a dense backward map mp (H, W, 2) at fractional (x, y) points."""
    H, W = mp.shape[:2]
    x = np.clip(pts[:, 0], 0, W - 1.001); y = np.clip(pts[:, 1], 0, H - 1.001)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int); fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    return (mp[y0, x0] * (1 - fx) * (1 - fy) + mp[y0, x0 + 1] * fx * (1 - fy) + mp[y0 + 1, x0] * (1 - fx) * fy + mp[y0 + 1, x0 + 1] * fx * fy)


def affine_map(M, shape):
    """Dense backward map (ref px -> mov px) of the forward affine M, as (H, W, 2) float32."""
    Minv = np.linalg.inv(M)
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)
    return apply_affine(Minv, pts).reshape(shape[0], shape[1], 2).astype(np.float32)


def tps_backward_map(src, dst, shape, lam, step=8):
    """Dense backward map ref px -> mov px from a TPS fitted on (dst -> src) pairs, evaluated on a step grid and
    bilinearly interpolated (exact at grid nodes)."""
    tps = TPS(dst, src, lam=lam)
    ys = np.arange(0, shape[0] + step, step); xs = np.arange(0, shape[1] + step, step)
    gx, gy = np.meshgrid(xs, ys)
    m = tps(np.stack([gx.ravel(), gy.ravel()], 1)).reshape(len(ys), len(xs), 2).astype(np.float32)
    out = np.empty((shape[0], shape[1], 2), np.float32)
    for r0 in range(0, shape[0], 512):
        r1 = min(shape[0], r0 + 512)
        yy, xx = np.mgrid[r0:r1, 0:shape[1]]
        pts = np.stack([xx.ravel() / step, yy.ravel() / step], 1)
        out[r0:r1] = bilinear_map(m, pts).reshape(r1 - r0, shape[1], 2)
    return out


def block_match(dm, dr, mr, bmap, um, block_um=200, stride_um=200, search_um=40, min_ncc=0.45, min_std=8.0):
    """Reference blocks vs. the moving density image warped by the backward map `bmap` (ref px -> mov px).
    Returns (mov_pts_work_px, ref_pts_work_px, ncc) for accepted blocks; the moving point is bmap(ref center + shift)."""
    wm = cv2.remap(dm, bmap[..., 0], bmap[..., 1], cv2.INTER_LINEAR, borderValue=0)
    B = int(round(block_um / um)); S = int(round(stride_um / um)); R = int(round(search_um / um))
    H, W = dr.shape
    src, dst, nccs = [], [], []
    for y in range(R, H - B - R, S):
        for x in range(R, W - B - R, S):
            if mr[y:y + B, x:x + B].mean() < 0.6:
                continue
            tpl = dr[y:y + B, x:x + B]
            if tpl.std() < min_std:
                continue
            win = wm[y - R:y + B + R, x - R:x + B + R]
            if (win > 0).mean() < 0.5:
                continue
            res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if mx < min_ncc or loc[0] in (0, res.shape[1] - 1) or loc[1] in (0, res.shape[0] - 1):
                continue
            dx = dy = 0.0
            cx, cy = loc
            l, c, r = res[cy, cx - 1], res[cy, cx], res[cy, cx + 1]
            if (l - 2 * c + r) < 0: dx = 0.5 * (l - r) / (l - 2 * c + r)
            u, d = res[cy - 1, cx], res[cy + 1, cx]
            if (u - 2 * c + d) < 0: dy = 0.5 * (u - d) / (u - 2 * c + d)
            shift = np.array([cx + dx - R, cy + dy - R])   # where the reference block's content sits in the warped moving image
            p_ref = np.array([x + B / 2, y + B / 2])
            src.append(bilinear_map(bmap, (p_ref + shift)[None])[0]); dst.append(p_ref); nccs.append(float(mx))
    return np.array(src).reshape(-1, 2), np.array(dst).reshape(-1, 2), np.array(nccs)


def ransac_affine(src, dst, thresh_px, max_trials=5000):
    if len(src) < 6:
        return None, np.zeros(len(src), bool)
    model, inl = measure.ransac((src, dst), sktf.AffineTransform, min_samples=3, residual_threshold=thresh_px, max_trials=max_trials, rng=0)
    if model is None or inl is None or inl.sum() < 6:
        return None, np.zeros(len(src), bool)
    A = sktf.estimate_transform("affine", src[inl], dst[inl])   # least squares on inliers
    return A.params, inl


# ----------------------------------------------------------------------------- landmark error
def stratified_holdout(dst, n_hold, seed):
    from sklearn.cluster import KMeans
    n_hold = min(n_hold, len(dst) // 3)
    km = KMeans(n_clusters=n_hold, n_init=3, random_state=seed).fit(dst)
    rng = np.random.default_rng(seed)
    return np.array(sorted({int(rng.choice(np.nonzero(km.labels_ == k)[0])) for k in range(n_hold)}))


def landmark_errors(src, dst, um, n_hold=30, n_rep=5, tps_lam_um2=400.0):
    res_aff, res_tps, holds = [], [], []
    for rep in range(n_rep):
        h = stratified_holdout(dst, n_hold, seed=rep)
        fit = np.setdiff1d(np.arange(len(src)), h)
        A = sktf.estimate_transform("affine", src[fit], dst[fit])
        res_aff.append(np.linalg.norm(A(src[h]) - dst[h], axis=1) * um)
        tps = TPS(src[fit], dst[fit], lam=tps_lam_um2 / um ** 2)
        res_tps.append(np.linalg.norm(tps(src[h]) - dst[h], axis=1) * um)
        holds.append(h)
    ra = np.concatenate(res_aff); rt = np.concatenate(res_tps)
    stats = lambda r: dict(n=int(len(r)), median_um=float(np.median(r)), p90_um=float(np.percentile(r, 90)), mean_um=float(r.mean()), max_um=float(r.max()))
    return dict(affine=stats(ra), tps=stats(rt), per_rep_affine_median_um=[float(np.median(r)) for r in res_aff],
                per_rep_tps_median_um=[float(np.median(r)) for r in res_tps], n_hold_per_rep=[int(len(h)) for h in holds],
                tps_lambda_um2=tps_lam_um2), holds[0], res_aff[0]


# ----------------------------------------------------------------------------- figures
def overlay_figure(ref_rgb, mov_mask, ref_mask, M, um, out_png, title, pts=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    w = warp(mov_mask.astype(np.uint8), M, ref_rgb.shape[:2], nearest=True) > 0
    g = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)
    fig, ax = plt.subplots(figsize=(9, 9 * g.shape[0] / g.shape[1]))
    ax.imshow(g, cmap="gray", vmin=0, vmax=255)
    for c in measure.find_contours(w.astype(float), 0.5):
        if len(c) * um > 300: ax.plot(c[:, 1], c[:, 0], color="#d62728", lw=0.5)
    for c in measure.find_contours(ref_mask.astype(float), 0.5):
        if len(c) * um > 300: ax.plot(c[:, 1], c[:, 0], color="#1f77b4", lw=0.35, alpha=0.7)
    if pts is not None and len(pts):
        ax.plot(pts[:, 0], pts[:, 1], ".", color="#2ca02c", ms=2)
    ax.set_title(title, fontsize=8); ax.set_axis_off()
    sb = 1000 / um
    ax.plot([50, 50 + sb], [g.shape[0] - 60, g.shape[0] - 60], color="k", lw=3); ax.text(50, g.shape[0] - 90, "1 mm", fontsize=8)
    fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)


def landmark_panel(mov_rgb, ref_rgb, src, dst, hold, res, M, um, out_png, title, half_um=150):
    """Grid of held-out landmark patches: moving (warped with the affine) next to the reference patch, same window."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    warped = cv2.warpAffine(mov_rgb, np.asarray(M[:2], np.float64), (ref_rgb.shape[1], ref_rgb.shape[0]), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    h = int(half_um / um)
    n = len(hold); ncol = 6; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol * 2, figsize=(ncol * 2 * 1.6, nrow * 1.75))
    axes = np.atleast_2d(axes)
    for a in axes.ravel(): a.set_axis_off()
    for j, i in enumerate(hold):
        x, y = dst[i]; r, c = j // ncol, (j % ncol) * 2
        sl = (slice(max(0, int(y) - h), int(y) + h), slice(max(0, int(x) - h), int(x) + h))
        for k, im in enumerate((warped, ref_rgb)):
            ax = axes[r, c + k]; ax.imshow(im[sl]); ax.plot(h, h, "+", color="lime", ms=9, mew=1.2)
        axes[r, c].set_title(f"#{j} {res[j]:.0f} um", fontsize=6)
    fig.suptitle(title + f"  (left: moving warped by affine, right: reference; {2 * half_um} um patches; green + = same reference position)", fontsize=8)
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-um", type=float, default=1.1)
    ap.add_argument("--mid-um", type=float, default=4.4)
    ap.add_argument("--coarse-um", type=float, default=8.8)
    ap.add_argument("--sections", default="ovff_xenium_v1,ovff_hd_probe_deep,ovff_hd3p_min")
    ap.add_argument("--out", default=str(REPO / "results/ovff/registration"))
    ap.add_argument("--fig", default=str(REPO / "figures/quicklook"))
    ap.add_argument("--cache", default=str(REPO / "data/ovff/registration_cache"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--n-hold", type=int, default=30)
    ap.add_argument("--n-rep", type=int, default=5)
    ap.add_argument("--block-um", type=float, default=200)
    ap.add_argument("--search-um", type=float, default=40)
    ap.add_argument("--gross-um", type=float, default=60, help="RANSAC threshold for gross outliers among NCC-validated blocks")
    ap.add_argument("--min-ncc", type=float, default=0.35)
    ap.add_argument("--min-ncc-mid", type=float, default=0.3)
    ap.add_argument("--tps-lambda-um2", type=float, default=400.0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); fig = Path(args.fig); fig.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else None
    tag = f"_{args.tag}" if args.tag else ""
    t_all = time.time()

    print(f"== reference {REFERENCE}", flush=True)
    LV = dict(coarse=args.coarse_um, mid=args.mid_um, work=args.work_um)
    ref = {u: load_work_image(REFERENCE, LV[u], cache) for u in LV}
    ref_masks = {u: tissue_mask(ref[u][0], ref[u][1]) for u in ref}
    ref_dens = {u: density(ref[u][0], ref_masks[u], ref[u][1]) for u in ref}
    ref_w, um_w, Mw2f_ref, ref_src = ref['work']
    A_ref, ref_frame = image_px_to_um(REFERENCE)   # ref image px -> ref um
    result = dict(reference=REFERENCE, reference_frame=ref_frame, reference_image=ref_src,
                  reference_image_px_to_um=A_ref.tolist(), work_um_per_px=um_w,
                  convention="3x3 matrices act on column vectors [x, y, 1]; x = column (horizontal), y = row (vertical)",
                  parameters=vars(args),
                  sections={REFERENCE: dict(um_to_reference_um=np.eye(3).tolist(), image_px_to_reference_um=A_ref.tolist(),
                                            own_frame=ref_frame, image=ref_src, method="identity (reference)",
                                            tissue_area_mm2=float(ref_masks['work'].sum() * um_w ** 2 / 1e6))},
                  landmark_errors={}, timing={})

    for sec in args.sections.split(","):
        t0 = time.time(); print(f"== {sec}", flush=True)
        mov = {u: load_work_image(sec, LV[u], cache) for u in LV}
        mov_masks = {u: tissue_mask(mov[u][0], mov[u][1]) for u in mov}
        mov_dens = {u: density(mov[u][0], mov_masks[u], mov[u][1]) for u in mov}
        mov_w, umw, Mw2f_mov, mov_src = mov['work']
        # transfer of a work-px affine between resolutions: M_b = inv(F_ref_b) F_ref_a M_a inv(F_mov_a) F_mov_b
        def lift(M, ua, ub):
            return np.linalg.inv(ref[ub][2]) @ ref[ua][2] @ M @ np.linalg.inv(mov[ua][2]) @ mov[ub][2]
        uc, umid, uw = 'coarse', 'mid', 'work'
        M0, info_c = coarse_rigid(mov[uc][0], ref[uc][0], mov_masks[uc], ref_masks[uc], mov[uc][1])
        print(f"  coarse rigid: {info_c}", flush=True)
        # ECC affine at coarse then mid resolution
        M1, info_e1 = ecc_affine(density(mov[uc][0], mov_masks[uc], mov[uc][1], 8), density(ref[uc][0], ref_masks[uc], ref[uc][1], 8), M0, mov[uc][1])
        M1m = lift(M1, uc, umid)
        M2, info_e2 = ecc_affine(density(mov[umid][0], mov_masks[umid], mov[umid][1], 6), density(ref[umid][0], ref_masks[umid], ref[umid][1], 6), M1m, mov[umid][1])
        c_mid, iou_mid = score(M2, mov_dens[umid], ref_dens[umid], mov_masks[umid], ref_masks[umid])
        print(f"  ecc: coarse {info_e1} mid {info_e2}; corr {c_mid:.3f} iou {iou_mid:.3f}", flush=True)
        # hierarchical block matching: (A) mid res, large blocks, wide search, affine pre-warp; (B) work res, TPS pre-warp
        Mm = M2; bm = {}
        lam_mid = args.tps_lambda_um2 / mov["mid"][1] ** 2; lam_w = args.tps_lambda_um2 / um_w ** 2
        srcA, dstA, nccA = block_match(mov_dens[umid], ref_dens[umid], ref_masks[umid], affine_map(Mm, ref_dens[umid].shape), mov[umid][1],
                                       block_um=args.block_um * 2, stride_um=args.block_um, search_um=args.search_um * 4, min_ncc=args.min_ncc_mid)
        MA, inlA = ransac_affine(srcA, dstA, args.gross_um / mov[umid][1])
        bm["A_mid"] = dict(block_um=args.block_um * 2, search_um=args.search_um * 4, n_blocks=int(len(srcA)), n_inliers=int(inlA.sum()),
                           ncc_median=float(np.median(nccA)) if len(nccA) else None)
        print(f"  block A (mid): {bm['A_mid']}", flush=True)
        if MA is not None:
            Mm = MA
        Mw = lift(Mm, umid, uw)
        # pre-warp for the fine pass: TPS from the mid-level inliers (lifted to work px), else the affine
        if inlA.sum() >= 12:
            srcW = apply_affine(np.linalg.inv(mov[uw][2]) @ mov[umid][2], srcA[inlA]); dstW = apply_affine(np.linalg.inv(ref[uw][2]) @ ref[umid][2], dstA[inlA])
            bmap = tps_backward_map(srcW, dstW, ref_dens[uw].shape, lam_w)
        else:
            bmap = affine_map(Mw, ref_dens[uw].shape)
        for p, (b, s_) in enumerate(((args.block_um, args.search_um), (args.block_um * 0.75, args.search_um * 0.75))):
            src, dst, ncc = block_match(mov_dens[uw], ref_dens[uw], ref_masks[uw], bmap, um_w, block_um=b, stride_um=b * 0.75, search_um=s_, min_ncc=args.min_ncc)
            Ma, inl = ransac_affine(src, dst, args.gross_um / um_w)
            bm[f"B{p}_work"] = dict(block_um=b, search_um=s_, n_blocks=int(len(src)), n_inliers=int(inl.sum()), ncc_median=float(np.median(ncc)) if len(ncc) else None)
            print(f"  block B{p} (work): {bm[f'B{p}_work']}", flush=True)
            if Ma is not None:
                Mw = Ma
            if inl.sum() >= 12 and p == 0:
                bmap = tps_backward_map(src[inl], dst[inl], ref_dens[uw].shape, lam_w)
        c_w, iou_w = score(Mw, mov_dens[uw], ref_dens[uw], mov_masks[uw], ref_masks[uw])
        info_f = dict(**bm, corr=c_w, mask_iou=iou_w, gross_outlier_um=args.gross_um, ncc_quartiles=[float(q) for q in np.percentile(ncc[inl], [25, 50, 75])] if inl.sum() else None)
        src_i, dst_i = src[inl], dst[inl]
        if len(src_i) >= 12:
            errs, hold, res0 = landmark_errors(src_i, dst_i, um_w, n_hold=args.n_hold, n_rep=args.n_rep, tps_lam_um2=args.tps_lambda_um2)
            print(f"  landmarks: affine median {errs['affine']['median_um']:.1f} p90 {errs['affine']['p90_um']:.1f} | "
                  f"tps median {errs['tps']['median_um']:.1f} p90 {errs['tps']['p90_um']:.1f} (n={errs['affine']['n']})", flush=True)
            landmark_panel(mov_w, ref_w, src_i, dst_i, hold, res0, Mw, um_w, fig / f"ovff_landmarks_{sec}{tag}.png", f"{sec} -> {REFERENCE}")
        else:
            errs, hold = dict(affine=None, tps=None, note="too few accepted blocks"), np.array([], int)
        M_img = Mw2f_ref @ Mw @ np.linalg.inv(Mw2f_mov)     # full-res mov image px -> full-res ref image px
        A_mov, mov_frame = image_px_to_um(sec)
        M_img2ref = A_ref @ M_img                            # mov image px -> reference um
        M_um2ref = M_img2ref @ np.linalg.inv(A_mov)          # mov own um -> reference um
        # non-rigid: TPS in um frames on all validated blocks, kept only if the affine hold-out error exceeds 20 um
        nonrigid = dict(kept=False)
        if errs.get("affine"):
            src_um = apply_affine(A_mov @ Mw2f_mov, src_i)      # moving work px -> moving own um
            dst_um = apply_affine(A_ref @ Mw2f_ref, dst_i)
            fwd = TPS(src_um, dst_um, lam=args.tps_lambda_um2); rev = TPS(dst_um, src_um, lam=args.tps_lambda_um2)
            rt = np.linalg.norm(rev(fwd(src_um)) - src_um, axis=1)
            nonrigid = dict(kept=bool(errs["affine"]["median_um"] > 20.0), file=f"nonrigid_{sec}{tag}.json", n_control=int(len(src_um)),
                            lambda_um2=args.tps_lambda_um2, holdout=errs["tps"], roundtrip_um=dict(median=float(np.median(rt)), p90=float(np.percentile(rt, 90))),
                            note="TPS own um -> reference um fitted on all validated blocks; hold-out numbers from landmark_errors.tps")
            with open(out / nonrigid["file"], "w") as f:
                json.dump(dict(section=sec, reference=REFERENCE, to_reference=fwd.to_dict(), from_reference=rev.to_dict(),
                               control_src_um=src_um.tolist(), control_dst_um=dst_um.tolist()), f)
        overlay_figure(ref_w, mov_masks[uw], ref_masks[uw], Mw, um_w, fig / f"ovff_reg_{sec}{tag}.png",
                       f"{sec} (red: moved tissue outline; green: landmark blocks) on {REFERENCE} H&E (blue outline); affine, "
                       f"{int(inl.sum())} blocks, corr {c_w:.3f}, mask IoU {iou_w:.3f}", pts=dst_i)
        result["sections"][sec] = dict(
            um_to_reference_um=M_um2ref.tolist(), image_px_to_reference_um=M_img2ref.tolist(),
            image_px_to_reference_image_px=M_img.tolist(), image_px_to_um=A_mov.tolist(), own_frame=mov_frame, image=mov_src,
            method=f"rigid sweep @ {mov[uc][1]:.2f} um/px -> ECC affine @ {mov[umid][1]:.2f} um/px -> NCC block matching + RANSAC affine @ {um_w:.2f} um/px",
            coarse=info_c, ecc=dict(coarse=info_e1, mid=info_e2, corr_mid=c_mid, iou_mid=iou_mid), fine=info_f,
            decomposition_um_to_ref=decompose_affine(M_um2ref), decomposition_image_px_to_ref_image_px=decompose_affine(M_img),
            tissue_area_mm2=float(mov_masks[uw].sum() * umw ** 2 / 1e6), landmark_errors=errs, nonrigid=nonrigid)
        result["landmark_errors"][sec] = errs
        np.savez(out / f"matches_{sec}{tag}.npz", src_work_px=src_i, dst_work_px=dst_i, um_per_px=um_w, hold0=hold,
                 M_work=Mw, Mw2f_mov=Mw2f_mov, Mw2f_ref=Mw2f_ref)
        result["timing"][sec] = dict(seconds=time.time() - t0)
        print(f"  {sec} done in {time.time() - t0:.0f}s; um->ref: {decompose_affine(M_um2ref)}", flush=True)

    result["timing"]["total_seconds"] = time.time() - t_all
    result["timing"]["max_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    with open(out / f"transforms{tag}.json", "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {out / f'transforms{tag}.json'}; total {result['timing']['total_seconds']:.0f}s, max RSS {result['timing']['max_rss_gb']:.1f} GB")


if __name__ == "__main__":
    main()
