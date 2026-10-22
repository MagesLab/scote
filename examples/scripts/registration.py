"""Common frame for the serial sections of the 10x FF ovarian 5-assay set.

Reference frame: the Xenium Prime 5K section's own micron coordinates (`ovff_xenium_5k`, transcripts x/y in µm).
A transforms JSON file (see `load_transforms`) stores, per section,
3x3 affine matrices acting on column vectors [x, y, 1] (x = column / horizontal, y = row / vertical):

  * ``um_to_reference_um``      section's own µm frame -> reference µm   (what later stages use)
  * ``image_px_to_reference_um``  section's full-resolution H&E pixel -> reference µm
  * ``own_frame``: how the own µm frame is defined (Xenium: transcript µm; Visium HD: full-res image px x microns_per_pixel,
    i.e. the coordinates written by scote.io.convert_visium_hd).

Conventions:
  * Xenium ``he_imagealignment.csv`` (Xenium Explorer export) maps H&E full-res px -> *morphology image px* (0.2125 µm/px);
    multiply by the morphology pixel size to get µm. Its scale (0.6445) equals 0.1369 µm (H&E px) / 0.2125 µm (morphology px).
  * Visium HD ``tissue_positions.parquet`` gives bin centers in full-res image px; µm = px * scalefactors ``microns_per_pixel``.

  * ``nonrigid`` (optional): a regularized thin-plate spline own µm -> reference µm (and the reverse fit) on the validated
    landmark blocks, kept only where the affine hold-out error exceeds 20 µm; `to_reference(..., nonrigid=True)` uses it.
    The affine is the default; the TPS corrects the residual tissue distortion (tens of µm) of serial sectioning.

Registered coordinates are for region / tile level use only (sections are 5-10 µm apart; cells differ), never for cell matching.

Default locations (used when no ``path`` / ``masks_dir`` is passed), relative to the working directory:
``results/ovff/registration/transforms.json`` (override with the environment variable ``SCOTE_TRANSFORMS``) and
``data/ovff/masks`` (override with ``SCOTE_MASKS_DIR``), the layout of the preprint examples.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

REFERENCE = "ovff_xenium_5k"
XENIUM_MORPHOLOGY_UM_PER_PX = 0.2125
DEFAULT_TRANSFORMS = Path(os.environ.get("SCOTE_TRANSFORMS", "results/ovff/registration/transforms.json"))
DEFAULT_MASKS_DIR = Path(os.environ.get("SCOTE_MASKS_DIR", "data/ovff/masks"))
# HD probe deep and min are the same library and section (identical scalefactors, tissue positions, tissue_hires md5).
ALIASES = {"ovff_hd_probe_min": "ovff_hd_probe_deep", "ovff_hd_probe": "ovff_hd_probe_deep", "ovff_hd3p_deep": "ovff_hd3p_min"}


# ----------------------------------------------------------------------------- basic affine helpers
def apply_affine(M, xy):
    """Apply 3x3 matrix M to (n, 2) points. Column vector convention: [x', y', 1]^T = M [x, y, 1]^T."""
    M = np.asarray(M, dtype=float)
    xy = np.asarray(xy, dtype=float)
    out = xy @ M[:2, :2].T + M[:2, 2]
    return out


def scale_matrix(sx, sy=None, tx=0.0, ty=0.0):
    sy = sx if sy is None else sy
    return np.array([[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]])


def decompose_affine(M):
    """Return dict(scale_x, scale_y, rotation_deg, shear_deg, tx, ty) of a 3x3 affine (x-axis-first QR-style decomposition)."""
    M = np.asarray(M, dtype=float)
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    sx = float(np.hypot(a, c))
    theta = float(np.degrees(np.arctan2(c, a)))
    # remove rotation, read off shear and y scale
    cos, sin = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    b2 = cos * b + sin * d
    d2 = -sin * b + cos * d
    sy = float(d2)
    shear = float(np.degrees(np.arctan2(b2, d2)))
    return dict(scale_x=sx, scale_y=sy, rotation_deg=theta, shear_deg=shear, tx=float(M[0, 2]), ty=float(M[1, 2]),
                det=float(np.linalg.det(M[:2, :2])))


def xenium_he_to_um(alignment_csv: Path, morphology_um_per_px: float = XENIUM_MORPHOLOGY_UM_PER_PX) -> np.ndarray:
    """H&E full-res px -> Xenium µm, from a Xenium Explorer `*_imagealignment.csv` (H&E px -> morphology px)."""
    A = np.loadtxt(alignment_csv, delimiter=",").reshape(3, 3)
    return scale_matrix(morphology_um_per_px) @ A


def visium_hd_px_to_um(microns_per_pixel: float) -> np.ndarray:
    """Visium HD full-res image px -> the µm frame used by scote.io.convert_visium_hd (px * microns_per_pixel)."""
    return scale_matrix(float(microns_per_pixel))


# ----------------------------------------------------------------------------- transforms.json access
def _canon(section: str) -> str:
    return ALIASES.get(section, section)


def load_transforms(path: Path | str | None = None) -> dict:
    p = Path(path) if path is not None else DEFAULT_TRANSFORMS
    with open(p) as f:
        return json.load(f)


def load_transform(section: str, path: Path | str | None = None, kind: str = "um_to_reference_um") -> np.ndarray:
    """3x3 matrix for `section` (see module docstring for `kind`). Reference section -> identity."""
    section = _canon(section)
    if section == REFERENCE and kind == "um_to_reference_um":
        return np.eye(3)
    t = load_transforms(path)
    if section not in t["sections"]:
        raise KeyError(f"{section} not in {path or DEFAULT_TRANSFORMS}; have {list(t['sections'])}")
    return np.asarray(t["sections"][section][kind], dtype=float).reshape(3, 3)


def to_reference(xy, section: str, path: Path | str | None = None, kind: str = "um_to_reference_um", nonrigid: bool = False) -> np.ndarray:
    """Map (n, 2) coordinates of `section` (own µm frame by default) into the reference frame (µm).
    nonrigid=True applies the kept TPS (own µm -> reference µm) instead of the affine where one exists (own-µm frame only)."""
    if nonrigid and kind == "um_to_reference_um":
        tps = load_nonrigid(section, path, "to_reference")
        if tps is not None:
            return tps(np.asarray(xy, float) * _own_um_rescale(section, path))
    return apply_affine(load_transform(section, path, kind), xy)


def from_reference(xy, section: str, path: Path | str | None = None, kind: str = "um_to_reference_um", nonrigid: bool = False) -> np.ndarray:
    """Inverse of `to_reference`: reference µm -> `section`'s own frame (nonrigid=True: the reverse TPS fit)."""
    if nonrigid and kind == "um_to_reference_um":
        tps = load_nonrigid(section, path, "from_reference")
        if tps is not None:
            return tps(xy) / _own_um_rescale(section, path)
    return apply_affine(np.linalg.inv(load_transform(section, path, kind)), xy)


def _own_um_rescale(section: str, path) -> float:
    """Factor mapping the current own-µm frame onto the frame the TPS was fitted in (1 unless the 3' HD mpp was updated)."""
    nr = load_transforms(path)["sections"][_canon(section)].get("nonrigid") or {}
    return float(nr.get("own_um_rescale", 1.0))


def hd3p_um_to_reference_from_spaceranger(outs_spatial: Path | str, path: Path | str | None = None) -> np.ndarray:
    """Hook for the 3' HD section (`ovff_hd3p_min`, deep run = same section): once Space Ranger `outs/spatial/` exists,
    its `scalefactors_json.json: microns_per_pixel` defines µm = full-res (btf) px * mpp. The stored
    `image_px_to_reference_um` was fitted on the btf itself, so the own-µm transform is image_px_to_ref @ scale(1/mpp).
    Returns the matrix (and the caller may write it into transforms.json via `update_hd3p_transform`)."""
    sf = json.load(open(Path(outs_spatial) / "scalefactors_json.json"))
    mpp = float(sf["microns_per_pixel"])
    M_img = load_transform("ovff_hd3p_min", path, kind="image_px_to_reference_um")
    return M_img @ scale_matrix(1.0 / mpp)


def update_hd3p_transform(outs_spatial: Path | str, path: Path | str | None = None) -> dict:
    p = Path(path) if path is not None else DEFAULT_TRANSFORMS
    t = load_transforms(p)
    M = hd3p_um_to_reference_from_spaceranger(outs_spatial, p)
    sf = json.load(open(Path(outs_spatial) / "scalefactors_json.json"))
    sec = t["sections"]["ovff_hd3p_min"]
    sec["um_to_reference_um"] = M.tolist()
    if sec.get("nonrigid", {}).get("kept"):
        # the TPS was fitted in the provisional frame (btf px * OME mpp): new own um * (mpp_prov / mpp) = provisional um
        sec["nonrigid"]["own_um_rescale"] = float(sec["own_frame"]["microns_per_pixel"]) / float(sf["microns_per_pixel"]) if sec["own_frame"].get("provisional") else sec["nonrigid"].get("own_um_rescale", 1.0)
    sec["own_frame"] = dict(kind="fullres_image_px_times_mpp", microns_per_pixel=float(sf["microns_per_pixel"]),
                            source=str(Path(outs_spatial) / "scalefactors_json.json"), provisional=False)
    with open(p, "w") as f:
        json.dump(t, f, indent=1)
    return sec



# ----------------------------------------------------------------------------- thin-plate spline (regularized)
class TPS:
    """2-D thin-plate spline x -> y with smoothing `lam` (units of the source coordinates squared). Duplicate sources are dropped."""

    def __init__(self, src, dst, lam: float = 1.0):
        src = np.asarray(src, float); dst = np.asarray(dst, float)
        _, idx = np.unique(np.round(src, 3), axis=0, return_index=True)
        src, dst = src[idx], dst[idx]
        n = len(src); self.src = src; self.lam = float(lam)
        K = self._U(src, src) + lam * np.eye(n)
        P = np.hstack([np.ones((n, 1)), src])
        L = np.zeros((n + 3, n + 3)); L[:n, :n] = K; L[:n, n:] = P; L[n:, :n] = P.T
        Y = np.zeros((n + 3, 2)); Y[:n] = dst
        self.coef = np.linalg.lstsq(L, Y, rcond=None)[0]

    @staticmethod
    def _U(a, b):
        r2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.where(r2 > 0, r2 * np.log(r2 + 1e-300) / 2, 0.0)

    def __call__(self, pts):
        pts = np.asarray(pts, float).reshape(-1, 2); n = len(self.src)
        out = np.empty_like(pts)
        for i in range(0, len(pts), 20000):   # chunked: the kernel matrix is (chunk x n)
            q = pts[i:i + 20000]
            out[i:i + 20000] = self._U(q, self.src) @ self.coef[:n] + np.hstack([np.ones((len(q), 1)), q]) @ self.coef[n:]
        return out

    def to_dict(self):
        return dict(kind="tps", lambda_src_units2=self.lam, control_src=self.src.tolist(), coef=self.coef.tolist())

    @classmethod
    def from_dict(cls, d):
        obj = cls.__new__(cls); obj.src = np.asarray(d["control_src"], float); obj.coef = np.asarray(d["coef"], float); obj.lam = float(d["lambda_src_units2"])
        return obj


def load_nonrigid(section: str, path: Path | str | None = None, direction: str = "to_reference") -> TPS | None:
    """TPS own µm -> reference µm (`to_reference`) or reference µm -> own µm (`from_reference`, a separate fit, not the exact
    inverse; the two disagree by ~ the hold-out error). None if the section has no kept non-rigid transform."""
    section = _canon(section)
    if section == REFERENCE:
        return None
    t = load_transforms(path)
    nr = t["sections"][section].get("nonrigid")
    if not nr or not nr.get("kept"):
        return None
    base = (Path(path).parent if path is not None else DEFAULT_TRANSFORMS.parent)
    with open(base / nr["file"]) as f:
        d = json.load(f)
    return TPS.from_dict(d[direction])


# ----------------------------------------------------------------------------- masks
def load_region_mask(section: str | None = None, masks_dir: Path | str | None = None) -> dict:
    """Load the 10 µm region raster in the reference frame (section=None) or transferred into `section`'s own frame.
    Returns dict(labels (H, W) uint8, origin_um (x0, y0), pixel_um, names list[str], frame str)."""
    d = Path(masks_dir) if masks_dir is not None else DEFAULT_MASKS_DIR
    fn = d / ("regions_10um.npz" if section is None or _canon(section) == REFERENCE else f"regions_10um_{_canon(section)}.npz")
    z = np.load(fn, allow_pickle=True)
    return dict(labels=z["labels"], origin_um=tuple(float(v) for v in z["origin_um"]), pixel_um=float(z["pixel_um"]),
                names=[str(s) for s in z["names"]], frame=str(z["frame"]))


def region_at(xy, mask: dict) -> np.ndarray:
    """Label index for (n, 2) µm coordinates in the mask's frame (0 = outside raster/tissue)."""
    xy = np.asarray(xy, dtype=float)
    x0, y0 = mask["origin_um"]
    px = mask["pixel_um"]
    col = np.floor((xy[:, 0] - x0) / px).astype(int)
    row = np.floor((xy[:, 1] - y0) / px).astype(int)
    lab = mask["labels"]
    ok = (col >= 0) & (row >= 0) & (col < lab.shape[1]) & (row < lab.shape[0])
    out = np.zeros(len(xy), dtype=lab.dtype)
    out[ok] = lab[row[ok], col[ok]]
    return out
