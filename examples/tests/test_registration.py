"""Synthetic tests for scripts/registration.py (transform conventions, json round trip, masks, 3' HD hook); run with
``python -m pytest tests`` from the examples root."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import registration as R  # noqa: E402


def _transforms_json(tmp_path: Path, M_um: np.ndarray, M_img: np.ndarray) -> Path:
    t = dict(reference=R.REFERENCE, sections={
        R.REFERENCE: dict(um_to_reference_um=np.eye(3).tolist(), image_px_to_reference_um=R.scale_matrix(0.2).tolist()),
        "ovff_xenium_v1": dict(um_to_reference_um=M_um.tolist(), image_px_to_reference_um=M_img.tolist()),
        "ovff_hd3p_min": dict(um_to_reference_um=(M_img @ R.scale_matrix(1 / 0.27381)).tolist(), image_px_to_reference_um=M_img.tolist(),
                              own_frame=dict(kind="fullres_image_px_times_mpp", microns_per_pixel=0.27381, provisional=True)),
    })
    p = tmp_path / "transforms.json"
    p.write_text(json.dumps(t))
    return p


def _rot(theta_deg, s=1.0, tx=0.0, ty=0.0):
    t = np.radians(theta_deg); c, si = np.cos(t), np.sin(t)
    return np.array([[s * c, -s * si, tx], [s * si, s * c, ty], [0, 0, 1.0]])


def test_apply_affine_column_vector_convention():
    M = _rot(90, tx=10, ty=0)
    out = R.apply_affine(M, [[1.0, 0.0]])
    assert np.allclose(out, [[10.0, 1.0]])   # x -> y under +90 deg rotation (x = column, y = row), then translate


def test_decompose_affine_roundtrip():
    M = _rot(-37.5, s=1.02, tx=123.4, ty=-56.7)
    d = R.decompose_affine(M)
    assert abs(d["rotation_deg"] + 37.5) < 1e-9 and abs(d["scale_x"] - 1.02) < 1e-9 and abs(d["scale_y"] - 1.02) < 1e-9
    assert abs(d["shear_deg"]) < 1e-9 and abs(d["det"] - 1.02 ** 2) < 1e-9


def test_to_and_from_reference_roundtrip(tmp_path):
    M_um = _rot(7.3, s=0.99, tx=410.0, ty=-940.0)
    p = _transforms_json(tmp_path, M_um, _rot(97.3, s=0.135, tx=300, ty=7000))
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 8000, (500, 2))
    ref = R.to_reference(xy, "ovff_xenium_v1", p)
    assert np.allclose(ref, R.apply_affine(M_um, xy))
    back = R.from_reference(ref, "ovff_xenium_v1", p)
    assert np.allclose(back, xy, atol=1e-8)
    # reference section is the identity, aliases resolve, unknown sections raise
    assert np.allclose(R.to_reference(xy, R.REFERENCE, p), xy)
    assert np.allclose(R.load_transform("ovff_hd3p_deep", p), R.load_transform("ovff_hd3p_min", p))
    with pytest.raises(KeyError):
        R.load_transform("nope", p)


def test_xenium_alignment_direction(tmp_path):
    """A Xenium Explorer matrix maps H&E px -> morphology px; xenium_he_to_um multiplies by 0.2125 um/px."""
    A = np.array([[0.0038, 0.6445, 316.8], [-0.6445, 0.0038, 33707.5], [0, 0, 1]])
    f = tmp_path / "align.csv"
    np.savetxt(f, A, delimiter=",")
    M = R.xenium_he_to_um(f)
    assert np.allclose(M, R.scale_matrix(0.2125) @ A)
    # an H&E pixel of 0.1369 um maps to 0.6445 morphology px = 0.137 um: scale consistency of the vendor file
    assert abs(0.6445 * 0.2125 - 0.13687) < 1e-3
    p = R.apply_affine(M, [[0, 0]])
    assert abs(p[0, 1] - 33707.5 * 0.2125) < 1e-6


def test_hd3p_hook_updates_from_scalefactors(tmp_path):
    M_img = _rot(3.0, s=0.2738, tx=100, ty=200)
    p = _transforms_json(tmp_path, _rot(0), M_img)
    outs = tmp_path / "spatial"; outs.mkdir()
    (outs / "scalefactors_json.json").write_text(json.dumps(dict(microns_per_pixel=0.27388, tissue_hires_scalef=0.2)))
    sec = R.update_hd3p_transform(outs, p)
    assert sec["own_frame"]["provisional"] is False
    # a bin at full-res px (1000, 2000) -> um via the new mpp -> reference must equal image_px_to_reference applied to the px
    px = np.array([[1000.0, 2000.0]])
    um = px * 0.27388
    assert np.allclose(R.to_reference(um, "ovff_hd3p_min", p), R.apply_affine(M_img, px), atol=1e-9)


def test_region_mask_lookup(tmp_path):
    lab = np.zeros((20, 30), np.uint8); lab[5:10, 10:20] = 3
    np.savez(tmp_path / "regions_10um.npz", labels=lab, origin_um=np.array([100.0, 50.0]), pixel_um=10.0,
             names=np.array(["outside", "tumor", "stroma", "immune"], dtype=object), frame="ref")
    m = R.load_region_mask(None, tmp_path)
    assert m["names"][3] == "immune" and m["pixel_um"] == 10.0
    xy = np.array([[100 + 10 * 15 + 1, 50 + 10 * 7 + 1], [0, 0], [100 + 10 * 25, 50 + 10 * 7]])
    assert list(R.region_at(xy, m)) == [3, 0, 0]


def test_tps_reproduces_affine_plus_smooth_warp(tmp_path):
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 5000, (300, 2))
    warp = lambda p: R.apply_affine(_rot(3.0, tx=50, ty=-20), p) + 15 * np.sin(p / 900.0)
    dst = warp(src)
    tps = R.TPS(src, dst, lam=1.0)
    q = rng.uniform(200, 4800, (50, 2))
    assert np.median(np.linalg.norm(tps(q) - warp(q), axis=1)) < 2.0   # smooth warp recovered to ~ um
    # serialization and the nonrigid loader
    d = tps.to_dict(); tps2 = R.TPS.from_dict(d)
    assert np.allclose(tps2(q), tps(q))
    t = dict(reference=R.REFERENCE, sections={"ovff_xenium_v1": dict(um_to_reference_um=_rot(3.0, tx=50, ty=-20).tolist(),
                                                                   nonrigid=dict(kept=True, file="nr.json"))})
    (tmp_path / "transforms.json").write_text(json.dumps(t))
    (tmp_path / "nr.json").write_text(json.dumps(dict(to_reference=d, from_reference=R.TPS(dst, src, lam=1.0).to_dict())))
    p = tmp_path / "transforms.json"
    assert np.allclose(R.to_reference(q, "ovff_xenium_v1", p, nonrigid=True), tps(q))
    assert np.allclose(R.to_reference(q, "ovff_xenium_v1", p, nonrigid=False), R.apply_affine(_rot(3.0, tx=50, ty=-20), q))
    back = R.from_reference(R.to_reference(q, "ovff_xenium_v1", p, nonrigid=True), "ovff_xenium_v1", p, nonrigid=True)
    assert np.median(np.linalg.norm(back - q, axis=1)) < 3.0
