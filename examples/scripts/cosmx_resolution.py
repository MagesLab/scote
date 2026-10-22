#!/usr/bin/env python3
"""Resolution floor of preprint Supplementary Fig. S4c: what is the smallest honest r bin for CosMx, and does the
sub-micron regime survive its own controls?

  * the precision of a CosMx call is not the camera pixel: each call is a consensus over `target_call_observations` spots
    whose scatter is `spread`, so the centroid precision is ~ spread / sqrt(n_obs) where the decoder fields exist (the
    AtoMx flat files of GEO GSE282059 carry neither, so it is recorded as absent);
  * z is quantized in planes while xy is sub-micron, so this works **in-plane** (pairs within one z plane) and treats z as a
    stratum, not a coordinate.

The decisive control: **negative probes and system controls**, codes with no transcriptome match or no physical target.
Their two-point function must be flat (log g = 0 at every r); anything they show at small r is the instrument, the decoder
or the estimator, and it is a direct measurement of the artifact floor.

    python scripts/cosmx_resolution.py --processed data/processed/cosmx_kidney_gse282059/piece_2 --n-fov 16 \
        --delta 0.05 --n-distance-bins 60 --tile-um 200 --top-genes 150 --threads 8 \
        --title "CosMx 6k kidney GSE282059 (analysis piece)" \
        --out-json results/cosmx_kidney/resolution.json --out-fig figures/quicklook/cosmx_kidney_resolution.png
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scote import random_label_reference, twopoint  # noqa: E402


def enumerated_counts_per_tile(xy, codes, G, delta, n_distance_bins, tile_um, threads=None):
    """Per-tile pair counts of the enumerated level alone (scote.twopoint with n_levels=0, so the level has
    n_distance_bins linear bins of width delta): counts (T, n_distance_bins, G, G) float64, totals (T, G) float64, and
    the bins (DataFrame with level, shell, r_from, r_to, r_central)."""
    res = twopoint(xy, codes, names=[str(i) for i in range(G)], delta=delta, shells_per_level=n_distance_bins,
                   n_levels=0, tile_um=tile_um, threads=threads or None, keep_tiles=True)
    return res.counts_tiles, res.totals_tiles, res.bins


def stacked_log_g(counts: np.ndarray, totals: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Count-weighted log g over a set of (a, b) entries, pooled over tiles (pooled estimator)."""
    P = counts.sum(0)
    ref = random_label_reference(P[None], totals.sum(0)[None])[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(P[:, sel].sum(1) / ref[:, sel].sum(1))


def run_group(xy: np.ndarray, codes: np.ndarray, G: int, delta: float, n_bins: int, tile_um: float,
              threads: int):
    counts, totals, iv = enumerated_counts_per_tile(xy, codes, G, delta, n_bins, tile_um, threads)
    # int64 counts can overflow inside random_label_reference (Ptot * n_a * n_b is formed before the
    # division by n_tot^2). Cast before any normalization.
    return counts.astype(np.float64), totals.astype(np.float64), iv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed/cosmx_kidney_gse282059/piece_2")
    ap.add_argument("--n-fov", type=int, default=12)
    ap.add_argument("--delta", type=float, default=0.05, help="r bin width in um")
    ap.add_argument("--n-distance-bins", dest="n_bins", type=int, default=60, help="r bins of width delta")
    ap.add_argument("--tile-um", type=float, default=200.0,
                    help="block size. A CosMx FOV is 4256 px = 512 um, so 500 um tiles would be one tile "
                         "per FOV and give no within-FOV blocks; 200 um keeps several per FOV")
    ap.add_argument("--top-genes", type=int, default=200)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-fig", required=True)
    ap.add_argument("--title", default="CosMx 6k kidney GSE282059 (analysis piece)")
    a = ap.parse_args()

    files = sorted(glob.glob(f"{a.processed}/fov_*.parquet"))[: a.n_fov]
    if not files:
        raise SystemExit(f"no parquet in {a.processed}")
    df = pl.concat([pl.read_parquet(f) for f in files])
    print(f"{len(files)} FOVs, {len(df):,} molecules", flush=True)

    out: dict = {"n_fov": len(files), "n_molecules": int(len(df)), "delta_um": a.delta,
                 "n_bins": a.n_bins, "tile_um": a.tile_um}

    # ---- 1. the physical localization precision -----------------------------------------------------
    PX = 0.120280945
    e = df.filter(pl.col("codeclass") == "Endogenous")
    has_prec = "spread" in df.columns and "n_obs" in df.columns
    if has_prec:
        prec = (e["spread"].to_numpy() / np.sqrt(np.maximum(e["n_obs"].to_numpy(), 1))) * PX
        out["centroid_precision_um"] = {f"p{q}": float(np.percentile(prec, q)) for q in (5, 25, 50, 75, 95)}
        out["spread_um"] = {f"p{q}": float(np.percentile(e["spread"].to_numpy() * PX, q))
                            for q in (5, 50, 95)}
        print(f"  centroid precision: median {np.median(prec)*1000:.0f} nm "
              f"(IQR {np.percentile(prec,25)*1000:.0f}-{np.percentile(prec,75)*1000:.0f})", flush=True)
    else:   # AtoMx flat files (e.g. GSE282059) carry no decoder spread / n_obs
        out["centroid_precision_um"] = None
        out["centroid_precision_note"] = "absent: the input has no per-molecule spread / n_obs (decoder fields)"
        print("  centroid precision: not available (no spread / n_obs columns)", flush=True)

    # ---- 2. in-plane 2-point for three code classes -------------------------------------------------
    # every class is run on the SAME geometry and the same estimator; only the labels differ
    results = {}
    for cls, label in (("Endogenous", "endogenous"), ("Negative", "negative probes"),
                       ("SystemControl", "system controls")):
        d = df.filter(pl.col("codeclass") == cls)
        if len(d) < 5000:
            print(f"  {label}: only {len(d):,} molecules, skipped", flush=True)
            continue
        # the cap applies to EVERY class, not just the endogenous one: CosMx carries thousands of
        # system-control barcodes, and taking all of them makes the kernel's (tiles, bins, G, G) size
        # arithmetic overflow.
        n_codes = int(d["feature"].n_unique())
        top = (d.group_by("feature").len().sort("len", descending=True)
                 .head(a.top_genes)["feature"].to_list())
        d = d.filter(pl.col("feature").is_in(top))
        names = sorted(set(d["feature"].to_list()))
        gi = {g: i for i, g in enumerate(names)}
        G = len(names)
        acc_c = acc_t = None
        ivs = None
        for zp in sorted(set(d["z_plane"].to_list())):          # in-plane: z is a stratum, not an axis
            s = d.filter(pl.col("z_plane") == zp)
            if len(s) < 1000:
                continue
            xy = np.c_[s["x"].to_numpy(), s["y"].to_numpy()].astype(np.float64)
            codes = np.array([gi[f] for f in s["feature"].to_list()], dtype=np.int64)
            c, t, iv = run_group(xy, codes, G, a.delta, a.n_bins, a.tile_um, a.threads)
            acc_c = c.sum(0) if acc_c is None else acc_c + c.sum(0)
            acc_t = t.sum(0) if acc_t is None else acc_t + t.sum(0)
            ivs = iv
        eye = np.eye(G, dtype=bool)
        P = acc_c
        ref = random_label_reference(P[None], acc_t[None])[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            same = np.log(P[:, eye].sum(1) / ref[:, eye].sum(1))
            off = np.log(P[:, ~eye].sum(1) / ref[:, ~eye].sum(1))
        r = np.asarray(ivs["r_central"])
        results[label] = {"n_features": G, "n_codes_available": n_codes,
                          "n_molecules": int(len(d)),
                          "r_um": r.tolist(), "log_g_same": same.tolist(), "log_g_diff": off.tolist(),
                          "pairs_same_bin0": float(P[0][eye].sum())}
        print(f"  {label}: G={G}, {len(d):,} molecules, "
              f"log g_aa bin0 {same[0]:+.3f} bin1 {same[1]:+.3f}", flush=True)
    out["classes"] = results

    # ---- 3. the artifact floor from the controls ----------------------------------------------------
    ctrl = [k for k in ("negative probes", "system controls") if k in results]
    if ctrl:
        floor = {}
        for k in ctrl:
            v = np.array(results[k]["log_g_same"])
            fin = np.isfinite(v)
            floor[k] = {"max_abs_over_r": float(np.max(np.abs(v[fin]))) if fin.any() else None,
                        "bin0": float(v[0]) if np.isfinite(v[0]) else None,
                        "first_bin_within_0.1": int(np.argmax(np.abs(np.where(fin, v, np.inf)) < 0.1))
                        if fin.any() else None}
        out["control_floor"] = floor
        for k, v in floor.items():
            print(f"  {k}: |log g_aa| max {v['max_abs_over_r']}, bin0 {v['bin0']}")

    # ---- figure ---------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    ax = axes[0]
    if not has_prec:
        ax.text(0.5, 0.5, "no per-molecule precision fields\nin this input", transform=ax.transAxes, ha="center")
    else:
        ax.hist(prec * 1000, bins=120, range=(0, 600), color="tab:purple", histtype="step", lw=1.4,
                label="centroid precision  spread/$\\sqrt{n}$")
        ax.hist(e["spread"].to_numpy() * PX * 1000, bins=120, range=(0, 1200), color="0.55",
                histtype="step", lw=1.2, label="raw spot scatter  spread")
        ax.axvline(np.median(prec) * 1000, color="k", lw=1.1, ls="--",
                   label=f"median {np.median(prec)*1000:.0f} nm")
        ax.axvline(120.280945, color="tab:red", lw=1.0, ls=":", label="camera pixel 120 nm")
        ax.set_xlabel("nm")
        ax.set_ylabel("molecules")
        ax.legend(fontsize=7)
    ax.set_title("(a) how well is a CosMx molecule actually located?", fontsize=9)

    ax = axes[1]
    col = {"endogenous": "tab:purple", "negative probes": "tab:orange", "system controls": "tab:cyan"}
    for k, v in results.items():
        r = np.array(v["r_um"])
        ax.plot(r, v["log_g_same"], color=col[k], lw=1.5, label=f"{k} — same feature")
        ax.plot(r, v["log_g_diff"], color=col[k], lw=1.0, ls="--", alpha=0.7)
    ax.axhline(0, color="0.5", lw=0.8)
    if has_prec:
        ax.axvline(np.median(prec), color="k", lw=1.0, ls=":",
                   label=f"localization precision {np.median(prec)*1000:.0f} nm")
    ax.set_xlabel("r (µm), in-plane")
    ax.set_ylabel(r"stacked $\log g$")
    ax.set_title("(b) the controls are the artifact floor\n(dashed = different feature)", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[2]
    if "endogenous" in results:
        r = np.array(results["endogenous"]["r_um"])
        ax.plot(r, results["endogenous"]["log_g_same"], color="tab:purple", lw=1.6, marker="o", ms=2.5)
        for k in ctrl:
            ax.plot(np.array(results[k]["r_um"]), results[k]["log_g_same"], color=col[k], lw=1.1)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xlim(0, min(3.0, r.max()))
        if has_prec:
            ax.axvspan(0, np.median(prec), color="0.88", zorder=0)
        ax.set_xlabel("r (µm), in-plane")
        ax.set_ylabel(r"stacked $\log g_{aa}$")
        ax.set_title("(c) the sub-micron regime, zoomed\n(shaded = below the localization precision)",
                     fontsize=9)

    fig.suptitle(f"{a.title} — localization precision and the control floor "
                 f"({len(files)} FOVs, Δ = {a.delta} µm, in-plane)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    Path(a.out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out_fig, dpi=170)
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out_fig} and {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
