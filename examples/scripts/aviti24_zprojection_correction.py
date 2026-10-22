#!/usr/bin/env python3
"""z-projection of preprint Supplementary Fig. S4b: AVITI24 molecule-level two-point functions per z-plane vs projected.

`scote.twopoint` uses x and y only. AVITI24 records 7 planes at 1 µm, so an xy-only estimate on all molecules is a projection
through a ~6 µm slab: two molecules close in xy but far apart in z enter the estimator as a close pair. This script computes
the stacked same-gene and different-gene log g of the 50 most abundant genes in the six baseline wells twice: projected (all
planes pooled) and per-slice (pairs within one plane only, counts summed over planes).

It also decomposes the coincidence term. AVITI24 localizations sit on a 0.48 µm pixel lattice, and a cross-plane duplicate
re-localized by one pixel (0.48 µm, inside the first 0.5 µm bin) is invisible to a test that counts only pairs at zero 2-D
separation. Binned finely enough, the lattice makes g(r) a comb, and each integer offset can be read separately: the
projected-minus-per-slice difference at each shell is the cross-plane term, shell (0,0) the exact coincidences, shell (1,0)
the one-pixel neighbors, and the rest says how far the leakage reaches. Counts are pooled across the bins carrying a shell
before the log, never averaged as logs, because a lattice distance can fall on a bin edge (0.48 µm does).

Input: data/processed/aviti24_<well>/molecules.parquet (python -m scote convert --platform aviti24, one per well).

    python scripts/aviti24_zprojection_correction.py --top-genes 50 --baseline-only \
        --out-json results/cosmx/aviti24_zprojection_correction.json \
        --out-fig figures/quicklook/aviti24_zprojection_correction.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

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

WELLS = ["A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2", "E1", "E2", "F1", "F2"]
BASELINE = ["A1", "A2", "B1", "B2", "C1", "C2"]      # the 6 baseline wells
PITCH = 0.48
LATTICE_D = np.array([0.0, 1.0, np.sqrt(2), 2.0, np.sqrt(5), 2 * np.sqrt(2), 3.0])
LATTICE_LBL = ["(0,0)", "(1,0)", "(1,1)", "(2,0)", "(2,1)", "(2,2)", "(3,0)"]


def curves(accC, accT, js=None):
    """Stacked same- and different-gene log g. If js is given, pool those bins before the log."""
    ref = random_label_reference(accC[None], accT[None])[0]
    eye = np.eye(len(accT), dtype=bool)
    C = accC if js is None else accC[js].sum(0)[None]
    R = ref if js is None else ref[js].sum(0)[None]
    with np.errstate(divide="ignore", invalid="ignore"):
        same = np.log(C[:, eye].sum(1) / R[:, eye].sum(1))
        diff = np.log(C[:, ~eye].sum(1) / R[:, ~eye].sum(1))
    return same, diff


def half_decay(r, g):
    """r at which the stacked log g falls to half its first-bin value, by linear interpolation."""
    if not np.isfinite(g[0]) or g[0] <= 0:
        return float("nan")
    target = g[0] / 2.0
    for i in range(1, len(g)):
        if np.isfinite(g[i]) and g[i] <= target:
            x0, x1, y0, y1 = r[i - 1], r[i], g[i - 1], g[i]
            return float(x0 + (y0 - target) * (x1 - x0) / (y0 - y1)) if y0 != y1 else float(x1)
    return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-genes", type=int, default=50, help="panel: the N most abundant genes")
    ap.add_argument("--panel-file", default=None,
                    help="explicit gene list instead of top-N by abundance")
    ap.add_argument("--compartment", default="all", choices=["all", "nuclear", "cytoplasmic"])
    ap.add_argument("--wells", default=",".join(WELLS))
    ap.add_argument("--baseline-only", action="store_true",
                    help="only the 6 baseline wells (A1-C2)")
    ap.add_argument("--delta", type=float, default=0.024, help="1/20 pixel; the lattice spikes separate")
    ap.add_argument("--n-distance-bins", dest="n_bins", type=int, default=126, help="bins of width delta, out to ~3 µm")
    ap.add_argument("--coarse", type=float, default=0.5, help="coarse bin width, rebuilt by pooling")
    ap.add_argument("--tile-um", type=float, default=9000.0)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-fig", required=True)
    a = ap.parse_args()

    wells = [w for w in a.wells.split(",") if not a.baseline_only or w in BASELINE]
    if a.panel_file:
        feats = [l.strip() for l in Path(a.panel_file).read_text().splitlines()
                 if l.strip() and not l.startswith("#")]
        tot = {}
    else:
        tot = {}
        for w in wells:
            tbl = pq.read_table(f"data/processed/aviti24_{w}/molecules.parquet",
                                columns=["feature", "batch"])
            tbl = tbl.filter(pc.not_equal(tbl.column("batch").cast("string"), "B01"))
            vc = pd.Series(tbl.column("feature").cast("string").to_pandas()).value_counts()
            for g, n in vc.items():
                tot[g] = tot.get(g, 0) + int(n)
        feats = [g for g, _ in sorted(tot.items(), key=lambda kv: -kv[1])[: a.top_genes]]
    gi = {g: i for i, g in enumerate(feats)}
    G = len(feats)
    print(f"{len(wells)} wells, panel = {G} genes "
          f"({'file ' + a.panel_file if a.panel_file else 'top by abundance'}), "
          f"compartment {a.compartment}", flush=True)

    out = {"n_wells": len(wells), "wells": wells, "n_genes": G, "delta_um": a.delta,
           "compartment": a.compartment, "panel": a.panel_file or f"top{a.top_genes}",
           "pitch_um": PITCH, "modes": {}}
    acc = {}
    for mode in ("projected", "per-slice"):
        accC = accT = None
        iv = None
        for w in wells:
            cols = ["x", "y", "z_plane", "feature", "batch"] + (
                [] if a.compartment == "all" else ["nucleus"])
            tbl = pq.read_table(f"data/processed/aviti24_{w}/molecules.parquet", columns=cols)
            tbl = tbl.filter(pc.not_equal(tbl.column("batch").cast("string"), "B01"))
            if a.compartment != "all":
                tbl = tbl.filter(pc.equal(tbl.column("nucleus"), a.compartment == "nuclear"))
            f = tbl.column("feature").cast("string").to_pandas().to_numpy()
            code = pd.Series(gi, dtype="float64").reindex(f).to_numpy()
            ok = np.isfinite(code)
            x, y = tbl.column("x").to_numpy()[ok], tbl.column("y").to_numpy()[ok]
            zp = tbl.column("z_plane").to_numpy()[ok]
            cd = code[ok].astype(np.int64)
            groups = ([(None, np.ones(len(cd), bool))] if mode == "projected"
                      else [(p, zp == p) for p in np.unique(zp)])
            for _, m in groups:
                if m.sum() < 50:
                    continue
                c, t, iv2 = enumerated_counts_per_tile(np.c_[x[m], y[m]].astype(np.float64), cd[m], G,
                                                   a.delta, a.n_bins, a.tile_um, a.threads)
                accC = c.astype(np.float64).sum(0) if accC is None else accC + c.astype(np.float64).sum(0)
                accT = t.astype(np.float64).sum(0) if accT is None else accT + t.astype(np.float64).sum(0)
                iv = iv2 if iv2 is not None else iv
            print(f"  {mode:10s} {w} done", flush=True)
        acc[mode] = (accC, accT, np.asarray(iv["r_central"]))

    r = acc["per-slice"][2]
    k = int(round(a.coarse / a.delta))
    n_coarse = len(r) // k
    r_coarse = np.array([r[i * k:(i + 1) * k].mean() for i in range(n_coarse)])
    for mode, (accC, accT, _) in acc.items():
        same_c = np.array([curves(accC, accT, list(range(i * k, (i + 1) * k)))[0][0]
                           for i in range(n_coarse)])
        diff_c = np.array([curves(accC, accT, list(range(i * k, (i + 1) * k)))[1][0]
                           for i in range(n_coarse)])
        shells = []
        for i, d in enumerate(LATTICE_D):
            js = [j for j in range(len(r)) if abs(r[j] - d * PITCH) <= a.delta]
            if not js:
                continue
            s, _ = curves(accC, accT, js)
            ref = random_label_reference(accC[None], accT[None])[0]
            eye = np.eye(G, dtype=bool)
            shells.append({"offset": LATTICE_LBL[i], "r_um": float(d * PITCH),
                           "log_g_same": float(s[0]),
                           "same_gene_pairs": float(accC[js][:, eye].sum()),
                           "all_pairs": float(accC[js].sum())})
        out["modes"][mode] = {
            "r_coarse_um": r_coarse.tolist(),
            "same": same_c.tolist(), "diff": diff_c.tolist(),
            "bin0_same": float(same_c[0]), "bin0_diff": float(diff_c[0]),
            "fold_above_random": float(np.exp(same_c[0])),
            "half_decay_um": half_decay(r_coarse, same_c),
            "n_molecules": float(accT.sum()), "shells": shells}
        q = out["modes"][mode]
        print(f"\n{mode}: bin0 same {q['bin0_same']:+.4f} ({q['fold_above_random']:.2f}x random), "
              f"diff {q['bin0_diff']:+.4f}, half-decay {q['half_decay_um']:.3f} µm")
        print("  " + "  ".join(f"{x:+.3f}" for x in same_c[:6]))

    p, s = out["modes"]["projected"], out["modes"]["per-slice"]
    out["projection_inflation"] = {
        "bin0_log_g_difference": p["bin0_same"] - s["bin0_same"],
        "bin0_factor_in_g": float(np.exp(p["bin0_same"] - s["bin0_same"])),
        "half_decay_projected_um": p["half_decay_um"],
        "half_decay_per_slice_um": s["half_decay_um"],
        "by_shell": [{"offset": ps["offset"], "r_um": ps["r_um"],
                      "projected": ps["log_g_same"], "per_slice": ss["log_g_same"],
                      "cross_plane_term": ps["log_g_same"] - ss["log_g_same"]}
                     for ps, ss in zip(p["shells"], s["shells"])]}
    pi = out["projection_inflation"]
    print(f"\nPROJECTION INFLATION at bin 0: {pi['bin0_log_g_difference']:+.4f} in log g "
          f"= {pi['bin0_factor_in_g']:.2f}x in g")
    print(f"half-decay {pi['half_decay_projected_um']:.3f} µm projected -> "
          f"{pi['half_decay_per_slice_um']:.3f} µm per-slice")
    print(f"\n{'offset':>7} {'r(µm)':>7} {'projected':>10} {'per-slice':>10} {'cross-plane':>12}")
    for q in pi["by_shell"]:
        print(f"{q['offset']:>7} {q['r_um']:7.3f} {q['projected']:+10.4f} {q['per_slice']:+10.4f} "
              f"{q['cross_plane_term']:+12.4f}")
    Path(a.out_json).write_text(json.dumps(out, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    for mode, col in (("projected", "C3"), ("per-slice", "C0")):
        q = out["modes"][mode]
        ax.plot(q["r_coarse_um"], q["same"], "o-", color=col, label=f"{mode} same-gene")
        ax.plot(q["r_coarse_um"], q["diff"], "s--", color=col, alpha=0.5, label=f"{mode} different-gene")
    ax.axhline(0, lw=0.6, color="k")
    ax.set_xlabel("r (µm)"); ax.set_ylabel("stacked log g")
    ax.set_title(f"top {G} genes, {len(wells)} wells\n"
                 f"bin 0: {p['bin0_same']:+.3f} projected → {s['bin0_same']:+.3f} per-slice "
                 f"({pi['bin0_factor_in_g']:.1f}× in g)", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    xs = [q["r_um"] for q in pi["by_shell"]]
    ax.plot(xs, [q["projected"] for q in pi["by_shell"]], "o-", color="C3", label="projected")
    ax.plot(xs, [q["per_slice"] for q in pi["by_shell"]], "o-", color="C0", label="per-slice")
    ax.bar(xs, [q["cross_plane_term"] for q in pi["by_shell"]], width=0.08, color="0.7",
           label="cross-plane term")
    ax.axvline(0, color="C2", ls=":", lw=1.0)
    ax.annotate("exact coincidences", (0, ax.get_ylim()[1] * 0.9), fontsize=6, color="C2")
    ax.annotate("one-pixel neighbors", (PITCH, ax.get_ylim()[1] * 0.8), fontsize=6, color="C2")
    ax.set_xlabel("lattice separation (µm)"); ax.set_ylabel("stacked log g$_{aa}$")
    ax.set_title("coincidence term, decomposed by lattice shell", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out_fig, dpi=170)
    print(f"\nwrote {a.out_json} and {a.out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
