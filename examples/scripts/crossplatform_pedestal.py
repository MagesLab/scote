#!/usr/bin/env python3
"""The composition pedestal on both platforms (AVITI24 and CosMx), the helpers of scripts/pedestal_direct.py.

The random-label reference is built from sample-pooled totals while pairs are counted per block, so a gene confined to the
cells that express it scores as clustered: cell composition sits inside "same-gene clustering". To separate the two, gene
labels are permuted WITHIN each cell (positions and per-cell composition preserved exactly, only the sub-cellular
arrangement destroyed), per z-plane, on a gene panel shared by both platforms. Observed and permuted are computed on the
IDENTICAL subset (cell_id != -1 on both platforms, since a molecule outside any cell cannot be permuted within one).

The full (bins, G, G) tensors are cached for every run including each permutation replicate.

    python scripts/crossplatform_pedestal.py --compartments all --n-perm 3 \
        --tensor-dir data/twopoint/pedestal --out-json results/cosmx_kidney/crossplatform_pedestal.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scote import random_label_reference, twopoint  # noqa: E402


def enumerated_counts_per_tile(xy, codes, G, delta, n_distance_bins, tile_um, threads=None):
    """Per-tile pair counts of the enumerated level alone (scote.twopoint with n_levels=0, so the level has
    n_distance_bins linear bins of width delta): counts (T, n_distance_bins, G, G) float64, totals (T, G) float64, and
    the bins (DataFrame with level, shell, r_from, r_to, r_central)."""
    res = twopoint(xy, codes, names=[str(i) for i in range(G)], delta=delta, shells_per_level=n_distance_bins,
                   n_levels=0, tile_um=tile_um, threads=threads or None, keep_tiles=True)
    return res.counts_tiles, res.totals_tiles, res.bins

WELLS = ["A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2", "E1", "E2", "F1", "F2"]
FOV_UM = 4256 * 0.120280945
# One tile per block on both platforms. The kernel returns (tiles, bins, G, G) and its per-tile
# overhead dominates: a 500 µm tiling of a 7.6 mm AVITI24 well is 240 tiles and ran ~10x slower per
# block than a single tile, for a 2.96 GB allocation instead of 12 MB. Pairs are only counted WITHIN a
# tile, so a larger tile also loses FEWER boundary pairs -- at r <= 3 µm the difference either way is
# negligible. Observed and permuted are computed with the identical tiling, so the excess is unaffected.
SPEC = {"aviti24": dict(delta=0.5, n_bins=8, tile=9000.0),
        "cosmx": dict(delta=0.25, n_bins=12, tile=2047.0)}


def permute_within_cell(cell: np.ndarray, code: np.ndarray, rng) -> np.ndarray:
    """Permute labels among the molecules of each cell. Positions and per-cell composition untouched."""
    r = rng.random(len(cell))
    shuffled = np.lexsort((r, cell))          # grouped by cell, random order inside each
    base = np.argsort(cell, kind="stable")    # grouped by cell, original order inside each
    out = np.empty_like(code)
    out[base] = code[shuffled]
    return out


def load_aviti24(comp, feats, gi):
    """Return a list of (key, x, y, cell, code) blocks -- one per (well, z plane)."""
    blocks = []
    for w in WELLS:
        cols = ["x", "y", "feature", "batch", "z_plane", "cell_id"] + (
            [] if comp == "all" else ["nucleus"])
        tbl = pq.read_table(f"data/processed/aviti24_{w}/molecules.parquet", columns=cols)
        tbl = tbl.filter(pc.not_equal(tbl.column("batch").cast("string"), "B01"))
        tbl = tbl.filter(pc.not_equal(tbl.column("cell_id"), -1))
        if comp != "all":
            tbl = tbl.filter(pc.equal(tbl.column("nucleus"), comp == "nuclear"))
        f = tbl.column("feature").cast("string").to_pandas().to_numpy()
        code = pd.Series(gi, dtype="float64").reindex(f).to_numpy()
        ok = np.isfinite(code)
        x, y = tbl.column("x").to_numpy()[ok], tbl.column("y").to_numpy()[ok]
        zp = tbl.column("z_plane").to_numpy()[ok]
        # cell ids are per well, so make them globally unique before any permutation
        cid = tbl.column("cell_id").to_numpy()[ok].astype(np.int64) + WELLS.index(w) * 10_000_000
        cd = code[ok].astype(np.int64)
        for p in np.unique(zp):
            m = zp == p
            if m.sum() >= 50:
                blocks.append((f"{w}z{p}", x[m], y[m], cid[m], cd[m]))
        print(f"    loaded {w}: {ok.sum():,} in-cell molecules", flush=True)
    return blocks


def load_cosmx(comp, feats, gi, root="data/processed/cosmx_kidney_gse282059/piece_2"):
    """root: directory of fov_*.parquet files (scripts/cosmx_kidney_convert.py writes one per FOV of the analysis piece)."""
    blocks = []
    cols = ["x", "y", "z_plane", "feature", "codeclass", "nucleus", "compartment", "cell_id", "fov"]
    for f_ in sorted(glob.glob(f"{root}/fov_*.parquet")):
        d = (pl.read_parquet(f_, columns=cols)
             .filter((pl.col("codeclass") == "Endogenous") & (pl.col("cell_id") != -1)))
        if comp == "nuclear":
            d = d.filter(pl.col("nucleus"))
        elif comp == "cytoplasmic":
            d = d.filter(pl.col("compartment") == "Cytoplasm")
        d = d.with_columns(pl.col("feature").cast(pl.Utf8)).filter(pl.col("feature").is_in(feats))
        if len(d) < 50:
            continue
        for zp in sorted(set(d["z_plane"].to_list())):
            s = d.filter(pl.col("z_plane") == zp)
            if len(s) < 20:
                continue
            blocks.append((f"{Path(f_).stem}z{zp}",
                           s["x"].to_numpy(), s["y"].to_numpy(),
                           s["cell_id"].to_numpy().astype(np.int64),
                           np.array([gi[g] for g in s["feature"].to_list()], dtype=np.int64)))
    print(f"    loaded cosmx: {sum(len(b[4]) for b in blocks):,} in-cell molecules", flush=True)
    return blocks


def accumulate(blocks, G, spec, threads, rng=None):
    """Sum level-0 pair counts over blocks; if rng is given, permute labels within cells first."""
    accC = accT = None
    for _, x, y, cell, code in blocks:
        cd = permute_within_cell(cell, code, rng) if rng is not None else code
        c, t, _ = enumerated_counts_per_tile(np.c_[x, y].astype(np.float64), cd, G,
                                         spec["delta"], spec["n_bins"], spec["tile"], threads)
        accC = c.astype(np.float64).sum(0) if accC is None else accC + c.astype(np.float64).sum(0)
        accT = t.astype(np.float64).sum(0) if accT is None else accT + t.astype(np.float64).sum(0)
    return accC, accT


def same_curve(accC, accT):
    ref = random_label_reference(accC[None], accT[None])[0]
    eye = np.eye(len(accT), dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(accC[:, eye].sum(1) / ref[:, eye].sum(1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="results/cosmx_kidney/common_genes_matched_aviti24.txt")
    ap.add_argument("--platforms", default="aviti24,cosmx")
    ap.add_argument("--compartments", default="all")
    ap.add_argument("--n-perm", type=int, default=3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tensor-dir", default="data/twopoint/pedestal")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()

    Path(a.tensor_dir).mkdir(parents=True, exist_ok=True)
    feats = [l.strip() for l in Path(a.panel).read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    gi = {g: i for i, g in enumerate(feats)}
    G = len(feats)
    print(f"matched panel {G} genes", flush=True)

    out = {"panel": a.panel, "n_genes": G, "n_perm": a.n_perm, "seed": a.seed,
           "construction": "per-slice, in-cell molecules only, within-cell label permutation",
           "cells": {}}
    for comp in a.compartments.split(","):
        for plat in a.platforms.split(","):
            spec = SPEC[plat]
            print(f"  {plat}|{comp}: loading ...", flush=True)
            blocks = (load_aviti24 if plat == "aviti24" else load_cosmx)(comp, feats, gi)
            n_mol = sum(len(b[4]) for b in blocks)
            n_cell = len(np.unique(np.concatenate([b[3] for b in blocks])))
            print(f"  {plat}|{comp}: {len(blocks)} blocks, {n_mol:,} molecules, "
                  f"{n_cell:,} cells; observed ...", flush=True)
            obsC, obsT = accumulate(blocks, G, spec, a.threads)
            np.savez_compressed(f"{a.tensor_dir}/{plat}_{comp}_observed.npz",
                                counts=obsC.astype(np.float32), totals=obsT)
            obs = same_curve(obsC, obsT)
            print(f"    observed  {np.array2string(obs, precision=3)}", flush=True)
            perms = []
            for k in range(a.n_perm):
                rng = np.random.default_rng(a.seed + 1000 * k)
                pC, pT = accumulate(blocks, G, spec, a.threads, rng=rng)
                np.savez_compressed(f"{a.tensor_dir}/{plat}_{comp}_perm{k}.npz",
                                    counts=pC.astype(np.float32), totals=pT)
                cur = same_curve(pC, pT)
                perms.append(cur.tolist())
                print(f"    perm {k}    {np.array2string(cur, precision=3)}", flush=True)
            P = np.asarray(perms)
            out["cells"][f"{plat}|{comp}"] = {
                "delta_um": spec["delta"], "n_bins": spec["n_bins"],
                "n_molecules": n_mol, "n_cells": int(n_cell),
                "observed": obs.tolist(),
                "pedestal_mean": P.mean(0).tolist(), "pedestal_sd": P.std(0, ddof=1).tolist(),
                "excess": (obs - P.mean(0)).tolist()}
            Path(a.out_json).write_text(json.dumps(out, indent=2))
    print(f"wrote {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
