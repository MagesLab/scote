#!/usr/bin/env python3
"""Convert the public CosMx 6k human kidney slide (GEO GSE282059, Reck et al. 2025) to the common molecule model.

Source: the AtoMx flat file `GSE282059_2B_FFPE_tx_file.csv.gz` (columns fov, cell_ID, cell, x_local_px, y_local_px,
x_global_px, y_global_px, z, target, CellComp). Unlike the decoder `AnalysisResults` tables of other CosMx releases,
this deposit has no qv / spread / target_call_observations, so those columns are absent.

Conversion (scote.io.convert_cosmx): x, y = x_global_px, y_global_px * 0.120280945 um (the CosMx camera pixel; the
global coordinates of this deposit are sub-pixel, the local ones are rounded to integers and are not used);
z_plane = z (integer plane index, -1 .. 9) and z = z_plane * 0.8 um (9 planes at 0.8 um, paper Methods);
cell_id = fov * 100000 + cell_ID (-1 where cell_ID == 0); nucleus = CellComp == "Nuclear"; codeclass from the target
name; controls are kept. Added here: piece = tissue piece from single-linkage clustering of the FOV corner positions at
0.6 mm (adjacent FOVs are 0.512 mm apart), numbered 1.. in order of decreasing FOV count.

Outputs
  data/processed/cosmx_kidney_gse282059/molecules.parquet             whole slide, one row per molecule
  data/processed/cosmx_kidney_gse282059/piece_<k>/fov_NNNNN.parquet   the analysis piece, one file per FOV (the layout
                                                                       cosmx_resolution.py reads via --processed)
  results/cosmx_kidney/convert_summary.json                           per-piece summary; "analysis_piece" = most in-cell
                                                                       molecules

    python scripts/cosmx_kidney_convert.py --threads 4
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv

import scote.io

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
PX_UM = 0.120280945
Z_STEP_UM = 0.8
SAMPLE = "GSE282059_2B_FFPE"
RAW = Path("data/raw/cosmx_kidney_gse282059")
OUT = Path("data/processed/cosmx_kidney_gse282059")
COMPS = ["Nuclear", "Cytoplasm", "Membrane", "None"]
CLASSES = ["Endogenous", "Negative", "SystemControl", "FalseCode"]
ZMIN, ZMAX = -3, 12


def pieces_from_fov_positions(link_mm: float):
    from scipy.cluster.hierarchy import fcluster, linkage
    with gzip.open(RAW / f"{SAMPLE}_fov_positions_file.csv.gz", "rb") as fh:
        t = pcsv.read_csv(fh)
    fov = t.column("FOV").to_numpy()
    xy = np.c_[t.column("x_global_mm").to_numpy(), t.column("y_global_mm").to_numpy()]
    lab = fcluster(linkage(xy, method="single"), t=link_mm, criterion="distance")
    # renumber 1.. by decreasing FOV count (ties: smallest first FOV)
    labs = sorted(np.unique(lab), key=lambda l: (-(lab == l).sum(), fov[lab == l].min()))
    remap = {l: i + 1 for i, l in enumerate(labs)}
    piece_of = np.zeros(fov.max() + 1, np.int8)
    piece_of[fov] = [remap[l] for l in lab]
    bbox = {}
    for l in labs:
        m = lab == l
        bbox[remap[l]] = dict(x_mm=[float(xy[m, 0].min()), float(xy[m, 0].max() + 0.512)],
                              y_mm=[float(xy[m, 1].min()), float(xy[m, 1].max() + 0.512)])
    return piece_of, bbox, len(labs)


class PieceStats:
    """Per-piece counts of the converted transcripts, accumulated block by block (the transform of convert_cosmx,
    which also adds the piece column)."""

    def __init__(self, piece_of: np.ndarray, n_piece: int):
        self.piece_of, self.P1 = piece_of, n_piece + 1
        self.feat_idx: dict[str, int] = {}
        self.feat_cls: list[int] = []
        self.cnt_feat = np.zeros((self.P1, 8192), np.int64)
        self.cnt_feat_in = np.zeros((self.P1, 8192), np.int64)
        self.cnt_z = np.zeros((self.P1, ZMAX - ZMIN + 1), np.int64)
        self.cnt_comp = np.zeros((self.P1, len(COMPS)), np.int64)
        self.cnt_cls = np.zeros((self.P1, len(CLASSES)), np.int64)
        self.cnt_in = np.zeros(self.P1, np.int64)
        self.cnt_fov = np.zeros(piece_of.size, np.int64)
        self.incell_comp_none = self.outcell_comp_not_none = self.n_rows = 0
        self.cells = [set() for _ in range(self.P1)]

    def __call__(self, t: pa.Table) -> pa.Table:
        P1, NF, NZ = self.P1, self.cnt_feat.shape[1], self.cnt_z.shape[1]
        fov = t.column("fov").to_numpy().astype(np.int64)
        cell_id = t.column("cell_id").to_numpy()
        zp = t.column("z_plane").to_numpy().astype(np.int64)
        feat = t.column("feature").combine_chunks().dictionary_encode()
        for name in feat.dictionary.to_pylist():
            if name not in self.feat_idx:
                self.feat_idx[name] = len(self.feat_idx)
                self.feat_cls.append(CLASSES.index(
                    "Negative" if name.startswith("Negative") else "SystemControl" if name.startswith("SystemControl")
                    else "FalseCode" if name.startswith("FalseCode") else "Endogenous"))
        gidx = np.array([self.feat_idx[n] for n in feat.dictionary.to_pylist()], np.int64)[feat.indices.to_numpy()]
        cls = np.asarray(self.feat_cls, np.int64)[gidx]
        cc = t.column("compartment").combine_chunks().dictionary_encode()
        comp = np.array([COMPS.index(c) for c in cc.dictionary.to_pylist()], np.int64)[cc.indices.to_numpy()]
        pc_ = self.piece_of[fov].astype(np.int64)
        incell = cell_id != -1
        self.cnt_feat += np.bincount(pc_ * NF + gidx, minlength=P1 * NF).reshape(P1, NF)
        self.cnt_feat_in += np.bincount(pc_[incell] * NF + gidx[incell], minlength=P1 * NF).reshape(P1, NF)
        self.cnt_z += np.bincount(pc_ * NZ + np.clip(zp, ZMIN, ZMAX) - ZMIN, minlength=P1 * NZ).reshape(P1, NZ)
        self.cnt_comp += np.bincount(pc_ * len(COMPS) + comp, minlength=P1 * len(COMPS)).reshape(P1, len(COMPS))
        self.cnt_cls += np.bincount(pc_ * len(CLASSES) + cls, minlength=P1 * len(CLASSES)).reshape(P1, len(CLASSES))
        self.cnt_in += np.bincount(pc_[incell], minlength=P1)
        self.cnt_fov += np.bincount(fov, minlength=self.piece_of.size)
        self.incell_comp_none += int((incell & (comp == 3)).sum())
        self.outcell_comp_not_none += int((~incell & (comp != 3)).sum())
        for p in np.unique(pc_[incell]):
            self.cells[p].update(np.unique(cell_id[incell & (pc_ == p)]).tolist())
        self.n_rows += len(fov)
        return t.append_column("piece", pa.array(pc_.astype(np.int8)))


def subpixel_check(block_mb: int) -> list[int]:
    """Distinct (fov, fractional x pixel) pairs and rows of the first csv block: the global coordinates are
    sub-pixel."""
    with gzip.open(RAW / f"{SAMPLE}_tx_file.csv.gz", "rb") as fh:
        reader = pcsv.open_csv(fh, read_options=pcsv.ReadOptions(block_size=block_mb << 20),
                               convert_options=pcsv.ConvertOptions(include_columns=["fov", "x_global_px"]))
        b = reader.read_next_batch()
    fov, xg = b.column("fov").to_numpy(), b.column("x_global_px").to_numpy()
    return [len(np.unique(np.c_[fov, np.round(xg % 1, 6)], axis=0)), len(fov)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--link-mm", type=float, default=0.6)
    ap.add_argument("--block-mb", type=int, default=64)
    ap.add_argument("--summary", default="results/cosmx_kidney/convert_summary.json")
    a = ap.parse_args()
    pa.set_cpu_count(a.threads)
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    Path(a.summary).parent.mkdir(parents=True, exist_ok=True)

    piece_of, bbox, n_piece = pieces_from_fov_positions(a.link_mm)
    print(f"{n_piece} tissue pieces from {int((piece_of > 0).sum())} FOV positions", flush=True)
    stats = PieceStats(piece_of, n_piece)
    scote.io.convert_cosmx(RAW / f"{SAMPLE}_tx_file.csv.gz", OUT, SAMPLE, px_um=PX_UM, z_step_um=Z_STEP_UM,
                           drop_controls=False, block_mb=a.block_mb, transform=stats)
    t_conv = time.time() - t0
    n_rows = stats.n_rows
    print(f"{n_rows:,} molecules, {len(stats.feat_idx):,} targets; conversion {t_conv:.0f} s", flush=True)

    P1 = stats.P1
    fcls = np.asarray(stats.feat_cls)
    nf = len(stats.feat_idx)
    zlab = list(range(ZMIN, ZMAX + 1))
    cnt_feat, cnt_feat_in, cnt_cls, cnt_in, cnt_fov = (stats.cnt_feat, stats.cnt_feat_in, stats.cnt_cls, stats.cnt_in,
                                                       stats.cnt_fov)

    def piece_rec(p):
        tot = int(cnt_cls[p].sum())
        fovs = np.flatnonzero((piece_of == p) & (cnt_fov > 0)) if p > 0 else np.flatnonzero((piece_of == 0) & (cnt_fov > 0))
        endo = cnt_feat[p, :nf] * (fcls == 0)
        return dict(piece=int(p), n_fov=int(len(fovs)), fovs=fovs.tolist(), bbox_mm=bbox.get(int(p)),
                    n_molecules=tot, n_in_cell=int(cnt_in[p]), in_cell_fraction=float(cnt_in[p] / max(tot, 1)),
                    n_endogenous=int(endo.sum()),
                    n_endogenous_in_cell=int((cnt_feat_in[p, :nf] * (fcls == 0)).sum()),
                    n_cells=len(stats.cells[p]),
                    molecules_per_z_plane={str(z): int(v) for z, v in zip(zlab, stats.cnt_z[p]) if v},
                    compartment={c: int(v) for c, v in zip(COMPS, stats.cnt_comp[p])},
                    codeclass={c: int(v) for c, v in zip(CLASSES, cnt_cls[p])},
                    n_genes_detected=int((endo > 0).sum()), n_genes_ge_100=int((endo >= 100).sum()))

    recs = [piece_rec(p) for p in range(1, P1)]
    unassigned = piece_rec(0) if cnt_cls[0].sum() else None
    best = max(recs, key=lambda r: r["n_in_cell"])
    frac_px = subpixel_check(a.block_mb)
    summary = dict(source=f"GEO GSE282059 {SAMPLE}_tx_file.csv.gz", script="scripts/cosmx_kidney_convert.py",
                   px_um=PX_UM, z_step_um=Z_STEP_UM, link_mm=a.link_mm, n_molecules=n_rows,
                   n_targets=nf, n_genes=int((fcls == 0).sum()), n_negative=int((fcls == 1).sum()),
                   n_systemcontrol=int((fcls == 2).sum()), n_falsecode=int((fcls == 3).sum()),
                   n_fov=int((cnt_fov > 0).sum()), n_in_cell=int(cnt_in.sum()),
                   in_cell_fraction=float(cnt_in.sum() / n_rows),
                   check_incell_with_CellComp_None=stats.incell_comp_none,
                   check_outcell_with_CellComp_not_None=stats.outcell_comp_not_none,
                   subpixel_check_first_batch=dict(distinct_fov_fraction=frac_px[0], rows=frac_px[1]),
                   molecules_per_z_plane={str(z): int(v) for z, v in zip(zlab, stats.cnt_z.sum(0)) if v},
                   pieces=recs, molecules_outside_pieces=unassigned, analysis_piece=best["piece"],
                   analysis_piece_rule="most in-cell molecules", runtime_convert_s=round(t_conv, 1))
    Path(a.summary).write_text(json.dumps(summary, indent=1))
    for r in recs:
        print(f"  piece {r['piece']}: {r['n_fov']:3d} FOV, {r['n_molecules'] / 1e6:6.1f} M, in-cell {r['in_cell_fraction']:.3f}, "
              f"{r['n_cells']:,} cells, FOV {r['fovs'][:1]}..{r['fovs'][-1:]}", flush=True)
    print(f"analysis piece {best['piece']}", flush=True)
    # ---- per-FOV files of the analysis piece (the layout cosmx_resolution.py reads) ---------------------
    import polars as pl
    t1 = time.time()
    pdir = OUT / f"piece_{best['piece']}"
    pdir.mkdir(exist_ok=True)
    d = (pl.scan_parquet(OUT / "molecules.parquet").filter(pl.col("piece") == best["piece"]).collect()
         .with_columns([pl.col(c).cast(pl.Categorical) for c in ("feature", "sample", "compartment", "codeclass")]))
    for (f,), s in d.partition_by("fov", as_dict=True, maintain_order=True).items():
        s.write_parquet(pdir / f"fov_{int(f):05d}.parquet", compression="zstd", statistics=True)
    summary["analysis_piece_dir"] = str(pdir)
    summary["runtime_piece_split_s"] = round(time.time() - t1, 1)
    Path(a.summary).write_text(json.dumps(summary, indent=1))
    print(f"wrote {pdir} ({d.height:,} rows, {time.time() - t1:.0f} s) and {a.summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
