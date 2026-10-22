#!/usr/bin/env python
"""Gene panels of the two-point runs, built from the converted data (one gene per line, no header).

  common    genes present in every input (molecule tables: their `feature` column; bin tables: their `features`),
            sorted alphabetically: the CRC panel (Xenium, Visium HD, Visium v2 of P2; 408 genes) and the spleen panel
            (Xenium, MERSCOPE, Visium HD of WT709; 90 genes)
  features  the features of a two-point run, in the order of the run: the top-200 genes of liver slice 1, used for
            the other three slices
  aviti24   the top-N 3' genes of the AVITI24 screen by count pooled over the wells, restricted to genes present in
            every well; the targeted batch B01 and the mitochondrial aggregates are left out
  matched   the top-N CosMx genes of the kidney piece by count (Endogenous probes, no MT- genes) that are also in a
            given panel, in CosMx order: the panel shared by CosMx and AVITI24 in Fig. S4a

  python scripts/gene_panel.py common --molecules data/processed/xenium_crc_p2/molecules.parquet \
      --bins data/processed/hd_crc_p2/bins_square_002um.npz data/processed/v2_crc_p2/bins_spots55um.npz --out genes.txt
  python scripts/gene_panel.py features --npz data/twopoint/error_scaling/vizgen_liver1slice1_bin2um_top200.npz --out genes.txt
  python scripts/gene_panel.py aviti24 --molecules data/processed/aviti24_{A1,...,F2}/molecules.parquet --top 1000 --out a.txt
  python scripts/gene_panel.py matched --cosmx-dir data/processed/cosmx_kidney_gse282059/piece_2 --panel a.txt \
      --top 1000 --out genes.txt
"""
import argparse
import glob
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

AVITI24_EXCLUDE = ("chrM_umbrella", "chrM_light", "chrM_heavy")


def molecule_features(path):
    """Distinct values of the `feature` column of a molecule table."""
    out = set()
    pf = pq.ParquetFile(path)
    for rg in range(pf.metadata.num_row_groups):
        col = pf.read_row_group(rg, columns=["feature"]).column("feature").combine_chunks()
        if hasattr(col, "dictionary"):
            used = np.unique(col.indices.to_numpy(zero_copy_only=False))
            out.update(col.dictionary.take(used).to_pylist())
        else:
            out.update(pc.unique(col).to_pylist())
    return out


def bin_features(path):
    return {str(f) for f in np.load(path, allow_pickle=True)["features"]}


def read_panel(path):
    return [line.strip() for line in open(path) if line.strip() and not line.startswith("#")]


def write_panel(genes, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(genes) + "\n")
    print(f"{path}: {len(genes)} genes")


def cmd_common(a):
    sets = [molecule_features(p) for p in a.molecules] + [bin_features(p) for p in a.bins]
    write_panel(sorted(set.intersection(*sets)), a.out)


def cmd_features(a):
    write_panel([str(f) for f in np.load(a.npz, allow_pickle=True)["features"]], a.out)


def cmd_aviti24(a):
    pooled, per_well = Counter(), []
    for path in a.molecules:
        counts = Counter()
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg, columns=["feature", "batch"])
            batch = t.column("batch").combine_chunks().dictionary_decode()
            feat = t.column("feature").combine_chunks().dictionary_decode()
            counts.update(feat.filter(pc.not_equal(batch, "B01")).to_pylist())
        for k in AVITI24_EXCLUDE:
            counts.pop(k, None)
        per_well.append(set(counts))
        pooled.update(counts)
        print(f"{path}: {len(counts)} 3' features, {sum(counts.values()):,} molecules", flush=True)
    common = set.intersection(*per_well)
    write_panel([g for g, _ in pooled.most_common() if g in common][: a.top], a.out)


def cmd_matched(a):
    import polars as pl
    files = sorted(glob.glob(f"{a.cosmx_dir}/fov_*.parquet"))
    if not files:
        raise SystemExit(f"no fov_*.parquet in {a.cosmx_dir}")
    counts = (pl.scan_parquet(f"{a.cosmx_dir}/fov_*.parquet").filter(pl.col("codeclass") == "Endogenous")
              .group_by("feature").len().collect().rename({"len": "n"}).sort("n", descending=True)
              .with_columns(pl.col("feature").cast(pl.Utf8)).filter(~pl.col("feature").str.starts_with("MT-")))
    top = counts["feature"].to_list()[: a.top]
    other = set(read_panel(a.panel))
    write_panel([g for g in top if g in other], a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("common"); p.add_argument("--molecules", nargs="*", default=[]); p.add_argument("--bins", nargs="*", default=[])
    p.add_argument("--out", required=True)
    p = sub.add_parser("features"); p.add_argument("--npz", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("aviti24"); p.add_argument("--molecules", nargs="+", required=True); p.add_argument("--top", type=int, default=1000)
    p.add_argument("--out", required=True)
    p = sub.add_parser("matched"); p.add_argument("--cosmx-dir", required=True); p.add_argument("--panel", required=True)
    p.add_argument("--top", type=int, default=1000); p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(common=cmd_common, features=cmd_features, aviti24=cmd_aviti24, matched=cmd_matched)[a.cmd](a)


if __name__ == "__main__":
    main()
