#!/usr/bin/env python
"""Composition pedestal of preprint Supplementary Fig. S4a: CosMx 6k kidney and AVITI24 counted directly in 0.5 um level-0 bins.

Uses scripts/crossplatform_pedestal.py (imported, not copied): the gene panel shared by both platforms (--panel), Endogenous
in-cell molecules only, one block per (FOV, z-plane), stacked same-gene log g against the random-label reference
(same_curve), within-cell label permutation with rng = default_rng(seed + 1000 k), seed 0, k = 0, 1, 2. Delta = 0.5 um,
6 bins (0-3 um) on CosMx; AVITI24 (--aviti24 recompute) is counted on the same panel in 0.5 um bins with the same
permutation seeds.

    python scripts/pedestal_direct.py --threads 8 --aviti24 recompute \
        --panel results/cosmx_kidney/common_genes_matched_aviti24.txt \
        --cosmx-dir data/processed/cosmx_kidney_gse282059/piece_2 --tag kidney_matched \
        --tensor-dir data/twopoint/pedestal_bin05_kidney \
        --out-json results/cosmx_kidney/crossplatform_pedestal_bin05_direct.json
        -> data/twopoint/pedestal_bin05_kidney/*_{observed,perm0,perm1,perm2}.npz   (cached (bins, G, G) counts, totals)
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)                                   # crossplatform_pedestal.py uses repo-relative paths
import crossplatform_pedestal as cp  # noqa: E402

PANEL = "results/cosmx_kidney/common_genes_matched_aviti24.txt"
SPEC = dict(delta=0.5, n_bins=6, tile=cp.SPEC["cosmx"]["tile"])


def pedestal_aviti24(feats, gi, a, rp):
    """AVITI24 observed + within-cell permutations on the given panel, 0.5 um bins, 8 bins (cp.SPEC["aviti24"]); the same
    construction as crossplatform_pedestal.py's "aviti24|all" entry, so the two platforms share the panel."""
    t0 = time.time()
    spec = cp.SPEC["aviti24"]
    G = len(feats)
    blocks = cp.load_aviti24("all", feats, gi)
    n_mol = sum(len(b[4]) for b in blocks)
    n_cell = len(np.unique(np.concatenate([b[3] for b in blocks])))
    print(f"aviti24|all: {G} genes, {len(blocks)} blocks, {n_mol:,} molecules, {n_cell:,} cells; load {time.time() - t0:.0f} s", flush=True)
    C, T = cp.accumulate(blocks, G, spec, a.threads)
    np.savez_compressed(f"{a.tensor_dir}/aviti24_{a.tag}_observed.npz", counts=C.astype(np.float32), totals=T)
    obs = cp.same_curve(C, T)
    perms = []
    for k in range(a.n_perm):
        rng = np.random.default_rng(a.seed + 1000 * k)
        pC, pT = cp.accumulate(blocks, G, spec, a.threads, rng=rng)
        np.savez_compressed(f"{a.tensor_dir}/aviti24_{a.tag}_perm{k}.npz", counts=pC.astype(np.float32), totals=pT)
        perms.append(cp.same_curve(pC, pT))
        print(f"  aviti24 perm {k} {np.array2string(perms[-1], precision=4)}", flush=True)
    P = np.asarray(perms)
    ped, sd = P.mean(0), P.std(0, ddof=1)
    return dict(native_delta_um=spec["delta"], n_bins=spec["n_bins"], tile_um=spec["tile"], n_blocks=len(blocks),
                n_molecules=n_mol, n_cells=int(n_cell), observed=obs.tolist(), pedestal_mean=ped.tolist(),
                pedestal_sd=sd.tolist(), pedestal_perms=P.tolist(), excess=(obs - ped).tolist(), runtime_s=round(time.time() - t0, 1),
                read_point=dict(r_from=0.5, r_to=1.0, observed=float(obs[rp]), pedestal=float(ped[rp]),
                                pedestal_sd=float(sd[rp]), subcellular_share=float(1 - ped[rp] / obs[rp])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-perm", type=int, default=3)
    ap.add_argument("--max-blocks", type=int, default=0, help="smoke test only: first FOV file, first N blocks")
    ap.add_argument("--tensor-dir", default="data/twopoint/pedestal_bin05_kidney")
    ap.add_argument("--out-json", default="results/cosmx_kidney/crossplatform_pedestal_bin05_direct.json")
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--cosmx-dir", default="data/processed/cosmx_kidney_gse282059/piece_2", help="directory of fov_*.parquet")
    ap.add_argument("--tag", default="kidney_matched", help="prefix of the cached tensor files")
    ap.add_argument("--aviti24", choices=("recompute", "skip"), default="recompute",
                    help="recompute: count AVITI24 on --panel in 0.5 um bins, same permutation seeds; skip")
    a = ap.parse_args()

    feats = [l.strip() for l in Path(a.panel).read_text().splitlines() if l.strip() and not l.startswith("#")]
    gi = {g: i for i, g in enumerate(feats)}
    G = len(feats)
    t0 = time.time()
    if a.max_blocks:                                             # smoke test: read only the first FOV file
        _glob = cp.glob.glob
        cp.glob.glob = lambda pat: sorted(_glob(pat))[:1]
    blocks = cp.load_cosmx("all", feats, gi, root=a.cosmx_dir)
    if a.max_blocks:
        blocks = blocks[: a.max_blocks]
    n_mol = sum(len(b[4]) for b in blocks)
    n_cell = len(np.unique(np.concatenate([b[3] for b in blocks])))
    print(f"cosmx|all: {G} genes, {len(blocks)} blocks, {n_mol:,} molecules, {n_cell:,} cells; load {time.time() - t0:.0f} s", flush=True)
    Path(a.tensor_dir).mkdir(parents=True, exist_ok=True)

    t1 = time.time()
    C, T = cp.accumulate(blocks, G, SPEC, a.threads)
    np.savez_compressed(f"{a.tensor_dir}/{a.tag}_observed.npz", counts=C.astype(np.float32), totals=T)
    obs = cp.same_curve(C, T)
    print(f"  observed {np.array2string(obs, precision=4)}  ({time.time() - t1:.0f} s)", flush=True)
    perms = []
    for k in range(a.n_perm):
        t1 = time.time()
        rng = np.random.default_rng(a.seed + 1000 * k)
        pC, pT = cp.accumulate(blocks, G, SPEC, a.threads, rng=rng)
        np.savez_compressed(f"{a.tensor_dir}/{a.tag}_perm{k}.npz", counts=pC.astype(np.float32), totals=pT)
        perms.append(cp.same_curve(pC, pT))
        print(f"  perm {k}   {np.array2string(perms[-1], precision=4)}  ({time.time() - t1:.0f} s)", flush=True)
    P = np.asarray(perms)
    ped, sd = P.mean(0), P.std(0, ddof=1)

    rp = 1                                                       # read point: bin [0.5, 1.0) um
    t_cosmx = time.time() - t0
    if a.aviti24 == "recompute":
        cells, src_av = {"aviti24": pedestal_aviti24(feats, gi, a, rp)}, a.tensor_dir + " (AVITI24, recomputed on this panel)"
    else:
        cells, src_av = {}, "AVITI24 not computed"
    out = dict(panel=a.panel, source=a.tensor_dir + " (CosMx, " + a.cosmx_dir + "); " + src_av, delta_um=SPEC["delta"], n_genes=G,
               n_perm=a.n_perm, seed=a.seed,
               construction="per-slice, in-cell molecules only, within-cell label permutation; both platforms counted directly in "
                            "0.5 um level-0 bins (no pooling)",
               script="scripts/pedestal_direct.py", runtime_s=round(time.time() - t0, 1),
               runtime_cosmx_s=round(t_cosmx, 1), cells=cells)
    out["cells"]["cosmx"] = dict(native_delta_um=SPEC["delta"], n_bins=SPEC["n_bins"], tile_um=SPEC["tile"], n_blocks=len(blocks),
                                 n_molecules=n_mol, n_cells=int(n_cell),
                                 observed=obs.tolist(), pedestal_mean=ped.tolist(), pedestal_sd=sd.tolist(),
                                 pedestal_perms=P.tolist(), excess=(obs - ped).tolist(),
                                 read_point=dict(r_from=0.5, r_to=1.0, observed=float(obs[rp]), pedestal=float(ped[rp]),
                                                 pedestal_sd=float(sd[rp]), subcellular_share=float(1 - ped[rp] / obs[rp])))
    if a.max_blocks:
        a.out_json = a.out_json.replace(".json", f"_smoke{a.max_blocks}.json")
    Path(a.out_json).write_text(json.dumps(out, indent=1))
    print("read point", out["cells"]["cosmx"]["read_point"])
    print("wrote", a.out_json, f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
