#!/usr/bin/env python
"""Preprint Fig. 3 (molecule level): operator labels per molecule on the SpatialBench spleen Xenium sections, the
molecule-level two-point run, and the three-group comparison (WT, Tbx21-KO, Ctrl).

  run       operator labels from config/operators_spleen.yaml (gene-defined labels, and score-defined labels resolved by a
            smoothed marker fraction), then scote.twopoint (level 0 at delta = 0.5 um + lattice levels, 500 um
            tiles, per-tile counts kept) -> data/twopoint/spleen_operators/<section>.npz and results/spleen_operators/<section>.json
  compare3  every group of the YAML: per-section log g_ab(r) with its closed-form tile block bootstrap uncertainty, group
            means, and the contrasts WT - KO, Ctrl - WT, Ctrl - KO, WT+Ctrl - KO as difference +- SE with z
            -> results/spleen_operators/three_groups.{json,md}

  python scripts/spleen_operators.py run --section sb_xenium_WT713 --threads 4
  python scripts/spleen_operators.py compare3
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parents[1]
OPS = yaml.safe_load(open(REPO / "config/operators_spleen.yaml"))
OUT = REPO / "data/twopoint/spleen_operators"
RES = REPO / "results/spleen_operators"
QL = REPO / "figures/quicklook"


def feature_group(names, a):
    """Indices of the features named `a` (a list, so that a pair of groups can be collapsed in one call)."""
    return [i for i, n in enumerate(names) if n == a]


def smoothed_fraction(xy, is_marker, origin, shape, grid, sigma):
    ij = np.floor((xy - origin) / grid).astype(np.int64)
    flat = ij[:, 0] * shape[1] + ij[:, 1]
    tot = gaussian_filter(np.bincount(flat, minlength=shape[0] * shape[1]).reshape(shape).astype(np.float32), sigma / grid)
    mk = gaussian_filter(np.bincount(flat[is_marker], minlength=shape[0] * shape[1]).reshape(shape).astype(np.float32), sigma / grid)
    return (mk / np.maximum(tot, 1e-6))[ij[:, 0], ij[:, 1]]


def label_molecules(xy, gene):
    """Operator label per molecule ('' = no operator). Returns (labels, info dict)."""
    grid, sigma = OPS["grid_um"], OPS["sigma_um"]
    origin = xy.min(0) - grid; shape = tuple(np.floor((xy.max(0) - origin) / grid).astype(int) + 2)
    lab = np.full(len(xy), "", dtype=object); info = {}
    for name, op in OPS["operators"].items():
        m = np.isin(gene, op["genes"])
        if op["rule"] == "gene":
            lab[m] = name; info[name] = dict(n=int(m.sum()))
        else:
            sc = smoothed_fraction(xy, np.isin(gene, op["score_genes"]), origin, shape, grid, sigma)
            thr = np.quantile(sc[m], op["quantile"])
            hi = m & (sc >= thr)
            lab[hi] = name; lab[m & ~hi] = op["else"]
            info[name] = dict(n=int(hi.sum()), n_else=int((m & ~hi).sum()), threshold_fraction=float(thr), else_label=op["else"])
    return lab, info


def stage_run(a):
    import pyarrow.parquet as pq
    from scote import CPUOptions, twopoint
    t0 = time.time()
    tbl = pq.read_table(REPO / f"data/processed/{a.section}/molecules.parquet", columns=["x", "y", "feature"])
    xy = np.stack([tbl["x"].to_numpy(), tbl["y"].to_numpy()], 1).astype(np.float64); gene = tbl["feature"].to_pandas().astype(str).to_numpy()
    lab, info = label_molecules(xy, gene)
    keep = lab != ""
    xy, lab = xy[keep], lab[keep]
    names = sorted(set(lab)); codes = pd.Categorical(lab, categories=names).codes.astype(np.int64)
    print(f"{a.section}: {keep.sum():,} operator molecules of {len(keep):,}; labels {names}; {time.time()-t0:.0f}s", flush=True)
    res = twopoint(xy, codes, names=names, delta=0.5, shells_per_level=8, n_levels=a.n_levels, tile_um=a.tile_um,
                   backend=CPUOptions(mode=a.cpu_mode), threads=a.threads or None, keep_tiles=True)
    OUT.mkdir(parents=True, exist_ok=True)
    res.save(OUT / f"{a.section}.npz", compress=True)
    summ = dict(section=a.section, n_molecules_total=int(len(keep)), n_operator_molecules=int(keep.sum()), operators=info,
                labels=names, totals={n: float(t) for n, t in zip(names, res.totals)}, n_tiles=res.n_tiles, timings=res.timings,
                seconds=time.time() - t0)
    RES.mkdir(parents=True, exist_ok=True)
    json.dump(summ, open(RES / f"{a.section}.json", "w"), indent=1, default=float)


# ---------------------------------------------------------------- three groups (WT, KO, Ctrl)
PAIRS3 = [("B", "T"), ("T", "FDC"), ("Tfh", "FDC"), ("B", "FDC"), ("GCB", "FDC"), ("GCB", "Tfh"), ("B", "B"), ("GCB", "GCB")]
CONTRASTS3 = [("WT", "KO"), ("Ctrl", "WT"), ("Ctrl", "KO"), ("WT+Ctrl", "KO")]   # Ctrl - WT is the no-difference-expected calibration
GROUP_COL = {"WT": "#0072B2", "KO": "#D55E00", "Ctrl": "#009E73"}
TARGETS3 = (2, 5, 10, 20, 50, 100, 200)
Z95 = 1.959963984540054   # norm.ppf(0.975): intervals log g +- Z95 sd


def merged_closed_form(ct, tt, ia, ib, with_cov=False):
    """log g_ab(r) of a stored tile tensor for feature groups a and b (pair counts and totals of the members of a and of b
    summed; the section's P_tot and n kept as the reference) and its closed-form tile block bootstrap uncertainty.

    ct (T, K, G, G), tt (T, G); ia, ib: feature indices of a and b (disjoint, or identical for a self pair).
    The uncertainty is the closed form on the collapsed partition {a, b, rest} (`scote.uncertainty.collapse_tensor`,
    `scote.uncertainty.closed_form_log_g`): the rest group keeps every tile's full P_tot and n, so the collapsed random-label
    reference is that of the full tensor. The collapsed point estimate must reproduce the direct one (checked;
    `max_abs_diff_collapsed` reports the difference).
    Returns dict(log_g, sd, lo, hi, t_eff, flag, max_abs_diff_collapsed[, cov, influence]): lo / hi = log g -+ 1.96 sd, NaN
    where the reliability flag is 2 (T_eff < 5); flag 1 (5 <= T_eff < 20) keeps the interval. with_cov: also the (K, K)
    closed-form cross-shell covariance of log g (sum_t l_t(k) l_t(k'), `scote.uncertainty.cross_shell_cov`) and the per-tile
    influence (T, K) it is built from."""
    from scote import uncertainty as TV
    ia, ib = list(ia), list(ib)
    T = ct.shape[0]
    if set(ia) & set(ib) and sorted(ia) != sorted(ib):
        raise ValueError("feature groups must be disjoint or identical")
    sub = ct[:, :, ia][:, :, :, ib].sum(axis=(2, 3)); subtot = ct.sum(axis=(2, 3))
    w = np.ones(T); Pb, Ptb, nbt = w @ sub, w @ subtot, w @ tt
    with np.errstate(all="ignore"):
        lg = np.log(Pb / (Ptb * nbt[ia].sum() * nbt[ib].sum() / nbt.sum() ** 2))
    labels = np.full(ct.shape[2], -1, dtype=np.int64); labels[ia] = 0
    B = 0 if sorted(ia) == sorted(ib) else 1
    labels[ib] = B
    Cc, Nc = TV.collapse_tensor(ct, tt, labels)
    res = TV.closed_form_log_g(Cc, Nc, min_tiles_warn=0)
    lg_c = res.log_g[:, 0, B]; ok = np.isfinite(lg) & np.isfinite(lg_c)
    if not np.array_equal(np.isfinite(lg), np.isfinite(lg_c)) or (ok.any() and np.abs(lg[ok] - lg_c[ok]).max() > 1e-10):
        raise AssertionError("collapsed point estimate does not reproduce the merged log g")
    sd = res.sd[:, 0, B]; flag = res.flag[:, 0, B].astype(int)
    lo, hi = lg - Z95 * sd, lg + Z95 * sd
    lo[flag == 2] = np.nan; hi[flag == 2] = np.nan
    out = dict(log_g=lg, sd=sd, lo=lo, hi=hi, t_eff=np.asarray(res.t_eff[:, 0, B], dtype=np.float64), flag=flag,
               max_abs_diff_collapsed=float(np.abs(lg[ok] - lg_c[ok]).max()) if ok.any() else 0.0)
    if with_cov:
        l = TV.influence_log_g(Cc, Nc)
        out["cov"] = TV.cross_shell_cov(l, (0, B)); out["influence"] = np.ascontiguousarray(l[:, :, 0, B])   # (K, K), (T, K)
    return out


def merged_log_g(d, ga, gb, with_cov=False):
    """log g_ab(r) of a stored run (features ga, gb) with the closed-form tile block bootstrap uncertainty
    (`merged_closed_form`); None if a feature is absent."""
    feats = list(map(str, d["features"])); ia, ib = feature_group(feats, ga), feature_group(feats, gb)
    if not ia or not ib: return None
    return merged_closed_form(d["counts_tiles"], d["totals_tiles"], ia, ib, with_cov=with_cov)


def group_mean_var(vals, sds):
    """Group mean and variance of the mean: mean tile-bootstrap variance / n + replicate variance / n (n = sections)."""
    vals, sds = np.asarray(vals, float), np.asarray(sds, float); n = len(vals)
    v = np.nanmean(sds ** 2) / n + (np.var(vals, ddof=1) / n if n > 1 else 0.0)
    return float(np.nanmean(vals)), float(v)


def contrast(vals1, sds1, vals2, sds2):
    m1, v1 = group_mean_var(vals1, sds1); m2, v2 = group_mean_var(vals2, sds2); err = float(np.sqrt(v1 + v2))
    return dict(diff=m1 - m2, err=err, z=(m1 - m2) / err if err > 0 else None)


def fmt_c(c):
    return f"{c['diff']:+.2f} ± {c['err']:.2f} (z {c['z']:+.1f})" if c and c["z"] is not None else "—"


def stage_compare3(a):
    """All groups of OPS['sections'] (WT, KO, Ctrl): per-section pairwise log g_ab(r), group means,
    and every contrast of CONTRASTS3 as difference +- SE with z at the r values of TARGETS3.
    Writes results/spleen_operators/three_groups.json, .md and figures/quicklook/spleen_three_groups.png."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    groups = list(OPS["sections"])
    secs = {g: [s for s in OPS["sections"][g] if (OUT / f"{s}.npz").exists()] for g in groups}
    data = {s: np.load(OUT / f"{s}.npz", allow_pickle=True) for g in secs for s in secs[g]}
    if not data:
        raise SystemExit(f"no two-point runs in {OUT}")
    res = dict(sections=secs, contrasts=[c for c in CONTRASTS3], uncertainty="closed", pairs={}, summary=[])
    ncol = len(PAIRS3)
    fig, axs = plt.subplots(1, ncol, figsize=(3.1 * ncol, 3.2), squeeze=False)
    for k, (ga, gb) in enumerate(PAIRS3):
        ax = axs[0, k]; rows = {}
        for g in secs:
            for s in secs[g]:
                m = merged_log_g(data[s], ga, gb)
                if m is None: continue
                lg, sd = m["log_g"], m["sd"]
                r = data[s]["r_central"]
                ax.plot(r, lg, color=GROUP_COL.get(g, "k"), lw=1, alpha=0.9, label=f"{g} {s.split('_')[2]}"); ax.fill_between(r, lg - sd, lg + sd, color=GROUP_COL.get(g, "k"), alpha=0.12)
                rows[s] = dict(group=g, r_um=r.tolist(), log_g=lg.tolist(), sd=sd.tolist(), t_eff=m["t_eff"].tolist(), flag=m["flag"].tolist())
        ax.set_xscale("log"); ax.axhline(0, c="k", lw=0.5); ax.set_title(f"g_{ga},{gb}", fontsize=9); ax.set_xlabel("r [µm]")
        if k == 0: ax.legend(fontsize=6, frameon=False); ax.set_ylabel("log g (closed-form tile bootstrap ±1 SD)")
        res["pairs"][f"{ga}:{gb}"] = rows
    for ax in axs.ravel():
        if not ax.lines: ax.axis("off")
    # summary: group values and all contrasts at the standard r grid
    for key, rows in res["pairs"].items():
        if not rows: continue
        vkey = "log_g"
        r = np.array(next(iter(rows.values()))["r_um"])
        for target in TARGETS3:
            k = int(np.argmin(np.abs(r - target)))
            vals = {g: [rows[s][vkey][k] for s in rows if rows[s]["group"] == g] for g in groups}
            sds = {g: [rows[s]["sd"][k] for s in rows if rows[s]["group"] == g] for g in groups}
            pooled = lambda name: (sum((vals[x] for x in name.split("+")), []), sum((sds[x] for x in name.split("+")), []))
            entry = dict(quantity=key, r_um=float(r[k]), values={g: [float(v) for v in vals[g]] for g in groups if vals[g]},
                         means={g: group_mean_var(vals[g], sds[g])[0] for g in groups if vals[g]}, contrasts={})
            # reliability flag of the closed-form interval (2: T_eff < 5, no interval; such entries get no contrast)
            entry["flag_max"] = int(max(rows[s]["flag"][k] for s in rows))
            if entry["flag_max"] == 2:
                res["summary"].append(entry); continue
            for g1, g2 in CONTRASTS3:
                v1, s1 = pooled(g1); v2, s2 = pooled(g2)
                if not v1 or not v2: continue
                entry["contrasts"][f"{g1}-{g2}"] = contrast(v1, s1, v2, s2)
            res["summary"].append(entry)
    RES.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(RES / "three_groups.json", "w"), indent=1, default=float)
    fig.suptitle("SpatialBench spleen Xenium: WT (blue) vs Tbx21-KO (red) vs Ctrl (green), molecule-level operators", fontsize=10)
    fig.tight_layout(); QL.mkdir(parents=True, exist_ok=True); fig.savefig(QL / "spleen_three_groups.png", dpi=140)
    n = {g: len(secs[g]) for g in groups}
    lines = [f"# Three-group comparison (molecule level; n = {', '.join(f'{g} {n[g]}' for g in groups)}; difference ± SE with z; SE² = Σ_groups [mean tile-bootstrap variance / n + replicate variance / n]; pooled WT+Ctrl treated as one group of n = {n.get('WT', 0) + n.get('Ctrl', 0)})",
             "", "| quantity | r [µm] | " + " | ".join(f"{g} ({n[g]})" for g in groups) + " | " + " | ".join(f"{g1} − {g2}" for g1, g2 in CONTRASTS3) + " |",
             "|---|---|" + "---|" * (len(groups) + len(CONTRASTS3))]
    for e in res["summary"]:
        cells = [", ".join(f"{v:+.2f}" for v in e["values"].get(g, [])) or "—" for g in groups]
        cells += [fmt_c(e["contrasts"].get(f"{g1}-{g2}")) for g1, g2 in CONTRASTS3]
        lines.append(f"| {e['quantity']} | {e['r_um']:.1f} | " + " | ".join(cells) + " |")
    (RES / "three_groups.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["run", "compare3"])
    ap.add_argument("--section"); ap.add_argument("--cpu-mode", default="auto", help="CPU product strategy of the lattice levels")
    ap.add_argument("--n-levels", type=int, default=8); ap.add_argument("--tile-um", type=float, default=500.0)
    ap.add_argument("--threads", type=int, default=0, help="numba threads (0 = scote's default)")
    a = ap.parse_args()
    dict(run=stage_run, compare3=stage_compare3)[a.stage](a)


if __name__ == "__main__":
    main()
