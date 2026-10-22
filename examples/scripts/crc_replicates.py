#!/usr/bin/env python
"""Replicate axis of preprint Fig. 2c (CRC P1 / P2 / P5 serial-section sets): (i) Xenium-vs-HD agreement per section
(scripts/platform_overlay.py on the two 2 µm runs), (ii) between-section spread of log g on the same platform
(SD over sections of log g per pair per r bin, median over pairs) compared with the within-section tile-bootstrap SD,
(iii) Pearson of log g between sections. All three use every bin of the common 2 µm lattice with r_from >= R_MIN, i.e. from
the intra-bin shell [0, 2) µm on (no resolution cut; small pair counts enter through the SD < 0.5 filter); every row carries
r_from and r_central (the lattice-shell distance of the npz, 1 µm for the intra-bin shell). Output
results/platforms/crc_replicates.json (+ md table)."""
import json, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]; TP = REPO / "data/twopoint"; RES = REPO / "results/platforms"
SECTIONS = ["p1", "p2", "p5"]
R_MIN = 0.0   # µm, lower bound on r_from of the compared bins (0: from the intra-bin shell on)


def sd_filter(d):
    """Stored closed-form SD for the pair filters (SD < 0.5 and the like), with the pairs without an interval (reliability
    flag 2, T_eff < 5) set to +inf so that every SD filter excludes them."""
    sd = d["sd"]
    return np.where(d["flag"] == 2, np.inf, sd) if "flag" in d.files else sd


def load(path):
    d = np.load(path, allow_pickle=True)
    return dict(feats=list(map(str, d["features"])), lg=d["log_g"], sd=sd_filter(d), r_from=d["r_from"], r=d["r_central"])


def main():
    out = dict(per_section=[], between_sections={})
    have = [s for s in SECTIONS if (TP / f"xenium_crc_{s}_bin2um_common.npz").exists() and (TP / f"hd_crc_{s}_bin2um_common.npz").exists()]
    for s in have:
        summ = RES / f"crc_{s}_xen_hd_summary.json"
        subprocess.run([sys.executable, str(REPO / "scripts/platform_overlay.py"), "--xenium-bin", str(TP / f"xenium_crc_{s}_bin2um_common.npz"),
                        "--hd", str(TP / f"hd_crc_{s}_bin2um_common.npz"), "--out", str(REPO / f"figures/quicklook/crc_{s}_xen_hd.png"), "--summary", str(summ),
                        "--r-min", str(R_MIN)], check=True)
        bins = json.load(open(summ))["comparisons"][0]["bins"]
        d = np.load(TP / f"hd_crc_{s}_bin2um_common.npz", allow_pickle=True); rc = dict(zip(np.round(d["r_from"], 6), d["r_central"]))
        out["per_section"].append(dict(section=s, bins=[{k: b[k] for k in ("r_from", "n_pairs", "pearson", "rms_diff", "frac_within_2sigma")}
                                                       | dict(r_central=float(rc[round(b["r_from"], 6)])) for b in bins]))
    for plat in ("xenium", "hd"):
        runs = {s: load(TP / f"{plat}_crc_{s}_bin2um_common.npz") for s in SECTIONS if (TP / f"{plat}_crc_{s}_bin2um_common.npz").exists()}
        if len(runs) < 2: continue
        common = sorted(set.intersection(*[set(v["feats"]) for v in runs.values()]))
        iu = np.triu_indices(len(common)); rows = []
        ref = next(iter(runs.values()))
        for k, rf in enumerate(ref["r_from"]):
            if rf < R_MIN: continue
            L, S = [], []
            for v in runs.values():
                kk = int(np.argmin(np.abs(v["r_from"] - rf)))
                if abs(v["r_from"][kk] - rf) > 1e-6: break
                idx = [v["feats"].index(g) for g in common]
                L.append(v["lg"][kk][idx][:, idx][iu]); S.append(v["sd"][kk][idx][:, idx][iu])
            if len(L) != len(runs): continue
            L, S = np.array(L), np.array(S)
            ok = np.isfinite(L).all(0) & (S < 0.5).all(0)
            if ok.sum() < 100: continue
            between = L[:, ok].std(0, ddof=1); within = np.sqrt((S[:, ok] ** 2).mean(0))
            pear = [float(np.corrcoef(L[i, ok], L[j, ok])[0, 1]) for i in range(len(L)) for j in range(i + 1, len(L))]
            rows.append(dict(r_from=float(rf), r_central=float(ref["r"][k]), n_pairs=int(ok.sum()), median_between_section_sd=float(np.median(between)), median_within_section_sd=float(np.median(within)),
                             ratio_between_within=float(np.median(between / np.maximum(within, 1e-9))), pearson_between_sections=pear))
        out["between_sections"][plat] = dict(sections=list(runs), rows=rows)
    json.dump(out, open(RES / "crc_replicates.json", "w"), indent=1)
    md = ["| r_from [µm] | " + " | ".join(f"{p['section']}: Pearson / RMS (Xen vs HD)" for p in out["per_section"]) + " | " +
          " | ".join(f"{pl}: between-section SD / within SD (ratio), Pearson between sections" for pl in out["between_sections"]) + " |"]
    md.append("|---|" + "---|" * (len(out["per_section"]) + len(out["between_sections"])))
    for rf in (0, 2, 4, 6, 12, 24, 48, 96, 256, 512, 1024, 2048):
        cells = [f"{rf}"]
        for p in out["per_section"]:
            b = min(p["bins"], key=lambda b: abs(b["r_from"] - rf)); cells.append(f"{b['pearson']:.3f} / {b['rms_diff']:.2f}" if abs(b["r_from"] - rf) <= 0.5 * rf else "—")
        for pl, v in out["between_sections"].items():
            if not v["rows"]: cells.append("—"); continue
            b = min(v["rows"], key=lambda b: abs(b["r_from"] - rf)); cells.append(f"{b['median_between_section_sd']:.3f} / {b['median_within_section_sd']:.3f} ({b['ratio_between_within']:.1f}), " + ", ".join(f"{x:.2f}" for x in b["pearson_between_sections"]) if abs(b["r_from"] - rf) <= 0.5 * rf else "—")
        md.append("| " + " | ".join(cells) + " |")
    open(RES / "crc_replicates.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
