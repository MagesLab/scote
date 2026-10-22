#!/usr/bin/env python
"""Preprint Fig. 3: added value of molecule-level functions over cell-level baselines on the SpatialBench spleen.
Regenerates figures/fig3_added_value.{pdf,png} (17.5 cm wide, all text >= 7 pt) from results/spleen_operators/ and writes
results/spleen_operators/fig3_bracket_10um.json (every number of panel d). Panels:
  (a) B-T on the typical WT section (`typical_section`: the WT section whose molecule- and centroid-level B-T curves lie closest to
      the WT-group medians), molecule level solid, centroid level dashed, Squidpy co-occurrence open squares at the outer edge r of
      each distance interval (Squidpy 1.8.3 `co_occurrence` counts the pairs within distance r, so each value is cumulative over 0-r);
  (b) 95 % CI width of log g vs r, median over the nine sections (lines) and interquartile range (bands), style = level; the centroid
      curve runs down to the smallest bin in which the majority of sections have an interval (T_eff >= 5);
  (c) molecule-level B-T per section and group mean +- error by group (WT, Tbx21-KO, Ctrl; 3 + 3 + 3 sections;
      results/spleen_operators/three_groups.json), dotted line at the 10 um read point of (d);
  (d) per-section B-T values at 10 um for Squidpy co-occurrence (log of the co-occurrence ratio of (a) for the pairs within 10 um,
      `SQ_INTERVAL`), the cell-centroid function, and the molecule-level function (bin 10-12 um), grouped WT / KO / Ctrl, on one shared
      y range, with brackets carrying z of WT - KO, Ctrl - KO, Ctrl - WT (error model of Methods "Replicate groups").
Pairs with score-defined operators (FDC, Tfh) are drawn only if one of them passes the knockout calibration rule at 10 um at molecule
level (`OPERATOR_RULE`); otherwise the figure shows B-T only. Colors: groups (WT/KO/Ctrl) in (a), (c), (d); line style = level.
The bin edges of (d) are read from the two-point runs in data/twopoint/spleen_operators/ (scripts/spleen_operators.py run), otherwise
from an existing results/spleen_operators/fig3_bracket_10um.json.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results/spleen_operators"; DATA = REPO / "data/twopoint/spleen_operators"
C = dict(mol="#0072B2", cen="#E69F00", sq="#009E73", WT="#0072B2", KO="#D55E00", Ctrl="#009E73", ln="#CC79A7", gray="#7f7f7f")
GROUPS = ("WT", "KO", "Ctrl")
CELL_D = 8.0    # µm, typical cell diameter: centroid-level functions are not interpretable below it
R_MAX = 500.0
LEGFS = 7.0     # legend / in-panel text size [pt]
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "legend.fontsize": 6.0, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})


def jload(p):
    p = Path(p); return json.load(open(p)) if p.exists() else None


def group_band(rows, key):
    """Per-group mean and error (tile-bootstrap variance / n + replicate variance / n) vs r from the operator-run rows."""
    out = {}
    for g in GROUPS:
        vals = np.array([rows[s][key] for s in rows if rows[s]["group"] == g], dtype=float)
        sds = np.array([rows[s]["sd"] for s in rows if rows[s]["group"] == g], dtype=float)
        if len(vals) == 0: continue
        n = len(vals); m = np.nanmean(vals, 0)
        err = np.sqrt(np.nanmean(sds ** 2, 0) / n + (np.nanvar(vals, 0, ddof=1) / n if n > 1 else 0))
        out[g] = (m, err, vals)
    return out


CPAIR = {"B:T": "#000000", "T:FDC": "#CC79A7", "Tfh:FDC": "#E69F00"}    # preprint (b): color = pair (groups keep their own colors in c)


def typical_section(fn, pair="B:T", group="WT"):
    """The section of `group` whose molecule- and centroid-level log g(r) of `pair` lie closest to the group medians
    (RMS over 1–500 µm for molecules, 8–500 µm for centroids, combined in quadrature)."""
    per = fn[pair]; secs = [s for s in per if f"_{group}" in s]; d2 = np.zeros(len(secs))
    r = np.array(per[secs[0]]["molecules"]["r_um"])
    for level, rmin in (("molecules", 1.0), ("centroids", CELL_D)):
        ok = (r >= rmin) & (r <= R_MAX); M = np.array([per[s][level]["log_g"] for s in secs], float)[:, ok]
        d2 += np.sqrt(np.nanmean((M - np.nanmedian(M, 0)) ** 2, 1)) ** 2
    return secs[int(np.argmin(d2))], dict(zip(secs, np.sqrt(d2)))


def panel_a_bt(ax):
    """Preprint (a): B–T on a typical WT section at molecule level (solid) and cell-centroid level (dashed), Squidpy co-occurrence."""
    fn = jload(RES / "baseline_functions.json"); sec, dist = typical_section(fn); lv = fn["B:T"][sec]
    print("panel a: typical WT section", sec, {k: round(float(v), 3) for k, v in dist.items()})
    for level, ls, a_band in (("molecules", "-", 0.25), ("centroids", "--", 0.13)):
        r = np.array(lv[level]["r_um"]); lg = np.array(lv[level]["log_g"]); lo = np.array(lv[level]["lo"]); hi = np.array(lv[level]["hi"])
        ok = np.isfinite(lg) & (r <= R_MAX) & ((r >= CELL_D) if level == "centroids" else True)
        ax.fill_between(r[ok], lo[ok], hi[ok], color=C["WT"], alpha=a_band, lw=0)
        ax.plot(r[ok], lg[ok], ls, color=C["WT"], lw=1.1)
    sq = jload(RES / f"squidpy_{sec}.json")
    if sq is not None:
        ti = {t: i for i, t in enumerate(sq["types"])}; occ = np.array(sq["co_occurrence"]["occ"]); iv = np.array(sq["co_occurrence"]["interval_um"])
        re = iv[1:]; y = np.log(occ[ti["B"], ti["T"]])    # value k counts the pairs within iv[k + 1]
        ax.plot(re[re <= R_MAX], y[re <= R_MAX], "s", ms=2.2, color=C["WT"], mfc="none", mew=0.7)
    ax.axvspan(0.3, CELL_D, color=C["gray"], alpha=0.12, lw=0)
    ax.axhline(0, color="k", lw=0.4); ax.set_xscale("log"); ax.set_xlim(0.3, R_MAX * 1.1); ax.set_xlabel("r [µm]", labelpad=1); ax.set_ylabel("log g_B,T(r)")
    lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_, hi_ + 0.40 * (hi_ - lo_))
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="k", lw=1.1, label="molecules"), Line2D([], [], color="k", lw=1.1, ls="--", label="cell centroids"),
         Line2D([], [], color="k", ls="", marker="s", ms=2.5, mfc="none", mew=0.7, label="Squidpy co-occurrence")]
    ax.legend(handles=h, frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0.1, fontsize=LEGFS, labelspacing=0.25)
    ax.set_title(f"B – T, {sec.split('_')[2]}", pad=3)


CEN_MIN_SECTIONS = 5    # (b): a centroid bin is drawn if at least this many of the nine sections have an interval there (majority)


def panel_b_bands(ax, pairs=("B:T",)):
    """Preprint (b): 95 % CI width, median over the nine sections (lines) and interquartile range over sections (bands); style = level.
    Molecules from 1 µm; centroids wherever the majority of sections have a closed-form interval (reliability flag < 2, T_eff >= 5)."""
    fn = jload(RES / "baseline_functions.json"); from matplotlib.lines import Line2D
    ext = {}
    for pair in pairs:
        col = CPAIR[pair] if len(pairs) > 1 else "k"; per = fn.get(pair, {})
        for level, ls in (("molecules", "-"), ("centroids", "--")):
            W = []
            for s, lv in per.items():
                r = np.array(lv[level]["r_um"]); W.append(np.array(lv[level]["hi"]) - np.array(lv[level]["lo"]))
            W = np.array(W); W[~(W > 0)] = np.nan; n_ok = np.isfinite(W).sum(0)
            with np.errstate(all="ignore"):
                import warnings; warnings.simplefilter("ignore", RuntimeWarning)
                w = np.nanmedian(W, 0); q1, q3 = np.nanpercentile(W, 25, 0), np.nanpercentile(W, 75, 0)
            ok = (r <= R_MAX) & np.isfinite(w) & ((n_ok >= CEN_MIN_SECTIONS) if level == "centroids" else (r >= 1))
            ax.fill_between(r[ok], q1[ok], q3[ok], color=col, alpha=0.13 if level == "molecules" else 0.10, lw=0)
            ax.plot(r[ok], w[ok], ls=ls, color=col, lw=1.0)
            ext[f"{pair} {level}"] = float(r[ok].min())
    print("panel b: smallest r drawn", ext)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.8, R_MAX * 1.1); ax.set_xlabel("r [µm]", labelpad=1); ax.set_ylabel("95 % CI width of log g", labelpad=1)
    ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1]); ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1"]); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0.04, 2.2)
    h = [Line2D([], [], color="k", lw=1.0, label="molecules"), Line2D([], [], color="k", lw=1.0, ls="--", label="cell centroids")]
    if len(pairs) > 1: h = [Line2D([], [], color=CPAIR[p], lw=1.6, label=p.replace(":", "–")) for p in pairs] + h
    ax.legend(handles=h, frameon=False, ncol=2 if len(pairs) > 1 else 1, loc="upper right", handlelength=1.6, columnspacing=0.8, borderaxespad=0.1, fontsize=LEGFS, labelspacing=0.2)
    ax.set_title(("B – T, " if len(pairs) == 1 else "") + f"CI width, {len(fn['B:T'])} sections", pad=3)


# ---- panel d: per-section values at 10 µm with significance brackets -------------------------------------------------------------
BIN_10 = 13                  # index of the bin containing 10 µm in baseline_functions.json (edges 10–12 µm, see bin_edges())
CONTRASTS_D = (("WT", "KO"), ("Ctrl", "KO"), ("Ctrl", "WT"))
OPERATOR_PAIRS = ("T:FDC", "Tfh:FDC", "B:FDC")
OPERATOR_RULE = "WT-KO and Ctrl-KO both |z| >= 2 with the same sign and |z(Ctrl-WT)| < 2 (Methods 'Replicate groups')"
SQ_INTERVAL = 0              # Squidpy co_occurrence value read in (d): index 0 = the pairs within interval_um[1] = 10 µm


def bin_edges(k=BIN_10):
    """Edges of bin k from the run's .npz; fallback: the edges recorded in an existing bracket JSON."""
    for f in sorted(DATA.glob("sb_xenium_*.npz")):
        d = np.load(f, allow_pickle=True)
        if "r_from" in d.files and "r_to" in d.files: return float(d["r_from"][k]), float(d["r_to"][k]), float(d["r_central"][k])
    old = jload(RES / "fig3_bracket_10um.json")
    if old is not None: b = old["bin"]; return b["r_from_um"], b["r_to_um"], b["r_central_um"]
    return np.nan, np.nan, np.nan


def _contrast(v1, s1, v2, s2):
    """Group contrast of Methods 'Replicate groups' (spleen_operators.contrast): diff of group means, error^2 = sum over the two
    groups of (mean bootstrap variance / n + between-section variance / n); plus Welch's two-sided t-test on the per-section values."""
    from scipy import stats
    def mv(v, s):
        v, s = np.asarray(v, float), np.asarray(s, float); n = len(v)
        return v.mean(), np.nanmean(s ** 2) / n + np.var(v, ddof=1) / n
    m1, var1 = mv(v1, s1); m2, var2 = mv(v2, s2); err = float(np.sqrt(var1 + var2)); t = stats.ttest_ind(v1, v2, equal_var=False)
    a1, a2 = np.var(v1, ddof=1) / len(v1), np.var(v2, ddof=1) / len(v2)
    df = (a1 + a2) ** 2 / (a1 ** 2 / (len(v1) - 1) + a2 ** 2 / (len(v2) - 1))
    return dict(diff=float(m1 - m2), err=err, z=float((m1 - m2) / err), welch_t=float(t.statistic), welch_df=float(df), welch_p=float(t.pvalue))


def calibration(cs):
    a, b, c = cs["WT-KO"]["z"], cs["Ctrl-KO"]["z"], cs["Ctrl-WT"]["z"]
    if abs(a) >= 2 and abs(b) >= 2 and np.sign(a) == np.sign(b): return "passed" if abs(c) < 2 else "failed"
    return "failed" if (abs(a) >= 2 or abs(b) >= 2) else "no effect"


def bracket_data(write=True):
    """Per-section values at 10 µm for Squidpy (log co-occurrence ratio within 10 µm, from squidpy_<section>.json), centroids and
    molecules (log g, closed-form tile-bootstrap SD), group contrasts, nine-section medians; operator pairs at molecule level for the
    drop/keep decision. Centroid and molecule contrasts are cross-checked against baselines_compare.json (Supplementary Table S2)."""
    fn = jload(RES / "baseline_functions.json"); bc = jload(RES / "baselines_compare.json")
    groups = {s: q["group"] for s, q in bc["squidpy"].items()}
    r_from, r_to, r_c = bin_edges()
    sq = {s: jload(RES / f"squidpy_{s}.json") for s in groups}
    iv = np.array(next(iter(sq.values()))["co_occurrence"]["interval_um"]); sq_r = float(iv[SQ_INTERVAL + 1])
    assert all(np.allclose(q["co_occurrence"]["interval_um"], iv) for q in sq.values())
    out = dict(description="Fig. 3d (label fig:2): B-T at 10 um per section, by method and group; contrasts with the error model of Methods "
               "'Replicate groups' (z = diff/err, err^2 = sum_groups [mean(sd_boot^2)/n + var_between/n]) and Welch's two-sided t-test. "
               f"Squidpy: log of the co_occurrence ratio of panel a (Squidpy 1.8.3, 50,000-cell subsample per section) at interval index "
               f"{SQ_INTERVAL}, i.e., the cell pairs within {sq_r:g} um (Squidpy counts the pairs with d <= the interval's outer edge, so the "
               f"ratio is cumulative over 0-{sq_r:g} um); chosen as the Squidpy value closest to the 10-12 um bin of the other two methods "
               "(the next value, within 20 um, is dominated by pairs at 10-20 um). No per-section error: sd = 0.",
               sources=["results/spleen_operators/baseline_functions.json", "results/spleen_operators/baselines_compare.json",
                        "results/spleen_operators/squidpy_<section>.json"],
               bin=dict(index=BIN_10, r_from_um=r_from, r_to_um=r_to, r_central_um=r_c),
               squidpy_interval=dict(index=SQ_INTERVAL, r_within_um=sq_r, interval_um=iv.tolist()), operator_rule=OPERATOR_RULE, methods={})
    def method_entry(pair, level):
        rows = []
        for s in groups:
            if level == "squidpy":
                q = sq[s]; ti = {t: i for i, t in enumerate(q["types"])}; a, b = pair.split(":")
                v = float(np.log(np.array(q["co_occurrence"]["occ"])[ti[a], ti[b], SQ_INTERVAL]))
                rows.append(dict(section=s, group=groups[s], value=v, sd=0.0, lo=None, hi=None, n_cells=int(q["co_occurrence"]["n_cells"])))
            else:
                lv = fn[pair][s][level]; assert abs(lv["r_um"][BIN_10] - r_c) < 1e-6 or not np.isfinite(r_c)
                v, sd = float(lv["log_g"][BIN_10]), float(lv["sd"][BIN_10])
                rows.append(dict(section=s, group=groups[s], value=v, sd=sd, lo=float(lv["lo"][BIN_10]), hi=float(lv["hi"][BIN_10]),
                                 t_eff=float(lv["t_eff"][BIN_10]), flag=int(lv["flag"][BIN_10])))
        cs = {}
        for g1, g2 in CONTRASTS_D:
            v1 = [q["value"] for q in rows if q["group"] == g1]; s1 = [q["sd"] for q in rows if q["group"] == g1]
            v2 = [q["value"] for q in rows if q["group"] == g2]; s2 = [q["sd"] for q in rows if q["group"] == g2]
            cs[f"{g1}-{g2}"] = _contrast(v1, s1, v2, s2)
        key = None if level == "squidpy" else f"{level} log g(10 µm) {pair}"
        if key is not None:
            ref = bc["group_contrasts"][key]["contrasts"]
            for k, c in cs.items(): assert abs(c["z"] - ref[k]["z"]) < 1e-9, (key, k)       # identical to Supplementary Table S2
        vals = np.array([q["value"] for q in rows])
        return dict(pair=pair, level=level, quantity=f"log co-occurrence ratio, Squidpy co_occurrence, pairs within {sq_r:g} um" if level == "squidpy" else "log g_ab(r), bin 10-12 um",
                    sections=rows, group_means={g: float(np.mean([q["value"] for q in rows if q["group"] == g])) for g in GROUPS},
                    median_nine_sections=float(np.median(vals)), contrasts=cs, calibration=calibration(cs), table_S2_key=key)
    for level in ("squidpy", "centroids", "molecules"): out["methods"][f"{level} B:T"] = method_entry("B:T", level)
    m, c = out["methods"]["molecules B:T"]["sections"], out["methods"]["centroids B:T"]["sections"]
    out["molecules_deeper_than_centroids_every_section"] = bool(all(a["value"] < b["value"] for a, b in zip(m, c)))
    for pair in OPERATOR_PAIRS:
        for level in ("squidpy", "centroids", "molecules"): out["methods"][f"{level} {pair}"] = method_entry(pair, level)
    out["operator_pairs_pass_at_10um_molecules"] = {p: out["methods"][f"molecules {p}"]["calibration"] == "passed" for p in OPERATOR_PAIRS}
    out["operator_pairs_shown"] = bool(any(out["operator_pairs_pass_at_10um_molecules"].values()))
    if write:
        json.dump(out, open(RES / "fig3_bracket_10um.json", "w"), indent=1)
        print("wrote", RES / "fig3_bracket_10um.json")
    return out


def _zfmt(z):
    t = f"{z:+.1f}"
    return "z = " + ("0.0" if t in ("+0.0", "-0.0") else t.replace("-", "\u2212"))


def _bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y - h, y, y, y - h], color="k", lw=0.6, clip_on=False)
    ax.text((x1 + x2) / 2, y + 0.4 * h, text, ha="center", va="bottom", fontsize=LEGFS)


def panel_d_brackets(axs, bd, pair="B:T"):
    """Preprint (d): one axis per method; per-section points (with 95 % tile-bootstrap intervals where defined), group mean (short black
    line), nine-section median (gray dotted), brackets with z of WT − KO, Ctrl − KO, Ctrl − WT."""
    xg = {"WT": 0, "KO": 1, "Ctrl": 2}; jit = (-0.18, 0.0, 0.18)
    r_sq = bd["squidpy_interval"]["r_within_um"]
    titles = dict(squidpy=f"Squidpy co-occurrence, ≤ {r_sq:g} µm", centroids="cell centroids, 10 µm", molecules="molecules, 10 µm")
    for ax, level in zip(axs, ("squidpy", "centroids", "molecules")):
        e = bd["methods"][f"{level} {pair}"]
        for g in GROUPS:
            rows = [q for q in e["sections"] if q["group"] == g]
            for j, q in zip(jit, rows):
                x = xg[g] + j
                if q["lo"] is not None: ax.plot([x, x], [q["lo"], q["hi"]], color=C[g], lw=0.9, solid_capstyle="butt")
                ax.plot(x, q["value"], "o", ms=3.2, color=C[g], mec="none" if level != "squidpy" else C[g], mfc=C[g] if level != "squidpy" else "none", mew=0.8)
            ax.plot([xg[g] - 0.32, xg[g] + 0.32], [e["group_means"][g]] * 2, color="k", lw=0.9)
        ax.axhline(e["median_nine_sections"], color=C["gray"], lw=0.7, ls=":")
        ax.set_xticks(range(3)); ax.set_xticklabels(list(GROUPS)); ax.set_xlim(-0.55, 2.55)
        ax.set_title(titles[level], pad=3)
        lo = min(q["lo"] if q["lo"] is not None else q["value"] for q in e["sections"])
        hi = max(q["hi"] if q["hi"] is not None else q["value"] for q in e["sections"])
        e["_range"] = (lo, hi)
    # one y range for the three axes (all are log ratios against the label-shuffle expectation); brackets above the data
    def lims(ranges):
        lo, hi = min(r[0] for r in ranges), max(r[1] for r in ranges); span = hi - lo
        return lo - 0.08 * span, hi, span
    for axset, levels in ((tuple(axs), ("squidpy", "centroids", "molecules")),):
        lo, hi, span = lims([bd["methods"][f"{l} {pair}"]["_range"] for l in levels])
        h = 0.035 * span; step = 0.17 * span
        for ax, level in zip(axset, levels):
            cs = bd["methods"][f"{level} {pair}"]["contrasts"]
            y0 = hi + 0.10 * span
            _bracket(ax, 0, 1, y0, h, _zfmt(cs['WT-KO']['z']))
            _bracket(ax, 1, 2, y0 + step, h, _zfmt(cs['Ctrl-KO']['z']))
            _bracket(ax, 0, 2, y0 + 2 * step, h, _zfmt(cs['Ctrl-WT']['z']))
            ax.set_ylim(lo, y0 + 2 * step + 0.17 * span)
            ax.axhline(0, color="k", lw=0.4) if ax.get_ylim()[1] > 0 > ax.get_ylim()[0] else None
    for l in ("squidpy", "centroids", "molecules"): bd["methods"][f"{l} {pair}"].pop("_range", None)
    axs[0].set_ylabel("log co-occurrence ratio", labelpad=1)
    for ax in axs[1:]: ax.set_ylabel(f"log g_{pair.replace(':', ',')} (10 µm)", labelpad=1)


def panel_c(ax):
    wk = jload(RES / "three_groups.json")
    if wk is None or "B:T" not in wk["pairs"]:
        ax.text(0.5, 0.5, "three_groups.json missing", ha="center", va="center", transform=ax.transAxes); return
    rows = wk["pairs"]["B:T"]; r = np.array(next(iter(rows.values()))["r_um"]); ok = (r <= R_MAX) & (r >= 1)
    gb = group_band(rows, "log_g")
    for g, (m, err, vals) in gb.items():
        for v in vals: ax.plot(r[ok], v[ok], color=C[g], lw=0.5, alpha=0.5)
        ax.fill_between(r[ok], (m - err)[ok], (m + err)[ok], color=C[g], alpha=0.25, lw=0)
        ax.plot(r[ok], m[ok], color=C[g], lw=1.3, label=f"{g} (n = {len(vals)})")
    ax.axvline(10.0, ymax=0.62, color=C["gray"], lw=0.7, ls=":")    # read point of panel d (stops below the legend)
    ax.axhline(0, color="k", lw=0.4); ax.set_xscale("log"); ax.set_xlim(0.8, R_MAX * 1.1); ax.set_xlabel("r [µm]", labelpad=1); ax.set_ylabel("log g_B,T(r)")
    lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_, hi_ + 0.38 * (hi_ - lo_))    # room for the legend above the curves
    ax.set_title("B – T, three groups", pad=3); ax.legend(frameon=False, loc="upper left", handlelength=1.2, borderaxespad=0.1, fontsize=LEGFS, labelspacing=0.25)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preprint", action="store_true", help="accepted for compatibility; the preprint cut is the only output")
    ap.parse_args()
    bd = bracket_data()
    print("operator pairs at 10 µm (molecules):", {p: (bd["methods"][f"molecules {p}"]["calibration"],
          {k: round(c["z"], 2) for k, c in bd["methods"][f"molecules {p}"]["contrasts"].items()}) for p in OPERATOR_PAIRS})
    if bd["operator_pairs_shown"]:
        raise SystemExit("an operator pair passes the calibration at 10 µm: add its bracket row to panel d before regenerating")
    fig = plt.figure(figsize=(17.5 / 2.54, 5.15))
    top = fig.add_gridspec(1, 3, left=0.075, right=0.995, top=0.94, bottom=0.585, wspace=0.36)
    bot = fig.add_gridspec(1, 3, left=0.075, right=0.995, top=0.43, bottom=0.07, wspace=0.36)
    axs = [fig.add_subplot(top[0, i]) for i in range(3)]; daxs = [fig.add_subplot(bot[0, i]) for i in range(3)]
    panel_a_bt(axs[0]); panel_b_bands(axs[1]); panel_c(axs[2]); panel_d_brackets(daxs, bd)
    for ax, lab in zip(axs + daxs[:1], "abcd"):
        ax.text(-0.24, 1.1, lab, transform=ax.transAxes, fontweight="bold", fontsize=9, va="top")
    out = REPO / "figures/fig3_added_value"
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=600)
    print("wrote", out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
