#!/usr/bin/env python
"""Preprint Fig. 6: compute scaling on real data with molecule-level comparators.

Inputs: the benchmark rows of workflows/bench.smk in results/bench/ (one JSON line per run):
  scote_cpu1.jsonl, scote_cpu4.jsonl, scote_cpu16.jsonl, scote_gpu.jsonl   SCOTE rows (scripts/bench/bench_ours.py) on 1, 4
      and 16 pinned CPU threads and on one GPU with 16 host threads; every row run twice, --uncertainty closed --write DIR
      (per-tile path with the closed-form tile block bootstrap: log g, sd, n_eff_pairs, T_eff, flag and P of all gene pairs and
      shells written to disk) and --uncertainty none --write DIR (global counts, P written). Missing files are skipped.
  kdtree.jsonl, squidpy.jsonl, tacco.jsonl   comparators (scripts/bench/bench_{kdtree,squidpy,tacco}.py); for identical
      settings the later row wins; the full-section KD-tree row (open square, panel a) is the budget-stopped
      molecules_full_partial row with 16 workers, extrapolated linearly in the gene pairs done.
SCOTE times are the time to the result in host memory, without the write to disk (TIME_DEFINITION): closed rows compute_s,
counts-only rows compute_s + d2h_s; load_ours sets wall_s to that value and keeps the recorded wall as wall_disk_s.
The dedup key OURS_KEY includes the uncertainty and the write state.

Drawn: SCOTE rows with uncertainties (closed form) on the GPU and on 16 CPU threads only; comparators at 16 threads /
workers. The 1- and 4-thread rows, the counts-only rows, the liver rows, the KD-tree gene-panel extrapolation and the KD-tree
memory rows are not drawn; all of them are in results/bench/summary.json.
  (a) wall time vs molecules at fixed area (Xenium CRC P2, top-200 genes, thinning): SCOTE CPU 16 threads and GPU; Squidpy
      co_occurrence and KD-tree cross-K (16 workers), measured points and model extrapolation (open: extrapolated); cell level;
  (b) wall time vs gene-panel size on full sections (lymph node, CRC P2, Visium HD); star: the Atera whole-transcriptome call;
  (c) area vs density: thinning at full area (solid) vs central crops at full density (dashed), CPU 16 threads and GPU; time
      lines (1 s, 1 min) as in (a);
  (d) peak memory vs molecules; Atera whole-transcriptome device peak (star); Squidpy measured and its buffer, drawn up to
      SQ_BUF_MAX_GB; the GPU's device memory and the GPU node's RAM (recorded in the GPU rows).
Legends: color = device (top legend, all panels; comparators by color in panel a); marker / line style = data set (b), measured
vs extrapolated (a), thinning vs crop (c), device vs host memory (d).
Writes figures/fig6_compute.{pdf,png} and results/bench/summary.json.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results/bench"
OURS_FILES = ("scote_cpu1.jsonl", "scote_cpu4.jsonl", "scote_cpu16.jsonl", "scote_gpu.jsonl")
OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#000000"]
LEG = 6.0                                                   # preprint legend font size (pt)
ANN = 6.0                                                   # preprint in-plot annotations (pt)
W_PRE, H_PRE = 160, 124                                     # preprint figure size (mm), printed at \linewidth = 160 mm
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "legend.fontsize": LEG, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
C_T = {1: "#6BAED6", 4: "#2171B5", 16: "#08306B"}          # CPU threads: one hue, light -> dark
C_GPU, C_SQ, C_TAC = OKABE[4], OKABE[3], OKABE[1]
C_KD = {16: "#00704F", 1: "#66C2A4"}                        # KD-tree workers: dark / light green (lightness = threads, as for SCOTE)
C_DEV = "#8C2D04"                                          # SCOTE GPU rows
C_PRE = {"cpu16": "#4292C6", "gpu": "#EF3B2C"}             # SCOTE colors of the figure (only the 16-thread CPU rows are drawn)
GPU_THR = 16                                                # host threads of the GPU runs
MIN, HOUR, DAY = 60.0, 3600.0, 86400.0
FULL_N = 47077697


def jl(p):
    p = Path(p)
    if not p.is_absolute():
        p = RES / p
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def pick(rows, **kw):
    return [r for r in rows if all((r.get(k) == v) if not callable(v) else v(r.get(k)) for k, v in kw.items())]


def dedup(rows, key):
    """last record wins for identical settings (re-runs)"""
    d = {}
    for r in rows:
        d[key(r)] = r
    return list(d.values())


def series(rows, x="n_molecules"):
    rows = sorted(rows, key=lambda r: r[x])
    return np.array([r[x] for r in rows], float), np.array([r["wall_s"] for r in rows], float), rows


def fit_power(x, y, xmin):
    m = (x >= xmin) & np.isfinite(y) & (y > 0)
    if m.sum() < 2:
        return None
    lx, ly = np.log(x[m]), np.log(y[m])
    A = np.vstack([lx, np.ones_like(lx)]).T
    (b, c), *_ = np.linalg.lstsq(A, ly, rcond=None)
    se = None
    if m.sum() > 2:
        s2 = ((ly - A @ np.array([b, c])) ** 2).sum() / (m.sum() - 2)
        se = float(np.sqrt(s2 / ((lx - lx.mean()) ** 2).sum()))
    pred = np.exp(c) * x[m] ** b
    return dict(beta=float(b), logc=float(c), se_beta=se, n=int(m.sum()), xmin=float(x[m].min()), xmax=float(x[m].max()),
                max_rel_err=float(np.abs(pred / y[m] - 1).max()))


def fit_squidpy(rows):
    """Squidpy 1.8.3 cost model from its kernel: t = c0 + a N k^2 (zeroing + reducing the N x L x k^2 int32 buffer)
    + b N^2 (the all-pairs loop); non-negative least squares on relative residuals, all G pooled."""
    from scipy.optimize import nnls
    rows = [r for r in rows if r["n_points"] >= 3000]
    if len(rows) < 3:
        return None
    t = np.array([r["wall_s"] for r in rows]); N = np.array([r["n_points"] for r in rows], float); k = np.array([r["k_present"] for r in rows], float)
    A = np.vstack([np.ones_like(N), N * k * k, N * N]).T / t[:, None]
    p, _ = nnls(A, np.ones_like(t))
    pred = A @ p
    return dict(kind="squidpy2", c0=float(p[0]), a=float(p[1]), b=float(p[2]), n=len(rows), max_rel_err=float(np.abs(pred - 1).max()),
                xmax=float(N.max()), xmax_G200=float(max([q["n_points"] for q in rows if q["G"] == 200], default=0)),
                tmax=float(t.max()))


def fpow(f, x, k=200):
    x = np.asarray(x, float)
    if f.get("kind") == "squidpy2":
        return f["c0"] + f["a"] * x * k * k + f["b"] * x * x
    return np.exp(f["logc"]) * x ** f["beta"]


def crossover(fc, xo, yo, k=200):
    """smallest N (from max(our smallest measured size, the comparator fit's smallest N) up to 1e10) above which the comparator
    model fc stays slower than our measured curve (log-log interpolation; continued beyond the measured range by the last two
    points). Below the fit range use crossovers_measured (measured comparator points against our measured curve)."""
    grid = np.logspace(np.log10(max(xo[0], fc.get("xmin", xo[0]))), 10, 4001)
    lx, ly = np.log(xo), np.log(yo)
    s_hi = (ly[-1] - ly[-2]) / (lx[-1] - lx[-2])
    lg = np.log(grid)
    ours = np.where(lg > lx[-1], ly[-1] + s_hi * (lg - lx[-1]), np.interp(lg, lx, ly))
    above = np.log(fpow(fc, grid, k)) > ours
    if above.all():
        return float(grid[0]), f"comparator slower at every size >= {grid[0]:.0f} (smallest measured)"
    if not above.any():
        return None, "never slower up to 1e10"
    last_below = np.where(~above)[0].max()
    if last_below == len(grid) - 1:
        return None, "comparator faster again at 1e10"
    return float(grid[last_below + 1]), ""


def crossover_measured(xc, yc, xo, yo):
    """same, between two measured curves (both log-log interpolated, only on the overlap of their ranges)"""
    lo, hi = max(xc.min(), xo.min()), min(xc.max(), xo.max())
    if lo >= hi:
        return None
    grid = np.logspace(np.log10(lo), np.log10(hi), 2001)
    c = np.interp(np.log(grid), np.log(xc), np.log(yc)); o = np.interp(np.log(grid), np.log(xo), np.log(yo))
    above = c > o
    if above.all():
        return float(grid[0])
    if not above.any():
        return None
    return float(grid[np.where(~above)[0].max() + 1]) if np.where(~above)[0].max() < len(grid) - 1 else None


def time_axis(ax, ylo, yhi, fs=ANN, skip=()):
    for v, lab in ((1, "1 s"), (MIN, "1 min"), (HOUR, "1 h"), (DAY, "1 day"), (30 * DAY, "1 month")):
        if ylo < v < yhi and lab not in skip:
            ax.axhline(v, color="0.85", lw=0.5, zorder=0)
            ax.text(1.0, v, lab, transform=ax.get_yaxis_transform(), fontsize=fs, color="0.45", va="bottom", ha="right")


# ---------------------------------------------------------------------------------------------------------------------
OURS_KEY = lambda r: (r["device"], r["threads"], r["dataset"], r["G"], r["crop"], r["thin_frac"], r["thin_n"], r["stored"],
                      r.get("uncertainty", "none"), bool(r.get("write_dir")))
FULL = lambda r: r["crop"] == 1.0 and r["thin_frac"] == 1.0 and r["thin_n"] == 0


def load_ours():
    """SCOTE benchmark rows (OURS_FILES). A run whose counts differ from a reference tensor is dropped (none expected).
    Deduplicated with OURS_KEY (device, threads, input, uncertainty, write)."""
    rows = [r for f in OURS_FILES for r in jl(f)]
    bad = [r for r in rows if r.get("counts_identical_to_reference") is False]
    if bad:
        print("WARNING: dropping rows with counts different from the reference:", [(r["device"], r["threads"], r["dataset"], r["G"]) for r in bad])
    rows = [r for r in rows if r.get("counts_identical_to_reference") is not False]
    for r in rows:              # time = time to the result in host memory, without the write to disk (TIME_DEFINITION)
        r["wall_disk_s"] = r["wall_s"]
        r["wall_s"] = r["compute_s"] + ((r.get("d2h_s") or 0.0) if r.get("uncertainty") == "none" else 0.0)
    return dedup(rows, OURS_KEY)


TIME_DEFINITION = ("SCOTE times (wall_s of every row as drawn and summarized) = time to the result in host memory, excluding "
                   "the write to disk: closed rows compute_s (the device-to-host copies are inside the "
                   "accumulator finish; write_s = memory-map stores, flush, bins.csv, features.txt), counts-only rows compute_s + "
                   "d2h_s (per-shell device-to-host copies; about 0 on the CPU). The recorded wall time incl. the write is kept as "
                   "wall_disk_s. Comparator times are the recorded wall times.")
OURS = load_ours()
CLOSED = [r for r in OURS if r.get("uncertainty") == "closed" and r.get("write_dir")]
NONE = [r for r in OURS if r.get("uncertainty") == "none" and r.get("write_dir")]
GPU_SWEEPS_SEP = ("liver_slice", "liver_all", "atera_cervical")


def load_all():
    """ours = CPU closed-form rows; liver = GPU closed-form section rows; comparators"""
    ours = [r for r in CLOSED if r["device"] == "cpu"]
    liver = [r for r in CLOSED if r["device"] == "gpu" and r.get("sweep") == "liver_slice"]
    kd = dedup(jl("kdtree.jsonl"), lambda r: (r["threads"], r["G"], r["thin_n"], r["thin_frac"], r["sweep"]))
    sq = dedup(jl("squidpy.jsonl"), lambda r: (r["method"], r["threads"], r["G"], r["n_points"]))
    tac = dedup(jl("tacco.jsonl"), lambda r: (r["n_cells"],))
    return ours, liver, kd, sq, tac


GPU = [r for r in CLOSED if r["device"] == "gpu"]
GPU_OURS = [r for r in GPU if r.get("sweep") not in GPU_SWEEPS_SEP]
GPU_LIVER = pick(GPU, sweep="liver_slice")
GPU_LIVER_ALL = (pick(GPU, sweep="liver_all") or [None])[-1]
ATERA = sorted(pick(GPU, sweep="atera_cervical"), key=lambda r: r["G"])
GPU_NONE = [r for r in NONE if r["device"] == "gpu"]
GPU_NONE_OURS = [r for r in GPU_NONE if r.get("sweep") not in GPU_SWEEPS_SEP]


def gpu_info(rows=GPU + GPU_NONE):
    """Name, device memory (GB) and host RAM (GB) of the GPU rows: legend label and the memory lines of panel d."""
    r = next((q for q in rows if q.get("gpu")), None)
    if r is None:
        return None, None, None
    name = str(r["gpu"]).replace("NVIDIA ", "").split()[0]
    mem = r.get("gpu_total_MB"); ram = r.get("env", {}).get("mem_total_GB")
    return name, (mem / 1024 if mem else None), ram


GPU_NAME, GPU_MEM_GB, NODE_MEM_GB = gpu_info()
LAB_GPU = f"GPU {GPU_NAME} ({GPU_MEM_GB:.0f} GB)" if GPU_MEM_GB else f"GPU {GPU_NAME}"
CPU_NONE = [r for r in NONE if r["device"] == "cpu"]
LS_NONE = dict(ls=":", lw=0.8, marker=None)                  # counts-only rows (no intervals; not drawn in the figure)
SQ_BUF_MAX_GB = 1e4                                         # Squidpy buffer extrapolation drawn up to 10^4 GB (panel d)
PRE_THREADS = (16,)                                         # CPU thread counts drawn in the preprint figure (all rows in the summary)


def atera_billion(rows=None):
    """the whole-transcriptome single call (G >= 18,000) only; absent while the job is running"""
    rows = [r for r in (ATERA if rows is None else rows) if r["G"] >= 18000]
    return max(rows, key=lambda r: r["n_molecules"]) if rows else None


def molecule_sweep(ours, dev, thr):
    rows = [r for r in pick(ours, dataset="xenium_crc_p2", G=200, crop=1.0, device=dev, threads=thr) if r["sweep"] == "molecules"]
    return series(rows) if len(rows) >= 2 else None


def kd_rows(kd, thr, G=200):
    return [r for r in pick(kd, G=G, threads=thr, sweep="molecules")]


def kd_ok(r):
    """complete runs, and budget-stopped runs with >= 25 % of the gene pairs done (extrapolated linearly in the pairs; drawn open)"""
    return not r.get("partial") or r["n_pairs_done"] >= 0.25 * r["n_pairs"]


def kd_full(kd, G=200):
    """full-section budget-stopped KD-tree row (16 workers)"""
    r = pick(kd, G=G, threads=16, sweep="molecules_full_partial")
    return r[-1] if r else None


def sq_rows(sq, thr=16, G=200):
    return pick(sq, method="squidpy_co_occurrence", G=G, threads=thr, skipped=False)


def _legend(ax, h, l, loc, fs=None, **kw):
    from matplotlib.legend_handler import HandlerTuple
    kw = dict(dict(frameon=False, handlelength=2.0, labelspacing=0.25, borderaxespad=0.2, handletextpad=0.5), **kw)
    return ax.legend(h, l, loc=loc, fontsize=fs or LEG, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)}, **kw)


def color_handles(ms=2.4):
    """the top legend: color = device / threads (no line style), identical in all panels; the figure shows the
    16-thread CPU rows and the GPU rows with uncertainties only"""
    thr = PRE_THREADS
    h = [plt.Line2D([], [], color=C_T[t], marker="o", ls="-", ms=ms, lw=1.6) for t in thr]
    l = [f"SCOTE CPU, {t} thread{'s' if t > 1 else ''}" for t in thr]
    if GPU_OURS:
        h.append(plt.Line2D([], [], color=C_DEV, marker="o", ls="-", ms=ms, lw=1.6)); l.append(f"SCOTE {LAB_GPU}")
    return h, l


# ---------------------------------------------------------------------------------------------------------------------
def _comparators_a(ax, kd, sq, S, ms, xmax_ext, fs, full_labels=True, kd_threads=(16, 1), notes=True):
    """full_labels=False: short notes; notes=False (as drawn): no notes, no thread counts in the labels, one KD-tree
    legend entry (the open full-section square is covered by the 'extrapolated' style entry)"""
    h, l = [], []
    # Squidpy, 16 threads, G = 200: measured + model (fit over all measured N, G) to the full section
    rows = sq_rows(sq)
    if rows:
        x, y, _ = series(rows, "n_points")
        ax.plot(x, y, "^", ms=ms, color=C_SQ, mec="none")
        f = S["fits"].get("squidpy_t16")
        if f:
            xe = np.logspace(np.log10(x.max()), np.log10(xmax_ext), 40)
            ax.plot(xe, fpow(f, xe, 200), "--", lw=0.9, color=C_SQ)
            ts = float(fpow(f, FULL_N, 200))
            ax.plot(FULL_N, ts, "^", ms=ms + 0.6, mfc="white", mec=C_SQ, mew=0.8)
            if notes:
                ax.annotate(f"≈ {ts / DAY:.0f} days", (FULL_N, ts), xytext=(5, -3) if full_labels else (5, 0), textcoords="offset points", fontsize=fs,
                            ha="left", va="top" if full_labels else "center", color="0.15")
        h.append(plt.Line2D([], [], ls="none", marker="^", ms=ms, color=C_SQ, mec="none"))
        l.append("Squidpy co_occurrence, 16 threads" if notes else "Squidpy co_occurrence")
    # KD-tree, 16 and 1 workers
    for thr in kd_threads:
        rows = [r for r in kd_rows(kd, thr) if kd_ok(r)]
        if len(rows) < 2:
            continue
        x, y, rr = series(rows)
        part = np.array([bool(r.get("partial")) for r in rr])
        ax.plot(x[~part], y[~part], "s", ms=ms - 0.2, color=C_KD[thr], mec="none")
        ax.plot(x[part], y[part], "s", ms=ms - 0.2, mfc="white", mec=C_KD[thr], mew=0.7)
        f = S["fits"].get(f"kdtree_G200_t{thr}")
        if f:
            xe = np.logspace(np.log10(x.max()), np.log10(FULL_N), 30)
            ax.plot(xe, fpow(f, xe), "--", lw=0.9, color=C_KD[thr])
        h.append(plt.Line2D([], [], ls="none", marker="s", ms=ms - 0.2, color=C_KD[thr], mec="none"))
        l.append(f"KD-tree cross-K, {thr} worker{'s' if thr > 1 else ''}" if notes else "KD-tree cross-K")
    r = kd_full(kd)
    if r:
        ax.plot(r["n_molecules"], r["wall_s"], "s", ms=ms + 0.4, mfc="white", mec=C_KD[16], mew=0.8)
        if not notes:
            pass
        elif full_labels:
            h.append(plt.Line2D([], [], ls="none", marker="s", ms=ms + 0.4, mfc="white", mec=C_KD[16], mew=0.8))
            l.append("KD-tree, full section, 16 workers")
            frac = r["n_pairs_done"] / r["n_pairs"]
            ax.annotate(f"≈ {r['wall_s'] / HOUR:.0f} h\n({frac:.0%} of pairs\nin {r['wall_measured_s'] / HOUR:.1f} h)", (r["n_molecules"], r["wall_s"]),
                        xytext=(5, -1), textcoords="offset points", fontsize=fs, color="0.15", va="top")
        else:
            ax.annotate(f"≈ {r['wall_s'] / HOUR:.0f} h", (r["n_molecules"], r["wall_s"]), xytext=(5, 0),
                        textcoords="offset points", fontsize=fs, va="center", color="0.15")
    return h, l


def panel_a(ax, D, S):
    """SCOTE with uncertainties on 16 CPU threads and the GPU; comparators at 16 threads / workers; cell-level references"""
    ours, liver, kd, sq, tac = D
    ylo, yhi = 0.1, 1e7
    for thr in PRE_THREADS:
        s = molecule_sweep(ours, "cpu", thr)
        if s:
            ax.plot(s[0], s[1], "o-", ms=2.0, lw=1.1, color=C_T[thr])
    s = molecule_sweep(GPU_OURS, "gpu", GPU_THR)
    if s:
        ax.plot(s[0], s[1], "o-", ms=2.0, lw=1.3, color=C_DEV)
    h, l = _comparators_a(ax, kd, sq, S, 3, FULL_N, ANN, kd_threads=(16,), notes=False)
    for rows, col, lab in ((pick(sq, method="squidpy_co_occurrence_cells", skipped=False), C_SQ, "Squidpy, cell level"),
                           (pick(tac, method="tacco_co_occurrence_cells"), C_TAC, "TACCO, cell level")):
        if rows:
            r = max(rows, key=lambda q: q["n_cells"])
            ax.plot(r["n_cells"], r["wall_s"], "D", ms=3.2, mfc="white", mec=col, mew=0.8)
            h.append(plt.Line2D([], [], ls="none", marker="D", ms=3.2, mfc="white", mec=col, mew=0.8)); l.append(lab)
    h += [plt.Line2D([], [], color="0.3", ls="none", marker="s", ms=2.8, mec="none"),
          (plt.Line2D([], [], color="0.3", ls="none", marker="s", ms=2.8, mfc="white", mew=0.7), plt.Line2D([], [], color="0.3", ls="--", lw=0.9))]
    l += ["measured", "extrapolated"]
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(ylo, yhi); ax.set_xlim(7e2, 5e8)     # right margin: time-line labels clear of the data
    time_axis(ax, ylo, yhi)
    ax.set_xlabel("molecules (Xenium CRC section, top-200 genes, thinned)"); ax.set_ylabel("wall time [s]")
    _legend(ax, h, l, "upper left")


DS_B = (("x5k_lymph_node", "s"), ("xenium_crc_p2", "o"), ("hd_crc_p2", "D"))


def _panel_b_series(ax, ours, ms, none=True, threads=None):
    for thr in (threads or (1, 4, 16)):
        for ds, mk in DS_B:
            rows = [r for r in pick(ours, dataset=ds, device="cpu", threads=thr) if FULL(r)]
            if rows:
                x, y, _ = series(rows, "G")
                ax.plot(x, y, mk + "-", ms=ms, lw=0.9, color=C_T[thr])
            rows = [r for r in pick(CPU_NONE, dataset=ds, device="cpu", threads=thr) if FULL(r)] if none else []
            if len(rows) >= 2:
                x, y, _ = series(rows, "G")
                ax.plot(x, y, color=C_T[thr], **LS_NONE)
    for ds, mk in DS_B:
        rows = [r for r in pick(GPU_NONE_OURS, dataset=ds, device="gpu", threads=GPU_THR) if FULL(r)] if none else []
        if len(rows) >= 2:
            x, y, _ = series(rows, "G")
            ax.plot(x, y, color=C_DEV, **LS_NONE)
    for ds, mk in DS_B:
        rows = [r for r in pick(GPU_OURS, dataset=ds, device="gpu", threads=GPU_THR) if FULL(r)]
        if rows:
            x, y, _ = series(rows, "G")
            ax.plot(x, y, mk + "-", ms=ms, lw=1.1, color=C_DEV, zorder=4)
    b = atera_billion()
    if b:
        ax.plot(b["G"], b["wall_s"], "*", ms=12, color=C_DEV, mec="k", mew=0.4, zorder=1.5)
    return b


def panel_b(ax, D, S):
    """full sections, SCOTE with uncertainties on 16 CPU threads and the GPU; star: the Atera whole-transcriptome call"""
    ours = D[0]
    ylo, yhi = 3, 3e5
    ms = 2.6
    _panel_b_series(ax, ours, ms, none=False, threads=PRE_THREADS)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(ylo, yhi); ax.set_xlim(35, 4e4)
    time_axis(ax, ylo, yhi, skip=("1 day",))
    ax.set_xlabel("genes (top-G, full section)"); ax.set_ylabel("wall time [s]")
    h = [plt.Line2D([], [], color="0.3", marker=m, ls="-", ms=ms, lw=0.8) for _, m in DS_B]
    h += [plt.Line2D([], [], ls="none", color=C_DEV, marker="*", ms=7, mec="k", mew=0.4)]
    _legend(ax, h, ["Xenium 5K lymph node (202 M molecules)", "Xenium CRC P2 (51 M molecules)", "Visium HD CRC P2 (267 M UMI)",
                    "Atera WTA cervical, all 18,028 genes"], "upper left")


def panel_c(ax, D, S):
    ours = D[0]
    for src, dev, thr, col in [(ours, "cpu", t, C_T[t]) for t in PRE_THREADS] + [(GPU_OURS, "gpu", GPU_THR, C_DEV)]:
        th = [r for r in pick(src, dataset="xenium_crc_p2", G=200, device=dev, threads=thr, crop=1.0) if r["sweep"] == "molecules" and r["n_molecules"] >= 4e5]
        cr = pick(src, dataset="xenium_crc_p2", G=200, device=dev, threads=thr, sweep="area") + [r for r in th if FULL(r)]
        if len(th) >= 2:
            x, y, _ = series(th); ax.plot(x, y, "o-", ms=2.2, lw=1.0, color=col)
        if len(cr) >= 2:
            x, y, _ = series(cr); ax.plot(x, y, "s--", ms=2.4, lw=1.0, color=col, mfc="white", mew=0.7)
    ylo, yhi = 0.5, 5e2
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(ylo, yhi); ax.set_xlim(3e5, 2.5e8)     # right margin: time-line labels clear of the data
    time_axis(ax, ylo, yhi)
    ax.set_xlabel("molecules (Xenium CRC P2, top-200 genes)"); ax.set_ylabel("wall time [s]")
    h = [plt.Line2D([], [], color="0.3", marker="o", ls="-", ms=2.2, lw=1.0), plt.Line2D([], [], color="0.3", marker="s", ls="--", ms=2.4, lw=1.0, mfc="white")]
    _legend(ax, h, ["random thinning at full area", "central crops at full density (1/16, 1/4, 1/2, full)"], "upper left")


def panel_d(ax, D, S):
    """peak memory: SCOTE (16 CPU threads: host; GPU: device solid, host dotted), the Atera whole-transcriptome device peak (star),
    Squidpy measured and its buffer 4 N L G^2 bytes extrapolated up to 1e4 GB; GPU device memory and the GPU node's RAM"""
    ours, liver, kd, sq, tac = D
    for src, dev, thr, key, col, ls in ([(ours, "cpu", t, "peak_rss_MB", C_T[t], "-") for t in PRE_THREADS] +
                                        [(GPU_OURS, "gpu", GPU_THR, "gpu_peak_alloc_MB", C_DEV, "-"), (GPU_OURS, "gpu", GPU_THR, "peak_rss_MB", C_DEV, ":")]):
        s_ = molecule_sweep(src, dev, thr)
        if s_:
            ax.plot([r["n_molecules"] for r in s_[2]], [r[key] / 1024 for r in s_[2]], "o" + ls, ms=1.8, lw=1.0, color=col)
    h = [plt.Line2D([], [], color="0.3", ls="-", marker="o", ms=2.0, lw=1.0), plt.Line2D([], [], color="0.3", ls=":", marker="o", ms=2.0, lw=1.0)]
    l = ["CPU host / GPU device", "GPU run, host"]
    b = atera_billion()
    if b:
        ax.plot(b["n_molecules"], b["gpu_peak_alloc_MB"] / 1024, "*", ms=7, color=C_DEV, mec="k", mew=0.4, zorder=5)
        h.append(plt.Line2D([], [], ls="none", marker="*", ms=7, color=C_DEV, mec="k", mew=0.4)); l.append("Atera, device")
    hs = []
    for G, mk, al in ((200, "^", 1.0), (50, "v", 0.5)):
        rows = sorted(sq_rows(sq, 16, G), key=lambda r: r["n_points"])
        if rows:
            ax.plot([r["n_points"] for r in rows], [r["peak_rss_MB"] / 1024 for r in rows], mk, ms=3, color=C_SQ, mec="none", alpha=al)
            n_max = SQ_BUF_MAX_GB * 2**30 / (4.0 * 48 * G * G)          # buffer drawn up to SQ_BUF_MAX_GB
            xe = np.logspace(np.log10(rows[-1]["n_points"]), np.log10(n_max), 30)
            ax.plot(xe, 4.0 * xe * 48 * G * G / 2**30, "--", lw=0.8, color=C_SQ, alpha=al)
            hs.append(plt.Line2D([], [], ls="none", marker=mk, ms=3, color=C_SQ, mec="none", alpha=al))
    if hs:
        h.append(tuple(hs) + (plt.Line2D([], [], ls="--", lw=0.8, color=C_SQ),)); l.append("Squidpy, G = 200 / 50")
    if GPU_MEM_GB:
        ax.axhline(GPU_MEM_GB, color=C_DEV, lw=0.6, ls="--")
        h.append(plt.Line2D([], [], color=C_DEV, lw=0.6, ls="--")); l.append(f"{GPU_NAME} memory ({GPU_MEM_GB:.0f} GB)")
    if NODE_MEM_GB:
        ax.axhline(NODE_MEM_GB, color="0.5", lw=0.6, ls="-.")
        h.append(plt.Line2D([], [], color="0.5", lw=0.6, ls="-."))
        l.append(f"node RAM ({NODE_MEM_GB / 1024:.0f} TB)" if NODE_MEM_GB >= 1024 else f"node RAM ({NODE_MEM_GB:.0f} GB)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.01, 1e6); ax.set_xlim(7e2, 2.5e9)
    ax.set_xlabel("molecules (Xenium CRC, top-200; star: whole section)"); ax.set_ylabel("peak memory [GB]")
    _legend(ax, h, l, "upper left", ncol=2, columnspacing=1.0, handlelength=1.8)


# ---------------------------------------------------------------------------------------------------------------------
def summarize(D):
    ours, liver, kd, sq, tac = D
    S = dict(time_definition=TIME_DEFINITION, code="scripts/bench/bench_ours.py rows in results/bench/scote_*.jsonl; SCOTE rows = closed-form "
                  "tile bootstrap with the output written (wall_disk_s includes the write); counts-only rows under *_none keys",
             fits={}, crossovers={}, crossovers_measured={})
    # comparator fits over the extended measured range
    for thr in (1, 16):
        f = fit_squidpy(pick(sq, method="squidpy_co_occurrence", threads=thr, skipped=False))
        if f:
            S["fits"][f"squidpy_t{thr}"] = f
        for G in (50, 200):
            rows = [r for r in kd_rows(kd, thr, G) if kd_ok(r)]
            if len(rows) >= 2:
                x, y, _ = series(rows); f = fit_power(x, y, 3e4)
                if f:
                    S["fits"][f"kdtree_G{G}_t{thr}"] = f
    # KD-tree G exponent at matched N (16 workers)
    k50 = {r["thin_n"]: r["wall_s"] for r in kd_rows(kd, 16, 50)}
    k200 = {r["thin_n"]: r["wall_s"] for r in kd_rows(kd, 16, 200)}
    common = [n for n in k50 if n in k200 and n >= 1e5]
    f = S["fits"].get("kdtree_G200_t16")
    if common and f:
        gam = float(np.mean([np.log(k200[n] / k50[n]) / np.log(4) for n in common])); S["kdtree_gamma_G"] = gam
        out = {}
        for ds in ("x5k_lymph_node", "xenium_crc_p2"):
            rows = sorted([r for r in pick(GPU_OURS, dataset=ds, device="gpu") if FULL(r)], key=lambda r: r["G"])
            out[ds] = [dict(G=r["G"], N=r["n_molecules"], t_est_s=float(fpow(f, r["n_molecules"]) * (r["G"] / 200) ** gam)) for r in rows]
        S["kdtree_fullsection"] = out
    # comparator measurement basis
    mb = {}
    for thr in (16, 1):
        rows = [r for r in kd_rows(kd, thr) if kd_ok(r)]
        if rows:
            r = max(rows, key=lambda q: q["n_molecules"])
            mb[f"kdtree_G200_w{thr}"] = dict(N=r["n_molecules"], wall_s=round(r["wall_s"], 1), partial=bool(r.get("partial")),
                                              measured_s=round(r.get("wall_measured_s", r["wall_s"]), 1), pairs_done=r["n_pairs_done"])
    r = kd_full(kd)
    if r:
        mb["kdtree_G200_w16_fullsection_partial"] = dict(N=r["n_molecules"], pairs_done=r["n_pairs_done"], pairs=r["n_pairs"],
                                                        measured_s=round(r.get("wall_measured_s", 0), 1), extrapolated_s=round(r["wall_s"], 1),
                                                        pairs_s=round(r["pairs_s"], 1), s_per_pair=round(r["pairs_s"] / r["n_pairs_done"], 3),
                                                        cpu=r.get("env", {}).get("cpu"), workers=r["threads"])
    for G in (200, 50):
        rows = sq_rows(sq, 16, G)
        if rows:
            r = max(rows, key=lambda q: q["n_points"]); mb[f"squidpy_G{G}_t16"] = dict(N=r["n_points"], wall_s=round(r["wall_s"], 1), peak_rss_GB=round(r["peak_rss_MB"] / 1024, 1))
        sk = [q for q in pick(sq, method="squidpy_co_occurrence", G=G, threads=16, skipped=True)]
        if sk:
            mb[f"squidpy_G{G}_t16_skipped"] = [dict(N=q["n_points"], predicted_buffer_GB=round(q["predicted_buffer_GB"], 0), reason=q.get("reason")) for q in sk]
    S["comparator_measured_max"] = mb
    # SCOTE curves and crossovers on the CRC molecule sweep
    curves = {f"cpu{t}": molecule_sweep(ours, "cpu", t) for t in (1, 4, 16)}
    curves["gpu"] = molecule_sweep(GPU_OURS, "gpu", GPU_THR)
    curves_none = {f"cpu{t}": molecule_sweep(CPU_NONE, "cpu", t) for t in (1, 4, 16)}
    curves_none["gpu"] = molecule_sweep(GPU_NONE_OURS, "gpu", GPU_THR)
    for ref, s in curves_none.items():
        if s:
            S[f"{ref}_crc_sweep_none"] = [dict(N=int(r["n_molecules"]), wall_s=round(r["wall_s"], 3)) for r in s[2]]
    S["gpu_crossovers"], S["gpu_crossovers_measured"] = {}, {}
    for ref, s in curves.items():
        if not s:
            continue
        # GPU (16 host threads): sweep and crossovers under gpu_* keys, CPU rows under ours_cpu<threads>_* keys
        cx, cxm = (S["gpu_crossovers"], S["gpu_crossovers_measured"]) if ref == "gpu" else (S["crossovers"], S["crossovers_measured"])
        S["gpu_crc_sweep" if ref == "gpu" else f"ours_{ref}_crc_sweep"] = [dict(N=int(r["n_molecules"]), wall_s=round(r["wall_s"], 3), peak_rss_GB=round(r["peak_rss_MB"] / 1024, 2),
                                          **({"device_GB": round(r["gpu_peak_alloc_MB"] / 1024, 2)} if r.get("gpu_peak_alloc_MB") else {}),
                                          counts_identical=r.get("counts_identical_to_reference")) for r in s[2]]
        for key, k in (("squidpy_t16", 200), ("squidpy_t1", 200), ("kdtree_G200_t16", 200), ("kdtree_G200_t1", 200), ("kdtree_G50_t16", 200)):
            f = S["fits"].get(key)
            if f:
                n, note = crossover(f, s[0], s[1], k); cx[f"{key}_vs_{ref}"] = dict(N=n, note=note)
        for thr in (16, 1):
            rows = [r for r in kd_rows(kd, thr) if kd_ok(r)]
            if len(rows) >= 2:
                x, y, _ = series(rows)
                cxm[f"kdtree_G200_t{thr}_vs_{ref}"] = crossover_measured(x, y, s[0], s[1])
    # full sections, every device
    S["full_sections"] = [dict(dataset=r["dataset"], device=r["device"], threads=r["threads"], G=r["G"], N=int(r["n_molecules"]),
                               uncertainty=r["uncertainty"], wall_s=round(r["wall_s"], 2), compute_s=round(r["compute_s"], 2),
                               write_s=round(r["write_s"], 2), host_GB=round(r["peak_rss_MB"] / 1024, 1),
                               **({"device_GB": round(r["gpu_peak_alloc_MB"] / 1024, 1)} if r.get("gpu_peak_alloc_MB") else {}),
                               n_tiles=r.get("n_tiles"), output_GB=round(r.get("output_bytes", 0) / 1e9, 3), write_dir=r.get("write_dir"),
                               counts_identical=r.get("counts_identical_to_reference"))
                          for r in sorted([r for r in OURS if FULL(r) and r.get("sweep") not in GPU_SWEEPS_SEP],
                                          key=lambda r: (r["dataset"], r["G"], r["device"], r["threads"], r["uncertainty"]))]
    S["crops"] = [dict(device=r["device"], threads=r["threads"], crop=r["crop"], N=int(r["n_molecules"]), wall_s=round(r["wall_s"], 2))
                  for r in sorted(pick(ours + GPU_OURS, sweep="area"), key=lambda r: (r["device"], r["threads"], r["crop"]))]
    # crop vs thinning at equal N (thinning curve interpolated log-log at the crop's N)
    ratios = []
    for dev, thr in (("cpu", 1), ("cpu", 4), ("cpu", 16), ("gpu", GPU_THR)):
        s = curves["gpu" if dev == "gpu" else f"cpu{thr}"]
        for r in pick(ours + GPU_OURS, sweep="area", device=dev, threads=thr):
            if s:
                tthin = float(np.exp(np.interp(np.log(r["n_molecules"]), np.log(s[0]), np.log(s[1]))))
                ratios.append(dict(device=dev, threads=thr, crop=r["crop"], N=int(r["n_molecules"]), crop_s=round(r["wall_s"], 2),
                                   thin_s_at_N=round(tthin, 2), ratio=round(tthin / r["wall_s"], 2)))
    S["crop_vs_thinning"] = ratios
    # full-section comparator estimates
    for key in ("squidpy_t16", "kdtree_G200_t16"):
        f = S["fits"].get(key)
        if f:
            S[f"{key}_fullsection_est_s"] = float(fpow(f, FULL_N, 200))
    S["squidpy_fullsection_buffer_TB"] = 4.0 * FULL_N * 48 * 200 ** 2 / 1e12
    summarize_devices(S, ours)
    return S


def _row(r):
    return dict(dataset=r["dataset"], device=r["device"], threads=r["threads"], G=r["G"], N=int(r["n_molecules"]), uncertainty=r["uncertainty"],
                host_s=round(r["wall_s"], 2), wall_disk_s=round(r.get("wall_disk_s", r["wall_s"]), 2), compute_s=round(r["compute_s"], 2),
                write_s=round(r["write_s"], 2), d2h_s=r.get("d2h_s"),
                load_s=round(r.get("load_s", 0), 1), host_GB=round(r["peak_rss_MB"] / 1024, 1),
                **({"device_GB": round(r["gpu_peak_alloc_MB"] / 1024, 1), "device_reserved_GB": round(r.get("gpu_peak_reserved_MB", 0) / 1024, 1)}
                   if r.get("gpu_peak_alloc_MB") else {}),
                n_tiles=r.get("n_tiles"), acc_dtype=r.get("acc_dtype"), output_GB=round(r.get("output_bytes", 0) / 1e9, 3),
                counts_identical=r.get("counts_identical_to_reference"),
                **({"total_pairs": r["total_pairs"]} if r.get("total_pairs") is not None else {}))


def summarize_devices(S, ours):
    """summary keys: environment, GPU full sections (closed and counts only), closed / counts-only ratios on every device,
    GPU vs 16 CPU threads, liver, Atera"""
    if not OURS:
        return
    g = (GPU or [None])[0]; c = ([r for r in OURS if r["device"] == "cpu"] or [None])[0]
    S["env"] = dict(gpu=g.get("gpu") if g else None, gpu_host_cpu=g["env"].get("cpu") if g else None,
                    cpu=c["env"].get("cpu") if c else None, gpu_host_threads=GPU_THR, numba=(g or c)["env"].get("numba"),
                    scote=(g or c)["env"].get("scote"), torch=g.get("torch") if g else None, cuda=g.get("cuda") if g else None,
                    files=[f for f in OURS_FILES if (RES / f).exists()])
    S["gpu_full_sections"] = [_row(r) for r in sorted([r for r in GPU_OURS + GPU_NONE_OURS if FULL(r)],
                                                       key=lambda r: (r["dataset"], r["G"], r["uncertainty"]))]
    S["gpu_crops"] = [dict(crop=r["crop"], N=int(r["n_molecules"]), wall_s=round(r["wall_s"], 2)) for r in sorted(pick(GPU_OURS, sweep="area"), key=lambda r: r["crop"])]
    key = lambda r: (r["device"], r["threads"], r["dataset"], r["G"], r["crop"], r["thin_frac"], r["thin_n"], r.get("sweep") == "liver_all")
    none = {key(r): r for r in NONE}
    S["closed_over_none"] = [dict(device=r["device"], threads=r["threads"], dataset=r["dataset"], G=r["G"], N=int(r["n_molecules"]),
                                  sweep=r["sweep"], closed_s=round(r["wall_s"], 2), none_s=round(none[key(r)]["wall_s"], 2),
                                  ratio=round(r["wall_s"] / none[key(r)]["wall_s"], 2))
                             for r in sorted(CLOSED, key=lambda r: (r["device"], r["threads"], r["sweep"], r["dataset"], r["G"], r["n_molecules"]))
                             if key(r) in none]
    cpu16 = {key(r)[2:]: r for r in ours if r["threads"] == 16}
    S["cpu16_over_gpu"] = [dict(dataset=r["dataset"], G=r["G"], N=int(r["n_molecules"]), gpu_s=round(r["wall_s"], 2),
                                 cpu16_s=round(cpu16[key(r)[2:]]["wall_s"], 2), ratio=round(cpu16[key(r)[2:]]["wall_s"] / r["wall_s"], 2))
                            for r in sorted(GPU_OURS, key=lambda r: (r["dataset"], r["G"], r["n_molecules"])) if FULL(r) and key(r)[2:] in cpu16]
    d = dict(sections=[_row(r) for r in GPU_LIVER], sections_wall_total_s=round(sum(r["wall_s"] for r in GPU_LIVER), 1),
             sections_N_total=int(sum(r["n_molecules"] for r in GPU_LIVER)))
    if GPU_LIVER_ALL:
        d["single_call"] = dict(_row(GPU_LIVER_ALL), sections_in_call=GPU_LIVER_ALL.get("sections"),
                                note="all four sections side by side in one call (no cross-section pairs); top-200 genes of the union")
    nl = [r for r in GPU_NONE if r.get("sweep") == "liver_all"]
    if nl:
        d["single_call_none"] = _row(nl[-1])
    S["gpu_liver"] = d
    S["gpu_atera_cervical"] = [_row(r) for r in ATERA] + [_row(r) for r in sorted(pick(GPU_NONE, sweep="atera_cervical"), key=lambda r: r["G"])]


def main():
    global C_DEV
    D = load_all()
    S = summarize(D)
    json.dump(S, open(RES / "summary.json", "w"), indent=1)
    C_T[16], C_DEV = C_PRE["cpu16"], C_PRE["gpu"]
    fig, axs = plt.subplots(2, 2, figsize=(W_PRE / 25.4, H_PRE / 25.4))
    panel_a(axs[0, 0], D, S); panel_b(axs[0, 1], D, S); panel_c(axs[1, 0], D, S); panel_d(axs[1, 1], D, S)
    for ax, lab in zip(axs.ravel(), "abcd"):
        ax.text(-0.17, 1.03, lab, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    h, l = color_handles()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=LEG, frameon=False, handlelength=1.6, columnspacing=1.5, labelspacing=0.25,
               bbox_to_anchor=(0.53, 1.0), borderaxespad=0.2)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.08, top=0.91, wspace=0.27, hspace=0.32)
    small = [t for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip() and t.get_fontsize() < min(LEG, ANN) - 1e-6]
    if small:
        raise SystemExit(f"text below {min(LEG, ANN)} pt: {[(t.get_text(), t.get_fontsize()) for t in small]}")
    fig.savefig(REPO / "figures/fig6_compute.pdf"); fig.savefig(REPO / "figures/fig6_compute.png", dpi=600)
    print("wrote figures/fig6_compute.{pdf,png}")
    print(json.dumps({k: v for k, v in S.items() if k not in ("kdtree_fullsection", "full_sections", "crop_vs_thinning", "gpu_full_sections",
                                                               "gpu_liver", "env", "closed_over_none", "gpu_atera_cervical")
                      and not k.endswith("_crc_sweep") and not k.endswith("_crc_sweep_none")}, indent=1))


if __name__ == "__main__":
    main()
