"""Shared input selection for the compute-scaling benchmarks of preprint Fig. 6 (scripts/bench/*).

Every method (SCOTE CPU/GPU, Squidpy co_occurrence, KD-tree cross-K) gets *identical* inputs from `select()`:
  cache  data/bench_scaling/<dataset>.npz  (x, y float32 um; code int16 = abundance rank of the gene in the full
         section, so 'top-G genes' = code < G; features = gene names in rank order)  -- written by prep.py
  genes  top-G by abundance in the full section
  crop   area fraction f: axis-aligned rectangle with the bbox aspect ratio, side lengths sqrt(f) x bbox, centered on
         the median molecule position (density stays ~ that of the tissue there)
  thin   random subsample without replacement (numpy default_rng(seed), seed 0), either a fraction or an absolute N
Radii: the shell edges of the SCOTE lattice hierarchy (pitch p, shells_per_level 8, n_levels L): level 0 edges 0, p, ..., 8p; level
l >= 1 adds 4 shells of width 2^l p: [8*2^(l-1)p .. 16*2^(l-1)p]. With p = 2 um, L = 10: 49 edges, 0 .. 16,384 um.
"""
import json, os, platform, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "data/bench_scaling"
OUTDIR = REPO / "results/bench"


def shell_edges(pitch=2.0, shells_per_level=8, n_levels=10):
    e = [k * pitch for k in range(shells_per_level + 1)]
    p = pitch
    for _ in range(n_levels):
        p *= 2
        e += [k * p for k in range(shells_per_level // 2 + 1, shells_per_level + 1)]
    return np.array(e, dtype=np.float64)


def load_cache(dataset):
    z = np.load(CACHE / f"{dataset}.npz", allow_pickle=True)
    return dict(x=z["x"], y=z["y"], code=z["code"], features=list(z["features"]), counts=z["counts"])


def select(c, G=200, crop=1.0, thin_frac=1.0, thin_n=0, seed=0):
    """Returns xy (N,2) float64, codes (N,) int64, info dict. Order of operations: genes -> crop -> thin."""
    code = c["code"]
    G = len(c["features"]) if (G is None or G <= 0) else min(G, len(c["features"]))
    m = code < G
    x = c["x"][m]; y = c["y"][m]; k = code[m]
    xmin, xmax, ymin, ymax = float(x.min()), float(x.max()), float(y.min()), float(y.max())
    bbox_area = (xmax - xmin) * (ymax - ymin)
    if crop < 1.0:
        s = np.sqrt(crop)
        w, h = (xmax - xmin) * s, (ymax - ymin) * s
        cx, cy = float(np.median(x)), float(np.median(y))
        x0 = min(max(cx - w / 2, xmin), xmax - w); y0 = min(max(cy - h / 2, ymin), ymax - h)
        mm = (x >= x0) & (x < x0 + w) & (y >= y0) & (y < y0 + h)
        x, y, k = x[mm], y[mm], k[mm]
        region_area = w * h
    else:
        region_area = bbox_area
    n_before = len(k)
    n_target = thin_n if thin_n > 0 else int(round(thin_frac * n_before))
    if n_target < n_before:
        idx = np.random.default_rng(seed).choice(n_before, n_target, replace=False)
        idx.sort()
        x, y, k = x[idx], y[idx], k[idx]
    xy = np.stack([x, y], 1).astype(np.float64)
    # occupied area: 50 um tiles holding >= 1 molecule of the *unthinned* crop would be better; use the thinned set's
    # 50 um occupancy, which is stable down to ~1e4 molecules per mm^2
    t = np.unique(np.floor(xy / 50.0).astype(np.int64) @ np.array([1, 1 << 20], dtype=np.int64)) if len(k) else []
    occ_area = len(t) * 2500.0
    info = dict(G=int(G), crop=float(crop), thin_frac=float(thin_frac), thin_n=int(thin_n), seed=seed,
                n_molecules=int(len(k)), n_before_thin=int(n_before), region_area_mm2=region_area / 1e6,
                occupied_area_mm2_50um=occ_area / 1e6,
                density_per_um2=(len(k) / occ_area if occ_area else float("nan")))
    return xy, k.astype(np.int64), info


def env_info():
    """Software and machine record of a benchmark row: versions (the installed `scote` included), CPU model, CPUs available
    to the process, SLURM job, load average at the start, and the thread environment variables."""
    import numpy, scipy
    d = dict(python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__,
             cpu=_cpu_model(), affinity=len(os.sched_getaffinity(0)),
             mem_total_GB=_mem_total_gb(), slurm_job=os.environ.get("SLURM_JOB_ID"), loadavg=_loadavg(),
             threads_env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                         "OMP_WAIT_POLICY")})
    try:
        import numba; d["numba"] = numba.__version__
    except ImportError:
        pass
    d["scote"] = _version("scote")
    return d


def _version(dist):
    """Installed version of a distribution (None if it is not installed)."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def _mem_total_gb():
    """RAM of the node in GB (MemTotal of /proc/meminfo; None if unavailable)."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 2**20
    except (OSError, ValueError):
        pass
    return None


def _loadavg():
    """1, 5, 15 min load averages of the node at the start of the measurement (None if /proc is unavailable)."""
    try:
        return [float(v) for v in open("/proc/loadavg").read().split()[:3]]
    except (OSError, ValueError):
        return None


def _cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def peak_rss_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def append_jsonl(path, rec):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("RESULT", json.dumps(rec), flush=True)
