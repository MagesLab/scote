"""Reference implementations for the tests: the Monte Carlo tile block bootstrap (a loop over resamples) and the
two-pass moments of the closed form, written from the definitions."""
import numpy as np

from scote import random_label_reference


def bootstrap_weights(T: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Tile multiplicities (n_boot, T) of the block bootstrap; row b is bincount(rng.integers(0, T, T))."""
    W = np.empty((n_boot, T), dtype=np.float64)
    for b in range(n_boot):
        W[b] = np.bincount(rng.integers(0, T, T), minlength=T)
    return W


def loop_bootstrap_log_g(counts_tiles, totals_tiles, n_boot=200, seed=0, ci=(2.5, 97.5)):
    """Monte Carlo block bootstrap over tiles, one tensordot per resample: log_g, lo, hi (percentiles), sd,
    n_eff_pairs = P^2 / Var(P), P, totals."""
    rng = np.random.default_rng(seed)
    T, K, G, _ = counts_tiles.shape
    P = counts_tiles.sum(0)
    n = totals_tiles.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_g = np.log(P / random_label_reference(P, n))
    logs = np.empty((n_boot,) + P.shape, dtype=np.float32)
    Ps = np.empty((n_boot,) + P.shape, dtype=np.float64)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, T, T), minlength=T).astype(np.float64)
        Pb = np.tensordot(w, counts_tiles, axes=(0, 0))
        nb = w @ totals_tiles
        Ps[b] = Pb
        with np.errstate(divide="ignore", invalid="ignore"):
            logs[b] = np.log(Pb / random_label_reference(Pb, nb))
    with np.errstate(invalid="ignore"):                    # -inf - (-inf) in the bins of zero counts
        lo, hi = np.nanpercentile(logs, ci, axis=0)
        sd = np.nanstd(logs, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_eff = P ** 2 / np.maximum(Ps.var(0), 1e-300)
    return dict(log_g=log_g, lo=lo, hi=hi, sd=sd, n_eff_pairs=n_eff, P=P, totals=n, n_tiles=T)


def direct_moments(counts_tiles, totals_tiles, symmetrize=True):
    """Two-pass reference from a stored tile tensor: Var(P) = sum_t (P_t - P/T)^2, sd(log g) = sqrt(sum_t l_t^2) with
    l_t formed explicitly, t_eff = P^2 / sum_t P_t^2; on the slabs (P_t + P_t^T) / 2 when symmetrize."""
    C = np.asarray(counts_tiles, dtype=np.float64)
    N = np.asarray(totals_tiles, dtype=np.float64)
    if symmetrize:
        C = 0.5 * (C + np.swapaxes(C, 2, 3))
    T = C.shape[0]
    P = C.sum(0)
    n_a = N.sum(0)
    n = n_a.sum()
    m = N.sum(1)
    Qt = C.sum(axis=(2, 3))
    Q = P.sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(n_a > 0, N / n_a, 0.0)
        var_P = ((C - P[None] / T) ** 2).sum(0)
        infl = (C / P[None] - (Qt / Q)[:, :, None, None] - R[:, None, :, None] - R[:, None, None, :]
             + 2.0 * (m / n)[:, None, None, None])
        sd = np.sqrt((infl ** 2).sum(0))
        t_eff = P ** 2 / (C ** 2).sum(0)
    return dict(P=P, var_P=var_P, sd=sd, t_eff=t_eff)


def max_threads(n: int) -> int:
    """n, clamped to numba's maximum thread count of this process."""
    import numba
    return max(1, min(n, numba.config.NUMBA_NUM_THREADS))
