"""Weight matrices for fits across shells with the closed-form tile-bootstrap covariance (generalized least squares).

The closed-form covariance of log g across shells is a sum over tiles of outer products of the per-tile influence vectors,
Cov(k, k') = sum_t l_t(k) l_t(k') (scote.uncertainty.cross_shell_cov), i.e. T times the sample covariance of T vectors with
mean zero (sum_t l_t = 0). Neighboring shells are correlated close to 1, so the correlation matrix has eigenvalues down to
1e-5 to 1e-6 of the largest; those are estimated from only T tiles (T = 50-250 against K = 10-25 fitted shells) and are
biased low, and the plain inverse over-weights the shell-to-shell differences of the curve. The weight matrix is therefore
the inverse of a Ledoit-Wolf shrinkage estimate: the correlation matrix R is shrunk toward the identity,
R_s = (1 - s) R + s I, with the intensity s of Ledoit and Wolf (2004, J. Multivar. Anal. 88, 365) computed from the same
T influence vectors; the SDs are unchanged. `shrunk_cov(L, s=0.0)` gives the unregularized covariance.

Several independent sections (e.g. a difference Xenium - Visium HD) contribute their tiles as further rows of L.
"""
from __future__ import annotations

import numpy as np


def lw_intensity(L: np.ndarray) -> float:
    """Ledoit-Wolf shrinkage intensity of the correlation matrix of the rows of L (n_tiles, K), target the identity.
    The rows are the per-tile influence vectors (mean zero by construction); columns are standardized first."""
    L = np.asarray(L, dtype=np.float64)
    n, K = L.shape
    sd = np.sqrt((L ** 2).sum(0))
    X = L / sd * np.sqrt(n)                     # X^T X / n = R
    S = X.T @ X / n
    F = np.trace(S) / K * np.eye(K)
    d2 = float(((S - F) ** 2).sum())
    if d2 <= 0:
        return 0.0
    b2 = float(sum(((np.outer(x, x) - S) ** 2).sum() for x in X)) / n ** 2
    return min(b2, d2) / d2


def shrunk_cov(L: np.ndarray, s: float | None = None):
    """Covariance (K, K) = sum_t l_t l_t^T of the rows of L with the correlation matrix shrunk toward the identity by the
    intensity s (None: Ledoit-Wolf, `lw_intensity`; 0: none). Returns (cov, s, condition number of the shrunk correlation)."""
    L = np.asarray(L, dtype=np.float64)
    C = L.T @ L
    sd = np.sqrt(np.diag(C))
    R = C / np.outer(sd, sd)
    if s is None:
        s = lw_intensity(L)
    Rs = (1.0 - s) * R + s * np.eye(len(R))
    ev = np.linalg.eigvalsh(Rs)
    return Rs * np.outer(sd, sd), float(s), float(ev[-1] / ev[0]) if ev[0] > 0 else float("inf")
