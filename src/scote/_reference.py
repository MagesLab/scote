"""Random-label reference of the pair counts."""
from __future__ import annotations

import numpy as np

__all__ = ["random_label_reference"]


def random_label_reference(P: np.ndarray, totals: np.ndarray) -> np.ndarray:
    r"""Expected ordered pair counts for independent labels, :math:`P_\mathrm{tot}(k)\, n_a n_b / n_\mathrm{tot}^2`.

    :math:`g_{ab}(k) = P_{ab}(k) / \mathrm{reference}_{ab}(k)`: the labels of the molecules (or counts) are
    exchangeable under the reference, so the spatial pattern of all molecules together cancels and only the
    association of the labels remains.

    Parameters
    ----------
    P : numpy.ndarray
        (..., K, G, G) ordered pair counts; leading dimensions (e.g. bootstrap replicates) are allowed.
    totals : numpy.ndarray
        (..., G) feature totals :math:`n_a`, with the same leading dimensions.

    Returns
    -------
    numpy.ndarray
        (..., K, G, G) float64 reference counts. Both inputs are cast to float64 first: with integer inputs the
        product :math:`P_\mathrm{tot}\, n_a n_b`, formed before the division by :math:`n_\mathrm{tot}^2`, would
        overflow int64 at the scale of a whole section.
    """
    P = np.asarray(P, dtype=np.float64)
    totals = np.asarray(totals, dtype=np.float64)
    n_tot = totals.sum(-1)
    Ptot = P.sum(axis=(-1, -2))
    outer = totals[..., :, None] * totals[..., None, :]
    return Ptot[..., None, None] * outer[..., None, :, :] / np.maximum(n_tot, 1)[..., None, None, None] ** 2
