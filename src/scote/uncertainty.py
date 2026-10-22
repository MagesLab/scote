r"""Closed-form tile block bootstrap of :math:`\log g` and of statistics derived from it.

A block bootstrap over the :math:`T` spatial tiles of a section draws :math:`T` tiles with replacement, i.e.
multinomial weights :math:`W \sim \mathrm{Multinomial}(T; 1/T)` with :math:`E\,W_t = 1`,
:math:`\mathrm{Var}\,W_t = 1 - 1/T`, :math:`\mathrm{Cov}(W_t, W_{t'}) = -1/T`. For a tile-additive statistic
:math:`x = \sum_t x_t` the resampling variance for infinitely many resamples is

.. math::

    \mathrm{Var}_\mathrm{boot}(x) = \sum_t x_t^2 - x^2 / T = \sum_t (x_t - x / T)^2 .

For :math:`\log g_{ab} = \log P_{ab} - \log P_\mathrm{tot} - \log n_a - \log n_b + 2 \log n_\mathrm{tot}`
(random-label reference) the delta method gives :math:`\mathrm{Var}_\mathrm{boot}(\log g_{ab}) = \sum_t l_t(ab)^2`
with the per-tile contribution (influence)

.. math::

    l_t(ab) = \frac{P_t}{P} - \frac{Q_t}{Q} - r_t(a) - r_t(b) + \frac{2 m_t}{n_\mathrm{tot}} ,
    \qquad \sum_t l_t(ab) = 0,

where :math:`P_t` is the per-tile slab of the shell, :math:`Q_t = \sum_{ab} P_t`, :math:`Q = \sum_t Q_t`,
:math:`r_t(a) = n_t(a) / n(a)` is the tile share of feature :math:`a`, :math:`m_t = \sum_a n_t(a)`, and
:math:`n_\mathrm{tot} = \sum_t m_t`. The weight :math:`w_t(ab) = 2 m_t / n_\mathrm{tot} - r_t(a) - r_t(b)` is known
before the pass over the slabs, so expanding the square needs four :math:`G \times G` accumulators per shell,

.. math::

    A_0 = \sum_t P_t \; (= P), \quad A_1 = \sum_t P_t \odot P_t, \quad A_2 = \sum_t Q_t P_t,
    \quad A_v = \sum_t P_t \odot w_t ,

.. math::

    \sum_t l_t^2 = \frac{A_1}{P^2} - \frac{2 A_2}{P Q} + \frac{2 A_v}{P} + \sum_t v_t^2 ,

with the tile-only part :math:`\sum_t v_t(ab)^2 = w(a) + w(b) + 2 (R^\top R)(ab)`,
:math:`w = C_2 / 2 + \sum_t r_t^2 - 2 c^\top R`, :math:`c_t = -Q_t / Q + 2 m_t / n_\mathrm{tot}`,
:math:`C_2 = \sum_t c_t^2`, and :math:`R = (r_t(a))` of shape (T, G).

:math:`\mathrm{Var}_\mathrm{boot}(P_{ab}) = A_1 - A_0^2 / T`, and ``n_eff_pairs``
:math:`= P^2 / \mathrm{Var}_\mathrm{boot}(P)` describes the tile scatter of the count.
:math:`T_\mathrm{eff} = P^2 / \sum_t P_t^2` is the effective number of tiles that carry the pair; it decides whether the
normal interval :math:`\log g \mp z\,\mathrm{sd}` is used (:func:`scote.uncertainty.reliability_flag`). By default the
slabs are symmetrized, :math:`P_t \to (P_t + P_t^\top) / 2` in :math:`A_1`, :math:`A_2`, and :math:`A_v`: a pair is
attributed to the tile of its first member, so the slab of one tile is not symmetric, and symmetrizing gives half of
each pair to the tile of each member and :math:`\mathrm{sd}_{ab} = \mathrm{sd}_{ba}`. Tiles without a slab in a shell
contribute :math:`P_t = 0` and :math:`Q_t = 0` and enter only through their feature totals.

:math:`\sqrt{\sum_t l_t^2}` is the infinitesimal jackknife over tiles, the :math:`B \to \infty` limit of the
bootstrap SD of the linearized :math:`\log g`. It has no resampling noise; the variance estimate itself has a
relative error of order :math:`\sqrt{2 / T_\mathrm{eff}}`.

Statistics of many :math:`\log g` values (Pearson :math:`r`, regression slopes, RMS and mean differences, pooled pair
sets) and smooth functions of tile-additive sums get their SD from the same per-tile contributions by the delta method
(:func:`scote.uncertainty.influence_log_g`, :func:`scote.uncertainty.delta_var`,
:func:`scote.uncertainty.tile_sum_stat`).
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from numba import njit, prange

from ._reference import random_label_reference
from ._twopoint._common import resolve_backend

__all__ = [
    "MIN_TILES_WARN", "T_EFF_MIN", "T_EFF_OK", "TileMomentAccumulator", "closed_form_from_influence",
    "closed_form_log_g", "collapse_tensor", "collapsed_log_g", "cross_shell_corr", "cross_shell_cov", "delta_var",
    "grad_deming", "grad_mean_diff", "grad_numeric", "grad_ols_intercept", "grad_ols_slope", "grad_pearson",
    "grad_rms", "influence_log_g", "pooled_log_g", "reliability_flag", "stat_ci", "tile_sum_stat", "tile_sum_var",
]

MIN_TILES_WARN = 30
_FEW_TILES_MSG = ("closed-form tile bootstrap with {T} tiles: a variance estimate from fewer than {m} tiles is itself "
                  "uncertain (relative SD of the SD about 1/sqrt(2(T-1)) = {rel:.0%}). With tiles smaller than the "
                  "correlation length (the distance beyond which log g has decayed to 0 within its interval), the "
                  "error is not calibrated, and at distances approaching the tile side the interval is approximate. "
                  "See the documentation on uncertainties.")

T_EFF_OK = 20.0     # T_eff >= 20: normal interval (flag 0)
T_EFF_MIN = 5.0     # 5 <= T_eff < 20: interval kept but flagged (flag 1); T_eff < 5: no interval (flag 2)

# slabs with fewer elements than this are updated by the serial kernel (a parallel region per slab costs more than
# the update itself for small panels)
NUMBA_PARALLEL_MIN_ELEMENTS = 1 << 16


@njit(cache=True, nogil=True)
def _update_row(a, Pg, A0, A1, A2, Av, q, c, r, sym):
    """Update row a of the four accumulators with the slab Pg.

    Parameters
    ----------
    a : int
        Row index.
    Pg : numpy.ndarray
        (G, G) float64 slab P_t.
    A0, A1, A2, Av : numpy.ndarray
        (G, G) accumulators, updated in place.
    q : float
        Slab total Q_t.
    c : float
        Tile weight 2 m_t / n.
    r : numpy.ndarray
        (G,) tile shares r_t(a).
    sym : bool
        Use the symmetrized slab (P_t + P_t^T) / 2 in A1, A2, and Av; A0 always gets the raw slab.
    """
    ra = r[a]
    for b in range(Pg.shape[1]):
        p = Pg[a, b]
        A0[a, b] += p
        ps = 0.5 * (p + Pg[b, a]) if sym else p
        A1[a, b] += ps * ps
        A2[a, b] += ps * q
        Av[a, b] += ps * (c - ra - r[b])


@njit(parallel=True, cache=True)
def _update_numba_par(Pg, A0, A1, A2, Av, q, c, r, sym):
    """One pass over the slab updating the four accumulators, rows in parallel.

    Parameters
    ----------
    Pg, A0, A1, A2, Av, q, c, r, sym
        As in ``_update_row``.
    """
    for a in prange(Pg.shape[0]):
        _update_row(a, Pg, A0, A1, A2, Av, q, c, r, sym)


@njit(cache=True)
def _update_numba_ser(Pg, A0, A1, A2, Av, q, c, r, sym):
    """Serial variant of ``_update_numba_par`` (same arguments)."""
    for a in range(Pg.shape[0]):
        _update_row(a, Pg, A0, A1, A2, Av, q, c, r, sym)


def _update_numba(Pg, *args):
    """Update the accumulators with the parallel kernel for large slabs and the serial one for small slabs.

    Parameters
    ----------
    Pg : numpy.ndarray
        (G, G) float64 slab.
    *args
        The other arguments of ``_update_row`` after the row index and the slab.
    """
    (_update_numba_par if Pg.size >= NUMBA_PARALLEL_MIN_ELEMENTS else _update_numba_ser)(Pg, *args)


def _update_numpy(Pg, A0, A1, A2, Av, q, c, r, sym, buf, buf2):
    """Pure-NumPy update of the four accumulators (no G x G temporaries beyond the two scratch buffers).

    Parameters
    ----------
    Pg, A0, A1, A2, Av, q, c, r, sym
        As in ``_update_row``.
    buf : numpy.ndarray
        (G, G) float64 scratch.
    buf2 : numpy.ndarray or None
        (G, G) float64 scratch for the symmetrized slab.
    """
    A0 += Pg
    if sym:
        np.add(Pg, Pg.T, out=buf2)
        buf2 *= 0.5
        Ps = buf2
    else:
        Ps = Pg
    np.multiply(Ps, Ps, out=buf)
    A1 += buf
    np.multiply(Ps, q, out=buf)
    A2 += buf
    np.subtract(c - r[:, None], r[None, :], out=buf)                 # combined weight (c - r_a) - r_b, as the kernel
    buf *= Ps
    Av += buf


def reliability_flag(t_eff: np.ndarray) -> np.ndarray:
    r"""Reliability of the normal interval from the effective number of tiles carrying a pair.

    Parameters
    ----------
    t_eff : numpy.ndarray
        :math:`T_\mathrm{eff} = P^2 / \sum_t P_t^2` (NaN where :math:`P = 0`).

    Returns
    -------
    numpy.ndarray
        int8 array of the shape of ``t_eff``: 0 = :math:`T_\mathrm{eff} \ge 20` (normal interval
        :math:`\log g \mp z\,\mathrm{sd}`; SD within about 1.5 % of a large-:math:`B` bootstrap);
        1 = :math:`5 \le T_\mathrm{eff} < 20` (interval reported but flagged: SD 2-8 % low, lower end too high by
        0.1-0.3 half-widths); 2 = :math:`T_\mathrm{eff} < 5` or undefined (no interval).
    """
    t = np.asarray(t_eff)
    flag = np.full(t.shape, 2, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        flag[t >= T_EFF_MIN] = 1
        flag[t >= T_EFF_OK] = 0
    return flag


class TileMomentAccumulator:
    r"""Accumulators of the closed-form tile bootstrap, filled one per-tile slab at a time.

    Four :math:`G \times G` accumulators per live shell slot (module docstring);
    :meth:`scote.uncertainty.TileMomentAccumulator.finish` turns them into :math:`P`, :math:`\log g`, its SD,
    :math:`\mathrm{Var}(P)`, ``n_eff_pairs``, :math:`T_\mathrm{eff}`, and the reliability flag of the shell and frees
    them. Only shells that received a slab and are not finished hold memory.

    Parameters
    ----------
    N : numpy.ndarray
        (T, G) per-tile feature totals :math:`n_t(a)`, tiles in the order of the tile index passed to
        :meth:`scote.uncertainty.TileMomentAccumulator.update`.
    n_slots : int
        Number of shell slots :math:`K` (the bin index over all levels).
    backend : {"cpu", "gpu"}, scote.CPUOptions, or scote.GPUOptions
        ``"cpu"`` (or a :class:`scote.CPUOptions`, whose fields concern only the pair counts): host arrays, updated by
        a parallel numba kernel (or NumPy, see ``kernel``). ``"gpu"``: PyTorch tensors on the default CUDA device,
        updated by fused in-place operations; a :class:`scote.GPUOptions` selects the device of its ``device`` field,
        e.g. ``scote.GPUOptions(device="cpu")`` for the PyTorch CPU device.
    dtype : numpy.dtype
        Dtype of :math:`A_1`, :math:`A_2`, and :math:`A_v`: float64 (default) or float32 (half the memory;
        accumulation rounding of order :math:`T \cdot 6 \times 10^{-8}` relative, amplified by the cancellation in
        the expansion). :math:`A_0`, the point estimate, is always float64 and exact for integer counts below
        :math:`2^{53}`.
    kernel : {"auto", "numba", "numpy"}
        Update on the CPU backend; ``"auto"`` is ``"numba"``.
    keep_P : bool
        Keep every finished shell's :math:`P` (float64, (G, G)) in ``self.P[k]``.
    min_tiles_warn : int
        Warn (``UserWarning``) when :math:`T` is smaller.
    symmetrize : bool
        Use the symmetrized slabs :math:`(P_t + P_t^\top) / 2` in :math:`A_1`, :math:`A_2`, and :math:`A_v`, so that
        ``sd``, ``n_eff_pairs``, and ``t_eff`` are symmetric. The point estimate is unchanged. ``False`` reproduces the
        convention of a Monte Carlo bootstrap that attributes each pair to the tile of its first member.
    return_var_P : bool
        Whether :meth:`scote.uncertainty.TileMomentAccumulator.finish` returns :math:`\mathrm{Var}(P)`; with ``False``
        the key holds ``None`` (the GPU backend then neither computes nor copies it).

    Raises
    ------
    ValueError
        If ``N`` is not two-dimensional, or for an unknown backend name, kernel, or dtype.
    TypeError
        For a ``backend`` that is neither a name nor an options object.
    ImportError
        For the GPU backend without PyTorch.
    """

    def __init__(self, N, n_slots: int, backend="cpu", dtype=np.float64, kernel: str = "auto",
                 keep_P: bool = True, min_tiles_warn: int = MIN_TILES_WARN, symmetrize: bool = True,
                 return_var_P: bool = True):
        """See the class docstring for the parameters."""
        N = np.asarray(N, dtype=np.float64)
        if N.ndim != 2:
            raise ValueError(f"N must be (T, G), got shape {N.shape}")
        self.N = N
        self.T, self.G = N.shape
        self.n_slots = int(n_slots)
        self.symmetrize = bool(symmetrize)
        self.n_a = N.sum(0)                                   # feature totals n_a
        self.m = N.sum(1)                                     # tile totals m_t
        self.n = float(self.n_a.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.where(self.n_a > 0, N / self.n_a, 0.0)     # absent features: r = 0 (their P is 0 -> NaN anyway)
        self.R = R
        self.c = 2.0 * self.m / self.n if self.n > 0 else np.zeros(self.T)   # 2 m_t / n
        self.s2 = (R * R).sum(0)                              # sum_t r_t(a)^2
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.float64, np.float32):
            raise ValueError(f"accumulator dtype must be float64 or float32, got {self.dtype}")
        self.backend, options = resolve_backend(backend)
        self.keep_P = keep_P
        self.return_var_P = bool(return_var_P)
        self.P: dict[int, np.ndarray] = {}
        self._live: dict[int, dict] = {}
        self._finished: set[int] = set()
        if self.backend == "gpu":
            from ._twopoint.gpu import _torch
            torch = self._torch = _torch()
            self.device = torch.device(options.device)
            self._tdt = torch.float64 if self.dtype == np.float64 else torch.float32
            self._R_dev = torch.from_numpy(R).to(self.device)                  # float64 (finish)
            self._R_acc = self._R_dev.to(self._tdt)                            # accumulator dtype (update)
        else:
            self.device = None
            if kernel == "auto":
                kernel = "numba"
            if kernel not in ("numba", "numpy"):
                raise ValueError(f"kernel must be 'auto', 'numba', or 'numpy', got {kernel!r}")
            self.kernel = kernel
        if self.T < min_tiles_warn:
            warnings.warn(_FEW_TILES_MSG.format(T=self.T, m=min_tiles_warn, rel=1 / np.sqrt(2 * max(1, self.T - 1))),
                          UserWarning, stacklevel=2)

    def _alloc(self, k: int) -> dict:
        """Zeroed accumulators of a new live slot.

        Parameters
        ----------
        k : int
            Slot index.

        Returns
        -------
        dict
            A0, A1, A2, Av (G, G), the per-tile totals Q (T,), the seen flags, and NumPy scratch buffers.
        """
        G = self.G
        if self.backend == "gpu":
            torch = self._torch
            st = dict(A0=torch.zeros((G, G), dtype=torch.float64, device=self.device),
                      Q=torch.zeros(self.T, dtype=torch.float64, device=self.device))
            for name in ("A1", "A2", "Av"):
                st[name] = torch.zeros((G, G), dtype=self._tdt, device=self.device)
        else:
            st = dict(A0=np.zeros((G, G)), Q=np.zeros(self.T))
            for name in ("A1", "A2", "Av"):
                st[name] = np.zeros((G, G), dtype=self.dtype)
            if self.kernel == "numpy":
                st["buf"] = np.empty((G, G))
                st["buf2"] = np.empty((G, G)) if self.symmetrize else None
        st["seen"] = np.zeros(self.T, dtype=bool)
        return st

    def live_slots(self) -> list[int]:
        """Slots that hold accumulators (received a slab, not finished).

        Returns
        -------
        list of int
            The slot indices in increasing order.
        """
        return sorted(self._live)

    def discard(self, k: int) -> None:
        """Drop the partial accumulators of slot ``k`` (a producer restarts the shell, e.g. after running out of
        memory).

        Parameters
        ----------
        k : int
            Slot index.
        """
        self._live.pop(k, None)

    def update(self, k: int, t: int, Pg) -> None:
        """Add the slab of one tile to slot ``k``.

        Parameters
        ----------
        k : int
            Slot index.
        t : int
            Tile index in 0 .. T - 1.
        Pg : numpy.ndarray or torch.Tensor
            (G, G) slab :math:`P_t` (float64 for exact point estimates).

        Raises
        ------
        ValueError
            If the slot is finished or the tile was already added to it.
        IndexError
            If ``t`` is out of range.
        """
        if k in self._finished:
            raise ValueError(f"slot {k} is already finished")
        if not 0 <= t < self.T:
            raise IndexError(f"tile {t} out of range [0, {self.T})")
        st = self._live.get(k)
        if st is None:
            st = self._live[k] = self._alloc(k)
        if st["seen"][t]:
            raise ValueError(f"tile {t} of slot {k} was added twice")
        st["seen"][t] = True
        c_t = float(self.c[t])
        if self.backend == "gpu":
            torch = self._torch
            P = Pg if isinstance(Pg, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(Pg))
            P = P.to(device=self.device, dtype=torch.float64)
            q = P.sum()                                           # 0-d tensor, stays on the device (no sync)
            st["Q"][t] = q
            st["A0"].add_(P)
            Pa = P if self._tdt == torch.float64 else P.to(self._tdt)
            if self.symmetrize:
                Pa = Pa.add(Pa.T).mul_(0.5)                       # one G^2 scratch tensor
            qa = q if self._tdt == torch.float64 else q.to(self._tdt)
            r = self._R_acc[t]
            st["A1"].addcmul_(Pa, Pa)
            st["A2"].addcmul_(Pa, qa)
            Av = st["Av"]
            Av.add_(Pa, alpha=c_t)
            Av.addcmul_(Pa, r[:, None], value=-1.0)
            Av.addcmul_(Pa, r[None, :], value=-1.0)
            return
        if not isinstance(Pg, np.ndarray):                        # torch tensor into the CPU backend
            Pg = Pg.detach().cpu().numpy()
        P = np.ascontiguousarray(Pg, dtype=np.float64)
        q = float(P.sum())
        st["Q"][t] = q
        r = self.R[t]
        if self.kernel == "numba":
            _update_numba(P, st["A0"], st["A1"], st["A2"], st["Av"], q, c_t, r, self.symmetrize)
        else:
            _update_numpy(P, st["A0"], st["A1"], st["A2"], st["Av"], q, c_t, r, self.symmetrize, st["buf"],
                          st["buf2"])

    __call__ = update

    def finish(self, k: int, out_dtype=np.float64) -> dict:
        r"""Close slot ``k``, free its accumulators, and return its estimates.

        Parameters
        ----------
        k : int
            Slot index. A slot that never received a slab gives the estimates of all-zero moments.
        out_dtype : numpy.dtype
            Dtype of ``log_g``, ``sd``, and ``n_eff_pairs``.

        Returns
        -------
        dict
            Host arrays (G, G) under the names of :class:`scote.TwoPointResult`: ``P`` (float64), ``log_g`` (log of
            :math:`P` over the random-label reference), ``sd`` (square root of the delta-method variance of
            :math:`\log g`, negative rounding residues clipped to 0, NaN where :math:`P = 0`), ``n_eff_pairs``
            :math:`= P^2 / \max(\mathrm{Var}(P), 10^{-300})`, ``t_eff`` (float32, :math:`P^2 / \sum_t P_t^2`), ``flag``
            (int8, :func:`scote.uncertainty.reliability_flag`), and ``var_P`` (float64 or ``None``); scalars ``P_tot``
            (sum of :math:`P`) and ``n_tiles``. With ``symmetrize``, :math:`\mathrm{Var}(P)`, ``n_eff_pairs``, and
            ``t_eff`` refer to the symmetrized slabs.

        Raises
        ------
        ValueError
            If the slot is already finished.
        """
        if k in self._finished:
            raise ValueError(f"slot {k} is already finished")
        st = self._live.pop(k, None)
        if st is None:
            st = self._alloc(k)                                    # shell without any slab: all-zero moments
        self._finished.add(k)
        if self.backend == "gpu":
            res = self._finish_torch(st, out_dtype)          # sd, n_eff_pairs, t_eff, flag final (device-side)
        else:
            res = self._finish_numpy(st)
            if not self.return_var_P:
                res["var_P"] = None
        del st
        P = res["P"]
        with np.errstate(divide="ignore", invalid="ignore"):
            log_g = np.log(P / random_label_reference(P[None], self.n_a)[0])
        res["log_g"] = log_g.astype(out_dtype, copy=False)
        res["sd"] = res["sd"].astype(out_dtype, copy=False)
        res["n_eff_pairs"] = res["n_eff_pairs"].astype(out_dtype, copy=False)
        if "flag" not in res:
            res["flag"] = reliability_flag(res["t_eff"])
        res["n_tiles"] = self.T
        if self.keep_P:
            self.P[k] = P
        return res

    def _tile_terms(self, Qt: np.ndarray, Q: float):
        """w(a) of the tile-only part sum_t v_t^2 = w(a) + w(b) + 2 (R^T R)(ab), with
        c_t = -Q_t / Q + 2 m_t / n.

        Parameters
        ----------
        Qt : numpy.ndarray
            (T,) per-tile slab totals.
        Q : float
            Their sum.

        Returns
        -------
        numpy.ndarray
            (G,) w.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            c = (-Qt / Q if Q != 0 else np.zeros_like(Qt)) + self.c
        w = 0.5 * float(c @ c) + self.s2 - 2.0 * (c @ self.R)
        return w

    def _finish_numpy(self, st: dict) -> dict:
        """Host-side finish of one slot (the expansion of the module docstring).

        Parameters
        ----------
        st : dict
            The slot's accumulators (consumed).

        Returns
        -------
        dict
            P, var_P, n_eff_pairs, sd, t_eff, P_tot.
        """
        P = st["A0"]
        Q = float(P.sum())
        T = self.T
        Pv = 0.5 * (P + P.T) if self.symmetrize else P             # P of the (symmetrized) slabs
        A1 = _float64(st.pop("A1"))
        var_P = A1 - Pv * Pv / T
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            n_eff = Pv ** 2 / np.maximum(var_P, 1e-300)
            t_eff = (Pv * Pv / A1).astype(np.float32)
            V = A1                                                 # private buffer (popped): reused in place
            V /= Pv
            if Q != 0:
                V -= (2.0 / Q) * _float64(st.pop("A2"))
            V += 2.0 * _float64(st.pop("Av"))
            V /= Pv
        w = self._tile_terms(st["Q"], Q)
        V += w[:, None]
        V += w[None, :]
        V += 2.0 * (self.R.T @ self.R)
        np.maximum(V, 0.0, out=V, where=~np.isnan(V))
        np.sqrt(V, out=V)
        return dict(P=P, var_P=var_P, n_eff_pairs=n_eff, sd=V, t_eff=t_eff, P_tot=Q)

    def _finish_torch(self, st: dict, out_dtype=np.float64) -> dict:
        """Device-side finish of one slot.

        The elementwise operations of ``_finish_numpy`` in the same order (IEEE fp64, so n_eff_pairs, t_eff, and
        the flag equal the CPU backend's for equal accumulators); only P, sd, n_eff_pairs, t_eff, the flag, and
        Var(P) if requested are copied to the host.

        Parameters
        ----------
        st : dict
            The slot's accumulators (consumed).
        out_dtype : numpy.dtype
            Dtype of sd and n_eff_pairs.

        Returns
        -------
        dict
            P, var_P, n_eff_pairs, sd, t_eff, flag, P_tot.
        """
        torch = self._torch
        P = st["A0"]
        Q = float(P.sum().item())
        T = self.T
        tdt_out = torch.float32 if np.dtype(out_dtype) == np.float32 else torch.float64
        Pv = P.add(P.T).mul_(0.5) if self.symmetrize else P
        A1 = _tensor_float64(st.pop("A1"))
        tmp = Pv * Pv                                               # Pv^2 (one G^2 scratch tensor, reused below)
        var_P = A1 - tmp / T                                        # numpy: A1 - Pv * Pv / T
        var_P_h = var_P.cpu().numpy() if self.return_var_P else None
        torch.div(tmp, A1, out=tmp)                                 # numpy: Pv * Pv / A1
        t_eff = tmp.to(torch.float32)
        torch.mul(Pv, Pv, out=tmp)
        tmp.div_(torch.maximum(var_P, torch.tensor(1e-300, dtype=torch.float64, device=self.device)))
        del var_P
        n_eff = tmp.to(tdt_out)
        del tmp
        flag = torch.full(t_eff.shape, 2, dtype=torch.int8, device=self.device)   # reliability_flag on the device
        flag[t_eff >= T_EFF_MIN] = 1
        flag[t_eff >= T_EFF_OK] = 0
        V = A1.div_(Pv)                                           # private buffer (popped): reused in place
        if Q != 0:
            V.add_(_tensor_float64(st.pop("A2")), alpha=-2.0 / Q)
        V.add_(_tensor_float64(st.pop("Av")), alpha=2.0)
        V.div_(Pv)
        Qt = st["Q"].cpu().numpy()
        w = torch.from_numpy(self._tile_terms(Qt, Q)).to(self.device)
        V.add_(w[:, None]).add_(w[None, :])
        V.addmm_(self._R_dev.T, self._R_dev, alpha=2.0)
        V.clamp_(min=0.0).sqrt_()                                   # NaN (P = 0) stays NaN
        sd = V.to(tdt_out)
        del V, Pv
        P_h = P.cpu().numpy()
        return dict(P=P_h, var_P=var_P_h, n_eff_pairs=n_eff.cpu().numpy(), sd=sd.cpu().numpy(),
                    t_eff=t_eff.cpu().numpy(), flag=flag.cpu().numpy(), P_tot=Q)


def _float64(A: np.ndarray) -> np.ndarray:
    """An array as float64 (the array itself if it already is).

    Parameters
    ----------
    A : numpy.ndarray
        The array.

    Returns
    -------
    numpy.ndarray
        float64 array.
    """
    return A if A.dtype == np.float64 else A.astype(np.float64)


def _tensor_float64(A):
    """A torch tensor as float64 (the tensor itself if it already is).

    Parameters
    ----------
    A : torch.Tensor
        The tensor.

    Returns
    -------
    torch.Tensor
        float64 tensor.
    """
    return A if str(A.dtype) == "torch.float64" else A.double()


def closed_form_log_g(counts_tiles: np.ndarray, totals_tiles: np.ndarray, bins=None, features=None, *,
                      backend="cpu", dtype=np.float64, min_tiles_warn: int = MIN_TILES_WARN,
                      symmetrize: bool = True):
    r""":math:`\log g` with the closed-form tile bootstrap from a stored tile tensor.

    The slabs go through :class:`scote.uncertainty.TileMomentAccumulator` one shell at a time, so this function and the
    streamed accumulation of :func:`scote.twopoint` give identical values. Use it on subsets of tiles, on collapsed
    feature groups (:func:`scote.uncertainty.collapse_tensor`), or on the tile tensor of a result.

    Parameters
    ----------
    counts_tiles : numpy.ndarray
        (T, K, G, G) per-tile pair counts.
    totals_tiles : numpy.ndarray
        (T, G) per-tile feature totals.
    bins : pandas.DataFrame or None
        The bins table of the K bins (e.g. ``res.bins`` of the run the tensor comes from); ``None`` gives a table
        with ``level`` 0, ``shell`` 0 .. K - 1, and NaN distances.
    features : sequence of str or None
        Names of the G features; ``None`` gives ``"0"``, ``"1"``, ...
    backend : {"cpu", "gpu"}, scote.CPUOptions, or scote.GPUOptions
        Where the accumulators run, as in :class:`scote.uncertainty.TileMomentAccumulator`.
    dtype : numpy.dtype
        Accumulator dtype of :math:`A_1`, :math:`A_2`, :math:`A_v`.
    min_tiles_warn : int
        Warn when :math:`T` is smaller.
    symmetrize : bool
        Symmetrize the per-tile slabs (:class:`scote.uncertainty.TileMomentAccumulator`).

    Returns
    -------
    scote.TwoPointResult
        ``log_g``, ``sd`` (NaN where :math:`P = 0`), ``n_eff_pairs``
        (:math:`P^2 / \max(\mathrm{Var}(P), 10^{-300})`), ``P`` (float64, (K, G, G)), ``t_eff`` (float32), ``flag``
        (int8), ``P_tot``, ``totals``, ``totals_tiles``, ``bins``, ``features``, and ``counts_tiles`` (the input);
        intervals from :meth:`scote.TwoPointResult.interval` (``lo`` = ``hi`` = NaN where the flag is 2, i.e.
        :math:`T_\mathrm{eff} < 5` or :math:`P = 0`).

    Raises
    ------
    ValueError
        If the shapes do not match, or for an unknown backend name or dtype.
    TypeError
        For a ``backend`` that is neither a name nor an options object.
    ImportError
        For the GPU backend without PyTorch.
    """
    import dataclasses

    import pandas as pd

    from ._result import TwoPointResult
    t0 = time.perf_counter()
    C = np.asarray(counts_tiles)
    if C.ndim != 4 or C.shape[2] != C.shape[3]:
        raise ValueError(f"counts_tiles must have shape (T, K, G, G), got {C.shape}")
    T, K, G, _ = C.shape
    N = np.asarray(totals_tiles, dtype=np.float64)
    if N.shape != (T, G):
        raise ValueError(f"totals_tiles must have shape ({T}, {G}), got {N.shape}")
    name, options = resolve_backend(backend)
    acc = TileMomentAccumulator(N, K, backend=options, dtype=dtype, keep_P=False,
                                min_tiles_warn=min_tiles_warn, symmetrize=symmetrize)
    log_g = np.empty((K, G, G))
    sd = np.empty((K, G, G))
    n_eff = np.empty((K, G, G))
    P = np.empty((K, G, G))
    P_tot = np.empty(K)
    t_eff = np.empty((K, G, G), dtype=np.float32)
    flag = np.empty((K, G, G), dtype=np.int8)
    for k in range(K):
        for t in range(T):
            acc.update(k, t, np.asarray(C[t, k], dtype=np.float64))
        r = acc.finish(k)
        log_g[k], sd[k], n_eff[k], P[k], t_eff[k], flag[k] = (r["log_g"], r["sd"], r["n_eff_pairs"], r["P"],
                                                              r["t_eff"], r["flag"])
        P_tot[k] = r["P_tot"]
    sd[P == 0] = np.nan
    if bins is None:
        bins = pd.DataFrame(dict(level=np.zeros(K, dtype=np.int64), shell=np.arange(K), r_from=np.full(K, np.nan),
                                 r_to=np.full(K, np.nan), r_central=np.full(K, np.nan)))
    elif len(bins) != K:
        raise ValueError(f"bins has {len(bins)} rows for {K} bins")
    names = [str(f) for f in features] if features is not None else [str(i) for i in range(G)]
    if len(names) != G:
        raise ValueError(f"{len(names)} feature names for {G} features")
    params = dict(input="tile tensor", symmetrize=bool(symmetrize), backend=name,
                  options=dataclasses.asdict(options), acc_dtype=str(np.dtype(dtype)))
    return TwoPointResult(log_g=log_g, sd=sd, n_eff_pairs=n_eff, P=P, t_eff=t_eff, flag=flag, P_tot=P_tot,
                          totals=acc.n_a, totals_tiles=N, bins=bins.reset_index(drop=True), features=names,
                          counts_tiles=counts_tiles, params=params,
                          timings=dict(total=time.perf_counter() - t0))


# ----------------------------------------------------------------------------- statistics beyond per-pair log g
# Collapsed feature partitions, per-tile influences, delta-method variances of smooth functions of log g across pairs
# and sections, cross-shell covariances, tile-additive sums, and pooled pair sets. All are the B -> infinity limit of
# the multinomial tile bootstrap with first-order (delta-method) linearization, Var = sum_t c_t^2 with sum_t c_t = 0,
# and process one shell at a time.

def _shell_slab(C: np.ndarray, s) -> np.ndarray:
    """Per-tile slab of one shell or of a summed set of shells.

    Parameters
    ----------
    C : numpy.ndarray
        (T, K, G, G) per-tile counts.
    s : int or sequence of int
        A shell index, or the shells summed into one bin.

    Returns
    -------
    numpy.ndarray
        (T, G, G) float64 counts.
    """
    if np.ndim(s) == 0:
        return np.asarray(C[:, int(s)], dtype=np.float64)
    idx = [int(i) for i in s]
    out = np.array(C[:, idx[0]], dtype=np.float64)
    for i in idx[1:]:
        out += C[:, i]
    return out


def collapse_tensor(counts_tiles: np.ndarray, totals_tiles: np.ndarray, labels) -> tuple[np.ndarray, np.ndarray]:
    r"""Collapse the features of a tile tensor to the groups of a partition.

    Features labeled -1 leave the pairs but not the reference: when any label is -1, a last group ("rest") collects
    them, so that every tile keeps its full pair total :math:`P_{\mathrm{tot},t}(k)` and feature total
    :math:`n_{\mathrm{tot},t}`. The random-label reference of a collapsed pair :math:`(A, B)` is then
    :math:`P_\mathrm{tot}(k) N_A N_B / n_\mathrm{tot}^2` with the full :math:`P_\mathrm{tot}` and
    :math:`n_\mathrm{tot}`, the reference of a merged feature group, and ``closed_form_log_g(C, N)`` gives its point
    estimate and SD for all group pairs at once. Results for pairs with the rest group are not meaningful
    (:func:`scote.uncertainty.collapsed_log_g` removes them).

    Parameters
    ----------
    counts_tiles : numpy.ndarray
        (T, K, G, G) per-tile pair counts.
    totals_tiles : numpy.ndarray
        (T, G) per-tile feature totals.
    labels : array-like of int
        (G,) group 0 .. L - 1 of every feature, or -1 for none.

    Returns
    -------
    C : numpy.ndarray
        (T, K, L', L') float64 summed counts, :math:`C_{t,k,A,B} = \sum_{a \in A, b \in B} C_{t,k,a,b}`;
        L' = L + 1 if any label is -1, else L.
    N : numpy.ndarray
        (T, L') float64 summed totals.

    Raises
    ------
    ValueError
        If ``labels`` does not have shape (G,) or contains values below -1.
    """
    labels = np.asarray(labels, dtype=np.int64)
    C = counts_tiles
    T, K, G, _ = C.shape
    if labels.shape != (G,):
        raise ValueError(f"labels must have shape ({G},), got {labels.shape}")
    if labels.min(initial=0) < -1:
        raise ValueError("labels must be >= -1")
    L = int(labels.max(initial=-1)) + 1
    Lp = L + int(bool((labels == -1).any()))
    M = np.zeros((G, Lp))
    M[np.arange(G), np.where(labels < 0, L, labels)] = 1.0
    out = np.empty((T, K, Lp, Lp))
    for k in range(K):
        Ck = np.asarray(C[:, k], dtype=np.float64)
        out[:, k] = np.matmul(M.T[None], np.matmul(Ck, M))
    N = np.asarray(totals_tiles, dtype=np.float64) @ M
    return out, N


def collapsed_log_g(counts_tiles: np.ndarray, totals_tiles: np.ndarray, labels, names=None, bins=None, **kw):
    r""":func:`scote.uncertainty.closed_form_log_g` of collapsed feature groups, without the rest group.

    Parameters
    ----------
    counts_tiles : numpy.ndarray
        (T, K, G, G) per-tile pair counts.
    totals_tiles : numpy.ndarray
        (T, G) per-tile feature totals.
    labels : array-like of int
        (G,) group 0 .. L - 1 of every feature, -1 for none (:func:`scote.uncertainty.collapse_tensor`).
    names : sequence of str or None
        Names of the L groups; ``None`` gives ``"0"``, ``"1"``, ...
    bins : pandas.DataFrame or None
        The bins table (:func:`scote.uncertainty.closed_form_log_g`).
    **kw
        Passed to :func:`scote.uncertainty.closed_form_log_g` (``backend``, ``dtype``, ``min_tiles_warn``,
        ``symmetrize``).

    Returns
    -------
    scote.TwoPointResult
        The result of :func:`scote.uncertainty.closed_form_log_g` for the L groups: ``log_g``, ``sd``, ``n_eff_pairs``,
        ``P``, ``t_eff``, ``flag`` of shape (K, L, L), ``totals`` (L,) and ``totals_tiles`` (T, L) of the groups.
        ``P_tot`` is the total pair count of every bin including the pairs of the rest group (the reference of the
        groups), ``params["n_total"]`` the full feature total :math:`n_\mathrm{tot}` of the reference, and
        ``counts_tiles`` is ``None``.
    """
    import dataclasses
    labels = np.asarray(labels)
    C, N = collapse_tensor(counts_tiles, totals_tiles, labels)
    L = int(labels.max(initial=-1)) + 1
    res = closed_form_log_g(C, N, bins, None, **kw)
    names = [str(f) for f in names] if names is not None else [str(i) for i in range(L)]
    if len(names) != L:
        raise ValueError(f"{len(names)} group names for {L} groups")
    params = dict(res.params, n_total=float(N.sum()), collapsed_groups=L)
    return dataclasses.replace(
        res, log_g=res.log_g[:, :L, :L], sd=res.sd[:, :L, :L], n_eff_pairs=res.n_eff_pairs[:, :L, :L],
        P=res.P[:, :L, :L], t_eff=res.t_eff[:, :L, :L], flag=res.flag[:, :L, :L], totals=res.totals[:L],
        totals_tiles=res.totals_tiles[:, :L], features=names, counts_tiles=None, params=params)


def influence_log_g(counts_tiles: np.ndarray, totals_tiles: np.ndarray, shells=None,
                    symmetrize: bool = True) -> np.ndarray:
    r"""Per-tile linearized contributions :math:`l_t` of :math:`\log g` (module docstring).

    The closed-form tile-bootstrap covariance of any two :math:`\log g` values (other pairs, other shells, or the same
    tiles of another section on a common tile grid) is :math:`\sum_t l_t(i)\, l_t(j)`, and
    :math:`\mathrm{Var}(\log g_{ab}) = \sum_t l_t(ab)^2`.

    Parameters
    ----------
    counts_tiles : numpy.ndarray
        (T, K, G, G) per-tile pair counts.
    totals_tiles : numpy.ndarray
        (T, G) per-tile feature totals.
    shells : sequence or None
        ``None`` (all K shells) or a sequence whose items are shell indices or sequences of shell indices (a sequence
        is summed into one bin: counts summed, reference from the summed :math:`P_\mathrm{tot}`).
    symmetrize : bool
        Use the symmetrized slabs :math:`(P_t + P_t^\top) / 2`, as :func:`scote.uncertainty.closed_form_log_g`, so that
        :math:`l_t(ab) = l_t(ba)`.

    Returns
    -------
    numpy.ndarray
        (T, K', G, G) float64 influences; NaN where :math:`P_{ab} = 0`.
    """
    C = counts_tiles
    T, K, G, _ = C.shape
    N = np.asarray(totals_tiles, dtype=np.float64)
    sel = list(range(K)) if shells is None else list(shells)
    n_a = N.sum(0)
    n = float(n_a.sum())
    m = N.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(n_a > 0, N / n_a, 0.0)
    base = -R[:, :, None] - R[:, None, :]
    base += (2.0 * m / n)[:, None, None]
    out = np.empty((T, len(sel), G, G))
    for j, s in enumerate(sel):
        Ck = _shell_slab(C, s)
        if symmetrize:
            Ck += np.swapaxes(Ck, 1, 2)
            Ck *= 0.5
        P = Ck.sum(0)
        Qt = Ck.sum(axis=(1, 2))
        Q = float(Qt.sum())
        o = out[:, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(Ck, P, out=o)
        del Ck
        if Q != 0:
            o -= (Qt / Q)[:, None, None]
        o += base
    return out


def closed_form_from_influence(influence: np.ndarray) -> np.ndarray:
    r"""SD of :math:`\log g` from an influence array.

    Parameters
    ----------
    influence : numpy.ndarray
        (T, K', G, G) influences of :func:`scote.uncertainty.influence_log_g`.

    Returns
    -------
    numpy.ndarray
        (K', G, G) :math:`\sqrt{\sum_t l_t^2}`, equal to the ``sd`` of :func:`scote.uncertainty.closed_form_log_g` on
        the same shells up to rounding; NaN where influence is NaN.
    """
    influence = np.asarray(influence)
    out = np.empty(influence.shape[1:])
    for j in range(influence.shape[1]):
        out[j] = np.sqrt(np.einsum("t...,t...->...", influence[:, j], influence[:, j]))
    return out


def delta_var(grad, l_list, joint: bool) -> float:
    r"""Delta-method variance of a smooth statistic :math:`\psi` of the :math:`\log g` values of one or several
    sections.

    Parameters
    ----------
    grad : numpy.ndarray or list of numpy.ndarray
        Per section :math:`s`, :math:`\partial \psi / \partial \log g` over the pairs used, shape (n_s,). A single
        array is one section.
    l_list : numpy.ndarray or list of numpy.ndarray
        The matching influences restricted to those pairs, shape (T_s, n_s) (e.g.
        ``influence_log_g(C, N, [k])[:, 0][:, ia, ib]``).
    joint : bool
        ``True``: the sections share the tile index (registered sections on one tile grid, equal T) and one
        multinomial weight vector, so the per-tile contributions :math:`c_t = \sum_s l_{s,t} \cdot \nabla_s \psi` are
        summed before squaring;
        ``False``: independent tile resamples per section, the variances add.

    Returns
    -------
    float
        :math:`\sum_t c_t^2` (joint) or :math:`\sum_s \sum_t (l_{s,t} \cdot \nabla_s \psi)^2` (independent). NaN in a
        used influence or gradient propagates (select finite pairs first).

    Raises
    ------
    ValueError
        If the numbers or shapes of gradients and influences do not match, or ``joint`` sections differ in
        :math:`T`.
    """
    if not isinstance(grad, (list, tuple)):
        grad, l_list = [grad], [l_list]
    grad = [np.asarray(g, dtype=np.float64).ravel() for g in grad]
    l_list = [np.asarray(infl, dtype=np.float64) for infl in l_list]
    if len(grad) != len(l_list):
        raise ValueError(f"{len(grad)} gradients but {len(l_list)} influence arrays")
    for g, infl in zip(grad, l_list):
        if infl.ndim != 2 or infl.shape[1] != g.shape[0]:
            raise ValueError(f"influence shape {infl.shape} does not match gradient length {g.shape[0]}")
    contrib = [infl @ g for g, infl in zip(grad, l_list)]                # (T_s,) each
    if joint:
        if any(c.shape[0] != contrib[0].shape[0] for c in contrib):
            raise ValueError("joint=True needs the same tiles (equal T) in every section")
        c = np.sum(contrib, axis=0)
        return float(c @ c)
    return float(sum(c @ c for c in contrib))


def stat_ci(psi_value, var, level: float = 0.95) -> tuple:
    r"""Symmetric two-sided normal interval of a statistic.

    Parameters
    ----------
    psi_value : float or numpy.ndarray
        The statistic.
    var : float or numpy.ndarray
        Its variance.
    level : float
        Coverage of the interval.

    Returns
    -------
    tuple
        :math:`(\psi - z \sqrt{\mathrm{var}}, \psi + z \sqrt{\mathrm{var}})` with
        ``z = scipy.stats.norm.ppf((1 + level) / 2)``.
    """
    from scipy.stats import norm
    z = norm.ppf(0.5 + level / 2.0)
    s = np.sqrt(var)
    return psi_value - z * s, psi_value + z * s


def _xy(x, y):
    """Two statistics inputs as flat float64 arrays of equal length.

    Parameters
    ----------
    x, y : array-like
        Values of the two sections.

    Returns
    -------
    tuple
        (x, y, n).

    Raises
    ------
    ValueError
        If the lengths differ.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"x and y differ in length: {x.shape} vs {y.shape}")
    return x, y, x.size


def _moments(x, y):
    """Centered values and second moments (divisor n).

    Parameters
    ----------
    x, y : numpy.ndarray
        (n,) values.

    Returns
    -------
    tuple
        (dx, dy, sxx, syy, sxy).
    """
    dx, dy = x - x.mean(), y - y.mean()
    return dx, dy, float(np.mean(dx * dx)), float(np.mean(dy * dy)), float(np.mean(dx * dy))


def grad_pearson(x, y):
    r"""Pearson :math:`r` of :math:`(x, y)` and its gradient.

    Parameters
    ----------
    x : array-like
        (n,) values of the first section (e.g. :math:`\log g` of a set of pairs on platform A).
    y : array-like
        (n,) values of the second section, same pairs.

    Returns
    -------
    r : float
        Pearson correlation.
    grad : list of numpy.ndarray
        :math:`[\partial r / \partial x, \partial r / \partial y]`, each (n,).

    Raises
    ------
    ValueError
        If x and y differ in length.
    """
    x, y, n = _xy(x, y)
    dx, dy, sxx, syy, sxy = _moments(x, y)
    s = np.sqrt(sxx * syy)
    r = sxy / s
    return r, [(dy / s - r * dx / sxx) / n, (dx / s - r * dy / syy) / n]


def grad_rms(x, y):
    r"""RMS difference :math:`\sqrt{\langle (y - x)^2 \rangle}` and its gradient.

    Parameters
    ----------
    x : array-like
        (n,) values of the first section (e.g. :math:`\log g` of a set of pairs on platform A).
    y : array-like
        (n,) values of the second section, same pairs.

    Returns
    -------
    rms : float
        Root mean square of :math:`y - x`.
    grad : list of numpy.ndarray
        The derivatives with respect to ``x`` and ``y``, each (n,).

    Raises
    ------
    ValueError
        If x and y differ in length.
    """
    x, y, n = _xy(x, y)
    d = y - x
    R = float(np.sqrt(np.mean(d * d)))
    g = d / (n * R)
    return R, [-g, g]


def grad_mean_diff(x, y):
    r"""Mean difference :math:`\langle y - x \rangle` and its gradient.

    Parameters
    ----------
    x : array-like
        (n,) values of the first section (e.g. :math:`\log g` of a set of pairs on platform A).
    y : array-like
        (n,) values of the second section, same pairs.

    Returns
    -------
    mean_diff : float
        Mean of :math:`y - x`.
    grad : list of numpy.ndarray
        :math:`[-1/n, 1/n]` as two (n,) arrays.

    Raises
    ------
    ValueError
        If x and y differ in length.
    """
    x, y, n = _xy(x, y)
    return float(np.mean(y - x)), [np.full(n, -1.0 / n), np.full(n, 1.0 / n)]


def grad_ols_slope(x, y):
    r"""OLS slope :math:`b = s_{xy} / s_{xx}` of ``y`` on ``x`` (``np.polyfit(x, y, 1)[0]``) and its gradient.

    Parameters
    ----------
    x : array-like
        (n,) values of the first section (e.g. :math:`\log g` of a set of pairs on platform A).
    y : array-like
        (n,) values of the second section, same pairs.

    Returns
    -------
    slope : float
        The slope.
    grad : list of numpy.ndarray
        :math:`[\partial b / \partial x, \partial b / \partial y]`, each (n,).

    Raises
    ------
    ValueError
        If x and y differ in length.
    """
    x, y, n = _xy(x, y)
    dx, dy, sxx, syy, sxy = _moments(x, y)
    b = sxy / sxx
    return b, [(dy - 2.0 * b * dx) / (n * sxx), dx / (n * sxx)]


def grad_ols_intercept(x, y):
    r"""OLS intercept :math:`a = \bar y - b \bar x` of ``y`` on ``x`` (``np.polyfit(x, y, 1)[1]``) and its gradient.

    Parameters
    ----------
    x : array-like
        (n,) values of the first section (e.g. :math:`\log g` of a set of pairs on platform A).
    y : array-like
        (n,) values of the second section, same pairs.

    Returns
    -------
    intercept : float
        The intercept.
    grad : list of numpy.ndarray
        :math:`[\partial a / \partial x, \partial a / \partial y]`, each (n,).

    Raises
    ------
    ValueError
        If x and y differ in length.
    """
    x, y, n = _xy(x, y)
    b, (gbx, gby) = grad_ols_slope(x, y)
    xm = float(x.mean())
    return float(y.mean() - b * xm), [-b / n - xm * gbx, 1.0 / n - xm * gby]


def grad_deming(x, y, lam: float):
    r"""Deming slope of ``y`` on ``x`` and its gradient (``lam`` held fixed).

    .. math::

        \beta = \frac{s_{yy} - \lambda s_{xx} + \sqrt{(s_{yy} - \lambda s_{xx})^2 + 4 \lambda s_{xy}^2}}{2 s_{xy}} .

    Parameters
    ----------
    x : array-like
        (n,) values of the first section (e.g. :math:`\log g` of a set of pairs on platform A).
    y : array-like
        (n,) values of the second section, same pairs.
    lam : float
        Error-variance ratio :math:`\lambda` = Var(error of ``y``) / Var(error of ``x``); 1 gives the major axis.

    Returns
    -------
    slope : float
        The Deming slope.
    grad : list of numpy.ndarray
        :math:`[\partial \beta / \partial x, \partial \beta / \partial y]`, each (n,).

    Raises
    ------
    ValueError
        If x and y differ in length.
    """
    x, y, n = _xy(x, y)
    dx, dy, sxx, syy, sxy = _moments(x, y)
    D = syy - lam * sxx
    S = np.sqrt(D * D + 4.0 * lam * sxy * sxy)
    beta = (D + S) / (2.0 * sxy)
    dD = (1.0 + D / S) / (2.0 * sxy)                                 # dbeta/dD; D = syy - lam sxx
    d_sxy = 2.0 * lam / S - beta / sxy
    gx = (-lam * dD * 2.0 * dx + d_sxy * dy) / n
    gy = (dD * 2.0 * dy + d_sxy * dx) / n
    return beta, [gx, gy]


def grad_numeric(psi, x, eps: float = 1e-6):
    """Central finite-difference gradient of a scalar statistic.

    Parameters
    ----------
    psi : callable
        Statistic; called with the structure of ``x`` and returns a scalar.
    x : numpy.ndarray or list of numpy.ndarray
        Point of evaluation: one array, or one array per section.
    eps : float
        Relative step: the step of element ``i`` is ``eps * max(1, abs(x[i]))``.

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        The gradient in the structure of ``x``. Costs two evaluations of ``psi`` per element.
    """
    single = not isinstance(x, (list, tuple))
    xs = [np.array(v, dtype=np.float64) for v in ([x] if single else x)]
    call = (lambda v: psi(v[0])) if single else psi
    out = []
    for v in xs:
        flat = v.reshape(-1)
        g = np.empty(flat.size)
        for i in range(flat.size):
            x0 = flat[i]
            h = eps * max(1.0, abs(x0))
            flat[i] = x0 + h
            fp = float(call(xs))
            flat[i] = x0 - h
            fm = float(call(xs))
            flat[i] = x0
            g[i] = (fp - fm) / (2.0 * h)
        out.append(g.reshape(v.shape))
    return out[0] if single else out


def cross_shell_cov(influence: np.ndarray, pair) -> np.ndarray:
    r"""Closed-form tile-bootstrap covariance of :math:`\log g_{ab}` across shells.

    Parameters
    ----------
    influence : numpy.ndarray
        (T, K', G, G) influences of :func:`scote.uncertainty.influence_log_g`.
    pair : tuple of int
        Feature indices (a, b).

    Returns
    -------
    numpy.ndarray
        (K', K') covariance :math:`\sum_t l_t(k)\, l_t(k')`.
    """
    a, b = pair
    L = np.asarray(influence[:, :, a, b], dtype=np.float64)                 # (T, K')
    return L.T @ L


def cross_shell_corr(influence: np.ndarray, pair) -> np.ndarray:
    r"""Correlation of :math:`\log g_{ab}` across shells.

    Parameters
    ----------
    influence : numpy.ndarray
        (T, K', G, G) influences of :func:`scote.uncertainty.influence_log_g`.
    pair : tuple of int
        Feature indices (a, b).

    Returns
    -------
    numpy.ndarray
        (K', K') correlation matrix (:func:`scote.uncertainty.cross_shell_cov` normalized by the SDs).
    """
    c = cross_shell_cov(influence, pair)
    s = np.sqrt(np.diag(c))
    with np.errstate(divide="ignore", invalid="ignore"):
        return c / np.outer(s, s)


def tile_sum_var(S_tiles: np.ndarray, grad):
    r"""Closed-form tile-bootstrap variance of a function of tile-additive sums.

    Parameters
    ----------
    S_tiles : numpy.ndarray
        (T, m) per-tile values of m tile-additive sums :math:`S = \sum_t S_t` (or (T,) for m = 1).
    grad : numpy.ndarray
        (m,) gradient :math:`\partial f / \partial S`, or (p, m) for p statistics.

    Returns
    -------
    float or numpy.ndarray
        :math:`\nabla f^\top \left[\sum_t (S_t - S/T)(S_t - S/T)^\top\right] \nabla f`; a (p, p) covariance
        matrix for a (p, m) gradient.
    """
    S = np.asarray(S_tiles, dtype=np.float64)
    if S.ndim == 1:
        S = S[:, None]
    D = S - S.mean(0)
    g = np.asarray(grad, dtype=np.float64)
    if g.ndim <= 1:
        c = D @ g.reshape(-1)
        return float(c @ c)
    Cg = D @ g.T
    return Cg.T @ Cg


def tile_sum_stat(f, S_tiles: np.ndarray, eps: float = 1e-6):
    r"""Value and closed-form SD of a smooth statistic of tile-additive sums (numerical gradient).

    Parameters
    ----------
    f : callable
        Maps the (m,) vector of sums to a scalar or a (p,) vector.
    S_tiles : numpy.ndarray
        (T, m) per-tile values.
    eps : float
        Relative finite-difference step (:func:`scote.uncertainty.grad_numeric`).

    Returns
    -------
    value : numpy.ndarray
        :math:`f(\sum_t S_t)`, shape () or (p,).
    sd : numpy.ndarray
        Its SD, same shape.
    """
    S_tiles = np.asarray(S_tiles, dtype=np.float64)
    S = S_tiles.sum(0)
    val = np.asarray(f(S), dtype=np.float64)
    if val.ndim == 0:
        g = grad_numeric(lambda v: float(f(v)), S, eps)
        return val, np.sqrt(tile_sum_var(S_tiles, g))
    Gm = np.stack([grad_numeric(lambda v, j=j: float(np.asarray(f(v))[j]), S, eps) for j in range(val.size)])
    return val, np.sqrt(np.diag(tile_sum_var(S_tiles, Gm)))


def pooled_log_g(counts_tiles: np.ndarray, totals_tiles: np.ndarray, pairs_a, pairs_b,
                 return_influence: bool = False, symmetrize: bool = True) -> dict:
    r"""Pooled :math:`\log g` of a set of ordered pairs per shell, with its closed-form tile-bootstrap SD.

    .. math::

        \log g_\mathrm{pool}
        = \log \frac{\sum_p P(a_p b_p)}{\sum_p P_\mathrm{tot}\, n(a_p)\, n(b_p) / n_\mathrm{tot}^2} ,

    with the per-tile influence

    .. math::

        l_t = \frac{S_t}{S} - \frac{Q_t}{Q} - \sum_p w_p \left(r_t(a_p) + r_t(b_p)\right)
        + \frac{2 m_t}{n_\mathrm{tot}} ,
        \qquad w_p = \frac{n(a_p)\, n(b_p)}{\sum_{p'} n(a_{p'})\, n(b_{p'})} ,

    and :math:`S_t = \sum_p P_t(a_p b_p)`.

    Parameters
    ----------
    counts_tiles : numpy.ndarray
        (T, K, G, G) per-tile pair counts.
    totals_tiles : numpy.ndarray
        (T, G) per-tile feature totals.
    pairs_a : array-like of int
        First feature index of every pair.
    pairs_b : array-like of int
        Second feature index of every pair, same length.
    return_influence : bool
        Also return the (T, K) influence (to combine pooled sets, e.g. a difference, with
        :func:`scote.uncertainty.delta_var`).
    symmetrize : bool
        :math:`P_t(a_p b_p) \to (P_t(a_p b_p) + P_t(b_p a_p)) / 2` in :math:`S_t`, as
        :func:`scote.uncertainty.closed_form_log_g`.

    Returns
    -------
    dict
        ``log_g`` (K,), ``sd`` (K,), ``n_eff_pairs`` (K,) :math:`= S^2 / \sum_t (S_t - S/T)^2`, ``P`` (K,)
        :math:`= S`, ``n_pairs``, and with ``return_influence`` ``influence`` (T, K). A shell without any pair gives
        ``log_g`` = ``sd`` = NaN.

    Raises
    ------
    ValueError
        If the pair lists are empty or differ in length.
    """
    C = counts_tiles
    T, K, G, _ = C.shape
    pa = np.asarray(pairs_a, dtype=np.int64).ravel()
    pb = np.asarray(pairs_b, dtype=np.int64).ravel()
    if pa.shape != pb.shape or pa.size == 0:
        raise ValueError("pairs_a and pairs_b must be non-empty and of equal length")
    N = np.asarray(totals_tiles, dtype=np.float64)
    n_a = N.sum(0)
    n = float(n_a.sum())
    m = N.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(n_a > 0, N / n_a, 0.0)
    nn = n_a[pa] * n_a[pb]
    D = float(nn.sum())
    w = nn / D
    v = -(R[:, pa] @ w + R[:, pb] @ w) + 2.0 * m / n                 # (T,) feature part of l_t
    log_g = np.empty(K)
    sd = np.empty(K)
    n_eff = np.empty(K)
    Ssum = np.empty(K)
    L = np.empty((T, K)) if return_influence else None
    for k in range(K):
        Ck = C[:, k]
        Qt = np.asarray(Ck.sum(axis=(1, 2), dtype=np.float64))
        Q = float(Qt.sum())
        St = np.asarray(Ck[:, pa, pb], dtype=np.float64).sum(1)
        S = float(St.sum())                                          # point estimate: raw slabs
        if symmetrize:
            St = 0.5 * (St + np.asarray(Ck[:, pb, pa], dtype=np.float64).sum(1))
        Sv = float(St.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            log_g[k] = np.log(S / (Q * D / max(n, 1.0) ** 2)) if Q > 0 else np.nan   # empty shell: no pairs at all
            infl = St / Sv - (Qt / Q if Q != 0 else 0.0) + v
            dS = St - Sv / T
            n_eff[k] = Sv * Sv / max(float(dS @ dS), 1e-300)
        sd[k] = np.sqrt(infl @ infl) if S > 0 else np.nan
        Ssum[k] = S
        if L is not None:
            L[:, k] = infl
    out = dict(log_g=log_g, sd=sd, n_eff_pairs=n_eff, P=Ssum, n_pairs=int(pa.size))
    if L is not None:
        out["influence"] = L
    return out
