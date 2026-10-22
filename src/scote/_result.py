r"""Results of the two-point functions: :class:`scote.TwoPointResult` (:math:`\log g` with its closed-form
tile-bootstrap uncertainty and the distance bins) and :class:`scote.PairCounts` (global pair counts of the lattice
hierarchy)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .uncertainty import T_EFF_MIN, T_EFF_OK

__all__ = ["PairCounts", "TwoPointResult"]

# arrays of shape (K, G, G) and their file names in a result directory
_PAIR_ARRAYS = ("log_g", "sd", "n_eff_pairs", "P", "t_eff", "flag")
_BIN_COLUMNS = ("level", "shell", "r_from", "r_to", "r_central")
_TILE_COLUMNS = ("column", "row", "x_from", "y_from")
_CLOSED_METHOD = "closed-form tile block bootstrap (B -> infinity; delta method for log g; intervals log g -+ z sd)"
_RELIABILITY_RULE = (f"flag 0: T_eff >= {T_EFF_OK:g} (normal interval); 1: {T_EFF_MIN:g} <= T_eff < {T_EFF_OK:g} "
                     f"(flagged); 2: T_eff < {T_EFF_MIN:g} or P = 0 (no interval, lo = hi = NaN); "
                     "T_eff = P^2 / sum_t P_t^2")


@dataclass
class TwoPointResult:
    r"""Scale-resolved two-point functions of all feature pairs with their uncertainties.

    K distance bins over all levels of the hierarchy, G features, T tiles. Bin ``k`` covers the distances
    ``bins.r_from[k] <= r < bins.r_to[k]``; ``bins.r_central[k]`` is the central distance of the bin. Bins of
    consecutive lattice levels overlap (each level keeps its own window), so the counts of a pair over all bins sum to
    slightly more than its number of pairs; :math:`g` is a ratio within one bin and is not affected.

    Results of :func:`scote.twopoint`, and of the statistics of
    :mod:`scote.uncertainty` that work on stored tile tensors (:func:`scote.uncertainty.closed_form_log_g`,
    :func:`scote.uncertainty.collapsed_log_g`).

    Attributes
    ----------
    log_g : numpy.ndarray
        (K, G, G) :math:`\log g_{ab}(k)`, the log of the pair count over its random-label reference
        :math:`P_\mathrm{tot}(k)\, n_a n_b / n_\mathrm{tot}^2`; ``-inf`` where :math:`P = 0`. float64, or float32 when
        the run was asked for float32 output.
    sd : numpy.ndarray
        (K, G, G) SD of :math:`\log g` from the closed-form tile block bootstrap; NaN where :math:`P = 0`.
    n_eff_pairs : numpy.ndarray
        (K, G, G) :math:`P^2 / \mathrm{Var}(P)`, the effective number of independent pairs implied by the tile scatter
        of the count.
    P : numpy.ndarray
        (K, G, G) float64 ordered pair counts (each unordered pair of molecules is counted in (a, b) and (b, a)).
    t_eff : numpy.ndarray
        (K, G, G) float32 effective number of tiles carrying the pair, :math:`T_\mathrm{eff} = P^2 / \sum_t P_t^2`.
    flag : numpy.ndarray
        (K, G, G) int8 reliability of the interval: 0 normal (:math:`T_\mathrm{eff} \ge 20`), 1 flagged
        (:math:`5 \le T_\mathrm{eff} < 20`), 2 no interval (:math:`T_\mathrm{eff} < 5` or :math:`P = 0`).
    P_tot : numpy.ndarray
        (K,) float64 total pair count of every bin, :math:`P_\mathrm{tot} = \sum_{ab} P_{ab}`.
    totals : numpy.ndarray
        (G,) float64 feature totals :math:`n_a`.
    totals_tiles : numpy.ndarray
        (T, G) float64 feature totals per tile.
    bins : pandas.DataFrame
        One row per bin: ``level`` (0 = the finest level), ``shell`` (index of the bin within its level), ``r_from``,
        ``r_to``, ``r_central`` (µm).
    features : list of str
        Feature names in the order of the G axes.
    counts_tiles : numpy.ndarray or None
        (T, K, G, G) float64 per-tile pair counts (a pair belongs to the tile of its first member), if the run kept
        them; input of :func:`scote.uncertainty.closed_form_log_g`, :func:`scote.uncertainty.influence_log_g`, and the
        other statistics of :mod:`scote.uncertainty`.
    tiles : pandas.DataFrame or None
        One row per tile index: ``column`` and ``row`` of the tile on the grid of side ``params["tile_um"]`` from
        ``params["origin"]``, and its lower corner ``x_from``, ``y_from`` (µm).
    params : dict
        Parameters of the run (input type, ``delta``, bin size, number of levels, shells per level, tile size, origin,
        ``backend`` (``"cpu"`` or ``"gpu"``) and its ``options`` as a dict, ...).
    timings : dict
        Wall times of the steps (seconds) and per-level records.
    path : pathlib.Path or None
        Directory the arrays are memory-mapped from, if the run streamed them to disk or the result was loaded from
        a directory.
    """
    log_g: np.ndarray
    sd: np.ndarray
    n_eff_pairs: np.ndarray
    P: np.ndarray
    t_eff: np.ndarray
    flag: np.ndarray
    P_tot: np.ndarray
    totals: np.ndarray
    totals_tiles: np.ndarray
    bins: pd.DataFrame
    features: list
    counts_tiles: np.ndarray | None = None
    tiles: pd.DataFrame | None = None
    params: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    path: Path | None = None
    _interval_95: tuple | None = field(default=None, init=False, repr=False, compare=False)

    # ------------------------------------------------------------------ shape
    @property
    def n_tiles(self) -> int:
        """Number of tiles T."""
        return int(self.totals_tiles.shape[0])

    @property
    def n_features(self) -> int:
        """Number of features G."""
        return int(self.log_g.shape[1])

    @property
    def n_distance_bins(self) -> int:
        """Number of distance bins K over all levels."""
        return int(self.log_g.shape[0])

    @property
    def r(self) -> np.ndarray:
        """(K,) central distance of every bin (µm), ``bins.r_central``."""
        return self.bins["r_central"].to_numpy()

    # ------------------------------------------------------------------ intervals
    def interval(self, ci=(2.5, 97.5)) -> tuple[np.ndarray, np.ndarray]:
        r"""Normal interval of :math:`\log g` from the closed-form SD.

        Parameters
        ----------
        ci : tuple of float
            Percentages of the two ends; the default is the central 95 % interval
            :math:`\log g \mp 1.96\,\mathrm{sd}`.

        Returns
        -------
        lo : numpy.ndarray
            (K, G, G) float64 lower end, :math:`\log g + z_\mathrm{lo}\,\mathrm{sd}` with
            ``z_lo = scipy.stats.norm.ppf(ci[0] / 100)``; NaN where ``flag == 2``.
        hi : numpy.ndarray
            (K, G, G) float64 upper end; NaN where ``flag == 2``.
        """
        from scipy.stats import norm
        z_lo, z_hi = norm.ppf(np.asarray(ci, dtype=np.float64) / 100.0)
        log_g = np.asarray(self.log_g, dtype=np.float64)
        sd = np.asarray(self.sd, dtype=np.float64)
        lo = log_g + z_lo * sd
        hi = log_g + z_hi * sd
        no = np.asarray(self.flag) == 2
        lo[no] = np.nan
        hi[no] = np.nan
        return lo, hi

    @property
    def lo(self) -> np.ndarray:
        """(K, G, G) lower end of the central 95 % interval (:meth:`scote.TwoPointResult.interval`), computed on
        first access."""
        return self._default_interval()[0]

    @property
    def hi(self) -> np.ndarray:
        """(K, G, G) upper end of the central 95 % interval (:meth:`scote.TwoPointResult.interval`), computed on
        first access."""
        return self._default_interval()[1]

    def _default_interval(self) -> tuple[np.ndarray, np.ndarray]:
        """The central 95 % interval, computed once."""
        if self._interval_95 is None:
            self._interval_95 = self.interval()
        return self._interval_95

    # ------------------------------------------------------------------ access
    def feature_index(self, name) -> int:
        """Index of a feature.

        Parameters
        ----------
        name : str or int
            Feature name, or an index (returned unchanged).

        Returns
        -------
        int
            Position on the G axes.

        Raises
        ------
        KeyError
            If the name is not a feature of the result.
        """
        if isinstance(name, (int, np.integer)):
            return int(name)
        try:
            return self.features.index(str(name))
        except ValueError:
            raise KeyError(f"{name!r} is not a feature of this result") from None

    def pair(self, a, b, ci=(2.5, 97.5)) -> pd.DataFrame:
        """The two-point function of one feature pair as a table.

        Parameters
        ----------
        a : str or int
            First feature (name or index).
        b : str or int
            Second feature (name or index).
        ci : tuple of float
            Interval percentages (:meth:`scote.TwoPointResult.interval`).

        Returns
        -------
        pandas.DataFrame
            One row per bin: ``level``, ``r_from``, ``r_to``, ``r_central``, ``log_g``, ``sd``, ``lo``, ``hi``,
            ``t_eff``, ``flag``, ``P``.
        """
        i, j = self.feature_index(a), self.feature_index(b)
        lo, hi = self.interval(ci)
        out = self.bins.copy()
        out["log_g"] = np.asarray(self.log_g[:, i, j], dtype=np.float64)
        out["sd"] = np.asarray(self.sd[:, i, j], dtype=np.float64)
        out["lo"] = lo[:, i, j]
        out["hi"] = hi[:, i, j]
        out["t_eff"] = np.asarray(self.t_eff[:, i, j])
        out["flag"] = np.asarray(self.flag[:, i, j])
        out["P"] = np.asarray(self.P[:, i, j])
        return out

    def meta(self) -> dict:
        """Description of the result for the metadata file.

        Returns
        -------
        dict
            The parameters of the run (``params``, as top-level keys), the uncertainty method and reliability rule,
            the numbers of tiles, bins, and features, and the timings.
        """
        out = dict(self.params)
        out.update(uncertainty="closed", uncertainty_method=_CLOSED_METHOD, reliability_rule=_RELIABILITY_RULE,
                   n_tiles=self.n_tiles, n_distance_bins=self.n_distance_bins, n_features=self.n_features,
                   timings=self.timings)
        return out

    # ------------------------------------------------------------------ files
    def save(self, path, compress: bool = False) -> Path:
        """Write the result to an ``.npz`` file or a directory.

        Parameters
        ----------
        path : str or pathlib.Path
            A path ending in ``.npz``: one archive with the arrays ``log_g``, ``lo``, ``hi``, ``sd``, ``n_eff_pairs``,
            ``P``, ``t_eff``, ``flag``, ``P_tot``, ``totals``, ``totals_tiles``, ``features`` (unicode strings),
            ``level``, ``shell``, ``r_from``, ``r_to``, ``r_central``, the tile table as ``tile_column``, ``tile_row``,
            ``tile_x_from``, ``tile_y_from`` (if present), and ``counts_tiles`` (if present), plus the metadata in
            ``<path without .npz>.json``. Any other path: a directory with one ``.npy`` file per array, ``bins.csv``,
            ``tiles.csv``, ``features.txt``, and ``meta.json``. The archive needs no pickle to be read.
        compress : bool
            Compress the ``.npz`` archive (slow for large tile tensors).

        Returns
        -------
        pathlib.Path
            The path written.
        """
        path = Path(path)
        if path.suffix == ".npz":
            path.parent.mkdir(parents=True, exist_ok=True)
            lo, hi = self._default_interval()
            arrays = dict(log_g=self.log_g, lo=lo, hi=hi, sd=self.sd, n_eff_pairs=self.n_eff_pairs, P=self.P,
                          t_eff=self.t_eff, flag=self.flag, P_tot=self.P_tot, totals=self.totals,
                          totals_tiles=self.totals_tiles, features=np.array(self.features, dtype=str))
            for name in _BIN_COLUMNS:
                arrays[name] = self.bins[name].to_numpy()
            if self.tiles is not None:
                for name in _TILE_COLUMNS:
                    arrays[f"tile_{name}"] = self.tiles[name].to_numpy()
            if self.counts_tiles is not None:
                arrays["counts_tiles"] = self.counts_tiles
            (np.savez_compressed if compress else np.savez)(path, **arrays)
            self.write_meta(path)
            return path
        path.mkdir(parents=True, exist_ok=True)
        for name in _PAIR_ARRAYS:
            target = path / f"{name}.npy"
            arr = getattr(self, name)
            if not (isinstance(arr, np.memmap) and Path(arr.filename).resolve() == target.resolve()):
                np.save(target, arr)
        np.save(path / "P_tot.npy", self.P_tot)
        np.save(path / "totals.npy", self.totals)
        np.save(path / "totals_tiles.npy", self.totals_tiles)
        if self.counts_tiles is not None:
            np.save(path / "counts_tiles.npy", self.counts_tiles)
        self.bins.to_csv(path / "bins.csv", index=False)
        if self.tiles is not None:
            self.tiles.to_csv(path / "tiles.csv", index=False)
        (path / "features.txt").write_text("\n".join(self.features) + "\n")
        self.write_meta(path)
        return path

    def write_meta(self, path) -> Path:
        """Write the metadata (:meth:`scote.TwoPointResult.meta`) of a saved result.

        Parameters
        ----------
        path : str or pathlib.Path
            The ``.npz`` file (metadata go to ``<path without .npz>.json``) or the result directory
            (``meta.json``).

        Returns
        -------
        pathlib.Path
            The metadata file.
        """
        path = Path(path)
        target = path.with_suffix(".json") if path.suffix == ".npz" else path / "meta.json"
        with open(target, "w") as fh:
            json.dump(self.meta(), fh, indent=1, default=_json_default)
        return target

    @classmethod
    def load(cls, path, mmap: bool = True) -> "TwoPointResult":
        """Read a result written by :meth:`scote.TwoPointResult.save`.

        Parameters
        ----------
        path : str or pathlib.Path
            An ``.npz`` file or a result directory.
        mmap : bool
            Directory: memory-map the arrays instead of reading them.

        Returns
        -------
        TwoPointResult
            The result; ``params`` and ``timings`` come from the metadata file if it exists.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no result at {path}")
        if path.suffix == ".npz":
            z = np.load(path)
            bins = _bins_table({name: z[name] for name in _BIN_COLUMNS})
            tiles = (pd.DataFrame({name: z[f"tile_{name}"] for name in _TILE_COLUMNS})
                     if "tile_column" in z else None)
            meta_path = path.with_suffix(".json")
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            return cls(log_g=z["log_g"], sd=z["sd"], n_eff_pairs=z["n_eff_pairs"], P=z["P"], t_eff=z["t_eff"],
                       flag=z["flag"], P_tot=z["P_tot"], totals=z["totals"], totals_tiles=z["totals_tiles"],
                       bins=bins, features=[str(f) for f in z["features"]],
                       counts_tiles=z["counts_tiles"] if "counts_tiles" in z else None, tiles=tiles,
                       params=_params_of(meta), timings=meta.get("timings", {}))
        mode = "r" if mmap else None
        arrays = {name: np.load(path / f"{name}.npy", mmap_mode=mode) for name in _PAIR_ARRAYS}
        meta = json.loads((path / "meta.json").read_text()) if (path / "meta.json").exists() else {}
        ct = path / "counts_tiles.npy"
        bins = pd.read_csv(path / "bins.csv", float_precision="round_trip")
        tiles = (pd.read_csv(path / "tiles.csv", float_precision="round_trip") if (path / "tiles.csv").exists()
                 else None)
        return cls(**arrays, P_tot=np.load(path / "P_tot.npy"), totals=np.load(path / "totals.npy"),
                   totals_tiles=np.load(path / "totals_tiles.npy"), bins=_bins_table(bins),
                   features=(path / "features.txt").read_text().split("\n")[:-1],
                   counts_tiles=np.load(ct, mmap_mode=mode) if ct.exists() else None, tiles=tiles,
                   params=_params_of(meta), timings=meta.get("timings", {}), path=path)


@dataclass
class PairCounts:
    """Global ordered pair counts of the lattice hierarchy (:func:`scote.pair_counts`).

    Attributes
    ----------
    counts : numpy.ndarray
        (K, G, G) float64 ordered pair counts of the K distance bins, or the (K,) per-bin totals when the counts were
        passed to a sink.
    bins : pandas.DataFrame
        One row per bin: ``level``, ``shell``, ``r_from``, ``r_to``, ``r_central`` (µm), as in
        :attr:`scote.TwoPointResult.bins`.
    params : dict
        Parameters of the run (input type, ``delta``, number of levels, shells per level, origin, ``backend``
        (``"cpu"`` or ``"gpu"``) and its ``options`` as a dict, threads).
    timings : dict
        ``levels``: one dict per level (``level``, ``pitch``, ``n_cells``, ``nnz_X``, ``density_X``, ``mode``,
        ``seconds_S``, ``seconds_SX``, ``seconds_XtSX``, ``pair_updates``, ``cell_pairs``, ``seconds_total``);
        ``total``: wall time (s).
    features : list of str or None
        Feature names in the order of the G axes.
    """
    counts: np.ndarray
    bins: pd.DataFrame
    params: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    features: list | None = None

    @property
    def n_distance_bins(self) -> int:
        """Number of distance bins K over all levels."""
        return int(self.counts.shape[0])

    @property
    def r(self) -> np.ndarray:
        """(K,) central distance of every bin (µm), ``bins.r_central``."""
        return self.bins["r_central"].to_numpy()


def _bins_table(columns) -> pd.DataFrame:
    """The bins table from its columns.

    Parameters
    ----------
    columns : dict or pandas.DataFrame
        The columns level, shell, r_from, r_to, r_central.

    Returns
    -------
    pandas.DataFrame
        The table with these columns, in this order.
    """
    return pd.DataFrame({name: np.asarray(columns[name]) for name in _BIN_COLUMNS})


_META_KEYS = ("uncertainty", "uncertainty_method", "reliability_rule", "n_tiles", "n_distance_bins", "n_features",
              "timings")


def _params_of(meta: dict) -> dict:
    """The run parameters of a metadata dict.

    Parameters
    ----------
    meta : dict
        Metadata (:meth:`scote.TwoPointResult.meta`).

    Returns
    -------
    dict
        All keys that are not result descriptions.
    """
    return {k: v for k, v in meta.items() if k not in _META_KEYS}


def _json_default(o):
    """JSON encoding of NumPy scalars and arrays (and ``str()`` of anything else).

    Parameters
    ----------
    o : object
        Object the JSON encoder cannot encode.

    Returns
    -------
    int, float, list, or str
        Its encodable form.
    """
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
