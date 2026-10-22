r"""The per-tile hierarchy of :func:`scote.twopoint` and the global counts of :func:`scote.pair_counts`.

Molecules with enumeration (``_twopoint_molecules``): exact pair enumeration in the linear bins
:math:`[0, \Delta), \dots, [(s - 1) \Delta, s \Delta)` (level 0, :math:`s` = ``shells_per_level``), then lattice levels
:math:`L = 1 \dots n_\mathrm{levels}` on cells of pitch :math:`2^L \Delta`, each keeping the shells
:math:`s/2 \dots s - 1`. Lattices only (``_twopoint_lattice``; bins, or molecules with ``lattice_only``): level 0 =
all shells :math:`0 \dots s - 1` of the lattice of pitch :math:`\Delta` (shell 0 = pairs within one cell), then levels
coarse-grained by 2 with the shells :math:`s/2 \dots s - 1`.

Tiles: square tiles of side ``tile_um`` from the origin; a tile exists if it holds a molecule or bin. Every pair is
attributed to a tile through its first member: at the enumerated level and the first lattice, the tile of the molecule
or bin; at a lattice level, the tile of the center of its cell. Cells of coarse levels (pitch at least about the tile
side) can have their center in a tile without data, or beyond the data; such a pair goes to the next existing tile in
the order of the tiles (by column, then by row), or to the last tile. The global counts and :math:`\log g` do not
depend on this attribution; the tile scatter (``sd``, ``n_eff_pairs``, ``t_eff``, ``flag``) of the coarsest bins
does. The accumulators use the symmetrized slabs, half of every pair in the tile of each member.

The per-tile slabs of every shell stream into the accumulators of the closed-form tile block bootstrap
(:class:`scote.uncertainty.TileMomentAccumulator`), so the (tiles, bins, G, G) tensor is formed only on request
(``keep_tiles``), and every shell is finished as soon as its last slab arrived. The enumerated level gives all its
shells per tile, so its shells are live together. With ``out`` the results are written into memory-mapped files
shell by shell.
"""
from __future__ import annotations

import dataclasses
import sys
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .._threads import with_threads
from .._result import PairCounts, TwoPointResult
from ..uncertainty import TileMomentAccumulator
from ._common import CPUOptions, GPUOptions, LevelTiming, resolve_backend
from .enumeration import bin_intervals, enumeration_block_size, pair_counts_enumerated_groups
from .lattice import build_lattice, coarse_grain, onehot_csr, shell_central_distances

__all__ = ["pair_counts", "twopoint"]

# Host memory of a run above which scote.twopoint warns before allocating (bytes).
MEMORY_WARN_BYTES = 4 * 2**30

BIN_COLUMNS = ["level", "shell", "r_from", "r_to", "r_central"]


def _validate(*, shells_per_level=None, n_levels=None, tile_um=None, delta=None, bin_size=None) -> None:
    """Check the parameters shared by the public functions.

    Parameters
    ----------
    shells_per_level : int or None
        Distance bins per level.
    n_levels : int or None
        Number of coarser levels.
    tile_um, delta, bin_size : float or None
        Lengths (µm).

    Raises
    ------
    ValueError
        If ``shells_per_level`` is not an even integer >= 2, ``n_levels`` is not an integer >= 0, or ``tile_um``,
        ``delta``, or ``bin_size`` is not a positive finite number (parameters that are ``None`` are not checked).
    """
    if shells_per_level is not None and not (isinstance(shells_per_level, (int, np.integer))
                                             and shells_per_level >= 2 and shells_per_level % 2 == 0):
        raise ValueError(f"shells_per_level must be an even integer >= 2 (the coarser levels keep the upper half), "
                         f"got {shells_per_level!r}")
    if n_levels is not None and not (isinstance(n_levels, (int, np.integer)) and n_levels >= 0):
        raise ValueError(f"n_levels must be an integer >= 0, got {n_levels!r}")
    for name, value in (("tile_um", tile_um), ("delta", delta), ("bin_size", bin_size)):
        if value is not None and not (isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value)
                                      and value > 0):
            raise ValueError(f"{name} must be a positive number, got {value!r}")


def _check_memory(*, T: int, K: int, G: int, live: int, out, keep_tiles: bool, dtype, acc_dtype,
                  enumeration: bool) -> int:
    """Warn before allocating if the arrays of a run need more than ``MEMORY_WARN_BYTES`` of host memory.

    Counted: the (K, G, G) outputs (``log_g``, ``sd``, ``n_eff_pairs``, ``P``, ``t_eff``, ``flag``) unless they are
    memory-mapped into ``out``; the accumulators of the live shell slots (four (G, G) arrays each; the ``s`` slots of
    the enumerated level are live together); the (s, G, G) counts of one tile of the enumerated level; and the
    (T, K, G, G) tile tensor with ``keep_tiles``.

    Parameters
    ----------
    T, K, G : int
        Tiles, distance bins, features.
    live : int
        Shell slots that are live at the same time.
    out : str, pathlib.Path, or None
        Output directory of the run.
    keep_tiles : bool
        Whether the tile tensor is kept.
    dtype, acc_dtype : str or numpy.dtype
        Output and accumulator dtypes.
    enumeration : bool
        Whether level 0 is enumerated.

    Returns
    -------
    int
        The estimate (bytes).
    """
    g2 = float(G) * G
    item, acc = np.dtype(dtype).itemsize, np.dtype(acc_dtype).itemsize
    parts = {}
    if out is None:
        parts["the (K, G, G) outputs in memory"] = K * g2 * (3 * item + 8 + 4 + 1)
    parts["the bootstrap accumulators"] = live * g2 * (8 + 3 * acc)
    if enumeration:
        parts["the enumerated counts of one tile"] = live * g2 * 8
    if keep_tiles:
        parts["the (T, K, G, G) tile tensor (keep_tiles)"] = T * K * g2 * 8
    total = sum(parts.values())
    if total > MEMORY_WARN_BYTES:
        detail = ", ".join(f"{name} {v / 2**30:.1f} GiB" for name, v in parts.items() if v >= 0.05 * total)
        advice = ["select fewer features (memory grows as G^2)"]
        if keep_tiles:
            advice.append("keep_tiles=False")
        if out is None:
            advice.append("out=DIR (memory-maps the outputs)")
        if enumeration:
            advice.append("lattice_only=True (one live slot instead of shells_per_level)")
        if np.dtype(acc_dtype) == np.float64:
            advice.append('acc_dtype="float32"')
        _warn(f"this run needs about {total / 2**30:.1f} GiB of host memory with G = {G} features and K = {K} "
              f"distance bins ({detail}); to reduce it: {'; '.join(advice)}")
    return int(total)


def _check_coordinates(coords: np.ndarray, name: str) -> None:
    """Raise ValueError if any coordinate is NaN or infinite.

    Parameters
    ----------
    coords : numpy.ndarray
        Coordinates.
    name : str
        Argument name for the message.
    """
    if not np.isfinite(coords).all():
        raise ValueError(f"{name} contains NaN or infinite coordinates")


def _origin(coords: np.ndarray, origin) -> np.ndarray:
    """The lattice and tile origin.

    Parameters
    ----------
    coords : numpy.ndarray
        (n, 2) coordinates.
    origin : array-like or None
        The origin given, or ``None`` for ``coords.min(0)``.

    Returns
    -------
    numpy.ndarray
        (2,) float64 origin.

    Raises
    ------
    ValueError
        If the origin does not have two finite entries or data lie below it.
    """
    if origin is None:
        return coords.min(0)
    origin = np.asarray(origin, dtype=np.float64)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError(f"origin must be two finite numbers, got {origin!r}")
    if (coords < origin).any():
        low = coords.min(0)
        raise ValueError(f"data lie below the origin {origin.tolist()} (minimum {low.tolist()}); the origin must "
                         "not exceed the smallest coordinate in either dimension")
    return origin


def _level_products(X, lat, n_shells, shells, *, backend, options, include_intra, row_groups=None, sink=None,
                    timing=None):
    """The shell products of one lattice level on the chosen backend.

    Parameters
    ----------
    X, lat, n_shells, shells, include_intra, row_groups, sink, timing
        As in ``scote._twopoint.cpu.pair_counts_level``.
    backend : {"cpu", "gpu"}
        The backend.
    options : CPUOptions or GPUOptions
        Its options.

    Returns
    -------
    numpy.ndarray or None
        As ``scote._twopoint.cpu.pair_counts_level``.
    """
    if backend == "gpu":
        from . import gpu as module
    else:
        from . import cpu as module
    return module.pair_counts_level(X, lat, n_shells, shells, include_intra=include_intra, row_groups=row_groups,
                                    sink=sink, timing=timing, options=options)


class _SlotTileMap:
    """Adapter from a level's ``sink(i, g, slab)`` (slot within the level, group index) to the global
    ``sink(base + i, tiles_of_groups[g], slab)``; forwards ``discard`` when the target has it.

    Parameters
    ----------
    target : callable
        The global sink.
    base : int
        Global slot of the level's first shell.
    tiles_of_groups : numpy.ndarray
        Tile index of every group index of the level.
    """

    def __init__(self, target, base: int, tiles_of_groups: np.ndarray):
        """See the class docstring."""
        self.target, self.base, self.tiles = target, int(base), np.asarray(tiles_of_groups)
        if hasattr(target, "discard"):
            self.discard = lambda i: target.discard(self.base + i)

    def __call__(self, i, g, slab):
        """Forward the slab of slot i of the level and group g to the target as (base + i, tile of g).

        Parameters
        ----------
        i : int
            Slot within the level.
        g : int
            Group index.
        slab : numpy.ndarray or torch.Tensor
            (G, G) counts.
        """
        self.target(self.base + i, int(self.tiles[g]), slab)


class _TileGrid:
    """Square tiles of side ``tile_um`` from ``origin``; the tiles that hold data, in (column, row) order.

    The key of a tile, ``ix * 2^32 + (iy + 2^31)``, is unique for every column ix and row iy (including rows beyond
    the data and below the origin) and orders the tiles by column, then by row.

    Parameters
    ----------
    coords : numpy.ndarray
        (n, 2) positions of the data (µm).
    tile_um : float
        Tile side (µm).
    origin : numpy.ndarray
        (2,) grid origin (µm).
    """

    _ROW_OFFSET = 1 << 31

    def __init__(self, coords: np.ndarray, tile_um: float, origin: np.ndarray):
        """See the class docstring."""
        self.tile_um, self.origin = float(tile_um), np.asarray(origin, dtype=np.float64)
        extent = (coords.max(0) - self.origin) / self.tile_um
        if not (np.abs(extent) < 2**30).all():
            raise ValueError(f"the data span more than 2^30 tiles of {tile_um} µm")
        self.tiles = np.unique(self.key(coords))

    def key(self, c_um: np.ndarray) -> np.ndarray:
        """Integer key of the tile of every position.

        Parameters
        ----------
        c_um : numpy.ndarray
            (n, 2) positions (µm).

        Returns
        -------
        numpy.ndarray
            (n,) int64 keys.
        """
        ix = np.floor((c_um[:, 0] - self.origin[0]) / self.tile_um).astype(np.int64)
        iy = np.floor((c_um[:, 1] - self.origin[1]) / self.tile_um).astype(np.int64)
        return (ix << 32) + (iy + self._ROW_OFFSET)

    @property
    def n_tiles(self) -> int:
        """Number of tiles T (tiles that hold data)."""
        return len(self.tiles)

    def index(self, c_um: np.ndarray) -> np.ndarray:
        """Tile index 0 .. T - 1 of every position.

        Parameters
        ----------
        c_um : numpy.ndarray
            (n, 2) positions (µm).

        Returns
        -------
        numpy.ndarray
            (n,) tile index: the tile of the position if it holds data, else the next tile in (column, row) order,
            or the last tile.
        """
        return np.clip(np.searchsorted(self.tiles, self.key(c_um)), 0, self.n_tiles - 1)

    def table(self) -> pd.DataFrame:
        """The tiles as a table.

        Returns
        -------
        pandas.DataFrame
            One row per tile index: ``column``, ``row`` (tile indices from the origin), ``x_from``, ``y_from`` (lower
            corner, µm).
        """
        column = self.tiles >> 32
        row = (self.tiles & ((1 << 32) - 1)) - self._ROW_OFFSET
        return pd.DataFrame(dict(column=column, row=row, x_from=self.origin[0] + column * self.tile_um,
                                 y_from=self.origin[1] + row * self.tile_um))


class _ShellWriter:
    """Sink of the per-tile slabs.

    Feeds the slabs into a :class:`scote.uncertainty.TileMomentAccumulator` and, as soon as a shell slot is complete
    (a later slot arrives, or ``close``), writes log g, sd, n_eff_pairs, P, t_eff, and the flag of that slot into
    (K, G, G) arrays (memory-mapped ``.npy`` files in ``out_dir``, or host memory). Optionally keeps the tile tensor.
    Only the accumulators of the live slot exist at any time.

    Parameters
    ----------
    n_slots : int
        Number of shell slots K.
    n_features : int
        Number of features G.
    out_dir : str or pathlib.Path or None
        Directory for memory-mapped outputs, or ``None`` for host memory.
    backend : CPUOptions or GPUOptions
        Where the accumulators run (:class:`scote.uncertainty.TileMomentAccumulator`).
    acc_dtype : numpy.dtype
        Accumulator dtype.
    out_dtype : numpy.dtype
        Dtype of log_g, sd, and n_eff_pairs.
    keep_tiles : bool
        Also store the (T, K, G, G) tile tensor.
    """

    def __init__(self, n_slots: int, n_features: int, out_dir=None, backend: CPUOptions | GPUOptions = CPUOptions(),
                 acc_dtype="float64", out_dtype="float64", keep_tiles: bool = False):
        """See the class docstring."""
        self.K, self.G = int(n_slots), int(n_features)
        self.out_dir = None if out_dir is None else Path(out_dir)
        self.backend = backend
        self.device = backend.device if isinstance(backend, GPUOptions) else None
        self.acc_dtype, self.out_dtype = np.dtype(acc_dtype), np.dtype(out_dtype)
        self.keep_tiles = keep_tiles
        self.acc = None
        self.next = 0
        self.P_tot = np.zeros(self.K)
        self.t_update = self.t_finish = self.t_write = 0.0
        self.n_updates = 0
        self.counts_tiles = None
        self._cuda = str(self.device).startswith("cuda")
        if self._cuda:
            import torch
            self._torch = torch
            torch.cuda.reset_peak_memory_stats(self.device)

    def _sync(self) -> None:
        """Synchronize the CUDA device (no-op otherwise)."""
        if self._cuda:
            self._torch.cuda.synchronize(self.device)

    def start(self, N: np.ndarray) -> None:
        """Set up the accumulator and allocate the outputs.

        Parameters
        ----------
        N : numpy.ndarray
            (T, G) per-tile feature totals.

        Raises
        ------
        ValueError
            If N does not have G columns.
        """
        if N.shape[1] != self.G:
            raise ValueError(f"totals have {N.shape[1]} features, expected {self.G}")
        self.N = np.asarray(N, dtype=np.float64)
        self.acc = TileMomentAccumulator(self.N, self.K, backend=self.backend, dtype=self.acc_dtype, keep_P=False,
                                         return_var_P=False)
        shape = (self.K, self.G, self.G)
        spec = dict(log_g=self.out_dtype, sd=self.out_dtype, n_eff_pairs=self.out_dtype, P=np.float64,
                    t_eff=np.float32, flag=np.int8)
        if self.out_dir is not None:
            from numpy.lib.format import open_memmap
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.arrays = {k: open_memmap(self.out_dir / f"{k}.npy", mode="w+", dtype=dt, shape=shape)
                           for k, dt in spec.items()}
        else:
            self.arrays = {k: np.empty(shape, dtype=dt) for k, dt in spec.items()}
        if self.keep_tiles:
            self.counts_tiles = np.zeros((self.N.shape[0], self.K, self.G, self.G), dtype=np.float64)

    def __call__(self, slot: int, t: int, slab) -> None:
        """Add the slab of one tile to a shell slot, after finishing all earlier slots.

        Parameters
        ----------
        slot : int
            Shell slot; slots must not decrease.
        t : int
            Tile index.
        slab : numpy.ndarray or torch.Tensor
            (G, G) float64 counts.

        Raises
        ------
        ValueError
            If the slot was already finished.
        """
        if slot < self.next:
            raise ValueError(f"slab for slot {slot} after the slot was written (next = {self.next})")
        while self.next < slot:
            self._finalize(self.next)
            self.next += 1
        self.add(slot, t, slab)

    def add(self, slot: int, t: int, slab) -> None:
        """Add the slab of one tile to a shell slot without finishing the earlier slots.

        The slots of the enumerated level arrive tile by tile (all slots of tile 0, then of tile 1, ...), so they are
        live at the same time; ``__call__`` finishes them when the first later slot arrives.

        Parameters
        ----------
        slot : int
            Shell slot, at least the next slot to be written.
        t : int
            Tile index.
        slab : numpy.ndarray or torch.Tensor
            (G, G) float64 counts.

        Raises
        ------
        ValueError
            If the slot was already finished.
        """
        if slot < self.next:
            raise ValueError(f"slab for slot {slot} after the slot was written (next = {self.next})")
        self._sync()
        ta = time.perf_counter()
        self.acc.update(slot, t, slab)
        self._sync()
        self.t_update += time.perf_counter() - ta
        self.n_updates += 1
        if self.counts_tiles is not None:
            self.counts_tiles[t, slot] = slab if isinstance(slab, np.ndarray) else slab.detach().cpu().numpy()

    def discard(self, slot: int) -> None:
        """Drop the partial accumulators (and stored slabs) of a slot before the producer restarts it.

        Parameters
        ----------
        slot : int
            Shell slot.
        """
        self.acc.discard(slot)
        if self.counts_tiles is not None:
            self.counts_tiles[:, slot] = 0.0

    def _finalize(self, k: int) -> None:
        """Turn the accumulators of one slot into the output arrays.

        Parameters
        ----------
        k : int
            Shell slot.
        """
        self._sync()
        ta = time.perf_counter()
        res = self.acc.finish(k, out_dtype=self.out_dtype)
        tb = time.perf_counter()
        a = self.arrays
        sd = res["sd"]
        sd[res["P"] == 0] = np.nan
        a["log_g"][k] = res["log_g"]
        a["sd"][k] = sd
        a["n_eff_pairs"][k] = res["n_eff_pairs"]
        a["P"][k] = res["P"]
        a["t_eff"][k] = res["t_eff"]
        a["flag"][k] = res["flag"]
        self.P_tot[k] = res["P_tot"]
        self.t_finish += tb - ta
        self.t_write += time.perf_counter() - tb

    def close(self) -> dict:
        """Finalize the remaining slots and flush the memory maps.

        Returns
        -------
        dict
            The (K, G, G) arrays by name.
        """
        while self.next < self.K:
            self._finalize(self.next)
            self.next += 1
        tb = time.perf_counter()
        if self.out_dir is not None:
            for v in self.arrays.values():
                v.flush()
        self.t_write += time.perf_counter() - tb
        return self.arrays

    def peak_device_mb(self) -> float | None:
        """Peak allocated CUDA memory (MB) since construction, or ``None`` off CUDA."""
        return self._torch.cuda.max_memory_allocated(self.device) / 2**20 if self._cuda else None


def _bin_row(level: int, shell: int, r_from: float, r_to: float, r_central: float) -> dict:
    """One row of the bins table.

    Parameters
    ----------
    level : int
        Level index.
    shell : int
        Shell index within the level.
    r_from, r_to : float
        Bin edges (µm).
    r_central : float
        Central distance of the bin (µm).

    Returns
    -------
    dict
        The row with the keys of ``BIN_COLUMNS``.
    """
    return dict(level=int(level), shell=int(shell), r_from=float(r_from), r_to=float(r_to), r_central=float(r_central))


def _lattice_levels_per_tile(coords, X0, pitch0, grid: _TileGrid, shells_per_level, first_level, n_levels, *,
                             all_shells_first, include_intra, backend, options, sink, slot_base, verbose,
                             on_totals=None):
    """Lattice levels ``first_level`` .. ``first_level + n_levels - 1`` with per-tile slabs streamed into a sink.

    Parameters
    ----------
    coords : numpy.ndarray
        (n, 2) positions (µm).
    X0 : scipy.sparse.csr_matrix
        (n, G) counts per row.
    pitch0 : float
        Pitch of the first lattice (µm).
    grid : _TileGrid
        The tiles; ``grid.origin`` is also the lattice origin.
    shells_per_level : int
        Shells per level.
    first_level : int
        Level index of the first lattice.
    n_levels : int
        Number of lattices.
    all_shells_first : bool
        The first lattice keeps all shells (with the pairs within one cell when ``include_intra``); otherwise it
        keeps shells s/2 .. s - 1 like every coarser level.
    include_intra : bool
        Count the pairs within one cell of the first lattice.
    backend : {"cpu", "gpu"}
        The backend.
    options : CPUOptions or GPUOptions
        Its options.
    sink : callable
        Global sink ``sink(slot, tile, slab)``.
    slot_base : int
        Slot of the first shell.
    verbose : bool
        Print the time of every level.
    on_totals : callable or None
        Receives the (T, G) feature totals per tile of the first lattice (cells attributed by their centers) before
        the first slab.

    Returns
    -------
    rows : list of dict
        Bin rows (``BIN_COLUMNS``).
    timings : list of dict
        One ``LevelTiming`` record per level.
    """
    origin = grid.origin
    cells0 = np.floor((coords - origin) / pitch0).astype(np.int64)
    lat, X = build_lattice(cells0, X0, margin=shells_per_level)
    if on_totals is not None:
        pos0 = grid.index((lat.cells + 0.5) * pitch0 + origin)
        tile_of_cell = sp.csr_matrix((np.ones(lat.n_cells), (pos0, np.arange(lat.n_cells))),
                                     shape=(grid.n_tiles, lat.n_cells))
        totals = np.zeros((grid.n_tiles, X.shape[1]))
        totals[:] = (tile_of_cell @ X).toarray()
        on_totals(totals)
        del tile_of_cell
    central = shell_central_distances(shells_per_level)
    pitch = float(pitch0)
    base = slot_base
    rows, timings = [], []
    for i in range(n_levels):
        level = first_level + i
        full = all_shells_first and i == 0
        keep = list(range(shells_per_level)) if full else list(range(shells_per_level // 2, shells_per_level))
        pos = grid.index((lat.cells + 0.5) * pitch + origin)
        tm = LevelTiming(level=level, pitch=pitch, n_cells=lat.n_cells, nnz_X=int(X.nnz),
                         density_X=X.nnz / max(1, X.shape[0] * X.shape[1]), mode="")
        t0 = time.perf_counter()
        _level_products(X, lat, shells_per_level, keep, backend=backend, options=options,
                        include_intra=(include_intra and full), row_groups=pos,
                        sink=_SlotTileMap(sink, base, np.unique(pos)), timing=tm)
        tm.seconds_total = time.perf_counter() - t0
        base += len(keep)
        for k in keep:
            rows.append(_bin_row(level, k, k * pitch, (k + 1) * pitch, central[k] * pitch))
        timings.append(tm.as_dict())
        if verbose:
            print(f"  level {level}: pitch={pitch:g} cells={lat.n_cells} mode={tm.mode} {tm.seconds_total:.1f}s",
                  flush=True)
        if i < n_levels - 1:
            lat, X = coarse_grain(lat, X, 2, margin=shells_per_level)
            pitch *= 2
    return rows, timings


def _finish_result(writer: _ShellWriter, rows, features, params, timings, out, grid: _TileGrid) -> TwoPointResult:
    """The TwoPointResult of a closed writer.

    Parameters
    ----------
    writer : _ShellWriter
        The closed writer.
    rows : list of dict
        Bin rows.
    features : list of str or None
        Feature names (``None``: "0", "1", ...).
    params : dict
        Run parameters.
    timings : dict
        Timings (updated in place with the accumulator and write times).
    out : str or pathlib.Path or None
        Directory of a streamed result.
    grid : _TileGrid
        The tiles.

    Returns
    -------
    TwoPointResult
        The result.
    """
    a = writer.arrays
    G = writer.G
    names = [str(f) for f in features] if features is not None else [str(i) for i in range(G)]
    bins = pd.DataFrame(rows, columns=BIN_COLUMNS)
    timings.update(accumulate=writer.t_update + writer.t_finish, write=writer.t_write, n_updates=writer.n_updates,
                   peak_device_memory_MB=writer.peak_device_mb())
    return TwoPointResult(log_g=a["log_g"], sd=a["sd"], n_eff_pairs=a["n_eff_pairs"], P=a["P"], t_eff=a["t_eff"],
                          flag=a["flag"], P_tot=writer.P_tot, totals=writer.N.sum(0), totals_tiles=writer.N,
                          bins=bins, features=names, counts_tiles=writer.counts_tiles, tiles=grid.table(),
                          params=params, timings=timings, path=None if out is None else Path(out))


def _save_streamed(res: TwoPointResult, out) -> None:
    """Write the small files of a streamed result (the arrays are already memory-mapped in ``out``).

    Parameters
    ----------
    res : TwoPointResult
        The result.
    out : str or pathlib.Path
        Its directory.
    """
    tw = time.perf_counter()
    res.save(out)
    res.timings["write"] += time.perf_counter() - tw
    res.write_meta(out)


# ----------------------------------------------------------------------------------------------- input normalization
_FEATURES_TYPE_MSG = ("features must be one label per point (a pandas Categorical or Series, an array of strings, or "
                      "integer codes) or an (N, G) count matrix (scipy sparse, a 2-D numpy array, or a DataFrame)")
_ANNDATA_MSG = ("{arg} is an AnnData object; pass its parts: scote.twopoint(xy, adata.X, names=adata.var_names), with "
                "xy the (N, 2) bin or spot centers in micrometers (e.g. adata.obsm['spatial'] times the micrometers "
                "per pixel)")
_ONE_COLUMN_MSG = ("features has one column: one label per point must be 1-D (df['gene'], not df[['gene']]; "
                   "codes.ravel() for a column of codes); a single-feature count matrix needs names=[...]")


def _warn(message: str, category=UserWarning) -> None:
    """Warn with the stack level of the first caller outside the package.

    Parameters
    ----------
    message : str
        The message.
    category : type
        The warning class.
    """
    level, frame = 2, sys._getframe(1)
    while frame is not None and frame.f_globals.get("__name__", "").split(".")[0] == "scote":
        frame, level = frame.f_back, level + 1
    warnings.warn(message, category, stacklevel=level)


def _is_anndata(obj) -> bool:
    """Whether ``obj`` looks like an AnnData object (without importing anndata).

    Parameters
    ----------
    obj : object
        Any object.

    Returns
    -------
    bool
        True if its class is named ``AnnData`` or it has the attributes ``X``, ``obs``, ``var``, and ``var_names``.
    """
    return type(obj).__name__ == "AnnData" or all(hasattr(obj, a) for a in ("X", "obs", "var", "var_names"))


def _is_count_matrix(features) -> bool:
    """Whether ``features`` is an (N, G) count matrix rather than one label per point.

    Parameters
    ----------
    features : object
        The ``features`` argument of :func:`scote.twopoint`.

    Returns
    -------
    bool
        True for a SciPy sparse matrix or array, a pandas DataFrame, or an array with two dimensions.
    """
    if sp.issparse(features) or isinstance(features, pd.DataFrame):
        return True
    if isinstance(features, (pd.Series, pd.Categorical, pd.Index)):
        return False
    return getattr(features, "ndim", 1) == 2


def _unique_names(names: list[str], what: str) -> list[str]:
    """Check that feature names are unique.

    Parameters
    ----------
    names : list of str
        The names.
    what : str
        Where the names come from, for the message.

    Returns
    -------
    list of str
        The names.

    Raises
    ------
    ValueError
        If a name occurs more than once.
    """
    if len(set(names)) != len(names):
        seen, dup = set(), []
        for f in names:
            if f in seen and f not in dup:
                dup.append(f)
            seen.add(f)
        shown = ", ".join(repr(f) for f in dup[:5]) + (", ..." if len(dup) > 5 else "")
        raise ValueError(f"the feature names ({what}) are not unique: {shown} occur more than once; make them unique "
                         "(e.g. pandas.Index.drop_duplicates, or AnnData's var_names_make_unique) or merge the "
                         "features first")
    return names


def _labels(features, names, n: int) -> tuple[np.ndarray, list[str]]:
    """Integer codes and feature names of one label per point.

    Parameters
    ----------
    features : pandas.Categorical, pandas.Series, or array-like
        One label per point (see :func:`scote.twopoint`).
    names : sequence of str or None
        Names of integer codes; only with integer labels.
    n : int
        Number of points.

    Returns
    -------
    codes : numpy.ndarray
        (n,) int64 codes into ``names``.
    names : list of str
        Feature names: the categories of a Categorical that occur (in their order), the given names of integer
        codes (all of them), otherwise the sorted unique labels.

    Raises
    ------
    ValueError
        For missing labels, codes outside ``0 .. len(names) - 1``, a length other than ``n``, or names that are not
        unique.
    TypeError
        For float labels, or ``names`` with labels that carry their own names.
    """
    if isinstance(features, pd.Series) and isinstance(features.dtype, pd.CategoricalDtype):
        features = features.array
    if isinstance(features, pd.Categorical):
        if names is not None:
            raise TypeError("names apply to integer codes and count matrices; a Categorical carries its categories "
                            "(use select to choose or order a subset)")
        codes = np.asarray(features.codes).astype(np.int64)
        names = [str(c) for c in features.categories]
        used = np.zeros(len(names) + 1, dtype=bool)
        used[codes + 1] = True                            # index 0: code -1 (missing)
        if not used[1:].all():                            # categories without points are dropped, in their order
            new_code = np.full(len(names) + 1, -1, dtype=np.int64)
            new_code[1:][used[1:]] = np.arange(int(used[1:].sum()))
            codes = new_code[codes + 1]
            names = [f for f, u in zip(names, used[1:]) if u]
        what = "categories"
    else:
        values = np.asarray(features)
        if values.ndim != 1:
            raise TypeError(_FEATURES_TYPE_MSG)
        if values.dtype.kind in "iu" and names is not None:
            codes = values
            names = [str(f) for f in names]
            if len(codes) and (codes.min() < 0 or codes.max() >= len(names)):
                raise ValueError(f"integer codes must lie in 0 .. {len(names) - 1} (one per name)")
            what = "names"
        elif names is not None:
            raise TypeError("names apply to integer codes and count matrices; string labels are their own names")
        elif values.dtype.kind in "iubUSO":
            codes, uniques = pd.factorize(values, sort=True, use_na_sentinel=True)
            names = [str(u) for u in uniques]
            what = "labels as strings"
        else:
            raise TypeError(f"{_FEATURES_TYPE_MSG}; got dtype {values.dtype}")
    if len(codes) != n:
        raise ValueError(f"features has {len(codes)} labels for {n} points")
    if len(codes) and codes.min() < 0:
        raise ValueError("features has missing labels (NaN or None); drop those points or give them a label")
    return codes.astype(np.int64, copy=False), _unique_names(names, what)


def _numeric_columns(df: pd.DataFrame) -> bool:
    """Whether every column of a DataFrame holds numbers (dense or sparse, not bool-like strings or objects).

    Parameters
    ----------
    df : pandas.DataFrame
        The frame.

    Returns
    -------
    bool
        True if all column dtypes (or the subtypes of sparse columns) are boolean, integer, or float.
    """
    for t in df.dtypes:
        t = t.subtype if isinstance(t, pd.SparseDtype) else t
        if not (pd.api.types.is_numeric_dtype(t) or pd.api.types.is_bool_dtype(t)) or isinstance(
                t, pd.CategoricalDtype):
            return False
    return True


def _counts(features, names, n: int) -> tuple[sp.csr_matrix, list[str]]:
    """CSR counts and feature names of an (N, G) count matrix.

    Parameters
    ----------
    features : scipy.sparse matrix, numpy.ndarray, or pandas.DataFrame
        The counts.
    names : sequence of str or None
        Column names (taken from the columns of a DataFrame).
    n : int
        Number of points.

    Returns
    -------
    X : scipy.sparse.csr_matrix
        (n, G) counts.
    names : list of str
        Feature names; ``"0"``, ``"1"``, ... without names.

    Raises
    ------
    ValueError
        For a row count other than ``n``, a number of names other than G, names that are not unique, one column
        without names, a one-column DataFrame, or negative or non-finite counts.
    TypeError
        For ``names`` with a DataFrame, or a matrix that does not hold numbers (strings, objects, categories).
    """
    if isinstance(features, pd.DataFrame):
        if names is not None:
            raise TypeError("names of a DataFrame are its columns")
        if features.shape[1] == 1:
            raise ValueError("features is a DataFrame with one column: one label per point must be 1-D (df['gene'], "
                             "not df[['gene']]); a single-feature count matrix is a 2-D array with names=[...]")
        if not _numeric_columns(features):
            raise TypeError("features is a DataFrame with non-numeric columns; a count matrix holds numbers (one "
                            "column per feature). For one label per point pass a 1-D Series, e.g. df['gene']")
        names = [str(c) for c in features.columns]
        sparse = all(isinstance(t, pd.SparseDtype) for t in features.dtypes)
        X = sp.csr_matrix(features.sparse.to_coo()) if sparse and len(features.columns) else sp.csr_matrix(
            features.to_numpy())
    else:
        dtype = features.dtype if sp.issparse(features) else np.asarray(features).dtype
        if dtype.kind not in "biuf":
            raise TypeError(f"features is a 2-D array of dtype {dtype}; a count matrix holds numbers. For one label "
                            "per point pass a 1-D array or Series (df['gene'], not df[['gene']])")
        if features.shape[1:] == (1,) and names is None:
            raise ValueError(_ONE_COLUMN_MSG)
        X = sp.csr_matrix(features)
    if X.shape[0] != n:
        raise ValueError(f"the count matrix has {X.shape[0]} rows for {n} points")
    G = X.shape[1]
    names = [str(i) for i in range(G)] if names is None else [str(f) for f in names]
    if len(names) != G:
        raise ValueError(f"{len(names)} feature names for {G} columns")
    _check_count_values(X.data)
    return X, _unique_names(names, "columns")


def _check_count_values(data: np.ndarray) -> None:
    """Raise on negative or non-finite counts; warn on non-integer counts.

    Parameters
    ----------
    data : numpy.ndarray
        The stored values of the count matrix.

    Raises
    ------
    ValueError
        For NaN, infinite, or negative values.
    """
    if data.dtype.kind == "f":
        if not np.isfinite(data).all():
            raise ValueError("the count matrix holds NaN or infinite values")
        frac = data != np.rint(data)
        if frac.any():
            _warn(f"{frac.mean():.3g} of the stored counts are not integers: they are used as weights, while the "
                  "uncertainties and the reference assume counts of molecules (pass raw counts, not normalized or "
                  "log-transformed values)")
    if data.dtype.kind in "if" and len(data) and data.min() < 0:
        raise ValueError("the count matrix holds negative values; counts must be >= 0 (pass raw counts, not "
                         "centered or log-transformed values)")


def _selection(select, names: list[str], totals: np.ndarray) -> list[str]:
    """The names chosen by ``select``.

    Parameters
    ----------
    select : str or sequence of str
        A selection string of :func:`scote.io.select_features` (``"top:N"``, ``"min:N"``, ``"file:PATH"``, or a
        comma-separated list) or a sequence of names, in the order wanted.
    names : list of str
        All feature names.
    totals : numpy.ndarray
        (G,) totals of the features (molecules or counts), for ``"top:N"`` and ``"min:N"``.

    Returns
    -------
    list of str
        The chosen names.

    Raises
    ------
    KeyError
        For unknown names.
    ValueError
        For repeated names.
    """
    from ..io import select_features
    if isinstance(select, str):
        ranked = pd.Series(np.asarray(totals, dtype=np.float64), index=names).sort_values(ascending=False,
                                                                                          kind="stable")
        chosen = [str(f) for f in select_features(select, ranked)]
    else:
        chosen = [str(f) for f in select]
        unknown = [f for f in chosen if f not in set(names)]
        if unknown:
            raise KeyError(f"unknown feature(s) {unknown}; the data have {len(names)} features (the categories of a "
                           "Categorical count only if a point carries them)")
    if len(set(chosen)) != len(chosen):
        raise ValueError("select names a feature more than once")
    return chosen


def _infer_bin_size(xy: np.ndarray, n_per_window: int = 25_000, n_windows: int = 4, seed: int = 0) -> float:
    """The bin size of a bin table from the bin centers: the 10th percentile of the nearest-neighbor distances.

    The distances are measured in ``n_windows`` square windows around randomly chosen bins, each holding about
    ``n_per_window`` bins on average (all bins when there are few). A low percentile rather than the median keeps
    the value when bins without counts were left out of the table. It is rounded to three significant digits, so
    that the slight scatter and rotation of bin centers registered to an image do not change it (2.0 µm for Visium
    HD 2 µm bins, whose neighbors are 1.998 to 2.000 µm apart in the image frame). For a square bin array it is the
    bin size; for the Visium spot array it is the spot spacing (99.9 µm).

    Parameters
    ----------
    xy : numpy.ndarray
        (N, 2) bin centers (µm).
    n_per_window : int
        Mean number of bins per window.
    n_windows : int
        Number of windows.
    seed : int
        Seed of the window centers.

    Returns
    -------
    float
        The bin size (µm).

    Raises
    ------
    ValueError
        If fewer than two distinct positions are found.
    """
    from scipy.spatial import cKDTree
    n = len(xy)
    if n <= 4 * n_per_window:
        groups = [xy]
    else:
        span = np.maximum(xy.max(0) - xy.min(0), np.finfo(np.float64).tiny)
        side = np.sqrt(n_per_window * span.prod() / n)
        rng = np.random.default_rng(seed)
        groups = [xy[(np.abs(xy - c) <= side / 2).all(1)] for c in xy[rng.choice(n, n_windows, replace=False)]]
    distances = []
    for pts in groups:
        if len(pts) >= 2:
            d = cKDTree(pts).query(pts, k=2)[0][:, 1]
            distances.append(d[d > 0])
    distances = np.concatenate(distances) if distances else np.zeros(0)
    if not len(distances):
        raise ValueError("cannot infer the bin size: fewer than two distinct bin positions; pass bin_size")
    return float(f"{float(np.percentile(distances, 10)):.3g}")


# ----------------------------------------------------------------------------------------------- implementation
def _twopoint_molecules(xy, codes, names, params, *, delta, shells_per_level, n_levels, tile_um, origin, backend,
                        options, keep_tiles, out, dtype, acc_dtype, verbose) -> TwoPointResult:
    """Exact enumeration below ``shells_per_level * delta``, lattice levels above (validated inputs).

    Parameters
    ----------
    xy : numpy.ndarray
        (n, 2) float64 positions (µm).
    codes : numpy.ndarray
        (n,) int64 codes into ``names``.
    names : list of str
        Feature names.
    params : dict
        Parameters of the run, completed and stored in the result.
    delta, shells_per_level, n_levels, tile_um, origin, backend, keep_tiles, out, dtype, acc_dtype, verbose
        As in :func:`scote.twopoint`; ``origin`` as an array.
    options : CPUOptions or GPUOptions
        Options of the backend.

    Returns
    -------
    TwoPointResult
        The result.
    """
    t_start = time.perf_counter()
    G = len(names)
    s = int(shells_per_level)
    grid = _TileGrid(xy, tile_um, origin)
    T = grid.n_tiles
    K = s + (s // 2) * n_levels
    _check_memory(T=T, K=K, G=G, live=s, out=out, keep_tiles=keep_tiles, dtype=dtype, acc_dtype=acc_dtype,
                  enumeration=True)
    writer = _ShellWriter(K, G, out, backend=options, acc_dtype=acc_dtype, out_dtype=dtype,
                          keep_tiles=keep_tiles)
    # level 0: enumeration per tile (molecules of a tile in their input order), one block size for all tiles; the s
    # slabs of every tile stream into the accumulators of the s live slots
    t0 = time.perf_counter()
    tile_of_point = grid.index(xy)
    order = np.argsort(tile_of_point, kind="stable")
    bounds = np.searchsorted(tile_of_point[order], np.arange(T + 1))
    block_size = enumeration_block_size(xy, delta, s)
    totals = np.zeros((T, G), dtype=np.float64)
    for t in range(T):
        totals[t] = np.bincount(codes[order[bounds[t]:bounds[t + 1]]], minlength=G)
    writer.start(totals)
    update_before = writer.t_update
    for t, counts in pair_counts_enumerated_groups(xy[order], codes[order], bounds, G, delta, s,
                                                   block_size=block_size):
        counts = counts.astype(np.float64)
        for k in range(s):
            writer.add(k, t, counts[k])
    t1 = time.perf_counter()
    enumeration_update = writer.t_update - update_before
    if verbose:
        print(f"  level 0 (enumeration): {T} tiles {t1 - t0:.1f}s", flush=True)
    iv = bin_intervals(delta, s)
    rows = [_bin_row(0, k, a, b, c) for k, (a, b, c) in enumerate(zip(iv["from"], iv["to"], iv["central_distance"]))]
    level_timings = []
    if n_levels > 0:
        lattice_options = options if backend == "cpu" else dataclasses.replace(options, device_slabs=True)
        rows_lattice, level_timings = _lattice_levels_per_tile(
            xy, onehot_csr(codes, G), 2 * float(delta), grid, s, 1, n_levels, all_shells_first=False,
            include_intra=False, backend=backend, options=lattice_options, sink=writer, slot_base=s, verbose=verbose)
        rows += rows_lattice
    writer.close()
    t2 = time.perf_counter()
    params.update(origin=[float(o) for o in grid.origin], threads=_numba_threads())
    timings = dict(enumeration=(t1 - t0) - enumeration_update, lattice=None, levels=level_timings)
    res = _finish_result(writer, rows, names, params, timings, out, grid)
    timings["lattice"] = (t2 - t1) - (timings["accumulate"] - enumeration_update) - writer.t_write
    timings["total"] = time.perf_counter() - t_start
    if out is not None:
        _save_streamed(res, out)
    if verbose:
        print(f"enumeration {timings['enumeration']:.1f}s | lattice {timings['lattice']:.1f}s | accumulate "
              f"{timings['accumulate']:.1f}s | tiles {T}", flush=True)
    return res


def _twopoint_lattice(xy, X, names, params, *, delta, shells_per_level, n_levels, tile_um, origin, backend, options,
                      keep_tiles, out, dtype, acc_dtype, verbose) -> TwoPointResult:
    """All levels on lattices: level 0 of pitch ``delta`` with all shells, then coarse-grained levels (validated
    inputs).

    Parameters
    ----------
    xy : numpy.ndarray
        (n, 2) float64 positions of the bins or molecules (µm).
    X : scipy.sparse.csr_matrix
        (n, G) counts per row.
    names : list of str
        Feature names.
    params : dict
        Parameters of the run, completed and stored in the result.
    delta, shells_per_level, n_levels, tile_um, origin, backend, keep_tiles, out, dtype, acc_dtype, verbose
        As in :func:`scote.twopoint`; ``origin`` as an array.
    options : CPUOptions or GPUOptions
        Options of the backend.

    Returns
    -------
    TwoPointResult
        The result.
    """
    t_start = time.perf_counter()
    G = X.shape[1]
    s = int(shells_per_level)
    grid = _TileGrid(xy, tile_um, origin)
    K = s + (s // 2) * n_levels
    _check_memory(T=grid.n_tiles, K=K, G=G, live=1, out=out, keep_tiles=keep_tiles, dtype=dtype, acc_dtype=acc_dtype,
                  enumeration=False)
    writer = _ShellWriter(K, G, out, backend=options, acc_dtype=acc_dtype, out_dtype=dtype,
                          keep_tiles=keep_tiles)
    lattice_options = options if backend == "cpu" else dataclasses.replace(options, device_slabs=True)
    t0 = time.perf_counter()
    rows, level_timings = _lattice_levels_per_tile(
        xy, X, delta, grid, s, 0, n_levels + 1, all_shells_first=True, include_intra=True, backend=backend,
        options=lattice_options, sink=writer, slot_base=0, verbose=verbose, on_totals=writer.start)
    writer.close()
    t1 = time.perf_counter()
    params.update(origin=[float(o) for o in grid.origin], threads=_numba_threads())
    timings = dict(levels=level_timings)
    res = _finish_result(writer, rows, names, params, timings, out, grid)
    timings["lattice"] = (t1 - t0) - timings["accumulate"] - writer.t_write
    timings["total"] = time.perf_counter() - t_start
    if out is not None:
        _save_streamed(res, out)
    if verbose:
        print(f"lattice {timings['lattice']:.1f}s | accumulate {timings['accumulate']:.1f}s | tiles {grid.n_tiles}",
              flush=True)
    return res


# Default range of the hierarchy: the bins of n_levels=None reach this distance (µm), 16.4 mm.
DEFAULT_RANGE_UM = 2.0 ** 14


def _default_n_levels(delta: float, shells_per_level: int) -> int:
    """The default number of levels: the smallest whose bins reach ``DEFAULT_RANGE_UM``.

    Parameters
    ----------
    delta : float
        Finest distance resolution (µm).
    shells_per_level : int
        Distance bins per level.

    Returns
    -------
    int
        The smallest ``n >= 0`` with ``shells_per_level * delta * 2**n >= 2**14`` µm (12 for 0.5 µm, 10 for 2 µm, 5
        for 64 µm with 8 shells per level).
    """
    n = 0
    while shells_per_level * delta * 2.0 ** n < DEFAULT_RANGE_UM * (1 - 1e-12):
        n += 1
    return n


def _prepare(xy, features, *, names, select, delta, bin_size, lattice_only, n_levels, shells_per_level):
    """Validate and normalize the data arguments shared by :func:`scote.twopoint` and :func:`scote.pair_counts`.

    Parameters
    ----------
    xy, features, names, select, delta, bin_size, lattice_only, n_levels, shells_per_level
        As in :func:`scote.twopoint`.

    Returns
    -------
    dict
        ``xy`` (float64), ``codes`` (labels) or ``X`` (CSR counts; labels on lattices give one-hot rows),
        ``names``, ``lattice`` (bool), ``delta``, ``bin_size`` (count matrices, else ``None``), ``n_levels``, and
        ``params`` (the input description).
    """
    for arg, value in (("xy", xy), ("features", features)):
        if _is_anndata(value):
            raise TypeError(_ANNDATA_MSG.format(arg=arg))
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must have shape (N, 2), got {xy.shape}")
    if len(xy) == 0:
        raise ValueError("no points")
    _check_coordinates(xy, "xy")
    n = len(xy)
    counts = _is_count_matrix(features)
    if counts:
        X, names = _counts(features, names, n)
        if lattice_only is False:
            raise ValueError("a count matrix is always counted on lattices; lattice_only=False needs one label per "
                             "point")
        totals = np.asarray(X.sum(0)).ravel()
    else:
        codes, names = _labels(features, names, n)
        totals = np.bincount(codes, minlength=len(names))
    if select is not None:
        chosen = _selection(select, names, totals)
        index = pd.Index(names).get_indexer(chosen)
        if counts:
            X = X[:, index].tocsr()
        else:
            new_code = np.full(len(names), -1, dtype=np.int64)
            new_code[index] = np.arange(len(chosen))
            codes = new_code[codes]
            keep = codes >= 0
            if not keep.all():
                xy, codes = xy[keep], codes[keep]
            if len(codes) == 0:
                raise ValueError("no points with a selected feature")
        names = chosen
    if len(names) == 1:
        _warn("a single feature: g = 1 at every distance by construction, because n and P are taken over the "
              "selected features; select at least two features")
    params = dict(input="counts" if counts else "labels", n_points=int(len(xy)))
    if counts:
        if bin_size is None:
            bin_size = _infer_bin_size(xy)
            params["bin_size_source"] = "inferred from the coordinates (nearest-neighbor distances)"
        else:
            params["bin_size_source"] = "given"
        _validate(bin_size=bin_size, delta=delta)
        bin_size = float(bin_size)
        if delta is None:
            delta = bin_size
        elif delta < bin_size * (1 - 1e-9):
            raise ValueError(f"delta = {delta} µm is smaller than the bin size {bin_size} µm: the bins cannot resolve "
                             "smaller distances (pass bin_size if the bins are smaller than their spacing)")
        params["bin_size"] = bin_size
        lattice = True
    else:
        if bin_size is not None:
            raise ValueError("bin_size applies to count matrices")
        lattice = bool(lattice_only)
        delta = 0.5 if delta is None else delta
    _validate(delta=delta)
    if n_levels is None:
        n_levels = _default_n_levels(float(delta), int(shells_per_level))
    out = dict(xy=xy, names=names, lattice=lattice, delta=float(delta), bin_size=bin_size if counts else None,
               n_levels=n_levels, params=params)
    if counts:
        out["X"] = X
    elif lattice:
        out["X"] = onehot_csr(codes, len(names))
    else:
        out["codes"] = codes
    return out


def _check_backend(backend: str) -> None:
    """Import PyTorch for the GPU backend, raising early if it is missing.

    Parameters
    ----------
    backend : {"cpu", "gpu"}
        The backend.
    """
    if backend == "gpu":
        from .gpu import _torch
        _torch()


# ----------------------------------------------------------------------------------------------- public functions
@with_threads
def twopoint(xy, features, *, names=None, select=None, delta: float | None = None, bin_size: float | None = None,
             lattice_only: bool | None = None, shells_per_level: int = 8, n_levels: int | None = None,
             tile_um: float = 500.0, origin=None, backend: str | CPUOptions | GPUOptions = "cpu",
             threads: int | None = None, keep_tiles: bool = False, out=None, dtype="float64", acc_dtype="float64",
             verbose: bool = False) -> TwoPointResult:
    r"""Two-point functions of all feature pairs, with closed-form tile-bootstrap uncertainties.

    One function for molecule tables (one label per point) and bin tables (a count matrix); it dispatches on the
    type of ``features``. In both cases ``xy`` are positions in micrometers and ``delta`` (:math:`\Delta`) is the finest
    distance resolution. The result holds :math:`\log g_{ab}(r)` of all ordered feature pairs over a hierarchy of
    distance bins :math:`k`: :math:`g_{ab}(r_k) = P_{ab}(k) / (P_\mathrm{tot}(k)\, n_a n_b / n_\mathrm{tot}^2)`, the
    pair count of the features ``a`` and ``b`` over its value for labels independent of the positions
    (:func:`scote.random_label_reference`). The reference is that of the features analyzed: :math:`n_a` and
    :math:`n_\mathrm{tot}`, :math:`P_{ab}` and :math:`P_\mathrm{tot}` are taken over the selected features only
    (``select``), so results of different selections differ by the change of reference.

    *Molecules* (one label per point): level 0 enumerates all pairs closer than :math:`s\,\Delta`
    (:math:`s` = ``shells_per_level``) exactly, in :math:`s` bins of width :math:`\Delta`, one tile at a time. Pairs
    that cross a tile border are not counted at this level, so the pair count drops by about
    :math:`4 s \Delta /` ``tile_um``; averaged over the placement of the tile grid, the loss does not depend on the
    labels and leaves :math:`g` unchanged. Levels
    :math:`L = 1 \dots` ``n_levels`` count the shells :math:`s/2 \dots s - 1` on lattices of pitch
    :math:`2^L \Delta`, so the bins continue from :math:`s\,\Delta` up to :math:`s\,\Delta\,2^{n_\mathrm{levels}}`.
    With ``lattice_only=True`` all levels are lattices, as for bins below.

    *Bins* (a count matrix, or molecules with ``lattice_only=True``): the counts are placed on a lattice of pitch
    ``delta`` (cell index :math:`\lfloor (x - x_0) / \Delta \rfloor` per axis, :math:`x_0` = ``origin``). Level 0
    counts all shells :math:`0 \dots s - 1` of that lattice (shell 0: the pairs within one cell, without
    self-pairs); levels :math:`1 \dots` ``n_levels`` coarse-grain by 2 and keep the shells :math:`s/2 \dots s - 1`.
    ``delta`` defaults to the bin size; a larger ``delta`` coarse-grains the bins (useful where molecules diffuse
    farther than the bin size), a smaller one raises ``ValueError``.

    Tiles and attribution: the square tiles of side ``tile_um`` start at ``origin``; a pair belongs to the tile of its
    first molecule at the enumerated level and to the tile of the center of its first cell at a lattice level. At
    levels whose pitch reaches about the tile side, a cell center can fall into a tile without data or beyond the
    data; the pair then goes to the next tile with data in (column, row) order, or to the last tile. This changes
    only the tile scatter (``sd``, ``n_eff_pairs``, ``t_eff``, ``flag``) of the coarsest bins, not ``P`` or
    ``log_g``. ``counts_tiles`` holds the counts attributed this way; the tile scatter behind ``sd``,
    ``n_eff_pairs``, and ``t_eff`` uses the symmetrized slabs :math:`(P_t + P_t^\top) / 2`, that is, half of every
    pair in the tile of each member (:class:`scote.uncertainty.TileMomentAccumulator`).

    Memory: the per-tile counts stream into the accumulators of the tile bootstrap shell by shell, so the
    (tiles, bins, G, G) tensor exists only with ``keep_tiles``. What stays in memory grows as :math:`G^2`: the
    (K, G, G) outputs (memory-mapped with ``out``) and four (G, G) accumulators per live shell (all
    ``shells_per_level`` shells of the enumerated level are live together). A warning names the parts before they
    are allocated if they exceed 4 GiB.

    Parameters
    ----------
    xy : array-like
        (N, 2) positions of the molecules or bin centers (µm), finite.
    features : pandas.Categorical, pandas.Series, array-like, scipy.sparse matrix, or pandas.DataFrame
        Either one label per point (molecule tables): a Categorical or Series, an array of strings, or integer
        codes with ``names``; or an (N, G) count matrix (bin or spot tables): a SciPy sparse matrix, a 2-D NumPy
        array, or a DataFrame whose columns are the features. Order of the features in the result: the categories of
        a Categorical, the ``names`` of integer codes or of a matrix, the columns of a DataFrame, otherwise the
        sorted unique labels (integers without ``names`` are labels, too). The categories of a Categorical that no
        point carries are dropped, keeping the order of the others (a filtered DataFrame keeps all categories of its
        panel; dropping them keeps the memory, which grows as :math:`G^2`, to the features present). Feature names
        must be unique; a one-column 2-D input without ``names`` (such as ``df[["gene"]]``), a non-numeric matrix,
        and an AnnData object raise with a hint. Counts must be non-negative; non-integer counts give a warning.
    names : sequence of str or None
        Names of integer codes (code ``i`` is ``names[i]``) or of the columns of a count matrix; they are given in
        the call, so all of them are kept, also features without molecules (G = ``len(names)``, a fixed feature axis
        across calls). ``None``: ``"0"``, ``"1"``, ... for a matrix.
    select : str, sequence of str, or None
        Features to keep, in this order: a sequence of names, or a selection string of
        :func:`scote.io.select_features` (``"top:N"``, ``"min:N"``, ``"file:PATH"``, or a comma-separated list),
        ranked by the molecule or count totals. Molecules of other features are dropped. The selection defines the
        reference: :math:`n_\mathrm{tot}` and :math:`P_\mathrm{tot}` are taken over the selected features only.
        ``None`` keeps all.
    delta : float or None
        Finest distance resolution (µm): the bin width of the enumerated level, or the lattice pitch of level 0.
        ``None``: 0.5 for molecules, the bin size for a count matrix. For a count matrix, a larger ``delta``
        coarse-grains.
    bin_size : float or None
        Count matrices only: the size of the bins (µm), the default and the lower limit of ``delta``. ``None`` infers it
        from ``xy`` as the 10th percentile of the nearest-neighbor distances of the bin centers, rounded to three
        significant digits (the bin size of a square bin array). Pass it for bins smaller than their spacing, e.g. 55
        for the Visium spots, whose centers are 100 µm apart.
    lattice_only : bool or None
        Molecules: count also the shortest distances on a lattice of pitch ``delta`` instead of enumerating them
        (``None`` and ``False``: enumerate). Count matrices are always counted on lattices.
    shells_per_level : int
        Distance bins per level, even and at least 2; the coarser levels keep the upper half.
    n_levels : int or None
        Number of levels after level 0, at least 0. ``None``: the smallest number whose bins reach 16.4 mm
        (:math:`2^{14}` µm), so that the range does not depend on ``lattice_only``: with 8 shells per level 12 for
        ``delta`` = 0.5 µm, 10 for 2 µm, and 5 for 64 µm.
    tile_um : float
        Side of the square tiles of the block bootstrap (µm), positive.
    origin : array-like or None
        (2,) origin of the tiles and lattices (µm), at most the smallest coordinate in either dimension. ``None``:
        the smallest coordinates of ``xy``. A common origin gives registered sections the same tile grid.
    backend : {"cpu", "gpu"}, CPUOptions, or GPUOptions
        Where the lattice products and the bootstrap accumulators run, with the default options: ``"cpu"`` (numba
        kernels, BLAS, and SciPy) or ``"gpu"`` (PyTorch on a CUDA device). A :class:`scote.CPUOptions` or
        :class:`scote.GPUOptions` selects its backend with these options, e.g. ``scote.GPUOptions(device="cuda:1")``,
        or ``scote.GPUOptions(device="cpu")`` for the PyTorch CPU device. The enumerated level always runs on the CPU.
    threads : int or None
        CPU threads of this call for numba, BLAS, and the PyTorch CPU device (restored afterwards); ``None``:
        :func:`scote.default_threads`.
    keep_tiles : bool
        Also return the (T, K, G, G) per-tile counts (``counts_tiles``) for statistics across pairs and shells
        (:mod:`scote.uncertainty`).
    out : str, pathlib.Path, or None
        Directory to stream the (K, G, G) arrays into (memory-mapped ``.npy`` files, see
        :meth:`scote.TwoPointResult.save`); ``None`` keeps them in memory. It removes the outputs from the host
        memory, not the accumulators of the live shells.
    dtype : str or numpy.dtype
        Dtype of ``log_g``, ``sd``, and ``n_eff_pairs`` (``"float64"`` or ``"float32"``; ``P`` is always float64).
    acc_dtype : str or numpy.dtype
        Dtype of the bootstrap accumulators (``"float32"`` halves their memory).
    verbose : bool
        Print the time of every level.

    Returns
    -------
    TwoPointResult
        ``log_g``, ``sd``, reliability flags, counts, and the distance bins of all
        :math:`K = s + n_\mathrm{levels}\, s / 2` bins; ``params`` records the input type, ``delta``, the bin size and
        where it came from, and the other settings.

    Raises
    ------
    ValueError
        For inconsistent or invalid inputs (duplicate feature names, negative or non-finite counts, a one-column
        2-D input without ``names``), a ``delta`` below the bin size, data below the origin, or an unknown backend.
    TypeError
        For an unsupported type of ``features`` (a non-numeric matrix, an AnnData object), ``names`` with labels
        that carry their own names, or a ``backend`` that is neither a name nor an options object.
    KeyError
        For unknown names in ``select``.
    ImportError
        For ``backend="gpu"`` without PyTorch.

    See Also
    --------
    scote.pair_counts : The global pair counts of the lattice hierarchy, without tiles and uncertainties.
    scote.io : Readers that return the data of a platform in the form this function takes.

    Examples
    --------
    A molecule table with columns ``x``, ``y`` (µm), and ``gene``:

    >>> res = scote.twopoint(df[["x", "y"]].to_numpy(), df["gene"], select="top:50")   # doctest: +SKIP

    A Visium HD bin table (bin size 2 µm, inferred) and the same bins coarse-grained to 8 µm:

    >>> xy, X, names, bin_size = scote.io.read_visium_hd("outs/")                          # doctest: +SKIP
    >>> res = scote.twopoint(xy, X, names=names)                                          # doctest: +SKIP
    >>> res8 = scote.twopoint(xy, X, names=names, delta=8.0)                               # doctest: +SKIP
    """
    backend, options = resolve_backend(backend)
    _validate(shells_per_level=shells_per_level, n_levels=n_levels, tile_um=tile_um)
    data = _prepare(xy, features, names=names, select=select, delta=delta, bin_size=bin_size,
                    lattice_only=lattice_only, n_levels=n_levels, shells_per_level=shells_per_level)
    _validate(shells_per_level=shells_per_level, n_levels=data["n_levels"])
    _check_backend(backend)
    xy = data["xy"]
    origin = _origin(xy, origin)
    params = dict(data["params"], lattice_only=data["lattice"], delta=data["delta"],
                  shells_per_level=int(shells_per_level), n_levels=int(data["n_levels"]), tile_um=float(tile_um),
                  backend=backend, options=dataclasses.asdict(options), dtype=str(np.dtype(dtype)),
                  acc_dtype=str(np.dtype(acc_dtype)))
    run = dict(delta=data["delta"], shells_per_level=shells_per_level, n_levels=data["n_levels"], tile_um=tile_um,
               origin=origin, backend=backend, options=options, keep_tiles=keep_tiles, out=out, dtype=dtype,
               acc_dtype=acc_dtype, verbose=verbose)
    if data["lattice"]:
        return _twopoint_lattice(xy, data["X"], data["names"], params, **run)
    return _twopoint_molecules(xy, data["codes"], data["names"], params, **run)


@with_threads
def pair_counts(xy, features, *, names=None, select=None, delta: float | None = None, bin_size: float | None = None,
                shells_per_level: int = 8, n_levels: int = 10, include_intra: bool = True, origin=None,
                backend: str | CPUOptions | GPUOptions = "cpu", threads: int | None = None,
                sink: Callable[[int, np.ndarray], None] | None = None, verbose: bool = False) -> PairCounts:
    r"""Global ordered pair counts of the lattice hierarchy, without tiles or uncertainties.

    The data arguments are those of :func:`scote.twopoint` on lattices: the points (molecules with one label each,
    or bins with a count matrix) are placed on a lattice of pitch ``delta``; level 0 counts all its shells, and
    every further level coarse-grains by 2 and keeps the shells :math:`s/2 \dots s - 1`
    (:math:`s` = ``shells_per_level``), whose distances continue the previous level's range.

    Parameters
    ----------
    xy, features, names, select, delta, bin_size
        As in :func:`scote.twopoint` (molecules are always counted on lattices here).
    shells_per_level : int
        Distance bins (shells) per level, even and at least 2.
    n_levels : int
        Number of coarse-grained levels after level 0, at least 0.
    include_intra : bool
        Count shell 0 of level 0 as the pairs of distinct rows (molecules or bins) within one cell.
    origin : array-like or None
        (2,) origin of the lattice (µm), at most the smallest coordinate in either dimension; ``None`` uses the
        smallest coordinates of ``xy``.
    backend : {"cpu", "gpu"}, CPUOptions, or GPUOptions
        Where the products run, as in :func:`scote.twopoint`.
    threads : int or None
        CPU threads of this call (numba, BLAS, PyTorch CPU device); ``None``: :func:`scote.default_threads`.
    sink : callable or None
        ``sink(k, slab)`` receives the (G, G) float64 host array of every bin ``k`` in increasing order (a device
        tensor that is valid only during the call with ``backend=scote.GPUOptions(device_slabs=True)``); the
        ``counts`` of the result are then the (K,) per-bin totals.
    verbose : bool
        Print the time of every level.

    Returns
    -------
    PairCounts
        ``counts`` (K, G, G) float64 ordered pair counts (or (K,) totals with ``sink``), ``bins`` (``level``,
        ``shell``, ``r_from``, ``r_to``, ``r_central`` per bin), ``features``, ``params``, and ``timings``.

    Raises
    ------
    ValueError
        For invalid inputs, a ``delta`` below the bin size, data below the origin, or an unknown backend.
    TypeError
        For an unsupported type of ``features``, or a ``backend`` that is neither a name nor an options object.
    ImportError
        For ``backend="gpu"`` without PyTorch.

    See Also
    --------
    scote.twopoint : The two-point functions with their uncertainties.
    """
    backend, options = resolve_backend(backend)
    _validate(shells_per_level=shells_per_level, n_levels=n_levels)
    data = _prepare(xy, features, names=names, select=select, delta=delta, bin_size=bin_size, lattice_only=True,
                    n_levels=n_levels, shells_per_level=shells_per_level)
    _check_backend(backend)
    xy, X, pitch = data["xy"], data["X"], data["delta"]
    origin = _origin(xy, origin)
    t_start = time.perf_counter()
    s = int(shells_per_level)
    cells0 = np.floor((xy - origin) / pitch).astype(np.int64)
    lat, X = build_lattice(cells0, X, margin=s)
    counts, rows, level_timings = [], [], []
    p = float(pitch)
    central = shell_central_distances(s, cells0.shape[1])
    base = 0
    for level in range(n_levels + 1):
        keep = list(range(s)) if level == 0 else list(range(s // 2, s))
        tm = LevelTiming(level=level, pitch=p, n_cells=lat.n_cells, nnz_X=int(X.nnz),
                         density_X=X.nnz / max(1, X.shape[0] * X.shape[1]), mode="")
        t0 = time.perf_counter()
        level_sink = None if sink is None else (lambda i, slab, b0=base: sink(b0 + i, slab))
        P = _level_products(X, lat, s, keep, backend=backend, options=options,
                            include_intra=include_intra and level == 0, sink=level_sink, timing=tm)
        base += len(keep)
        tm.seconds_total = time.perf_counter() - t0
        if verbose:
            print(f"  level {level}: pitch={p:g} cells={lat.n_cells} nnz={X.nnz} mode={tm.mode} "
                  f"{tm.seconds_total:.2f}s", flush=True)
        counts.append(P)
        for k in keep:
            rows.append(_bin_row(level, k, k * p, (k + 1) * p, central[k] * p))
        level_timings.append(tm.as_dict())
        if level < n_levels:
            lat, X = coarse_grain(lat, X, 2, margin=s)
            p *= 2
    params = dict(data["params"], delta=float(pitch), shells_per_level=s, n_levels=int(n_levels),
                  include_intra=bool(include_intra), origin=[float(o) for o in origin], backend=backend,
                  options=dataclasses.asdict(options), threads=_numba_threads())
    return PairCounts(counts=np.concatenate(counts, axis=0), bins=pd.DataFrame(rows, columns=BIN_COLUMNS),
                      features=data["names"], params=params,
                      timings=dict(levels=level_timings, total=time.perf_counter() - t_start))


def _numba_threads() -> int:
    """numba's current thread count.

    Returns
    -------
    int
        ``numba.get_num_threads()``.
    """
    import numba
    return int(numba.get_num_threads())
