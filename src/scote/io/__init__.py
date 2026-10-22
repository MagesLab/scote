"""Reading vendor outputs into memory, and converting them to the common data model.

For every supported platform, ``read_<platform>`` returns the data in memory in the form :func:`scote.twopoint`
takes, and ``convert_<platform>`` writes the same data in the common data model (with a QC summary); both share the
same code. Molecule platforms give a DataFrame with ``x``, ``y`` (µm), and ``feature`` (a Categorical) and further
columns; bin and spot platforms give a :class:`scote.io.BinTable` (``xy``, ``counts``, ``names``, ``bin_size``):

============================  ================================  ===================================  =========
platform                      reader                            converter                            output
============================  ================================  ===================================  =========
Xenium                        :func:`~scote.io.read_xenium`     :func:`~scote.io.convert_xenium`     molecules
MERSCOPE                      :func:`~scote.io.read_merscope`   :func:`~scote.io.convert_merscope`   molecules
CosMx (AtoMx flat files)      :func:`~scote.io.read_cosmx`      :func:`~scote.io.convert_cosmx`      molecules
Atera (zarr-only 10x bundle)  :func:`~scote.io.read_atera`      :func:`~scote.io.convert_atera`      molecules
AVITI24 / Cytos               :func:`~scote.io.read_aviti24`    :func:`~scote.io.convert_aviti24`    molecules
Visium HD                     :func:`~scote.io.read_visium_hd`  :func:`~scote.io.convert_visium_hd`  bins
Visium                        :func:`~scote.io.read_visium`     :func:`~scote.io.convert_visium`     spots
Stereo-seq (GEM, GEF)         :func:`~scote.io.read_stereoseq`  :func:`~scote.io.convert_stereoseq`  bins
============================  ================================  ===================================  =========

Common data model. Molecule table: a parquet file ``molecules.parquet`` with one row per molecule and the columns
``x``, ``y`` (µm, float32), optionally ``z`` (µm), ``feature`` (dictionary-encoded string), ``sample``, ``cell_id``
(int64, -1 = not in a cell), ``nucleus`` (bool), and platform-specific extras (``qv``, ``z_plane``, ...), written in
row groups, so that sections of :math:`10^8` molecules never have to be in memory at once; read it with
:func:`scote.io.read_molecules`. Bin table: an ``.npz`` file with the CSR counts (``data``, ``indices``, ``indptr``,
``shape``; bins x features), ``coords_um`` (bin centers, µm), ``features`` (names), ``bin_um`` (bin size), and
optionally ``in_tissue`` (bool per bin); read it with :func:`scote.io.read_bins`.

Feature selection by name or abundance: :func:`scote.io.select_features` (also the ``select`` argument of
:func:`scote.twopoint`). Missing inputs raise ``FileNotFoundError`` with the path searched; unknown feature names
raise ``KeyError``. The Visium readers need the ``io`` extra (scanpy), the Atera reader ``zarr`` and ``numcodecs``,
the Stereo-seq GEF reader ``h5py`` (all in the ``io`` extra).

Defaults shared by the platforms: control probes and codewords are dropped (Xenium, Atera, MERSCOPE, CosMx); every
column of the common data model is kept (``columns=None``); vendor gene names that occur more than once are made
unique with the rule of AnnData's ``var_names_make_unique``, because :func:`scote.twopoint` needs unique names.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._schema import MOLECULE_SCHEMA_DOC, BinTable
from ._atera import convert_atera, read_atera
from ._aviti24 import aviti24_run_parameters, aviti24_wells, convert_aviti24, read_aviti24
from ._cosmx import convert_cosmx, read_cosmx
from ._merscope import convert_merscope, read_merscope
from ._stereoseq import convert_stereoseq, read_stereoseq
from ._visium import convert_visium, convert_visium_hd, read_visium, read_visium_hd, visium_um_per_pixel
from ._xenium import cell_id_to_int, convert_xenium, read_xenium

__all__ = ["MOLECULE_SCHEMA_DOC", "BinTable", "aviti24_run_parameters", "aviti24_wells", "cell_id_to_int",
           "convert_atera", "convert_aviti24", "convert_cosmx", "convert_merscope", "convert_stereoseq",
           "convert_visium", "convert_visium_hd", "convert_xenium", "read_atera", "read_aviti24", "read_bins",
           "read_cosmx", "read_merscope", "read_molecules", "read_stereoseq", "read_visium", "read_visium_hd",
           "read_xenium", "select_features", "visium_um_per_pixel"]


def select_features(spec: str, counts: pd.Series) -> list[str]:
    """Select features by a specification string.

    Parameters
    ----------
    spec : str
        ``"top:N"``: the N most abundant features; ``"min:N"``: all features with at least N counts;
        ``"file:PATH"``: the names listed in a text file (one per line, ``#`` starts a comment line), in file order,
        restricted to the features present (a panel file can list features of other platforms); anything else: a
        comma-separated list of names (spaces around the commas are ignored), all of which must be present.
    counts : pandas.Series
        Feature name -> total count, sorted in descending order.

    Returns
    -------
    list of str
        The selected feature names.

    Raises
    ------
    KeyError
        If a name of a comma-separated list is not a feature of the table.
    FileNotFoundError
        If the file of ``"file:PATH"`` does not exist.
    """
    if spec.startswith("top:"):
        return list(counts.index[: int(spec[4:])])
    if spec.startswith("min:"):
        return list(counts.index[counts >= int(spec[4:])])
    if spec.startswith("file:"):
        with open(spec[5:]) as fh:
            want = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
        have = set(counts.index)
        return [w for w in want if w in have]
    names = [name.strip() for name in spec.split(",") if name.strip()]
    unknown = [n for n in names if n not in set(counts.index)]
    if unknown:
        raise KeyError(f"unknown feature(s) {unknown}; the table has {len(counts)} features")
    return names


def _existing(path) -> Path:
    """A path that must exist.

    Parameters
    ----------
    path : str or pathlib.Path
        The path.

    Returns
    -------
    pathlib.Path
        The path.

    Raises
    ------
    FileNotFoundError
        If it does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    return path


def read_molecules(path, select: str | None = None, max_molecules: int = 0, seed: int = 0,
                   columns=None) -> pd.DataFrame:
    """Read a molecule table of the common data model into memory, in the form :func:`scote.twopoint` takes.

    Parameters
    ----------
    path : str or pathlib.Path
        ``molecules.parquet`` of the common data model.
    select : str or None
        Feature selection (:func:`scote.io.select_features`), applied to the feature counts of the whole table;
        molecules of other features are dropped. ``None`` keeps all features.
    max_molecules : int
        If positive and smaller than the number of selected molecules, a random subsample of this size
        (``numpy.random.default_rng(seed).choice(n, max_molecules, replace=False)``).
    seed : int
        Seed of the subsample.
    columns : sequence of str or None
        Columns to read (``x``, ``y``, and ``feature`` are always read); ``None`` reads all, as the platform readers
        do. ``columns=[]`` reads only ``x``, ``y``, and ``feature`` (the least memory).

    Returns
    -------
    pandas.DataFrame
        One row per molecule; ``feature`` is a Categorical whose categories are the selected features in the order
        of the selection (all features in alphabetical order without one).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    KeyError
        If a selected feature name is not in the table.

    Examples
    --------
    >>> df = scote.io.read_molecules("molecules.parquet", select="top:50")    # doctest: +SKIP
    >>> res = scote.twopoint(df[["x", "y"]].to_numpy(), df["feature"])        # doctest: +SKIP
    """
    import pyarrow.parquet as pq
    if columns is not None:
        columns = list(dict.fromkeys(["x", "y", "feature", *columns]))
    tbl = pq.read_table(_existing(path), columns=columns)
    df = tbl.to_pandas()
    feat = df["feature"].astype("category")
    if select is None:
        names = sorted(str(c) for c in feat.cat.categories)
    else:
        names = [str(n) for n in select_features(select, feat.value_counts())]
    code_of_category = pd.Index(names).get_indexer(feat.cat.categories.astype(str))   # -1: not selected
    codes = code_of_category[feat.cat.codes.to_numpy()]
    keep = codes >= 0
    df["feature"] = pd.Categorical.from_codes(np.where(keep, codes, 0), categories=names)
    if not keep.all():
        df = df[keep].reset_index(drop=True)
    if max_molecules and len(df) > max_molecules:
        idx = np.random.default_rng(seed).choice(len(df), max_molecules, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
    return df


def read_bins(path, select: str | None = None) -> BinTable:
    """Read a bin table of the common data model into memory, in the form :func:`scote.twopoint` takes.

    Parameters
    ----------
    path : str or pathlib.Path
        Bin table ``.npz`` of the common data model. Bins with ``in_tissue == False`` are dropped.
    select : str or None
        Feature selection (:func:`scote.io.select_features`), applied to the feature totals of the bins kept; ``None``
        keeps all features.

    Returns
    -------
    BinTable
        ``xy`` (bin centers, µm), ``counts`` (CSR, bins x selected features), ``names``, and ``bin_size`` (µm).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    KeyError
        If a selected feature name is not in the table.
    ValueError
        If the feature names are stored as an object array.

    Examples
    --------
    >>> xy, X, names, bin_size = scote.io.read_bins("bins_square_002um.npz", select="top:50")   # doctest: +SKIP
    >>> res = scote.twopoint(xy, X, names=names, bin_size=bin_size)                             # doctest: +SKIP
    """
    path = _existing(path)
    z = np.load(path)
    X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    try:
        feats = pd.Index(z["features"].astype(str))
    except ValueError as e:
        raise ValueError(f"{path}: the feature names are stored as an object array, which needs pickle to be read; "
                         "store them as a unicode array (numpy.array(names, dtype=str))") from e
    coords = z["coords_um"].astype(np.float64)
    if "in_tissue" in z:
        keep = z["in_tissue"].astype(bool)
        X = X[keep]
        coords = coords[keep]
    names = [str(n) for n in feats]
    if select is not None:
        tot = pd.Series(np.asarray(X.sum(0)).ravel(), index=feats)
        names = [str(n) for n in select_features(select, tot.sort_values(ascending=False))]
        X = X[:, feats.get_indexer(names)].tocsr()
    return BinTable(coords, X, names, float(z["bin_um"]))


def _document_here() -> None:
    """Name ``scote.io`` as the module of the objects exported here (each is documented once, under ``scote.io``)."""
    for name in __all__:
        obj = globals()[name]
        if callable(obj):
            obj.__module__ = __name__


_document_here()
