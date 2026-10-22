"""The tables of the common data model, and the search for vendor files."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["MOLECULE_SCHEMA_DOC", "BinTable", "find_file", "make_unique", "molecule_frame", "write_bin_table",
           "write_molecules"]


class BinTable(NamedTuple):
    """A bin or spot table in memory, in the form :func:`scote.twopoint` takes.

    Unpacks as ``xy, counts, names, bin_size = table``; pass it as
    ``scote.twopoint(xy, counts, names=names, bin_size=bin_size)``.

    Attributes
    ----------
    xy : numpy.ndarray
        (N, 2) float64 bin centers (µm).
    counts : scipy.sparse.csr_matrix
        (N, G) counts per bin.
    names : list of str
        Feature names, one per column of ``counts``.
    bin_size : float
        Size of the bins (µm): the side of a square bin, or the diameter of a spot.
    """
    xy: np.ndarray
    counts: sp.csr_matrix
    names: list
    bin_size: float

MOLECULE_SCHEMA_DOC = ("x,y,[z] µm float32; feature dictionary<string>; sample string; cell_id int64 (-1 unassigned); "
                       "nucleus bool; qv float32")


def make_unique(names) -> list[str]:
    """Feature names made unique, the rule of AnnData's ``var_names_make_unique``.

    The first occurrence of a name is kept; later ones get ``-1``, ``-2``, ... appended (the next number whose result
    is not already a name). Vendor matrices can list one gene symbol for several gene ids; :func:`scote.twopoint`
    needs unique names. The function follows AnnData's ``make_index_unique`` (https://github.com/scverse/anndata,
    BSD 3-Clause License; the notice is in ``NOTICES.md`` at the root of this repository).

    Parameters
    ----------
    names : sequence of str
        The names.

    Returns
    -------
    list of str
        The unique names, in the same order.
    """
    names = [str(n) for n in names]
    if len(set(names)) == len(names):
        return names
    taken = set(names)
    seen: set[str] = set()
    counter: dict[str, int] = {}
    out = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
            continue
        while True:
            counter[name] = counter.get(name, 0) + 1
            candidate = f"{name}-{counter[name]}"
            if candidate not in taken:
                taken.add(candidate)
                out.append(candidate)
                break
    return out


def find_file(rawdir, pattern: str) -> Path:
    """The first file under a directory that matches a pattern.

    Parameters
    ----------
    rawdir : str or pathlib.Path
        Directory searched recursively.
    pattern : str
        Glob pattern of the file name (``Path.rglob``).

    Returns
    -------
    pathlib.Path
        The first match of ``Path.rglob``.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist or holds no match.
    """
    rawdir = Path(rawdir)
    if not rawdir.is_dir():
        raise FileNotFoundError(f"directory {rawdir} does not exist")
    match = next(rawdir.rglob(pattern), None)
    if match is None:
        raise FileNotFoundError(f"no {pattern} under {rawdir}")
    return match


def write_molecules(tables: Iterable, dest) -> int:
    """Write molecule tables of the common data model to one parquet file, one or more row groups per table.

    Parameters
    ----------
    tables : iterable of pyarrow.Table
        Tables with the same schema.
    dest : str or pathlib.Path
        Output file (zstd-compressed parquet).

    Returns
    -------
    int
        Rows written.
    """
    import pyarrow.parquet as pq
    writer, n = None, 0
    try:
        for tbl in tables:
            if writer is None:
                writer = pq.ParquetWriter(dest, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            n += tbl.num_rows
    finally:
        if writer is not None:
            writer.close()
    return n


def molecule_frame(tables: Iterable, columns=None) -> pd.DataFrame:
    """Molecule tables of the common data model as one DataFrame in memory.

    Parameters
    ----------
    tables : iterable of pyarrow.Table
        Tables with the same schema.
    columns : sequence of str or None
        Columns to keep (``x``, ``y``, and ``feature`` are always kept); ``None`` keeps all.

    Returns
    -------
    pandas.DataFrame
        One row per molecule; ``feature`` is a Categorical with the feature names in alphabetical order, other
        dictionary columns are Categoricals as well.
    """
    import pyarrow as pa
    keep = None if columns is None else list(dict.fromkeys(["x", "y", "feature", *columns]))
    parts = [t if keep is None else t.select(keep) for t in tables]
    if not parts:
        return pd.DataFrame({"x": np.zeros(0, np.float32), "y": np.zeros(0, np.float32),
                             "feature": pd.Categorical([])})
    df = pa.concat_tables(parts).to_pandas()
    feature = df["feature"].astype("category")
    df["feature"] = feature.cat.reorder_categories(sorted(feature.cat.categories, key=str))
    return df


def write_bin_table(dest, xy: np.ndarray, counts, names, bin_size: float, in_tissue=None) -> None:
    """Write a bin table of the common data model (``.npz``).

    Parameters
    ----------
    dest : str or pathlib.Path
        Output file.
    xy : numpy.ndarray
        (N, 2) bin centers (µm), stored as float32.
    counts : scipy.sparse matrix
        (N, G) counts, stored as float32 CSR arrays.
    names : sequence of str
        Feature names.
    bin_size : float
        Bin size (µm).
    in_tissue : numpy.ndarray or None
        (N,) bool; ``None`` marks all bins as in the tissue.
    """
    X = sp.csr_matrix(counts)
    np.savez(dest, data=X.data.astype(np.float32), indices=X.indices, indptr=X.indptr, shape=np.array(X.shape),
             coords_um=np.asarray(xy, dtype=np.float32), features=np.array(list(names), dtype=str),
             bin_um=float(bin_size),
             in_tissue=np.ones(X.shape[0], bool) if in_tissue is None else np.asarray(in_tissue, dtype=bool))
