"""Stereo-seq (STOmics) gene expression matrices of the SAW pipeline -> bin tables of the common data model, on disk
or in memory.

Both formats are described in the public file-format manual of the SAW pipeline ("Stereo-seq Analysis Workflow File
Format", the STOmics download page and the ``Documents/FileFormat`` directory of the public SAW repository).

* GEM (``.gem`` or ``.gem.gz``): a tab-separated text table with one row per gene and spot (DNB) or bin: ``geneID``
  (and optionally ``geneName``), ``x``, ``y`` (integer DNB coordinates), ``MIDCount`` (also ``MIDCounts`` or
  ``UMICount`` in some versions), and optionally ``ExonCount`` and ``CellID``; lines starting with ``#`` are header
  comments.
* Square-bin GEF (``.gef``, HDF5; needs ``h5py``, part of the ``io`` extra): the group ``geneExp/bin1`` holds the
  dataset ``expression`` (compound ``x``, ``y`` in DNB coordinates and ``count``, rows grouped by gene; attribute
  ``resolution``, the DNB pitch in nm) and the dataset ``gene`` (compound ``gene`` (name), ``offset`` (first row in
  ``expression``), ``count`` (number of rows); newer versions name the fields ``geneID`` and ``geneName``). The
  coarser groups ``bin10`` ... ``bin500`` are not read: the manual does not state the unit of their coordinates, so
  the bins are formed from ``bin1`` here, exactly as for GEM. Cell-bin GEF files (``cellBin`` group) hold cells, not
  bins, and are not read.

The DNBs lie on a square grid of 500 nm pitch on the standard chip, so the coordinates times 0.5 are micrometers.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._schema import BinTable, make_unique, write_bin_table

__all__ = ["STEREOSEQ_DNB_PITCH_UM", "convert_stereoseq", "read_stereoseq"]

STEREOSEQ_DNB_PITCH_UM = 0.5        # center-to-center distance of the DNBs on the standard Stereo-seq chip
_COUNT_COLUMNS = ("MIDCount", "MIDCounts", "UMICount")


def _header_lines(path: Path) -> int:
    """Number of ``#`` comment lines at the top of a GEM file.

    Parameters
    ----------
    path : pathlib.Path
        The file (``.gz`` is decompressed).

    Returns
    -------
    int
        The number of leading lines that start with ``#``.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    n = 0
    with opener(path, "rt") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            n += 1
    return n


def _gef_gene_column(fields, gene_column: str | None) -> str:
    """The field of the GEF gene table that holds the feature names.

    Parameters
    ----------
    fields : sequence of str
        Field names of ``geneExp/bin1/gene``.
    gene_column : str or None
        The field asked for; ``None`` takes ``geneName``, else ``gene``, else ``geneID``.

    Returns
    -------
    str
        The field.

    Raises
    ------
    ValueError
        If the field is not in the table.
    """
    if gene_column is None:
        gene_column = next((f for f in ("geneName", "gene", "geneID") if f in fields), None)
    if gene_column not in fields:
        raise ValueError(f"no gene name field {gene_column!r} in the GEF gene table; fields {list(fields)}")
    return gene_column


class _BinAccumulator:
    """(bin, gene) sums of chunks of rows, reduced chunk by chunk.

    Parameters
    ----------
    bin_dnb : int
        Bin side in DNBs.
    """

    _Y_OFFSET = 1 << 31

    def __init__(self, bin_dnb: int):
        """See the class docstring."""
        self.bin_dnb = int(bin_dnb)
        self.keys, self.genes, self.values = [], [], []

    def add(self, x: np.ndarray, y: np.ndarray, gene: np.ndarray, value: np.ndarray, n_genes: int) -> None:
        """Add the rows of one chunk, reduced to one entry per (bin, gene).

        Parameters
        ----------
        x, y : numpy.ndarray
            Integer DNB coordinates.
        gene : numpy.ndarray
            Gene index of every row.
        value : numpy.ndarray
            Counts.
        n_genes : int
            Number of genes seen so far (larger than every index in ``gene``).
        """
        if not len(x):
            return
        key = ((np.asarray(x, np.int64) // self.bin_dnb) << 32) + (np.asarray(y, np.int64) // self.bin_dnb
                                                                   + self._Y_OFFSET)
        bins, row = np.unique(key, return_inverse=True)
        m = sp.coo_matrix((np.asarray(value, np.float64), (row.ravel(), np.asarray(gene, np.int64))),
                          shape=(len(bins), n_genes)).tocsr()
        m.sum_duplicates()
        m = m.tocoo()
        self.keys.append(bins[m.row])
        self.genes.append(m.col.astype(np.int64))
        self.values.append(m.data)

    def table(self, names: list[str], dnb_pitch_um: float) -> BinTable:
        """The bin table of all chunks.

        Parameters
        ----------
        names : list of str
            Gene names, one per gene index.
        dnb_pitch_um : float
            Distance of neighboring DNBs (µm).

        Returns
        -------
        BinTable
            Bins ordered by x index, then y index; positions at the centers of their DNBs.
        """
        keys = np.concatenate(self.keys) if self.keys else np.zeros(0, np.int64)
        bins, row = np.unique(keys, return_inverse=True)
        X = sp.csr_matrix((np.concatenate(self.values) if self.values else np.zeros(0),
                           (row.ravel(), np.concatenate(self.genes) if self.genes else np.zeros(0, np.int64))),
                          shape=(len(bins), len(names)))
        X.sum_duplicates()
        ij = np.stack([bins >> 32, (bins & ((1 << 32) - 1)) - self._Y_OFFSET], 1).astype(np.float64)
        xy = (ij * self.bin_dnb + (self.bin_dnb - 1) / 2.0) * float(dnb_pitch_um)
        return BinTable(xy, X, names, float(self.bin_dnb * dnb_pitch_um))


def _read_gem(path: Path, bin_dnb: int, gene_column: str | None, dnb_pitch_um: float, chunk_rows: int) -> BinTable:
    """The bins of a GEM file (see :func:`scote.io.read_stereoseq`).

    Parameters
    ----------
    path : pathlib.Path
        The file.
    bin_dnb, gene_column, dnb_pitch_um, chunk_rows
        As in :func:`scote.io.read_stereoseq`.

    Returns
    -------
    BinTable
        The bins.
    """
    skip = _header_lines(path)
    head = pd.read_csv(path, sep="\t", skiprows=skip, nrows=0).columns
    count_col = next((c for c in _COUNT_COLUMNS if c in head), None)
    if count_col is None:
        raise ValueError(f"no count column ({', '.join(_COUNT_COLUMNS)}) in {path}; columns {list(head)}")
    gene_col = gene_column or ("geneName" if "geneName" in head else "geneID")
    genes: dict[str, int] = {}
    acc = _BinAccumulator(bin_dnb)
    for ch in pd.read_csv(path, sep="\t", skiprows=skip, usecols=[gene_col, "x", "y", count_col],
                          dtype={gene_col: str}, chunksize=chunk_rows):
        g, uniq = pd.factorize(ch[gene_col], sort=False)
        gmap = np.array([genes.setdefault(u, len(genes)) for u in uniq], dtype=np.int64)
        acc.add(ch["x"].to_numpy(np.int64), ch["y"].to_numpy(np.int64), gmap[g], ch[count_col].to_numpy(np.float64),
                len(genes))
    return acc.table(list(genes), dnb_pitch_um)


def _read_gef(path: Path, bin_dnb: int, gene_column: str | None, dnb_pitch_um: float | None,
              chunk_rows: int) -> BinTable:
    """The bins of a square-bin GEF file (see :func:`scote.io.read_stereoseq`).

    Parameters
    ----------
    path : pathlib.Path
        The file.
    bin_dnb, gene_column, dnb_pitch_um, chunk_rows
        As in :func:`scote.io.read_stereoseq`; ``dnb_pitch_um=None`` takes the ``resolution`` attribute.

    Returns
    -------
    BinTable
        The bins.

    Raises
    ------
    ImportError
        Without ``h5py``.
    ValueError
        If the file has no ``geneExp/bin1`` group (e.g. a cell-bin GEF).
    """
    try:
        import h5py
    except ImportError as e:                                   # pragma: no cover - depends on the environment
        raise ImportError("reading GEF files needs h5py (pip install 'scote[io]')") from e
    with h5py.File(path, "r") as f:
        if "geneExp" not in f or "bin1" not in f["geneExp"]:
            raise ValueError(f"{path} has no geneExp/bin1 group (a square-bin GEF); cell-bin GEF files are not read")
        grp = f["geneExp"]["bin1"]
        expression, gene = grp["expression"], grp["gene"][:]
        if dnb_pitch_um is None:
            res = expression.attrs.get("resolution")
            dnb_pitch_um = float(np.ravel(res)[0]) / 1000.0 if res is not None else STEREOSEQ_DNB_PITCH_UM
        field = _gef_gene_column(gene.dtype.names, gene_column)
        names = make_unique([v.decode() if isinstance(v, bytes) else str(v) for v in gene[field]])
        offset = gene["offset"].astype(np.int64)
        order = np.argsort(offset, kind="stable")
        starts = offset[order]
        acc = _BinAccumulator(bin_dnb)
        n_rows = expression.shape[0]
        for a in range(0, n_rows, chunk_rows):
            b = min(n_rows, a + chunk_rows)
            rows = expression[a:b]
            g = order[np.searchsorted(starts, np.arange(a, b), side="right") - 1]
            acc.add(rows["x"], rows["y"], g, rows["count"], len(names))
    return acc.table(names, dnb_pitch_um)


def read_stereoseq(path, bin_dnb: int = 50, gene_column: str | None = None, dnb_pitch_um: float | None = None,
                   chunk_rows: int = 10_000_000) -> BinTable:
    """Read a Stereo-seq GEM or square-bin GEF file into memory as square bins, in the form :func:`scote.twopoint`
    takes.

    The DNBs are merged into square bins of ``bin_dnb`` x ``bin_dnb`` DNBs (bin index ``x // bin_dnb``,
    ``y // bin_dnb``); a bin's position is the center of its DNBs, ``(index * bin_dnb + (bin_dnb - 1) / 2)`` times
    the DNB pitch, and its size is ``bin_dnb`` times the pitch (bin 50: 25 µm). Bins without counts are not part of
    the table. The rows are read ``chunk_rows`` at a time and every chunk is reduced to its (bin, gene) sums, so the
    memory holds the bins, not the rows of the file.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``.gem`` or ``.gem.gz`` file, or the square-bin ``.gef`` file (``h5py`` needed).
    bin_dnb : int
        Bin side in DNBs (1 keeps the single DNBs).
    gene_column : str or None
        Column (GEM) or field (GEF) of the feature names: ``None`` takes ``geneName`` where present, else ``geneID``
        (GEM) or ``gene`` (GEF).
    dnb_pitch_um : float or None
        Distance of neighboring DNBs (µm); ``None``: the ``resolution`` attribute of a GEF file, 0.5 µm otherwise.
    chunk_rows : int
        Rows read at a time.

    Returns
    -------
    BinTable
        ``xy`` (bin centers, µm; bins ordered by x index, then y index), ``counts`` (bins x genes, CSR), ``names``
        (GEM: genes in order of first appearance, one feature per name; GEF: the gene table in its order, names made
        unique with the rule of AnnData's ``var_names_make_unique``), ``bin_size`` (µm).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If ``bin_dnb`` is not a positive integer, a GEM file has no count column, or a GEF file has no
        ``geneExp/bin1`` group.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    if not (isinstance(bin_dnb, (int, np.integer)) and bin_dnb >= 1):
        raise ValueError(f"bin_dnb must be a positive integer, got {bin_dnb!r}")
    if path.suffix.lower() == ".gef":
        return _read_gef(path, int(bin_dnb), gene_column, dnb_pitch_um, int(chunk_rows))
    return _read_gem(path, int(bin_dnb), gene_column,
                     STEREOSEQ_DNB_PITCH_UM if dnb_pitch_um is None else float(dnb_pitch_um), int(chunk_rows))


def convert_stereoseq(path, outdir, sample: str, bin_dnb: int = 50, gene_column: str | None = None,
                      dnb_pitch_um: float | None = None) -> dict:
    """Convert a Stereo-seq GEM or square-bin GEF file to a bin table of the common data model,
    ``outdir/bins_bin<bin_dnb>.npz``.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``.gem``, ``.gem.gz``, or ``.gef`` file.
    outdir : str or pathlib.Path
        Output directory (created).
    sample : str
        Sample name for the QC summary.
    bin_dnb, gene_column, dnb_pitch_um
        As in :func:`scote.io.read_stereoseq`.

    Returns
    -------
    dict
        QC summary: numbers of bins and genes, counts per bin, bin size, output path.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.

    See Also
    --------
    scote.io.read_stereoseq : The same bins in memory, without a copy on disk.
    """
    table = read_stereoseq(path, bin_dnb=bin_dnb, gene_column=gene_column, dnb_pitch_um=dnb_pitch_um)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / f"bins_bin{int(bin_dnb)}.npz"
    write_bin_table(dest, table.xy, table.counts, table.names, table.bin_size)
    tot = np.asarray(table.counts.sum(1)).ravel()
    return dict(sample=sample, platform="stereoseq", source=str(path), bin_dnb=int(bin_dnb),
                bin_um=table.bin_size, n_bins=int(table.counts.shape[0]), n_genes=int(table.counts.shape[1]),
                count_total=float(tot.sum()), counts_per_bin_median=float(np.median(tot)) if len(tot) else 0.0,
                output=str(dest))
