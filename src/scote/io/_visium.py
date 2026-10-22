"""Visium HD and Visium (10x Genomics) Space Ranger outputs -> bin tables of the common data model, on disk or in
memory.

Requires the ``io`` extra (scanpy) to read the feature-barcode matrices. Gene names that occur more than once in a
matrix (one symbol for several gene ids) are made unique with the rule of AnnData's ``var_names_make_unique``
(``-1``, ``-2``, ... appended to the later occurrences), in the readers and the converters alike.
"""
from __future__ import annotations

import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ._schema import BinTable, find_file, make_unique

__all__ = ["VISIUM_SPOT_PITCH_UM", "VISIUM_SPOT_DIAMETER_UM", "convert_visium", "convert_visium_hd", "read_visium",
           "read_visium_hd", "visium_um_per_pixel"]

VISIUM_SPOT_PITCH_UM = 100.0   # Visium spots: 55 µm diameter, 100 µm center to center on a hexagonal array
VISIUM_SPOT_DIAMETER_UM = 55.0


def _read_matrix(h5: Path):
    """A Space Ranger feature-barcode matrix with unique gene names.

    Parameters
    ----------
    h5 : pathlib.Path
        The ``filtered_feature_bc_matrix.h5``.

    Returns
    -------
    X : scipy.sparse.csr_matrix
        (barcodes, genes) counts.
    barcodes : pandas.Index
        The barcodes.
    names : list of str
        The gene names, made unique (``make_unique``).
    """
    import scanpy as sc
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Variable names are not unique")   # made unique below
        ad = sc.read_10x_h5(h5)
    return sp.csr_matrix(ad.X), ad.obs_names, make_unique(ad.var_names)


def convert_visium_hd(rawdir, outdir, sample: str, bins=("square_002um", "square_008um", "square_016um")) -> dict:
    """Convert the binned outputs of a Visium HD run to bin tables.

    Bin centers are the full-resolution image pixels of ``tissue_positions.parquet`` times the
    ``microns_per_pixel`` scale factor (or bin size / ``spot_diameter_fullres`` where that factor is missing).

    Parameters
    ----------
    rawdir : str or pathlib.Path
        Directory searched recursively for ``<bin>/filtered_feature_bc_matrix.h5``.
    outdir : str or pathlib.Path
        Output directory (created); receives ``bins_<bin>.npz`` per bin size found.
    sample : str
        Sample name for the QC summary.
    bins : sequence of str
        Names of the bin-size directories to convert; missing ones are skipped.

    Returns
    -------
    dict
        QC summary per bin size: numbers of bins and genes, UMI totals per bin, area, pixel size, output path.

    Raises
    ------
    FileNotFoundError
        If ``rawdir`` does not exist or holds none of the bin sizes.
    """
    rawdir, outdir = Path(rawdir), Path(outdir)
    if not rawdir.is_dir():
        raise FileNotFoundError(f"directory {rawdir} does not exist")
    outdir.mkdir(parents=True, exist_ok=True)
    qc = dict(sample=sample, platform="visium_hd", bins={})
    for b in bins:
        cands = list(rawdir.rglob(f"{b}/filtered_feature_bc_matrix.h5"))
        if not cands:
            continue
        t0 = time.time()
        xy, X, names, bin_um, in_tissue, upp = _visium_hd_bins(cands[0], b)
        dest = outdir / f"bins_{b}.npz"
        np.savez(dest, data=X.data.astype(np.float32), indices=X.indices, indptr=X.indptr, shape=np.array(X.shape),
                 coords_um=xy, features=np.array(names, dtype=str), bin_um=bin_um, in_tissue=in_tissue)
        tot = np.asarray(X.sum(1)).ravel()
        gene_tot = np.asarray(X.sum(0)).ravel()
        qc["bins"][b] = dict(n_bins=int(X.shape[0]), n_genes=int(X.shape[1]), nnz=int(X.nnz),
                             umi_total=float(tot.sum()), umi_per_bin_mean=float(tot.mean()),
                             umi_per_bin_median=float(np.median(tot)),
                             genes_per_bin_mean=float(np.diff(X.indptr).mean()),
                             genes_ge_500=int((gene_tot >= 500).sum()), area_mm2=float(X.shape[0] * bin_um ** 2 / 1e6),
                             microns_per_pixel=upp, seconds=time.time() - t0, output=str(dest))
    if not qc["bins"]:
        raise FileNotFoundError(f"no {{{', '.join(bins)}}}/filtered_feature_bc_matrix.h5 under {rawdir}")
    return qc


def _visium_hd_bins(h5: Path, b: str):
    """The bins of one bin size of a Visium HD output.

    Parameters
    ----------
    h5 : pathlib.Path
        ``<bin>/filtered_feature_bc_matrix.h5``; the ``spatial`` directory next to it holds the positions.
    b : str
        Name of the bin-size directory, e.g. ``"square_002um"``.

    Returns
    -------
    xy : numpy.ndarray
        (N, 2) float32 bin centers (µm).
    X : scipy.sparse.csr_matrix
        (N, G) counts.
    names : list of str
        Gene names (made unique, ``make_unique``).
    bin_um : float
        Bin size (µm).
    in_tissue : numpy.ndarray
        (N,) in-tissue flags of ``tissue_positions.parquet`` (all True where the column is missing).
    upp : float
        Micrometers per full-resolution pixel.
    """
    import pandas as pd
    bdir = h5.parent
    X, barcodes, names = _read_matrix(h5)
    sf = json.load(open(bdir / "spatial" / "scalefactors_json.json"))
    upp = sf.get("microns_per_pixel")
    bin_um = float(re.findall(r"(\d+)um", b)[0])
    if upp is None:
        upp = bin_um / sf["spot_diameter_fullres"]
    pos = pd.read_parquet(bdir / "spatial" / "tissue_positions.parquet").set_index("barcode")
    pos = pos.reindex(barcodes)
    xy = np.stack([pos["pxl_col_in_fullres"].to_numpy(float) * upp,
                   pos["pxl_row_in_fullres"].to_numpy(float) * upp], 1).astype(np.float32)
    in_tissue = pos["in_tissue"].to_numpy() if "in_tissue" in pos else np.ones(len(pos), bool)
    return xy, X, names, bin_um, in_tissue, upp


def read_visium_hd(rawdir, bin: str = "square_002um", in_tissue_only: bool = True) -> BinTable:
    """Read one bin size of a Visium HD output into memory, in the form :func:`scote.twopoint` takes.

    The bins of :func:`scote.io.convert_visium_hd` (followed by :func:`scote.io.read_bins`), without writing them to
    disk.

    Parameters
    ----------
    rawdir : str or pathlib.Path
        Directory searched recursively for ``<bin>/filtered_feature_bc_matrix.h5`` (e.g. ``outs/binned_outputs``).
    bin : str
        Name of the bin-size directory: ``"square_002um"``, ``"square_008um"``, or ``"square_016um"``.
    in_tissue_only : bool
        Keep only the bins under the tissue.

    Returns
    -------
    BinTable
        ``xy`` (bin centers, µm), ``counts``, ``names``, and ``bin_size`` (µm).

    Raises
    ------
    FileNotFoundError
        If ``rawdir`` holds no ``<bin>/filtered_feature_bc_matrix.h5``.

    Examples
    --------
    >>> xy, X, names, bin_size = scote.io.read_visium_hd("outs/binned_outputs")      # doctest: +SKIP
    >>> res = scote.twopoint(xy, X, names=names, bin_size=bin_size)                  # doctest: +SKIP
    """
    h5 = find_file(rawdir, f"{bin}/filtered_feature_bc_matrix.h5")
    xy, X, names, bin_um, in_tissue, _ = _visium_hd_bins(h5, bin)
    keep = np.asarray(in_tissue).astype(bool) if in_tissue_only else np.ones(len(xy), bool)
    return BinTable(xy[keep].astype(np.float64), X[keep], names, bin_um)


def visium_um_per_pixel(pos) -> tuple[float, float]:
    """Micrometers per full-resolution pixel of a Visium slide, measured from the spot array.

    Within one array row, adjacent spots are two array columns and 100 µm apart, so the median pixel distance between
    such neighbors over the whole array (all spots, not only those under tissue) gives the scale. The
    ``spot_diameter_fullres`` scale factor is not used: Space Ranger defines it for a nominal 65 µm spot.

    Parameters
    ----------
    pos : pandas.DataFrame
        Tissue positions with the columns array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres.

    Returns
    -------
    um_per_pixel : float
        The scale.
    pitch_px : float
        The median spot pitch in pixels.

    Raises
    ------
    ValueError
        If fewer than 100 in-row neighbor pairs are found.
    """
    cols = ["array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    p = pos[cols].astype(float)
    q = p.assign(array_col=p["array_col"] - 2)
    m = p.merge(q, on=["array_row", "array_col"], suffixes=("", "_n"))
    if len(m) < 100:
        raise ValueError(f"too few in-row spot neighbors ({len(m)}) to measure the Visium pitch")
    d = np.hypot(m["pxl_row_in_fullres"] - m["pxl_row_in_fullres_n"],
                 m["pxl_col_in_fullres"] - m["pxl_col_in_fullres_n"])
    pitch_px = float(np.median(d))
    return VISIUM_SPOT_PITCH_UM / pitch_px, pitch_px


def convert_visium(rawdir, outdir, sample: str) -> dict:
    """Convert a Visium (55 µm spots) Space Ranger output to a bin table.

    Spot centers are the full-resolution pixels times the scale of :func:`scote.io.visium_um_per_pixel`; ``bin_um`` is
    55.

    Parameters
    ----------
    rawdir : str or pathlib.Path
        Directory searched recursively for ``filtered_feature_bc_matrix.h5``, ``scalefactors_json.json``, and
        ``tissue_positions*``.
    outdir : str or pathlib.Path
        Output directory (created); receives ``bins_spots55um.npz``.
    sample : str
        Sample name for the QC summary.

    Returns
    -------
    dict
        QC summary: numbers of spots and genes, UMI totals, pixel size and spot pitch, output path.

    Raises
    ------
    FileNotFoundError
        If the matrix, the scale factors, or the tissue positions are not found under ``rawdir``.
    """
    rawdir, outdir = Path(rawdir), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    xy, X, names, info = _visium_spots(rawdir)
    dest = outdir / "bins_spots55um.npz"
    np.savez(dest, data=X.data.astype(np.float32), indices=X.indices, indptr=X.indptr, shape=np.array(X.shape),
             coords_um=xy, features=np.array(names, dtype=str), bin_um=VISIUM_SPOT_DIAMETER_UM,
             in_tissue=np.ones(X.shape[0], bool))
    tot = np.asarray(X.sum(1)).ravel()
    return dict(sample=sample, platform="visium", n_spots=int(X.shape[0]), n_genes=int(X.shape[1]),
                umi_total=float(tot.sum()), umi_per_spot_median=float(np.median(tot)), **info, output=str(dest))


def _visium_spots(rawdir: Path):
    """The spots of a Visium output.

    Parameters
    ----------
    rawdir : pathlib.Path
        Directory searched recursively for ``filtered_feature_bc_matrix.h5``, ``scalefactors_json.json``, and
        ``tissue_positions*``.

    Returns
    -------
    xy : numpy.ndarray
        (N, 2) float32 spot centers (µm).
    X : scipy.sparse.csr_matrix
        (N, G) counts.
    names : list of str
        Gene names (made unique, ``make_unique``).
    info : dict
        ``microns_per_pixel``, ``spot_pitch_px``, ``spot_diameter_fullres_um``.
    """
    import pandas as pd
    h5 = find_file(rawdir, "filtered_feature_bc_matrix.h5")
    sf = json.loads(find_file(rawdir, "scalefactors_json.json").read_text())
    posf = find_file(rawdir, "tissue_positions*")
    X, barcodes, names = _read_matrix(h5)
    pos = pd.read_csv(posf, header=0 if "list" not in posf.name else None)
    if pos.shape[1] == 6 and "barcode" not in pos.columns:
        pos.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    upp, pitch_px = visium_um_per_pixel(pos)
    pos = pos.set_index("barcode").reindex(barcodes)
    xy = np.stack([pos["pxl_col_in_fullres"].to_numpy(float) * upp,
                   pos["pxl_row_in_fullres"].to_numpy(float) * upp], 1).astype(np.float32)
    info = dict(microns_per_pixel=upp, spot_pitch_px=pitch_px,
                spot_diameter_fullres_um=float(sf["spot_diameter_fullres"] * upp))
    return xy, X, names, info


def read_visium(rawdir) -> BinTable:
    r"""Read a Visium (55 µm spots) output into memory, in the form :func:`scote.twopoint` takes.

    The spots of :func:`scote.io.convert_visium`, without writing them to disk. The bin size is the spot diameter,
    55 µm; the spot centers are 100 µm apart on a hexagonal array.

    Parameters
    ----------
    rawdir : str or pathlib.Path
        Directory searched recursively for ``filtered_feature_bc_matrix.h5``, ``scalefactors_json.json``, and
        ``tissue_positions*``.

    Returns
    -------
    BinTable
        ``xy`` (spot centers, µm), ``counts``, ``names``, and ``bin_size`` (55 µm).

    Raises
    ------
    FileNotFoundError
        If the matrix, the scale factors, or the tissue positions are not found under ``rawdir``.
    """
    xy, X, names, _ = _visium_spots(Path(rawdir))
    return BinTable(xy.astype(np.float64), X, names, VISIUM_SPOT_DIAMETER_UM)
