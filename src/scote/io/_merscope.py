"""MERSCOPE (Vizgen) ``detected_transcripts.csv`` -> molecule table of the common data model, on disk or in memory."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from ._schema import find_file, molecule_frame, write_molecules
from ._xenium import cell_id_to_int

__all__ = ["convert_merscope", "read_merscope"]


def _merscope_tables(src: Path, sample: str, chunk_rows: int, stats: dict | None):
    """The molecules of a ``detected_transcripts.csv`` as tables of the common data model, chunk by chunk.

    Parameters
    ----------
    src : pathlib.Path
        The csv file (optionally compressed).
    sample : str
        Value of the ``sample`` column.
    chunk_rows : int
        Rows per csv chunk.
    stats : dict or None
        Updated in place with the counts of the QC summary; ``None`` skips them (the readers).

    Yields
    ------
    pyarrow.Table
        The kept molecules of one chunk.
    """
    if stats is not None:
        stats.update(n_in=0, n_out=0, n_ctrl=0, n_incell=0, feature_counts={},
                     bbox=[np.inf, np.inf, -np.inf, -np.inf])
        feature_counts, bbox = stats["feature_counts"], stats["bbox"]
    columns = {"global_x", "global_y", "global_z", "gene", "cell_id", "transcript_id", "barcode_id"}
    for ch in pd.read_csv(src, usecols=columns.__contains__, chunksize=chunk_rows):
        gene = ch["gene"].astype(str).to_numpy()
        is_ctrl = np.char.startswith(gene.astype(str), "Blank")
        keep = ~is_ctrl
        cid = ch["cell_id"].to_numpy() if "cell_id" in ch else np.full(len(ch), -1)
        cid = np.where(pd.isna(cid), -1, cid)
        cid = np.array([cell_id_to_int(c) if isinstance(c, str) else int(c) for c in cid], dtype=np.int64)
        cid[cid == 0] = -1  # -1 and 0: not in a cell
        x = ch["global_x"].to_numpy(np.float32)[keep]
        y = ch["global_y"].to_numpy(np.float32)[keep]
        z = (ch["global_z"].to_numpy(np.float32)[keep] if "global_z" in ch
             else np.zeros(int(keep.sum()), np.float32))
        if stats is not None:
            stats["n_in"] += len(ch)
            stats["n_ctrl"] += int(is_ctrl.sum())
            stats["n_out"] += int(keep.sum())
            stats["n_incell"] += int((cid[keep] >= 0).sum())
        if stats is not None and keep.any():
            bbox[0] = min(bbox[0], float(x.min()))
            bbox[2] = max(bbox[2], float(x.max()))
            bbox[1] = min(bbox[1], float(y.min()))
            bbox[3] = max(bbox[3], float(y.max()))
            u, c = np.unique(gene[keep], return_counts=True)
            for uu, cc in zip(u, c):
                feature_counts[uu] = feature_counts.get(uu, 0) + int(cc)
        yield pa.table({"x": pa.array(x), "y": pa.array(y), "z": pa.array(z),
                        "feature": pa.array(gene[keep].tolist(), type=pa.string()).dictionary_encode(),
                        "sample": pa.array([sample] * int(keep.sum()), type=pa.string()).dictionary_encode(),
                        "cell_id": pa.array(cid[keep]), "nucleus": pa.array(np.zeros(int(keep.sum()), bool))})


def convert_merscope(rawdir, outdir, sample: str, chunk_rows: int = 5_000_000) -> dict:
    """Convert a MERSCOPE ``detected_transcripts.csv`` to ``outdir/molecules.parquet``.

    Columns ``global_x``, ``global_y`` (µm), ``global_z`` (z-plane index), ``gene``, ``cell_id``; ``Blank-*``
    control barcodes are dropped. The csv is streamed in chunks (liver sections have about :math:`3 \\times 10^8`
    rows). The file has no nucleus flag, so ``nucleus`` is False.

    Parameters
    ----------
    rawdir : str or pathlib.Path
        Directory searched recursively for ``detected_transcripts.csv`` (optionally compressed).
    outdir : str or pathlib.Path
        Output directory (created); receives ``molecules.parquet`` and ``feature_counts.json``.
    sample : str
        Value of the ``sample`` column.
    chunk_rows : int
        Rows per csv chunk.

    Returns
    -------
    dict
        QC summary: molecules read and kept, dropped controls, fraction in cells, number of features, bounding box,
        the 20 most abundant features, timing, output path.

    Raises
    ------
    FileNotFoundError
        If no ``detected_transcripts.csv`` is found under ``rawdir``.

    See Also
    --------
    read_merscope : The same molecules in memory, without a copy on disk.
    """
    rawdir, outdir = Path(rawdir), Path(outdir)
    src = find_file(rawdir, "detected_transcripts.csv*")
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / "molecules.parquet"
    t0 = time.time()
    st: dict = {}
    write_molecules(_merscope_tables(src, sample, chunk_rows, st), dest)
    n_out, feature_counts = st["n_out"], st["feature_counts"]
    xmin, ymin, xmax, ymax = st["bbox"]
    area = (xmax - xmin) * (ymax - ymin) / 1e6 if n_out else 0.0
    with open(outdir / "feature_counts.json", "w") as f:
        json.dump(feature_counts, f)
    return dict(sample=sample, platform="merscope", source=str(src), molecules_in=st["n_in"],
                dropped_control=st["n_ctrl"], molecules=n_out,
                frac_in_cells=(st["n_incell"] / n_out if n_out else None), n_features=len(feature_counts),
                bbox_um=[xmin, ymin, xmax, ymax], bbox_area_mm2=area,
                top_features=sorted(feature_counts.items(), key=lambda kv: -kv[1])[:20],
                features_ge_500=sum(1 for v in feature_counts.values() if v >= 500), seconds=time.time() - t0,
                output=str(dest), note="nucleus flag not available in detected_transcripts.csv; z = z-slice index")


def read_merscope(rawdir, sample: str = "", columns=None, chunk_rows: int = 5_000_000) -> pd.DataFrame:
    """Read the molecules of a MERSCOPE output into memory, in the form :func:`scote.twopoint` takes.

    The molecules and columns of :func:`scote.io.convert_merscope`, without writing them to disk.

    Parameters
    ----------
    rawdir : str or pathlib.Path
        Directory searched recursively for ``detected_transcripts.csv`` (optionally compressed).
    sample : str
        Value of the ``sample`` column.
    columns : sequence of str or None
        Columns to keep (``x``, ``y``, and ``feature`` always); ``None`` keeps all.
    chunk_rows : int
        Rows per csv chunk.

    Returns
    -------
    pandas.DataFrame
        One row per molecule: ``x``, ``y`` (µm), ``z`` (z-slice index), ``feature`` (Categorical), ``sample``,
        ``cell_id``, ``nucleus``.

    Raises
    ------
    FileNotFoundError
        If no ``detected_transcripts.csv`` is found under ``rawdir``.
    """
    src = find_file(rawdir, "detected_transcripts.csv*")
    return molecule_frame(_merscope_tables(src, sample, chunk_rows, None), columns)
