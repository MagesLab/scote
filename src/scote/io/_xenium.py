"""Xenium (10x Genomics) ``transcripts.parquet`` -> molecule table of the common data model, on disk or in memory."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ._schema import MOLECULE_SCHEMA_DOC, find_file, molecule_frame, write_molecules

__all__ = ["XENIUM_CONTROL_RE", "cell_id_to_int", "convert_xenium", "read_xenium"]

# control codewords and probes dropped by convert_xenium
XENIUM_CONTROL_RE = re.compile(r"^(NegControl|BLANK|Unassigned|Deprecated|Intergenic|UnassignedCodeword|antisense)",
                               re.I)


def cell_id_to_int(c: str) -> int:
    """Integer form of a string cell id of the Xenium and MERSCOPE outputs.

    Parameters
    ----------
    c : str
        Cell id such as ``"ffkpbaba-1"``: letters a .. p in base 16, then ``-`` and a data-set suffix.

    Returns
    -------
    int
        The base-16 value of the part before the first ``-`` (letter a = digit 0).
    """
    core = c.split("-")[0]
    v = 0
    for ch in core:
        v = v * 16 + (ord(ch) - ord("a"))
    return v


def _xenium_tables(src: Path, sample: str, qv_min: float, batch_rows: int, stats: dict | None):
    """The molecules of a ``transcripts.parquet`` as tables of the common data model, batch by batch.

    Parameters
    ----------
    src : pathlib.Path
        The ``transcripts.parquet``.
    sample : str
        Value of the ``sample`` column.
    qv_min : float
        Minimum quality value.
    batch_rows : int
        Rows per batch.
    stats : dict or None
        Updated in place with the counts of the QC summary (``n_in``, ``n_qv``, ``n_ctrl``, ``n_out``, ``n_incell``,
        ``n_nuc``, ``feature_counts``, ``bbox``); ``None`` skips them (the readers).

    Yields
    ------
    pyarrow.Table
        The kept molecules of one batch.
    """
    pf = pq.ParquetFile(src)
    xcol, ycol, zcol = ("x_location", "y_location", "z_location")
    cell_col = "cell_id"
    nuc_col = "overlaps_nucleus"
    feat = "feature_name"
    qvc = "qv"
    if stats is not None:
        stats.update(n_in=0, n_qv=0, n_ctrl=0, n_out=0, n_incell=0, n_nuc=0, feature_counts={},
                     bbox=[np.inf, np.inf, -np.inf, -np.inf])
        feature_counts, bbox = stats["feature_counts"], stats["bbox"]
    for batch in pf.iter_batches(batch_size=batch_rows, columns=[xcol, ycol, zcol, feat, cell_col, nuc_col, qvc]):
        x = batch.column(xcol).to_numpy(zero_copy_only=False).astype(np.float32)
        y = batch.column(ycol).to_numpy(zero_copy_only=False).astype(np.float32)
        z = batch.column(zcol).to_numpy(zero_copy_only=False).astype(np.float32)
        qv = batch.column(qvc).to_numpy(zero_copy_only=False).astype(np.float32)
        fn = batch.column(feat)
        if pa.types.is_binary(fn.type) or pa.types.is_large_binary(fn.type):
            fn = fn.cast(pa.string())
        fn_np = np.asarray(fn.to_pylist() if not pa.types.is_dictionary(fn.type)
                           else fn.dictionary_decode().to_pylist(), dtype=object)
        cid = batch.column(cell_col)
        if pa.types.is_string(cid.type) or pa.types.is_large_string(cid.type) or pa.types.is_binary(cid.type):
            cid_np = np.array([-1 if (c is None or c == "UNASSIGNED") else cell_id_to_int(c) for c in cid.to_pylist()],
                              dtype=np.int64)
        else:
            cid_np = cid.to_numpy(zero_copy_only=False).astype(np.int64)
            cid_np[cid_np == 0] = -1
        nuc = batch.column(nuc_col).to_numpy(zero_copy_only=False).astype(bool)
        keep_qv = qv >= qv_min
        is_ctrl = np.array([bool(XENIUM_CONTROL_RE.match(f)) for f in fn_np])
        keep = keep_qv & ~is_ctrl
        if stats is not None:
            stats["n_in"] += batch.num_rows
            stats["n_qv"] += int((~keep_qv).sum())
            stats["n_ctrl"] += int((keep_qv & is_ctrl).sum())
            stats["n_out"] += int(keep.sum())
            stats["n_incell"] += int((cid_np[keep] >= 0).sum())
            stats["n_nuc"] += int(nuc[keep].sum())
        if stats is not None and keep.any():
            bbox[0] = min(bbox[0], float(x[keep].min()))
            bbox[2] = max(bbox[2], float(x[keep].max()))
            bbox[1] = min(bbox[1], float(y[keep].min()))
            bbox[3] = max(bbox[3], float(y[keep].max()))
            u, c = np.unique(fn_np[keep], return_counts=True)
            for uu, cc in zip(u, c):
                feature_counts[uu] = feature_counts.get(uu, 0) + int(cc)
        yield pa.table({
            "x": pa.array(x[keep]), "y": pa.array(y[keep]), "z": pa.array(z[keep]),
            "feature": pa.array(fn_np[keep].tolist(), type=pa.string()).dictionary_encode(),
            "sample": pa.array([sample] * int(keep.sum()), type=pa.string()).dictionary_encode(),
            "cell_id": pa.array(cid_np[keep]), "nucleus": pa.array(nuc[keep]), "qv": pa.array(qv[keep]),
        })


def convert_xenium(rawdir, outdir, sample: str, qv_min: float = 20.0, batch_rows: int = 5_000_000) -> dict:
    """Convert a Xenium ``transcripts.parquet`` to ``outdir/molecules.parquet``.

    Molecules with ``qv < qv_min`` and control codewords and probes (names starting with ``NegControl``, ``BLANK``,
    ``Unassigned``, ``Deprecated``, ``Intergenic``, ``UnassignedCodeword``, or ``antisense``) are dropped. Cell ids:
    string ids (Xenium 1.7 and later) are mapped with :func:`scote.io.cell_id_to_int`, ``"UNASSIGNED"`` to -1; integer
    ids (Xenium before 1.7) keep their value, 0 (unassigned) becomes -1. ``nucleus`` is the ``overlaps_nucleus`` flag.

    Parameters
    ----------
    rawdir : str or pathlib.Path
        Directory searched recursively for ``transcripts.parquet``.
    outdir : str or pathlib.Path
        Output directory (created); receives ``molecules.parquet`` and ``feature_counts.json``.
    sample : str
        Value of the ``sample`` column.
    qv_min : float
        Minimum Phred-scaled quality value.
    batch_rows : int
        Rows per streamed batch.

    Returns
    -------
    dict
        QC summary: molecules read and kept, drops by quality and controls, fractions in cells and nuclei, number of
        features, bounding box and density, the 20 most abundant features, timing, output path.

    Raises
    ------
    FileNotFoundError
        If no ``transcripts.parquet`` is found under ``rawdir``.

    See Also
    --------
    read_xenium : The same molecules in memory, without a copy on disk.
    """
    rawdir, outdir = Path(rawdir), Path(outdir)
    src = find_file(rawdir, "transcripts.parquet")
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / "molecules.parquet"
    t0 = time.time()
    st: dict = {}
    write_molecules(_xenium_tables(src, sample, qv_min, batch_rows, st), dest)
    n_out, feature_counts = st["n_out"], st["feature_counts"]
    xmin, ymin, xmax, ymax = st["bbox"]
    area_bbox = (xmax - xmin) * (ymax - ymin) / 1e6 if n_out else 0.0
    qc = dict(sample=sample, platform="xenium", source=str(src), molecules_in=st["n_in"], dropped_qv=st["n_qv"],
              dropped_control=st["n_ctrl"], molecules=n_out, frac_in_cells=(st["n_incell"] / n_out if n_out else None),
              frac_in_nucleus=(st["n_nuc"] / n_out if n_out else None), n_features=len(feature_counts),
              bbox_um=[xmin, ymin, xmax, ymax], bbox_area_mm2=area_bbox,
              molecules_per_mm2_bbox=(n_out / area_bbox if area_bbox else None),
              top_features=sorted(feature_counts.items(), key=lambda kv: -kv[1])[:20],
              features_ge_500=sum(1 for v in feature_counts.values() if v >= 500),
              seconds=time.time() - t0, output=str(dest), schema=MOLECULE_SCHEMA_DOC)
    with open(outdir / "feature_counts.json", "w") as f:
        json.dump(feature_counts, f)
    return qc


def read_xenium(rawdir, sample: str = "", qv_min: float = 20.0, columns=None,
                batch_rows: int = 5_000_000) -> pd.DataFrame:
    """Read the molecules of a Xenium output into memory, in the form :func:`scote.twopoint` takes.

    The molecules and columns of :func:`scote.io.convert_xenium`, without writing them to disk.

    Parameters
    ----------
    rawdir : str or pathlib.Path
        Directory searched recursively for ``transcripts.parquet`` (e.g. the ``outs`` directory).
    sample : str
        Value of the ``sample`` column.
    qv_min : float
        Minimum Phred-scaled quality value.
    columns : sequence of str or None
        Columns to keep (``x``, ``y``, and ``feature`` always); ``None`` keeps all columns of the common data model.
    batch_rows : int
        Rows per batch read.

    Returns
    -------
    pandas.DataFrame
        One row per molecule: ``x``, ``y``, ``z`` (µm), ``feature`` (Categorical), ``sample``, ``cell_id``,
        ``nucleus``, ``qv``.

    Raises
    ------
    FileNotFoundError
        If no ``transcripts.parquet`` is found under ``rawdir``.

    Examples
    --------
    >>> df = scote.io.read_xenium("outs/")                                    # doctest: +SKIP
    >>> res = scote.twopoint(df[["x", "y"]].to_numpy(), df["feature"])        # doctest: +SKIP
    """
    src = find_file(rawdir, "transcripts.parquet")
    return molecule_frame(_xenium_tables(src, sample, qv_min, batch_rows, None), columns)
