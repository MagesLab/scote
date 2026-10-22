"""10x Genomics output bundles without ``transcripts.parquet`` (Atera): molecules from the Xenium Explorer zarr stores
-> molecule table of the common data model, on disk or in memory.

The bundle ships the zarr stores of the Xenium Explorer instead of the transcript table: ``transcripts.zarr.zip``
(the transcripts in spatial grids; level 0 holds all of them, per grid cell ``location`` (N, 3) float32 µm,
``gene_identity`` (N, 1) into the root attribute ``gene_names``, ``quality_score`` (N, 1), the Xenium quality value,
``valid``, and ``status``) and ``cells.zarr.zip`` (the segmentation label masks ``masks/1`` (cells) and ``masks/0``
(nuclei), label k = cell index k - 1, pixel = µm times ``masks/homogeneous_transform``, and ``cell_id``). The layout
is the Xenium Onboard Analysis zarr output described in the public 10x Genomics documentation. Cell and nucleus
membership are looked up in the label masks, the 2-D assignment of the instrument software. Requires the ``io``
extra (``zarr``, ``numcodecs``).
"""
from __future__ import annotations

import contextlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from ._schema import MOLECULE_SCHEMA_DOC, molecule_frame, write_molecules
from ._xenium import XENIUM_CONTROL_RE

__all__ = ["convert_atera", "read_atera"]


@contextlib.contextmanager
def _zopen(path: Path):
    """A zarr v2 group in a zip archive, read-only; the archive is closed on exit.

    Parameters
    ----------
    path : pathlib.Path
        The ``.zarr.zip`` archive.

    Yields
    ------
    zarr.Group
        The root group.
    """
    import zarr
    from zarr.storage import ZipStore
    store = ZipStore(str(path), mode="r")
    try:
        yield zarr.open_group(store, mode="r", zarr_format=2)
    finally:
        store.close()


def _read_v2(zf: zipfile.ZipFile, path: str) -> np.ndarray:
    """A zarr v2 array of an archive, read chunk by chunk along the first axis.

    Used for the per-grid-cell arrays, whose dtype strings (e.g. ``"u1"``) some zarr versions reject.

    Parameters
    ----------
    zf : zipfile.ZipFile
        The archive.
    path : str
        Path of the array inside the archive.

    Returns
    -------
    numpy.ndarray
        The array (missing chunks hold the fill value).
    """
    import numcodecs
    md = json.loads(zf.read(f"{path}/.zarray"))
    dt = np.dtype(md["dtype"])
    shape, ch = md["shape"], md["chunks"]
    codec = numcodecs.get_codec(md["compressor"]) if md["compressor"] else None
    out = np.full(shape, md["fill_value"] or 0, dtype=dt)
    for i in range(0, shape[0], ch[0]):
        name = f"{path}/{i // ch[0]}" + (".0" * (len(shape) - 1))
        try:
            b = zf.read(name)
        except KeyError:
            continue
        b = codec.decode(b) if codec else b
        blk = np.frombuffer(b, dtype=dt).reshape(ch, order=md.get("order", "C"))
        n = min(ch[0], shape[0] - i)
        out[i:i + n] = blk[:n]
    return out


def _atera_tables(rawdir: Path, sample: str, qv_min: float, batch_rows: int, stats: dict):
    """The molecules of the zarr stores as tables of the common data model.

    Parameters
    ----------
    rawdir : pathlib.Path
        Directory with ``transcripts.zarr.zip`` and ``cells.zarr.zip``.
    sample : str
        Value of the ``sample`` column.
    qv_min : float
        Minimum quality value.
    batch_rows : int
        Molecules buffered per table.
    stats : dict
        Updated in place with the counts of the QC summary.

    Yields
    ------
    pyarrow.Table
        Kept molecules, about ``batch_rows`` at a time, in the order of the grid cells.
    """
    with _zopen(rawdir / "transcripts.zarr.zip") as tz:
        gene_names = np.array(tz.attrs["gene_names"], dtype=object)
        gene_category = tz["gene_category"][:]
        grid_attrs = tz["grids"].attrs.asdict()
    is_ctrl = np.array([bool(XENIUM_CONTROL_RE.match(str(n))) for n in gene_names])
    new_index = np.full(len(gene_names), -1, np.int32)
    new_index[~is_ctrl] = np.arange((~is_ctrl).sum(), dtype=np.int32)
    dict_arr = pa.array(gene_names[~is_ctrl].tolist(), type=pa.string())
    keys0, nobj0 = grid_attrs["grid_keys"][0], grid_attrs["grid_number_objects"][0]
    with _zopen(rawdir / "cells.zarr.zip") as cz:
        transform = cz["masks/homogeneous_transform"][:].astype(np.float64)
        cid_of_index = cz["cell_id"][:][:, 0].astype(np.int64)
        mask_cell = cz["masks/1"][:]
        mask_nuc = cz["masks/0"][:]
    n_cells = len(cid_of_index)
    tzip = zipfile.ZipFile(rawdir / "transcripts.zarr.zip")
    H, W = mask_cell.shape
    stats.update(n_in=0, n_qv=0, n_ctrl=0, n_out=0, n_incell=0, n_nuc=0, n_invalid=0, n_status_nz=0,
                 n_invalid_kept=0, n_nuc_other_cell=0, n_nuc_no_cell=0, gene_counts=np.zeros(len(gene_names), np.int64),
                 per_cell=np.zeros(n_cells + 1, np.int64), bbox=[np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf],
                 gene_names=gene_names, cid_of_index=cid_of_index, mask=(mask_cell, transform))
    sample_dict = pa.array([sample], type=pa.string())
    buf, buf_n = [], 0

    def table(parts):
        """The buffered molecules as one table of the common data model.

        Parameters
        ----------
        parts : list of dict
            Arrays per grid cell (``x``, ``y``, ``z``, ``f``, ``cid``, ``nuc``, ``qv``).

        Returns
        -------
        pyarrow.Table
            The molecules.
        """
        cols = {k: np.concatenate([b[k] for b in parts]) for k in parts[0]}
        n = len(cols["x"])
        return pa.table({
            "x": pa.array(cols["x"]), "y": pa.array(cols["y"]), "z": pa.array(cols["z"]),
            "feature": pa.DictionaryArray.from_arrays(pa.array(cols["f"], type=pa.int32()), dict_arr),
            "sample": pa.DictionaryArray.from_arrays(pa.array(np.zeros(n, np.int32)), sample_dict),
            "cell_id": pa.array(cols["cid"]), "nucleus": pa.array(cols["nuc"]), "qv": pa.array(cols["qv"]),
        })

    try:
        b = stats["bbox"]
        for i, key in enumerate(keys0):
            if nobj0[i] == 0:
                continue
            loc = _read_v2(tzip, f"grids/0/{key}/location")
            gi = _read_v2(tzip, f"grids/0/{key}/gene_identity")[:, 0].astype(np.int64)
            qv = _read_v2(tzip, f"grids/0/{key}/quality_score")[:, 0].astype(np.float32)
            valid = _read_v2(tzip, f"grids/0/{key}/valid").reshape(len(gi), -1)[:, 0]
            status = _read_v2(tzip, f"grids/0/{key}/status").reshape(len(gi), -1)[:, 0]
            stats["n_in"] += len(gi)
            stats["n_invalid"] += int((valid == 0).sum())
            stats["n_status_nz"] += int((status != 0).sum())
            keep_qv = qv >= qv_min
            ctrl = is_ctrl[gi]
            keep = keep_qv & ~ctrl
            stats["n_qv"] += int((~keep_qv).sum())
            stats["n_ctrl"] += int((keep_qv & ctrl).sum())
            k = int(keep.sum())
            if k == 0:
                continue
            stats["n_out"] += k
            x, y, z = (loc[keep, d].astype(np.float32) for d in range(3))
            g, q = gi[keep], qv[keep]
            stats["n_invalid_kept"] += int((valid[keep] == 0).sum())
            px = np.clip(np.floor(x * transform[0, 0] + transform[0, 3]).astype(np.int64), 0, W - 1)
            py = np.clip(np.floor(y * transform[1, 1] + transform[1, 3]).astype(np.int64), 0, H - 1)
            lc = mask_cell[py, px].astype(np.int64)
            ln = mask_nuc[py, px].astype(np.int64)
            cid = np.where(lc > 0, cid_of_index[np.maximum(lc - 1, 0)], -1)
            nuc = ln > 0
            stats["n_nuc_other_cell"] += int(((ln > 0) & (lc > 0) & (ln != lc)).sum())
            stats["n_nuc_no_cell"] += int(((ln > 0) & (lc == 0)).sum())
            stats["n_incell"] += int((lc > 0).sum())
            stats["n_nuc"] += int(nuc.sum())
            stats["gene_counts"] += np.bincount(g, minlength=len(gene_names))
            stats["per_cell"] += np.bincount(lc[gene_category[g, 0].astype(bool)], minlength=n_cells + 1)
            b[0], b[3] = min(b[0], float(x.min())), max(b[3], float(x.max()))
            b[1], b[4] = min(b[1], float(y.min())), max(b[4], float(y.max()))
            b[2], b[5] = min(b[2], float(z.min())), max(b[5], float(z.max()))
            buf.append(dict(x=x, y=y, z=z, f=new_index[g], cid=cid, nuc=nuc, qv=q))
            buf_n += k
            if buf_n >= batch_rows:
                yield table(buf)
                buf, buf_n = [], 0
        if buf:
            yield table(buf)
    finally:
        tzip.close()


def _matrix_check(rawdir: Path, stats: dict) -> dict | None:
    """Per-cell gene counts of the converted molecules against the cell-feature matrix of the bundle.

    Parameters
    ----------
    rawdir : pathlib.Path
        The bundle directory; ``cell_feature_matrix.zarr.zip`` is used if present.
    stats : dict
        Statistics of the conversion (per-cell counts of gene features).

    Returns
    -------
    dict or None
        Totals, the fraction of cells with identical counts, and the correlation; ``None`` without a matrix.
    """
    path = rawdir / "cell_feature_matrix.zarr.zip"
    if not path.exists():
        return None
    with _zopen(path) as root:
        cf = root["cell_features"]
        indptr = cf["csc/indptr"][:].astype(np.int64)
        gene = np.array(cf.attrs["feature_types"]) == "gene"
        data = cf["csc/data"][:].astype(np.int64) * gene[cf["csc/indices"][:]]
        cell_ids = cf["cell_id"][:, 0].astype(np.int64)
    cum = np.concatenate([[0], np.cumsum(data)])
    matrix = cum[indptr[1:]] - cum[indptr[:-1]]
    cid = stats["cid_of_index"]
    order = np.argsort(cid)
    pos = order[np.searchsorted(cid, cell_ids, sorter=order)]
    ours = stats["per_cell"][1:][pos]
    return dict(matrix_cells=int(len(matrix)), matrix_total=int(matrix.sum()), ours_total=int(ours.sum()),
                frac_cells_exact=float(np.mean(ours == matrix)),
                pearson_r=float(np.corrcoef(ours, matrix)[0, 1]) if len(matrix) > 1 else None)


def convert_atera(rawdir, outdir, sample: str, qv_min: float = 20.0, batch_rows: int = 5_000_000) -> dict:
    """Convert the transcripts of a zarr-only 10x bundle (Atera) to ``outdir/molecules.parquet``.

    The filters are those of :func:`scote.io.convert_xenium`: molecules with ``qv < qv_min`` are dropped, then the
    control features (the control names of :func:`scote.io.convert_xenium`). ``cell_id`` is the first column of the
    cell's ``cell_id`` (the integer that :func:`scote.io.cell_id_to_int` gives for the string form; -1 outside
    cells), ``nucleus`` whether the nucleus mask covers the molecule.

    Parameters
    ----------
    rawdir : str or pathlib.Path
        The bundle directory with ``transcripts.zarr.zip`` and ``cells.zarr.zip`` (and optionally
        ``cell_feature_matrix.zarr.zip`` for a cross-check).
    outdir : str or pathlib.Path
        Output directory (created); receives ``molecules.parquet`` and ``feature_counts.json``.
    sample : str
        Value of the ``sample`` column.
    qv_min : float
        Minimum quality value.
    batch_rows : int
        Molecules per row group.

    Returns
    -------
    dict
        QC summary with the keys of :func:`scote.io.convert_xenium` and an ``atera`` block: z range, molecules with
        ``valid == 0`` or ``status != 0``, nucleus labels outside or different from the cell label, and the per-cell
        cross-check against the cell-feature matrix.

    Raises
    ------
    FileNotFoundError
        If a zarr store is missing.

    See Also
    --------
    read_atera : The same molecules in memory, without a copy on disk.
    """
    rawdir, outdir = Path(rawdir), Path(outdir)
    for name in ("transcripts.zarr.zip", "cells.zarr.zip"):
        if not (rawdir / name).exists():
            raise FileNotFoundError(f"no {name} in {rawdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / "molecules.parquet"
    t0 = time.time()
    st: dict = {}
    write_molecules(_atera_tables(rawdir, sample, qv_min, batch_rows, st), dest)
    st.pop("mask")
    names = st["gene_names"]
    fc = {str(names[j]): int(c) for j, c in enumerate(st["gene_counts"]) if c > 0}
    n_out = st["n_out"]
    xmin, ymin, zmin, xmax, ymax, zmax = st["bbox"]
    area = (xmax - xmin) * (ymax - ymin) / 1e6 if n_out else 0.0
    with open(outdir / "feature_counts.json", "w") as f:
        json.dump(fc, f)
    return dict(sample=sample, platform="atera", source=str(rawdir / "transcripts.zarr.zip"),
                molecules_in=st["n_in"], dropped_qv=st["n_qv"], dropped_control=st["n_ctrl"], molecules=n_out,
                frac_in_cells=(st["n_incell"] / n_out if n_out else None),
                frac_in_nucleus=(st["n_nuc"] / n_out if n_out else None), n_features=len(fc),
                bbox_um=[xmin, ymin, xmax, ymax], bbox_area_mm2=area,
                molecules_per_mm2_bbox=(n_out / area if area else None),
                top_features=sorted(fc.items(), key=lambda kv: -kv[1])[:20],
                features_ge_500=sum(1 for v in fc.values() if v >= 500), seconds=time.time() - t0,
                output=str(dest), schema=MOLECULE_SCHEMA_DOC,
                atera=dict(z_range_um=[zmin, zmax], valid_eq_0=st["n_invalid"], status_ne_0=st["n_status_nz"],
                           valid_eq_0_among_kept=st["n_invalid_kept"],
                           nucleus_label_ne_cell_label=st["n_nuc_other_cell"],
                           nucleus_outside_cell=st["n_nuc_no_cell"],
                           cell_assignment="2-D label-mask lookup (cells.zarr masks/1, masks/0)",
                           matrix_check=_matrix_check(rawdir, st)))


def read_atera(rawdir, sample: str = "", qv_min: float = 20.0, columns=None,
               batch_rows: int = 5_000_000) -> pd.DataFrame:
    """Read the transcripts of a zarr-only 10x bundle (Atera) into memory, in the form :func:`scote.twopoint` takes.

    The molecules and columns of :func:`scote.io.convert_atera`, without writing them to disk.

    Parameters
    ----------
    rawdir : str or pathlib.Path
        The bundle directory with ``transcripts.zarr.zip`` and ``cells.zarr.zip``.
    sample : str
        Value of the ``sample`` column.
    qv_min : float
        Minimum quality value.
    columns : sequence of str or None
        Columns to keep (``x``, ``y``, and ``feature`` always); ``None`` keeps all.
    batch_rows : int
        Molecules per table read.

    Returns
    -------
    pandas.DataFrame
        One row per molecule: ``x``, ``y``, ``z`` (µm), ``feature`` (Categorical), ``sample``, ``cell_id``,
        ``nucleus``, ``qv``.

    Raises
    ------
    FileNotFoundError
        If a zarr store is missing.
    """
    rawdir = Path(rawdir)
    for name in ("transcripts.zarr.zip", "cells.zarr.zip"):
        if not (rawdir / name).exists():
            raise FileNotFoundError(f"no {name} in {rawdir}")
    return molecule_frame(_atera_tables(rawdir, sample, qv_min, batch_rows, {}), columns)
