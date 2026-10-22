"""CosMx (Bruker Spatial Biology) AtoMx flat-file transcripts -> molecule table of the common data model, on disk or
in memory.

The AtoMx flat-file export holds one transcript table, ``<run>_tx_file.csv`` (often gzip-compressed), with the
columns ``fov``, ``cell_ID`` (0 = not in a cell), ``x_global_px``, ``y_global_px`` (pixels of the slide frame), ``z``
(z-plane index), ``target`` (the probe target), and ``CellComp`` (``Nuclear``, ``Cytoplasm``, ``Membrane``, or
``None``); further columns are not used.
"""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from ._schema import find_file, molecule_frame, write_molecules

__all__ = ["COSMX_CODECLASSES", "COSMX_PIXEL_UM", "convert_cosmx", "read_cosmx"]

COSMX_PIXEL_UM = 0.120280945        # micrometers per pixel of the CosMx camera (global slide coordinates)
COSMX_CODECLASSES = ("Endogenous", "Negative", "SystemControl", "FalseCode")
_COMPARTMENTS = ("Nuclear", "Cytoplasm", "Membrane", "None")
_SCHEMA = pa.schema([("x", pa.float32()), ("y", pa.float32()), ("z", pa.float32()), ("z_plane", pa.int8()),
                     ("feature", pa.string()), ("sample", pa.string()), ("cell_id", pa.int64()),
                     ("nucleus", pa.bool_()), ("compartment", pa.string()), ("fov", pa.int16()),
                     ("codeclass", pa.string())])


def _codeclass(target: str) -> int:
    """Index into ``COSMX_CODECLASSES`` of a target name.

    Parameters
    ----------
    target : str
        Target name.

    Returns
    -------
    int
        1 for ``Negative*``, 2 for ``SystemControl*``, 3 for ``FalseCode*``, else 0 (``Endogenous``).
    """
    return (1 if target.startswith("Negative") else 2 if target.startswith("SystemControl")
            else 3 if target.startswith("FalseCode") else 0)


def _tx_file(path) -> Path:
    """The transcript file: ``path`` itself, or the first ``*tx_file.csv*`` under the directory ``path``.

    Parameters
    ----------
    path : str or pathlib.Path
        File or directory.

    Returns
    -------
    pathlib.Path
        The transcript file.

    Raises
    ------
    FileNotFoundError
        If it does not exist.
    """
    path = Path(path)
    if path.is_file():
        return path
    return find_file(path, "*tx_file.csv*")


def _cosmx_tables(src: Path, sample: str, px_um: float, z_step_um: float | None, drop_controls: bool,
                  block_mb: int, stats: dict | None):
    """The transcripts of an AtoMx transcript file as tables of the common data model, block by block.

    Parameters
    ----------
    src : pathlib.Path
        The transcript file (``.csv`` or ``.csv.gz``; multi-member gzip files are read whole).
    sample : str
        Value of the ``sample`` column.
    px_um : float
        Micrometers per pixel.
    z_step_um : float or None
        Distance of the z planes (µm); ``z = z_plane * z_step_um``, or NaN without it.
    drop_controls : bool
        Drop the negative, system-control, and false-code probes.
    block_mb : int
        Size of the csv blocks (MB).
    stats : dict or None
        Updated in place with the counts of the QC summary; ``None`` skips them (the readers).

    Yields
    ------
    pyarrow.Table
        The transcripts of one block.
    """
    import pyarrow.csv as pcsv
    conv = pcsv.ConvertOptions(column_types={"fov": pa.int32(), "cell_ID": pa.int64(), "x_global_px": pa.float64(),
                                             "y_global_px": pa.float64(), "z": pa.int32(), "target": pa.string(),
                                             "CellComp": pa.string()},
                               include_columns=["fov", "cell_ID", "x_global_px", "y_global_px", "z", "target",
                                                "CellComp"])
    if stats is not None:
        stats.update(n_in=0, n_out=0, n_incell=0, codeclass=dict.fromkeys(COSMX_CODECLASSES, 0),
                     compartment=dict.fromkeys(_COMPARTMENTS, 0), z_planes={}, feature_counts={}, fovs=set(),
                     bbox=[np.inf, np.inf, -np.inf, -np.inf])
    feature_class: dict[str, int] = {}
    opener = gzip.open if src.suffix == ".gz" else open
    with opener(src, "rb") as fh:
        reader = pcsv.open_csv(fh, read_options=pcsv.ReadOptions(block_size=block_mb << 20), convert_options=conv)
        for batch in reader:
            fov = batch.column("fov").to_numpy()
            cid = batch.column("cell_ID").to_numpy()
            xg = batch.column("x_global_px").to_numpy()
            yg = batch.column("y_global_px").to_numpy()
            zp = batch.column("z").to_numpy()
            target = batch.column("target").dictionary_encode()
            names = target.dictionary.to_pylist()
            for t in names:
                if t not in feature_class:
                    feature_class[t] = _codeclass(t)
            cls = np.array([feature_class[t] for t in names], np.int64)[target.indices.to_numpy()]
            comp_col = batch.column("CellComp")
            comp = comp_col.dictionary_encode()
            unknown = sorted(set(comp.dictionary.to_pylist()) - set(_COMPARTMENTS))
            if unknown:
                raise ValueError(f"unexpected CellComp values {unknown} in {src}")
            keep = cls == 0 if drop_controls else np.ones(len(fov), bool)
            incell = cid != 0
            cell_id = np.where(incell, fov.astype(np.int64) * 100000 + cid, -1)
            table = pa.table({"x": pa.array((xg * px_um).astype(np.float32)),
                              "y": pa.array((yg * px_um).astype(np.float32)),
                              "z": pa.array((zp * z_step_um).astype(np.float32) if z_step_um is not None
                                            else np.full(len(fov), np.nan, np.float32)),
                              "z_plane": pa.array(zp.astype(np.int8)), "feature": batch.column("target"),
                              "sample": pa.array(np.full(len(fov), sample)), "cell_id": pa.array(cell_id),
                              "nucleus": pc.fill_null(pc.equal(comp_col, "Nuclear"), False),
                              "compartment": comp_col, "fov": pa.array(fov.astype(np.int16)),
                              "codeclass": pa.array(np.asarray(COSMX_CODECLASSES, dtype=object)[cls])},
                             schema=_SCHEMA)
            if not keep.all():
                table = table.filter(pa.array(keep))
            if stats is None:
                yield table
                continue
            n = table.num_rows
            stats["n_in"] += len(fov)
            stats["n_out"] += n
            stats["n_incell"] += int((incell & keep).sum())
            stats["fovs"].update(np.unique(fov[keep]).tolist())
            for i, c in enumerate(COSMX_CODECLASSES):
                stats["codeclass"][c] += int(((cls == i) & keep).sum())
            for column, counts in (("compartment", stats["compartment"]), ("feature", stats["feature_counts"])):
                encoded = table.column(column).combine_chunks().dictionary_encode()
                values = np.bincount(encoded.indices.fill_null(-1).to_numpy(zero_copy_only=False) + 1,
                                     minlength=len(encoded.dictionary) + 1)[1:]
                for name, v in zip(encoded.dictionary.to_pylist(), values):
                    if v:
                        counts[name] = counts.get(name, 0) + int(v)
            for z, v in zip(*np.unique(zp[keep], return_counts=True)):
                stats["z_planes"][int(z)] = stats["z_planes"].get(int(z), 0) + int(v)
            if n:
                x, y = table.column("x").to_numpy(), table.column("y").to_numpy()
                b = stats["bbox"]
                b[0], b[1] = min(b[0], float(x.min())), min(b[1], float(y.min()))
                b[2], b[3] = max(b[2], float(x.max())), max(b[3], float(y.max()))
            yield table


def convert_cosmx(rawdir, outdir, sample: str, px_um: float = COSMX_PIXEL_UM, z_step_um: float | None = None,
                  drop_controls: bool = True, block_mb: int = 64, transform=None) -> dict:
    r"""Convert the transcript file of an AtoMx flat-file export to ``outdir/molecules.parquet``.

    Columns: ``x``, ``y`` = ``x_global_px``, ``y_global_px`` times ``px_um`` (µm); ``z_plane`` = ``z`` (plane
    index) and ``z`` = ``z_plane * z_step_um`` (NaN without ``z_step_um``; the plane distance is a setting of the
    run); ``feature`` = ``target``; ``cell_id`` = ``fov * 100000 + cell_ID`` (-1 where ``cell_ID == 0``);
    ``nucleus`` = ``CellComp == "Nuclear"``; ``compartment`` = ``CellComp``; ``fov``; ``codeclass`` from the target
    name (``Negative*``, ``SystemControl*``, ``FalseCode*``, otherwise ``Endogenous``). The rows keep the order of
    the file.

    Parameters
    ----------
    rawdir : str or pathlib.Path
        The transcript file (``*_tx_file.csv`` or ``.csv.gz``) or a directory searched recursively for it.
    outdir : str or pathlib.Path
        Output directory (created); receives ``molecules.parquet`` and ``feature_counts.json``.
    sample : str
        Value of the ``sample`` column.
    px_um : float
        Micrometers per pixel of the global coordinates.
    z_step_um : float or None
        Distance of the z planes (µm).
    drop_controls : bool
        Drop the control probes (codeclasses other than ``Endogenous``), as the converters of the other platforms
        drop theirs; ``False`` keeps them, marked by ``codeclass``.
    block_mb : int
        Size of the csv blocks read at a time (MB).
    transform : callable or None
        Applied to every block (a ``pyarrow.Table`` of the common data model) before it is written, e.g. to add a
        column; it returns the table to write.

    Returns
    -------
    dict
        QC summary: transcripts read and kept, fraction in cells, transcripts per codeclass, compartment, and z
        plane, number of FOVs and features, bounding box, the 20 most abundant features, timing, output path.

    Raises
    ------
    FileNotFoundError
        If no transcript file is found.
    ValueError
        For ``CellComp`` values other than the four compartments.

    See Also
    --------
    read_cosmx : The same transcripts in memory, without a copy on disk.
    """
    src = _tx_file(rawdir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / "molecules.parquet"
    t0 = time.time()
    st: dict = {}
    tables = _cosmx_tables(src, sample, px_um, z_step_um, drop_controls, block_mb, st)
    write_molecules(tables if transform is None else (transform(t) for t in tables), dest)
    n_out, fc = st["n_out"], st["feature_counts"]
    xmin, ymin, xmax, ymax = st["bbox"]
    area = (xmax - xmin) * (ymax - ymin) / 1e6 if n_out else 0.0
    with open(outdir / "feature_counts.json", "w") as f:
        json.dump(fc, f)
    return dict(sample=sample, platform="cosmx", source=str(src), px_um=px_um, z_step_um=z_step_um,
                molecules_in=st["n_in"], molecules=n_out, frac_in_cells=(st["n_incell"] / n_out if n_out else None),
                codeclass=st["codeclass"], compartment=st["compartment"],
                molecules_per_z_plane={str(k): v for k, v in sorted(st["z_planes"].items())},
                n_fov=len(st["fovs"]), n_features=len(fc), bbox_um=[xmin, ymin, xmax, ymax], bbox_area_mm2=area,
                top_features=sorted(fc.items(), key=lambda kv: -kv[1])[:20], seconds=time.time() - t0,
                output=str(dest))


def read_cosmx(rawdir, sample: str = "", px_um: float = COSMX_PIXEL_UM, z_step_um: float | None = None,
               drop_controls: bool = True, columns=None, block_mb: int = 64) -> pd.DataFrame:
    r"""Read the transcripts of an AtoMx flat-file export into memory, in the form :func:`scote.twopoint` takes.

    The transcripts and columns of :func:`scote.io.convert_cosmx`, without writing them to disk.

    Parameters
    ----------
    rawdir : str or pathlib.Path
        The transcript file (``*_tx_file.csv`` or ``.csv.gz``) or a directory searched recursively for it.
    sample : str
        Value of the ``sample`` column.
    px_um : float
        Micrometers per pixel of the global coordinates.
    z_step_um : float or None
        Distance of the z planes (µm).
    drop_controls : bool
        Drop the control probes, as the readers of the other platforms drop theirs; ``False`` keeps them (they then
        enter :math:`n_\mathrm{tot}` and :math:`P_\mathrm{tot}` unless ``select`` leaves them out).
    columns : sequence of str or None
        Columns to keep (``x``, ``y``, and ``feature`` always); ``None`` keeps all.
    block_mb : int
        Size of the csv blocks read at a time (MB).

    Returns
    -------
    pandas.DataFrame
        One row per transcript with the columns of :func:`scote.io.convert_cosmx`.

    Raises
    ------
    FileNotFoundError
        If no transcript file is found.
    """
    src = _tx_file(rawdir)
    return molecule_frame(_cosmx_tables(src, sample, px_um, z_step_um, drop_controls, block_mb, None), columns)
