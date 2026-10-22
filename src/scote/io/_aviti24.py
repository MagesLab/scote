"""Element Biosciences AVITI24 (Cytos) cytoprofiling runs: the SpatialData archive of a run -> molecule tables of the
common data model (one per well), on disk or in memory.

A cytoprofiling run delivers a SpatialData object of the run, ``<run>.zarr.zip`` (zarr v3, stored without
compression in the zip; Element's run-output documentation lists it as the spatial data object for downstream
analysis). It is read from the archive without extracting it. Its points elements, one per well and batch, named
``<well>-points-<batch>``, are parquet directories inside the zip; only the columns used here are read. The run
metadata (root attribute ``run_metadata``) give the pixel size (``run_parameters.ImageInfo.PixelSizeUm``), the
focus offsets of the z planes (``run_parameters.ZPositions``, read as µm), and the label of every well.

Columns used: pixel coordinates ``x``, ``y`` in the well frame and the z-plane index ``z``; ``target_name``;
``label_Cell_global`` (0 = not in a cell); ``is_nuclear``; ``MapQ``; ``tile_id``; and ``IsCounted`` where a batch
has it (the 3' transcriptome batches). Kept molecules: ``IsCounted`` in batches that have the column; in the other
batches (targeted panels, e.g. guide RNAs), all molecules except ``target_name == "Unassigned"``. The per-tile
``rawreads.parquet`` and ``reads.parquet`` files of the run folder are not read: their columns are not described in
the public documentation.
"""
from __future__ import annotations

import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ._schema import molecule_frame, write_molecules

__all__ = ["AVITI24_SCHEMA_DOC", "aviti24_run_parameters", "aviti24_wells", "convert_aviti24", "read_aviti24"]

AVITI24_SCHEMA_DOC = (
    "x,y,z µm float32 (z = ZPositions[z_plane]); z_plane int8; feature dictionary<string>; "
    "sample dictionary<string>; cell_id int64 (-1 unassigned); nucleus bool; qv float32 (MapQ); "
    "batch/condition/fov dictionary<string>"
)
UNASSIGNED = "Unassigned"
_PART_RE = re.compile(r"part\.(\d+)\.parquet$")
_COLUMNS = ["x", "y", "z", "target_name", "label_Cell_global", "is_nuclear", "MapQ", "tile_id"]


def _zarr_root(zf: zipfile.ZipFile) -> str:
    """Name of the zarr root group inside the archive.

    Parameters
    ----------
    zf : zipfile.ZipFile
        The opened archive.

    Returns
    -------
    str
        The single top-level directory that holds a ``zarr.json``.

    Raises
    ------
    ValueError
        If there is not exactly one such directory.
    """
    roots = sorted({n.split("/")[0] for n in zf.namelist() if n.endswith("/zarr.json") and n.count("/") == 1})
    if len(roots) != 1:
        raise ValueError(f"expected one top-level .zarr group in the archive, found {len(roots)}")
    return roots[0]


def _elements(zf: zipfile.ZipFile) -> dict[str, list[str]]:
    """The points elements of every well, batches in run order.

    Parameters
    ----------
    zf : zipfile.ZipFile
        The opened archive.

    Returns
    -------
    dict
        Well name -> element names.
    """
    out: dict[str, list[str]] = {}
    root = _zarr_root(zf)
    for name in {n.split("/")[2] for n in zf.namelist() if n.startswith(f"{root}/points/") and len(n.split("/")) > 3}:
        out.setdefault(name.split("-points-")[0], []).append(name)
    return {w: sorted(v, key=lambda e: e.split("-points-")[1]) for w, v in sorted(out.items())}


def _parts(zf: zipfile.ZipFile, element: str) -> list[str]:
    """Archive members of the parquet parts of one points element, in part order.

    Parameters
    ----------
    zf : zipfile.ZipFile
        The opened archive.
    element : str
        Element name.

    Returns
    -------
    list of str
        Member names.
    """
    prefix = f"{_zarr_root(zf)}/points/{element}/points.parquet/"
    return sorted((n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".parquet")),
                  key=lambda n: int(_PART_RE.search(n).group(1)))


def _run_metadata(zf: zipfile.ZipFile) -> dict:
    """The ``run_metadata`` attribute of the root group.

    Parameters
    ----------
    zf : zipfile.ZipFile
        The opened archive.

    Returns
    -------
    dict
        The run metadata (empty if absent).
    """
    attrs = json.loads(zf.read(f"{_zarr_root(zf)}/zarr.json")).get("attributes", {})
    rm = attrs.get("run_metadata", {})
    return json.loads(rm) if isinstance(rm, str) else rm


def aviti24_run_parameters(zip_path) -> dict:
    """Pixel size and z positions of a run.

    Parameters
    ----------
    zip_path : str or pathlib.Path
        The ``.zarr.zip`` archive.

    Returns
    -------
    dict
        ``pixel_size_um`` (``run_parameters.ImageInfo.PixelSizeUm``) and ``z_positions_um``
        (``run_parameters.ZPositions``, one per z plane); ``None`` where the metadata lack them.
    """
    with zipfile.ZipFile(Path(zip_path)) as zf:
        rp = _run_metadata(zf).get("run_parameters", {})
    pixel = rp.get("ImageInfo", {}).get("PixelSizeUm")
    z = rp.get("ZPositions")
    return dict(pixel_size_um=None if pixel is None else float(pixel),
                z_positions_um=None if z is None else [float(v) for v in z])


def aviti24_wells(zip_path) -> dict[str, dict]:
    """Wells of a run with their labels and batches.

    Parameters
    ----------
    zip_path : str or pathlib.Path
        The ``.zarr.zip`` archive.

    Returns
    -------
    dict
        Well name -> dict(condition=well label of the run metadata, batches=list of str in run order).
    """
    with zipfile.ZipFile(Path(zip_path)) as zf:
        cond = {w["well_name"]: w["well_label"] for w in _run_metadata(zf).get("well_metadata", [])}
        return {well: dict(condition=cond.get(well, ""), batches=[e.split("-points-")[1] for e in elements])
                for well, elements in _elements(zf).items()}


def _part_table(t: pa.Table, sample: str, condition: str, batch: str, pixel_um: float, z_um: np.ndarray):
    """One parquet part -> the common data model, filtered.

    Parameters
    ----------
    t : pyarrow.Table
        The columns of ``_COLUMNS`` (and ``IsCounted`` where the batch has it).
    sample, condition, batch : str
        Values of the ``sample``, ``condition``, and ``batch`` columns.
    pixel_um : float
        Micrometers per pixel.
    z_um : numpy.ndarray
        z position (µm) of every plane index.

    Returns
    -------
    pyarrow.Table or None
        The kept molecules, or ``None`` if none is kept.
    """
    if "IsCounted" in t.column_names:
        keep = pc.fill_null(t.column("IsCounted"), False)
    else:
        keep = pc.not_equal(t.column("target_name"), UNASSIGNED)
    t = t.filter(keep)
    n = t.num_rows
    if n == 0:
        return None
    x = t.column("x").to_numpy(zero_copy_only=False).astype(np.float32) * pixel_um
    y = t.column("y").to_numpy(zero_copy_only=False).astype(np.float32) * pixel_um
    plane = t.column("z").to_numpy(zero_copy_only=False).astype(np.int8)
    z = z_um[plane].astype(np.float32)
    cid = t.column("label_Cell_global").to_numpy(zero_copy_only=False).astype(np.int64)
    cid[cid == 0] = -1
    nuc = t.column("is_nuclear").to_numpy(zero_copy_only=False).astype(bool)
    qv = pc.fill_null(t.column("MapQ"), -1).to_numpy(zero_copy_only=False).astype(np.float32)
    return pa.table({
        "x": pa.array(x), "y": pa.array(y), "z": pa.array(z), "z_plane": pa.array(plane),
        "feature": t.column("target_name").cast(pa.string()).dictionary_encode(),
        "sample": pa.array([sample] * n, type=pa.string()).dictionary_encode(),
        "cell_id": pa.array(cid), "nucleus": pa.array(nuc), "qv": pa.array(qv),
        "batch": pa.array([batch] * n, type=pa.string()).dictionary_encode(),
        "condition": pa.array([condition] * n, type=pa.string()).dictionary_encode(),
        "fov": t.column("tile_id").cast(pa.string()).dictionary_encode(),
    })


def _aviti24_tables(zip_path: Path, well: str, sample: str, batches, flush_rows: int, pixel_um,
                    stats: dict | None):
    """The molecules of one well as tables of the common data model; no table spans two batches.

    Parameters
    ----------
    zip_path : pathlib.Path
        The archive.
    well : str
        Well name.
    sample : str
        Value of the ``sample`` column.
    batches : list of str or None
        Batches to read; ``None`` reads all.
    flush_rows : int
        Molecules buffered per table.
    pixel_um : float or None
        Micrometers per pixel; ``None`` takes the run metadata.
    stats : dict or None
        Updated in place with the counts of the QC summary; ``None`` skips them (the readers).

    Yields
    ------
    pyarrow.Table
        Kept molecules.

    Raises
    ------
    ValueError
        If the well is not in the archive, or the pixel size or the z positions are neither given nor in the run
        metadata.
    """
    params = aviti24_run_parameters(zip_path)
    pixel_um = params["pixel_size_um"] if pixel_um is None else float(pixel_um)
    if pixel_um is None or params["z_positions_um"] is None:
        raise ValueError("the run metadata lack run_parameters.ImageInfo.PixelSizeUm or ZPositions")
    z_um = np.asarray(params["z_positions_um"], dtype=np.float64)
    condition = aviti24_wells(zip_path).get(well, {}).get("condition", "")
    if stats is not None:
        stats.update(n_in=0, n_out=0, n_incell=0, n_nuc=0, per_batch={}, feature_counts={}, pixel_um=pixel_um,
                     z_um=z_um.tolist(), condition=condition, bbox=[np.inf, np.inf, -np.inf, -np.inf])
    with zipfile.ZipFile(zip_path) as zf:
        elements = _elements(zf).get(well)
        if not elements:
            raise ValueError(f"well {well!r} not in {sorted(_elements(zf))}")
        for element in elements:
            batch = element.split("-points-")[1]
            if batches and batch not in batches:
                continue
            buf, buf_rows, b_in, b_out = [], 0, 0, 0
            for part in _parts(zf, element):
                with zf.open(part) as fh:
                    f = pq.ParquetFile(fh)
                    cols = _COLUMNS + (["IsCounted"] if "IsCounted" in f.schema_arrow.names else [])
                    t = f.read(columns=cols)
                b_in += t.num_rows
                out = _part_table(t, sample, condition, batch, pixel_um, z_um)
                if out is None:
                    continue
                b_out += out.num_rows
                if stats is not None:
                    _part_stats(out, stats)
                buf.append(out)
                buf_rows += out.num_rows
                if buf_rows >= flush_rows:
                    yield pa.concat_tables(buf, promote_options="default").combine_chunks()
                    buf, buf_rows = [], 0
            if buf:
                yield pa.concat_tables(buf, promote_options="default").combine_chunks()
            if stats is not None:
                stats["per_batch"][batch] = {"molecules_in": b_in, "molecules": b_out,
                                             "dropped_frac": round(1 - b_out / b_in, 4) if b_in else None}
                stats["n_in"] += b_in
                stats["n_out"] += b_out


def _part_stats(out: pa.Table, stats: dict) -> None:
    """Add the kept molecules of one part to the QC counts.

    Parameters
    ----------
    out : pyarrow.Table
        The kept molecules (``_part_table``).
    stats : dict
        The QC counts, updated in place.
    """
    b = stats["bbox"]
    x, y = out.column("x").to_numpy(), out.column("y").to_numpy()
    b[0], b[2] = min(b[0], float(x.min())), max(b[2], float(x.max()))
    b[1], b[3] = min(b[1], float(y.min())), max(b[3], float(y.max()))
    stats["n_incell"] += int((out.column("cell_id").to_numpy() >= 0).sum())
    stats["n_nuc"] += int(out.column("nucleus").to_numpy(zero_copy_only=False).sum())
    u, c = np.unique(np.asarray(out.column("feature").combine_chunks().dictionary_decode()), return_counts=True)
    for uu, cc in zip(u, c):
        stats["feature_counts"][str(uu)] = stats["feature_counts"].get(str(uu), 0) + int(cc)


def convert_aviti24(zip_path, outdir, well: str, sample: str | None = None, batches: list[str] | None = None,
                    pixel_um: float | None = None, flush_rows: int = 4_000_000) -> dict:
    """Convert one well of an AVITI24 run archive to ``outdir/molecules.parquet``.

    Columns: ``x``, ``y`` (µm, pixels times the pixel size), ``z`` (the focus offset of the plane, µm), ``z_plane``,
    ``feature`` (``target_name``), ``sample``, ``cell_id`` (``label_Cell_global``, -1 outside cells), ``nucleus``
    (``is_nuclear``), ``qv`` (``MapQ``, -1 where missing), ``batch``, ``condition`` (the well label), ``fov``
    (``tile_id``). Row groups never span two batches, so a reader can select batches cheaply.

    Parameters
    ----------
    zip_path : str or pathlib.Path
        The ``<run>.zarr.zip`` archive.
    outdir : str or pathlib.Path
        Output directory (created); receives ``molecules.parquet`` and ``feature_counts.json``.
    well : str
        Well name (:func:`scote.io.aviti24_wells`).
    sample : str or None
        Value of the ``sample`` column; ``None`` gives ``aviti24_<well>``.
    batches : list of str or None
        Batches to convert (e.g. ``["B02", "B03"]``); ``None`` converts all.
    pixel_um : float or None
        Micrometers per pixel; ``None`` takes ``run_parameters.ImageInfo.PixelSizeUm`` of the run metadata.
    flush_rows : int
        Molecules buffered before a row group is written.

    Returns
    -------
    dict
        QC summary: molecules read and kept (also per batch), fractions in cells and nuclei, number of features,
        bounding box and density, the 20 most abundant features, pixel size and z positions, timing, output path and
        size.

    Raises
    ------
    ValueError
        If the well is not in the archive or the run metadata lack the pixel size or the z positions.

    See Also
    --------
    read_aviti24 : The same molecules in memory, without a copy on disk.
    """
    zip_path, outdir = Path(zip_path), Path(outdir)
    sample = sample or f"aviti24_{well}"
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / "molecules.parquet"
    t0 = time.time()
    st: dict = {}
    write_molecules(_aviti24_tables(zip_path, well, sample, batches, flush_rows, pixel_um, st), dest)
    n_in, n_out, fc = st["n_in"], st["n_out"], st["feature_counts"]
    xmin, ymin, xmax, ymax = st["bbox"]
    area = (xmax - xmin) * (ymax - ymin) / 1e6 if n_out else 0.0
    z_um = st["z_um"]
    with open(outdir / "feature_counts.json", "w") as f:
        json.dump(fc, f)
    return dict(sample=sample, platform="aviti24", well=well, condition=st["condition"], source=str(zip_path),
                molecules_in=n_in, molecules=n_out, dropped=n_in - n_out,
                dropped_frac=round((n_in - n_out) / n_in, 4) if n_in else None, per_batch=st["per_batch"],
                frac_in_cells=(st["n_incell"] / n_out if n_out else None),
                frac_in_nucleus=(st["n_nuc"] / n_out if n_out else None), n_features=len(fc),
                features_ge_500=sum(1 for v in fc.values() if v >= 500), bbox_um=[xmin, ymin, xmax, ymax],
                bbox_area_mm2=area, molecules_per_mm2_bbox=(n_out / area if area else None),
                top_features=sorted(fc.items(), key=lambda kv: -kv[1])[:20], pixel_size_um=st["pixel_um"],
                z_planes=len(z_um), z_um_of_plane0=z_um[0],
                z_step_um=(z_um[0] - z_um[1]) if len(z_um) > 1 else None, seconds=time.time() - t0,
                output=str(dest), output_bytes=dest.stat().st_size if dest.exists() else 0,
                schema=AVITI24_SCHEMA_DOC)


def read_aviti24(zip_path, well: str, sample: str | None = None, batches: list[str] | None = None,
                 pixel_um: float | None = None, columns=None) -> pd.DataFrame:
    """Read the molecules of one well of an AVITI24 run archive into memory, in the form :func:`scote.twopoint`
    takes.

    The molecules and columns of :func:`scote.io.convert_aviti24`, without writing them to disk.

    Parameters
    ----------
    zip_path : str or pathlib.Path
        The ``<run>.zarr.zip`` archive.
    well : str
        Well name (:func:`scote.io.aviti24_wells`).
    sample : str or None
        Value of the ``sample`` column; ``None`` gives ``aviti24_<well>``.
    batches : list of str or None
        Batches to read; ``None`` reads all.
    pixel_um : float or None
        Micrometers per pixel; ``None`` takes the run metadata.
    columns : sequence of str or None
        Columns to keep (``x``, ``y``, and ``feature`` always); ``None`` keeps all.

    Returns
    -------
    pandas.DataFrame
        One row per molecule with the columns of :func:`scote.io.convert_aviti24`.

    Raises
    ------
    ValueError
        If the well is not in the archive or the run metadata lack the pixel size or the z positions.
    """
    return molecule_frame(_aviti24_tables(Path(zip_path), well, sample or f"aviti24_{well}", batches, 4_000_000,
                                          pixel_um, None), columns)
