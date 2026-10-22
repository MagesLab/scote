"""Command-line interface: ``python -m scote <command>``.

The library is the main interface (:func:`scote.twopoint`, :mod:`scote.io`); these commands wrap it for workflows that
run one step per process:

* ``convert``: vendor output -> common data model (:mod:`scote.io` converters), with a QC JSON;
* ``twopoint``: a molecule or bin table of the common data model -> two-point functions with their closed-form
  tile-bootstrap uncertainties (:func:`scote.io.read_molecules` or :func:`scote.io.read_bins`, then
  :func:`scote.twopoint`, then :meth:`scote.TwoPointResult.save`).

Run ``python -m scote <command> --help`` for the options.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

__all__ = ["main"]


def _convert(a) -> int:
    """Run the ``convert`` command.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    int
        Exit status.
    """
    from . import io
    if a.platform == "aviti24" and a.list_wells:
        print(json.dumps(io.aviti24_wells(a.rawdir), indent=1))
        return 0
    if not (a.outdir and a.qc):
        raise SystemExit("--outdir and --qc are required to convert")
    sample = a.sample or Path(a.outdir).resolve().name
    if a.platform == "aviti24":
        if not a.well:
            raise SystemExit("--platform aviti24 needs --well (list them with --list-wells)")
        batches = [b.strip() for b in a.batches.split(",") if b.strip()] or None
        qc = io.convert_aviti24(a.rawdir, a.outdir, a.well, sample, batches)
    elif a.platform == "cosmx":
        qc = io.convert_cosmx(a.rawdir, a.outdir, sample, z_step_um=a.z_step_um, drop_controls=not a.keep_controls)
    elif a.platform == "stereoseq":
        qc = io.convert_stereoseq(a.rawdir, a.outdir, sample, bin_dnb=a.bin_dnb)
    else:
        converters = dict(xenium=io.convert_xenium, visium_hd=io.convert_visium_hd, visium=io.convert_visium,
                          merscope=io.convert_merscope, atera=io.convert_atera)
        qc = converters[a.platform](a.rawdir, a.outdir, sample)
    Path(a.qc).parent.mkdir(parents=True, exist_ok=True)
    with open(a.qc, "w") as f:
        json.dump(qc, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in qc.items() if k != "top_features"}, indent=1, default=str)[:3000])
    return 0


def _twopoint(a) -> int:
    """Run the ``twopoint`` command.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    int
        Exit status.
    """
    from . import io
    from ._twopoint.pipeline import twopoint
    t0 = time.time()
    to_npz = str(a.out).endswith(".npz")
    run = dict(shells_per_level=a.shells_per_level, tile_um=a.tile_um, backend=a.backend, threads=a.threads or None,
               acc_dtype=a.acc_dtype, verbose=True, keep_tiles=to_npz and not a.no_store_tiles,
               out=None if to_npz else a.out, dtype=a.dtype, n_levels=a.n_levels, delta=a.delta)
    if a.molecules:
        df = io.read_molecules(a.molecules, a.select, a.max_molecules, columns=[])
        n_units = len(df)
        names = list(df["feature"].cat.categories)
        print(f"loaded {n_units:,} molecules of {len(names)} features in {time.time() - t0:.1f}s", flush=True)
        res = twopoint(df[["x", "y"]].to_numpy(np.float64), df["feature"], lattice_only=a.lattice_only, **run)
        unit = f"molecules, {'lattice' if a.lattice_only else 'enumeration'}, delta={res.params['delta']:g} µm"
    else:
        if a.lattice_only:
            raise SystemExit("--lattice-only applies to --molecules (bins are always counted on lattices)")
        coords, X, names, bin_um = io.read_bins(a.bins, a.select)
        n_units = int(X.shape[0])
        print(f"loaded {n_units:,} bins x {len(names)} features ({X.sum():.3g} counts) in {time.time() - t0:.1f}s",
              flush=True)
        res = twopoint(coords, X, names=names, bin_size=bin_um, **run)
        unit = f"bins {bin_um:g} µm, delta={res.params['delta']:g} µm"
    res.params.update(input_file=str(a.molecules or a.bins), unit=unit, n_units=n_units,
                      feature_selection=a.select, max_molecules=a.max_molecules,
                      median_sd_log_g=float(np.nanmedian(np.asarray(res.sd))),
                      flag_fractions=[float((np.asarray(res.flag) == f).mean()) for f in (0, 1, 2)])
    tw = time.perf_counter()
    res.save(a.out)
    res.timings["save"] = time.perf_counter() - tw
    res.timings["total_cli"] = time.time() - t0
    res.write_meta(a.out)
    print(json.dumps({k: v for k, v in res.meta().items() if k != "timings"}, indent=1, default=str))
    return 0


def _backend(spec: str):
    """The ``backend`` of a ``--backend`` string.

    Parameters
    ----------
    spec : str
        ``cpu`` or ``gpu``, optionally followed by comma-separated ``KEY=VALUE`` fields of :class:`scote.CPUOptions`
        or :class:`scote.GPUOptions` (``cpu,mode=sparse``, ``gpu,device=cuda:1``); the values are converted to the
        types of the fields.

    Returns
    -------
    str, CPUOptions, or GPUOptions
        The name without fields, otherwise the options.

    Raises
    ------
    argparse.ArgumentTypeError
        For an unknown backend, a field without ``=``, an unknown field, or an invalid value.
    """
    import dataclasses
    from ._twopoint._common import CPUOptions, GPUOptions
    name, *pairs = (p.strip() for p in spec.split(","))
    if name not in ("cpu", "gpu"):
        raise argparse.ArgumentTypeError(f"{spec!r} does not start with 'cpu' or 'gpu'")
    pairs = [p for p in pairs if p]
    if not pairs:
        return name
    cls = GPUOptions if name == "gpu" else CPUOptions
    defaults = {f.name: getattr(cls(), f.name) for f in dataclasses.fields(cls)}
    fields = {}
    for pair in pairs:
        key, sep, value = (x.strip() for x in pair.partition("="))
        if not sep or key not in defaults:
            raise argparse.ArgumentTypeError(f"{pair!r} is not KEY=VALUE with KEY one of {sorted(defaults)}")
        kind = type(defaults[key])
        try:
            fields[key] = (value.lower() in ("1", "true", "yes")) if kind is bool else kind(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{key} needs a {kind.__name__}, got {value!r}") from None
    try:
        return cls(**fields)
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err)) from None


def _parser() -> argparse.ArgumentParser:
    """The argument parser of the command line.

    Returns
    -------
    argparse.ArgumentParser
        The parser with the ``convert`` and ``twopoint`` commands; the defaults are those of the library functions
        (except ``--out``, which the command needs).
    """
    ap = argparse.ArgumentParser(prog="python -m scote", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="vendor output -> common data model")
    c.add_argument("--platform", required=True,
                   choices=["xenium", "merscope", "cosmx", "atera", "aviti24", "visium_hd", "visium", "stereoseq"])
    c.add_argument("--rawdir", required=True,
                   help="vendor output directory (searched recursively); cosmx: also the transcript file; aviti24: "
                        "the run archive <run>.zarr.zip; stereoseq: the GEM or square-bin GEF file")
    c.add_argument("--outdir", default="", help="output directory")
    c.add_argument("--sample", default=None, help="sample name (default: the name of the output directory)")
    c.add_argument("--qc", default="", help="QC JSON to write")
    c.add_argument("--well", default="", help="aviti24: the well to convert")
    c.add_argument("--batches", default="", help="aviti24: comma-separated batches (default: all)")
    c.add_argument("--list-wells", action="store_true", help="aviti24: print the wells and batches, then exit")
    c.add_argument("--z-step-um", type=float, default=None, help="cosmx: distance of the z planes (µm)")
    c.add_argument("--keep-controls", action="store_true",
                   help="cosmx: keep the control probes (dropped by default, as on the other platforms)")
    c.add_argument("--bin-dnb", type=int, default=50, help="stereoseq: bin side in DNBs")

    t = sub.add_parser("twopoint", help="two-point functions of a molecule or bin table")
    src = t.add_mutually_exclusive_group(required=True)
    src.add_argument("--molecules", default=None, help="molecule table (parquet, common data model)")
    src.add_argument("--bins", default=None, help="bin table (npz, common data model)")
    t.add_argument("--lattice-only", action="store_true",
                   help="molecules: count the shortest distances on a lattice of pitch --delta instead of "
                        "enumerating the pairs")
    t.add_argument("--select", default=None,
                   help="'top:N' most abundant, 'min:N' at least N counts, 'file:PATH' one name per line, or a "
                        "comma-separated list (default: all features)")
    t.add_argument("--delta", type=float, default=None,
                   help="finest distance resolution (µm): bin width of the enumeration, or the pitch of the first "
                        "lattice (default: 0.5 for molecules, the bin size of the table for bins)")
    t.add_argument("--shells-per-level", type=int, default=8, help="distance bins per level (even)")
    t.add_argument("--n-levels", type=int, default=None,
                   help="levels after level 0 (default: the smallest number whose bins reach 16.4 mm, as the "
                        "library)")
    t.add_argument("--tile-um", type=float, default=500.0, help="tile side of the block bootstrap (µm)")
    t.add_argument("--max-molecules", type=int, default=0, help="random subsample of the molecules (0 = all)")
    t.add_argument("--threads", type=int, default=0,
                   help="CPU threads of numba, BLAS, and the PyTorch CPU device (0 = the library default, "
                        "scote.default_threads())")
    t.add_argument("--backend", type=_backend, default="cpu", metavar="{cpu,gpu}[,KEY=VALUE...]",
                   help="where the lattice products and accumulators run, optionally with fields of scote.CPUOptions "
                        "(e.g. 'cpu,mode=sparse') or scote.GPUOptions (e.g. 'gpu,device=cuda:1'); default: cpu")
    t.add_argument("--dtype", choices=["float64", "float32"], default="float64",
                   help="dtype of log_g, sd, and n_eff_pairs (P is always float64)")
    t.add_argument("--acc-dtype", choices=["float64", "float32"], default="float64",
                   help="dtype of the bootstrap accumulators")
    t.add_argument("--out", required=True,
                   help="'x.npz': one archive (with the per-tile counts unless --no-store-tiles) plus x.json; any "
                        "other path: a directory that the results stream into")
    t.add_argument("--no-store-tiles", action="store_true", help="npz output without the per-tile counts")
    return ap


def main(argv=None) -> int:
    """Run the command line.

    Parameters
    ----------
    argv : list of str or None
        Arguments without the program name; ``None`` uses ``sys.argv``.

    Returns
    -------
    int
        Exit status.
    """
    a = _parser().parse_args(argv)
    if a.cmd == "convert":
        return _convert(a)
    return _twopoint(a)
