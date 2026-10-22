#!/usr/bin/env python
"""Extract selected members (glob patterns) from zip / tar.gz archives into --outdir, flattening leading
directories that are common to all extracted members. Non-archive inputs (e.g. .h5) are hard-linked/copied.

Usage: extract_members.py --archive A.zip [B.tar.gz ...] --outdir DIR --members PATTERN [PATTERN ...] [--delete]
"""
import argparse, fnmatch, os, shutil, tarfile, zipfile, json, time
from pathlib import Path


def match(name, patterns):
    return any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(name, "*/" + p) for p in patterns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--delete", action="store_true", help="delete archives after successful extraction")
    ap.add_argument("True", nargs="?", help=argparse.SUPPRESS)   # tolerate Snakemake passing the boolean
    ap.add_argument("False", nargs="?", help=argparse.SUPPRESS)
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for arc in a.archive:
        arc = Path(arc); t0 = time.time()
        if zipfile.is_zipfile(arc):
            with zipfile.ZipFile(arc) as z:
                names = [n for n in z.namelist() if not n.endswith("/")]
                sel = [n for n in names if match(n, a.members)]
                for n in sel:
                    dest = out / Path(n).name if len({Path(s).name for s in sel}) == len(sel) else out / n
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(n) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst, 16 * 2**20)
                    manifest.append(dict(archive=arc.name, member=n, dest=str(dest), bytes=dest.stat().st_size))
                print(f"{arc.name}: {len(names)} members, extracted {len(sel)} in {time.time()-t0:.0f} s")
                skipped = [n for n in names if n not in sel]
                print("  not extracted (first 30):", skipped[:30])
        elif tarfile.is_tarfile(arc):
            with tarfile.open(arc) as t:
                sel = [m for m in t.getmembers() if m.isfile() and match(m.name, a.members)]
                for m in sel:
                    dest = out / m.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with t.extractfile(m) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst, 16 * 2**20)
                    manifest.append(dict(archive=arc.name, member=m.name, dest=str(dest), bytes=dest.stat().st_size))
                print(f"{arc.name}: extracted {len(sel)} members in {time.time()-t0:.0f} s")
        else:
            # non-archive input (e.g. .h5, or a csv placed/symlinked under data/archives/): hard-link the *target*
            # (os.link on a relative symlink links the symlink itself -> dangling in outdir)
            dest = out / arc.name
            if dest.is_symlink() and not dest.exists():
                dest.unlink()
            if not dest.exists():
                real = os.path.realpath(arc)
                try: os.link(real, dest)
                except OSError: shutil.copy2(real, dest)
            manifest.append(dict(archive=arc.name, member=None, dest=str(dest), bytes=dest.stat().st_size))
            print(f"{arc.name}: not an archive, linked")
    with open(out / "extract_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    if a.delete:
        for arc in a.archive:
            if zipfile.is_zipfile(arc) or tarfile.is_tarfile(arc):
                os.remove(arc); print("deleted", arc)


if __name__ == "__main__":
    main()
