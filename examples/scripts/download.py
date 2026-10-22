"""Download the files of one data set into its archive directory (rule `download` of workflows/fetch.smk).

    python scripts/download.py --outdir DIR --size-gb N NAME=URL [NAME=URL ...]
    python scripts/download.py --outdir DIR --manual --archive NAME --page URL

Every download is guarded by scripts/disk_guard.sh (the free-space floor comes from MIN_FREE_FRAC and MIN_FREE_GB) and
resumes (curl -C -). A data set without a direct link (--manual) is only checked: the archive must be in place,
otherwise the script stops with a message that points to the vendor page. Messages go to stdout, errors to stderr.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", required=True, help="archive directory of the data set")
    ap.add_argument("--size-gb", type=float, default=0.0, help="download size, for the free-space check")
    ap.add_argument("--manual", action="store_true", help="no direct link: check that the archive is in place")
    ap.add_argument("--archive", help="file name of the archive (with --manual)")
    ap.add_argument("--page", help="vendor landing page (with --manual)")
    ap.add_argument("files", nargs="*", metavar="NAME=URL", help="file name in the archive directory and its URL")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.manual:
        dest = outdir / args.archive
        if not dest.exists():
            sys.exit(f"no direct download: obtain the data from the vendor page {args.page} and place it as {dest} "
                     "(see README.md, 'Data sources'), then run the workflow again.")
        print(f"{time.strftime('%F %T')} {dest} is in place", flush=True)
        return
    subprocess.run(["scripts/disk_guard.sh", str(outdir), str(args.size_gb)], check=True)
    for item in args.files:
        name, url = item.split("=", 1)
        dest = outdir / name
        if dest.exists():
            continue
        t0 = time.time()
        cmd = ["curl", "-L", "--fail", "--no-progress-meter", "--retry", "5", "--retry-delay", "30", "-C", "-",
               "-o", f"{dest}.part", url]
        print(f"{time.strftime('%F %T')} {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        Path(f"{dest}.part").rename(dest)
        print(f"{time.strftime('%F %T')} done {name}: {dest.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f} s",
              flush=True)


if __name__ == "__main__":
    main()
