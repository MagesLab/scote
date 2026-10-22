# Data staging: download the public data sets of config/datasets.yaml, extract the needed archive members, convert every
# sample to the common data model of the `scote` package (molecules: parquet with x, y in um and a feature column; bins:
# sparse count matrix plus bin coordinates), and write a QC summary per sample to results/qc/<dataset>.json.
#
#   snakemake --configfile config/datasets.yaml config/host.yaml -j 3 download_all
#   snakemake --configfile config/datasets.yaml config/host.yaml -j 2 convert_all
#
# Every download is guarded by scripts/disk_guard.sh (never below `min_free_frac` free). Downloads resume (curl -C -).
# Data sets marked `manual_download: true` (Vizgen liver, Element AVITI24) have no direct link: their download rule checks
# that the file is in place and otherwise stops with a message that points to the vendor landing page.
from pathlib import Path

DATA = Path("data")
DS = {k: v for k, v in config["datasets"].items() if not v.get("blocked")}
MINFREE = config.get("min_free_frac", 0.05)
MINFREE_GB = config.get("min_free_gb", 0)
CONVERT = [k for k, v in DS.items() if not v.get("no_extract")]

localrules: download_all, convert_all

rule download_all:
    input: expand(str(DATA / "archives/{ds}/.downloaded"), ds=list(DS))

rule convert_all:
    input: expand("results/qc/{ds}.json", ds=CONVERT)

def _download_args(ds):
    d = DS[ds]
    if d.get("manual_download"):
        return ["--manual", "--archive", d["archive"], "--page", d["url"]]
    urls = {d["archive"]: d["url"], **d.get("extra_urls", {})}
    return ["--size-gb", str(d["size_gb"])] + [f"{name}={url}" for name, url in urls.items()]

rule download:
    output: touch(str(DATA / "archives/{ds}/.downloaded"))
    params:
        outdir=lambda w: str(DATA / "archives" / w.ds),
        args=lambda w: _download_args(w.ds),
    log: "logs/fetch/download_{ds}.log"
    conda: ENV_SCOTE
    threads: 1
    resources: mem_mb=500, runtime=720, download=1
    retries: 2
    shell:
        "MIN_FREE_FRAC={MINFREE} MIN_FREE_GB={MINFREE_GB} python scripts/download.py --outdir {params.outdir} "
        "{params.args:q} >> {log} 2>&1"

rule extract:
    input: str(DATA / "archives/{ds}/.downloaded")
    output: touch(str(DATA / "raw/{ds}/.extracted"))
    params:
        members=lambda w: DS[w.ds]["members"],
        archive=lambda w: str(DATA / "archives" / w.ds / DS[w.ds]["archive"]),
        extras=lambda w: [str(DATA / "archives" / w.ds / f) for f in DS[w.ds].get("extra_urls", {})],
        outdir=lambda w: str(DATA / "raw" / w.ds),
        delete=config.get("delete_archives", False),
        size=lambda w: DS[w.ds]["size_gb"],
    log: "logs/fetch/extract_{ds}.log"
    conda: ENV_SCOTE
    threads: 1
    resources: mem_mb=2000, runtime=360
    shell:
        "MIN_FREE_FRAC={MINFREE} MIN_FREE_GB={MINFREE_GB} scripts/disk_guard.sh {params.outdir} {params.size} > {log} 2>&1 && "
        "python scripts/extract_members.py --archive {params.archive} {params.extras} --outdir {params.outdir} "
        "--members {params.members:q} {params.delete:q} >> {log} 2>&1"

def _platform(*names):
    return "|".join(k for k in CONVERT if DS[k]["platform"] in names) or "__none__"

# one conversion rule per output layout, so that downstream rules can depend on the converted files themselves
CONVERT_OUT = {"molecules": ("xenium", "merscope", "atera"), "hdbins": ("visium_hd",), "spots": ("visium",)}
for _kind, _plats in CONVERT_OUT.items():
    _main = {"molecules": "molecules.parquet", "hdbins": "bins_square_002um.npz", "spots": "bins_spots55um.npz"}[_kind]
    rule:
        name: f"convert_{_kind}"
        input: str(DATA / "raw/{ds}/.extracted")
        output: qc="results/qc/{ds}.json", main=str(DATA / "processed/{ds}") + "/" + _main
        wildcard_constraints: ds=_platform(*_plats)
        params:
            platform=lambda w: DS[w.ds]["platform"],
            rawdir=lambda w: str(DATA / "raw" / w.ds),
            outdir=lambda w: str(DATA / "processed" / w.ds),
        log: "logs/fetch/convert_{ds}.log"
        conda: ENV_SCOTE
        threads: 2
        resources: mem_mb=8000, runtime=360
        shell:
            "python -m scote convert --platform {params.platform} --sample {wildcards.ds} --rawdir {params.rawdir} "
            "--outdir {params.outdir} --qc {output.qc} > {log} 2>&1"
