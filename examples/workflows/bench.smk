# Compute benchmark of preprint Fig. 6: SCOTE rows on 1, 4 and 16 pinned CPU threads and on one GPU, and the comparators
# (KD-tree cross-K, Squidpy co_occurrence at molecule and cell level, TACCO at cell level) on identical inputs
# (scripts/bench/common.py). Included by the driver Snakefile; run from the examples root:
#
#   snakemake -j 1 --resources gpu=1 bench_all
#
# One run per job; every job asks for a 16-CPU allocation (the thread count of a row is pinned inside it with the
# *_NUM_THREADS variables), except the single-worker KD-tree rows (1 CPU). Timings are single realizations whose values
# depend on the machine and on its load; run the jobs one at a time (-j 1) or on otherwise idle nodes. The rows record
# the installed versions, the CPU model and the GPU (env, gpu fields).
#
# A SCOTE row writes its full output (closed: log g, sd, n_eff_pairs, T_eff, flag and P of every gene pair and shell; none: P)
# to `bench_write_dir` (config/host.yaml), up to about 400 GB for the whole-transcriptome rows; the output is removed
# after the row.
BENCH = "results/bench"
CACHE = "data/bench_scaling"
WRITE = config.get("bench_write_dir", "data/bench/write")
HD_BINS = "data/processed/hd_crc_p2/bins_square_002um.npz"
LIVER_DS = [f"vizgen_liver{s}" for s in ("1slice1", "1slice2", "2slice1", "2slice2")]

# ---------------------------------------------------------------- SCOTE rows: tag -> (sweep, bench_ours arguments, inputs)
def _cache(ds): return f"{CACHE}/{ds}.npz"

ROWS = {}
for n in (1000, 3000, 10000, 30000, 100000, 300000):
    ROWS[f"crc_G200_n{n}"] = ("molecules", f"--G 200 --thin-n {n}", [_cache("xenium_crc_p2")])
for f in ("0.01", "0.03", "0.1", "0.3", "1.0"):
    ROWS[f"crc_G200_thin{f}"] = ("molecules", f"--G 200 --thin-frac {f}", [_cache("xenium_crc_p2")])
for c in ("0.0625", "0.25", "0.5"):
    ROWS[f"crc_G200_crop{c}"] = ("area", f"--G 200 --crop {c}", [_cache("xenium_crc_p2")])
for g in (50, 422):
    ROWS[f"crc_G{g}"] = ("genes_crc", f"--G {g}", [_cache("xenium_crc_p2")])
for g, tag in ((50, "50"), (200, "200"), (1000, "1000"), (0, "all")):
    ROWS[f"ln_G{tag}"] = ("genes_ln", f"--dataset x5k_lymph_node --G {g}", [_cache("x5k_lymph_node")])
for g, tag in ((1000, "1000"), (5000, "5000"), (0, "all")):
    ROWS[f"hd_G{tag}"] = ("genes_hd", f"--hd {HD_BINS} --G {g}", [HD_BINS])
for g, tag in ((200, "200"), (1000, "1000"), (0, "all")):
    ROWS[f"atera_G{tag}"] = ("atera_cervical", f"--dataset atera_cervical --G {g}", [_cache("atera_cervical")])
for ds in LIVER_DS:
    ROWS[f"liver_{ds}"] = ("liver_slice", f"--dataset {ds} --G 200", [_cache(ds)])
ROWS["liver_all"] = ("liver_all", f"--datasets {','.join(LIVER_DS)} --G 200", [_cache(ds) for ds in LIVER_DS])

CRC = [t for t in ROWS if t.startswith("crc_")]
BOTH = ("closed", "none")
# profile -> (device, threads, [(tag, uncertainties)])
PROFILES = {
    "cpu1": ("cpu", 1, [(t, BOTH) for t in CRC + ["ln_G50", "ln_G200", "ln_G1000"]]),
    "cpu4": ("cpu", 4, [(t, BOTH) for t in CRC + ["ln_G50", "ln_G200", "ln_G1000"]]),
    "cpu16": ("cpu", 16, [(t, BOTH) for t in CRC + ["ln_G50", "ln_G200", "ln_G1000", "ln_Gall", "hd_G1000", "hd_G5000"]]
                         + [("hd_Gall", ("none",))]),
    "gpu": ("cuda", 16, [(t, BOTH) for t in CRC + ["ln_G50", "ln_G200", "ln_G1000", "ln_Gall", "hd_G1000", "hd_G5000", "hd_Gall",
                                                    "atera_G200", "atera_G1000", "atera_Gall"]
                         + [f"liver_{ds}" for ds in LIVER_DS] + ["liver_all"]]),
}
MEM = {"ln_Gall": 192000, "hd_G1000": 192000, "hd_G5000": 192000, "hd_Gall": 288000, "atera_G200": 256000,
       "atera_G1000": 256000, "atera_Gall": 1048576, "liver_all": 640000}
MEM.update({f"liver_{ds}": 256000 for ds in LIVER_DS})
RUNTIME = {"cpu1": 60, "cpu4": 40, "cpu16": 30, "gpu": 15}            # minutes per row; longer rows below
RUNTIME_LONG = {"ln_G1000": 2.0, "ln_Gall": 3.0, "hd_G1000": 3.0, "hd_G5000": 4.0, "hd_Gall": 6.0, "atera_Gall": 6.0, "liver_all": 2.0}


def _row_files(profile):
    return [f"{BENCH}/rows/{profile}/{t}_{u}.jsonl" for t, us in PROFILES[profile][2] for u in us]


rule bench_all:
    input: expand(BENCH + "/scote_{p}.jsonl", p=list(PROFILES)), BENCH + "/kdtree.jsonl", BENCH + "/squidpy.jsonl", BENCH + "/tacco.jsonl"

rule bench_cache:
    # benchmark input cache: x, y (float32 um), gene code = abundance rank in the full section (scripts/bench/prep.py)
    input: "data/processed/{ds}/molecules.parquet"
    output: CACHE + "/{ds}.npz"
    wildcard_constraints: ds="[a-z0-9_]+"
    log: "logs/bench/cache_{ds}.log"
    conda: ENV_SCOTE
    threads: 8
    resources: mem_mb=lambda w: 640000 if w.ds.startswith("atera") else 128000, runtime=120
    shell: "python scripts/bench/prep.py {wildcards.ds} > {log} 2>&1"

rule bench_cells:
    # cell table of Xenium CRC P2 (centroids, 20 k-means clusters) for the cell-level comparators
    input: "data/raw/xenium_crc_p2/.extracted"
    output: CACHE + "/xenium_crc_p2_cells.parquet"
    log: "logs/bench/cells.log"
    conda: ENV_SCOTE
    threads: 8
    resources: mem_mb=32000, runtime=60
    shell: "python scripts/bench/prep.py --cells xenium_crc_p2 > {log} 2>&1"

rule bench_scote_row:
    input: lambda w: ROWS[w.tag][2]
    output: BENCH + "/rows/{profile}/{tag}_{unc}.jsonl"
    wildcard_constraints: profile="|".join(PROFILES), unc="closed|none"
    params:
        dev=lambda w: PROFILES[w.profile][0], n=lambda w: PROFILES[w.profile][1], sweep=lambda w: ROWS[w.tag][0],
        args=lambda w: ROWS[w.tag][1], wdir=lambda w: f"{WRITE}/{w.profile}/{w.tag}_{w.unc}",
        wait=lambda w: "export OMP_WAIT_POLICY=PASSIVE; " if PROFILES[w.profile][:2] == ("cpu", 16) else "",
    log: "logs/bench/{profile}/{tag}_{unc}.log"
    conda: ENV_SCOTE
    threads: 16
    resources:
        mem_mb=lambda w: MEM.get(w.tag, 64000) if w.profile in ("cpu16", "gpu") else 128000,
        runtime=lambda w: int(RUNTIME[w.profile] * RUNTIME_LONG.get(w.tag, 1.0)),
        gpu=lambda w: 1 if w.profile == "gpu" else 0,
    shell:
        # a closed GPU row that runs out of device memory with float64 accumulators is repeated with float32 accumulators
        # (acc_dtype is recorded in the row)
        "exec > {log} 2>&1; "
        "export OMP_NUM_THREADS={params.n} MKL_NUM_THREADS={params.n} OPENBLAS_NUM_THREADS={params.n} NUMBA_NUM_THREADS={params.n}; "
        "{params.wait}rm -rf {params.wdir} {output}; mkdir -p $(dirname {params.wdir}); "
        "B=\"python scripts/bench/bench_ours.py --device {params.dev} --threads {params.n} --sweep {params.sweep} "
        "--uncertainty {wildcards.unc} --write {params.wdir} {params.args} --out {output}\"; "
        "$B || {{ [ {params.dev} = cuda ] && [ {wildcards.unc} = closed ] && grep -qi 'out of memory' {log} && "
        "rm -rf {params.wdir} {output} && $B --acc-dtype float32; }}; rc=$?; rm -rf {params.wdir}; exit $rc"

rule bench_scote_profile:
    input: lambda w: _row_files(w.profile)
    output: BENCH + "/scote_{profile}.jsonl"
    wildcard_constraints: profile="|".join(PROFILES)
    log: "logs/bench/scote_{profile}.log"
    conda: ENV_SCOTE
    shell: "cat {input} > {output} 2> {log}"

# ---------------------------------------------------------------- comparators
# KD-tree cross-K (scripts/bench/bench_kdtree.py): (workers, G, selection, sweep, pair budget s, extra arguments)
KD = {}
for g in (50, 200):
    for n in (1000, 3000, 10000, 30000, 100000, 300000, 1000000, 3000000):
        KD[f"w16_G{g}_n{n}"] = (16, g, f"--thin-n {n}", "molecules", 600, "")
    for n in (1000, 3000, 10000, 30000, 100000):
        KD[f"w1_G{g}_n{n}"] = (1, g, f"--thin-n {n}", "molecules", 300, "")
for n, budget in ((300000, 6000), (1000000, 6000), (3000000, 11700)):
    KD[f"w1_G200_n{n}"] = (1, 200, f"--thin-n {n}", "molecules", budget, "")
KD["w16_G200_full"] = (16, 200, "--thin-frac 1.0", "molecules_full_partial", 28800, "--chunk 1")   # one gene pair per task

rule bench_kdtree_row:
    input: _cache("xenium_crc_p2")
    output: BENCH + "/rows/kdtree/{tag}.jsonl"
    params: k=lambda w: KD[w.tag]
    log: "logs/bench/kdtree/{tag}.log"
    conda: ENV_SCOTE
    threads: lambda w: KD[w.tag][0]
    resources: mem_mb=64000, runtime=lambda w: int(KD[w.tag][4] / 60 * 1.1) + 20
    shell:
        "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1; rm -f {output}; "
        "python scripts/bench/bench_kdtree.py --G {params.k[1]} {params.k[2]} --workers {params.k[0]} --budget-s {params.k[4]} "
        "{params.k[5]} --sweep {params.k[3]} --out {output} > {log} 2>&1"

# Squidpy co_occurrence at molecule level (16 and 1 numba threads; runs whose predicted buffer exceeds --mem-gb are recorded as
# skipped) and at cell level (8 threads)
SQ = {}
for n in (1000, 3000, 10000, 30000, 100000, 300000):
    SQ[f"t16_G50_n{n}"] = (16, f"--G 50 --thin-n {n} --mem-gb 200 --sweep molecules")
for n in (1000, 3000, 10000, 20000, 100000):
    SQ[f"t16_G200_n{n}"] = (16, f"--G 200 --thin-n {n} --mem-gb 200 --sweep molecules")
for n in (1000, 3000, 10000, 30000):
    SQ[f"t1_G50_n{n}"] = (1, f"--G 50 --thin-n {n} --mem-gb 200 --sweep molecules")
for n in (1000, 3000, 10000):
    SQ[f"t1_G200_n{n}"] = (1, f"--G 200 --thin-n {n} --mem-gb 200 --sweep molecules")
CELLS_N = (10000, 30000, 100000, 0)                              # 0 = all segmented cells
for n in CELLS_N:
    SQ[f"cells_n{n}"] = (8, f"--cells {CACHE}/xenium_crc_p2_cells.parquet --thin-n {n} --interval cell300 --mem-gb 50 --sweep cells")

rule bench_squidpy_row:
    input: lambda w: [CACHE + "/xenium_crc_p2_cells.parquet"] if w.tag.startswith("cells") else [_cache("xenium_crc_p2")]
    output: BENCH + "/rows/squidpy/{tag}.jsonl"
    params: n=lambda w: SQ[w.tag][0], args=lambda w: SQ[w.tag][1]
    log: "logs/bench/squidpy/{tag}.log"
    conda: ENV_SCOTE
    threads: lambda w: SQ[w.tag][0]
    resources: mem_mb=lambda w: 64000 if w.tag.startswith("cells") else 220000, runtime=40
    shell:
        "export OMP_NUM_THREADS={params.n} MKL_NUM_THREADS={params.n} OPENBLAS_NUM_THREADS={params.n} NUMBA_NUM_THREADS={params.n}; "
        "rm -f {output}; python scripts/bench/bench_squidpy.py {params.args} --threads {params.n} --out {output} > {log} 2>&1"

rule bench_tacco_row:
    input: CACHE + "/xenium_crc_p2_cells.parquet"
    output: BENCH + "/rows/tacco/cells_n{n}.jsonl"
    log: "logs/bench/tacco/cells_n{n}.log"
    conda: ENV_SCOTE
    threads: 8
    resources: mem_mb=64000, runtime=40
    shell:
        "export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8; rm -f {output}; "
        "python scripts/bench/bench_tacco.py --cells {input} --thin-n {wildcards.n} --threads 8 --out {output} > {log} 2>&1"

rule bench_comparators:
    input:
        kd=expand(BENCH + "/rows/kdtree/{t}.jsonl", t=list(KD)),
        sq=expand(BENCH + "/rows/squidpy/{t}.jsonl", t=list(SQ)),
        tac=expand(BENCH + "/rows/tacco/cells_n{n}.jsonl", n=CELLS_N),
    output: BENCH + "/kdtree.jsonl", BENCH + "/squidpy.jsonl", BENCH + "/tacco.jsonl"
    log: "logs/bench/comparators.log"
    conda: ENV_SCOTE
    shell: "exec 2> {log}; cat {input.kd} > {output[0]}; cat {input.sq} > {output[1]}; cat {input.tac} > {output[2]}"
