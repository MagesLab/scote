# Two-point runs of the preprint (python -m scote twopoint): per-tile pair-count tensors with the closed-form tile block
# bootstrap, written to data/twopoint/ (several GB each). Included by the driver Snakefile; run from the examples root:
#
#   snakemake --configfile config/datasets.yaml config/host.yaml -j 4 twopoint_all
#
# Every run uses the 2 um lattice hierarchy with 500 um tiles; molecule inputs are the parquet tables of workflows/fetch.smk,
# Visium HD inputs the 2 um bin tables. Resources: measured peak memory of the preprint runs plus a margin.
from pathlib import Path
DATA = Path("data")
TP = str(DATA / "twopoint")
TILE = 500
MODE = config.get("twopoint_cpu_mode", "auto")      # lattice shell products: numba kernel at sparse levels, BLAS when dense
GENES_CRC = "results/platforms/common_genes_crc.txt"              # the 408 genes common to the CRC P2 Xenium, Visium HD and Visium v2 sections
GENES_SPLEEN = "results/platforms/common_genes_spleen.txt"        # the 90 genes common to the three spleen platforms
GENES_LIVER = "results/error_scaling/vizgen_liver_top200_genes.txt"   # top-200 genes of liver slice 1, used on the other slices
SPLEEN_MOL = ["sb_xenium_WT709_rep1", "sb_xenium_KO168", "sb_merscope_WT709"]
SPLEEN_HD = ["sb_hd_WT709", "sb_hd_KO168"]
MOL_FRACTIONS = [0.05, 0.1, 0.25, 0.5]                            # molecule subsamples at fixed area (preprint Fig. 4c)

def mol(ds): return str(DATA / "processed" / ds / "molecules.parquet")
def hd(ds): return str(DATA / "processed" / ds / "bins_square_002um.npz")
def v2(ds): return str(DATA / "processed" / ds / "bins_spots55um.npz")

SUB = {"xenium_crc_p2": dict(full=TP + "/xenium_crc_p2_bin2um_common.npz", genes="file:" + GENES_CRC, extra=[GENES_CRC]),
       "x5k_lymph_node": dict(full=TP + "/x5k_lymph_node_bin2um_top200.npz", genes="top:200", extra=[])}

rule twopoint_all:
    input:
        TP + "/xenium_crc_p2_mol_common.npz",
        expand(TP + "/{p}_crc_{s}_bin2um_common.npz", p=["xenium", "hd"], s=["p1", "p2", "p5"]),
        TP + "/v2_crc_p2_spots_common.npz",
        expand(TP + "/{ds}_bin2um_common.npz", ds=SPLEEN_MOL + SPLEEN_HD),
        TP + "/x5k_lymph_node_bin2um_top200.npz",
        expand(TP + "/error_scaling/vizgen_liver{s}_bin2um_top200.npz", s=["1slice1", "1slice2", "2slice1", "2slice2"]),
        expand(TP + "/error_scaling/xenium_crc_p2_bin2um_common_mol{m}.npz", m=MOL_FRACTIONS),
        expand(TP + "/error_scaling/x5k_lymph_node_bin2um_top200_mol{m}.npz", m=MOL_FRACTIONS),

rule twopoint_xenium_molecules:
    # molecule level: SCOTE level-0 kernel at delta = 0.5 um for r < 4 um, lattice shells above; 12 levels
    input: mol("xenium_crc_{s}"), GENES_CRC
    output: TP + "/xenium_crc_{s}_mol_common.npz"
    log: "logs/twopoint/xenium_crc_{s}_mol.log"
    conda: ENV_SCOTE
    threads: 4
    resources: mem_mb=64000, runtime=240
    shell:
        "python -m scote twopoint --molecules {input[0]} --select file:{input[1]} --delta 0.5 --n-levels 12 "
        "--tile-um {TILE} --threads {threads} --backend cpu,mode={MODE} --out {output} > {log} 2>&1"

rule twopoint_xenium_binned:
    # the same molecules on the Visium HD grid (2 um lattice): identical r bins as the HD run
    input: mol("xenium_crc_{s}"), GENES_CRC
    output: TP + "/xenium_crc_{s}_bin2um_common.npz"
    log: "logs/twopoint/xenium_crc_{s}_bin2um.log"
    conda: ENV_SCOTE
    threads: 4
    resources: mem_mb=64000, runtime=240
    shell:
        "python -m scote twopoint --molecules {input[0]} --select file:{input[1]} --lattice-only --delta 2 --n-levels 11 "
        "--tile-um {TILE} --threads {threads} --backend cpu,mode={MODE} --out {output} > {log} 2>&1"

rule twopoint_hd_bins:
    input: hd("hd_crc_{s}"), GENES_CRC
    output: TP + "/hd_crc_{s}_bin2um_common.npz"
    log: "logs/twopoint/hd_crc_{s}_bin2um.log"
    conda: ENV_SCOTE
    threads: 4
    resources: mem_mb=96000, runtime=360
    shell:
        "python -m scote twopoint --bins {input[0]} --select file:{input[1]} --n-levels 11 "
        "--tile-um {TILE} --threads {threads} --backend cpu,mode={MODE} --out {output} > {log} 2>&1"

rule twopoint_visium_spots:
    # 55 um spots at 100 um center pitch: lattice pitch 64 um = 2^5 x 2 um (each spot in its own cell), so the shells
    # coincide with the 2 um hierarchy from r = 256 um on; 5 levels
    input: v2("v2_crc_{s}"), GENES_CRC
    output: TP + "/v2_crc_{s}_spots_common.npz"
    log: "logs/twopoint/v2_crc_{s}_spots.log"
    conda: ENV_SCOTE
    threads: 2
    resources: mem_mb=8000, runtime=60
    shell:
        "python -m scote twopoint --bins {input[0]} --select file:{input[1]} --delta 64 --n-levels 5 "
        "--tile-um {TILE} --threads {threads} --backend cpu,mode={MODE} --out {output} > {log} 2>&1"

rule twopoint_spleen_molecules:
    # Xenium and MERSCOPE spleen sections on the 2 um lattice, 90 common genes
    input: lambda w: mol(w.ds), GENES_SPLEEN
    output: TP + "/{ds}_bin2um_common.npz"
    wildcard_constraints: ds="|".join(SPLEEN_MOL)
    log: "logs/twopoint/{ds}_bin2um.log"
    conda: ENV_SCOTE
    threads: 4
    resources: mem_mb=32000, runtime=120
    shell:
        "python -m scote twopoint --molecules {input[0]} --select file:{input[1]} --lattice-only --delta 2 --n-levels 11 "
        "--tile-um {TILE} --threads {threads} --backend cpu,mode={MODE} --out {output} > {log} 2>&1"

rule twopoint_spleen_hd:
    input: lambda w: hd(w.ds), GENES_SPLEEN
    output: TP + "/{ds}_bin2um_common.npz"
    wildcard_constraints: ds="|".join(SPLEEN_HD)
    log: "logs/twopoint/{ds}_bin2um.log"
    conda: ENV_SCOTE
    threads: 4
    resources: mem_mb=64000, runtime=180
    shell:
        "python -m scote twopoint --bins {input[0]} --select file:{input[1]} --n-levels 11 "
        "--tile-um {TILE} --threads {threads} --backend cpu,mode={MODE} --out {output} > {log} 2>&1"

rule twopoint_lymph_node_top200:
    input: mol("x5k_lymph_node")
    output: TP + "/x5k_lymph_node_bin2um_top200.npz"
    log: "logs/twopoint/x5k_lymph_node_top200.log"
    conda: ENV_SCOTE
    threads: 4
    resources: mem_mb=64000, runtime=240
    shell:
        "python -m scote twopoint --molecules {input} --select top:200 --lattice-only --delta 2 --n-levels 11 "
        "--tile-um {TILE} --threads {threads} --backend cpu,mode={MODE} --out {output} > {log} 2>&1"

rule twopoint_liver_top200:
    # slice 1 defines the top-200 genes; the other three slices use the same list
    input: lambda w: [mol(f"vizgen_liver{w.s}")] + ([] if w.s == "1slice1" else [GENES_LIVER])
    output: TP + "/error_scaling/vizgen_liver{s}_bin2um_top200.npz"
    params: genes=lambda w: "top:200" if w.s == "1slice1" else f"file:{GENES_LIVER}"
    log: "logs/twopoint/vizgen_liver{s}_top200.log"
    conda: ENV_SCOTE
    threads: 4
    resources: mem_mb=120000, runtime=600
    shell:
        "python -m scote twopoint --molecules {input[0]} --select {params.genes} --lattice-only --delta 2 --n-levels 11 "
        "--tile-um {TILE} --threads {threads} --backend cpu,mode={MODE} --out {output} > {log} 2>&1"

rule twopoint_molecule_subsample:
    # seeded random subsample of a fraction m of the molecules of the full run at fixed area: --max-molecules = round(m N),
    # N = n_units of the full run (its summary JSON next to the .npz)
    input: lambda w: [mol(w.ds), SUB[w.ds]["full"]] + SUB[w.ds]["extra"]
    output: TP + "/error_scaling/{ds}_{tag}_mol{m}.npz"
    wildcard_constraints: ds="|".join(SUB), tag="bin2um_common|bin2um_top200"
    params:
        genes=lambda w: SUB[w.ds]["genes"],
        summary=lambda w: SUB[w.ds]["full"].replace(".npz", ".json"),
    log: "logs/twopoint/{ds}_{tag}_mol{m}.log"
    conda: ENV_SCOTE
    threads: 4
    resources: mem_mb=64000, runtime=240
    shell:
        "exec > {log} 2>&1; n=$(python -c \"import json, sys; print(int(round(float(sys.argv[2]) * json.load(open(sys.argv[1]))['n_units'])))\" "
        "{params.summary} {wildcards.m}) && "
        "python -m scote twopoint --molecules {input[0]} --select {params.genes} --lattice-only --delta 2 --n-levels 11 --max-molecules $n "
        "--tile-um {TILE} --threads {threads} --backend cpu,mode={MODE} --out {output}"
