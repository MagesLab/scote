# Examples: SCOTE on public spatial omics data

Worked examples of the `scote` package of this repository on public data sets of imaging and sequencing-based platforms:
reading vendor output into the common data model, two-point functions of molecule and bin tables with their closed-form
uncertainties, and the statistics built on them (agreement across platforms, power and depth scaling, artifact
diagnostics, comparisons with cell-level methods, compute scaling). Together they produce every figure of the SCOTE
methods preprint https://doi.org/10.64898/2026.10.09.757915. The directory holds no data and no results: one Snakemake
workflow (`Snakefile`) downloads the data, converts them, runs the two-point functions and the analyses, and draws the
figures. Run every command from this directory; the scripts in `scripts/` show how the library is called.

## Running the workflow

```bash
conda env create -f envs/snakemake.yml && conda activate scote-snakemake
ln -s /path/to/a/volume/with/1TB data                     # downloads, conversions and two-point tensors go to data/
snakemake -n fig4                                         # dry run: what would be computed for Fig. 4
snakemake --sdm conda --configfile config/datasets.yaml config/host.yaml -j 4 --resources gpu=1 fig4
```

Every rule runs in the environment of `envs/scote.yml` (the packages of the preprint runs), which Snakemake creates on
first use with `--sdm conda` (or `--use-conda`); without either, the workflow stops at its start. The rules import
`scote` from `../src`, the package of this checkout, which is not installed in the environment.
Snakemake itself runs from any installation of Snakemake 8 or later with conda, such as the small environment
`envs/snakemake.yml`.

Targets: `all` (every figure), `fig1` ... `fig6` and `figS1` ... `figS4` (preprint numbering), `convert_all` (downloads and
conversions), `twopoint_all` (the two-point runs), `bench_all` (the compute benchmark of Fig. 6). Each figure target
computes only what its figure needs. Every rule declares threads, memory and run time, so the workflow runs unchanged under
a Snakemake executor plugin for a cluster (`--executor slurm` and the like); rules that need a GPU ask for `gpu=1`. The
complete workflow needs days of compute and about 1 TB of disk (660 GB of downloads).

Results are written to `results/` (tables, including the numbers quoted in the captions), `figures/` (PDF and PNG) and
`logs/` (the output and the errors of every rule). All results except the compute timings of Fig. 6 should reproduce
the preprint; the timings are single realizations whose values depend on the machine and on its load.

## Figures, scripts and tables

| Preprint | Target | Figure script (`figures/`) | Tables read (`results/`) |
|---|---|---|---|
| Fig. 1 | `fig1` | `make_fig1.py` | the CRC P2 molecule table and two-point runs (`data/`) |
| Fig. 2 | `fig2` | `make_fig2.py` | `platforms/{scatter_stats,crc_replicates}.json`, the CRC two-point runs |
| Fig. 3 | `fig3` | `make_fig3.py` | `spleen_operators/{baseline_functions,baselines_compare,three_groups,squidpy_*}.json`, the spleen operator runs |
| Fig. 4 | `fig4` | `make_fig4.py` | `error_scaling/*.json` |
| Fig. 5 | `fig5` | `make_fig5.py` | `artifacts/{collapse,acta2_regions,mastcells_*}.json` |
| Fig. 6 | `fig6` | `make_fig6.py` | `bench/{scote_*,kdtree,squidpy,tacco}.jsonl` |
| Fig. S1 | `figS1` | `make_figS1.py` | `platforms/scatter_stats.json`, the CRC two-point runs |
| Fig. S2 | `figS2` | `make_figS2.py` | `platforms/{scatter_stats,spleen_replicates}.json`, the spleen two-point runs |
| Fig. S3 | `figS3` | `make_figS3.py` | `ovff/depth/*.json` |
| Fig. S4 | `figS4` | `make_figS4.py` | `cosmx_kidney/{crossplatform_pedestal_bin05_direct,resolution}.json`, `cosmx/aviti24_zprojection_correction.json` |

A figure script can be run on its own once its tables exist, e.g. `python figures/make_fig4.py`.

## Data sources

| Data set | Figures | Source | Download |
|---|---|---|---|
| 10x Genomics CRC serial sections P1, P2, P5 (Xenium, Visium HD, Visium v2) | 1, 2, 4, 5, 6, S1 | 10x Genomics datasets, https://www.10xgenomics.com/datasets (CC BY 4.0) | workflow |
| 10x Genomics Xenium Prime 5K human lymph node | 4, 6 | 10x Genomics datasets (CC BY 4.0) | workflow |
| SpatialBench mouse spleen (Xenium, Visium HD, MERSCOPE) | 3, S2 | EBI BioStudies S-BSST2361, https://www.ebi.ac.uk/biostudies/studies/S-BSST2361 | workflow |
| 10x Genomics fresh-frozen ovarian cancer (Xenium Prime 5K, Visium HD at two depths) | S3 | 10x Genomics datasets (CC BY 4.0) | workflow |
| 10x Genomics Atera whole-transcriptome preview, cervical cancer | 6 | 10x Genomics datasets (CC BY 4.0) | workflow |
| CosMx 6k human kidney (Reck et al. 2025) | S4 | GEO GSE282059, https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE282059 | workflow |
| Vizgen MERFISH Mouse Liver Map | 4, 6 | vendor landing page https://info.vizgen.com/mouse-liver-access | by hand |
| Element Biosciences AVITI24 / Cytos A549 multimodal pooled screen | S4 | vendor landing page https://www.elementbiosciences.com/resources/dataset-explore-multimodal-pooled-screening-on-aviti24 | by hand |

The two data sets marked "by hand" are distributed by the vendors on their own pages and are not downloaded by the
workflow; it stops with a message pointing to the landing page. Obtain them there under the vendor's terms and place:

- Vizgen liver: `detected_transcripts.csv.gz` of Liver1Slice1, Liver1Slice2, Liver2Slice1 and Liver2Slice2 as
  `data/archives/vizgen_liver<N>slice<M>/detected_transcripts.csv.gz` (for example `data/archives/vizgen_liver1slice1/`);
- Element AVITI24: the SpatialData zarr zip of the screen as `data/archives/aviti24_a549_screen/cytos_a549_screen.zarr.zip`
  (read in place, never extracted).

## What has been checked

`snakemake -n` builds the complete graph from an empty directory (every figure target has a path from the downloads), and
the figure scripts reproduce the figures of the posted preprint pixel for pixel from the tables of the preprint runs. The
workflow has not been run end to end in this form; its steps are the scripts and parameters of the preprint runs.
`python -m pytest tests` runs the synthetic tests of the registration helpers (`scripts/registration.py`).
