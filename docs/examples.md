# Examples

The directory `examples/` of the repository shows SCOTE at work on public data sets of single-molecule resolved and
bin-based platforms: from the downloads, through the conversion to the common data model and the two-point runs, to
the analyses built on them (cross-platform comparisons, the effect of sequencing depth and resolution, tissue
operators, power and replicate analyses, and a compute benchmark). One Snakemake workflow runs every step and produces
every figure of the preprint (<https://doi.org/10.64898/2026.10.09.757915>); the scripts can be read and adapted one by
one. Its [README](https://github.com/MagesLab/scote/blob/main/examples/README.md) maps each figure to its workflow
target and scripts and gives the commands.
