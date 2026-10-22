```{include} ../README.md
:start-after: <!-- use-start -->
:end-before: <!-- use-end -->
```

### Inputs

- Labels (molecule tables): a pandas Categorical or Series, an array of strings, or integer codes with `names`. Pairs
  closer than `8 * delta` are enumerated exactly in linear bins of width `delta`, then `n_levels` lattice levels
  follow (four bins per factor of two in distance; `shells_per_level`, default 8, sets the number of bins per level).
  `lattice_only=True` counts the shortest distances on a lattice of pitch `delta` as well, as for bins.
- Count matrices (bin and spot tables): an `(n, G)` SciPy sparse or NumPy matrix with `names`, or a DataFrame whose
  columns are the features. The bins are placed on a lattice of pitch `delta`, by default the bin size: the
  `bin_size` that the bin readers return (`read_visium_hd`, `read_bins`, ...), inferred from the coordinates if not
  given. A larger `delta` coarse-grains.
- The features are the categories of a Categorical that occur (unused categories are dropped), the `names` (all of
  them, also features without molecules), or the sorted labels; `select` takes a subset in a given order (a list of
  names, or `"top:N"`, `"min:N"`, `"file:PATH"`). The selection also defines the reference: $n_\mathrm{tot}$ and
  $P_\mathrm{tot}$ count the selected features only, so $\log g$ of a pair can differ between runs with different
  selections.

{mod}`scote.io` reads the outputs of Xenium, MERSCOPE, CosMx (AtoMx flat files), Atera, AVITI24 / Cytos, Visium HD,
Visium, and Stereo-seq (GEM and square-bin GEF) into memory (`read_xenium`, `read_merscope`, `read_cosmx`,
`read_atera`, `read_aviti24`, `read_visium_hd`, `read_visium`, `read_stereoseq`); control probes are dropped by
default, and gene names that occur more than once are made unique as AnnData's `var_names_make_unique` does. For
workflows that read a section many times, the converters of the same names (`convert_xenium`, ...) write it once to
the common data model, which `read_molecules` and `read_bins` read back: a molecule table is a parquet file with `x`,
`y` in µm and `feature` (`z`, `sample`, `cell_id`, `nucleus` optional); a bin table is an `.npz` file with CSR counts,
bin centers, feature names, and the bin size.

### Parameters

- `delta`: the finest distance resolution. For molecules, the width of the exact distance bins at short range; take
  it near the localization precision of the platform (e.g., 0.5 µm for many imaging platforms). For bins, the bin
  size (the reader's `bin_size`), or a multiple of it where the finest details are not of interest (e.g., where RNA
  diffuses over more than a bin).
- `n_levels`: every level doubles the range. The bins reach `8 * delta * 2**n_levels`, e.g. 0.5 µm and 12 levels give
  about 16 mm; the default is the smallest number of levels that reaches 16.4 mm (12 for `delta` 0.5 µm, 10 for
  2 µm). Cover the largest distance of interest; a range beyond the size of the section only adds bins with few or no
  pairs.
- `tile_um`: the side of the tiles of the uncertainties, default 500 µm; see
  [Choosing the tile size](uncertainties.md#choosing-the-tile-size).
- `backend`, `threads`, `out`, `acc_dtype`: see [Performance](performance.md).

### Result

- The result is a {class}`scote.TwoPointResult`: `log_g`, `sd`, `lo` / `hi`, `t_eff`, `flag`, `n_eff_pairs`, the
  counts `P` and `P_tot`, the feature totals, the distance bins `bins`, the run `params`, and the `timings`. The
  uncertainties are described under [Uncertainties](uncertainties.md).
- `keep_tiles=True` keeps the per-tile counts `counts_tiles` `(n_tiles, n_distance_bins, G, G)` for statistics beyond
  single pairs; `res.tiles` lists the column, row, and lower corner of every tile.
- `scote.pair_counts(xy, features, ...)`: global pair counts of the lattice hierarchy, without tiles or
  uncertainties, as a {class}`scote.PairCounts` (`counts`, `bins`, `features`, `params`, `timings`).
