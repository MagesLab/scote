# SCOTE: scale-resolved two-point correlation functions for spatial omics

<!-- intro-start -->
SCOTE (Spatial COvariance TEnsors) computes normalized two-point correlation functions $g_{ab}(r)$ for
single-molecule resolved (like MERFISH, CosMx, Xenium, AVITI24, Atera) and bin-based spatial omics methods (like Visium,
Stereo-seq, Visium HD) from sub-micrometer to centimeter scales. Pairs are counted on a log-hierarchical lattice as
sparse matrix products, normalized by the random-label expectation on the same lattice, with tile block-bootstrap
uncertainties. The CPU (numba / SciPy) and GPU (PyTorch) backends give bit-identical integer counts.

For two features $a$ and $b$ (genes, or any other label of a molecule or of the counts in a bin), the two-point
correlation function is

$$
g_{ab}(r_k) = \frac{P_{ab}(k)}{P_\mathrm{tot}(k) \thinspace n_a n_b / n_\mathrm{tot}^2},
$$

where $P_{ab}(k)$ is the number of ordered pairs of an $a$ molecule and a $b$ molecule at a distance in bin $k$
(central distance $r_k$), $P_\mathrm{tot}(k)$ the number of all ordered pairs of all features in that bin, $n_a$ and
$n_b$ the totals of $a$ and $b$, and $n_\mathrm{tot}$ the total of all features. $\log g = 0$ means no association
beyond the spatial pattern that all molecules share, $\log g > 0$ co-localization, and $\log g < 0$ separation. The
uncertainty comes from tiles, squares of side `tile_um` that cut the section
(see [Uncertainties](https://mageslab.github.io/scote/uncertainties.html)).

The method is described in the preprint <https://doi.org/10.64898/2026.10.09.757915>.
<!-- intro-end -->
Documentation: <https://mageslab.github.io/scote>.

<!-- install-start -->
## Install

Python >= 3.11.

```bash
pip install "scote @ git+https://github.com/MagesLab/scote"            # the package (import name `scote`)
pip install "scote[gpu] @ git+https://github.com/MagesLab/scote"       # + PyTorch for the GPU backend
```

| Extra | Adds |
|---|---|
| `gpu` | PyTorch, for `backend="gpu"` |
| `io` | scanpy, anndata, zarr, numcodecs, h5py, for the Visium and Visium HD (scanpy), Atera (zarr), and Stereo-seq GEF (h5py) readers of `scote.io` |
| `dev` | pytest |
| `test` | pytest and PyTorch, for the whole test suite including the GPU-backend tests |
| `docs` | Sphinx, MyST, the Read the Docs theme, sphinx-copybutton, for `docs/` (see `docs/README.md`) |
<!-- install-end -->

<!-- quickstart-start -->
## Quick start

Synthetic molecules placed uniformly at random: three features keep random labels, and a fourth, `clustered`, takes
the place of the `background` molecules near 500 random centers. The positions of all molecules stay uniform.

```python
import numpy as np
from scipy.spatial import cKDTree
import scote

### Prepare Data

rng = np.random.default_rng(0)
# 90k molecules placed uniformly at random on a 600 µm x 600 µm section, with random labels
xy = rng.uniform(0, 600, (90_000, 2))
labels = rng.choice(["uniform_a", "uniform_b", "uniform_c", "background"], len(xy), p=[0.2, 0.2, 0.2, 0.4])
# background molecules within 4 µm of one of 500 random centers become "clustered"
distance, _ = cKDTree(rng.uniform(0, 600, (500, 2))).query(xy)
labels[(labels == "background") & (distance < 4)] = "clustered"

### Run SCOTE

# log g_ab(r) of all label pairs from 0.5 µm to about 1 mm, uncertainties from 100 µm x 100 µm tiles
res = scote.twopoint(xy, labels, delta=0.5, n_levels=8, tile_um=100.0)

### Access results

# one row per distance bin: r_central, log_g, sd, lo, hi, ...
clustered, uniform = res.pair("clustered", "clustered"), res.pair("uniform_a", "uniform_b")
for k in range(0, res.n_distance_bins, 4):
    if clustered.r_central[k] > 100.0:      # beyond the tile side, the intervals are approximate
        break
    print(f"r = {clustered.r_central[k]:5.1f} µm"
          f"  clustered, clustered: {clustered.log_g[k]:+.2f} +- {1.96 * clustered.sd[k]:.2f}"
          f"  uniform_a, uniform_b: {uniform.log_g[k]:+.3f} +- {1.96 * uniform.sd[k]:.3f}")
```

The clustered feature has $\log g$ of about +2.6 below 1 µm and +1.8 at 4 µm, and falls to 0 at 8 µm, the diameter of
a cluster. The uniform features stay at $\log g = 0$ within their intervals. The rows stop at the tile side, 100 µm:
at distances approaching it, the intervals are approximate (see
[Uncertainties](https://mageslab.github.io/scote/uncertainties.html)). The first call compiles the numba kernels,
which takes from a few seconds to a minute depending on the machine; the kernels are cached on disk, so later calls,
also in a new Python process, take about a second for this example.
<!-- quickstart-end -->

## Uncertainties

The error bars are the closed-form limit of the tile block bootstrap: pair counts are attributed to square spatial tiles
(`tile_um`, default 500 µm), and the delta method gives the standard deviation `sd` of $\log g$ from the scatter of the
per-tile counts, without resampling. The interval `lo`, `hi` is `log_g` ± 1.96 `sd`; it is flagged (`flag` 1) or
withheld (`flag` 2, `lo` = `hi` = NaN) when fewer than 20 or 5 effective tiles carry the pair. The tiles should be
larger than the correlation length of the tissue, and at distances approaching the tile side the interval is
approximate; the [documentation](https://mageslab.github.io/scote/uncertainties.html) explains how to read the
correlation length from the two-point functions.

<!-- use-start -->
## Use

One function, `scote.twopoint`, takes the positions `xy` in micrometers and `features`, either one label per point (molecule
tables) or a count matrix (bin and spot tables), and dispatches on the type of `features`. For a molecule table of
your own, a DataFrame `df` with `x`, `y` in micrometers and the `gene` of every molecule:

```python
import scote
res = scote.twopoint(df[["x", "y"]].to_numpy(), df["gene"])

res.pair("CD3E", "MS4A1")            # one pair as a table over the distance bins
res.log_g, res.sd, res.flag          # (n_distance_bins, G, G); res.bins: level, shell, r_from, r_to, r_central
lo, hi = res.interval((2.5, 97.5))   # normal interval log g -+ 1.96 sd, NaN where flag == 2
res.save("section.npz")              # or a directory; scote.TwoPointResult.load reads both
```

The readers of `scote.io` return the data in this form, either from the vendor output directly or from the common
data model:

```python
import scote

# molecules, e.g. a Xenium output read into memory: a DataFrame with x, y (µm) and the gene of every molecule
df = scote.io.read_xenium("outs/")
res = scote.twopoint(df[["x", "y"]].to_numpy(), df["feature"], select="top:200", delta=0.5, n_levels=12)

# bins, e.g. the 2 µm bins of a Visium HD output: centers (µm), sparse counts, gene names, bin size
xy, X, names, bin_size = scote.io.read_visium_hd("outs/binned_outputs", bin="square_002um")
res = scote.twopoint(xy, X, names=names, bin_size=bin_size, select="top:200", n_levels=11)
```
<!-- use-end -->

The [documentation](https://mageslab.github.io/scote/usage.html) describes the inputs, the parameters, and the
result in detail.

## Examples

The directory [`examples/`](examples/README.md) shows SCOTE at work on public data sets of the platforms named above:
from the downloads, through the conversion to the common data model and the two-point runs, to the analyses built on
them (cross-platform comparisons, the effect of sequencing depth and resolution, tissue operators, power and replicate
analyses, and a compute benchmark). One Snakemake workflow runs every step and produces every figure of the preprint;
the scripts can be read and adapted one by one. The repository holds no data and no results.
[`examples/README.md`](examples/README.md) maps each figure to its workflow target and scripts and gives the commands.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

The tests check the integer counts of all backends against brute force and the enumeration kernel, the closed form
against Monte Carlo, and every reader against its converter on small synthetic vendor outputs.

## Citation

Ilin M., Singh Y., Mages S. W. (2026). Scale-resolved correlation functions unify spatial omics across platforms and
resolutions. bioRxiv, https://doi.org/10.64898/2026.10.09.757915 (`CITATION.cff`).
