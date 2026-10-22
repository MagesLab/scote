# Uncertainties

## The closed-form tile bootstrap

Every section is cut into square tiles of side `tile_um` (default 500 µm). All quantities of the estimator are sums
over tiles: the pair counts $P_{ab}$, the all-feature counts $P_\mathrm{tot}$, and the feature totals $n_a$ and
$n_\mathrm{tot}$. A block bootstrap draws the tiles with replacement and recomputes $\log g$, including its reference,
from the resampled sums. {mod}`scote.uncertainty` evaluates this bootstrap in closed form, as its limit for infinitely
many resamples with the delta method for $\log g$: the linearized contribution of tile $t$ is

$$
l_t = \frac{P^{(t)}_{ab}}{P_{ab}} - \frac{P^{(t)}_\mathrm{tot}}{P_\mathrm{tot}} - \frac{n^{(t)}_a}{n_a}
      - \frac{n^{(t)}_b}{n_b} + \frac{2 n^{(t)}_\mathrm{tot}}{n_\mathrm{tot}},
      \qquad \operatorname{Var}(\log g_{ab}) = \sum_t l_t^2 .
$$

The sum is accumulated tile by tile in four $G \times G$ arrays per distance bin while the pairs are counted, so no
resampling is done and no per-tile tensor has to be held in memory. For the uncertainty, each pair counts half in the
tile of each of its two members, so `sd` of $(a, b)$ equals that of $(b, a)$; the point estimates do not depend on
this attribution.

A run reports, per pair and distance bin:

- `sd`, the standard deviation of $\log g$, and the interval `lo`, `hi` = `log_g` ± 1.96 `sd`
  ({meth}`scote.TwoPointResult.interval` gives other levels);
- `n_eff_pairs` $= P_{ab}^2 / \operatorname{Var}(P_{ab})$, the tile scatter of the count;
- `t_eff` $= P_{ab}^2 / \sum_t (P^{(t)}_{ab})^2$, the effective number of tiles that carry the pair;
- `flag`: 0 for $T_\mathrm{eff} \ge 20$ (normal interval), 1 for $5 \le T_\mathrm{eff} < 20$ (interval reported but
  flagged), 2 for $T_\mathrm{eff} < 5$ (no interval, `lo` = `hi` = NaN). Pairs of rare features, or of features
  confined to a few tiles, come out flagged.

## Choosing the tile size

The bootstrap treats the tiles as independent samples of the tissue, which requires tiles larger than the correlation
length: the distance beyond which the two-point functions have decayed, so that $\log g$ is 0 within its interval for
the pairs of interest. The two-point functions show this length directly. Run once with tiles that are certainly large
enough (the default of 500 µm for a section of several millimeters), plot $\log g$ with its interval against $r$ for
the pairs you will report, and read off the distance from which all of them stay at 0.

- Tiles may be smaller than the default as long as they stay larger than that distance. Smaller tiles mean more of
  them, and the estimate of the variance from $T$ tiles is itself uncertain by a relative SD of about
  $1/\sqrt{2(T-1)}$; a run with fewer than 30 tiles warns for this reason.
- With tiles smaller than the correlation length, `sd` is not calibrated.
- At distances approaching the tile side, the interval is approximate. A synthetic test found `sd` conservative there
  (larger than the scatter between replicate sections); this is not a general rule.
- Structure on the scale of the whole section, such as a gradient across it, has a correlation length that no tile
  smaller than the section exceeds. For such structure, compare sections or replicates.

The quick start, with clusters of 8 µm diameter on a 600 µm section, uses 100 µm tiles; the sections of the preprint,
several millimeters across, used 500 µm tiles (81 to 356 tiles per section).

## Statistics beyond single pairs

Statistics of many $\log g$ values get their SD from the same per-tile contributions (run with `keep_tiles=True` to keep
the per-tile counts). {func}`scote.uncertainty.influence_log_g` gives the contribution $l_t$ of every $\log g$;
{func}`scote.uncertainty.delta_var` combines them with the gradient of a statistic
({func}`~scote.uncertainty.grad_pearson`, {func}`~scote.uncertainty.grad_rms`,
{func}`~scote.uncertainty.grad_mean_diff`, {func}`~scote.uncertainty.grad_ols_slope`,
{func}`~scote.uncertainty.grad_ols_intercept`, {func}`~scote.uncertainty.grad_deming`, or
{func}`~scote.uncertainty.grad_numeric`), for matched tiles of one section or for independent sections, and
{func}`scote.uncertainty.stat_ci` turns the variance into an interval. {func}`scote.uncertainty.collapsed_log_g` merges
features into groups, {func}`scote.uncertainty.pooled_log_g` pools pair sets,
{func}`scote.uncertainty.cross_shell_cov` gives the covariance of $\log g$ across distance bins, and
{func}`scote.uncertainty.tile_sum_var` and {func}`scote.uncertainty.tile_sum_stat` handle smooth functions of
tile-additive sums.
