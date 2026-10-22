"""Closed-form building blocks beyond per-pair log g (scote.uncertainty): collapsed partitions, per-tile influence,
delta-method variances across pairs and sections, cross-shell covariance, tile-additive sums, pooled pair sets.
Each is checked against the explicit two-pass definition and, where it is a bootstrap statistic, against a Monte Carlo
block bootstrap with B = 20,000 (the draws of _helpers.bootstrap_weights, i.e. those of the loop bootstrap)."""
import numpy as np
import pytest

import scote
from scote import random_label_reference
from scote import uncertainty as TV
from _helpers import bootstrap_weights, loop_bootstrap_log_g

B_MC = 20_000


def _toy(T_=40, K=3, G=12, seed=0, spread=0.5):
    rng = np.random.default_rng(seed)
    N = rng.poisson(rng.uniform(20, 200, (1, G)) * rng.uniform(1 - spread, 1 + spread, (T_, 1))).astype(np.float64)
    lam = rng.uniform(2, 300, (1, K, G, G)) * rng.uniform(1 - spread, 1 + spread, (T_, 1, 1, 1))
    return _sym(rng.poisson(lam).astype(np.float64)), N


def _sym(C):
    """Per-tile symmetric slabs: then the default symmetrization of the closed form is a no-op and the Monte Carlo
    bootstrap of the raw tensor estimates the same quantity (symmetrization itself is tested in test_uncertainty.py)."""
    return 0.5 * (C + np.swapaxes(C, -1, -2))


def _explicit_l(C, N, k):
    """l_t(ab) written out tile by tile from the definition."""
    T_ = C.shape[0]
    P = C[:, k].sum(0)
    Q = P.sum()
    n_a = N.sum(0)
    n = n_a.sum()
    out = np.empty((T_,) + P.shape)
    for t in range(T_):
        with np.errstate(divide="ignore", invalid="ignore"):
            out[t] = (C[t, k] / P - C[t, k].sum() / Q - (N[t] / n_a)[:, None] - (N[t] / n_a)[None, :]
                      + 2 * N[t].sum() / n)
    return out


def _mc_log_g(C, N, k, W):
    """log g of every resample (B, G, G) at shell k for tile weights W (B, T)."""
    T_, _, G, _ = C.shape
    Pb = (W @ C[:, k].reshape(T_, -1)).reshape(-1, 1, G, G)
    nb = W @ N
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(Pb / random_label_reference(Pb, nb))[:, 0]


# ------------------------------------------------------------------------------------------------ 1. collapse
def _merged_log_g_reference(ct, tt, ia, ib, w):
    """The statistic of a merged feature group for one bootstrap weight vector (collapsed counts, full P_tot and n)."""
    sub = ct[:, :, ia][:, :, :, ib].sum(axis=(2, 3))
    subtot = ct.sum(axis=(2, 3))
    Pb, Ptb, nbt = w @ sub, w @ subtot, w @ tt
    with np.errstate(all="ignore"):
        return np.log(Pb / (Ptb * nbt[ia].sum() * nbt[ib].sum() / nbt.sum() ** 2))


def test_collapse_tensor_explicit_and_rest_group():
    C, N = _toy(T_=20, K=2, G=9, seed=4)
    labels = np.array([0, 0, 1, -1, 1, 2, -1, 2, 2])
    Cc, Nc = TV.collapse_tensor(C, N, labels)
    assert Cc.shape == (20, 2, 4, 4) and Nc.shape == (20, 4)              # L = 3 groups + rest
    groups = [np.where(labels == g)[0] for g in range(3)] + [np.where(labels == -1)[0]]
    for A, ga in enumerate(groups):
        assert np.array_equal(Nc[:, A], N[:, ga].sum(1))
        for B, gb in enumerate(groups):
            assert np.array_equal(Cc[:, :, A, B], C[:, :, ga][:, :, :, gb].sum(axis=(2, 3)))
    assert np.array_equal(Cc.sum(axis=(2, 3)), C.sum(axis=(2, 3)))      # full P_tot per tile and shell
    assert np.array_equal(Nc.sum(1), N.sum(1))                           # full n per tile
    Cn, Nn = TV.collapse_tensor(C, N, np.where(labels < 0, 0, labels))  # no -1: no rest group
    assert Cn.shape == (20, 2, 3, 3)
    with pytest.raises(ValueError):
        TV.collapse_tensor(C, N, labels[:-1])


def test_collapsed_closed_form_reproduces_merged_log_g():
    C, N = _toy(T_=40, K=3, G=12, seed=1)
    labels = np.array([0, 0, 1, 1, 1, 2, 2, -1, -1, 3, 3, -1])
    L = 4
    res = TV.collapsed_log_g(C, N, labels, names=["A", "B", "C", "D"])
    assert res.log_g.shape == (3, L, L) and res.params["n_total"] == N.sum() and res.features == ["A", "B", "C", "D"]
    assert res.totals_tiles.shape == (40, L) and res.counts_tiles is None
    assert np.array_equal(res.P_tot, C.sum(0).sum(axis=(1, 2)))           # the full reference total
    T_ = C.shape[0]
    for A in range(L):
        for B in range(L):
            ia, ib = list(np.where(labels == A)[0]), list(np.where(labels == B)[0])
            lg_ref = _merged_log_g_reference(C, N, ia, ib, np.ones(T_))
            np.testing.assert_allclose(res.log_g[:, A, B], lg_ref, rtol=1e-13, atol=1e-13)
            # two-pass delta method on the ORIGINAL tensor with the merged definition
            Pt = C[:, :, ia][:, :, :, ib].sum(axis=(2, 3))
            Qt = C.sum(axis=(2, 3))
            nA, nB, n = N[:, ia].sum(1), N[:, ib].sum(1), N.sum(1)
            infl = (Pt / Pt.sum(0) - Qt / Qt.sum(0) - (nA / nA.sum())[:, None] - (nB / nB.sum())[:, None]
                    + (2 * n / n.sum())[:, None])
            np.testing.assert_allclose(res.sd[:, A, B], np.sqrt((infl ** 2).sum(0)), rtol=1e-9)
    # Monte Carlo: the loop bootstrap on the collapsed tensor (with the rest group) is the merged-group statistic
    Cc, Nc = TV.collapse_tensor(C, N, labels)
    boot = loop_bootstrap_log_g(Cc, Nc, n_boot=B_MC, seed=2)
    ratio = res.sd / boot["sd"][:, :L, :L]
    assert np.abs(ratio - 1).max() < 0.03, ratio
    # one resample: the same statistic as the merged-group Monte Carlo
    w = bootstrap_weights(T_, 1, np.random.default_rng(5))[0]
    lb = _mc_log_g(Cc, Nc, 1, w[None])[0][:L, :L]
    for A, B in [(0, 1), (2, 3), (1, 1)]:
        ref = _merged_log_g_reference(C, N, list(np.where(labels == A)[0]), list(np.where(labels == B)[0]), w)[1]
        np.testing.assert_allclose(lb[A, B], ref, rtol=1e-12)


# --------------------------------------------------------------------------------------------- 2. influence
def test_influence_explicit_sum_zero_and_closed_form():
    C, N = _toy(seed=2)
    C[:, 1, 0, 1] = C[:, 1, 1, 0] = 0.0                                  # a pair without counts
    infl = TV.influence_log_g(C, N)
    assert infl.shape == (40, 3, 12, 12) and infl.dtype == np.float64
    for k in range(3):
        np.testing.assert_allclose(infl[:, k], _explicit_l(C, N, k), rtol=1e-12, atol=1e-15, equal_nan=True)
    ok = np.isfinite(infl).all(0)
    assert (~ok).sum() == 2 and np.isnan(infl[:, 1, 0, 1]).all()
    assert np.abs(infl.sum(0)[ok]).max() < 1e-12
    cf = TV.closed_form_log_g(C, N)
    sd = TV.closed_form_from_influence(infl)
    np.testing.assert_allclose(sd[ok], cf.sd[ok], rtol=1e-9)
    assert np.isnan(sd[~ok]).all()
    # shell subsets and summed bin sets
    l2 = TV.influence_log_g(C, N, shells=[2, [0, 1]])
    np.testing.assert_allclose(l2[:, 0], infl[:, 2], rtol=0, atol=0)
    Cs = C[:, [0, 1]].sum(1, keepdims=True)
    np.testing.assert_allclose(TV.closed_form_from_influence(l2[:, 1:]), TV.closed_form_log_g(Cs, N).sd,
                               rtol=1e-9, equal_nan=True)


# ------------------------------------------------------------------------------------- 3. gradients, delta_var
GRADS = dict(pearson=TV.grad_pearson, rms=TV.grad_rms, mean_diff=TV.grad_mean_diff, ols_slope=TV.grad_ols_slope,
             ols_intercept=TV.grad_ols_intercept, deming=lambda x, y: TV.grad_deming(x, y, 1.7),
             major_axis=lambda x, y: TV.grad_deming(x, y, 1.0))


def _stats_vec(x, y, name):
    """Vectorized statistic over rows of x, y (B, n) (direct formulas, independent of the gradient code)."""
    dx = x - x.mean(1, keepdims=True)
    dy = y - y.mean(1, keepdims=True)
    sxx, syy, sxy = (dx * dx).mean(1), (dy * dy).mean(1), (dx * dy).mean(1)
    if name == "pearson":
        return sxy / np.sqrt(sxx * syy)
    if name == "rms":
        return np.sqrt(((y - x) ** 2).mean(1))
    if name == "mean_diff":
        return (y - x).mean(1)
    if name == "ols_slope":
        return sxy / sxx
    if name == "ols_intercept":
        return y.mean(1) - sxy / sxx * x.mean(1)
    lam = 1.7 if name == "deming" else 1.0
    D = syy - lam * sxx
    return (D + np.sqrt(D * D + 4 * lam * sxy ** 2)) / (2 * sxy)


@pytest.mark.parametrize("name", list(GRADS))
def test_gradients_match_finite_differences_and_values(name):
    rng = np.random.default_rng(7)
    x = rng.normal(0, 0.5, 60)
    y = 1.3 * x + rng.normal(0.2, 0.3, 60)
    val, g = GRADS[name](x, y)
    assert np.isclose(val, _stats_vec(x[None], y[None], name)[0], rtol=1e-12)
    gn = TV.grad_numeric(lambda v: _stats_vec(v[0][None], v[1][None], name)[0], [x, y], eps=1e-6)
    for a, b in zip(g, gn):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-9)
    if name in ("ols_slope", "ols_intercept"):
        assert np.isclose(val, np.polyfit(x, y, 1)[0 if name == "ols_slope" else 1], rtol=1e-10)
    assert np.allclose(TV.grad_numeric(lambda v: float(np.sum(v ** 2)), x), 2 * x, rtol=1e-6)


def _two_sections(joint: bool, T_=40, G=10, seed=11):
    """Section B has the structure 1.3 f + h + 0.2 of section A's f (fixed h = platform difference) and, when joint,
    the same tiles: shared tile intensity u_t and correlated tile-by-pair modulations."""
    rng = np.random.default_rng(seed)
    def sym(M):
        return (M + M.T) / np.sqrt(2)
    f = sym(rng.normal(0, 0.5, (G, G)))
    h = sym(rng.normal(0, 0.3, (G, G)))
    s = rng.uniform(0.5, 2.0, G)
    TB = T_ if joint else T_ + 13
    uA = rng.uniform(0.5, 1.5, T_)
    uB = uA if joint else rng.uniform(0.5, 1.5, TB)
    eA = rng.normal(0, 0.3, (T_, G, G))
    eB = 0.6 * eA + 0.8 * rng.normal(0, 0.3, (T_, G, G)) if joint else rng.normal(0, 0.3, (TB, G, G))
    lamA = 30 * np.outer(s, s)[None] * np.exp(f)[None] * uA[:, None, None] * np.exp(eA)
    lamB = 30 * np.outer(s, s)[None] * np.exp(1.3 * f + h + 0.2)[None] * uB[:, None, None] * np.exp(eB)
    CA = _sym(rng.poisson(lamA)[:, None].astype(np.float64))
    CB = _sym(rng.poisson(lamB)[:, None].astype(np.float64))
    NA = rng.poisson(500 * s[None] * uA[:, None]).astype(np.float64)
    NB = rng.poisson(500 * s[None] * uB[:, None]).astype(np.float64)
    return (CA, NA), (CB, NB)


@pytest.mark.parametrize("joint", [True, False])
def test_delta_var_matches_loop_bootstrap(joint):
    (CA, NA), (CB, NB) = _two_sections(joint)
    G = CA.shape[2]
    iu = np.triu_indices(G)
    lA = TV.influence_log_g(CA, NA)[:, 0][:, iu[0], iu[1]]
    lB = TV.influence_log_g(CB, NB)[:, 0][:, iu[0], iu[1]]
    PA, PB = CA[:, 0].sum(0)[iu], CB[:, 0].sum(0)[iu]
    assert min(PA.min(), PB.min()) >= 100                                 # well-populated pairs only
    x = TV.closed_form_log_g(CA, NA).log_g[0][iu]
    y = TV.closed_form_log_g(CB, NB).log_g[0][iu]
    WA = bootstrap_weights(CA.shape[0], B_MC, np.random.default_rng(0))
    WB = WA if joint else bootstrap_weights(CB.shape[0], B_MC, np.random.default_rng(1))
    xb = _mc_log_g(CA, NA, 0, WA)[:, iu[0], iu[1]]
    yb = _mc_log_g(CB, NB, 0, WB)[:, iu[0], iu[1]]
    for name, fn in GRADS.items():
        val, (gx, gy) = fn(x, y)
        var = TV.delta_var([gx, gy], [lA, lB], joint=joint)
        sd_mc = _stats_vec(xb, yb, name).std()
        assert abs(np.sqrt(var) / sd_mc - 1) < 0.03, (name, joint, np.sqrt(var), sd_mc)
        lo, hi = TV.stat_ci(val, var)
        assert np.isclose(hi - val, 1.959963984540054 * np.sqrt(var)) and np.isclose(val - lo, hi - val)
    # the single-section form: Var of mean log g over the pairs of A
    g = np.full(x.size, 1.0 / x.size)
    assert np.isclose(TV.delta_var(g, lA, joint=False), TV.delta_var([g], [lA], joint=True))
    assert abs(np.sqrt(TV.delta_var(g, lA, joint=False)) / xb.mean(1).std() - 1) < 0.03
    if not joint:
        with pytest.raises(ValueError):
            TV.delta_var([gx, gy], [lA, lB], joint=True)


def test_joint_differs_from_independent_for_registered_sections():
    (CA, NA), (CB, NB) = _two_sections(True)
    iu = np.triu_indices(CA.shape[2])
    lA = TV.influence_log_g(CA, NA)[:, 0][:, iu[0], iu[1]]
    lB = TV.influence_log_g(CB, NB)[:, 0][:, iu[0], iu[1]]
    x = TV.closed_form_log_g(CA, NA).log_g[0][iu]
    y = TV.closed_form_log_g(CB, NB).log_g[0][iu]
    _, g = TV.grad_mean_diff(x, y)
    vj, vi = TV.delta_var(g, [lA, lB], True), TV.delta_var(g, [lA, lB], False)
    assert vj < 0.8 * vi                                                   # positively correlated tiles cancel


# ----------------------------------------------------------------------------------------------- 4. cross-shell
def test_cross_shell_cov_explicit_and_monte_carlo():
    C, N = _toy(T_=40, K=4, G=6, seed=3, spread=0.3)
    infl = TV.influence_log_g(C, N)
    for pair in [(0, 1), (2, 2), (5, 3)]:
        cov = TV.cross_shell_cov(infl, pair)
        Lx = np.stack([_explicit_l(C, N, k)[:, pair[0], pair[1]] for k in range(4)], 1)
        np.testing.assert_allclose(cov, Lx.T @ Lx, rtol=1e-10)
        np.testing.assert_allclose(np.sqrt(np.diag(cov)), TV.closed_form_log_g(C, N).sd[:, pair[0], pair[1]], rtol=1e-9)
        corr = TV.cross_shell_corr(infl, pair)
        assert np.allclose(np.diag(corr), 1.0)
        W = bootstrap_weights(40, B_MC, np.random.default_rng(4))
        mc = np.stack([_mc_log_g(C, N, k, W)[:, pair[0], pair[1]] for k in range(4)], 1)
        cmc = np.corrcoef(mc.T)
        assert np.abs(corr - cmc).max() < 0.03, (corr, cmc)


# ------------------------------------------------------------------------------------------ 5. tile-additive sums
def test_tile_sum_var_and_stat_against_monte_carlo():
    # pixel_intercept-style: per-pixel target t and co-marker c in tiles; statistics m0 / m_all and the OLS intercept
    rng = np.random.default_rng(8)
    T_ = 60
    tile_of = rng.integers(0, T_, 30_000)
    u = rng.uniform(0.6, 1.4, T_)[tile_of]
    c = rng.poisson(1.5 * u)
    t = rng.poisson(0.7 + 0.8 * c * u)
    S_tiles = np.stack([np.bincount(tile_of, weights=w, minlength=T_) for w in
                        (np.ones_like(t), t, (c == 0) * 1.0, (c == 0) * t, c, c * t, c * c)], 1)

    def f(S):
        n, st, n0, st0, sc, sct, scc = S
        m_all, m0 = st / n, st0 / n0
        beta = (sct / n - sc / n * m_all) / (scc / n - (sc / n) ** 2)
        return np.array([m0 / m_all, (m_all - beta * sc / n) / m_all])

    val, sd = TV.tile_sum_stat(f, S_tiles)
    W = bootstrap_weights(T_, B_MC, np.random.default_rng(9))
    mc = np.array([f(S) for S in W @ S_tiles])
    np.testing.assert_allclose(val, f(S_tiles.sum(0)))
    np.testing.assert_allclose(sd / mc.std(0), 1.0, atol=0.03)
    # analytic gradient of a ratio of sums: Var(S1/S0) = grad^T Cov grad, against the explicit two-pass sum
    S = S_tiles.sum(0)
    g = np.zeros(7)
    g[1] = 1 / S[0]
    g[0] = -S[1] / S[0] ** 2
    lt = S_tiles[:, 1] / S[0] - S[1] * S_tiles[:, 0] / S[0] ** 2
    assert np.isclose(TV.tile_sum_var(S_tiles, g), ((lt - lt.mean()) ** 2).sum(), rtol=1e-10)
    cov = TV.tile_sum_var(S_tiles, np.stack([g, 2 * g]))
    assert cov.shape == (2, 2) and np.isclose(cov[1, 1], 4 * cov[0, 0]) and np.isclose(cov[0, 1], 2 * cov[0, 0])


def test_pooled_log_g_against_definition_and_monte_carlo():
    C, N = _toy(T_=50, K=3, G=10, seed=6, spread=0.4)
    pa = np.array([0, 1, 2, 4, 7])
    pb = np.array([3, 3, 5, 4, 9])
    res = TV.pooled_log_g(C, N, pa, pb, return_influence=True)
    T_ = C.shape[0]
    n = N.sum(0)
    for k in range(3):                                                   # mastcells block_bootstrap arithmetic
        P = C[:, k].sum(0)
        Pt = P.sum()
        ref = Pt * n[pa] * n[pb] / n.sum() ** 2
        assert np.isclose(res["log_g"][k], np.log(P[pa, pb].sum() / ref.sum()), rtol=1e-13)
        St = C[:, k][:, pa, pb].sum(1)
        Qt = C[:, k].sum((1, 2))
        infl = (St / St.sum() - Qt / Qt.sum() + 2 * N.sum(1) / n.sum()
             - sum((N[:, a] * n[b] + n[a] * N[:, b]) for a, b in zip(pa, pb)) / (n[pa] * n[pb]).sum())
        np.testing.assert_allclose(res["influence"][:, k], infl, rtol=1e-10, atol=1e-14)
        assert abs(infl.sum()) < 1e-12
        np.testing.assert_allclose(res["sd"][k], np.sqrt((infl ** 2).sum()), rtol=1e-12)
        np.testing.assert_allclose(res["n_eff_pairs"][k], St.sum() ** 2 / ((St - St.mean()) ** 2).sum(), rtol=1e-12)
    W = bootstrap_weights(T_, B_MC, np.random.default_rng(10))
    for k in range(3):
        Pb = (W @ C[:, k].reshape(T_, -1)).reshape(-1, 10, 10)
        nb = W @ N
        refb = Pb.sum((1, 2))[:, None] * nb[:, pa] * nb[:, pb] / nb.sum(1)[:, None] ** 2
        mc = np.log(Pb[:, pa, pb].sum(1) / refb.sum(1))
        assert abs(res["sd"][k] / mc.std() - 1) < 0.03
        assert abs(res["P"][k] ** 2 / res["n_eff_pairs"][k] / Pb[:, pa, pb].sum(1).var() - 1) < 0.03
    # a single pair equals the per-pair closed form
    one = TV.pooled_log_g(C, N, [2], [5])
    cf = TV.closed_form_log_g(C, N)
    np.testing.assert_allclose(one["log_g"], cf.log_g[:, 2, 5], rtol=1e-13)
    np.testing.assert_allclose(one["sd"], cf.sd[:, 2, 5], rtol=1e-9)


def test_influence_and_pooled_symmetrize_flag():
    rng = np.random.default_rng(12)
    C = rng.poisson(50.0, (30, 2, 5, 5)).astype(np.float64)
    N = rng.poisson(300.0, (30, 5)).astype(np.float64)
    C = C + np.swapaxes(np.roll(C, 1, axis=0), 2, 3)                   # symmetric total, asymmetric single tiles
    for sym in (True, False):
        infl = TV.influence_log_g(C, N, symmetrize=sym)
        cf = TV.closed_form_log_g(C, N, symmetrize=sym, min_tiles_warn=0)
        np.testing.assert_allclose(TV.closed_form_from_influence(infl), cf.sd, rtol=1e-9)
        one = TV.pooled_log_g(C, N, [1], [3], symmetrize=sym)
        np.testing.assert_allclose(one["sd"], cf.sd[:, 1, 3], rtol=1e-9)
        np.testing.assert_allclose(one["n_eff_pairs"], cf.n_eff_pairs[:, 1, 3], rtol=1e-9)
    infl = TV.influence_log_g(C, N)
    np.testing.assert_allclose(infl, np.swapaxes(infl, 2, 3), rtol=1e-12)


def test_pooled_log_g_empty_shell_is_nan():
    C, N = _toy(T_=30, K=3, G=8, seed=13)
    C[:, 1] = 0.0                                                        # shell 1 holds no pair at all (Q = 0)
    pa, pb = [0, 2, 5], [1, 2, 7]
    res = TV.pooled_log_g(C, N, pa, pb, return_influence=True)
    assert np.isnan(res["log_g"][1]) and np.isnan(res["sd"][1]) and res["P"][1] == 0
    assert np.isnan(res["influence"][:, 1]).all()
    ref = TV.pooled_log_g(C[:, [0, 2]], N, pa, pb)                      # the other shells are unaffected
    np.testing.assert_allclose(res["log_g"][[0, 2]], ref["log_g"], rtol=1e-14)
    np.testing.assert_allclose(res["sd"][[0, 2]], ref["sd"], rtol=1e-14)


# ---- device-side finish of the torch backend -------------------------------------------------------------------
@pytest.mark.parametrize("symmetrize", [True, False])
@pytest.mark.parametrize("out_dtype", [np.float64, np.float32])
def test_gpu_finish_matches_cpu_backend(symmetrize, out_dtype):
    """The GPU backend (PyTorch CPU device) returns the CPU backend's sd, n_eff, t_eff (1e-12 relative), flag and log g
    (exact; log g from P on the host), with and without Var(P), for an input with an empty
    tile and an absent feature (P = 0 rows: NaN log g and sd, flag 2)."""
    torch = pytest.importorskip("torch")
    C, N = _toy(T_=30, K=3, G=10, seed=3)
    C[5] = 0.0
    C[:, :, 2, :] = 0.0
    C[:, :, :, 2] = 0.0
    N[:, 2] = 0.0          # feature 2 absent
    T_, K = C.shape[:2]
    res = {}
    gpu_cpu = scote.GPUOptions(device="cpu")                     # the GPU backend on the PyTorch CPU device
    for name, kw in (("numpy", dict(backend="cpu", kernel="numpy")), ("torch", dict(backend=gpu_cpu)),
                     ("torch_novar", dict(backend=gpu_cpu, return_var_P=False))):
        acc = TV.TileMomentAccumulator(N, K, symmetrize=symmetrize, min_tiles_warn=0, **kw)
        out = []
        for k in range(K):
            for t in range(T_):
                if C[t, k].any():
                    acc.update(k, t, C[t, k] if kw["backend"] == "cpu" else torch.from_numpy(C[t, k].copy()))
            out.append(acc.finish(k, out_dtype=out_dtype))
        res[name] = out
    rtol = 1e-12 if out_dtype == np.float64 else 1e-6
    for a, b, c in zip(res["numpy"], res["torch"], res["torch_novar"]):
        assert b["var_P"] is not None and c["var_P"] is None
        np.testing.assert_allclose(b["var_P"], a["var_P"], rtol=1e-12)
        for r in (b, c):
            assert np.array_equal(r["P"], a["P"])
            assert np.array_equal(r["log_g"], a["log_g"], equal_nan=True)
            assert np.array_equal(r["flag"], a["flag"])
            assert r["flag"].dtype == np.int8 and r["t_eff"].dtype == np.float32
            for key in ("sd", "n_eff_pairs"):
                assert r[key].dtype == np.dtype(out_dtype)
                np.testing.assert_allclose(r[key], a[key], rtol=rtol, equal_nan=True)
            np.testing.assert_allclose(r["t_eff"], a["t_eff"], rtol=rtol, equal_nan=True)
            assert r["P_tot"] == a["P_tot"]
        assert np.isnan(a["log_g"][2]).all() and (a["flag"][2] == 2).all()
