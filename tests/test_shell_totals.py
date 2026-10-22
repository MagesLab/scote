"""Shell totals of the lattice hierarchy vs exact pair counts.

The hierarchy assigns a pair to shell k of level L by its *lattice* offset o_L = floor(c/2^L) - floor(c'/2^L):
level 0 takes |o_0| < 8, every further level takes 4 <= |o_L| < 8. Because o_{L+1} = o_L/2 + delta with delta in
{-1/2, 0, 1/2} per axis (cell parities), the windows of consecutive levels overlap (|o_L| < 8 and |o_{L+1}| >= 4) and
leave gaps (|o_L| >= 8 and |o_{L+1}| < 4). Doubles outnumber misses by (D(8) - 4 D(4)) = 193 - 180 = 13 offsets per
transition (D(r) = lattice points with |o| < r), so the shell totals exceed N(N-1) by a few percent (5.1 % on the
Xenium CRC P2 geometry, 2.2 % for a flat pair spectrum). This is a property of the estimator's windows, not a counting
error: every shell count equals the brute-force count of the pairs inside its window, exactly, and g_ab = P_ab / P_ref
is a ratio within one window. These tests pin that down."""
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.spatial import cKDTree

from scote import pair_counts, random_label_reference


PITCH, NB, NL, G = 2.0, 8, 6, 3


def _edges():
    e = [k * PITCH for k in range(NB + 1)]
    p = PITCH
    for _ in range(NL):
        p *= 2
        e += [k * p for k in range(NB // 2 + 1, NB + 1)]
    return np.array(e)


def _data(N=3000, side=120.0, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, side, (N, 2))
    p = np.column_stack([0.2 + 0.5 * (xy[:, 0] < side / 2), np.full(N, 0.3), np.full(N, 0.3)])
    p /= p.sum(1, keepdims=True)
    codes = np.array([rng.choice(G, p=pi) for pi in p])
    cells = np.floor(xy / PITCH).astype(np.int64)
    X0 = sp.csr_matrix((np.ones(N), (np.arange(N), codes)), shape=(N, G))
    return xy, codes, cells, X0


def _brute(xy, codes, intervals):
    """Ordered pair counts per (level, shell) from the lattice-offset definition, plus pair multiplicities."""
    N = len(xy)
    c = np.floor(xy / PITCH).astype(np.int64)
    i, j = np.triu_indices(N, 1)
    slot = {(v["level"], v["shell"]): s for s, v in enumerate(intervals)}
    P = np.zeros((len(intervals), G, G))
    mult = np.zeros(len(i), int)
    for L in range(NL + 1):
        o = (c[i] >> L) - (c[j] >> L)
        k = np.floor(np.hypot(o[:, 0], o[:, 1])).astype(int)
        m = (k < NB) if L == 0 else ((k >= NB // 2) & (k < NB))
        mult += m
        for kk in np.unique(k[m]):
            sel = m & (k == kk)
            np.add.at(P[slot[(L, int(kk))]], (codes[i[sel]], codes[j[sel]]), 1)
    return P + P.transpose(0, 2, 1), mult


@pytest.fixture(scope="module")
def counts():
    xy, codes, cells, X0 = _data()
    res = pair_counts(xy, X0, delta=PITCH, bin_size=PITCH, shells_per_level=NB, n_levels=NL, origin=(0.0, 0.0))
    return xy, codes, res.counts, res.bins.to_dict("records")


def test_exact_shells_partition_the_pairs(counts):
    xy, codes, P, iv = counts
    N = len(xy)
    T = cKDTree(xy)
    cum = T.count_neighbors(T, _edges()[1:], cumulative=True) - N
    assert cum[-1] == N * (N - 1)                      # the 48 Euclidean shells cover every pair exactly once
    assert len(iv) == len(_edges()) - 1 == 8 + 4 * NL


def test_shell_counts_equal_their_window_enumeration(counts):
    """Each shell count is exactly the number of ordered pairs whose lattice offset lies in the shell's window."""
    xy, codes, P, iv = counts
    Pb, mult = _brute(xy, codes, iv)
    assert np.array_equal(P, Pb)


def test_total_surplus_is_the_window_overlap(counts):
    """Sum of all shells = N(N-1) + doubled - missed ordered pairs; surplus a few percent, never negative here."""
    xy, codes, P, iv = counts
    N = len(xy)
    _, mult = _brute(xy, codes, iv)
    doubled = 2 * int((mult == 2).sum())
    missed = 2 * int((mult == 0).sum())
    assert mult.max() <= 2 and doubled > 0 and missed > 0
    assert P.sum() == N * (N - 1) + doubled - missed
    ratio = P.sum() / (N * (N - 1))
    assert 1.01 < ratio < 1.10, ratio                   # 5.1 % on Xenium CRC P2; 2.2 % for a flat spectrum


def test_level_surplus_follows_D8_minus_4D4():
    """On a periodic uniform lattice the net excess per transition is exactly (D(8) - 4 D(4)) = 13 pairs per
    (level-L cell, unit of pair density): check the closed form by enumerating all offsets with uniform parities."""
    r = np.arange(-20, 21)
    X, Y = np.meshgrid(r, r)
    d = np.hypot(X, Y)
    def D(R):
        return int((d < R).sum())
    assert (D(4), D(8)) == (45, 193)
    # enumerate fine offsets o with all four parity combinations (c, c') -> coarse offset o' = (c>>1) - (c'>>1)
    o = np.stack([X.ravel(), Y.ravel()], 1)
    dbl = mis = 0.0
    for px in (0, 1):
        for py in (0, 1):
            c = np.array([px, py])
            cp = c - o                        # c' = c - o
            o1 = (c >> 1) - (cp >> 1)
            h, h1 = np.hypot(*o.T), np.hypot(*o1.T)
            dbl += ((h < 8) & (h1 >= 4)).sum() / 4
            mis += ((h >= 8) & (h1 < 4)).sum() / 4
    assert round(dbl - mis, 9) == D(8) - 4 * D(4) == 13


def test_g_is_a_ratio_within_one_window(counts):
    """g_ab(k) = P_ab / (P_tot n_a n_b / n^2) uses the same window in numerator and reference: the random-label
    reference sums to P_tot(k) shell by shell (so the surplus cancels in the normalization identity), and g from our
    counts equals g from the window enumeration bit for bit."""
    xy, codes, P, iv = counts
    totals = np.bincount(codes, minlength=G).astype(float)
    ref = random_label_reference(P, totals)
    assert np.allclose(ref.sum((1, 2)), P.sum((1, 2)), rtol=1e-12)
    Pb, _ = _brute(xy, codes, iv)
    with np.errstate(divide="ignore", invalid="ignore"):
        g, gb = P / ref, Pb / random_label_reference(Pb, totals)
    assert np.array_equal(np.nan_to_num(g), np.nan_to_num(gb))
