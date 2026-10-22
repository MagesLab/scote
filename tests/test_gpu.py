"""GPU backend (scote._twopoint.gpu) on the torch CPU device (and CUDA where available): bit-identical to the CPU
backend for global and per-tile counts, shell operators built on the device, and the end-to-end pipelines."""
import numpy as np
import pytest
import scipy.sparse as sp

import scote
from scote._twopoint import cpu, gpu
from scote._twopoint.lattice import build_lattice, onehot_csr, shell_offsets, shell_operator

try:
    import torch
except ImportError:
    torch = None

# every test of this module is collected in every environment and skipped one by one where PyTorch or CUDA is
# missing, so that "passed + skipped" is the same total everywhere
pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch is not installed (the gpu extra)")
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(torch is None or not torch.cuda.is_available(),
                                                                  reason="no CUDA device"))]
DENSE = {"semidense": 0, "dense": 4 * 2**30}      # dense_bytes_limit that selects the path
SPARSE = scote.CPUOptions(mode="sparse")


def _case(n=30_000, G=7, side=200.0, h=2.0, tile=100.0, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, side, (n, 2))
    codes = rng.choice(G, n, p=np.arange(G, 0, -1) / np.arange(G, 0, -1).sum())
    lat, X = build_lattice(np.floor(xy / h).astype(np.int64), onehot_csr(codes, G), margin=8)
    center = (lat.cells + 0.5) * h
    tiles = np.floor(center[:, 0] / tile).astype(np.int64) * 100 + np.floor(center[:, 1] / tile).astype(np.int64)
    return lat, X, tiles


@pytest.mark.parametrize("device", DEVICES)
def test_shell_operator_on_device_matches_host(device):
    lat, X, _ = _case()
    offs = shell_offsets(8)
    for k in (1, 3, 7):
        S = shell_operator(lat, offs[k])
        for rows in (None, (1000, 4000)):
            Sd = gpu.shell_operator_rows(lat, offs[k], device, rows).to_dense().cpu().numpy()
            ref = S.toarray() if rows is None else S[rows[0]:rows[1]].toarray()
            assert np.array_equal(Sd, ref)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("path", ["semidense", "dense"])
def test_level_matches_cpu(device, path):
    lat, X, _ = _case()
    ref = cpu.pair_counts_level(X, lat, 8, include_intra=True, options=SPARSE)
    P = gpu.pair_counts_level(X, lat, 8, include_intra=True, options=scote.GPUOptions(
        device=device, dense_bytes_limit=DENSE[path], row_block_bytes=64 * 1024))   # tiny blocks -> many blocks
    assert np.array_equal(P, ref)
    tot = gpu.pair_counts_level(X, lat, 8, include_intra=True, totals_only=True,
                                options=scote.GPUOptions(device=device, dense_bytes_limit=DENSE[path]))
    assert np.array_equal(tot, ref.sum(axis=(1, 2)))
    assert np.array_equal(tot, cpu.pair_counts_level(X, lat, 8, include_intra=True, totals_only=True))


@pytest.mark.parametrize("device", DEVICES)
def test_float64_exact_large_counts(device):
    """Dense coarse level with counts above 2^24 per entry: exact in fp64."""
    lat, X, _ = _case(n=400_000, G=3, side=60.0)
    ref = cpu.pair_counts_level(X, lat, 8, include_intra=True, options=SPARSE)
    assert ref.max() > 2 ** 24
    P = gpu.pair_counts_level(X, lat, 8, include_intra=True, options=scote.GPUOptions(device=device))
    assert np.array_equal(P, ref)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("path", ["semidense", "dense"])
def test_groups_match_cpu_bit_exact(device, path):
    lat, X, tiles = _case(n=40_000, G=6, side=300.0)
    ref = cpu.pair_counts_level(X, lat, 8, include_intra=True, row_groups=tiles, options=SPARSE)
    # tiny row blocks -> groups are split into several segments and blocks pack several groups
    P = gpu.pair_counts_level(X, lat, 8, include_intra=True, row_groups=tiles, options=scote.GPUOptions(
        device=device, dense_bytes_limit=DENSE[path], row_block_bytes=48 * 1024))
    assert P.shape == ref.shape == (len(np.unique(tiles)), 8, X.shape[1], X.shape[1])
    assert np.array_equal(P, ref)
    P4 = gpu.pair_counts_level(X, lat, 8, [4, 5, 6, 7], row_groups=tiles,
                               options=scote.GPUOptions(device=device, dense_bytes_limit=DENSE[path]))
    assert np.array_equal(P4, ref[:, 4:8])


@pytest.mark.parametrize("device", DEVICES)
def test_pipelines_gpu_equal_cpu(device):
    rng = np.random.default_rng(5)
    n, G, pitch = 30_000, 3, 2.0
    xy = rng.uniform(0, 400, (n, 2))
    codes = rng.choice(G, n, p=[0.6, 0.3, 0.1])
    X0 = sp.csr_matrix(onehot_csr(codes, G))
    with pytest.warns(UserWarning):
        c = scote.twopoint(xy, X0, delta=pitch, bin_size=pitch, tile_um=100.0, n_levels=4, backend=SPARSE,
                           keep_tiles=True)
    with pytest.warns(UserWarning):
        g = scote.twopoint(xy, X0, delta=pitch, bin_size=pitch, tile_um=100.0, n_levels=4,
                           backend=scote.GPUOptions(device=device), keep_tiles=True)
    assert np.array_equal(c.counts_tiles, g.counts_tiles) and np.array_equal(c.totals_tiles, g.totals_tiles)
    for key in ("log_g", "P", "flag"):
        assert np.array_equal(getattr(c, key), getattr(g, key), equal_nan=True), key
    np.testing.assert_allclose(g.sd, c.sd, rtol=1e-9)   # device vs host accumulation order
    with pytest.warns(UserWarning):
        cm = scote.twopoint(xy, codes, n_levels=3, tile_um=100.0, keep_tiles=True)
    with pytest.warns(UserWarning):
        gm = scote.twopoint(xy, codes, n_levels=3, tile_um=100.0, backend=scote.GPUOptions(device=device),
                            keep_tiles=True)
    assert np.array_equal(cm.counts_tiles, gm.counts_tiles)
    np.testing.assert_allclose(gm.sd, cm.sd, rtol=1e-9)
    pc = scote.pair_counts(xy, X0, delta=pitch, bin_size=pitch, n_levels=4)
    pg = scote.pair_counts(xy, X0, delta=pitch, bin_size=pitch, n_levels=4, backend=scote.GPUOptions(device=device))
    assert np.array_equal(pc.counts, pg.counts) and pc.bins.equals(pg.bins)


@pytest.mark.parametrize("device", DEVICES)
def test_timings_book_the_same_stages(device):
    """Both backends book the shell operators under seconds_S and the products under seconds_SX / seconds_XtSX,
    with the same nnz(S) per shell."""
    lat, X, _ = _case()
    for path in ("semidense", "dense"):
        tc = scote._twopoint.LevelTiming(0, 2.0, 0, 0, 0.0, "")
        tg = scote._twopoint.LevelTiming(0, 2.0, 0, 0, 0.0, "")
        cpu.pair_counts_level(X, lat, 8, timing=tc, options=SPARSE)
        gpu.pair_counts_level(X, lat, 8, timing=tg,
                              options=scote.GPUOptions(device=device, dense_bytes_limit=DENSE[path]))
        assert tc.cell_pairs == tg.cell_pairs > 0
        assert [r["nnz_S"] for r in tc.per_shell] == [r["nnz_S"] for r in tg.per_shell if r["k"] >= 0]
        assert tg.seconds_S > 0 and tg.seconds_SX > 0 and tg.seconds_XtSX > 0
