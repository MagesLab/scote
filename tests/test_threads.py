"""Thread count per call: default min(numba's setting, CPUs available to the process), restored afterwards."""
import os
import subprocess
import sys

import numba
import numpy as np
import pytest

import scote
from scote import _threads


def test_default_threads_is_bounded_by_numba_affinity_and_slurm(monkeypatch):
    for name in _threads._SLURM_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    assert 1 <= scote.default_threads() <= numba.get_num_threads()
    assert scote.default_threads() <= scote.available_cpus() <= (os.cpu_count() or 1)
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "1(x2),4")
    assert scote.available_cpus() == 1 and scote.default_threads() == 1
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")
    assert scote.available_cpus() == min(3, len(os.sched_getaffinity(0)))
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "")
    assert scote.available_cpus() == 1                      # falls through to SLURM_JOB_CPUS_PER_NODE


def test_thread_limit_sets_and_restores():
    before = numba.get_num_threads()
    with scote.thread_limit(1) as n:
        assert n == 1 and numba.get_num_threads() == 1
    assert numba.get_num_threads() == before
    with pytest.raises(ValueError, match="NUMBA_NUM_THREADS"):
        with scote.thread_limit(numba.config.NUMBA_NUM_THREADS + 1):
            pass
    with pytest.raises(ValueError):
        with scote.thread_limit(0):
            pass
    assert numba.get_num_threads() == before


def test_public_calls_restore_the_thread_count():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 200, (3_000, 2))
    codes = rng.integers(0, 3, 3_000)
    before = numba.get_num_threads()
    res = scote.twopoint(xy, codes, n_levels=1, tile_um=30.0, threads=1)
    assert res.params["threads"] == 1 and numba.get_num_threads() == before
    res = scote.twopoint(xy, codes, n_levels=1, tile_um=30.0)
    assert res.params["threads"] == scote.default_threads() and numba.get_num_threads() == before


def test_thread_limit_covers_blas_and_torch():
    """BLAS pools (threadpoolctl) and the PyTorch CPU threads follow the call's thread count and are restored."""
    from threadpoolctl import threadpool_info

    def blas_threads():
        return {d["filepath"]: d["num_threads"] for d in threadpool_info() if d["user_api"] == "blas"}

    before = blas_threads()
    if not before:
        pytest.skip("no BLAS library loaded")
    with scote.thread_limit(1):
        assert set(blas_threads().values()) == {1}
    assert blas_threads() == before
    torch = pytest.importorskip("torch")
    torch_before = torch.get_num_threads()
    with scote.thread_limit(1):
        assert torch.get_num_threads() == 1
    assert torch.get_num_threads() == torch_before


def test_import_leaves_numba_threads_unchanged():
    code = ("import numba; numba.set_num_threads(1); import scote, scote.io, scote._cli, scote._twopoint.cpu; "
            "print(numba.get_num_threads())")
    env = dict(os.environ, NUMBA_NUM_THREADS="2")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "1"


def _cgroup_tree(tmp_path, proc_line, files):
    """A fake /proc/self/cgroup and cgroup mount with the given control files."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    proc = tmp_path / "proc_cgroup"
    proc.write_text(proc_line + "\n")
    root = tmp_path / "cgroup"
    for rel, text in files.items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    return proc, root


def test_cgroup_v2_quota_of_the_process_and_its_parents(tmp_path):
    proc, root = _cgroup_tree(tmp_path, "0::/user.slice/job.scope",
                              {"user.slice/job.scope/cpu.max": "max 100000\n", "user.slice/cpu.max": "250000 100000\n",
                               "cpu.max": "max 100000\n"})
    assert _threads._cgroup_cpus(proc, root) == 3                         # 2.5 CPUs of the parent, rounded up
    proc, root = _cgroup_tree(tmp_path / "b", "0::/", {"cpu.max": "max 100000\n"})
    assert _threads._cgroup_cpus(proc, root) is None
    proc, root = _cgroup_tree(tmp_path / "c", "0::/", {"cpu.max": "50000 100000\n"})
    assert _threads._cgroup_cpus(proc, root) == 1                         # half a CPU: at least 1


def test_cgroup_v1_quota(tmp_path):
    proc, root = _cgroup_tree(tmp_path, "12:memory:/x\n4:cpu,cpuacct:/slurm/job7",
                              {"cpu,cpuacct/slurm/job7/cpu.cfs_quota_us": "400000\n",
                               "cpu,cpuacct/slurm/job7/cpu.cfs_period_us": "100000\n",
                               "cpu,cpuacct/slurm/cpu.cfs_quota_us": "-1\n",
                               "cpu,cpuacct/slurm/cpu.cfs_period_us": "100000\n"})
    assert _threads._cgroup_cpus(proc, root) == 4
    assert _threads._cgroup_cpus(tmp_path / "missing", root) is None


def test_available_cpus_honors_the_cgroup_quota(monkeypatch):
    for name in _threads._SLURM_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_threads, "_cgroup_cpus", lambda *a, **k: 2)
    assert scote.available_cpus() == min(2, len(os.sched_getaffinity(0)))
    assert scote.default_threads() <= 2


def test_cgroup_quota_tells_no_limit_from_no_information(tmp_path):
    proc, root = _cgroup_tree(tmp_path, "0::/a", {"a/cpu.max": "max 100000\n", "cpu.max": "max 100000\n"})
    assert _threads._cgroup_quota(proc, root) == float("inf") and _threads._cgroup_cpus(proc, root) is None
    proc, root = _cgroup_tree(tmp_path / "b", "0::/a", {})                # no cgroup file system
    assert _threads._cgroup_quota(proc, root) is None and _threads._cgroup_cpus(proc, root) is None


def test_default_threads_is_capped_where_no_limit_is_visible(monkeypatch):
    monkeypatch.setattr(_threads, "_limit_visible", lambda: False)
    assert scote.default_threads() == max(1, min(numba.get_num_threads(), scote.available_cpus(),
                                                 _threads.BLIND_DEFAULT_THREADS))
    monkeypatch.setattr(_threads, "_limit_visible", lambda: True)
    assert scote.default_threads() == max(1, min(numba.get_num_threads(), scote.available_cpus()))
    monkeypatch.undo()
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")                     # a SLURM allocation is a visible limit
    assert _threads._limit_visible()
