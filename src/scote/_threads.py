"""CPU threads of a call: numba's thread count, the BLAS thread pools (through ``threadpoolctl``), and the PyTorch CPU
threads, set per call and restored afterwards."""
from __future__ import annotations

import contextlib
import functools
import inspect
import math
import os
import re
import sys
from pathlib import Path

import numba
from threadpoolctl import threadpool_limits

from ._twopoint._common import GPUOptions

__all__ = ["available_cpus", "default_threads", "thread_limit", "with_threads"]

_SLURM_VARIABLES = ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE")
# the most threads of the default where the process cannot see whether its CPUs are limited (``default_threads``)
BLIND_DEFAULT_THREADS = 8


def _slurm_cpus() -> int | None:
    """CPUs of the SLURM allocation on this node, or ``None`` outside SLURM.

    The first of ``SLURM_CPUS_PER_TASK``, ``SLURM_CPUS_ON_NODE``, and ``SLURM_JOB_CPUS_PER_NODE`` that is set and
    starts with a positive integer is used; for ``SLURM_JOB_CPUS_PER_NODE`` (e.g. ``"16(x2),8"``) that is the count
    of the first node group.

    Returns
    -------
    int or None
        The CPU count, or ``None`` if none of the variables is set.
    """
    for name in _SLURM_VARIABLES:
        value = os.environ.get(name, "")
        match = re.match(r"\s*(\d+)", value)
        if match and int(match.group(1)) > 0:
            return int(match.group(1))
    return None


def _read_quota(cgroup_dir: Path) -> float | None:
    """CPU quota of one cgroup directory in CPUs (quota / period).

    Reads ``cpu.max`` ("QUOTA PERIOD" or "max PERIOD", cgroup v2) or ``cpu.cfs_quota_us`` and ``cpu.cfs_period_us``
    (quota -1 = unlimited, cgroup v1).

    Parameters
    ----------
    cgroup_dir : pathlib.Path
        The cgroup directory.

    Returns
    -------
    float or None
        The quota, ``math.inf`` if the directory sets none, or ``None`` if it has no readable CPU control files.
    """
    try:
        v2 = cgroup_dir / "cpu.max"
        if v2.is_file():
            quota, *rest = v2.read_text().split()
            period = float(rest[0]) if rest else 100_000.0
            return math.inf if quota == "max" or period <= 0 else float(quota) / period
        v1_quota, v1_period = cgroup_dir / "cpu.cfs_quota_us", cgroup_dir / "cpu.cfs_period_us"
        if v1_quota.is_file() and v1_period.is_file():
            quota, period = float(v1_quota.read_text()), float(v1_period.read_text())
            return math.inf if quota <= 0 or period <= 0 else quota / period
    except (OSError, ValueError, IndexError):
        return None
    return None


def _cgroup_quota(proc_cgroup="/proc/self/cgroup", root="/sys/fs/cgroup") -> float | None:
    """CPU quota of the cgroups of this process in CPUs.

    The cgroup of the process (``/proc/self/cgroup``: the v2 entry ``0::PATH``, or the v1 entries of the ``cpu``
    controller) and all its ancestors are searched under the cgroup mount ``root``; the smallest quota counts.

    Parameters
    ----------
    proc_cgroup : str or pathlib.Path
        The cgroup file of the process.
    root : str or pathlib.Path
        Mount point of the cgroup file system.

    Returns
    -------
    float or None
        The smallest quota, ``math.inf`` if the CPU control files can be read and set none, or ``None`` if none can
        be read (no cgroup file system, as in some containers).
    """
    root = Path(root)
    try:
        lines = Path(proc_cgroup).read_text().splitlines()
    except OSError:
        return None
    candidates = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        controllers, path = parts[1], parts[2].strip()
        if controllers == "":                                     # cgroup v2
            bases = [root]
        elif "cpu" in controllers.split(","):                     # cgroup v1, cpu controller
            bases = [root / controllers, root / "cpu", root / "cpu,cpuacct", root / "cpuacct,cpu"]
        else:
            continue
        rel = Path(path.lstrip("/"))
        chain = [rel, *rel.parents] if str(rel) not in ("", ".") else [Path(".")]
        for base in bases:
            candidates += [base / p for p in chain]
    quotas = [q for q in (_read_quota(d) for d in dict.fromkeys(candidates)) if q is not None]
    return min(quotas) if quotas else None


def _cgroup_cpus(proc_cgroup="/proc/self/cgroup", root="/sys/fs/cgroup") -> int | None:
    """CPU limit of the cgroups of this process, or ``None`` without a limit (or without cgroup information).

    Parameters
    ----------
    proc_cgroup : str or pathlib.Path
        The cgroup file of the process.
    root : str or pathlib.Path
        Mount point of the cgroup file system.

    Returns
    -------
    int or None
        The quota of ``_cgroup_quota`` rounded up to whole CPUs (at least 1), or ``None``.
    """
    quota = _cgroup_quota(proc_cgroup, root)
    if quota is None or math.isinf(quota):
        return None
    return max(1, math.ceil(quota - 1e-9))


def _affinity_cpus() -> int:
    """CPUs in the affinity mask of the process (``os.cpu_count()`` where the mask is not available)."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _limit_visible() -> bool:
    """Whether the process can see a CPU limit or its absence.

    Returns
    -------
    bool
        True inside a SLURM allocation, with an affinity mask smaller than the machine, or with readable cgroup CPU
        control files (with or without a quota); False otherwise, e.g. in a container without the cgroup file system,
        where a CPU quota cannot be seen.
    """
    return (_slurm_cpus() is not None or _affinity_cpus() < (os.cpu_count() or 1)
            or _cgroup_quota() is not None)


def available_cpus() -> int:
    """Number of CPUs this process may use.

    Returns
    -------
    int
        The minimum of the CPUs in the affinity mask of the process (``os.sched_getaffinity``, or
        ``os.cpu_count()`` where the affinity mask is not available), the CPUs of the SLURM allocation, if the
        process runs in one (``SLURM_CPUS_PER_TASK``, ``SLURM_CPUS_ON_NODE``, or ``SLURM_JOB_CPUS_PER_NODE``), and the
        CPU quota of its cgroups (cgroup v2 ``cpu.max``, v1 ``cpu.cfs_quota_us`` / ``cpu.cfs_period_us``, rounded up;
        containers and login nodes often set one).
    """
    n = _affinity_cpus()
    for limit in (_slurm_cpus(), _cgroup_cpus()):
        if limit is not None:
            n = min(n, limit)
    return max(1, n)


def default_threads() -> int:
    """Default thread count of a call: ``min(numba.get_num_threads(), available_cpus())``, and at most
    ``BLIND_DEFAULT_THREADS`` (8) where the process cannot see whether its CPUs are limited.

    numba starts with one thread per CPU of the machine, which oversubscribes a partial-node allocation (for example
    16 cores of a 128-core node in SLURM) and slows the kernels down. The default therefore never exceeds the CPUs
    available to the process; a lower thread count set with ``numba.set_num_threads`` or ``NUMBA_NUM_THREADS`` is
    kept. Without a SLURM allocation, an affinity mask smaller than the machine, or readable cgroup CPU control
    files (for example in a container without the cgroup file system), a CPU quota may be in place that the process
    cannot see; the default is then at most ``BLIND_DEFAULT_THREADS``, and ``threads`` raises it.

    Returns
    -------
    int
        The thread count, at least 1.
    """
    n = min(numba.get_num_threads(), available_cpus())
    if not _limit_visible():
        n = min(n, BLIND_DEFAULT_THREADS)
    return max(1, n)


@contextlib.contextmanager
def thread_limit(threads: int | None = None):
    """Context manager that sets the CPU threads for its body and restores the previous values afterwards.

    Three thread pools are limited to the same count: numba's (the parallel kernels), the BLAS libraries loaded in
    the process (``threadpoolctl.threadpool_limits(user_api="blas")``; the dense products of the CPU backend), and
    PyTorch's intra-op threads (if PyTorch is imported; the GPU backend on the PyTorch CPU device).

    Parameters
    ----------
    threads : int or None
        Thread count. ``None`` uses :func:`scote.default_threads`.

    Yields
    ------
    int
        The thread count in effect inside the context.

    Raises
    ------
    ValueError
        If ``threads`` is not between 1 and ``numba.config.NUMBA_NUM_THREADS`` (the maximum numba fixes at start-up,
        set by the environment variable ``NUMBA_NUM_THREADS``).
    """
    if threads is None:
        n = default_threads()
    else:
        n = int(threads)
        if not 1 <= n <= numba.config.NUMBA_NUM_THREADS:
            raise ValueError(f"threads={threads} is outside 1 .. {numba.config.NUMBA_NUM_THREADS} (numba's maximum, "
                             "set by the environment variable NUMBA_NUM_THREADS before numba is imported)")
    before = numba.get_num_threads()
    torch = sys.modules.get("torch")
    torch_before = torch.get_num_threads() if torch is not None else None
    numba.set_num_threads(n)
    try:
        with threadpool_limits(limits=n, user_api="blas"):
            if torch is not None:
                torch.set_num_threads(n)
            yield n
    finally:
        numba.set_num_threads(before)
        if torch is not None:
            torch.set_num_threads(torch_before)


def with_threads(func):
    """Decorator: run ``func`` inside :func:`scote.thread_limit` with the value of its ``threads`` argument.

    With the GPU backend (``backend="gpu"`` or a :class:`scote.GPUOptions`), PyTorch is imported before the threads are
    set, so that its CPU threads are limited too.

    Parameters
    ----------
    func : callable
        A function with a ``threads`` parameter (int or None).

    Returns
    -------
    callable
        The wrapped function.

    Raises
    ------
    TypeError
        If ``func`` has no ``threads`` parameter.
    """
    signature = inspect.signature(func)
    if "threads" not in signature.parameters:
        raise TypeError(f"{func.__name__} has no 'threads' parameter")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Call ``func`` inside :func:`scote.thread_limit` with its ``threads`` argument."""
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        backend = bound.arguments.get("backend")
        if isinstance(backend, GPUOptions) or (isinstance(backend, str) and backend == "gpu"):
            with contextlib.suppress(ImportError):
                import torch  # noqa: F401
        with thread_limit(bound.arguments["threads"]):
            return func(*args, **kwargs)

    return wrapper
