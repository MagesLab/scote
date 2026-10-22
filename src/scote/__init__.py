"""scote: scale-resolved two-point correlation functions for spatial omics.

The main function, :func:`scote.twopoint`, takes positions in micrometers and either one label per point (molecule
tables of imaging platforms) or a count matrix (bin and spot tables of sequencing-based platforms), and returns a
:class:`scote.TwoPointResult` with :math:`\\log g_{ab}(r)` of all feature pairs over a hierarchy of distance bins and
its closed-form tile-bootstrap uncertainty. Reading vendor outputs into memory or converting them to the common data
model: :mod:`scote.io`. Statistics derived from :math:`\\log g` with their uncertainties: :mod:`scote.uncertainty`. Both
submodules are available after ``import scote``.

numba-parallel kernels, BLAS, and the PyTorch CPU device run on :func:`scote.default_threads` threads per call (at most
the CPUs available to the process); pass ``threads`` to a function to choose another count.
"""
from . import io, uncertainty
from ._reference import random_label_reference
from ._threads import available_cpus, default_threads, thread_limit
from ._twopoint._common import CPUOptions, GPUOptions
from ._twopoint.pipeline import pair_counts, twopoint
from ._result import PairCounts, TwoPointResult

__version__ = "0.1.0"

__all__ = ["CPUOptions", "GPUOptions", "PairCounts", "TwoPointResult", "available_cpus", "default_threads", "io",
           "pair_counts", "random_label_reference", "thread_limit", "twopoint", "uncertainty",
           "__version__"]

# every public object is documented once, under the name users import it by
for _obj in (CPUOptions, GPUOptions, PairCounts, TwoPointResult, available_cpus, default_threads, pair_counts,
             random_label_reference, thread_limit, twopoint):
    _obj.__module__ = __name__
del _obj
