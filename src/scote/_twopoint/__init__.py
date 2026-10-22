"""Implementation of :func:`scote.twopoint` and :func:`scote.pair_counts` (private).

* ``enumeration``: exact enumeration of molecule pairs at short distances;
* ``lattice``: lattices of occupied cells, coarse-graining, and the shell operators :math:`S_k`;
* ``cpu`` and ``gpu``: the shell products :math:`P_k = X^\\top S_k X` of one lattice level on the CPU (numba, BLAS,
  SciPy) and with PyTorch (on a GPU or the PyTorch CPU device); both ``pair_counts_level`` functions take the same
  arguments, deliver host arrays to a sink, and give bit-identical counts;
* ``pipeline``: the per-tile hierarchy with its closed-form tile bootstrap, and the public functions.

The public objects are exported by :mod:`scote`.
"""
from ._common import LevelTiming

__all__ = ["LevelTiming"]
