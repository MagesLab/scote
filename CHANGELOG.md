# Changelog

## 0.1.0

First public release. It reproduces the numbers of the preprint https://doi.org/10.64898/2026.10.09.757915.

- Two-point functions $g_{ab}(r)$ of all feature pairs, normalized by the random-label expectation, from one function
  for molecule tables (one label per molecule: exact pair enumeration at short distances plus lattice shell
  products) and for bin and spot tables (a count matrix on a lattice of pitch `delta`): `scote.twopoint`, result
  class `scote.TwoPointResult`. Inputs are checked, with the fix named in the error (one label per point as a 1-D
  array, AnnData, duplicate feature names, negative counts).
- Closed-form tile block bootstrap for the uncertainties, with the effective-tile diagnostics and reliability flags,
  streamed output of the result arrays, and delta-method uncertainties of statistics derived from $\log g$
  (`scote.uncertainty`).
- CPU (numba, BLAS, SciPy) and GPU (PyTorch) shell products with bit-identical integer counts, selected by one
  `backend` argument (also for the accumulators of `scote.uncertainty`): `"cpu"`, `"gpu"`, or the options
  `scote.CPUOptions` / `scote.GPUOptions`; global pair counts as `scote.PairCounts` (`scote.pair_counts`).
- Readers of the outputs of Xenium, MERSCOPE, CosMx, Atera, AVITI24 / Cytos, Visium HD, Visium, and Stereo-seq (GEM
  and GEF) into memory, and converters to the common molecule and bin data model (`scote.io`); a command line for
  workflows (`python -m scote`).
- Thread count per call for numba, BLAS, and PyTorch: at most the CPUs available to the process (affinity mask,
  SLURM allocation, cgroup CPU quota), and at most 8 where no limit is visible; restored after the call.
- Documentation with one API page per module, function, and class (https://mageslab.github.io/scote).
