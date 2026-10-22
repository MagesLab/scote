# Performance and threads

## Threads

scote chooses the thread count of every call itself; nothing has to be set for a laptop, a workstation, or a job on
part of a cluster node.

- The numba kernels run on {func}`scote.default_threads` threads per call: the minimum of numba's thread count
  (`NUMBA_NUM_THREADS` or `numba.set_num_threads`) and the CPUs available to the process
  ({func}`scote.available_cpus`: its CPU affinity mask, the CPUs of a SLURM allocation, and the CPU quota of its
  cgroups). Where the process cannot see such a limit (as in some containers), the default is at most 8; pass
  `threads=` to use more. More threads than CPUs cost a modest factor, not a stall.
- The dense products of the coarse lattice levels run in BLAS (through NumPy and SciPy), and the `gpu` backend on a
  PyTorch CPU device runs PyTorch's threads. Both pools are limited per call in the same way, BLAS with
  `threadpoolctl`, to the same count.
- Pass `threads=n` to a function (or `--threads n` on the command line) to choose another count for all of them. The
  counts are set for the duration of the call and restored afterward; importing `scote` does not change them.
  {func}`scote.thread_limit` sets them for a block of your own code.

## Backends

`backend="cpu"` (the default) runs the lattice shell products with numba kernels at sparse levels and BLAS products
at dense levels; `backend="gpu"` runs them, and the accumulators of the uncertainties, with PyTorch. Both give the same
integer counts. For other options, `backend` takes a {class}`scote.CPUOptions` or a {class}`scote.GPUOptions` instead
of the name, e.g. `backend=scote.GPUOptions(device="cuda:1")`; on the command line, `--backend gpu,device=cuda:1`. The
`gpu` backend also runs on a PyTorch CPU device (`backend=scote.GPUOptions(device="cpu")`), which is useful for testing
without a GPU; on the CPU, the `cpu` backend is faster (about 2.8-fold on a section of 3.2 million molecules of 20
genes, 4 threads, for molecules and for 2 µm bins). {func}`scote.uncertainty.closed_form_log_g` and
{class}`scote.uncertainty.TileMomentAccumulator` take the same `backend` for their accumulators.

## Memory

The arrays of a {class}`scote.TwoPointResult` (`log_g`, `sd`, the counts, ...) hold $K G^2$ values each for $K$
distance bins and $G$ features, 8 bytes per value for the float64 arrays. The per-tile counts stream into the
accumulators of the tile bootstrap distance bin by distance bin, so the $(T, K, G, G)$ tensor of $T$ tiles exists only
with `keep_tiles=True` ($8\,T K G^2$ bytes); the uncertainties do not need it. What stays in memory grows as $G^2$:

- the $(K, G, G)$ arrays of the result; `out="dir"` streams them into memory-mapped `.npy` files distance bin by
  distance bin, so that they leave the host memory;
- four $(G, G)$ accumulators per distance bin in progress: one on the lattice levels, and all `shells_per_level` bins
  of the exact enumeration at the short distances of molecule tables.

{func}`scote.twopoint` warns before allocating when these parts exceed 4 GiB, and names them. For large panels, take a
subset with `select`, leave `keep_tiles` off, stream with `out`, count all distances of molecules on the lattice
(`lattice_only=True`, one distance bin in progress), or halve the accumulators with `acc_dtype="float32"`.
