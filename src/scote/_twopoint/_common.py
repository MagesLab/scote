"""Options and timing records shared by the CPU and GPU backends."""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["BACKENDS", "CPUOptions", "GPUOptions", "LevelTiming", "resolve_backend"]

BACKENDS = ("cpu", "gpu")
CPU_MODES = ("auto", "numba", "dense", "sparse")
_BACKEND_TYPES = "backend must be 'cpu', 'gpu', a scote.CPUOptions, or a scote.GPUOptions"


@dataclass(frozen=True)
class CPUOptions:
    r"""Options of the CPU backend: ``backend=scote.CPUOptions(...)`` of :func:`scote.twopoint` and
    :func:`scote.pair_counts` (``backend="cpu"`` uses the defaults).

    The shell products :math:`P_k = X^\top S_k X` of a lattice level (:math:`X`: cells x features counts,
    :math:`S_k`: the 0/1 adjacency of the cell pairs in shell :math:`k`) run with numba kernels, BLAS, or SciPy.

    Attributes
    ----------
    mode : {"auto", "numba", "dense", "sparse"}
        Product strategy: ``"numba"`` (fused multi-threaded kernel), ``"dense"`` (:math:`X` densified, BLAS),
        ``"sparse"`` (SciPy, the reference), ``"auto"`` (``"dense"`` where the density of :math:`X` exceeds
        ``dense_threshold``, else ``"numba"``). All give identical counts.
    dense_threshold : float
        Density :math:`\mathrm{nnz}(X) / (n_\mathrm{cells} G)` above which mode ``"auto"`` uses ``"dense"``.
    row_block : int
        Rows of :math:`S_k` per block in mode ``"sparse"`` (bounds the memory of the intermediate :math:`S_k X`).
    host : {"numba", "numpy"}
        Construction of the shell operators and the sparse x dense product: parallel numba kernels, or the serial
        NumPy/SciPy reference (identical results).
    numba_acc_bytes : int
        Memory limit (bytes) of the per-thread (G, G) accumulators of mode ``"numba"``,
        :math:`8\,n_\mathrm{threads} G^2`; above it a level falls back to ``"sparse"`` with a warning. Raise it for many
        features on a machine with the memory (the counts are identical in every mode).
    """
    mode: str = "auto"
    dense_threshold: float = 0.05
    row_block: int = 200_000
    host: str = "numba"
    numba_acc_bytes: int = 8 * 2**30

    def __post_init__(self):
        """Validate the fields."""
        if self.mode not in CPU_MODES:
            raise ValueError(f"CPU mode must be one of {CPU_MODES}, got {self.mode!r}")
        if self.host not in ("numba", "numpy"):
            raise ValueError(f"host must be 'numba' or 'numpy', got {self.host!r}")
        if not self.row_block >= 1:
            raise ValueError(f"row_block must be positive, got {self.row_block}")
        if not self.numba_acc_bytes >= 0:
            raise ValueError(f"numba_acc_bytes must be >= 0, got {self.numba_acc_bytes}")


@dataclass(frozen=True)
class GPUOptions:
    r"""Options of the GPU backend: ``backend=scote.GPUOptions(...)`` of :func:`scote.twopoint` and
    :func:`scote.pair_counts` (``backend="gpu"`` uses the defaults).

    The backend runs the products with PyTorch, on a CUDA device or, with ``device="cpu"``, on the PyTorch CPU
    device (the same code on the host). The counts are bit-identical to those of the CPU backend.

    Attributes
    ----------
    device : str
        Torch device (``"cuda"``, ``"cuda:1"``, or ``"cpu"``, which runs the same code on the host).
    dense_bytes_limit : int
        Upper limit of :math:`2 n_\mathrm{cells} G \cdot 8` bytes for the dense path of a level (also at most 30 % of
        the device memory budget).
    row_block_bytes : int
        Memory bound of one dense row block of the semidense path.
    device_slabs : bool
        Pass the slabs to a sink as float64 device tensors that are valid only during the call, instead of host
        arrays (saves the device-to-host copy when the sink works on the device).
    """
    device: str = "cuda"
    dense_bytes_limit: int = 4 * 2**30
    row_block_bytes: int = 2 * 2**30
    device_slabs: bool = False


def resolve_backend(backend) -> tuple[str, CPUOptions | GPUOptions]:
    """The name and the options of a ``backend`` argument.

    Parameters
    ----------
    backend : {"cpu", "gpu"}, CPUOptions, or GPUOptions
        A backend name (with the default options) or the options of a backend.

    Returns
    -------
    tuple of str and CPUOptions or GPUOptions
        ``"cpu"`` or ``"gpu"``, and the options.

    Raises
    ------
    ValueError
        For an unknown backend name.
    TypeError
        For any other type, including a dict or an options class instead of an instance.
    """
    if isinstance(backend, CPUOptions):
        return "cpu", backend
    if isinstance(backend, GPUOptions):
        return "gpu", backend
    if isinstance(backend, str):
        if backend not in BACKENDS:
            raise ValueError(f"{_BACKEND_TYPES}, got {backend!r}")
        return backend, CPUOptions() if backend == "cpu" else GPUOptions()
    if backend is CPUOptions or backend is GPUOptions:
        raise TypeError(f"{_BACKEND_TYPES}, got the class {backend.__name__}; pass an instance, "
                        f"e.g. scote.{backend.__name__}()")
    hint = "; pass the fields as scote.CPUOptions(**fields) or scote.GPUOptions(**fields)" \
        if isinstance(backend, dict) else ""
    raise TypeError(f"{_BACKEND_TYPES}, got {type(backend).__name__}{hint}")


@dataclass
class LevelTiming:
    """Time and work of the shell products of one lattice level.

    Both backends book the same stages: building the shell operators S_k (``seconds_S``), the products S_k X
    (``seconds_SX``), and the products X^T (S_k X) (``seconds_XtSX``). Where one kernel forms both products at once
    (CPU mode "numba"), its time is booked under ``seconds_XtSX`` and ``seconds_SX`` stays 0.

    Attributes
    ----------
    level : int
        Level index of the hierarchy.
    pitch : float
        Lattice pitch of the level (µm).
    n_cells : int
        Occupied cells.
    nnz_X : int
        Stored entries of the count matrix X.
    density_X : float
        nnz(X) / (n_cells G).
    mode : str
        Strategy that ran (CPU mode or a description of the GPU path).
    seconds_S : float
        Time to build the shell operators.
    seconds_SX : float
        Time of the products S_k X.
    seconds_XtSX : float
        Time of the products X^T (S_k X).
    pair_updates : float
        Multiply-adds of the products.
    cell_pairs : float
        nnz(S_k) summed over the shells.
    seconds_total : float
        Wall time of the level, including uploads to the device.
    per_shell : list of dict
        Per-shell records (shell index, nnz(S_k), times; on the GPU also the upload time and peak device memory).
    """
    level: int
    pitch: float
    n_cells: int
    nnz_X: int
    density_X: float
    mode: str
    seconds_S: float = 0.0
    seconds_SX: float = 0.0
    seconds_XtSX: float = 0.0
    pair_updates: float = 0.0
    cell_pairs: float = 0.0
    seconds_total: float = 0.0
    per_shell: list = field(default_factory=list)

    def add_shell(self, k: int, nnz_S: int, s_S: float, s_SX: float, s_XtSX: float, updates: float) -> None:
        """Book the work of one shell.

        Parameters
        ----------
        k : int
            Shell index.
        nnz_S : int
            Stored entries of the shell operator (rows used).
        s_S : float
            Seconds to build the shell operator.
        s_SX : float
            Seconds of the product S_k X.
        s_XtSX : float
            Seconds of the product X^T (S_k X).
        updates : float
            Multiply-adds of the products.
        """
        self.seconds_S += s_S
        self.seconds_SX += s_SX
        self.seconds_XtSX += s_XtSX
        self.pair_updates += updates
        self.cell_pairs += nnz_S
        self.per_shell.append(dict(k=k, nnz_S=int(nnz_S), s_S=s_S, s_SX=s_SX, s_XtSX=s_XtSX))

    def as_dict(self) -> dict:
        """The record as a plain dict.

        Returns
        -------
        dict
            All attributes except ``per_shell``.
        """
        return dict(level=self.level, pitch=self.pitch, n_cells=self.n_cells, nnz_X=self.nnz_X,
                    density_X=self.density_X, mode=self.mode, seconds_S=self.seconds_S, seconds_SX=self.seconds_SX,
                    seconds_XtSX=self.seconds_XtSX, pair_updates=self.pair_updates, cell_pairs=self.cell_pairs,
                    seconds_total=self.seconds_total)
