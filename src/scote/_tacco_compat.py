"""Helpers vendored from TACCO for the pair enumeration (``scote._twopoint.enumeration``).

Origin: TACCO, https://github.com/simonwm/tacco.
TACCO is distributed under the BSD 3-Clause license, Copyright (c) 2022-2025, Broad Institute, 2025-2026 The TACCO
Project Maintainers;
the full notice is reproduced in ``NOTICES.md`` at the root of this repository.

Contents and their TACCO sources: ``min_dtype``, ``bin``, ``hash``, ``_lexicographic_hash``, ``_numba_getNN``,
``_getNN``, ``_get_indptr``, ``_guess_blocksize``: ``tacco/utils/_dist.py``; ``enum_shifts``:
``tacco/tools/_annotate.py::_enum_shifts``.

Modifications: ``_getNN`` keeps only the numba code path with the coordinate dtype as given (TACCO selects among three
code paths that return the same neighbor table by a run-time self-test; the self-test and the other two paths are
removed, and the 64-bit range check of the lattice size, which cannot fail for two-dimensional int32 coordinates, is an
error instead of a fallback); docstrings were added to the helpers that had none, and long lines were wrapped.

``bin`` and ``hash`` keep their TACCO names and shadow the builtins inside this module.
"""
import numpy as np
import pandas as pd
from numba import njit

__all__ = ["bin", "enum_shifts", "hash", "min_dtype"]


def min_dtype(
    maximum,
):
    """\
    Gives the smallest signed integer dtype which is necessary to represent
    `maximum`.

    Parameters
    ----------
    maximum
        The integer to be representable in the dtype.

    Returns
    -------
    A signed integer dtype.

    """

    if maximum < 2**7 - 1:
        dtype = np.int8
    elif maximum < 2**15 - 1:
        dtype = np.int16
    elif maximum < 2**31 - 1:
        dtype = np.int32
    elif maximum < 2**63 - 1:
        dtype = np.int64
    else:
        raise ValueError('The maximum value to represent is larger than the range of 64bit integer! '
                         'This is not easily possible...')
    return dtype


def bin(
    data,
    bin_size,
    position_keys=['x', 'y'],
    bin_keys=None,
    shift=None,
):

    """\
    Bins points in position space.

    Parameters
    ----------
    data
        A :class:`~pandas.DataFrame`.
    bin_size
        The spatial size of a bin. Bins are of the same size in all
        dimensions.
    position_keys
        Array-like of column keys which contain the position of the points.
    bin_keys
        The names of the columns to contain the bin assignment. Has to be of
        the same length as `position_keys`. If `None`, a dataframe of bin
        assignments is returned.
    shift
        An array-like of the same length as `position_keys`, giving the
        shift of the start point of the bin grid in every dimension. If
        `None`, then no shift is used and the bin grid starts at the minimum
        value in each dimension.

    Returns
    -------
    Depending on `bin_key` returns either a :class:`~pandas.DataFrame` of bin\
    assignments or the updated input `data` contining the bin assignments\
    under `bin_keys`.

    """

    _coords = data[position_keys].to_numpy()
    mins, maxs = _coords.min(axis=0), _coords.max(axis=0)
    if shift is not None:
        mins += shift
    n_dim = _coords.shape[1]

    max_bin_value = (maxs - mins).max() // bin_size + 1
    dtype = min_dtype(max_bin_value)

    bin_coords = np.array([((_coords[:, d] - mins[d]) // bin_size).astype(dtype) for d in range(n_dim)]).T

    if bin_keys is None:
        return pd.DataFrame(bin_coords, index=data.index)
    else:
        if len(bin_keys) != n_dim:
            raise ValueError(f'`bin_keys` and `position_keys` have a diferent size: '
                             f'`(len(bin_keys),len(position_keys))=({len(bin_keys)},{n_dim})`')
        for d, bin_key in enumerate(bin_keys):
            data[bin_key] = bin_coords[:, d]
        return data


def hash(
    data,
    keys=None,
    hash_key=None,
    other=None,
    compress=True,
):

    """\
    Create collision-free hash of several categorical columns by
    lexicograhical indexing.

    Parameters
    ----------
    data
        A :class:`~pandas.DataFrame`.
    keys
        The names of the columns containing the categorical properties to
        hash. Not-categorical columns are transformed to categoricals first.
        If `None`, uses all columns.
    hash_key
        The name of the column to contain the hash values. If `None`, a series
        of hash assignments is returned.
    other
        Another :class:`~pandas.DataFrame`, which also has the `keys` with the
        same datatypes and should get the same hash trasformation as `data`.
    compress
        Whether to compress the lexicographical indices into contiguous 0-based
        hash values. This can take quite some time, but can also decrease the
        object size of the hash columns.

    Returns
    -------
    Depending on `hash_key` returns either a :class:`~pandas.Series` of hash\
    assignments or the updated input `data` contining the hash assignments\
    under `hash_key`. Depending on `other` returns this as a pair of the\
    results for `data and `other`.

    """

    if keys is None:
        keys = data.columns

    _data = data[keys]
    if other is not None:
        _other = other[keys]

    colrange = range(_data.shape[1])
    cats = []
    for i in colrange:
        unique = _data.iloc[:, i].unique()
        if hasattr(unique, 'to_numpy'):
            unique = unique.to_numpy()

        if other is not None:
            if _data.iloc[:, i].dtype != _other.iloc[:, i].dtype:
                raise ValueError(f'Columns {keys[i]!r} of `data` and `other` have different data types '
                                 f'{_data.iloc[:,i].dtype!r} and {_other.iloc[:,i].dtype!r}!')
            other_unique = _other.iloc[:, i].unique()
            if hasattr(other_unique, 'to_numpy'):
                other_unique = other_unique.to_numpy()
            unique = np.unique(np.concatenate([unique, other_unique]))

        cats.append(_data.iloc[:, i].astype(unique.dtype).astype(pd.CategoricalDtype(unique)))

    sizes = [len(cats[i].cat.categories) for i in colrange]
    dtype = min_dtype(np.prod(sizes))
    codes = [cats[i].cat.codes.to_numpy() for i in colrange]

    res = np.zeros(codes[0].shape, dtype=dtype)
    for size, code in zip(sizes, codes):
        res *= size
        res += code

    if other is None:
        if compress:
            res = pd.Series(res, index=data.index).astype('category').cat.codes

        if hash_key is None:
            return res
        else:
            data[hash_key] = res
            return data

    else:
        other_cats = [_other.iloc[:, i].astype(cats[i].cat.categories.dtype).astype(cats[i].dtype) for i in colrange]
        other_codes = [other_cats[i].cat.codes.to_numpy() for i in colrange]
        other_res = np.zeros(other_codes[0].shape, dtype=dtype)

        for size, code in zip(sizes, other_codes):
            other_res *= size
            other_res += code

        if compress:
            unique = np.unique(np.concatenate([np.unique(res), np.unique(other_res)]))
            res = pd.Series(res, index=data.index).astype(pd.CategoricalDtype(unique)).cat.codes
            other_res = pd.Series(other_res, index=other.index).astype(pd.CategoricalDtype(unique)).cat.codes

        if hash_key is None:
            return res, other_res
        else:
            data[hash_key] = res
            other[hash_key] = other_res
            return data, other


def enum_shifts(n_shifts, n_dim, shifts=None):
    """\
    Enumerates all integer shift vectors with entries 0 .. n_shifts - 1.

    Parameters
    ----------
    n_shifts
        Number of values per dimension.
    n_dim
        Number of dimensions still to enumerate.
    shifts
        The shift vectors of the dimensions enumerated so far (used by the
        recursion); `None` to start.

    Returns
    -------
    An array of shape (n_shifts**n_dim, n_dim) in lexicographic order.

    """
    if n_dim == 0:
        return np.array(shifts)
    if shifts is None:
        shifts = [[s] for s in range(n_shifts)]
    else:
        shifts = [[s, *shift] for s in range(n_shifts) for shift in shifts]
    return enum_shifts(n_shifts, n_dim - 1, shifts=shifts)


@njit(cache=True)
def _lexicographic_hash(codes, sizes):
    """\
    Lexicographic index of an integer vector in a box.

    Parameters
    ----------
    codes
        The integer vector.
    sizes
        The size of the box in every dimension.

    Returns
    -------
    The index, or -1 if the vector lies outside the box.

    """
    res = 0
    for size, code in zip(sizes, codes):
        if code < 0 or code >= size:
            return -1
        res *= size
        res += code
    return res


@njit(cache=True)
def _numba_getNN(coords, bins, shifts):
    """\
    Neighbor table of the occupied blocks.

    Parameters
    ----------
    coords
        Integer coordinates of the occupied blocks, one row per block.
    bins
        The block index of every row of `coords`.
    shifts
        The shift vectors to the neighbor blocks (including the zero shift).

    Returns
    -------
    An array of shape (n_blocks, 3**n_dim): row `b` holds the number of
    neighbors of block `b` with a larger block index in entry 0, then their
    block indices.

    """
    sizes = np.array([np.max(coords[:, i]) + 1 for i in range(coords.shape[1])])

    xy2bin_dict = {}
    for i in range(len(bins)):
        xy2bin_dict[_lexicographic_hash(coords[i], sizes)] = bins[i]

    # for every coord there can be 3**nDimensions - 1 neighbors contributions
    # for every coord the first entry is the number of neighbors, adding + 1 to a total of 3**nDimensions
    others = np.zeros((coords.shape[0], 3**coords.shape[1]), dtype=bins.dtype)
    for xi in range(len(bins)):
        bi, ci = bins[xi], coords[xi]
        bi_counter = 0
        for shift in shifts:
            bj = xy2bin_dict.get(_lexicographic_hash(ci + shift, sizes), None)
            if bj is not None:
                if bj > bi:
                    bi_counter += 1
                    others[bi, bi_counter] = bj
        others[bi, 0] = bi_counter

    return others


def _getNN(coords, bins):
    """\
    Neighbor table of the occupied blocks (all blocks within one block
    in every dimension).

    Parameters
    ----------
    coords
        Integer coordinates of the occupied blocks, one row per block.
    bins
        The block index of every row of `coords`.

    Returns
    -------
    The neighbor table of ``_numba_getNN``.

    Raises
    ------
    ValueError
        If the box of the block coordinates has more than 2**63 - 1 cells.

    """
    # shifts in the bin-vector to find all neighbors
    shifts = enum_shifts(n_shifts=3, n_dim=coords.shape[1])
    shifts = shifts - 1

    total_size = 1
    for i in range(coords.shape[1]):
        total_size = total_size * (int(np.max(coords[:, i])) + 1)  # Python int: no overflow in the coordinate dtype
    if total_size > 2**63 - 1:
        raise ValueError('The block box is too large to be enumerated in a 64bit signed integer.')

    return _numba_getNN(coords, bins, shifts)


@njit(cache=True)
def _get_indptr(bins, max_bin, dtype):
    """\
    Boundaries of the runs of equal values in a sorted array.

    Parameters
    ----------
    bins
        Sorted block indices 0 .. max_bin, every index present.
    max_bin
        The largest block index.
    dtype
        Integer dtype of the result.

    Returns
    -------
    An array of length max_bin + 2: the points of block `b` are
    `bins[indptr[b]:indptr[b+1]]`.

    """
    indptr = np.zeros((max_bin + 2), dtype=dtype)
    bin0 = 0
    b = 1
    for i in range(len(bins)):
        if bins[i] != bin0:
            bin0 = bins[i]
            indptr[b] = i
            b += 1
    indptr[-1] = len(bins)
    return indptr


def _guess_blocksize(positions, max_distance, verbose=1):
    """\
    Heuristic block size for the pair enumeration: large enough that the
    fullest block has a high but not too high number of points, which makes
    the dense subproblems efficient while keeping the memory low and the
    sparsity gain high.

    Parameters
    ----------
    positions
        A :class:`~pandas.DataFrame` with one position column per dimension.
    max_distance
        The largest distance of the enumeration; the block size is at least
        this.
    verbose
        Print the chosen value if positive.

    Returns
    -------
    The block size, rounded to two significant digits.

    """
    n_dim = positions.shape[1]
    maxN = 0
    factor_i = 0
    blocks = 1
    numba_blocksize = max_distance
    while maxN < 200 and blocks >= 1:
        factor_i += 1
        factor = 2**(factor_i / n_dim)
        if factor > 10:  # different ballpark...
            break
        # round to 2 significant digits for convenience
        numba_blocksize = float(f'{max_distance * 2**(factor_i/n_dim):.1e}')
        bins = bin(positions, bin_size=numba_blocksize, position_keys=positions.columns)
        hashes = hash(bins)
        Avc = hashes.value_counts()
        maxN = Avc.max()
        blocks = len(Avc)
    factor_i -= 1
    # round to 2 significant digits for convenience
    numba_blocksize = float(f'{max_distance * 2**(factor_i/n_dim):.1e}')
    if verbose > 0:
        print(f'The heuristic value for the parameter `numba_blocksize` is {numba_blocksize}. Consider specifying '
              'this directly as argument to avoid (possibly significant) overhead and/or experiment with this value '
              'on (a subset of) the actual dataset at hand to obtain an optimal value in terms of speed and memory '
              'requirements.')
    return numba_blocksize
