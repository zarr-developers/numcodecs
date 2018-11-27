# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, print_function, division
import sys
import array


import numpy as np


PY2 = sys.version_info[0] == 2


if PY2:  # pragma: py3 no cover

    text_type = unicode
    binary_type = str
    integer_types = (int, long)
    reduce = reduce

else:  # pragma: py2 no cover

    text_type = str
    binary_type = bytes
    integer_types = int,
    from functools import reduce


def ensure_text(l, encoding='utf-8'):
    if isinstance(l, text_type):
        return l
    else:  # pragma: py3 no cover
        return text_type(l, encoding=encoding)


def ensure_ndarray(buf, dtype=None):
    """Convenience function to coerce `buf` to a numpy array, if it is not already a
    numpy array.

    Parameters
    ----------
    buf : array-like or bytes-like
        A numpy array or any object exporting a buffer interface.
    dtype : dtype, optional
        Request that the data be viewed as the given dtype.

    Returns
    -------
    arr : ndarray
        A numpy array, sharing memory with `buf`.

    Notes
    -----
    This function will not create a copy under any circumstances, it is guaranteed to
    return a view on memory exported by `buf`.

    """

    if isinstance(buf, np.ndarray):
        # already a numpy array
        arr = buf

    elif isinstance(buf, array.array) and buf.typecode == 'u':
        # guard condition, do not support array.array with unicode type, this is
        # problematic because numpy does not support it on all platforms
        raise TypeError('array.array with unicode type is not supported')

    else:

        # N.B., first take a memoryview to make sure that we subsequently create a
        # numpy array from a memory buffer with no copy

        if PY2:  # pragma: py3 no cover
            try:
                mem = memoryview(buf)
            except TypeError:
                # on PY2 also check if object exports old-style buffer interface
                mem = np.getbuffer(buf)

        else:  # pragma: py2 no cover
            mem = memoryview(buf)

        # instantiate array from memoryview, ensures no copy
        arr = np.array(mem, copy=False)

        if PY2 and isinstance(buf, array.array):  # pragma: py3 no cover
            # type information will not have been propagated via the old-style buffer
            # interface, so we have to manually hack it back in after the fact
            arr = arr.view(buf.typecode)

    if dtype is not None:
        # view as requested dtype
        arr = arr.view(dtype)

    return arr


def ensure_contiguous_ndarray(buf, dtype=None):
    """Convenience function to coerce `buf` to a numpy array, if it is not already a
    numpy array. Also ensures that the returned value exports fully contiguous memory,
    and supports the new-style buffer interface.

    Parameters
    ----------
    buf : array-like or bytes-like
        A numpy array or any object exporting a buffer interface.
    dtype : dtype, optional
        Request that the data be viewed as the given dtype.

    Returns
    -------
    arr : ndarray
        A numpy array, sharing memory with `buf`.

    Notes
    -----
    This function will not create a copy under any circumstances, it is guaranteed to
    return a view on memory exported by `buf`.

    """

    # ensure input is a numpy array
    arr = ensure_ndarray(buf, dtype=dtype)

    # check for datetime or timedelta ndarray, the buffer interface doesn't support those
    if isinstance(buf, np.ndarray) and buf.dtype.kind in 'Mm':
        arr = arr.view(np.int64)

    # check for object arrays, these are just memory pointers, actual memory holding
    # item data is scattered elsewhere
    if arr.dtype == object:
        raise TypeError('object arrays are not supported')

    # check memory is contiguous, if so flatten
    if arr.flags.c_contiguous or arr.flags.f_contiguous:
        # can flatten without copy
        arr = arr.reshape(-1, order='A')

    else:
        raise ValueError('an array with contiguous memory is required')

    return arr


def ensure_bytes(buf):
    """Obtain a bytes object from memory exposed by `buf`."""

    if not isinstance(buf, binary_type):

        # go via numpy, for convenience
        arr = ensure_ndarray(buf)

        # create bytes
        buf = arr.tobytes(order='A')

    return buf


def ndarray_copy(src, dst):
    """Copy the contents of the array from `src` to `dst`."""

    if dst is None:
        # no-op
        return src

    # ensure ndarrays
    src = ensure_ndarray(src)
    dst = ensure_ndarray(dst)

    # ensure same data type
    if dst.dtype != object:
        src = src.view(dst.dtype)

    # reshape source to match destination
    src = src.reshape(-1, order='A')
    if src.shape != dst.shape:
        if dst.flags.f_contiguous:
            order = 'F'
        else:
            order = 'C'
        src = src.reshape(dst.shape, order=order)

    # copy via numpy
    np.copyto(dst, src)

    return dst
