# flake8: noqa
import sys
import codecs
import array
from functools import reduce

import numpy as np


def ensure_ndarray(buf):
    """Convenience function to coerce `buf` to a numpy array, if it is not already a
    numpy array.

    Parameters
    ----------
    buf : array-like or bytes-like
        A numpy array or any object exporting a buffer interface.

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

    elif isinstance(buf, array.array) and buf.typecode in 'cu':
        # Guard condition, do not support array.array with unicode type, this is
        # problematic because numpy does not support it on all platforms. Also do not
        # support char as it was removed in Python 3.
        raise TypeError('array.array with char or unicode type is not supported')

    else:

        # N.B., first take a memoryview to make sure that we subsequently create a
        # numpy array from a memory buffer with no copy
        mem = memoryview(buf)

        # instantiate array from memoryview, ensures no copy
        arr = np.array(mem, copy=False)

    return arr


def ensure_contiguous_ndarray(buf, max_buffer_size=None):
    """Convenience function to coerce `buf` to a numpy array, if it is not already a
    numpy array. Also ensures that the returned value exports fully contiguous memory,
    and supports the new-style buffer interface. If the optional max_buffer_size is
    provided, raise a ValueError if the number of bytes consumed by the returned
    array exceeds this value.

    Parameters
    ----------
    buf : array-like or bytes-like
        A numpy array or any object exporting a buffer interface.
    max_buffer_size : int
        If specified, the largest allowable value of arr.nbytes, where arr
        is the retured array.

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
    arr = ensure_ndarray(buf)

    # check for object arrays, these are just memory pointers, actual memory holding
    # item data is scattered elsewhere
    if arr.dtype == object:
        raise TypeError('object arrays are not supported')

    # check for datetime or timedelta ndarray, the buffer interface doesn't support those
    if arr.dtype.kind in 'Mm':
        arr = arr.view(np.int64)

    # check memory is contiguous, if so flatten
    if arr.flags.c_contiguous or arr.flags.f_contiguous:
        # can flatten without copy
        arr = arr.reshape(-1, order='A')

    else:
        raise ValueError('an array with contiguous memory is required')

    if max_buffer_size is not None and arr.nbytes > max_buffer_size:
        msg = "Codec does not support buffers of > {} bytes".format(max_buffer_size)
        raise ValueError(msg)

    return arr


def ensure_bytes(buf):
    """Obtain a bytes object from memory exposed by `buf`."""

    if not isinstance(buf, bytes):

        # go via numpy, for convenience
        arr = ensure_ndarray(buf)

        # check for object arrays, these are just memory pointers,
        # actual memory holding item data is scattered elsewhere
        if arr.dtype == object:
            raise TypeError('object arrays are not supported')

        # create bytes
        buf = arr.tobytes(order='A')

    return buf


def ensure_text(s, encoding='utf-8'):
    if not isinstance(s, str):
        s = ensure_contiguous_ndarray(s)
        s = codecs.decode(s, encoding)
    return s


def ndarray_copy(src, dst):
    """Copy the contents of the array from `src` to `dst`."""

    if dst is None:
        # no-op
        return src

    # ensure ndarrays
    src = ensure_ndarray(src)
    dst = ensure_ndarray(dst)

    # flatten source array
    src = src.reshape(-1, order='A')

    # ensure same data type
    if dst.dtype != object:
        src = src.view(dst.dtype)

    # reshape source to match destination
    if src.shape != dst.shape:
        if dst.flags.f_contiguous:
            order = 'F'
        else:
            order = 'C'
        src = src.reshape(dst.shape, order=order)

    # copy via numpy
    np.copyto(dst, src)

    return dst
