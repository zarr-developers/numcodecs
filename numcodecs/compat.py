# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, print_function, division
import io
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

    if max_buffer_size is not None and arr.nbytes > max_buffer_size:
        msg = "Codec does not support buffers of > {} bytes".format(max_buffer_size)
        raise ValueError(msg)

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


class MemoryViewIO(io.RawIOBase):
    def __init__(self, buf):
        self._pos = 0
        self._buf = ensure_contiguous_ndarray(buf).view('u1')

    def peek(self, size=1):
        if size >= 0:
            sl = slice(self._pos, self._pos + size)
        else:
            sl = slice(self._pos, None)

        data = self._buf[sl].tobytes()

        return data

    def read(self, size=-1):
        data = self.peek(size=size)

        self._pos += len(data)

        return data

    def read1(self, n=-1):
        return self.read(size=n)

    def readable(self):
        return True

    def readall(self):
        return self.read(size=-1)

    def readinto(self, buf):
        buf_view = ensure_contiguous_ndarray(buf)
        size = len(buf_view)

        sl = slice(self._pos, self._pos + size)
        _buf_view = self._buf[sl]
        size = min(size, len(_buf_view))

        np.copyto(buf_view[:size], _buf_view[:size])
        self._pos += size

        return buf

    def readinto1(self, buf):
        return self.readinto(buf)

    def seek(self, offset, whence=io.SEEK_SET):
        if offset == io.SEEK_SET:
            self._pos = offset
        elif offset == io.SEEK_CUR:
            self._pos += offset
        elif offset == io.SEEK_END:
            self._pos = len(self._buf) + offset
        else:
            raise ValueError("Invalid whence = %s" % str(whence))

        self._pos = np.clip(self._pos, 0, len(self._buf))

        return self._pos

    def seekable(self):
        return True

    def tell(self):
        return self._pos
