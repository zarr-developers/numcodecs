# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, print_function, division
import sys
import operator
import array


import numpy as np


PY2 = sys.version_info[0] == 2


if PY2:  # pragma: py3 no cover

    buffer = buffer
    text_type = unicode
    binary_type = str
    integer_types = (int, long)
    reduce = reduce

else:  # pragma: py2 no cover

    buffer = memoryview
    text_type = str
    binary_type = bytes
    integer_types = int,
    from functools import reduce


def getbuffer(v):
    if PY2:
        return np.getbuffer(v)
    else:
        return memoryview(v).cast('B')


def to_buffer(v):
    """Obtain a `buffer` or `memoryview` for `v`."""

    b = ndarray_from_buffer(v)

    if b.dtype.kind is 'O':
        raise ValueError('cannot encode object array')
    elif b.dtype.kind in 'Mm':
        b = b.view(np.uint64)

    b = b.reshape(-1, order='A')

    return b


def buffer_copy(buf, out=None):
    """Copy the contents of the memory buffer from `buf` to `out`."""

    if out is None:
        # no-op
        return buf

    # coerce to ndarrays
    buf = ndarray_from_buffer(buf)
    out = ndarray_from_buffer(out)

    # view source as destination dtype
    if out.dtype.kind is not 'O':
        buf = buf.view(out.dtype)

    # ensure shapes are compatible
    buf = buf.reshape(-1, order='A')
    if buf.shape != out.shape:
        if out.flags.f_contiguous:
            order = 'F'
        else:
            order = 'C'
        buf = buf.reshape(out.shape, order=order)

    # copy via numpy
    np.copyto(out, buf)

    return out


def ndarray_from_buffer(buf):
    arr = buf
    if not isinstance(arr, np.ndarray):
        try:
            arr = memoryview(arr)
        except TypeError:  # pragma: py3 no cover
            arr = np.getbuffer(arr)
        else:  # pragma: py2 no cover
            if isinstance(buf, array.array) and buf.typecode is 'u':
                arr = arr.cast('B').cast(np.dtype('u%i' % buf.itemsize).char)

        arr = np.array(arr, copy=False)
        if PY2 and isinstance(buf, array.array):  # pragma: py3 no cover
            if buf.typecode is 'u':
                arr = arr.view('u%i' % buf.itemsize)
            else:
                arr = arr.view(buf.typecode)

    return arr


def ensure_text(l, encoding='utf-8'):
    if isinstance(l, text_type):
        return l
    else:  # pragma: py3 no cover
        return text_type(l, encoding=encoding)
