# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, print_function, division
import sys
import operator
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


def buffer_tobytes(v):
    """Obtain a sequence of bytes for the memory buffer used by `v`."""
    if isinstance(v, binary_type):
        return v
    elif isinstance(v, np.ndarray):
        return v.tobytes(order='A')
    elif PY2 and isinstance(v, array.array):  # pragma: py3 no cover
        return v.tostring()
    else:
        return memoryview(v).tobytes()


def buffer_copy(buf, out=None):
    """Copy the contents of the memory buffer from `buf` to `out`."""

    if out is None:
        # no-op
        return buf

    # handle ndarray destination
    if isinstance(out, np.ndarray):

        # view source as destination dtype
        if isinstance(buf, np.ndarray):
            buf = buf.view(dtype=out.dtype).reshape(-1, order='A')
        else:
            buf = np.frombuffer(buf, dtype=out.dtype)

        # ensure shapes are compatible
        if buf.shape != out.shape:
            if out.flags.f_contiguous:
                order = 'F'
            else:
                order = 'C'
            buf = buf.reshape(out.shape, order=order)

        # copy via numpy
        np.copyto(out, buf)

    # handle generic buffer destination
    else:

        # obtain memoryview of destination
        dest = memoryview(out)

        # ensure source is 1D
        if isinstance(buf, np.ndarray):
            buf = buf.reshape(-1, order='A')
            # try to match itemsize
            dtype = 'u%s' % dest.itemsize
            buf = buf.view(dtype=dtype)

        # try to copy via memoryview
        dest[:] = buf

    return out


def ndarray_from_buffer(buf, dtype):
    if isinstance(buf, np.ndarray):
        arr = buf.reshape(-1, order='A').view(dtype)
    else:
        arr = np.frombuffer(buf, dtype=dtype)
    return arr


def ensure_text(l, encoding='utf-8'):
    if isinstance(l, text_type):
        return l
    else:  # pragma: py3 no cover
        return text_type(l, encoding=encoding)


def handle_datetime(buf):
    if hasattr(buf, 'dtype') and buf.dtype.kind in 'Mm':
        return buf.view('u8')
    return buf
