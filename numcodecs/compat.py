# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, print_function, division
import sys
import operator
import array


import numpy as np


PY2 = sys.version_info[0] == 2


if PY2:  # pragma: no cover

    text_type = unicode
    binary_type = str
    integer_types = (int, long)
    reduce = reduce

else:

    text_type = str
    binary_type = bytes
    integer_types = int,
    from functools import reduce


def buffer_size(v):
    """Return the size of the memory buffer used by `v`."""
    if PY2 and isinstance(v, array.array):  # pragma: no cover
        # special case array.array because does not support buffer
        # interface in PY2
        return v.buffer_info()[1] * v.itemsize
    else:
        v = memoryview(v)
        return reduce(operator.mul, v.shape) * v.itemsize


def buffer_tobytes(v):
    """Obtain a sequence of bytes for the memory buffer used by `v`."""
    if isinstance(v, np.ndarray):
        return v.tobytes(order='A')
    elif PY2 and isinstance(v, array.array):  # pragma: no cover
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
