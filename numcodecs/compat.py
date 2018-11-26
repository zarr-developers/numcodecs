# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, print_function, division
import sys


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


def ensure_contiguous_ndarray(o):
    """TODO"""

    if not isinstance(o, np.ndarray):

        # make that we create an array from a memory buffer with no copy

        if PY2:  # pragma: py3 no cover
            try:
                o = memoryview(o)
            except TypeError:
                # on PY2 also check if object exports old-style buffer interface
                o = np.getbuffer(o)

        else:  # pragma: py2 no cover
            o = memoryview(o)

        o = np.array(o, copy=False)

    # check for object arrays, these are just memory pointers, actual memory holding
    # item data is scattered elsewhere
    if o.dtype == object:
        raise ValueError('object arrays are not supported')

    # check for datetime or timedelta ndarray, cannot take a memoryview of those
    if o.dtype.kind in 'Mm':
        o = o.view(np.int64)

    # check memory is contiguous, if so flatten
    if o.flags.c_contiguous or o.flags.f_contiguous:

        # can flatten without copy
        o = o.reshape(-1, order='A')

    else:
        raise ValueError('an array with contiguous memory is required')

    return o


def ensure_bytes(o):
    """Obtain a bytes object from memory exposed by `o`."""

    if not isinstance(o, binary_type):

        # go via numpy, for convenience
        a = ensure_contiguous_ndarray(o)

        # create bytes
        o = a.tobytes()

    return o


def memory_copy(buf, out):
    """Copy the contents of the memory buffer from `buf` to `out`."""

    if out is None:
        # no-op
        return buf

    # ensure ndarrays viewing memory
    buf = ensure_contiguous_ndarray(buf)
    out = ensure_contiguous_ndarray(out)

    # ensure same data type, required for copy
    buf = buf.view(out.dtype)

    # copy via numpy
    np.copyto(out, buf)

    return out
