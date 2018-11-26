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


def ensure_ndarray(o, allow_copy=False):
    """TODO"""

    if not isinstance(o, np.ndarray):

        # If the object is not a numpy array, we will first check to see if it exports
        # a buffer interface, so we can then create a numpy array with a view onto the
        # same memory, rather than copying memory. This is a work-around for current
        # behaviour in np.array(o, copy=False), which will sometimes make a copy
        # and choose *not* to use the buffer interface, even though it is available,
        # e.g., if o is a bytes object.

        try:
            o = memoryview(o)
        except TypeError:
            if PY2:  # pragma: py3 no cover
                try:
                    # on PY2 also check if object exports old-style buffer interface
                    o = np.getbuffer(o)
                except TypeError:
                    if not allow_copy:
                        raise
            else:  # pragma: py2 no cover
                if not allow_copy:
                    raise

        o = np.array(o, copy=False)

    return o


def ensure_ndarray_exporting_memory(o, allow_copy=False):
    """TODO"""

    # ensure we have a numpy array
    o = ensure_ndarray(o, allow_copy=allow_copy)

    # check for object arrays, these are just memory pointers, actual memory holding
    # item data is scattered elsewhere
    if o.dtype == object:
        raise ValueError('object arrays are not supported')

    # check for datetime or timedelta ndarray, cannot take a memoryview of those
    if o.dtype.kind in 'Mm':
        o = o.view(np.int64)

    return o


def ensure_contiguous_ndarray(o, allow_copy=False):

    # ensure we have a numpy array that can export memory
    o = ensure_ndarray_exporting_memory(o, allow_copy=allow_copy)

    # check for contiguous memory
    if not (o.flags.c_contiguous or o.flags.f_contiguous):
        raise ValueError('array with contiguous memory is required')

    return o


def ensure_c_contiguous_ndarray(o, allow_copy=False):
    """TODO"""

    # ensure we have a numpy array that can export memory
    o = ensure_ndarray_exporting_memory(o, allow_copy=allow_copy)

    # check if C-contiguous
    if not o.flags.c_contiguous:

        if o.flags.f_contiguous:
            # convert F to C contiguous without copying memory
            o = o.T

        else:

            if allow_copy:
                # ensure we have a C contiguous array - may copy memory
                o = np.ascontiguousarray(o)

            else:
                raise ValueError('array is not contiguous')

    return o


def ensure_bytes(o):
    """Obtain a bytes object from memory exposed by `o`."""

    if not isinstance(o, binary_type):

        # go via numpy, for convenience
        a = ensure_ndarray_exporting_memory(o)

        # create bytes
        o = a.tobytes(order='A')

    return o


def memory_copy(buf, out=None):
    """Copy the contents of the memory buffer from `buf` to `out`."""

    if out is None:
        # no-op
        return buf

    # ensure ndarrays viewing memory
    buf = ensure_contiguous_ndarray(buf, allow_copy=False)
    out = ensure_contiguous_ndarray(out, allow_copy=False)

    # flatten both arrays - we have ensured they are contiguous, so this should not
    # introduce any copies
    buf = buf.reshape(-1, order='A')
    out = out.reshape(-1, order='A')

    # ensure same data type, required for copy
    buf = buf.view(out.dtype)

    # copy via numpy
    np.copyto(out, buf)

    return out
