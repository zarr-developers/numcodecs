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


def memory_copy(buf, out=None):
    """Copy the contents of the memory buffer from `buf` to `out`."""

    if out is None:
        # no-op
        return buf

    # obtain ndarrays, casting to the same data type
    buf = ensure_contiguous_ndarray(buf).view('u1')
    out = ensure_contiguous_ndarray(out).view('u1')

    # copy memory
    np.copyto(out, buf)

    return out


def ensure_text(l, encoding='utf-8'):
    if isinstance(l, text_type):
        return l
    else:  # pragma: py3 no cover
        return text_type(l, encoding=encoding)


def ensure_contiguous_ndarray(o):
    """Convenience function to obtain a numpy ndarray using memory exposed by object
    `o`. This function performs some additional checks and transformations to ensure
    that the returned ndarray can export a new-style buffer interface, and that the
    exposed memory is C contiguous.

    Parameters
    ----------
    o : array-like or bytes-like
        A numpy ndarray or any object exposing a memory buffer. On Python 3 this must
        be an object exposing the new-style buffer interface. On Python 2 this can also
        be an object exposing the old-style buffer interface.

    Returns
    -------
    o : ndarray

    Notes
    -----
    All efforts are made to ensure that no memory is copied, and that the returned
    ndarray provides a view on whatever memory was exposed by the original object `o`.
    However, in some circumstances a memory copy will be made, e.g., if the memory
    exposed by `o` is not contiguous.

    Arrays with an object dtype are not allowed as input to this function, as the memory
    exposed by these objects is just the object pointers, and the actual object values
    have memory scattered throughout the heap. In other words, there is no way to
    obtain a completely contiguous view of the memory of all items in the array,
    without some additional transformations that are beyond the scope of this function.

    """

    if not isinstance(o, np.ndarray):

        # first try to obtain a memoryview or buffer, needed to ensure that we don't
        # subsequently copy memory when going via np.array()

        if PY2:  # pragma: py3 no cover
            # accept objects exposing either old-style or new-style buffer interface
            try:
                o = memoryview(o)
            except TypeError:
                o = np.getbuffer(o)

        else:  # pragma: py2 no cover
            o = memoryview(o)

        # N.B., this is not well documented, but np.array() will accept an object exposing
        # a buffer interface, and will take a view of the memory rather than making a
        # copy, preserving type information where present
        o = np.array(o, copy=False)

    # check for object arrays, these are just memory pointers, actual memory holding
    # item data is scattered elsewhere
    if o.dtype == object:
        raise ValueError('object arrays are not supported')

    # check for datetime or timedelta ndarray, cannot take a memoryview of those directly
    if o.dtype.kind in 'Mm':
        o = o.view(np.int64)

    # flatten the array to 1 dimension - this will also ensure the array is contiguous
    # N.B., this will in some cases cause a memory copy to be made, see
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
    o = o.reshape(-1, order='A')

    return o


def ensure_bytes(o):
    """Obtain a bytes object from memory exposed by `o`."""

    if not isinstance(o, binary_type):

        # view as numpy array with contiguous memory
        m = ensure_contiguous_ndarray(o)

        # create bytes from memory
        o = m.tobytes()

    return o
