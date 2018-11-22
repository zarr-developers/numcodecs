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
    buf = ensure_ndarray_from_memory(buf).view('u1')
    out = ensure_ndarray_from_memory(out).view('u1')

    # copy memory
    np.copyto(out, buf)

    return out


def ensure_text(l, encoding='utf-8'):
    if isinstance(l, text_type):
        return l
    else:  # pragma: py3 no cover
        return text_type(l, encoding=encoding)


def ensure_ndarray_from_memory(o, flatten=True):
    """Convenience function to obtain a numpy ndarray using memory exposed by object `o`,
    ensuring that no memory copies are made, and that `o` is not an object array.

    Parameters
    ----------
    o : bytes-like
        Any object exposing a memory buffer. On Python 3 this must be an object exposing
        the new-style buffer interface. On Python 2 this can also be an object exposing
        the old-style buffer interface.
    flatten : bool, optional
        If True, flatten any multi-dimensional inputs into a one-dimensional memoryview.

    Returns
    -------
    o : ndarray

    """

    if not isinstance(o, np.ndarray):

        # first try to obtain a memoryview or buffer, needed to ensure that we don't
        # subsequently copy memory when going via np.array()

        if PY2:  # pragma: py3 no cover
            # accept objects exposing either old-style or new-style buffer interface
            try:
                o = memoryview(o)
            except TypeError:
                o = buffer(o)

        else:  # pragma: py2 no cover
            o = memoryview(o)

        # N.B., this is not documented, but np.array() will accept an object exposing
        # a buffer interface, and will take a view of the memory rather than making a
        # copy, preserving type information where present
        o = np.array(o, copy=False)

    # check for object arrays
    if o.dtype == object:
        raise ValueError('object arrays are not supported')

    if flatten:

        # flatten the array to 1 dimension
        o = o.reshape(-1, order='A')

    return o


def ensure_memoryview(o, flatten=True):
    """Obtain a :class:`memoryview` with a view of memory exposed by `o`.

    Parameters
    ----------
    o : bytes-like
        Any object exposing a memory buffer. On Python 3 this must be an object exposing
        the new-style buffer interface. On Python 2 this can also be an object exposing
        the old-style buffer interface.
    flatten : bool, optional
        If True, flatten any multi-dimensional inputs into a one-dimensional memoryview.

    Returns
    -------
    o : memoryview

    """

    # go via numpy, for convenience
    o = ensure_ndarray_from_memory(o, flatten=flatten)

    # check for datetime or timedelta ndarray, cannot take a memoryview of those
    if o.dtype.kind in 'Mm':
        o = o.view(np.int64)

    # expose as memoryview
    o = memoryview(o)

    return o


def ensure_bytes(o):
    """Obtain a bytes object from memory exposed by `o`."""

    if not isinstance(o, binary_type):

        # obtain memoryview
        m = ensure_memoryview(o)

        # create bytes from memory
        o = m.tobytes()

    return o


if PY2:  # pragma: py3 no cover
    # Under PY2 some codecs are happier if they are provided with a buffer object
    # rather than a memoryview, so here we provide a convenience function to obtain a
    # buffer object from a range of possible inputs.

    def ensure_buffer(o):
        """Obtain a :class:`buffer` with a view of memory exposed by `o`.

        Parameters
        ----------
        o : bytes-like
            Any object exposing a memory buffer. Can be an object exposing either
            old-style or new-style buffer interface.

        Returns
        -------
        o : buffer

        """

        # go via numpy, for convenience
        o = ensure_ndarray_from_memory(o)

        # expose as buffer
        o = buffer(o)

        return o
