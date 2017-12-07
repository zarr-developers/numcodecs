# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import zlib as _zlib


import numpy as np


from .abc import Codec
from .compat import buffer_copy, handle_datetime, buffer_tobytes, PY2


class Zlib(Codec):
    """Codec providing compression using zlib via the Python standard library.

    Parameters
    ----------
    level : int
        Compression level.

    """

    codec_id = 'zlib'

    def __init__(self, level=1):
        self.level = level

    def encode(self, buf):

        # deal with lack of buffer support for datetime64 and timedelta64
        buf = handle_datetime(buf)

        if isinstance(buf, np.ndarray):

            # cannot compress object array
            if buf.dtype == object:
                raise ValueError('cannot encode object array')

            # if numpy array, can only handle C contiguous directly
            if not buf.flags.c_contiguous:
                buf = buf.tobytes(order='A')

        if PY2:  # pragma: py3 no cover
            # ensure bytes, PY2 cannot handle things like bytearray
            buf = buffer_tobytes(buf)

        # do compression
        return _zlib.compress(buf, self.level)

    # noinspection PyMethodMayBeStatic
    def decode(self, buf, out=None):

        if PY2:  # pragma: py3 no cover
            # ensure bytes, PY2 cannot handle things like bytearray
            buf = buffer_tobytes(buf)

        # do decompression
        dec = _zlib.decompress(buf)

        # handle destination - Python standard library zlib module does not
        # support direct decompression into buffer, so we have to copy into
        # out if given
        return buffer_copy(dec, out)
