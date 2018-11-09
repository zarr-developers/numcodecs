# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import gzip as _gzip
import io


import numpy as np


from .abc import Codec
from .compat import buffer_copy, handle_datetime, PY2


class GZip(Codec):
    """Codec providing gzip compression using zlib via the Python standard library.

    Parameters
    ----------
    level : int
        Compression level.

    """

    codec_id = 'gzip'

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
            # ensure buffer, PY2 cannot handle things like bytearray
            buf = buffer(buf)

        # do compression
        compressed = io.BytesIO()
        with _gzip.GzipFile(fileobj=compressed,
                            mode='wb',
                            compresslevel=self.level) as compressor:
            compressor.write(buf)
        compressed = compressed.getvalue()

        return compressed

    # noinspection PyMethodMayBeStatic
    def decode(self, buf, out=None):

        if PY2:  # pragma: py3 no cover
            # ensure buffer, PY2 cannot handle things like bytearray
            buf = buffer(buf)

        # do decompression
        buf = io.BytesIO(buf)
        with _gzip.GzipFile(fileobj=buf, mode='rb') as decompressor:
            decompressed = decompressor.read()

        # handle destination - Python standard library zlib module does not
        # support direct decompression into buffer, so we have to copy into
        # out if given
        return buffer_copy(decompressed, out)
