# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import gzip as _gzip
import io


from .abc import Codec
from .compat import ndarray_copy, ensure_contiguous_ndarray


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

        # normalise inputs
        buf = ensure_contiguous_ndarray(buf)
        buf = memoryview(buf)  # needed on PY2 for unknown reasons

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

        # normalise inputs
        buf = ensure_contiguous_ndarray(buf)
        if out is not None:
            out = ensure_contiguous_ndarray(out)

        # do decompression
        buf = io.BytesIO(buf)
        with _gzip.GzipFile(fileobj=buf, mode='rb') as decompressor:
            decompressed = decompressor.read()

        # handle destination - Python standard library zlib module does not
        # support direct decompression into buffer, so we have to copy into
        # out if given
        return ndarray_copy(decompressed, out)
