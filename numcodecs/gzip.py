# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import gzip as _gzip
import io


from .abc import Codec
from .compat import buffer_copy, to_buffer


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

        buf = memoryview(to_buffer(buf))

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

        buf = to_buffer(buf)

        # do decompression
        buf = io.BytesIO(buf)
        with _gzip.GzipFile(fileobj=buf, mode='rb') as decompressor:
            decompressed = decompressor.read()

        # handle destination - Python standard library zlib module does not
        # support direct decompression into buffer, so we have to copy into
        # out if given
        return buffer_copy(decompressed, out)
