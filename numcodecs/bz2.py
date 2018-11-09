# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import bz2 as _bz2


from numcodecs.abc import Codec
from numcodecs.compat import buffer_copy, to_buffer


class BZ2(Codec):
    """Codec providing compression using bzip2 via the Python standard library.

    Parameters
    ----------
    level : int
        Compression level.

    """

    codec_id = 'bz2'

    def __init__(self, level=1):
        self.level = level

    def encode(self, buf):

        buf = to_buffer(buf)

        # do compression
        return _bz2.compress(buf, self.level)

    # noinspection PyMethodMayBeStatic
    def decode(self, buf, out=None):

        buf = to_buffer(buf)

        # do decompression
        dec = _bz2.decompress(buf)

        # handle destination - Python standard library bz2 module does not
        # support direct decompression into buffer, so we have to copy into
        # out if given
        return buffer_copy(dec, out)
