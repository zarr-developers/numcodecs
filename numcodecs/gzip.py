# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import gzip as _gzip
import io


from .abc import Codec
from .compat import ensure_bytes, ensure_ndarray, ensure_contiguous_ndarray, PY2


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
        if PY2:  # pragma: py3 no cover
            # view as u1 needed on PY2
            # ref: https://github.com/zarr-developers/numcodecs/pull/128#discussion_r236786466
            buf = buf.view('u1')

        # do compression
        compressed = io.BytesIO()
        with _gzip.GzipFile(fileobj=compressed,
                            mode='wb',
                            compresslevel=self.level) as compressor:
            compressor.write(buf)

        try:
            compressed = compressed.getbuffer()
        except AttributeError:  # pragma: py3 no cover
            compressed = compressed.getvalue()

        return ensure_ndarray(compressed)

    # noinspection PyMethodMayBeStatic
    def decode(self, buf, out=None):

        # normalise inputs
        if PY2:  # pragma: py3 no cover
            # On Python 2, BytesIO always copies.
            # Merely ensure the data supports the (new) buffer protocol.
            buf = ensure_contiguous_ndarray(buf)
        else:  # pragma: py2 no cover
            # BytesIO only copies if the data is not of `bytes` type.
            # This allows `bytes` objects to pass through without copying.
            buf = ensure_bytes(buf)

        # do decompression
        buf = io.BytesIO(buf)
        with _gzip.GzipFile(fileobj=buf, mode='rb') as decompressor:
            if out is not None:
                out_view = ensure_contiguous_ndarray(out)
                decompressor.readinto(out_view)
                if decompressor.read(1) != b'':
                    raise ValueError("Unable to fit data into `out`")
            else:
                out = ensure_ndarray(decompressor.read())

        # handle destination - Python standard library zlib module does not
        # support direct decompression into buffer, so we have to copy into
        # out if given
        return out
