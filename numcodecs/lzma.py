# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


_lzma = None
try:
    import lzma as _lzma
except ImportError:  # pragma: no cover
    try:
        from backports import lzma as _lzma
    except ImportError:
        pass


if _lzma:

    import numpy as np
    from .abc import Codec
    from .compat import buffer_copy, handle_datetime

    # noinspection PyShadowingBuiltins
    class LZMA(Codec):
        """Codec providing compression using lzma via the Python standard
        library (available on Python 3 and Python 2 with ``backports.lzma``).

        Parameters
        ----------
        format : integer, optional
            One of the lzma format codes, e.g., ``lzma.FORMAT_XZ``.
        check : integer, optional
            One of the lzma check codes, e.g., ``lzma.CHECK_NONE``.
        preset : integer, optional
            An integer between 0 and 9 inclusive, specifying the compression
            level.
        filters : list, optional
            A list of dictionaries specifying compression filters. If
            filters are provided, 'preset' must be None.

        """

        codec_id = 'lzma'

        def __init__(self, format=1, check=-1, preset=None, filters=None):
            self.format = format
            self.check = check
            self.preset = preset
            self.filters = filters

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

            # do compression
            return _lzma.compress(buf, format=self.format, check=self.check,
                                  preset=self.preset, filters=self.filters)

        def decode(self, buf, out=None):

            # do decompression
            dec = _lzma.decompress(buf, format=self.format,
                                   filters=self.filters)

            # handle destination
            return buffer_copy(dec, out)

        def __repr__(self):
            r = '%s(format=%r, check=%r, preset=%r, filters=%r)' % \
                (type(self).__name__, self.format, self.check, self.preset,
                 self.filters)
            return r
