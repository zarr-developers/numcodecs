# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


try:
    import lzma as _lzma
except ImportError:  # pragma: no cover
    pass
else:

    import numpy as np
    from numcodecs.abc import Codec
    from numcodecs.compat import buffer_copy

    # noinspection PyShadowingBuiltins
    class LZMA(Codec):
        """Codec providing compression using lzma via the Python standard
        library (only available under Python 3).

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

            # if numpy array, can only handle C contiguous directly
            if isinstance(buf, np.ndarray) and not buf.flags.c_contiguous:
                buf = buf.tobytes(order='A')

            # do compression
            return _lzma.compress(buf, format=self.format, check=self.check,
                                  preset=self.preset, filters=self.filters)

        def decode(self, buf, out=None):

            # setup filters
            if self.format == _lzma.FORMAT_RAW:
                # filters needed
                filters = self.filters
            else:
                # filters should not be specified
                filters = None

            # do decompression
            dec = _lzma.decompress(buf, format=self.format, filters=filters)

            # handle destination
            return buffer_copy(dec, out)

        def get_config(self):
            config = dict()
            config['id'] = self.codec_id
            config['format'] = self.format
            config['check'] = self.check
            config['preset'] = self.preset
            config['filters'] = self.filters
            return config

        def __repr__(self):
            r = '%s(format=%r, check=%r, preset=%r, filters=%r)' % \
                (type(self).__name__, self.format, self.check, self.preset,
                 self.filters)
            return r
