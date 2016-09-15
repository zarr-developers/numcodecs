# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from numcodecs.compat import PY2


if not PY2:

    import lzma as _lzma
    import itertools
    import numpy as np
    from numcodecs.lzma import LZMA
    from numcodecs.tests.common import check_encode_decode, check_config, \
        check_repr

    codecs = [
        LZMA(),
        LZMA(preset=1),
        LZMA(preset=5),
        LZMA(preset=9),
        LZMA(format=_lzma.FORMAT_RAW,
             filters=[dict(id=_lzma.FILTER_LZMA2, preset=1)])
    ]

    # mix of dtypes: integer, float, bool, string
    # mix of shapes: 1D, 2D, 3D
    # mix of orders: C, F
    arrays = [
        np.arange(1000, dtype='i4'),
        np.linspace(1000, 1001, 1000, dtype='f8'),
        np.random.normal(loc=1000, scale=1, size=(100, 10)),
        np.random.randint(0, 2, size=1000, dtype=bool).reshape(100, 10,
                                                               order='F'),
        np.random.choice([b'a', b'bb', b'ccc'], size=1000).reshape(10, 10, 10)
    ]

    def test_encode_decode():
        for arr, codec in itertools.product(arrays, codecs):
            check_encode_decode(arr, codec)

    def test_config():
        codec = LZMA(preset=1, format=_lzma.FORMAT_XZ,
                     check=_lzma.CHECK_NONE, filters=None)
        check_config(codec)

    def test_repr():
        check_repr('LZMA(format=1, check=0, preset=1, filters=None)')
