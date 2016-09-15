# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import itertools


import numpy as np
from nose.tools import assert_is_instance, eq_ as eq


from numcodecs.blosc import Blosc
from numcodecs.tests.common import check_encode_decode
from numcodecs.registry import get_codec


codecs = [
    Blosc(),
    Blosc(clevel=0),
    Blosc(cname='lz4'),
    Blosc(cname='lz4', clevel=1, shuffle=Blosc.NOSHUFFLE),
    Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE),
    Blosc(cname='lz4', clevel=9, shuffle=Blosc.BITSHUFFLE),
    Blosc(cname='zlib', clevel=1, shuffle=0),
    Blosc(cname='zstd', clevel=1, shuffle=1),
    Blosc(cname='blosclz', clevel=1, shuffle=2),
    Blosc(cname='snappy', clevel=1, shuffle=2),
]


# mix of dtypes: integer, float, bool, string
# mix of shapes: 1D, 2D, 3D
# mix of orders: C, F
arrays = [
    np.arange(1000, dtype='i4'),
    np.linspace(1000, 1001, 1000, dtype='f8'),
    np.random.normal(loc=1000, scale=1, size=(100, 10)),
    np.random.randint(0, 2, size=1000, dtype=bool).reshape(100, 10, order='F'),
    np.random.choice([b'a', b'bb', b'ccc'], size=1000).reshape(10, 10, 10)
]


def test_encode_decode():
    for arr, codec in itertools.product(arrays, codecs):
        check_encode_decode(arr, codec)


def test_get_config():
    codec = Blosc(cname='zstd', clevel=3, shuffle=1)
    config = codec.get_config()
    eq(codec, get_codec(config))


def test_repr():
    expect = "Blosc(cname='zstd', clevel=3, shuffle=1)"
    codec = eval(expect)
    actual = repr(codec)
    eq(expect, actual)
