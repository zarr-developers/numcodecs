# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import itertools


import numpy as np
from nose.tools import eq_ as eq


from numcodecs.bz2 import BZ2
from numcodecs.tests.common import check_encode_decode
from numcodecs.registry import get_codec


codecs = [
    BZ2(),
    BZ2(level=1),
    BZ2(level=5),
    BZ2(level=9),
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
    codec = BZ2(level=3)
    config = codec.get_config()
    eq(codec, get_codec(config))


def test_repr():
    expect = "BZ2(level=3)"
    codec = eval(expect)
    actual = repr(codec)
    eq(expect, actual)
