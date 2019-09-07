# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import unittest


import numpy as np
import pytest


from numcodecs.compat import PY2
try:
    from numcodecs.vlen_nd import VLenNDArray
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("vlen-ndarray not available")
from numcodecs.tests.common import (check_config, check_repr,
                                    check_encode_decode_array,
                                    check_backwards_compatibility,
                                    assert_array_items_equal)


arrays = [
    np.array([np.array([1, 2, 3]),
              np.array([[4]]),
              np.array([[5, 6], [7, 8]])] * 300, dtype=object),
    np.array([np.array([1, 2, 3]),
              np.array([[4]]),
              np.array([[5, 6], [7, 8]])] * 300, dtype=object).reshape(90, 10),
]


codecs = [
    VLenNDArray('<i1'),
    VLenNDArray('<i2'),
    VLenNDArray('<i4'),
    VLenNDArray('<i8'),
    VLenNDArray('<u1'),
    VLenNDArray('<u2'),
    VLenNDArray('<u4'),
    VLenNDArray('<u8'),
]


def test_encode_decode():
    for arr in arrays:
        for codec in codecs:
            check_encode_decode_array(arr, codec)


def test_config():
    codec = VLenNDArray('<i8')
    check_config(codec)


def test_repr():
    check_repr("VLenNDArray(dtype='<i8')")


def test_backwards_compatibility():
    check_backwards_compatibility(VLenNDArray.codec_id, arrays, codecs)


def test_encode_errors():
    codec = VLenNDArray('<i8')
    with pytest.raises(ValueError):
        codec.encode('foo')
    with pytest.raises(ValueError):
        codec.encode(['foo', 'bar'])


def test_decode_errors():
    codec = VLenNDArray('<i8')
    with pytest.raises(TypeError):
        codec.decode(1234)
    # these should look like corrupt data
    with pytest.raises(ValueError):
        codec.decode(b'foo')
    with pytest.raises(ValueError):
        codec.decode(np.arange(2, 3, dtype='i4'))
    with pytest.raises(ValueError):
        codec.decode(np.arange(10, 20, dtype='i4'))
    with pytest.raises(ValueError if PY2 else TypeError):
        # exports old-style buffer interface on PY2, hence ValueError
        codec.decode(u'foo')

    # test out parameter
    enc = codec.encode(arrays[0])
    with pytest.raises(TypeError):
        codec.decode(enc, out=b'foo')
    with pytest.raises(TypeError):
        codec.decode(enc, out=u'foo')
    with pytest.raises(TypeError):
        codec.decode(enc, out=123)
    with pytest.raises(ValueError):
        codec.decode(enc, out=np.zeros(10, dtype='i4'))


def test_encode_none():
    a = np.array([[1, 3], None, [[4, 7]]], dtype=object)
    codec = VLenNDArray(int)
    enc = codec.encode(a)
    dec = codec.decode(enc)
    expect = np.array([np.array([1, 3]),
                       np.array([]),
                       np.array([[4, 7]])], dtype=object)
    assert_array_items_equal(expect, dec)
