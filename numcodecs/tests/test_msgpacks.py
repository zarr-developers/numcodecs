# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np
import nose

try:
    from numcodecs.msgpacks import MsgPack
except ImportError:  # pragma: no cover
    raise nose.SkipTest("msgpack-python not available")

from numcodecs.tests.common import check_config, check_repr, check_encode_decode_array, \
    check_backwards_compatibility


# object array with strings
# object array with mix strings / nans
# object array with mix of string, int, float
arrays = [
    np.array(['foo', 'bar', 'baz'] * 300, dtype=object),
    np.array([['foo', 'bar', np.nan]] * 300, dtype=object),
    np.array(['foo', 1.0, 2] * 300, dtype=object),
    np.arange(1000, dtype='i4'),
    np.array(['foo', 'bar', 'baz'] * 300),
]


def test_encode_decode():
    for arr in arrays:
        codec = MsgPack()
        check_encode_decode_array(arr, codec)


def test_config():
    codec = MsgPack()
    check_config(codec)


def test_repr():
    check_repr("MsgPack(encoding='utf-8')")
    check_repr("MsgPack(encoding='ascii')")


def test_backwards_compatibility():
    check_backwards_compatibility(MsgPack.codec_id, arrays, [MsgPack()])
