# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import unittest
import itertools


import numpy as np


try:
    from numcodecs.msgpacks import LegacyMsgPack, MsgPack
    codecs = [LegacyMsgPack(), MsgPack()]
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("msgpack not available")


from numcodecs.tests.common import (check_config, check_repr, check_encode_decode_array,
                                    check_backwards_compatibility, greetings)


# object array with strings
# object array with mix strings / nans
# object array with mix of string, int, float
# ...
arrays = [
    np.array(['foo', 'bar', 'baz'] * 300, dtype=object),
    np.array([['foo', 'bar', np.nan]] * 300, dtype=object),
    np.array(['foo', 1.0, 2] * 300, dtype=object),
    np.arange(1000, dtype='i4'),
    np.array(['foo', 'bar', 'baz'] * 300),
    np.array(['foo', ['bar', 1.0, 2], {'a': 'b', 'c': 42}] * 300, dtype=object),
    np.array(greetings * 100),
    np.array(greetings * 100, dtype=object),
]


def test_encode_decode():
    for arr, codec in itertools.product(arrays, codecs):
        check_encode_decode_array(arr, codec)


def test_config():
    for codec in codecs:
        check_config(codec)


def test_repr():
    check_repr("MsgPack(encoding='utf-8')")
    check_repr("MsgPack(encoding='ascii')")
    check_repr("LegacyMsgPack(encoding='utf-8')")
    check_repr("LegacyMsgPack(encoding='ascii')")


def test_backwards_compatibility():
    for codec in codecs:
        check_backwards_compatibility(codec.codec_id, arrays, [codec])


def test_non_numpy_inputs():
    # numpy will infer a range of different shapes and dtypes for these inputs.
    # Make sure that round-tripping through encode preserves this.
    data = [
        [0, 1],
        [[0, 1], [2, 3]],
        [[0], [1], [2, 3]],
        [[[0, 0]], [[1, 1]], [[2, 3]]],
        ["1"],
        ["11", "11"],
        ["11", "1", "1"],
        [{}],
        [{"key": "value"}, ["list", "of", "strings"]],
    ]
    for input_data in data:
        for codec in codecs:
            output_data = codec.decode(codec.encode(input_data))
            assert np.array_equal(np.array(input_data), output_data)


def test_legacy_codec_broken():
    # Simplest demonstration of why the MsgPack codec needed to be changed.
    # The LegacyMsgPack codec didn't include shape information in the serialised
    # bytes, which gave different shapes in the input and output under certain
    # circumstances.
    a = np.empty(2, dtype=object)
    a[0] = [0, 1]
    a[1] = [2, 3]
    codec = LegacyMsgPack()
    b = codec.decode(codec.encode(a))
    assert a.shape == (2,)
    assert b.shape == (2, 2)
    assert not np.array_equal(a, b)

    # Now show that the MsgPack codec handles this case properly.
    codec = MsgPack()
    b = codec.decode(codec.encode(a))
    assert np.array_equal(a, b)
