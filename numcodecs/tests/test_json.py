# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import itertools


import numpy as np


from numcodecs.json import LegacyJSON, JSON
from numcodecs.tests.common import (check_config, check_repr, check_encode_decode_array,
                                    check_backwards_compatibility, greetings)
json_codecs = [
    JSON(),
    JSON(indent=True),
]

legacy_json_codecs = [
    LegacyJSON(),
    LegacyJSON(indent=True),
]

codecs = json_codecs + legacy_json_codecs


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
    r = (
        "JSON(encoding='utf-8', allow_nan=True, check_circular=True, ensure_ascii=True,\n"
        "     indent=None, separators=(',', ':'), skipkeys=False, sort_keys=True,\n"
        "     strict=True)"
    )
    check_repr(r)
    r = (
        "LegacyJSON(encoding='utf-8', allow_nan=True, check_circular=True,\n"
        "     ensure_ascii=True, indent=None, separators=(',', ':'), skipkeys=False,\n"
        "     sort_keys=True, strict=True)"
    )
    check_repr(r)


def test_backwards_compatibility():
    check_backwards_compatibility(LegacyJSON.codec_id, arrays, legacy_json_codecs)
    check_backwards_compatibility(JSON.codec_id, arrays, json_codecs)


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
    # Simplest demonstration of why the JSON codec needed to be changed.
    # The LegacyJSON codec didn't include shape information in the serialised
    # bytes, which gave different shapes in the input and output under certain
    # circumstances.
    a = np.empty(2, dtype=object)
    a[0] = [0, 1]
    a[1] = [2, 3]
    codec = LegacyJSON()
    b = codec.decode(codec.encode(a))
    assert a.shape == (2,)
    assert b.shape == (2, 2)
    assert not np.array_equal(a, b)

    # Now show that the JSON codec handles this case properly.
    codec = JSON()
    b = codec.decode(codec.encode(a))
    assert np.array_equal(a, b)
