import itertools

import numpy as np
import pytest

from numcodecs.json import JSON
from tests.common import (
    check_backwards_compatibility,
    check_config,
    check_encode_decode_array,
    check_repr,
    greetings,
)

codecs = [
    JSON(),
    JSON(indent=True),
]


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
    np.array([[0, 1], [2, 3]], dtype=object),
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


def test_backwards_compatibility():
    check_backwards_compatibility(JSON.codec_id, arrays, codecs)


@pytest.mark.parametrize(
    ('input_data', 'dtype'),
    [
        ([0, 1], None),
        ([[0, 1], [2, 3]], None),
        ([[0], [1], [2, 3]], object),
        ([[[0, 0]], [[1, 1]], [[2, 3]]], None),
        (["1"], None),
        (["11", "11"], None),
        (["11", "1", "1"], None),
        ([{}], None),
        ([{"key": "value"}, ["list", "of", "strings"]], object),
        ([0], None),
        ([{'hi': 0}], "object"),
        (["hi"], "object"),
        (0, None),
    ],
)
def test_non_numpy_inputs(input_data, dtype):
    # numpy will infer a range of different shapes and dtypes for these inputs.
    # Make sure that round-tripping through encode preserves this.
    data = np.array(input_data, dtype=dtype)
    for codec in codecs:
        output_data = codec.decode(codec.encode(data))
        assert input_data == output_data.tolist()


@pytest.mark.parametrize(
    "arr",
    [
        # from #518
        np.array(
            [
                (1, 'dataset_1', {'source': '.', 'path': '/dataset_1', 'object_id': None}),
                (2, 'dataset_2', {'source': '.', 'path': '/dataset_2', 'object_id': None}),
            ],
            dtype=[('id', '<i4'), ('name', 'O'), ('reference', 'O')],
        ),
        np.array((1, 2.5), dtype=[('x', '<i8'), ('y', '<f4')]),
        np.arange(12).astype('f8').view([('a', '<f8'), ('b', '<f8', (2,))]).reshape(2, 2),
        np.zeros(3, dtype=[('outer', [('p', '<i4'), ('q', '<f8')]), ('z', '<u1')]),
        np.zeros(2, dtype=[('inner', [('p', '<i4')], (2,))]),
    ],
)
def test_structured_dtype(arr):
    codec = JSON()
    dec = codec.decode(codec.encode(arr))
    assert dec.dtype == arr.dtype
    assert dec.shape == arr.shape
    np.testing.assert_equal(dec.tolist(), arr.tolist())
