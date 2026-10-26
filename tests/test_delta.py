import itertools

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from numcodecs import get_codec
from numcodecs.delta import Delta
from tests.common import (
    check_backwards_compatibility,
    check_config,
    check_encode_decode,
    check_repr,
)

# mix of dtypes: boolean, integer, float
# mix of shapes: 1D, 2D, 3D
# mix of orders: C, F
arrays = [
    np.tile([False, True, True, False, True], 22).reshape(10, 11),
    np.arange(1000, dtype='<i4'),
    np.linspace(1000, 1001, 1000, dtype='<f4').reshape(100, 10),
    np.random.normal(loc=1000, scale=1, size=(10, 10, 10)).astype('<f8'),
    np.random.randint(0, 200, size=1000, dtype='u2').astype('<u2').reshape(100, 10, order='F'),
]


def test_encode_decode():
    for arr in arrays:
        codec = Delta(dtype=arr.dtype)
        check_encode_decode(arr, codec)


def test_encode():
    dtype = 'i8'
    astype = 'i4'
    codec = Delta(dtype=dtype, astype=astype)
    arr = np.arange(10, 20, 1, dtype=dtype)
    expect = np.array([10] + ([1] * 9), dtype=astype)
    actual = codec.encode(arr)
    assert_array_equal(expect, actual)
    assert np.dtype(astype) == actual.dtype


@pytest.mark.parametrize('astype', ['?', 'i1', 'u1', '<i2', '<f4', '<c16'])
def test_boolean_encoded_compatibility(astype):
    codec = get_codec(Delta(dtype='?', astype=astype).get_config())
    original = np.array([True, True, False, True])
    # These transition values are the existing encoded format, including for
    # wider encoded dtypes. Decode must also read previously stored bytes.
    encoded = np.array([1, 0, 1, 1], dtype=astype)
    assert codec.encode(original).tobytes() == encoded.tobytes()
    assert_array_equal(codec.decode(encoded.tobytes()), original)
    out = np.empty_like(original)
    assert codec.decode(encoded.tobytes(), out=out) is out
    assert_array_equal(out, original)


@pytest.mark.parametrize('order', ['C', 'F'])
def test_boolean_layout(order):
    original = np.array([True, False, True, True, False, False]).reshape(2, 3, order=order)
    check_encode_decode(original, Delta(dtype='?'))


def test_boolean_patterns():
    codec = Delta(dtype='?')
    for size in range(1, 9):
        for pattern in itertools.product([False, True], repeat=size):
            # Independent transition oracle, without NumPy difference/cumsum.
            transitions = [pattern[0]] + [
                a != b for a, b in zip(pattern[1:], pattern[:-1], strict=True)
            ]
            encoded = np.array(transitions, dtype=bool)
            assert_array_equal(codec.encode(np.array(pattern)), encoded)
            assert_array_equal(codec.decode(encoded.tobytes()), pattern)


def test_config():
    codec = Delta(dtype='<i4', astype='<i2')
    check_config(codec)


def test_repr():
    check_repr("Delta(dtype='<i4', astype='<i2')")


def test_backwards_compatibility():
    for arr in arrays:
        codec = Delta(dtype=arr.dtype)
        check_backwards_compatibility(Delta.codec_id, [arr], [codec], prefix=str(arr.dtype))


def test_errors():
    with pytest.raises(ValueError):
        Delta(dtype=object)
    with pytest.raises(ValueError):
        Delta(dtype='i8', astype=object)
