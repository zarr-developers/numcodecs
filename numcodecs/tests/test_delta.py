import numpy as np
import pytest
from numpy.testing import assert_array_equal

from numcodecs.delta import Delta
from numcodecs.tests.common import (
    check_backwards_compatibility,
    check_config,
    check_encode_decode,
    check_repr,
)

# mix of dtypes: integer, float
# mix of shapes: 1D, 2D, 3D
# mix of orders: C, F
# mix of encoding types: All available types for each arrays
arrays = [
    (np.random.randint(0, 1, size=110, dtype='?').reshape(10, 11), ('?', '<u1', '<i1')),
    (np.arange(1000, dtype='<i4'), ('<i4', '<i2', '<u4', 'u2')),
    (np.linspace(1000, 1001, 1000, dtype='<f4').reshape(100, 10), ('<f4',)),
    (np.random.normal(loc=1000, scale=1, size=(10, 10, 10)).astype('<f8'), ('<f8',)),
    (
        np.random.randint(0, 200, size=1000, dtype='u2').astype('<u2').reshape(100, 10, order='F'),
        ('<i2',),
    ),
]


def test_encode_decode():
    for arr, encoding_types in arrays:
        for astype in encoding_types:
            codec = Delta(dtype=arr.dtype, astype=astype)
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


def test_config():
    codec = Delta(dtype='<i4', astype='<i2')
    check_config(codec)


def test_repr():
    check_repr("Delta(dtype='<i4', astype='<i2')")


def test_backwards_compatibility():
    for arr, _ in arrays:
        codec = Delta(dtype=arr.dtype)
        check_backwards_compatibility(Delta.codec_id, [arr], [codec], prefix=str(arr.dtype))


def test_errors():
    with pytest.raises(ValueError):
        Delta(dtype=object)
    with pytest.raises(ValueError):
        Delta(dtype='i8', astype=object)


# overflow tests
# Note: Before implementing similar test for integer -> integer types, check numpy/numpy#8987.
oveflow_proned_float_float_pairs = [
    ('f4', 'f2'),
    ('f8', 'f4'),
]


def test_oveflow_proned_float_float_encode():
    for dtype, astype in oveflow_proned_float_float_pairs:
        codec = Delta(dtype=dtype, astype=astype)
        arr = np.array([0, np.finfo(astype).max.astype(dtype) * 2], dtype=dtype)
        with pytest.warns(RuntimeWarning, match=r"overflow encountered"):
            codec.encode(arr)


overflow_proned_integer_float_paris = [
    ('i4', 'f2'),
    ('i8', 'f2'),
    ('u4', 'f2'),
    ('u8', 'f2'),
]


def test_oveflow_proned_integer_float_encode():
    for dtype, astype in overflow_proned_integer_float_paris:
        codec = Delta(dtype=dtype, astype=astype)
        arr = np.array([0, int(np.rint(np.finfo(astype).max)) * 2], dtype=dtype)
        with pytest.warns(RuntimeWarning, match=r"overflow encountered"):
            codec.encode(arr)
