import itertools

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from numcodecs.fixedscaleoffset import FixedScaleOffset
from numcodecs.tests.common import (
    check_backwards_compatibility,
    check_config,
    check_encode_decode,
    check_repr,
)

arrays = [
    np.linspace(1000, 1001, 1000, dtype='<f8'),
    np.random.normal(loc=1000, scale=1, size=1000).astype('<f8'),
    np.linspace(1000, 1001, 1000, dtype='<f8').reshape(100, 10),
    np.linspace(1000, 1001, 1000, dtype='<f8').reshape(100, 10, order='F'),
    np.linspace(1000, 1001, 1000, dtype='<f8').reshape(10, 10, 10),
]


codecs = [
    FixedScaleOffset(offset=1000, scale=10, dtype='<f8', astype='<i1'),
    FixedScaleOffset(offset=1000, scale=10**2, dtype='<f8', astype='<i2'),
    FixedScaleOffset(offset=1000, scale=10**6, dtype='<f8', astype='<i4'),
    FixedScaleOffset(offset=1000, scale=10**12, dtype='<f8', astype='<i8'),
    FixedScaleOffset(offset=1000, scale=10**12, dtype='<f8'),
]


def test_encode_decode():
    for arr, codec in itertools.product(arrays, codecs):
        precision = int(np.log10(codec.scale))
        check_encode_decode(arr, codec, precision=precision)


@pytest.mark.parametrize(
    ("offset", "scale", "expected"),
    [
        (1000, 10, [0, 6, 11, 17, 22, 28, 33, 39, 44, 50]),
        (1002.5, 10, [-25, -19, -14, -8, -3, 3, 8, 14, 19, 25]),
        (1000, 0.5, [0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
    ],
)
def test_encode(offset: float, scale: float, expected: list[int]):
    dtype = '<f8'
    astype = np.int16
    codec = FixedScaleOffset(scale=scale, offset=offset, dtype=dtype, astype=astype)
    arr = np.linspace(1000, 1005, 10, dtype=dtype)
    expect = np.array(expected, dtype=astype)
    actual = codec.encode(arr)
    assert_array_equal(expect, actual)
    assert np.dtype(astype) == actual.dtype


def test_config():
    codec = FixedScaleOffset(dtype='<f8', astype='<i4', scale=10, offset=100)
    check_config(codec)


def test_repr():
    stmt = "FixedScaleOffset(scale=10, offset=100, dtype='<f8', astype='<i4')"
    check_repr(stmt)


def test_backwards_compatibility():
    precision = [int(np.log10(codec.scale)) for codec in codecs]
    check_backwards_compatibility(FixedScaleOffset.codec_id, arrays, codecs, precision=precision)


def test_errors():
    with pytest.raises(ValueError):
        FixedScaleOffset(dtype=object, astype='i4', scale=10, offset=100)
    with pytest.raises(ValueError):
        FixedScaleOffset(dtype='f8', astype=object, scale=10, offset=100)
