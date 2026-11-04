import numpy as np
from numpy.testing import assert_array_equal

from numcodecs.packbits import PackBits
from tests.common import (
    check_backwards_compatibility,
    check_config,
    check_encode_decode,
    check_repr,
)

arrays = [
    np.random.randint(0, 2, size=1000, dtype=bool),
    np.random.randint(0, 2, size=(100, 10), dtype=bool),
    np.random.randint(0, 2, size=(10, 10, 10), dtype=bool),
    np.random.randint(0, 2, size=1000, dtype=bool).reshape(10, 10, 10, order='F'),
]


def test_encode_decode():
    codec = PackBits()
    for arr in arrays:
        check_encode_decode(arr, codec, order='C')
    # check different number of left-over bits
    arr = np.random.randint(0, 2, size=1000, dtype=bool)
    for size in list(range(1, 17)):
        check_encode_decode(arr[:size], codec, order='C')


def test_config():
    codec = PackBits()
    check_config(codec)


def test_repr():
    check_repr("PackBits()")


def test_backwards_compatibility():
    check_backwards_compatibility(PackBits.codec_id, arrays, [PackBits()])


def test_encode_f_contiguous_logical_order():
    # regression test for https://github.com/zarr-developers/numcodecs/issues/850
    # the encoded stream must hold elements in logical (C) order, so encoding
    # an F-contiguous array produces the same bytes as its C-ordered equivalent
    arr_f = np.asfortranarray(np.array([[True, False, True], [False, True, True]]))
    arr_c = np.ascontiguousarray(arr_f)
    codec = PackBits()
    assert_array_equal(codec.encode(arr_f), codec.encode(arr_c))

    # a consumer reshaping the decoded 1-D stream in C order gets the array back
    dec = codec.decode(codec.encode(arr_f)).reshape(arr_f.shape)
    assert_array_equal(dec, arr_f)

    # decoding into an F-ordered output buffer lands values in the right places
    out = np.empty(arr_f.shape, dtype=bool, order='F')
    codec.decode(codec.encode(arr_f), out=out)
    assert_array_equal(out, arr_f)
