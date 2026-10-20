import numpy as np
import pytest
from numpy.testing import assert_array_equal

from numcodecs.categorize import Categorize
from tests.common import (
    check_backwards_compatibility,
    check_config,
    check_encode_decode,
    check_encode_decode_array,
)

labels = ['ƒöõ', 'ßàř', 'ßāẑ', 'ƪùüx']
arrays = [
    np.random.choice(labels, size=1000),
    np.random.choice(labels, size=(100, 10)),
    np.random.choice(labels, size=(10, 10, 10)),
    np.random.choice(labels, size=1000).reshape(100, 10, order='F'),
]
arrays_object = [a.astype(object) for a in arrays]


def test_encode_decode():
    # unicode dtype
    for arr in arrays:
        codec = Categorize(labels, dtype=arr.dtype)
        check_encode_decode(arr, codec, order='C')

    # object dtype
    for arr in arrays_object:
        codec = Categorize(labels, dtype=arr.dtype)
        check_encode_decode_array(arr, codec, order='C')


def test_encode():
    for dtype in 'U', object:
        arr = np.array(['ƒöõ', 'ßàř', 'ƒöõ', 'ßāẑ', 'ƪùüx'], dtype=dtype)
        # miss off quux
        codec = Categorize(labels=labels[:-1], dtype=arr.dtype, astype='u1')

        # test encoding
        expect = np.array([1, 2, 1, 3, 0], dtype='u1')
        enc = codec.encode(arr)
        assert_array_equal(expect, enc)
        assert expect.dtype == enc.dtype

        # test decoding with unexpected value
        dec = codec.decode(enc)
        expect = arr.copy()
        expect[expect == 'ƪùüx'] = ''
        assert_array_equal(expect, dec)
        assert arr.dtype == dec.dtype


def test_config():
    codec = Categorize(labels=labels, dtype='U4')
    check_config(codec)


def test_repr():
    dtype = '<U3'
    astype = '|u1'
    codec = Categorize(labels=['foo', 'bar', 'baz', 'qux'], dtype=dtype, astype=astype)
    expect = "Categorize(dtype='<U3', astype='|u1', labels=['foo', 'bar', 'baz', ...])"
    actual = repr(codec)
    assert expect == actual

    dtype = '<U4'
    astype = '|u1'
    codec = Categorize(labels=labels, dtype=dtype, astype=astype)
    expect = "Categorize(dtype='<U4', astype='|u1', labels=['ƒöõ', 'ßàř', 'ßāẑ', ...])"
    actual = repr(codec)
    assert expect == actual


def test_backwards_compatibility():
    codec = Categorize(labels=labels, dtype='<U4', astype='u1')
    check_backwards_compatibility(Categorize.codec_id, arrays, [codec], prefix='U')
    codec = Categorize(labels=labels, dtype=object, astype='u1')
    check_backwards_compatibility(Categorize.codec_id, arrays_object, [codec], prefix='O')


def test_errors():
    with pytest.raises(TypeError):
        Categorize(labels=['foo', 'bar'], dtype='S6')
    with pytest.raises(TypeError):
        Categorize(labels=['foo', 'bar'], dtype='U6', astype=object)


def test_encode_f_contiguous_logical_order():
    # regression test for https://github.com/zarr-developers/numcodecs/issues/850
    # the encoded stream must hold elements in logical (C) order, so encoding
    # an F-contiguous array produces the same bytes as its C-ordered equivalent
    for dtype in 'U1', object:
        arr_f = np.asfortranarray(np.array([['a', 'b'], ['a', 'a']], dtype=dtype))
        arr_c = np.ascontiguousarray(arr_f)
        codec = Categorize(labels=['a', 'b'], dtype=dtype)
        assert_array_equal(codec.encode(arr_f), codec.encode(arr_c))

        # a consumer reshaping the decoded 1-D stream in C order gets the array back
        dec = codec.decode(codec.encode(arr_f)).reshape(arr_f.shape)
        assert_array_equal(dec, arr_f)

        # decoding into an F-ordered output buffer lands values in the right places
        out = np.empty(arr_f.shape, dtype=dtype, order='F')
        codec.decode(codec.encode(arr_f), out=out)
        assert_array_equal(out, arr_f)
