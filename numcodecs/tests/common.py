# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import array
import json as _json
import os
from glob import glob


import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest


from numcodecs.compat import buffer_tobytes, ndarray_from_buffer
from numcodecs import *  # flake8: noqa


greetings = [u'¡Hola mundo!', u'Hej Världen!', u'Servus Woid!', u'Hei maailma!',
             u'Xin chào thế giới', u'Njatjeta Botë!', u'Γεια σου κόσμε!',
             u'こんにちは世界', u'世界，你好！', u'Helló, világ!', u'Zdravo svete!',
             u'เฮลโลเวิลด์']


def compare_arrays(arr, res, precision=None):

    # ensure numpy array
    if not isinstance(res, np.ndarray):
        res = ndarray_from_buffer(res, dtype=arr.dtype)
    elif res.dtype != arr.dtype:
        res = res.view(arr.dtype)

    # convert to correct shape
    if arr.flags.f_contiguous:
        order = 'F'
    else:
        order = 'C'
    res = res.reshape(arr.shape, order=order)

    # exact compare
    if precision is None:
        assert_array_equal(arr, res)

    # fuzzy compare
    else:
        assert_array_almost_equal(arr, res, decimal=precision)


def check_encode_decode(arr, codec, precision=None):

    # N.B., watch out here with blosc compressor, if the itemsize of
    # the source buffer is different then the results of encoding
    # (i.e., compression) may be different. Hence we *do not* require that
    # the results of encoding be identical for all possible inputs, rather
    # we just require that the results of the encode/decode round-trip can
    # be compared to the original array.

    # encoding should support any object exporting the buffer protocol,
    # as well as array.array in PY2

    # test encoding of numpy array
    enc = codec.encode(arr)
    dec = codec.decode(enc)
    compare_arrays(arr, dec, precision=precision)

    # test encoding of bytes
    buf = arr.tobytes(order='A')
    enc = codec.encode(buf)
    dec = codec.decode(enc)
    compare_arrays(arr, dec, precision=precision)

    # test encoding of bytearray
    buf = bytearray(arr.tobytes(order='A'))
    enc = codec.encode(buf)
    dec = codec.decode(enc)
    compare_arrays(arr, dec, precision=precision)

    # test encoding of array.array
    buf = array.array('b', arr.tobytes(order='A'))
    enc = codec.encode(buf)
    dec = codec.decode(enc)
    compare_arrays(arr, dec, precision=precision)

    # decoding should support any object exporting the buffer protocol,
    # as well as array.array in PY2

    # setup
    enc_bytes = buffer_tobytes(enc)

    # test decoding of raw bytes
    dec = codec.decode(enc_bytes)
    compare_arrays(arr, dec, precision=precision)

    # test decoding of bytearray
    dec = codec.decode(bytearray(enc_bytes))
    compare_arrays(arr, dec, precision=precision)

    # test decoding of array.array
    buf = array.array('b', enc_bytes)
    dec = codec.decode(buf)
    compare_arrays(arr, dec, precision=precision)

    # test decoding of numpy array
    buf = np.frombuffer(enc_bytes, dtype='u1')
    dec = codec.decode(buf)
    compare_arrays(arr, dec, precision=precision)

    # test decoding directly into numpy array
    out = np.empty_like(arr)
    codec.decode(enc_bytes, out=out)
    compare_arrays(arr, out, precision=precision)

    # test decoding directly into bytearray
    out = bytearray(arr.nbytes)
    codec.decode(enc_bytes, out=out)
    # noinspection PyTypeChecker
    compare_arrays(arr, out, precision=precision)


def assert_array_items_equal(res, arr):

    assert isinstance(res, np.ndarray)
    res = res.reshape(-1, order='A')
    arr = arr.reshape(-1, order='A')
    assert res.shape == arr.shape
    assert res.dtype == arr.dtype

    # numpy asserts don't compare object arrays
    # properly; assert that we have the same nans
    # and values
    arr = arr.ravel().tolist()
    res = res.ravel().tolist()
    for a, r in zip(arr, res):
        if isinstance(a, np.ndarray):
            assert_array_equal(a, r)
        elif a != a:
            assert r != r
        else:
            assert a == r


def check_encode_decode_array(arr, codec):

    enc = codec.encode(arr)
    dec = codec.decode(enc)
    assert_array_items_equal(arr, dec)

    out = np.empty_like(arr)
    codec.decode(enc, out=out)
    assert_array_items_equal(arr, out)


def check_config(codec):
    config = codec.get_config()
    # round-trip through JSON to check serialization
    config = _json.loads(_json.dumps(config))
    assert codec == get_codec(config)


def check_repr(stmt):
    # check repr matches instantiation statement
    codec = eval(stmt)
    actual = repr(codec)
    assert stmt == actual


def check_backwards_compatibility(codec_id, arrays, codecs, precision=None, prefix=None):

    # setup directory to hold data fixture
    if prefix:
        fixture_dir = os.path.join('fixture', codec_id, prefix)
    else:
        fixture_dir = os.path.join('fixture', codec_id)
    if not os.path.exists(fixture_dir):  # pragma: no cover
        os.makedirs(fixture_dir)

    # save fixture data
    for i, arr in enumerate(arrays):
        arr_fn = os.path.join(fixture_dir, 'array.{:02d}.npy'.format(i))
        if not os.path.exists(arr_fn):  # pragma: no cover
            np.save(arr_fn, arr)

    # load fixture data
    for arr_fn in glob(os.path.join(fixture_dir, 'array.*.npy')):

        # setup
        i = int(arr_fn.split('.')[-2])
        arr = np.load(arr_fn)
        arr_bytes = buffer_tobytes(arr)
        if arr.flags.f_contiguous:
            order = 'F'
        else:
            order = 'C'

        for j, codec in enumerate(codecs):

            # setup a directory to hold encoded data
            codec_dir = os.path.join(fixture_dir, 'codec.{:02d}'.format(j))
            if not os.path.exists(codec_dir):  # pragma: no cover
                os.makedirs(codec_dir)

            # file with codec configuration information
            codec_fn = os.path.join(codec_dir, 'config.json')

            # one time save config
            if not os.path.exists(codec_fn):  # pragma: no cover
                with open(codec_fn, mode='w') as cf:
                    _json.dump(codec.get_config(), cf, sort_keys=True, indent=4)

            # load config and compare with expectation
            with open(codec_fn, mode='r') as cf:
                config = _json.load(cf)
                assert codec == get_codec(config)

            enc_fn = os.path.join(codec_dir, 'encoded.{:02d}.dat'.format(i))

            # one time encode and save array
            if not os.path.exists(enc_fn):  # pragma: no cover
                enc = codec.encode(arr)
                with open(enc_fn, mode='wb') as ef:
                    ef.write(enc)

            # load and decode data
            with open(enc_fn, mode='rb') as ef:
                enc = ef.read()
                dec = codec.decode(enc)
                dec_arr = ndarray_from_buffer(dec, dtype=arr.dtype)
                dec_arr = dec_arr.reshape(arr.shape, order=order)
                if precision and precision[j] is not None:
                    assert_array_almost_equal(arr, dec_arr, decimal=precision[j])
                elif arr.dtype == 'object':
                    assert_array_items_equal(arr, dec_arr)
                else:
                    assert_array_equal(arr, dec_arr)
                    assert arr_bytes == buffer_tobytes(dec)


def check_err_decode_object_buffer(compressor):
    # cannot decode directly into object array, leads to segfaults
    a = np.arange(10)
    enc = compressor.encode(a)
    out = np.empty(10, dtype=object)
    with pytest.raises(ValueError):
        compressor.decode(enc, out=out)
    print(out[:])


def check_err_encode_object_buffer(compressor):
    # compressors cannot encode object array
    a = np.array(['foo', 'bar', 'baz'], dtype=object)
    with pytest.raises(ValueError):
        compressor.encode(a)
