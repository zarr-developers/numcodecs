# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import array
import json


import numpy as np
from nose.tools import eq_ as eq
from numpy.testing import assert_array_almost_equal


from numcodecs.compat import buffer_tobytes, ndarray_from_buffer
from numcodecs import *  # flake8: noqa


def check_encode_decode(arr, codec, precision=None):

    # N.B., watch out here with blosc compressor, if the itemsize of
    # the source buffer is different then the results of encoding
    # (i.e., compression) may be different. Hence we *do not* require that
    # the results of encoding be identical for all possible inputs, rather
    # we just require that the results of the encode/decode round-trip can
    # be compared to the original array.

    # setup
    arr_bytes = buffer_tobytes(arr)
    if arr.flags.f_contiguous:
        order = 'F'
    else:
        order = 'C'

    # function to compare result after encode/decode round-trip against
    # original array

    def compare(res):
        if precision is None:
            eq(arr_bytes, buffer_tobytes(res))
        else:
            res = ndarray_from_buffer(res, dtype=arr.dtype)
            res = res.reshape(arr.shape, order=order)
            assert_array_almost_equal(arr, res, decimal=precision)

    # encoding should support any object exporting the buffer protocol,
    # as well as array.array in PY2

    # test encoding of numpy array
    enc = codec.encode(arr)
    dec = codec.decode(enc)
    compare(dec)

    # test encoding of raw bytes
    buf = arr.tobytes(order='A')
    enc = codec.encode(buf)
    dec = codec.decode(enc)
    compare(dec)

    # test encoding of array.array
    buf = array.array('b', arr.tobytes(order='A'))
    enc = codec.encode(buf)
    dec = codec.decode(enc)
    compare(dec)

    # decoding should support any object exporting the buffer protocol,
    # as well as array.array in PY2

    # setup
    enc_bytes = buffer_tobytes(enc)

    # test decoding of raw bytes
    dec = codec.decode(enc_bytes)
    compare(dec)

    # test decoding of array.array
    buf = array.array('b', enc_bytes)
    dec = codec.decode(buf)
    compare(dec)

    # test decoding of numpy array
    buf = np.frombuffer(enc_bytes, dtype='u1')
    dec = codec.decode(buf)
    compare(dec)

    # test decoding directly into numpy array
    out = np.empty_like(arr)
    codec.decode(enc_bytes, out=out)
    compare(out)

    # test decoding directly into bytearray
    out = bytearray(arr.nbytes)
    codec.decode(enc_bytes, out=out)
    compare(out)


def check_config(codec):
    config = codec.get_config()
    # round-trip through JSON to check serialization
    config = json.loads(json.dumps(config))
    eq(codec, get_codec(config))


def check_repr(stmt):
    # check repr matches instantiation statement
    codec = eval(stmt)
    actual = repr(codec)
    eq(stmt, actual)
