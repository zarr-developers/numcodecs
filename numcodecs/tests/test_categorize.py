# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import eq_ as eq


from numcodecs.categorize import Categorize
from numcodecs.tests.common import check_encode_decode, check_config, check_backwards_compatibility
from numcodecs.compat import PY2


labels = [b'foo', b'bar', b'baz', b'quux']
labels_u = [u'ƒöõ', u'ßàř', u'ßāẑ', u'ƪùüx']
labels_num = [1000000, 2000000, 3000000]
arrays = [
    np.random.choice(labels, size=1000),
    np.random.choice(labels, size=(100, 10)),
    np.random.choice(labels, size=(10, 10, 10)),
    np.random.choice(labels, size=1000).reshape(100, 10, order='F'),
]
arrays_u = [
    np.random.choice(labels_u, size=1000),
    np.random.choice(labels_u, size=(100, 10)),
    np.random.choice(labels_u, size=(10, 10, 10)),
    np.random.choice(labels_u, size=1000).reshape(100, 10, order='F'),
]
arrays_num = [
    np.random.choice(labels_num, size=1000).astype('i8'),
    np.random.choice(labels_num, size=(100, 10)).astype('i8'),
    np.random.choice(labels_num, size=(10, 10, 10)).astype('i8'),
    np.random.choice(labels_num, size=1000).reshape(100, 10, order='F').astype('i8'),
]


def test_encode_decode():

    # string dtype
    for arr in arrays:
        codec = Categorize(labels, dtype=arr.dtype)
        check_encode_decode(arr, codec)

    # unicode dtype
    for arr in arrays_u:
        codec = Categorize(labels_u, dtype=arr.dtype)
        check_encode_decode(arr, codec)

    # other dtype
    for arr in arrays_num:
        codec = Categorize(labels_num, dtype=arr.dtype)
        check_encode_decode(arr, codec)


def test_encode():
    arr = np.array([b'foo', b'bar', b'foo', b'baz', b'quux'])
    # miss off quux
    codec = Categorize(labels=labels[:-1], dtype=arr.dtype, astype='u1')

    # test encoding
    expect = np.array([1, 2, 1, 3, 0], dtype='u1')
    enc = codec.encode(arr)
    assert_array_equal(expect, enc)
    eq(expect.dtype, enc.dtype)

    # test decoding with unexpected value
    dec = codec.decode(enc)
    expect = arr.copy()
    expect[expect == b'quux'] = b''
    assert_array_equal(expect, dec)
    eq(arr.dtype, dec.dtype)


def test_encode_unicode():
    arr = np.array([u'ƒöõ', u'ßàř', u'ƒöõ', u'ßāẑ', u'ƪùüx'])
    # miss off quux
    codec = Categorize(labels=labels_u[:-1], dtype=arr.dtype, astype='u1')

    # test encoding
    expect = np.array([1, 2, 1, 3, 0], dtype='u1')
    enc = codec.encode(arr)
    assert_array_equal(expect, enc)
    eq(expect.dtype, enc.dtype)

    # test decoding with unexpected value
    dec = codec.decode(enc)
    expect = arr.copy()
    expect[expect == u'ƪùüx'] = u''
    assert_array_equal(expect, dec)
    eq(arr.dtype, dec.dtype)


def test_config():
    codec = Categorize(labels=labels, dtype='S4')
    check_config(codec)
    codec = Categorize(labels=labels_u, dtype='U4')
    check_config(codec)


def test_repr():
    if not PY2:

        dtype = '|S5'
        astype = '|u1'
        codec = Categorize(labels=labels, dtype=dtype, astype=astype)
        expect = "Categorize(dtype='|S5', astype='|u1', " \
                 "labels=[b'foo', b'bar', b'baz', ...])"
        actual = repr(codec)
        eq(expect, actual)

        dtype = '<U5'
        astype = '|u1'
        codec = Categorize(labels=labels_u, dtype=dtype, astype=astype)
        expect = "Categorize(dtype='<U5', astype='|u1', " \
                 "labels=['ƒöõ', 'ßàř', 'ßāẑ', ...])"
        actual = repr(codec)
        eq(expect, actual)


def test_backwards_compatibility():
    codec = Categorize(labels=labels, dtype=arrays[0].dtype, astype='u1')
    check_backwards_compatibility(Categorize.codec_id, arrays, [codec], prefix='s')
    codec = Categorize(labels=labels_u, dtype=arrays_u[0].dtype, astype='u1')
    check_backwards_compatibility(Categorize.codec_id, arrays_u, [codec], prefix='u')
    codec = Categorize(labels=labels_num, dtype=arrays_num[0].dtype, astype='u1')
    check_backwards_compatibility(Categorize.codec_id, arrays_num, [codec], prefix='n')
