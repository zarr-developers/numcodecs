# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np
from numpy.testing import assert_array_equal
import pytest


from numcodecs.categorize import Categorize
from numcodecs.tests.common import (check_encode_decode, check_config,
                                    check_backwards_compatibility, compare_arrays)
from numcodecs.compat import PY2


labels = [u'ƒöõ', u'ßàř', u'ßāẑ', u'ƪùüx']
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
        check_encode_decode(arr, codec)

    # object dtype
    for arr in arrays_object:
        codec = Categorize(labels, dtype=arr.dtype)
        # can't do the full check because can only accept object array for
        # encoding
        enc = codec.encode(arr)
        dec = codec.decode(enc)
        compare_arrays(arr, dec)


def test_encode():
    for dtype in 'U', object:

        arr = np.array([u'ƒöõ', u'ßàř', u'ƒöõ', u'ßāẑ', u'ƪùüx'], dtype=dtype)
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
        expect[expect == u'ƪùüx'] = u''
        assert_array_equal(expect, dec)
        assert arr.dtype == dec.dtype


def test_config():
    codec = Categorize(labels=labels, dtype='U4')
    check_config(codec)


def test_repr():
    dtype = '<U3'
    astype = '|u1'
    codec = Categorize(labels=['foo', 'bar', 'baz', 'qux'],
                       dtype=dtype, astype=astype)
    expect = "Categorize(dtype='<U3', astype='|u1', "
    if PY2:  # pragma: py3 no cover
        expect += "labels=[u'foo', u'bar', u'baz', ...])"
    else:  # pragma: py2 no cover
        expect += "labels=['foo', 'bar', 'baz', ...])"
    actual = repr(codec)
    assert expect == actual

    if not PY2:  # pragma: py2 no cover

        dtype = '<U4'
        astype = '|u1'
        codec = Categorize(labels=labels, dtype=dtype, astype=astype)
        expect = "Categorize(dtype='<U4', astype='|u1', " \
                 "labels=['ƒöõ', 'ßàř', 'ßāẑ', ...])"
        actual = repr(codec)
        assert expect == actual


def test_backwards_compatibility():
    codec = Categorize(labels=labels, dtype='<U4', astype='u1')
    check_backwards_compatibility(Categorize.codec_id, arrays, [codec],
                                  prefix='U')
    codec = Categorize(labels=labels, dtype=object, astype='u1')
    check_backwards_compatibility(Categorize.codec_id, arrays_object,
                                  [codec], prefix='O')


def test_errors():
    with pytest.raises(ValueError):
        Categorize(labels=['foo', 'bar'], dtype='S6')
