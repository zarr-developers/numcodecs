# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import itertools


import numpy as np


from numcodecs.zfp import Zfp
from numcodecs.tests.common import (check_encode_decode_array, check_config, check_repr,
                                    check_backwards_compatibility,
                                    check_err_decode_object_buffer,
                                    check_err_encode_object_buffer)


codecs = [
    Zfp(),
    Zfp(mode='a',tol=0), 
    Zfp(mode='p',prec=64), 
    Zfp(mode='r',rate=60), 
    Zfp(mode='c'),
]


# mix of dtypes: integer, float, bool, string
# mix of shapes: 1D, 2D, 3D
# mix of orders: C, F
arrays = [
    np.linspace(1000, 1001, 1000, dtype='f4'),
    np.linspace(1000, 1001, 1000, dtype='f8'),
    np.random.normal(loc=1000, scale=1, size=(200, 100, 50)),
    np.random.normal(loc=1000, scale=1, size=(200, 100, 50, 10)),
    np.asfortranarray(np.random.normal(loc=1000, scale=1, size=(150, 350, 50))),
]


def test_encode_decode():
    for arr, codec in itertools.product(arrays, codecs):
        check_encode_decode_array(arr, codec)


def test_config():
    for codec in codecs:
        check_config(codec)


def test_repr():
    check_repr("Zfp(mode='a',tol=0)")
    check_repr("Zfp(mode='p',prec=64)")
    check_repr("Zfp(mode='r',rate=0)")


def test_backwards_compatibility():
    check_backwards_compatibility(Zfp.codec_id, arrays, codecs)


def test_err_decode_object_buffer():
    check_err_decode_object_buffer(Zfp())


def test_err_encode_object_buffer():
    check_err_encode_object_buffer(Zfp())
