# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import itertools
import unittest


import numpy as np


try:
    # noinspection PyProtectedMember
    from numcodecs.zfpy import ZFPY, _zfpy
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("ZFPY not available")


from numcodecs.tests.common import (check_encode_decode_array, check_config, check_repr,
                                    check_backwards_compatibility,
                                    check_err_decode_object_buffer,
                                    check_err_encode_object_buffer)


codecs = [
    ZFPY(),
    ZFPY(mode=_zfpy.mode_fixed_accuracy, tolerance=0.1),
    ZFPY(mode=_zfpy.mode_fixed_precision, precision=64),
    ZFPY(mode=_zfpy.mode_fixed_rate, rate=60),
    ZFPY(mode='c'),
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
    check_repr("ZFPY(mode=_zfpy.mode_fixed_accuracy,tolerance=0.1)")
    check_repr("ZFPY(mode=zfpy.mode_fixed_precision,precision=64)")
    check_repr("ZFPY(mode=_zfpy.mode_fixed_rate,rate=0)")


def test_backwards_compatibility():
    check_backwards_compatibility(ZFPY.codec_id, arrays, codecs)


def test_err_decode_object_buffer():
    check_err_decode_object_buffer(ZFPY())


def test_err_encode_object_buffer():
    check_err_encode_object_buffer(ZFPY())
