# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from numcodecs.pickles import Pickle
from numcodecs.tests.common import check_config, check_repr, check_encode_decode_array


# object array with strings
# object array with mix strings / nans
# object array with mix of string, int, float
arrays = [
    np.array(['foo', 'bar', 'baz'] * 300, dtype=object),
    np.array([['foo', 'bar', np.nan]] * 300, dtype=object),
    np.array(['foo', 1.0, 2] * 300, dtype=object),
    np.arange(1000, dtype='i4'),
    np.array(['foo', 'bar', 'baz'] * 300),
]


def test_encode_decode():
    codec = Pickle()
    for arr in arrays:
        check_encode_decode_array(arr, codec)


def test_config():
    codec = Pickle(protocol=-1)
    check_config(codec)


def test_repr():
    check_repr("Pickle(protocol=-1)")
