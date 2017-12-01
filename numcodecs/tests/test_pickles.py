# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import itertools


import numpy as np


from numcodecs.compat import PY2
from numcodecs.pickles import Pickle
from numcodecs.tests.common import (check_config, check_repr, check_encode_decode_array,
                                    check_backwards_compatibility, greetings)


codecs = [
    Pickle(protocol=0),
    Pickle(protocol=1),
    Pickle(protocol=2),
]
if not PY2:  # pragma: py2 no cover
    codecs.append(Pickle(protocol=3))
    codecs.append(Pickle(protocol=4))


# object array with strings
# object array with mix strings / nans
# object array with mix of string, int, float
# ...
arrays = [
    np.array(['foo', 'bar', 'baz'] * 300, dtype=object),
    np.array([['foo', 'bar', np.nan]] * 300, dtype=object),
    np.array(['foo', 1.0, 2] * 300, dtype=object),
    np.arange(1000, dtype='i4'),
    np.array(['foo', 'bar', 'baz'] * 300),
    np.array(['foo', ['bar', 1.0, 2], {'a': 'b', 'c': 42}] * 300, dtype=object),
    np.array(greetings * 100),
    np.array(greetings * 100, dtype=object),
]


def test_encode_decode():
    for arr, codec in itertools.product(arrays, codecs):
        check_encode_decode_array(arr, codec)


def test_config():
    codec = Pickle(protocol=-1)
    check_config(codec)
    for codec in codecs:
        check_config(codec)


def test_repr():
    check_repr("Pickle(protocol=-1)")


def test_backwards_compatibility():
    check_backwards_compatibility(Pickle.codec_id, arrays, codecs)
