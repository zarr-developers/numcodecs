# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import sys


import numcodecs as codecs
from numcodecs import blosc
import numpy as np
from numpy.testing import assert_array_equal


codec = codecs.Blosc()
data = np.arange(int(sys.argv[1]))
for i in range(int(sys.argv[2])):
    enc = codec.encode(data)
    dec = codec.decode(enc)
    arr = np.frombuffer(dec, dtype=data.dtype)
    assert_array_equal(data, arr)
