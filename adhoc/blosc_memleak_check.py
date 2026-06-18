import sys

import numpy as np
from numpy.testing import assert_array_equal

import numcodecs

codec = numcodecs.Blosc()
data = np.arange(int(sys.argv[1]))
for _ in range(int(sys.argv[2])):
    enc = codec.encode(data)
    dec = codec.decode(enc)
    arr = np.frombuffer(dec, dtype=data.dtype)
    assert_array_equal(data, arr)
