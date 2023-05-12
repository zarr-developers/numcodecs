import sys

import numcodecs
import numpy as np
from numpy.testing import assert_array_equal


codec = numcodecs.Blosc()
data = np.arange(int(sys.argv[1]))
for i in range(int(sys.argv[2])):
    enc = codec.encode(data)
    dec = codec.decode(enc)
    arr = np.frombuffer(dec, dtype=data.dtype)
    assert_array_equal(data, arr)
