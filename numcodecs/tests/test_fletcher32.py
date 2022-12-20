import numpy as np
import pytest

from numcodecs.fletcher32 import Fletcher32, fletcher32


@pytest.mark.parametrize("inval,outval", [
    [b"abcdef", 1448095018],
    [b"abcdefgh", 3957429649]
])
def test_vectors(inval, outval):
    arr = np.array(list(inval), dtype="uint8").view('uint16')
    assert fletcher32(arr) == outval


@pytest.mark.parametrize(
    "dtype",
    ["uint8", "int32", "float32"]
)
def test_with_data(dtype):
    data = np.empty(100, dtype=dtype)
    f = Fletcher32()
    arr = np.frombuffer(f.decode(f.encode(data)), dtype=dtype)
    assert (arr == data).all()
