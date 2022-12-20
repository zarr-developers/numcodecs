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


def test_error():
    data = np.arange(100)
    f = Fletcher32()
    enc = f.encode(data)
    enc2 = bytearray(enc)
    enc2[0] += 1
    with pytest.raises(ValueError) as e:
        f.decode(enc2)
    assert "fletcher32 checksum" in str(e.value)

