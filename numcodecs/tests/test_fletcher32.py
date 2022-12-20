import numpy as np
import pytest

from numcodecs.fletcher32 import Fletcher32


@pytest.mark.parametrize(
    "dtype",
    ["uint8", "int32", "float32"]
)
def test_with_data(dtype):
    data = np.arange(100, dtype=dtype)
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


def test_known():
    data = (
        b'\xf04\xfe\x1a\x03\xb2\xb1?^\x99j\xf3\xd6f\xef?\xbbm\x04n'
        b'\x9a\xdf\xeb?x\x9eIL\xdeW\xc8?A\xef\x88\xa8&\xad\xef?'
        b'\xf2\xc6a\x01a\xb8\xe8?#&\x96\xabY\xf2\xe7?\xe2Pw\xba\xd0w\xea?'
        b'\x80\xc5\xf8M@0\x9a?\x98H+\xb4\x03\xfa\xc6?\xb9P\x1e1'
    )
    data3 = Fletcher32().decode(data)
    outarr = np.frombuffer(data3, dtype="<f8")
    expected = [
        0.0691225, 0.98130367, 0.87104532, 0.19018153, 0.9898866,
        0.77250719, 0.74833377, 0.8271259, 0.02557469, 0.17950484
    ]
    assert np.allclose(outarr, expected)
