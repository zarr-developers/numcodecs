import numpy as np
import pytest
from numcodecs.fletcher32 import Fletcher32

from numcodecs.compat import ensure_contiguous_ndarray


@pytest.mark.parametrize("dtype", ["uint8", "int32", "float32"])
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
    with pytest.raises(RuntimeError) as e:
        f.decode(enc2)
    assert "fletcher32 checksum" in str(e.value)


def test_known():
    data = (
        b'w\x07\x00\x00\x00\x00\x00\x00\x85\xf6\xff\xff\xff\xff\xff\xff'
        b'i\x07\x00\x00\x00\x00\x00\x00\x94\xf6\xff\xff\xff\xff\xff\xff'
        b'\x88\t\x00\x00\x00\x00\x00\x00i\x03\x00\x00\x00\x00\x00\x00'
        b'\x93\xfd\xff\xff\xff\xff\xff\xff\xc3\xfc\xff\xff\xff\xff\xff\xff'
        b"'\x02\x00\x00\x00\x00\x00\x00\xba\xf7\xff\xff\xff\xff\xff\xff"
        b'\xfd%\x86d'
    )
    data3 = Fletcher32().decode(data)
    outarr = np.frombuffer(data3, dtype="<i8")
    expected = [
        1911,
        -2427,
        1897,
        -2412,
        2440,
        873,
        -621,
        -829,
        551,
        -2118,
    ]
    assert outarr.tolist() == expected


def test_out():
    data = np.frombuffer(bytearray(b"Hello World"), dtype="uint8")
    f = Fletcher32()
    result = f.encode(data)
    f.decode(result, out=data)


@pytest.mark.parametrize(
    "make_out",
    [
        lambda: np.empty(16, dtype="uint8"),
        lambda: np.empty((2, 2), dtype="uint32", order="F"),
        lambda: np.empty(2, dtype="float64"),
        lambda: bytearray(16),
        lambda: memoryview(bytearray(16)),
    ],
    ids=["uint8", "uint32-F-order", "float64", "bytearray", "memoryview"],
)
def test_decode_into_out(make_out):
    data = np.arange(16, dtype="uint8")
    f = Fletcher32()
    out = make_out()
    assert f.decode(f.encode(data), out=out) is out
    assert ensure_contiguous_ndarray(out).view("uint8").tobytes() == data.tobytes()


@pytest.mark.parametrize("shape", [(15,), (17,), ()], ids=["smaller", "larger", "0-d"])
def test_decode_rejects_out_of_wrong_size(shape):
    data = np.arange(16, dtype="uint8")
    f = Fletcher32()
    out = np.full(shape, 0xAA, dtype="uint8")
    with pytest.raises(ValueError, match=f"out has {out.size} bytes, but the decoded data has 16"):
        f.decode(f.encode(data), out=out)
    assert (out == 0xAA).all()
