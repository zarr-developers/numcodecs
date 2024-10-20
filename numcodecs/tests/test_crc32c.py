import numpy as np
import pytest

from numcodecs.crc32c_ import Crc32c
from numcodecs.tests.common import (
    check_encode_decode,
    check_err_decode_object_buffer,
    check_err_encode_object_buffer,
    check_repr,
)

# mix of dtypes: integer, float, bool, string
# mix of shapes: 1D, 2D, 3D
# mix of orders: C, F
arrays = [
    np.arange(1000, dtype='i4'),
    np.linspace(1000, 1001, 1000, dtype='f8'),
    np.random.normal(loc=1000, scale=1, size=(100, 10)),
    np.random.randint(0, 2, size=1000, dtype=bool).reshape(100, 10, order='F'),
    np.random.choice([b'a', b'bb', b'ccc'], size=1000).reshape(10, 10, 10),
    np.random.randint(0, 2**60, size=1000, dtype='u8').view('M8[ns]'),
    np.random.randint(0, 2**60, size=1000, dtype='u8').view('m8[ns]'),
    np.random.randint(0, 2**25, size=1000, dtype='u8').view('M8[m]'),
    np.random.randint(0, 2**25, size=1000, dtype='u8').view('m8[m]'),
    np.random.randint(-(2**63), -(2**63) + 20, size=1000, dtype='i8').view('M8[ns]'),
    np.random.randint(-(2**63), -(2**63) + 20, size=1000, dtype='i8').view('m8[ns]'),
    np.random.randint(-(2**63), -(2**63) + 20, size=1000, dtype='i8').view('M8[m]'),
    np.random.randint(-(2**63), -(2**63) + 20, size=1000, dtype='i8').view('m8[m]'),
]


def test_encode_decode():
    for arr in arrays:
        check_encode_decode(arr, Crc32c())


def test_checksum():
    arr = np.arange(0, 64, dtype="uint8")
    buf = Crc32c().encode(arr)
    assert np.frombuffer(buf, dtype="<u4", offset=(len(buf) - 4))[0] == np.uint32(4218238699)


def test_repr():
    check_repr("Crc32c()")


def test_eq():
    assert Crc32c() == Crc32c()
    assert not Crc32c() != Crc32c()
    assert Crc32c() != 'foo'
    assert 'foo' != Crc32c()
    assert not Crc32c() == 'foo'


def test_err_decode_object_buffer():
    check_err_decode_object_buffer(Crc32c())


def test_err_encode_object_buffer():
    check_err_encode_object_buffer(Crc32c())


def test_err_encode_list():
    data = ['foo', 'bar', 'baz']
    with pytest.raises(TypeError):
        Crc32c().encode(data)


def test_err_encode_non_contiguous():
    # non-contiguous memory
    arr = np.arange(1000, dtype='i4')[::2]
    with pytest.raises(ValueError):
        Crc32c().encode(arr)


def test_err_out_too_small():
    arr = np.arange(10, dtype='i4')
    out = np.empty_like(arr)[:-1]
    with pytest.raises(ValueError):
        Crc32c().decode(Crc32c().encode(arr), out)


def test_err_out_too_large():
    out = np.empty((10,), dtype='i4')
    arr = out[:-1]
    arr[:] = 5
    with pytest.raises(ValueError):
        Crc32c().decode(Crc32c().encode(arr), out)
