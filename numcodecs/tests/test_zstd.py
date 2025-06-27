import itertools

import numpy as np
import pytest

try:
    from numcodecs.zstd import Zstd
except ImportError:  # pragma: no cover
    pytest.skip("numcodecs.zstd not available", allow_module_level=True)


from numcodecs.tests.common import (
    check_backwards_compatibility,
    check_config,
    check_encode_decode,
    check_err_decode_object_buffer,
    check_err_encode_object_buffer,
    check_repr,
)

codecs = [
    Zstd(),
    Zstd(level=-1),
    Zstd(level=0),
    Zstd(level=1),
    Zstd(level=10),
    Zstd(level=22),
    Zstd(level=100),
    Zstd(checksum=True),
    Zstd(level=0, checksum=True),
    Zstd(level=22, checksum=True),
]


# mix of dtypes: integer, float, bool, string
# mix of shapes: 1D, 2D, 3D
# mix of orders: C, F
arrays = [
    np.arange(1000, dtype="i4"),
    np.linspace(1000, 1001, 1000, dtype="f8"),
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
    for arr, codec in itertools.product(arrays, codecs):
        check_encode_decode(arr, codec)


def test_config():
    for codec in codecs:
        check_config(codec)


def test_repr():
    check_repr("Zstd(level=3)")


def test_backwards_compatibility():
    check_backwards_compatibility(Zstd.codec_id, arrays, codecs)


def test_err_decode_object_buffer():
    check_err_decode_object_buffer(Zstd())


def test_err_encode_object_buffer():
    check_err_encode_object_buffer(Zstd())


def test_checksum():
    data = np.arange(0, 64, dtype="uint8")
    assert len(Zstd(level=0, checksum=False).encode(data)) + 4 == len(
        Zstd(level=0, checksum=True).encode(data)
    )


def test_native_functions():
    # Note, these assertions might need to be changed for new versions of zstd
    assert Zstd.default_level() == 3
    assert Zstd.min_level() == -131072
    assert Zstd.max_level() == 22


def test_streaming_decompression():
    codec = Zstd()
    # Bytes from streaming compression
    bytes_val = bytes(bytearray([
        40, 181, 47, 253, 0, 88, 97, 0, 0, 72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100, 33,
    ]))
    dec = codec.decode(bytes_val)
    assert dec == b'Hello World!'

    bytes2 = bytes(bytearray([
        40, 181, 47, 253, 0, 88, 36, 2, 0, 164, 3, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
        83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
        109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 1, 0, 58, 252, 223, 115, 5, 5, 76, 0, 0, 8,
        115, 1, 0, 252, 255, 57, 16, 2, 76, 0, 0, 8, 107, 1, 0, 252, 255, 57, 16, 2, 76, 0, 0, 8, 99, 1, 0, 252, 255, 57,
        16, 2, 76, 0, 0, 8, 91, 1, 0, 252, 255, 57, 16, 2, 76, 0, 0, 8, 83, 1, 0, 252, 255, 57, 16, 2, 76, 0, 0, 8, 75, 1,
        0, 252, 255, 57, 16, 2, 76, 0, 0, 8, 67, 1, 0, 252, 255, 57, 16, 2, 76, 0, 0, 8, 117, 1, 0, 252, 255, 57, 16, 2, 76,
        0, 0, 8, 109, 1, 0, 252, 255, 57, 16, 2, 76, 0, 0, 8, 101, 1, 0, 252, 255, 57, 16, 2, 76, 0, 0, 8, 93, 1, 0, 252,
        255, 57, 16, 2, 76, 0, 0, 8, 85, 1, 0, 252, 255, 57, 16, 2, 76, 0, 0, 8, 77, 1, 0, 252, 255, 57, 16, 2, 77, 0, 0, 8,
        69, 1, 0, 252, 127, 29, 8, 1,
    ]))
    dec2 = codec.decode(bytes2)
    assert dec2 == b'ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz' * 1024 * 32
