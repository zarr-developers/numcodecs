import itertools
from contextlib import suppress

import numpy as np
import pytest

from numcodecs.checksum32 import CRC32, Adler32
from numcodecs.tests.common import (
    check_backwards_compatibility,
    check_config,
    check_encode_decode,
    check_err_decode_object_buffer,
    check_err_encode_object_buffer,
    check_repr,
)

has_crc32c = False
with suppress(ImportError):
    from numcodecs.checksum32 import CRC32C

    has_crc32c = True

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

codecs = [
    CRC32(),
    CRC32(location="end"),
    Adler32(),
    Adler32(location="end"),
]
if has_crc32c:
    codecs.extend(
        [
            CRC32C(location="start"),
            CRC32C(),
        ]
    )


@pytest.mark.parametrize(("codec", "arr"), itertools.product(codecs, arrays))
def test_encode_decode(codec, arr):
    check_encode_decode(arr, codec)


@pytest.mark.parametrize(("codec", "arr"), itertools.product(codecs, arrays))
def test_errors(codec, arr):
    enc = codec.encode(arr)
    with pytest.raises(RuntimeError):
        codec.decode(enc[:-1])


@pytest.mark.parametrize("codec", codecs)
def test_config(codec):
    check_config(codec)


@pytest.mark.parametrize("codec", codecs)
def test_err_input_too_small(codec):
    buf = b'000'  # 3 bytes are too little for a 32-bit checksum
    with pytest.raises(ValueError):
        codec.decode(buf)


@pytest.mark.parametrize("codec", codecs)
def test_err_encode_non_contiguous(codec):
    # non-contiguous memory
    arr = np.arange(1000, dtype='i4')[::2]
    with pytest.raises(ValueError):
        codec.encode(arr)


@pytest.mark.parametrize("codec", codecs)
def test_err_encode_list(codec):
    data = ['foo', 'bar', 'baz']
    with pytest.raises(TypeError):
        codec.encode(data)


def test_err_location():
    with pytest.raises(ValueError):
        CRC32(location="foo")
    with pytest.raises(ValueError):
        Adler32(location="foo")
    if has_crc32c:
        with pytest.raises(ValueError):
            CRC32C(location="foo")


def test_repr():
    check_repr("CRC32(location='start')")
    check_repr("CRC32(location='end')")
    check_repr("Adler32(location='start')")
    check_repr("Adler32(location='end')")
    if has_crc32c:
        check_repr("CRC32C(location='start')")
        check_repr("CRC32C(location='end')")


def test_backwards_compatibility():
    check_backwards_compatibility(CRC32.codec_id, arrays, [CRC32()])
    check_backwards_compatibility(Adler32.codec_id, arrays, [Adler32()])
    if has_crc32c:
        check_backwards_compatibility(CRC32C.codec_id, arrays, [CRC32C()])


@pytest.mark.parametrize("codec", codecs)
def test_err_encode_object_buffer(codec):
    check_err_encode_object_buffer(codec)


@pytest.mark.parametrize("codec", codecs)
def test_err_decode_object_buffer(codec):
    check_err_decode_object_buffer(codec)


@pytest.mark.parametrize("codec", codecs)
def test_err_out_too_small(codec):
    arr = np.arange(10, dtype='i4')
    out = np.empty_like(arr)[:-1]
    with pytest.raises(ValueError):
        codec.decode(codec.encode(arr), out)


@pytest.mark.skipif(not has_crc32c, reason="Needs `crc32c` installed")
def test_crc32c_checksum():
    arr = np.arange(0, 64, dtype="uint8")
    buf = CRC32C(location="end").encode(arr)
    assert np.frombuffer(buf, dtype="<u4", offset=(len(buf) - 4))[0] == np.uint32(4218238699)


@pytest.mark.parametrize("codec", codecs)
def test_err_checksum(codec):
    arr = np.arange(0, 64, dtype="uint8")
    buf = bytearray(codec.encode(arr))
    buf[-1] = 0  # corrupt the checksum
    with pytest.raises(RuntimeError):
        codec.decode(buf)
