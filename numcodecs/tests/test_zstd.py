import itertools
import subprocess

import numpy as np
import pytest

from numcodecs.tests.common import (
    check_backwards_compatibility,
    check_config,
    check_encode_decode,
    check_err_decode_object_buffer,
    check_err_encode_object_buffer,
    check_repr,
)
from numcodecs.zstd import Zstd

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
    # Test input frames with unknown frame content size
    codec = Zstd()

    # If the zstd command line interface is available, check the bytes
    cli = zstd_cli_available()
    if cli:
        view_zstd_streaming_bytes()

    # Encode bytes directly that were the result of streaming compression
    bytes_val = b'(\xb5/\xfd\x00Xa\x00\x00Hello World!'
    dec = codec.decode(bytes_val)
    dec_expected = b'Hello World!'
    assert dec == dec_expected
    if cli:
        assert bytes_val == generate_zstd_streaming_bytes(dec_expected)
        assert dec_expected == generate_zstd_streaming_bytes(bytes_val, decompress=True)

    # Two consecutive frames given as input
    bytes2 = bytes(bytearray(bytes_val * 2))
    dec2 = codec.decode(bytes2)
    dec2_expected = b'Hello World!Hello World!'
    assert dec2 == dec2_expected
    if cli:
        assert dec2_expected == generate_zstd_streaming_bytes(bytes2, decompress=True)

    # Single long frame that decompresses to a large output
    bytes3 = b'(\xb5/\xfd\x00X$\x02\x00\xa4\x03ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz\x01\x00:\xfc\xdfs\x05\x05L\x00\x00\x08s\x01\x00\xfc\xff9\x10\x02L\x00\x00\x08k\x01\x00\xfc\xff9\x10\x02L\x00\x00\x08c\x01\x00\xfc\xff9\x10\x02L\x00\x00\x08[\x01\x00\xfc\xff9\x10\x02L\x00\x00\x08S\x01\x00\xfc\xff9\x10\x02L\x00\x00\x08K\x01\x00\xfc\xff9\x10\x02L\x00\x00\x08C\x01\x00\xfc\xff9\x10\x02L\x00\x00\x08u\x01\x00\xfc\xff9\x10\x02L\x00\x00\x08m\x01\x00\xfc\xff9\x10\x02L\x00\x00\x08e\x01\x00\xfc\xff9\x10\x02L\x00\x00\x08]\x01\x00\xfc\xff9\x10\x02L\x00\x00\x08U\x01\x00\xfc\xff9\x10\x02L\x00\x00\x08M\x01\x00\xfc\xff9\x10\x02M\x00\x00\x08E\x01\x00\xfc\x7f\x1d\x08\x01'
    dec3 = codec.decode(bytes3)
    dec3_expected = b'ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz' * 1024 * 32
    assert dec3 == dec3_expected
    if cli:
        assert bytes3 == generate_zstd_streaming_bytes(dec3_expected)
        assert dec3_expected == generate_zstd_streaming_bytes(bytes3, decompress=True)

    # Garbage input results in an error
    bytes4 = bytes(bytearray([0, 0, 0, 0, 0, 0, 0, 0]))
    with pytest.raises(RuntimeError, match='Zstd decompression error: invalid input data'):
        codec.decode(bytes4)


def generate_zstd_streaming_bytes(input: bytes, *, decompress: bool = False) -> bytes:
    """
    Use the zstd command line interface to compress or decompress bytes in streaming mode.
    """
    if decompress:
        args = ["-d"]
    else:
        args = []

    p = subprocess.run(["zstd", "--no-check", *args], input=input, capture_output=True)
    return p.stdout


def view_zstd_streaming_bytes():
    bytes_val = generate_zstd_streaming_bytes(b"Hello world!")
    print(f"    bytes_val = {bytes_val}")

    bytes3 = generate_zstd_streaming_bytes(
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz" * 1024 * 32
    )
    print(f"    bytes3 = {bytes3}")


def zstd_cli_available() -> bool:
    return not subprocess.run(
        ["zstd", "-V"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode
