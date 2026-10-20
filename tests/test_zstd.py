import itertools
import struct
import subprocess

import numpy as np
import pytest
from numcodecs.zstd import Zstd

from numcodecs import zstd
from tests.common import (
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


def test_multi_frame():
    codec = Zstd()

    hello_world = codec.encode(b"Hello world!")
    assert codec.decode(hello_world) == b"Hello world!"
    assert codec.decode(hello_world * 2) == b"Hello world!Hello world!"

    hola = codec.encode(b"Hola ")
    mundo = codec.encode(b"Mundo!")
    assert codec.decode(hola) == b"Hola "
    assert codec.decode(mundo) == b"Mundo!"
    assert codec.decode(hola + mundo) == b"Hola Mundo!"

    bytes_val = b'(\xb5/\xfd\x00Xa\x00\x00Hello World!'
    dec = codec.decode(bytes_val)
    dec_expected = b'Hello World!'
    assert dec == dec_expected
    cli = zstd_cli_available()
    if cli:
        assert bytes_val == generate_zstd_streaming_bytes(dec_expected)
        assert dec_expected == generate_zstd_streaming_bytes(bytes_val, decompress=True)

    # Concatenate frames of known sizes and unknown sizes
    # unknown size frame at the end
    assert codec.decode(hola + mundo + bytes_val) == b"Hola Mundo!Hello World!"
    # unknown size frame at the beginning
    assert codec.decode(bytes_val + hola + mundo) == b"Hello World!Hola Mundo!"
    # unknown size frame in the middle
    assert codec.decode(hola + bytes_val + mundo) == b"Hola Hello World!Mundo!"


def test_empty_frame():
    codec = Zstd()
    encoded = codec.encode(b"")
    assert codec.decode(encoded) == b""
    # decode into an empty destination buffer
    out = bytearray()
    codec.decode(encoded, out=out)
    assert out == b""
    # an empty frame can be concatenated with other frames
    assert codec.decode(encoded + codec.encode(b"abc") + encoded) == b"abc"


def test_decode_empty_input():
    with pytest.raises(RuntimeError, match='Zstd decompression error: invalid input data'):
        Zstd().decode(b"")


def test_truncated_input():
    codec = Zstd()
    known = codec.encode(b"Hello world!" * 100)
    unknown = b'(\xb5/\xfd\x00Xa\x00\x00Hello World!'
    for data in (known[:-3], unknown[:-3], known + known[:-3]):
        with pytest.raises(RuntimeError, match='invalid input data'):
            codec.decode(data)


def test_error_messages_are_text():
    # error names from zstd must be formatted as text, not as the repr of bytes
    codec = Zstd()
    encoded = bytearray(Zstd(checksum=True).encode(b"abc" * 10))
    encoded[-1] ^= 0xFF
    with pytest.raises(RuntimeError) as excinfo:
        codec.decode(bytes(encoded))
    assert "b'" not in str(excinfo.value)
    assert "b\"" not in str(excinfo.value)
    assert "checksum" in str(excinfo.value)


def test_decode_unknown_size_into_out():
    codec = Zstd()
    unknown = b'(\xb5/\xfd\x00Xa\x00\x00Hello World!'
    out = bytearray(12)
    codec.decode(unknown, out=out)
    assert out == b"Hello World!"


def test_repr_checksum():
    check_repr("Zstd(level=3)")
    check_repr("Zstd(level=3, checksum=True)")


def test_stream_decompress_growth():
    # Highly compressible data with unknown content size requires the output
    # buffer to grow many times.
    codec = Zstd()
    frame = b'(\xb5/\xfd\x00X$\x02\x00\xa4\x03ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz\x01\x00:\xfc\xdfs\x05\x05L\x00\x00\x08s\x01\x00\xfc\xff9\x10\x02L\x00\x00\x08k\x01\x00\xfc\xff9\x10\x02L\x00\x00\x08c\x01\x00\xfc\xff9\x10\x02L\x00\x00\x08[\x01\x00\xfc\xff9\x10\x02L\x00\x00\x08S\x01\x00\xfc\xff9\x10\x02L\x00\x00\x08K\x01\x00\xfc\xff9\x10\x02L\x00\x00\x08C\x01\x00\xfc\xff9\x10\x02L\x00\x00\x08u\x01\x00\xfc\xff9\x10\x02L\x00\x00\x08m\x01\x00\xfc\xff9\x10\x02L\x00\x00\x08e\x01\x00\xfc\xff9\x10\x02L\x00\x00\x08]\x01\x00\xfc\xff9\x10\x02L\x00\x00\x08U\x01\x00\xfc\xff9\x10\x02L\x00\x00\x08M\x01\x00\xfc\xff9\x10\x02M\x00\x00\x08E\x01\x00\xfc\x7f\x1d\x08\x01'
    expected = b'ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz' * 1024 * 32
    assert codec.decode(frame * 3) == expected * 3


def _frame_with_content_size(n):
    """A minimal single-segment frame whose header claims ``n`` bytes of content."""
    return b'\x28\xb5\x2f\xfd' + bytes([0xE0]) + struct.pack('<Q', n) + b'\x01\x00\x00'


def test_empty_frame_module_functions():
    for checksum in (False, True):
        for level in (-1, 0, 3, 22):
            encoded = zstd.compress(b"", level, checksum)
            assert zstd.decompress(encoded) == b""
            assert zstd.decompress(encoded * 3) == b""


def test_header_content_size_too_large():
    # content sizes beyond what a Python buffer can hold must raise a clean
    # RuntimeError rather than SystemError/MemoryError
    for n in (2**63, 2**64 - 3):
        with pytest.raises(RuntimeError, match='content size too large'):
            zstd.decompress(_frame_with_content_size(n))
    # sentinel value used by zstd for errors
    with pytest.raises(RuntimeError, match='invalid input data'):
        zstd.decompress(_frame_with_content_size(2**64 - 2))


def test_total_content_size_overflow():
    frame = _frame_with_content_size(2**63)
    with pytest.raises(OverflowError, match='total content size too large'):
        zstd.decompress(frame * 2)


def test_header_content_size_mismatch():
    # header claims 5 bytes, but the frame actually holds none
    with pytest.raises(RuntimeError, match='Zstd decompression error'):
        zstd.decompress(_frame_with_content_size(5))


def test_dest_too_small():
    encoded = zstd.compress(b"abc" * 10)
    with pytest.raises(ValueError, match='destination buffer too small'):
        zstd.decompress(encoded, np.zeros(3, dtype='u1'))
    # exact and oversized destinations are fine for known content sizes
    out = np.zeros(30, dtype='u1')
    zstd.decompress(encoded, out)
    assert out.tobytes() == b"abc" * 10


def test_unknown_size_dest_must_match():
    unknown = b'(\xb5/\xfd\x00Xa\x00\x00Hello World!'
    with pytest.raises(RuntimeError, match='expected to decompress'):
        zstd.decompress(unknown, bytearray(100))
    with pytest.raises(RuntimeError, match='Zstd decompression error'):
        zstd.decompress(unknown, bytearray(3))


def test_garbage_input_messages():
    for data in (b"\0" * 8, b"not zstd data", b"\x28\xb5\x2f"):
        with pytest.raises(RuntimeError, match='Zstd decompression error: invalid input data'):
            zstd.decompress(data)


@pytest.mark.parametrize(
    "make",
    [bytes, bytearray, memoryview, lambda b: np.frombuffer(b, dtype='u1')],
)
def test_buffer_types(make):
    data = b"Hello world!" * 50
    encoded = zstd.compress(make(data))
    assert zstd.decompress(make(encoded)) == data


def test_compress_level_clamped():
    data = b"abc" * 100
    assert zstd.decompress(zstd.compress(data, 10**6)) == data
    assert zstd.decompress(zstd.compress(data, -(10**6))) == data


def test_compress_bad_level_type():
    with pytest.raises(TypeError):
        zstd.compress(b"abc", level="high")


def test_repeated_failures_do_not_crash():
    # exercise the error paths many times to surface leaks or double frees
    unknown = b'(\xb5/\xfd\x00Xa\x00\x00Hello World!'
    for _ in range(2000):
        with pytest.raises(RuntimeError):
            zstd.decompress(unknown[:-3])
        with pytest.raises(RuntimeError):
            zstd.decompress(b"\0" * 8)
        assert zstd.decompress(unknown) == b"Hello World!"
