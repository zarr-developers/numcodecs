# Check Zstd against pyzstd package

import numpy as np
import pytest
import pyzstd

from numcodecs.zstd import Zstd

test_data = [
    b"Hello World!",
    np.arange(113).tobytes(),
    np.arange(10, 15).tobytes(),
    np.random.randint(3, 50, size=(53,), dtype=np.uint16).tobytes(),
]


@pytest.mark.parametrize("input", test_data)
def test_pyzstd_simple(input):
    """
    Test if Zstd.[decode, encode] can perform the inverse operation to
    pyzstd.[compress, decompress] in the simple case.
    """
    z = Zstd()
    assert z.decode(pyzstd.compress(input)) == input
    assert pyzstd.decompress(z.encode(input)) == input


@pytest.mark.xfail
@pytest.mark.parametrize("input", test_data)
def test_pyzstd_simple_multiple_frames_decode(input):
    """
    Test decompression of two concatenated frames of known sizes

    numcodecs.zstd.Zstd currently fails because it only assesses the size of the
    first frame. Rather, it should keep iterating through all the frames until
    the end of the input buffer.
    """
    z = Zstd()
    assert pyzstd.decompress(pyzstd.compress(input) * 2) == input * 2
    assert z.decode(pyzstd.compress(input) * 2) == input * 2


@pytest.mark.parametrize("input", test_data)
def test_pyzstd_simple_multiple_frames_encode(input):
    """
    Test if pyzstd can decompress two concatenated frames from Zstd.encode
    """
    z = Zstd()
    assert pyzstd.decompress(z.encode(input) * 2) == input * 2


@pytest.mark.parametrize("input", test_data)
def test_pyzstd_streaming(input):
    """
    Test if Zstd can decode a single frame and concatenated frames in streaming
    mode where the decompressed size is not recorded in the frame header.
    """
    pyzstd_c = pyzstd.ZstdCompressor()
    pyzstd_d = pyzstd.ZstdDecompressor()
    pyzstd_e = pyzstd.EndlessZstdDecompressor()
    z = Zstd()

    d_bytes = input
    pyzstd_c.compress(d_bytes)
    c_bytes = pyzstd_c.flush()
    assert z.decode(c_bytes) == d_bytes
    assert pyzstd_d.decompress(z.encode(d_bytes)) == d_bytes

    # Test multiple streaming frames
    assert z.decode(c_bytes * 2) == pyzstd_e.decompress(c_bytes * 2)
    assert z.decode(c_bytes * 3) == pyzstd_e.decompress(c_bytes * 3)
    assert z.decode(c_bytes * 4) == pyzstd_e.decompress(c_bytes * 4)
    assert z.decode(c_bytes * 5) == pyzstd_e.decompress(c_bytes * 5)
    assert z.decode(c_bytes * 7) == pyzstd_e.decompress(c_bytes * 7)
    assert z.decode(c_bytes * 11) == pyzstd_e.decompress(c_bytes * 11)
    assert z.decode(c_bytes * 13) == pyzstd_e.decompress(c_bytes * 13)
    assert z.decode(c_bytes * 99) == pyzstd_e.decompress(c_bytes * 99)
