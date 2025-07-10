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
    z = Zstd()
    assert z.decode(pyzstd.compress(input)) == input
    assert pyzstd.decompress(z.encode(input)) == input


@pytest.mark.xfail
@pytest.mark.parametrize("input", test_data)
def test_pyzstd_simple_multiple_frames_decode(input):
    z = Zstd()
    assert z.decode(pyzstd.compress(input) * 2) == input * 2


@pytest.mark.parametrize("input", test_data)
def test_pyzstd_simple_multiple_frames_encode(input):
    z = Zstd()
    assert pyzstd.decompress(z.encode(input) * 2) == input * 2


@pytest.mark.parametrize("input", test_data)
def test_pyzstd_streaming(input):
    pyzstd_c = pyzstd.ZstdCompressor()
    pyzstd_d = pyzstd.ZstdDecompressor()
    z = Zstd()

    d_bytes = input
    pyzstd_c.compress(d_bytes)
    c_bytes = pyzstd_c.flush()
    assert z.decode(c_bytes) == d_bytes
    assert pyzstd_d.decompress(z.encode(d_bytes)) == d_bytes

    # Test multiple streaming frames
    assert z.decode(c_bytes * 2) == d_bytes * 2
    assert z.decode(c_bytes * 3) == d_bytes * 3
    assert z.decode(c_bytes * 99) == d_bytes * 99
