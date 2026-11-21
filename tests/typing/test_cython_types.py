"""Static types of the Cython codecs, as declared in their `.pyi` stubs, checked against
runtime behavior."""

from typing import assert_type

import numpy as np
import numpy.typing as npt

from numcodecs.blosc import Blosc
from numcodecs.fletcher32 import Fletcher32
from numcodecs.jenkins import jenkins_lookup3
from numcodecs.lz4 import LZ4
from numcodecs.vlen import VLenArray, VLenBytes, VLenUTF8
from numcodecs.zstd import Zstd

DATA = np.arange(16, dtype="f8")


def test_compressors() -> None:
    for codec in (Blosc(), Zstd(), LZ4()):
        enc = assert_type(codec.encode(DATA), bytes)
        assert isinstance(enc, bytes)
        dec = assert_type(codec.decode(enc), bytes)
        assert dec == DATA.tobytes()
        out = bytearray(DATA.nbytes)
        assert assert_type(codec.decode(enc, out=out), bytearray) is out


def test_zstd_levels() -> None:
    for level in (Zstd.default_level(), Zstd.min_level(), Zstd.max_level()):
        assert isinstance(assert_type(level, int), int)


def test_fletcher32() -> None:
    codec = Fletcher32()
    enc = assert_type(codec.encode(DATA), bytes)
    dec = assert_type(codec.decode(enc), npt.NDArray[np.uint8])
    assert isinstance(dec, np.ndarray)
    assert dec.tobytes() == DATA.tobytes()


def test_jenkins_lookup3() -> None:
    assert isinstance(assert_type(jenkins_lookup3(b"abc"), int), int)


def test_vlen() -> None:
    utf8 = VLenUTF8()
    enc = assert_type(utf8.encode(["foo", "bar"]), bytearray)
    dec = assert_type(utf8.decode(enc), npt.NDArray[np.object_])
    assert dec.tolist() == ["foo", "bar"]

    raw = VLenBytes()
    assert raw.decode(assert_type(raw.encode([b"foo"]), bytearray)).tolist() == [b"foo"]

    arrays = VLenArray(dtype="i4")
    values = np.empty(2, dtype=object)
    values[0], values[1] = np.arange(3, dtype="i4"), np.arange(2, dtype="i4")
    enc = assert_type(arrays.encode(values), bytearray)
    assert assert_type(arrays.decode(enc), npt.NDArray[np.object_])[0].tolist() == [0, 1, 2]
