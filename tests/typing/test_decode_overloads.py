"""Static checks of the `decode` overloads of every codec.

Each codec overloads `decode` twice: without `out` it returns what the codec decodes to,
and with `out` it returns the type of `out`. `static_checks` is never called; it exists to
be type-checked by mypy and pyright, which verify every `assert_type` and report any
`ignore` comment that stops being needed. It must not run: some of its calls pass an
immutable `bytes` object as `out`.

The file is repetitive on purpose. A typing mistake in one codec's overloads (or in one
stub) is only caught by a line that names that codec.

Known limits, not checked here:

- With an `out` typed `Any`, mypy infers `Any` but pyright picks the first overload and
  infers the no-`out` type.
- JSON, MsgPack, Pickle, Categorize, the VLen codecs and PCodec need an ndarray `out` and
  raise for other buffers, which the inherited signature cannot express.
"""

# `memoryview[float]` in the signature below cannot be evaluated before Python 3.14
from __future__ import annotations

from typing import TYPE_CHECKING, Any, assert_type

import numpy as np
import numpy.typing as npt

from numcodecs import blosc, lz4, zstd
from numcodecs.astype import AsType
from numcodecs.base64 import Base64
from numcodecs.bitround import BitRound
from numcodecs.blosc import Blosc
from numcodecs.bz2 import BZ2
from numcodecs.categorize import Categorize
from numcodecs.checksum32 import CRC32, Adler32, JenkinsLookup3
from numcodecs.compat import ndarray_copy
from numcodecs.delta import Delta
from numcodecs.fixedscaleoffset import FixedScaleOffset
from numcodecs.fletcher32 import Fletcher32
from numcodecs.gzip import GZip
from numcodecs.json import JSON
from numcodecs.lz4 import LZ4
from numcodecs.lzma import LZMA
from numcodecs.ndarray_like import NDArrayLike
from numcodecs.packbits import PackBits
from numcodecs.pickles import Pickle
from numcodecs.quantize import Quantize
from numcodecs.registry import codec_registry
from numcodecs.shuffle import Shuffle
from numcodecs.vlen import VLenArray, VLenBytes, VLenUTF8
from numcodecs.zlib import Zlib
from numcodecs.zstd import Zstd

if TYPE_CHECKING:
    # These need optional dependencies at runtime. `static_checks` is never run, so the
    # names are only needed by the type checkers.
    from numcodecs.msgpacks import MsgPack
    from numcodecs.pcodec import PCodec
    from numcodecs.zfpy import ZFPY

# The codec ids checked in `static_checks`; test_every_codec_is_checked keeps this
# complete. CRC32C shares its `decode` with CRC32 and is only defined when a crc32c
# package is installed, so it is listed here without lines of its own.
CHECKED_IDS = frozenset(
    {
        "zlib",
        "gzip",
        "bz2",
        "lzma",
        "base64",
        "blosc",
        "zstd",
        "lz4",
        "shuffle",
        "crc32",
        "adler32",
        "crc32c",
        "jenkins_lookup3",
        "fletcher32",
        "astype",
        "delta",
        "quantize",
        "fixedscaleoffset",
        "bitround",
        "packbits",
        "categorize",
        "json2",
        "msgpack2",
        "pickle",
        "vlen-utf8",
        "vlen-bytes",
        "vlen-array",
        "zfpy",
        "pcodec",
    }
)


def static_checks(
    enc: bytes,
    f8: npt.NDArray[np.float64],
    nd: npt.NDArray[Any],
    ba: bytearray,
    maybe: bytearray | None,
    mv: memoryview[float],
) -> None:
    # Zlib
    assert_type(Zlib().decode(enc), bytes)
    assert_type(Zlib().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(Zlib().decode(enc, nd), npt.NDArray[Any])
    assert_type(Zlib().decode(enc, out=ba), bytearray)
    assert_type(Zlib().decode(enc, mv), memoryview[float])
    assert_type(Zlib().decode(enc, maybe), bytes | bytearray)
    Zlib().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # GZip
    assert_type(GZip().decode(enc), bytes)
    assert_type(GZip().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(GZip().decode(enc, nd), npt.NDArray[Any])
    assert_type(GZip().decode(enc, out=ba), bytearray)
    assert_type(GZip().decode(enc, mv), memoryview[float])
    assert_type(GZip().decode(enc, maybe), bytes | bytearray)
    GZip().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # BZ2
    assert_type(BZ2().decode(enc), bytes)
    assert_type(BZ2().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(BZ2().decode(enc, nd), npt.NDArray[Any])
    assert_type(BZ2().decode(enc, out=ba), bytearray)
    assert_type(BZ2().decode(enc, mv), memoryview[float])
    assert_type(BZ2().decode(enc, maybe), bytes | bytearray)
    BZ2().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # LZMA
    assert_type(LZMA().decode(enc), bytes)
    assert_type(LZMA().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(LZMA().decode(enc, nd), npt.NDArray[Any])
    assert_type(LZMA().decode(enc, out=ba), bytearray)
    assert_type(LZMA().decode(enc, mv), memoryview[float])
    assert_type(LZMA().decode(enc, maybe), bytes | bytearray)
    LZMA().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # Base64
    assert_type(Base64().decode(enc), bytes)
    assert_type(Base64().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(Base64().decode(enc, nd), npt.NDArray[Any])
    assert_type(Base64().decode(enc, out=ba), bytearray)
    assert_type(Base64().decode(enc, mv), memoryview[float])
    assert_type(Base64().decode(enc, maybe), bytes | bytearray)
    Base64().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # Blosc
    assert_type(Blosc().decode(enc), bytes)
    assert_type(Blosc().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(Blosc().decode(enc, nd), npt.NDArray[Any])
    assert_type(Blosc().decode(enc, out=ba), bytearray)
    assert_type(Blosc().decode(enc, mv), memoryview[float])
    assert_type(Blosc().decode(enc, maybe), bytes | bytearray)
    Blosc().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # Zstd
    assert_type(Zstd().decode(enc), bytes)
    assert_type(Zstd().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(Zstd().decode(enc, nd), npt.NDArray[Any])
    assert_type(Zstd().decode(enc, out=ba), bytearray)
    assert_type(Zstd().decode(enc, mv), memoryview[float])
    assert_type(Zstd().decode(enc, maybe), bytes | bytearray)
    Zstd().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # LZ4
    assert_type(LZ4().decode(enc), bytes)
    assert_type(LZ4().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(LZ4().decode(enc, nd), npt.NDArray[Any])
    assert_type(LZ4().decode(enc, out=ba), bytearray)
    assert_type(LZ4().decode(enc, mv), memoryview[float])
    assert_type(LZ4().decode(enc, maybe), bytes | bytearray)
    LZ4().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # Shuffle
    assert_type(Shuffle().decode(enc), npt.NDArray[np.uint8])
    assert_type(Shuffle().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(Shuffle().decode(enc, nd), npt.NDArray[Any])
    assert_type(Shuffle().decode(enc, out=ba), bytearray)
    assert_type(Shuffle().decode(enc, mv), memoryview[float])
    assert_type(Shuffle().decode(enc, maybe), npt.NDArray[np.uint8] | bytearray)
    Shuffle().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # CRC32
    assert_type(CRC32().decode(enc), npt.NDArray[np.uint8] | memoryview)
    assert_type(CRC32().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(CRC32().decode(enc, nd), npt.NDArray[Any])
    assert_type(CRC32().decode(enc, out=ba), bytearray)
    assert_type(CRC32().decode(enc, mv), memoryview[float])
    assert_type(CRC32().decode(enc, maybe), npt.NDArray[np.uint8] | memoryview | bytearray)
    CRC32().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # Adler32
    assert_type(Adler32().decode(enc), npt.NDArray[np.uint8] | memoryview)
    assert_type(Adler32().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(Adler32().decode(enc, nd), npt.NDArray[Any])
    assert_type(Adler32().decode(enc, out=ba), bytearray)
    assert_type(Adler32().decode(enc, mv), memoryview[float])
    assert_type(Adler32().decode(enc, maybe), npt.NDArray[np.uint8] | memoryview | bytearray)
    Adler32().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # JenkinsLookup3
    assert_type(JenkinsLookup3().decode(enc), memoryview)
    assert_type(JenkinsLookup3().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(JenkinsLookup3().decode(enc, nd), npt.NDArray[Any])
    assert_type(JenkinsLookup3().decode(enc, out=ba), bytearray)
    assert_type(JenkinsLookup3().decode(enc, mv), memoryview[float])
    assert_type(JenkinsLookup3().decode(enc, maybe), memoryview | bytearray)
    JenkinsLookup3().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # Fletcher32
    assert_type(Fletcher32().decode(enc), npt.NDArray[np.uint8])
    assert_type(Fletcher32().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(Fletcher32().decode(enc, nd), npt.NDArray[Any])
    assert_type(Fletcher32().decode(enc, out=ba), bytearray)
    assert_type(Fletcher32().decode(enc, mv), memoryview[float])
    assert_type(Fletcher32().decode(enc, maybe), npt.NDArray[np.uint8] | bytearray)
    Fletcher32().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # AsType
    assert_type(AsType("f4", "f8").decode(enc), npt.NDArray[Any])
    assert_type(AsType("f4", "f8").decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(AsType("f4", "f8").decode(enc, nd), npt.NDArray[Any])
    assert_type(AsType("f4", "f8").decode(enc, out=ba), bytearray)
    assert_type(AsType("f4", "f8").decode(enc, mv), memoryview[float])
    assert_type(AsType("f4", "f8").decode(enc, maybe), npt.NDArray[Any] | bytearray)
    AsType("f4", "f8").decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # Delta
    assert_type(Delta("f8").decode(enc), npt.NDArray[Any])
    assert_type(Delta("f8").decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(Delta("f8").decode(enc, nd), npt.NDArray[Any])
    assert_type(Delta("f8").decode(enc, out=ba), bytearray)
    assert_type(Delta("f8").decode(enc, mv), memoryview[float])
    assert_type(Delta("f8").decode(enc, maybe), npt.NDArray[Any] | bytearray)
    Delta("f8").decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # Quantize
    assert_type(Quantize(1, "f8").decode(enc), npt.NDArray[Any])
    assert_type(Quantize(1, "f8").decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(Quantize(1, "f8").decode(enc, nd), npt.NDArray[Any])
    assert_type(Quantize(1, "f8").decode(enc, out=ba), bytearray)
    assert_type(Quantize(1, "f8").decode(enc, mv), memoryview[float])
    assert_type(Quantize(1, "f8").decode(enc, maybe), npt.NDArray[Any] | bytearray)
    Quantize(1, "f8").decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # FixedScaleOffset
    assert_type(FixedScaleOffset(0, 1, "f8").decode(enc), npt.NDArray[Any])
    assert_type(FixedScaleOffset(0, 1, "f8").decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(FixedScaleOffset(0, 1, "f8").decode(enc, nd), npt.NDArray[Any])
    assert_type(FixedScaleOffset(0, 1, "f8").decode(enc, out=ba), bytearray)
    assert_type(FixedScaleOffset(0, 1, "f8").decode(enc, mv), memoryview[float])
    assert_type(FixedScaleOffset(0, 1, "f8").decode(enc, maybe), npt.NDArray[Any] | bytearray)
    FixedScaleOffset(0, 1, "f8").decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # BitRound
    assert_type(BitRound(10).decode(enc), NDArrayLike)
    assert_type(BitRound(10).decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(BitRound(10).decode(enc, nd), npt.NDArray[Any])
    assert_type(BitRound(10).decode(enc, out=ba), bytearray)
    assert_type(BitRound(10).decode(enc, mv), memoryview[float])
    assert_type(BitRound(10).decode(enc, maybe), NDArrayLike | bytearray)
    BitRound(10).decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # PackBits
    assert_type(PackBits().decode(enc), npt.NDArray[np.bool_])
    assert_type(PackBits().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(PackBits().decode(enc, nd), npt.NDArray[Any])
    assert_type(PackBits().decode(enc, out=ba), bytearray)
    assert_type(PackBits().decode(enc, mv), memoryview[float])
    assert_type(PackBits().decode(enc, maybe), npt.NDArray[np.bool_] | bytearray)
    PackBits().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # Categorize
    assert_type(Categorize(["a"], object).decode(enc), npt.NDArray[Any])
    assert_type(Categorize(["a"], object).decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(Categorize(["a"], object).decode(enc, nd), npt.NDArray[Any])
    assert_type(Categorize(["a"], object).decode(enc, out=ba), bytearray)
    assert_type(Categorize(["a"], object).decode(enc, mv), memoryview[float])
    assert_type(Categorize(["a"], object).decode(enc, maybe), npt.NDArray[Any] | bytearray)
    Categorize(["a"], object).decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # JSON
    assert_type(JSON().decode(enc), npt.NDArray[Any])
    assert_type(JSON().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(JSON().decode(enc, nd), npt.NDArray[Any])
    assert_type(JSON().decode(enc, out=ba), bytearray)
    assert_type(JSON().decode(enc, mv), memoryview[float])
    assert_type(JSON().decode(enc, maybe), npt.NDArray[Any] | bytearray)
    JSON().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # MsgPack
    assert_type(MsgPack().decode(enc), npt.NDArray[Any])
    assert_type(MsgPack().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(MsgPack().decode(enc, nd), npt.NDArray[Any])
    assert_type(MsgPack().decode(enc, out=ba), bytearray)
    assert_type(MsgPack().decode(enc, mv), memoryview[float])
    assert_type(MsgPack().decode(enc, maybe), npt.NDArray[Any] | bytearray)
    MsgPack().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # Pickle
    assert_type(Pickle().decode(enc), Any)
    assert_type(Pickle().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(Pickle().decode(enc, nd), npt.NDArray[Any])
    assert_type(Pickle().decode(enc, out=ba), bytearray)
    assert_type(Pickle().decode(enc, mv), memoryview[float])
    Pickle().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # VLenUTF8
    assert_type(VLenUTF8().decode(enc), npt.NDArray[np.object_])
    assert_type(VLenUTF8().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(VLenUTF8().decode(enc, nd), npt.NDArray[Any])
    assert_type(VLenUTF8().decode(enc, out=ba), bytearray)
    assert_type(VLenUTF8().decode(enc, mv), memoryview[float])
    assert_type(VLenUTF8().decode(enc, maybe), npt.NDArray[np.object_] | bytearray)
    VLenUTF8().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # VLenBytes
    assert_type(VLenBytes().decode(enc), npt.NDArray[np.object_])
    assert_type(VLenBytes().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(VLenBytes().decode(enc, nd), npt.NDArray[Any])
    assert_type(VLenBytes().decode(enc, out=ba), bytearray)
    assert_type(VLenBytes().decode(enc, mv), memoryview[float])
    assert_type(VLenBytes().decode(enc, maybe), npt.NDArray[np.object_] | bytearray)
    VLenBytes().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # VLenArray
    assert_type(VLenArray("i4").decode(enc), npt.NDArray[np.object_])
    assert_type(VLenArray("i4").decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(VLenArray("i4").decode(enc, nd), npt.NDArray[Any])
    assert_type(VLenArray("i4").decode(enc, out=ba), bytearray)
    assert_type(VLenArray("i4").decode(enc, mv), memoryview[float])
    assert_type(VLenArray("i4").decode(enc, maybe), npt.NDArray[np.object_] | bytearray)
    VLenArray("i4").decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # ZFPY
    assert_type(ZFPY().decode(enc), npt.NDArray[Any])
    assert_type(ZFPY().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(ZFPY().decode(enc, nd), npt.NDArray[Any])
    assert_type(ZFPY().decode(enc, out=ba), bytearray)
    assert_type(ZFPY().decode(enc, mv), memoryview[float])
    assert_type(ZFPY().decode(enc, maybe), npt.NDArray[Any] | bytearray)
    ZFPY().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # PCodec
    assert_type(PCodec().decode(enc), npt.NDArray[Any])
    assert_type(PCodec().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(PCodec().decode(enc, nd), npt.NDArray[Any])
    assert_type(PCodec().decode(enc, out=ba), bytearray)
    assert_type(PCodec().decode(enc, mv), memoryview[float])
    assert_type(PCodec().decode(enc, maybe), npt.NDArray[Any] | bytearray)
    PCodec().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # blosc.decompress
    assert_type(blosc.decompress(enc), bytes)
    assert_type(blosc.decompress(enc, nd), npt.NDArray[Any])
    assert_type(blosc.decompress(enc, ba), bytearray)
    assert_type(blosc.decompress(enc, maybe), bytes | bytearray)
    blosc.decompress(enc, enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # zstd.decompress
    assert_type(zstd.decompress(enc), bytes)
    assert_type(zstd.decompress(enc, nd), npt.NDArray[Any])
    assert_type(zstd.decompress(enc, ba), bytearray)
    assert_type(zstd.decompress(enc, maybe), bytes | bytearray)
    zstd.decompress(enc, enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # lz4.decompress
    assert_type(lz4.decompress(enc), bytes)
    assert_type(lz4.decompress(enc, nd), npt.NDArray[Any])
    assert_type(lz4.decompress(enc, ba), bytearray)
    assert_type(lz4.decompress(enc, maybe), bytes | bytearray)
    lz4.decompress(enc, enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # Shuffle.encode also takes `out`
    assert_type(Shuffle().encode(enc), npt.NDArray[np.uint8])
    assert_type(Shuffle().encode(enc, nd), npt.NDArray[Any])
    assert_type(Shuffle().encode(enc, out=ba), bytearray)
    assert_type(Shuffle().encode(enc, maybe), npt.NDArray[np.uint8] | bytearray)
    Shuffle().encode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # compat.ndarray_copy
    assert_type(ndarray_copy(enc, None), bytes)
    assert_type(ndarray_copy(enc, nd), npt.NDArray[Any])
    assert_type(ndarray_copy(enc, ba), bytearray)
    assert_type(ndarray_copy(enc, maybe), bytes | bytearray)
    ndarray_copy(enc, enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]


def test_every_codec_is_checked() -> None:
    """A codec added to numcodecs must be added to `static_checks` and `CHECKED_IDS`."""
    assert set(codec_registry) <= CHECKED_IDS
