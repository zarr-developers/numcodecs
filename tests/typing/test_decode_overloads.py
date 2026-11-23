"""Static checks of the `decode` overloads of every codec.

Each codec overloads `decode` twice: without `out` it returns what the codec decodes to,
and with `out` it returns the type of `out`. `static_checks` is never called; it exists to
be type-checked by mypy and pyright, which verify every `assert_type` and report any
`ignore` comment that stops being needed. It must not run: some of its calls pass an
immutable `bytes` object as `out`.

The file is repetitive on purpose. A typing mistake in one codec's overloads (or in one
stub) is only caught by a line that names that codec. For each one it checks the result
with no `out`, an ndarray of known dtype, `NDArray[Any]`, a `bytearray` (also with both
arguments passed by keyword), a cast
`memoryview` and an optional `out`; that `bytes` is rejected as `out`; and that the
no-`out` overload comes first, which is what `executor.map(codec.decode, ...)` and
`functools.partial(codec.decode, ...)` resolve to.

Known limits, not checked here:

- With an `out` typed `Any`, mypy infers `Any` but pyright picks the first overload and
  infers the no-`out` type.
- JSON, MsgPack, Pickle, Categorize, the VLen codecs and PCodec need an ndarray `out` and
  raise for other buffers, which the inherited signature cannot express.
"""

# `memoryview[float]` in the signature below cannot be evaluated before Python 3.14
from __future__ import annotations

import array
import ctypes
import inspect
import mmap
import pickle
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, assert_type, get_overloads

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
    # These exist only when an optional dependency is installed. `static_checks` is
    # never run, so the names are only needed by the type checkers.
    from numcodecs.checksum32 import CRC32C
    from numcodecs.msgpacks import MsgPack
    from numcodecs.pcodec import PCodec
    from numcodecs.zfpy import ZFPY


def _first_result[T](func: Callable[..., T]) -> T:
    """What type checkers take as the result of an overloaded callable used where any
    callable is expected: that of its first overload. Never called."""
    raise NotImplementedError


def static_checks(
    enc: bytes,
    f8: npt.NDArray[np.float64],
    nd: npt.NDArray[Any],
    ba: bytearray,
    maybe: bytearray | None,
    mv: memoryview[float],
    like: NDArrayLike,
) -> None:
    # Zlib
    assert_type(Zlib().decode(enc), bytes)
    assert_type(Zlib().decode(buf=enc), bytes)
    assert_type(Zlib().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(Zlib().decode(enc, nd), npt.NDArray[Any])
    assert_type(Zlib().decode(enc, out=ba), bytearray)
    assert_type(Zlib().decode(buf=enc, out=ba), bytearray)
    assert_type(Zlib().decode(enc, mv), memoryview[float])
    assert_type(Zlib().decode(enc, maybe), bytes | bytearray)
    assert_type(_first_result(Zlib().decode), bytes)
    Zlib().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # GZip
    assert_type(GZip().decode(enc), bytes)
    assert_type(GZip().decode(buf=enc), bytes)
    assert_type(GZip().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(GZip().decode(enc, nd), npt.NDArray[Any])
    assert_type(GZip().decode(enc, out=ba), bytearray)
    assert_type(GZip().decode(buf=enc, out=ba), bytearray)
    assert_type(GZip().decode(enc, mv), memoryview[float])
    assert_type(GZip().decode(enc, maybe), bytes | bytearray)
    assert_type(_first_result(GZip().decode), bytes)
    GZip().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # BZ2
    assert_type(BZ2().decode(enc), bytes)
    assert_type(BZ2().decode(buf=enc), bytes)
    assert_type(BZ2().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(BZ2().decode(enc, nd), npt.NDArray[Any])
    assert_type(BZ2().decode(enc, out=ba), bytearray)
    assert_type(BZ2().decode(buf=enc, out=ba), bytearray)
    assert_type(BZ2().decode(enc, mv), memoryview[float])
    assert_type(BZ2().decode(enc, maybe), bytes | bytearray)
    assert_type(_first_result(BZ2().decode), bytes)
    BZ2().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # LZMA
    assert_type(LZMA().decode(enc), bytes)
    assert_type(LZMA().decode(buf=enc), bytes)
    assert_type(LZMA().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(LZMA().decode(enc, nd), npt.NDArray[Any])
    assert_type(LZMA().decode(enc, out=ba), bytearray)
    assert_type(LZMA().decode(buf=enc, out=ba), bytearray)
    assert_type(LZMA().decode(enc, mv), memoryview[float])
    assert_type(LZMA().decode(enc, maybe), bytes | bytearray)
    assert_type(_first_result(LZMA().decode), bytes)
    LZMA().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # Base64
    assert_type(Base64().decode(enc), bytes)
    assert_type(Base64().decode(buf=enc), bytes)
    assert_type(Base64().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(Base64().decode(enc, nd), npt.NDArray[Any])
    assert_type(Base64().decode(enc, out=ba), bytearray)
    assert_type(Base64().decode(buf=enc, out=ba), bytearray)
    assert_type(Base64().decode(enc, mv), memoryview[float])
    assert_type(Base64().decode(enc, maybe), bytes | bytearray)
    assert_type(_first_result(Base64().decode), bytes)
    Base64().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # Blosc
    assert_type(Blosc().decode(enc), bytes)
    assert_type(Blosc().decode(buf=enc), bytes)
    assert_type(Blosc().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(Blosc().decode(enc, nd), npt.NDArray[Any])
    assert_type(Blosc().decode(enc, out=ba), bytearray)
    assert_type(Blosc().decode(buf=enc, out=ba), bytearray)
    assert_type(Blosc().decode(enc, mv), memoryview[float])
    assert_type(Blosc().decode(enc, maybe), bytes | bytearray)
    assert_type(_first_result(Blosc().decode), bytes)
    Blosc().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # Zstd
    assert_type(Zstd().decode(enc), bytes)
    assert_type(Zstd().decode(buf=enc), bytes)
    assert_type(Zstd().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(Zstd().decode(enc, nd), npt.NDArray[Any])
    assert_type(Zstd().decode(enc, out=ba), bytearray)
    assert_type(Zstd().decode(buf=enc, out=ba), bytearray)
    assert_type(Zstd().decode(enc, mv), memoryview[float])
    assert_type(Zstd().decode(enc, maybe), bytes | bytearray)
    assert_type(_first_result(Zstd().decode), bytes)
    Zstd().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # LZ4
    assert_type(LZ4().decode(enc), bytes)
    assert_type(LZ4().decode(buf=enc), bytes)
    assert_type(LZ4().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(LZ4().decode(enc, nd), npt.NDArray[Any])
    assert_type(LZ4().decode(enc, out=ba), bytearray)
    assert_type(LZ4().decode(buf=enc, out=ba), bytearray)
    assert_type(LZ4().decode(enc, mv), memoryview[float])
    assert_type(LZ4().decode(enc, maybe), bytes | bytearray)
    assert_type(_first_result(LZ4().decode), bytes)
    LZ4().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # Shuffle
    assert_type(Shuffle().decode(enc), npt.NDArray[np.uint8])
    assert_type(Shuffle().decode(buf=enc), npt.NDArray[np.uint8])
    assert_type(Shuffle().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(Shuffle().decode(enc, nd), npt.NDArray[Any])
    assert_type(Shuffle().decode(enc, out=ba), bytearray)
    assert_type(Shuffle().decode(buf=enc, out=ba), bytearray)
    assert_type(Shuffle().decode(enc, mv), memoryview[float])
    assert_type(Shuffle().decode(enc, maybe), npt.NDArray[np.uint8] | bytearray)
    assert_type(_first_result(Shuffle().decode), npt.NDArray[np.uint8])
    Shuffle().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # CRC32
    assert_type(CRC32().decode(enc), npt.NDArray[np.uint8] | memoryview)
    assert_type(CRC32().decode(buf=enc), npt.NDArray[np.uint8] | memoryview)
    assert_type(CRC32().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(CRC32().decode(enc, nd), npt.NDArray[Any])
    assert_type(CRC32().decode(enc, out=ba), bytearray)
    assert_type(CRC32().decode(buf=enc, out=ba), bytearray)
    assert_type(CRC32().decode(enc, mv), memoryview[float])
    assert_type(CRC32().decode(enc, maybe), npt.NDArray[np.uint8] | memoryview | bytearray)
    assert_type(_first_result(CRC32().decode), npt.NDArray[np.uint8] | memoryview)
    CRC32().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # Adler32
    assert_type(Adler32().decode(enc), npt.NDArray[np.uint8] | memoryview)
    assert_type(Adler32().decode(buf=enc), npt.NDArray[np.uint8] | memoryview)
    assert_type(Adler32().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(Adler32().decode(enc, nd), npt.NDArray[Any])
    assert_type(Adler32().decode(enc, out=ba), bytearray)
    assert_type(Adler32().decode(buf=enc, out=ba), bytearray)
    assert_type(Adler32().decode(enc, mv), memoryview[float])
    assert_type(Adler32().decode(enc, maybe), npt.NDArray[np.uint8] | memoryview | bytearray)
    assert_type(_first_result(Adler32().decode), npt.NDArray[np.uint8] | memoryview)
    Adler32().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # CRC32C
    assert_type(CRC32C().decode(enc), npt.NDArray[np.uint8] | memoryview)
    assert_type(CRC32C().decode(buf=enc), npt.NDArray[np.uint8] | memoryview)
    assert_type(CRC32C().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(CRC32C().decode(enc, nd), npt.NDArray[Any])
    assert_type(CRC32C().decode(enc, out=ba), bytearray)
    assert_type(CRC32C().decode(buf=enc, out=ba), bytearray)
    assert_type(CRC32C().decode(enc, mv), memoryview[float])
    assert_type(CRC32C().decode(enc, maybe), npt.NDArray[np.uint8] | memoryview | bytearray)
    assert_type(_first_result(CRC32C().decode), npt.NDArray[np.uint8] | memoryview)
    CRC32C().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # JenkinsLookup3
    assert_type(JenkinsLookup3().decode(enc), memoryview)
    assert_type(JenkinsLookup3().decode(buf=enc), memoryview)
    assert_type(JenkinsLookup3().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(JenkinsLookup3().decode(enc, nd), npt.NDArray[Any])
    assert_type(JenkinsLookup3().decode(enc, out=ba), bytearray)
    assert_type(JenkinsLookup3().decode(buf=enc, out=ba), bytearray)
    assert_type(JenkinsLookup3().decode(enc, mv), memoryview[float])
    assert_type(JenkinsLookup3().decode(enc, maybe), memoryview | bytearray)
    assert_type(_first_result(JenkinsLookup3().decode), memoryview)
    JenkinsLookup3().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # Fletcher32
    assert_type(Fletcher32().decode(enc), npt.NDArray[np.uint8])
    assert_type(Fletcher32().decode(buf=enc), npt.NDArray[np.uint8])
    assert_type(Fletcher32().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(Fletcher32().decode(enc, nd), npt.NDArray[Any])
    assert_type(Fletcher32().decode(enc, out=ba), bytearray)
    assert_type(Fletcher32().decode(buf=enc, out=ba), bytearray)
    assert_type(Fletcher32().decode(enc, mv), memoryview[float])
    assert_type(Fletcher32().decode(enc, maybe), npt.NDArray[np.uint8] | bytearray)
    assert_type(_first_result(Fletcher32().decode), npt.NDArray[np.uint8])
    Fletcher32().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # AsType
    assert_type(AsType("f4", "f8").decode(enc), npt.NDArray[Any])
    assert_type(AsType("f4", "f8").decode(buf=enc), npt.NDArray[Any])
    assert_type(AsType("f4", "f8").decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(AsType("f4", "f8").decode(enc, nd), npt.NDArray[Any])
    assert_type(AsType("f4", "f8").decode(enc, out=ba), bytearray)
    assert_type(AsType("f4", "f8").decode(buf=enc, out=ba), bytearray)
    assert_type(AsType("f4", "f8").decode(enc, mv), memoryview[float])
    assert_type(AsType("f4", "f8").decode(enc, maybe), npt.NDArray[Any] | bytearray)
    assert_type(_first_result(AsType("f4", "f8").decode), npt.NDArray[Any])
    AsType("f4", "f8").decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # Delta
    assert_type(Delta("f8").decode(enc), npt.NDArray[Any])
    assert_type(Delta("f8").decode(buf=enc), npt.NDArray[Any])
    assert_type(Delta("f8").decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(Delta("f8").decode(enc, nd), npt.NDArray[Any])
    assert_type(Delta("f8").decode(enc, out=ba), bytearray)
    assert_type(Delta("f8").decode(buf=enc, out=ba), bytearray)
    assert_type(Delta("f8").decode(enc, mv), memoryview[float])
    assert_type(Delta("f8").decode(enc, maybe), npt.NDArray[Any] | bytearray)
    assert_type(_first_result(Delta("f8").decode), npt.NDArray[Any])
    Delta("f8").decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # Quantize
    assert_type(Quantize(1, "f8").decode(enc), npt.NDArray[Any])
    assert_type(Quantize(1, "f8").decode(buf=enc), npt.NDArray[Any])
    assert_type(Quantize(1, "f8").decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(Quantize(1, "f8").decode(enc, nd), npt.NDArray[Any])
    assert_type(Quantize(1, "f8").decode(enc, out=ba), bytearray)
    assert_type(Quantize(1, "f8").decode(buf=enc, out=ba), bytearray)
    assert_type(Quantize(1, "f8").decode(enc, mv), memoryview[float])
    assert_type(Quantize(1, "f8").decode(enc, maybe), npt.NDArray[Any] | bytearray)
    assert_type(_first_result(Quantize(1, "f8").decode), npt.NDArray[Any])
    Quantize(1, "f8").decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # FixedScaleOffset
    assert_type(FixedScaleOffset(0, 1, "f8").decode(enc), npt.NDArray[Any])
    assert_type(FixedScaleOffset(0, 1, "f8").decode(buf=enc), npt.NDArray[Any])
    assert_type(FixedScaleOffset(0, 1, "f8").decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(FixedScaleOffset(0, 1, "f8").decode(enc, nd), npt.NDArray[Any])
    assert_type(FixedScaleOffset(0, 1, "f8").decode(enc, out=ba), bytearray)
    assert_type(FixedScaleOffset(0, 1, "f8").decode(buf=enc, out=ba), bytearray)
    assert_type(FixedScaleOffset(0, 1, "f8").decode(enc, mv), memoryview[float])
    assert_type(FixedScaleOffset(0, 1, "f8").decode(enc, maybe), npt.NDArray[Any] | bytearray)
    assert_type(_first_result(FixedScaleOffset(0, 1, "f8").decode), npt.NDArray[Any])
    FixedScaleOffset(0, 1, "f8").decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # BitRound
    assert_type(BitRound(10).decode(enc), NDArrayLike)
    assert_type(BitRound(10).decode(buf=enc), NDArrayLike)
    assert_type(BitRound(10).decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(BitRound(10).decode(enc, nd), npt.NDArray[Any])
    assert_type(BitRound(10).decode(enc, out=ba), bytearray)
    assert_type(BitRound(10).decode(buf=enc, out=ba), bytearray)
    assert_type(BitRound(10).decode(enc, mv), memoryview[float])
    assert_type(BitRound(10).decode(enc, maybe), NDArrayLike | bytearray)
    assert_type(_first_result(BitRound(10).decode), NDArrayLike)
    BitRound(10).decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # PackBits
    assert_type(PackBits().decode(enc), npt.NDArray[np.bool_])
    assert_type(PackBits().decode(buf=enc), npt.NDArray[np.bool_])
    assert_type(PackBits().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(PackBits().decode(enc, nd), npt.NDArray[Any])
    assert_type(PackBits().decode(enc, out=ba), bytearray)
    assert_type(PackBits().decode(buf=enc, out=ba), bytearray)
    assert_type(PackBits().decode(enc, mv), memoryview[float])
    assert_type(PackBits().decode(enc, maybe), npt.NDArray[np.bool_] | bytearray)
    assert_type(_first_result(PackBits().decode), npt.NDArray[np.bool_])
    PackBits().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # Categorize
    assert_type(Categorize(["a"], object).decode(enc), npt.NDArray[Any])
    assert_type(Categorize(["a"], object).decode(buf=enc), npt.NDArray[Any])
    assert_type(Categorize(["a"], object).decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(Categorize(["a"], object).decode(enc, nd), npt.NDArray[Any])
    assert_type(Categorize(["a"], object).decode(enc, out=ba), bytearray)
    assert_type(Categorize(["a"], object).decode(buf=enc, out=ba), bytearray)
    assert_type(Categorize(["a"], object).decode(enc, mv), memoryview[float])
    assert_type(Categorize(["a"], object).decode(enc, maybe), npt.NDArray[Any] | bytearray)
    assert_type(_first_result(Categorize(["a"], object).decode), npt.NDArray[Any])
    Categorize(["a"], object).decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # JSON
    assert_type(JSON().decode(enc), npt.NDArray[Any])
    assert_type(JSON().decode(buf=enc), npt.NDArray[Any])
    assert_type(JSON().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(JSON().decode(enc, nd), npt.NDArray[Any])
    assert_type(JSON().decode(enc, out=ba), bytearray)
    assert_type(JSON().decode(buf=enc, out=ba), bytearray)
    assert_type(JSON().decode(enc, mv), memoryview[float])
    assert_type(JSON().decode(enc, maybe), npt.NDArray[Any] | bytearray)
    assert_type(_first_result(JSON().decode), npt.NDArray[Any])
    JSON().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # MsgPack
    assert_type(MsgPack().decode(enc), npt.NDArray[Any])
    assert_type(MsgPack().decode(buf=enc), npt.NDArray[Any])
    assert_type(MsgPack().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(MsgPack().decode(enc, nd), npt.NDArray[Any])
    assert_type(MsgPack().decode(enc, out=ba), bytearray)
    assert_type(MsgPack().decode(buf=enc, out=ba), bytearray)
    assert_type(MsgPack().decode(enc, mv), memoryview[float])
    assert_type(MsgPack().decode(enc, maybe), npt.NDArray[Any] | bytearray)
    assert_type(_first_result(MsgPack().decode), npt.NDArray[Any])
    MsgPack().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # Pickle
    assert_type(Pickle().decode(enc), Any)
    assert_type(Pickle().decode(buf=enc), Any)
    assert_type(Pickle().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(Pickle().decode(enc, nd), npt.NDArray[Any])
    assert_type(Pickle().decode(enc, out=ba), bytearray)
    assert_type(Pickle().decode(buf=enc, out=ba), bytearray)
    assert_type(Pickle().decode(enc, mv), memoryview[float])
    assert_type(Pickle().decode(enc, maybe), Any | bytearray)
    assert_type(_first_result(Pickle().decode), Any)
    Pickle().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # VLenUTF8
    assert_type(VLenUTF8().decode(enc), npt.NDArray[np.object_])
    assert_type(VLenUTF8().decode(buf=enc), npt.NDArray[np.object_])
    assert_type(VLenUTF8().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(VLenUTF8().decode(enc, nd), npt.NDArray[Any])
    assert_type(VLenUTF8().decode(enc, out=ba), bytearray)
    assert_type(VLenUTF8().decode(buf=enc, out=ba), bytearray)
    assert_type(VLenUTF8().decode(enc, mv), memoryview[float])
    assert_type(VLenUTF8().decode(enc, maybe), npt.NDArray[np.object_] | bytearray)
    assert_type(_first_result(VLenUTF8().decode), npt.NDArray[np.object_])
    VLenUTF8().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # VLenBytes
    assert_type(VLenBytes().decode(enc), npt.NDArray[np.object_])
    assert_type(VLenBytes().decode(buf=enc), npt.NDArray[np.object_])
    assert_type(VLenBytes().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(VLenBytes().decode(enc, nd), npt.NDArray[Any])
    assert_type(VLenBytes().decode(enc, out=ba), bytearray)
    assert_type(VLenBytes().decode(buf=enc, out=ba), bytearray)
    assert_type(VLenBytes().decode(enc, mv), memoryview[float])
    assert_type(VLenBytes().decode(enc, maybe), npt.NDArray[np.object_] | bytearray)
    assert_type(_first_result(VLenBytes().decode), npt.NDArray[np.object_])
    VLenBytes().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # VLenArray
    assert_type(VLenArray("i4").decode(enc), npt.NDArray[np.object_])
    assert_type(VLenArray("i4").decode(buf=enc), npt.NDArray[np.object_])
    assert_type(VLenArray("i4").decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(VLenArray("i4").decode(enc, nd), npt.NDArray[Any])
    assert_type(VLenArray("i4").decode(enc, out=ba), bytearray)
    assert_type(VLenArray("i4").decode(buf=enc, out=ba), bytearray)
    assert_type(VLenArray("i4").decode(enc, mv), memoryview[float])
    assert_type(VLenArray("i4").decode(enc, maybe), npt.NDArray[np.object_] | bytearray)
    assert_type(_first_result(VLenArray("i4").decode), npt.NDArray[np.object_])
    VLenArray("i4").decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # ZFPY
    assert_type(ZFPY().decode(enc), npt.NDArray[Any])
    assert_type(ZFPY().decode(buf=enc), npt.NDArray[Any])
    assert_type(ZFPY().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(ZFPY().decode(enc, nd), npt.NDArray[Any])
    assert_type(ZFPY().decode(enc, out=ba), bytearray)
    assert_type(ZFPY().decode(buf=enc, out=ba), bytearray)
    assert_type(ZFPY().decode(enc, mv), memoryview[float])
    assert_type(ZFPY().decode(enc, maybe), npt.NDArray[Any] | bytearray)
    assert_type(_first_result(ZFPY().decode), npt.NDArray[Any])
    ZFPY().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # PCodec
    assert_type(PCodec().decode(enc), npt.NDArray[Any])
    assert_type(PCodec().decode(buf=enc), npt.NDArray[Any])
    assert_type(PCodec().decode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(PCodec().decode(enc, nd), npt.NDArray[Any])
    assert_type(PCodec().decode(enc, out=ba), bytearray)
    assert_type(PCodec().decode(buf=enc, out=ba), bytearray)
    assert_type(PCodec().decode(enc, mv), memoryview[float])
    assert_type(PCodec().decode(enc, maybe), npt.NDArray[Any] | bytearray)
    assert_type(_first_result(PCodec().decode), npt.NDArray[Any])
    PCodec().decode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # blosc.decompress
    assert_type(blosc.decompress(enc), bytes)
    assert_type(blosc.decompress(enc, f8), npt.NDArray[np.float64])
    assert_type(blosc.decompress(enc, nd), npt.NDArray[Any])
    assert_type(blosc.decompress(enc, ba), bytearray)
    assert_type(blosc.decompress(source=nd, dest=ba), bytearray)
    assert_type(blosc.decompress(source=nd), bytes)
    assert_type(blosc.decompress(enc, mv), memoryview[float])
    assert_type(blosc.decompress(enc, maybe), bytes | bytearray)
    assert_type(_first_result(blosc.decompress), bytes)
    blosc.decompress(enc, enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # zstd.decompress
    assert_type(zstd.decompress(enc), bytes)
    assert_type(zstd.decompress(enc, f8), npt.NDArray[np.float64])
    assert_type(zstd.decompress(enc, nd), npt.NDArray[Any])
    assert_type(zstd.decompress(enc, ba), bytearray)
    assert_type(zstd.decompress(source=nd, dest=ba), bytearray)
    assert_type(zstd.decompress(source=nd), bytes)
    assert_type(zstd.decompress(enc, mv), memoryview[float])
    assert_type(zstd.decompress(enc, maybe), bytes | bytearray)
    assert_type(_first_result(zstd.decompress), bytes)
    zstd.decompress(enc, enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # lz4.decompress
    assert_type(lz4.decompress(enc), bytes)
    assert_type(lz4.decompress(enc, f8), npt.NDArray[np.float64])
    assert_type(lz4.decompress(enc, nd), npt.NDArray[Any])
    assert_type(lz4.decompress(enc, ba), bytearray)
    assert_type(lz4.decompress(source=nd, dest=ba), bytearray)
    assert_type(lz4.decompress(source=nd), bytes)
    assert_type(lz4.decompress(enc, mv), memoryview[float])
    assert_type(lz4.decompress(enc, maybe), bytes | bytearray)
    assert_type(_first_result(lz4.decompress), bytes)
    lz4.decompress(enc, enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # Shuffle.encode also takes `out`
    assert_type(Shuffle().encode(enc), npt.NDArray[np.uint8])
    assert_type(Shuffle().encode(buf=enc), npt.NDArray[np.uint8])
    assert_type(Shuffle().encode(enc, out=f8), npt.NDArray[np.float64])
    assert_type(Shuffle().encode(enc, nd), npt.NDArray[Any])
    assert_type(Shuffle().encode(enc, out=ba), bytearray)
    assert_type(Shuffle().encode(buf=enc, out=ba), bytearray)
    assert_type(Shuffle().encode(enc, mv), memoryview[float])
    assert_type(Shuffle().encode(enc, maybe), npt.NDArray[np.uint8] | bytearray)
    assert_type(_first_result(Shuffle().encode), npt.NDArray[np.uint8])
    Shuffle().encode(enc, out=enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # compat.ndarray_copy(src, dst) has no default for `dst`
    assert_type(ndarray_copy(enc, None), bytes)
    assert_type(ndarray_copy(enc, f8), npt.NDArray[np.float64])
    assert_type(ndarray_copy(enc, nd), npt.NDArray[Any])
    assert_type(ndarray_copy(enc, ba), bytearray)
    assert_type(ndarray_copy(src=enc, dst=ba), bytearray)
    assert_type(ndarray_copy(src=enc, dst=None), bytes)
    # `src` may be an ndarray-like that does not export a buffer
    assert_type(ndarray_copy(like, ba), bytearray)
    assert_type(ndarray_copy(like, None), NDArrayLike)
    assert_type(ndarray_copy(enc, mv), memoryview[float])
    assert_type(ndarray_copy(enc, maybe), bytes | bytearray)
    ndarray_copy(enc, enc)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    # every kind of buffer WritableBufferLike names is accepted as `out`, and comes back
    assert_type(Zlib().decode(enc, array.array("B")), array.array[int])
    assert_type(Zlib().decode(enc, array.array("d")), array.array[float])
    assert_type(Zlib().decode(enc, mmap.mmap(-1, 8)), mmap.mmap)
    assert_type(Zlib().decode(enc, (ctypes.c_ubyte * 8)()), ctypes.Array[ctypes.c_ubyte])
    assert_type(Zlib().decode(enc, (ctypes.c_double * 1)()), ctypes.Array[ctypes.c_double])
    assert_type(Zlib().decode(enc, pickle.PickleBuffer(ba)), pickle.PickleBuffer)
    assert_type(
        Zlib().decode(enc, np.ma.masked_array(f8)),
        np.ma.MaskedArray[tuple[Any, ...], np.dtype[np.float64]],
    )
    assert_type(Zlib().decode(enc, like), NDArrayLike)


def test_every_codec_is_checked() -> None:
    """A codec added to numcodecs must get its own lines in `static_checks`."""
    source = inspect.getsource(static_checks)
    missing = [
        cls.__name__
        for cls in codec_registry.values()
        if f"{cls.__name__}(" not in source or f"# {cls.__name__}\n" not in source
    ]
    assert not missing


def test_ndarray_copy_overload_order() -> None:
    """The overload for `dst=None` comes first, as it does for every `decode`. Its result
    is generic, so `_first_result` cannot show the order to a type checker; this reads
    it at runtime instead."""
    first, second = get_overloads(ndarray_copy)
    assert inspect.get_annotations(first, eval_str=True)["dst"] is None
    assert inspect.get_annotations(second, eval_str=True)["dst"] is not None
