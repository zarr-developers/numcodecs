"""The behavior every codec promises through `numcodecs.abc.Codec`, checked for every codec.

These tests pin down the contract that the type annotations on `Codec` describe:

- `encode` and `decode` return a buffer-like object (one that exports the buffer protocol).
- `decode(buf, out=out)` writes into `out` and returns that same object.
- `get_config()` holds an `id` and JSON-serializable values, and `from_config` rebuilds an
  equal codec from it once `id` is removed.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

import numcodecs
from numcodecs import pickles
from numcodecs.abc import Codec
from numcodecs.compat import ensure_ndarray


@dataclass(frozen=True)
class Case:
    """A codec and an array it can round-trip."""

    make: Callable[[], Codec]
    data: np.ndarray

    @property
    def is_object(self) -> bool:
        return self.data.dtype == object


f8 = np.linspace(0, 100, 16, dtype="f8")
obj_str = np.array(["foo", "bar", "baz", "qux"], dtype=object)

CASES: dict[str, Case] = {
    "zlib": Case(lambda: numcodecs.Zlib(), f8),
    "gzip": Case(lambda: numcodecs.GZip(), f8),
    "bz2": Case(lambda: numcodecs.BZ2(), f8),
    "lzma": Case(lambda: numcodecs.LZMA(), f8),
    "blosc": Case(lambda: numcodecs.Blosc(), f8),
    "zstd": Case(lambda: numcodecs.Zstd(), f8),
    "lz4": Case(lambda: numcodecs.LZ4(), f8),
    "base64": Case(lambda: numcodecs.Base64(), f8),
    "shuffle": Case(lambda: numcodecs.Shuffle(elementsize=8), f8),
    "crc32": Case(lambda: numcodecs.CRC32(), f8),
    "adler32": Case(lambda: numcodecs.Adler32(), f8),
    "jenkins_lookup3": Case(lambda: numcodecs.JenkinsLookup3(), f8),
    "fletcher32": Case(lambda: numcodecs.Fletcher32(), f8),
    "astype": Case(lambda: numcodecs.AsType(encode_dtype="f4", decode_dtype="f8"), f8),
    "delta": Case(lambda: numcodecs.Delta(dtype="i4"), np.arange(16, dtype="i4")),
    "quantize": Case(lambda: numcodecs.Quantize(digits=1, dtype="f8"), f8),
    "fixedscaleoffset": Case(
        lambda: numcodecs.FixedScaleOffset(offset=0, scale=10, dtype="f8", astype="i4"), f8
    ),
    "bitround": Case(lambda: numcodecs.BitRound(keepbits=10), f8),
    "packbits": Case(lambda: numcodecs.PackBits(), np.array([True, False] * 8)),
    "categorize": Case(
        lambda: numcodecs.Categorize(labels=["foo", "bar", "baz", "qux"], dtype=object), obj_str
    ),
    "pickle": Case(lambda: numcodecs.Pickle(), obj_str),
    "json2": Case(lambda: numcodecs.JSON(), obj_str),
    "vlen-utf8": Case(lambda: numcodecs.VLenUTF8(), obj_str),
    "vlen-bytes": Case(
        lambda: numcodecs.VLenBytes(), np.array([b"foo", b"bar", b"baz", b"qux"], dtype=object)
    ),
    "vlen-array": Case(
        lambda: numcodecs.VLenArray(dtype="i4"),
        np.array([np.arange(3, dtype="i4"), np.arange(2, dtype="i4")] * 2, dtype=object),
    ),
    "msgpack2": Case(lambda: numcodecs.MsgPack(), obj_str),
    "zfpy": Case(lambda: numcodecs.ZFPY(), f8),
    "pcodec": Case(lambda: numcodecs.PCodec(), f8),
    "crc32c": Case(lambda: numcodecs.CRC32C(), f8),
}

# Codecs whose decode(..., out=bytearray) cannot work, because the codec needs `out` to carry
# a dtype: PCodec decodes straight into a typed array.
NO_BYTEARRAY_OUT = frozenset({"pcodec"})

# Known violations of the `out` contract. Each entry is a bug; remove it when fixed.
# `strict=True` makes a fixed entry fail, so this list cannot go stale.
OUT_NDARRAY_XFAIL: set[str] = set()
OUT_BYTEARRAY_XFAIL: set[str] = set()
DECODE_RETURNS_BUFFER_XFAIL: set[str] = set()


def _codec(codec_id: str) -> Codec:
    case = CASES[codec_id]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return case.make()
    except AttributeError:
        pytest.skip(f"{codec_id} is not available (optional dependency missing)")


def _params(
    xfail: set[str], *, skip: frozenset[str] = frozenset(), objects: bool = True
) -> list[object]:
    params: list[object] = []
    for codec_id, case in CASES.items():
        if codec_id in skip or (case.is_object and not objects):
            continue
        marks = [pytest.mark.xfail(strict=True, reason="known bug")] if codec_id in xfail else []
        params.append(pytest.param(codec_id, marks=marks, id=codec_id))
    return params


@pytest.fixture(autouse=True)
def _allow_pickle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pickles, "ALLOW_UNSAFE_PICKLE_DECODE", True)


def _is_buffer_like(obj: object) -> bool:
    try:
        memoryview(obj)  # type: ignore[arg-type]
    except TypeError:
        return False
    return True


def test_every_registered_codec_has_a_case() -> None:
    """A new codec must be added to `CASES` so that the contract is checked for it."""
    assert set(numcodecs.registry.codec_registry) <= set(CASES)


@pytest.mark.parametrize("codec_id", _params(DECODE_RETURNS_BUFFER_XFAIL))
def test_encode_decode_return_buffers(codec_id: str) -> None:
    codec, data = _codec(codec_id), CASES[codec_id].data
    enc = codec.encode(data)
    assert _is_buffer_like(enc), type(enc)
    dec = codec.decode(enc)
    assert _is_buffer_like(dec), type(dec)


@pytest.mark.parametrize("codec_id", _params(OUT_NDARRAY_XFAIL))
def test_decode_into_ndarray_returns_out(codec_id: str) -> None:
    codec, data = _codec(codec_id), CASES[codec_id].data
    enc = codec.encode(data)
    expected = codec.decode(enc)
    out = np.empty_like(data)
    assert codec.decode(enc, out=out) is out
    if CASES[codec_id].is_object:
        assert all(np.array_equal(a, b) for a, b in zip(out, ensure_ndarray(expected), strict=True))
    else:
        assert ensure_ndarray(expected).tobytes() == out.tobytes()


@pytest.mark.parametrize(
    "codec_id", _params(OUT_BYTEARRAY_XFAIL, skip=NO_BYTEARRAY_OUT, objects=False)
)
def test_decode_into_bytearray_returns_out(codec_id: str) -> None:
    codec, data = _codec(codec_id), CASES[codec_id].data
    enc = codec.encode(data)
    expected = codec.decode(enc)
    out = bytearray(data.nbytes)
    assert codec.decode(enc, out=out) is out
    assert bytes(out) == ensure_ndarray(expected).tobytes()


@pytest.mark.parametrize("codec_id", _params(set()))
def test_config_round_trip(codec_id: str) -> None:
    codec = _codec(codec_id)
    config = codec.get_config()
    assert config["id"] == codec.codec_id
    rebuilt_json = json.loads(json.dumps(config))
    assert numcodecs.get_codec(rebuilt_json) == codec
    without_id = {k: v for k, v in config.items() if k != "id"}
    assert type(codec).from_config(without_id) == codec
