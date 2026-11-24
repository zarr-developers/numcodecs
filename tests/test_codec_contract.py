"""The behavior every codec promises through `numcodecs.abc.Codec`, checked for every codec.

These tests pin down the contract that the type annotations on `Codec` describe:

- `encode` and `decode` return a buffer-like object (one that exports the buffer protocol).
- `decode(buf, out=out)` writes into `out` and returns that same object.
- `get_config()` holds an `id` and JSON-serializable values, and `from_config` rebuilds an
  equal codec from it once `id` is removed.
"""

from __future__ import annotations

import collections.abc
import json
import types
import typing
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import typing_extensions

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


def _c_2d(data: np.ndarray) -> np.ndarray:
    return np.empty((2, data.size // 2), dtype=data.dtype)


def _f_2d(data: np.ndarray) -> np.ndarray:
    return np.empty((2, data.size // 2), dtype=data.dtype, order="F")


def _strided_1d(data: np.ndarray) -> np.ndarray:
    return np.empty(data.size * 2, dtype=data.dtype)[::2]


def _strided_2d(data: np.ndarray) -> np.ndarray:
    return np.empty((4, data.size // 2), dtype=data.dtype)[::2]


AWKWARD_OUTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "2d-C": _c_2d,
    "2d-F": _f_2d,
    "strided-1d": _strided_1d,
    "strided-2d": _strided_2d,
}


# Codecs that write `out` as one block of memory, and so need it to be contiguous.
NEEDS_CONTIGUOUS_OUT = frozenset(
    {
        "zlib",
        "gzip",
        "bz2",
        "lzma",
        "blosc",
        "zstd",
        "lz4",
        "base64",
        "shuffle",
        "crc32",
        "adler32",
        "crc32c",
        "fletcher32",
        "zfpy",
        "pcodec",
    }
)
# Codecs that copy with `np.copyto(out, decoded)`, so `out` must have the decoded shape
# (1-D for the case data used here).
NEEDS_DECODED_SHAPE_OUT = frozenset({"json2", "msgpack2", "pickle"})


def _refuses(codec_id: str, out_kind: str) -> bool:
    """Whether `codec_id` is expected to refuse an `out` of this kind. Every codec not
    named above must fill every kind of `out`."""
    if codec_id in NEEDS_CONTIGUOUS_OUT:
        return out_kind.startswith("strided")
    if codec_id in NEEDS_DECODED_SHAPE_OUT:
        return out_kind != "strided-1d"
    return False


@pytest.mark.parametrize("out_kind", AWKWARD_OUTS)
@pytest.mark.parametrize("codec_id", _params(set()))
def test_decode_into_awkward_out(codec_id: str, out_kind: str) -> None:
    """Decoding into a 2-D or non-contiguous `out` either fills `out` and returns it, or
    raises `ValueError` and leaves `out` untouched. Which of the two is fixed per codec
    by `_refuses`, so a codec cannot pass by raising for a layout it should fill, and it
    can never return an `out` that does not hold the decoded data."""
    codec, case = _codec(codec_id), CASES[codec_id]
    enc = codec.encode(case.data)
    expected = ensure_ndarray(codec.decode(enc))
    out = AWKWARD_OUTS[out_kind](case.data)
    sentinel = object()
    out[...] = sentinel if case.is_object else np.zeros((), dtype=case.data.dtype)
    untouched = out.copy()

    if _refuses(codec_id, out_kind):
        with pytest.raises(ValueError):
            codec.decode(enc, out=out)
        if case.is_object:
            assert all(item is sentinel for item in out.flat)
        else:
            assert out.tobytes() == untouched.tobytes()
        return

    assert codec.decode(enc, out=out) is out
    # numcodecs fills `out` in memory order when it is contiguous (so an F-contiguous
    # `out` is filled column by column) and in C order otherwise; order="A" reads it back
    # the same way.
    flat = out.reshape(-1, order="A")
    if case.is_object:
        assert all(item is not sentinel for item in flat)
        assert all(np.array_equal(a, b) for a, b in zip(flat, expected.ravel(), strict=True))
    else:
        assert flat.tobytes() == expected.tobytes()


VLEN_NON_CONTIGUOUS_OUTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "every other row": lambda base: base[::2],
    "every other row and column": lambda base: base[::2, ::2],
    "rows reversed": lambda base: base[::-1],
    "transposed and strided": lambda base: base.T[::2],
    "3-D, strided on the middle axis": lambda base: base.reshape(2, 4, 3)[:, ::2],
}


@pytest.mark.parametrize("layout", VLEN_NON_CONTIGUOUS_OUTS)
@pytest.mark.parametrize("codec_id", ["vlen-utf8", "vlen-bytes", "vlen-array"])
def test_vlen_decode_fills_a_non_contiguous_out(codec_id: str, layout: str) -> None:
    """`out` cannot be flattened without a copy, so the decoded items have to be copied
    back into it, in logical (C) order. The rest of the memory `out` is a view of is left
    alone."""
    codec = _codec(codec_id)
    stale = object()
    base = np.full((6, 4), stale, dtype=object)
    out = VLEN_NON_CONTIGUOUS_OUTS[layout](base)
    # numpy cannot flatten any of these layouts without a copy
    assert not np.shares_memory(out.reshape(-1, order="A"), out)
    data = np.resize(CASES[codec_id].data, out.size)

    assert codec.decode(codec.encode(data), out=out) is out
    assert all(np.array_equal(a, b) for a, b in zip(out.ravel(), data, strict=True))
    assert sum(item is stale for item in base.flat) == base.size - out.size


@pytest.mark.parametrize("codec_id", ["vlen-utf8", "vlen-bytes", "vlen-array"])
def test_vlen_decode_refuses_a_read_only_non_contiguous_out(codec_id: str) -> None:
    """The decoded items cannot be copied back, and returning `out` unfilled would be
    wrong data."""
    codec = _codec(codec_id)
    stale = object()
    base = np.full((4, 2), stale, dtype=object)
    out = base[::2]
    out.setflags(write=False)
    with pytest.raises(ValueError, match="read-only"):
        codec.decode(codec.encode(np.resize(CASES[codec_id].data, out.size)), out=out)
    assert all(item is stale for item in base.flat)


@pytest.mark.parametrize("codec_id", ["vlen-utf8", "vlen-bytes", "vlen-array"])
def test_vlen_decode_into_an_empty_read_only_out(codec_id: str) -> None:
    """There is nothing to write, so a read-only `out` is not an error."""
    codec = _codec(codec_id)
    out = np.empty(0, dtype=object)
    out.setflags(write=False)
    assert codec.decode(codec.encode(np.empty(0, dtype=object)), out=out) is out


@pytest.mark.parametrize("codec_id", ["json2", "msgpack2", "pickle"])
def test_object_codec_decode_rejects_non_ndarray_out(codec_id: str) -> None:
    """These codecs copy with `np.copyto`, which needs an ndarray destination. A
    `bytearray` must not be silently broadcast into."""
    codec = _codec(codec_id)
    enc = codec.encode(np.array([7], dtype="u1"))
    with pytest.raises(TypeError):
        codec.decode(enc, out=bytearray(8))


def test_config_typeddicts_keep_their_required_keys() -> None:
    """Guards the typing_extensions floor: with too old a version, subclasses of
    `CodecConfig` silently lose their required keys on newer Pythons."""
    from numcodecs.checksum32 import Checksum32Config
    from numcodecs.zlib import ZlibConfig

    assert ZlibConfig.__required_keys__ == {"id", "level"}
    assert Checksum32Config.__required_keys__ == {"id"}
    assert Checksum32Config.__optional_keys__ == {"location"}


def _conforms(value: Any, annotation: Any) -> bool:
    """Whether `value` has the type `annotation`, for the annotations the configuration
    TypedDicts use. Stricter than a type checker in one way: `bool` does not pass for
    `int` or `float`. An annotation it does not know raises, so that it gets added."""
    if annotation is Any:
        return True
    if isinstance(annotation, typing.TypeAliasType):
        return _conforms(value, annotation.__value__)
    if annotation is None or annotation is type(None):
        return value is None
    if annotation is float:
        return type(value) in (int, float)
    if annotation in (int, str, bool):
        return type(value) is annotation
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        return any(_conforms(value, arg) for arg in args)
    if origin is typing.Literal:
        return any(type(value) is type(arg) and value == arg for arg in args)
    if origin is list:
        return type(value) is list and all(_conforms(item, args[0]) for item in value)
    if origin is collections.abc.Sequence:
        return type(value) in (list, tuple) and all(_conforms(item, args[0]) for item in value)
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return type(value) is tuple and all(_conforms(item, args[0]) for item in value)
        return (
            type(value) is tuple
            and len(value) == len(args)
            and all(_conforms(item, arg) for item, arg in zip(value, args, strict=True))
        )
    if origin in (dict, collections.abc.Mapping):
        return type(value) is dict and all(
            _conforms(k, args[0]) and _conforms(v, args[1]) for k, v in value.items()
        )
    raise NotImplementedError(f"cannot check {annotation!r}")


@pytest.mark.parametrize("codec_id", _params(set()))
def test_config_has_the_declared_types(codec_id: str) -> None:
    """`get_config()` returns what its TypedDict declares: the same keys, and values of
    the declared types. The codecs build their configuration untyped and `cast` it, so no
    type checker verifies this. (Codecs compiled from Cython carry no annotations at
    runtime; `test_stubs.py` checks those against their stubs.)"""
    codec = _codec(codec_id)
    declared: Any = typing.get_type_hints(type(codec).get_config).get("return")
    if not typing_extensions.is_typeddict(declared):
        assert type(codec).__module__ in {
            "numcodecs.blosc",
            "numcodecs.zstd",
            "numcodecs.lz4",
            "numcodecs.vlen",
            "numcodecs.fletcher32",
        }
        return
    config = codec.get_config()
    fields = typing.get_type_hints(declared)
    assert declared.__required_keys__ <= set(config) <= set(fields)
    wrong = {
        key: f"declared {fields[key]}, is {value!r}"
        for key, value in config.items()
        if not _conforms(value, fields[key])
    }
    assert not wrong


@pytest.mark.parametrize("codec_id", _params(set()))
def test_config_round_trip(codec_id: str) -> None:
    codec = _codec(codec_id)
    config = codec.get_config()
    assert config["id"] == codec.codec_id
    rebuilt_json = json.loads(json.dumps(config))
    assert numcodecs.get_codec(rebuilt_json) == codec
    without_id = {k: v for k, v in config.items() if k != "id"}
    assert type(codec).from_config(without_id) == codec
