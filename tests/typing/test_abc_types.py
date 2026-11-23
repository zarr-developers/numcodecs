"""Static types of `numcodecs.abc`, checked against runtime behavior."""

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self, TypedDict, assert_type, overload

import numpy as np
import numpy.typing as npt
import pytest

import numcodecs
from numcodecs.abc import Codec, CodecProtocol, JSONValue
from numcodecs.compat import ensure_bytes, ndarray_copy
from numcodecs.ndarray_like import BufferLike, OutT, WritableBufferLike


def _zlib() -> Codec:
    return numcodecs.get_codec({"id": "zlib", "level": 1})


def test_codec_methods() -> None:
    codec = _zlib()
    data = np.arange(16, dtype="f8")

    assert_type(codec.codec_id, str)
    assert codec.codec_id == "zlib"

    enc = assert_type(codec.encode(data), BufferLike)
    dec = assert_type(codec.decode(enc), BufferLike)
    assert ensure_bytes(dec) == data.tobytes()

    out: npt.NDArray[np.float64] = np.empty_like(data)
    # through the base class the result is only known to be buffer-like
    assert assert_type(codec.decode(enc, out=out), BufferLike) is out
    buf = bytearray(data.nbytes)
    assert assert_type(codec.decode(enc, out=buf), BufferLike) is buf

    config = assert_type(codec.get_config(), Mapping[str, Any])
    assert config["id"] == "zlib"
    rebuilt = assert_type(type(codec).from_config({"level": 1}), Codec)
    assert rebuilt == codec

    assert_type(codec == object(), bool)


def test_codec_satisfies_codec_protocol() -> None:
    codec: CodecProtocol = _zlib()
    assert codec.codec_id == "zlib"


class _Structural:
    """Implements the codec interface without subclassing `Codec`, and types `out` and
    `config` more narrowly than `Codec` does."""

    codec_id: str = "test-structural"

    def encode(self, buf: BufferLike, /) -> bytes:
        return ensure_bytes(buf)

    def decode(self, buf: BufferLike, /, out: npt.NDArray[Any] | None = None) -> npt.NDArray[Any]:
        return np.frombuffer(ensure_bytes(buf), dtype="u1")

    def get_config(self) -> dict[str, Any]:
        return {"id": self.codec_id}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        return cls()


def test_codec_protocol_matches_structural_codecs() -> None:
    codec: CodecProtocol = _Structural()
    assert ensure_bytes(codec.decode(codec.encode(b"abc"))) == b"abc"
    # every numcodecs codec satisfies it too, through its overloads
    concrete: CodecProtocol = numcodecs.Zlib()
    assert concrete.get_config()["id"] == "zlib"


class _PropertyId:
    """A structural codec whose `codec_id` is a read-only property."""

    @property
    def codec_id(self) -> str:
        return "test-property-id"

    def encode(self, buf: BufferLike, /) -> bytes:
        return ensure_bytes(buf)

    def decode(self, buf: BufferLike, /, out: Any = None) -> bytes:
        return ensure_bytes(buf)

    def get_config(self) -> dict[str, Any]:
        return {"id": self.codec_id}

    @classmethod
    def from_config(cls, config: Any) -> Self:
        return cls()


def test_codec_protocol_matches_a_read_only_codec_id() -> None:
    codec: CodecProtocol = _PropertyId()
    assert codec.codec_id == "test-property-id"


def _consume(codec: CodecProtocol, level: int) -> CodecProtocol:
    """What code that accepts a `CodecProtocol` can call, and what it gets back."""
    data = np.arange(4, dtype="f8")
    assert_type(codec.codec_id, str)
    enc = assert_type(codec.encode(data), BufferLike)
    assert ensure_bytes(assert_type(codec.decode(enc), BufferLike)) == data.tobytes()
    out = np.empty_like(data)
    assert assert_type(codec.decode(enc, out=out), BufferLike) is out
    buf = bytearray(data.nbytes)
    assert assert_type(codec.decode(enc, buf), BufferLike) is buf
    assert_type(codec.get_config(), Mapping[str, Any])
    return assert_type(type(codec).from_config({"level": level}), CodecProtocol)


def test_codec_protocol_as_a_consumer_uses_it() -> None:
    codec = numcodecs.Zlib(level=1)
    assert _consume(codec, 1) == codec


class _Identity(Codec):
    """A third-party codec typed as precisely as numcodecs' own."""

    codec_id = "test-identity"

    def encode(self, buf: BufferLike, /) -> bytes:
        return ensure_bytes(buf)

    @overload
    def decode(self, buf: BufferLike, /, out: None = None) -> bytes: ...
    @overload
    def decode(self, buf: BufferLike, /, out: OutT) -> OutT: ...
    def decode(self, buf: BufferLike, /, out: WritableBufferLike | None = None) -> BufferLike:
        return ndarray_copy(ensure_bytes(buf), out)


class _Plain(Codec):
    """A third-party codec with ordinary annotations: the base class's own signature,
    under the parameter name `buf` as a regular (not positional-only) parameter."""

    codec_id = "test-plain"

    def encode(self, buf: BufferLike) -> BufferLike:
        return buf

    def decode(self, buf: BufferLike, out: WritableBufferLike | None = None) -> BufferLike:
        return buf


class _ConcreteReturn(Codec):
    """A third-party codec that declares a concrete return type and renames `buf`."""

    codec_id = "test-concrete"

    def encode(self, data: BufferLike) -> bytes:
        return ensure_bytes(data)

    def decode(self, data: BufferLike, out: WritableBufferLike | None = None) -> bytes:
        return ensure_bytes(data)


class _AnyTyped(Codec):
    """A third-party codec annotated with `Any`, as imagecodecs does."""

    codec_id = "test-any"

    def encode(self, buf: Any) -> Any:
        return buf

    def decode(self, buf: Any, out: Any = None) -> Any:
        return buf

    def get_config(self) -> dict[str, Any]:
        return {"id": self.codec_id}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        return cls()


class _ZlibSubclass(numcodecs.Zlib):
    """A known limit. A subclass of a concrete codec that overrides `decode` with one
    plain signature does not fit the two overloads it inherits, so both type checkers
    report the override. (Unannotated, pyright alone reports it, because it infers the
    return type.) Repeating the overloads, as `_Identity` does, avoids it. The ignore
    comments are checked, so this fails if the limit goes away."""

    codec_id = "test-zlib-subclass"

    def decode(self, buf: BufferLike, out: Any = None) -> bytes:  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        return super().decode(buf)


def test_subclasses() -> None:
    identity = _Identity()
    assert assert_type(identity.encode(b"abc"), bytes) == b"abc"
    assert assert_type(identity.decode(b"abc"), bytes) == b"abc"
    out = bytearray(3)
    assert assert_type(identity.decode(b"abc", out=out), bytearray) is out

    assert ensure_bytes(_Plain().decode(b"abc")) == b"abc"
    assert assert_type(_ConcreteReturn().decode(b"abc"), bytes) == b"abc"

    any_typed = _AnyTyped()
    assert any_typed.decode(any_typed.encode(b"abc")) == b"abc"
    assert_type(_AnyTyped.from_config({}), _AnyTyped)

    assert _ZlibSubclass().decode(numcodecs.Zlib().encode(b"abc")) == b"abc"


class _ZlibLikeConfig(TypedDict):
    level: int


def test_from_config_accepts_dicts_and_typeddicts() -> None:
    # codecs override from_config with a `dict` parameter (see _AnyTyped above), and callers
    # pass TypedDicts; both must type-check
    config: _ZlibLikeConfig = {"level": 3}
    codec = assert_type(numcodecs.Zlib.from_config(config), numcodecs.Zlib)
    assert codec == numcodecs.Zlib.from_config({"level": 3})


def test_out_must_be_writable() -> None:
    """`bytes` exports a buffer but is immutable, so it is not accepted as `out`. The
    ignore comments below are checked: if `bytes` became acceptable they would be unused,
    which both type checkers report."""
    codec = numcodecs.Zlib()
    enc = codec.encode(b"abc")
    with pytest.raises((TypeError, ValueError)):
        codec.decode(enc, out=b"xyz")  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    with pytest.raises((TypeError, ValueError)):
        _zlib().decode(enc, out=b"xyz")  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]


def test_out_with_any_in_its_type_keeps_its_type() -> None:
    """An `out` typed `NDArray[Any]` (what most numpy calls return) must come back as
    `NDArray[Any]`, not degrade to `Any`."""
    codec = numcodecs.Zlib()
    enc = codec.encode(np.arange(4, dtype="f8"))
    out: npt.NDArray[Any] = np.empty(4, dtype="f8")
    assert assert_type(codec.decode(enc, out=out), npt.NDArray[Any]) is out


def test_optional_out_gives_the_union() -> None:
    """Forwarding an optional `out` gives "decoded or `out`", nothing wider."""

    def decode(out: bytearray | None = None) -> bytes | bytearray:
        codec = numcodecs.Zlib()
        return assert_type(codec.decode(codec.encode(b"abc"), out), bytes | bytearray)

    assert decode() == b"abc"
    buf = bytearray(3)
    assert decode(buf) is buf


def _apply(func: Callable[[npt.NDArray[Any]], bytes], data: npt.NDArray[Any]) -> bytes:
    return func(data)


def test_decode_is_usable_as_a_plain_callable() -> None:
    """zarr-python passes `codec.decode` to a function typed like `_apply`."""
    codec = numcodecs.Zlib()
    enc = np.frombuffer(codec.encode(b"abc"), dtype="u1")
    assert _apply(codec.decode, enc) == b"abc"


def _json(value: JSONValue) -> JSONValue:
    return value


def test_json_value_covers_every_json_kind() -> None:
    """`JSONValue` takes each kind of JSON value, with tuples and other read-only
    containers standing in for arrays and objects."""
    pair: tuple[int, ...] = (1, 2)
    read_only: Mapping[str, Sequence[float]] = {"a": [1.5]}
    values: list[JSONValue] = [
        _json(None),
        _json(True),
        _json(1),
        _json(1.5),
        _json("a"),
        _json([1, "a", None]),
        _json(pair),
        _json({"a": {"b": [1.5]}}),
        _json(read_only),
    ]
    assert json.loads(json.dumps(values)) == [
        None,
        True,
        1,
        1.5,
        "a",
        [1, "a", None],
        [1, 2],
        {"a": {"b": [1.5]}},
        {"a": [1.5]},
    ]
    # a known hole: `bytes` is a `Sequence[int]` to type checkers, so it is accepted
    with pytest.raises(TypeError):
        json.dumps(_json(b"bytes"))
    with pytest.raises(TypeError):
        json.dumps(_json({1, 2}))  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
