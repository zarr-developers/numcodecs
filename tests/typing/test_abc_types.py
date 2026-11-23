"""Static types of `numcodecs.abc`, checked against runtime behavior."""

from collections.abc import Callable, Mapping
from typing import Any, Self, TypedDict, assert_type, overload

import numpy as np
import numpy.typing as npt
import pytest

import numcodecs
from numcodecs.abc import Codec, CodecProtocol
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


class _Identity(Codec):
    """A third-party codec typed as precisely as numcodecs' own."""

    codec_id = "test-identity"

    def encode(self, buf: BufferLike, /) -> bytes:
        return ensure_bytes(buf)

    @overload
    def decode(self, buf: BufferLike, /, out: OutT) -> OutT: ...
    @overload
    def decode(self, buf: BufferLike, /, out: None = None) -> bytes: ...
    @overload
    def decode(
        self, buf: BufferLike, /, out: WritableBufferLike | None = None
    ) -> bytes | WritableBufferLike: ...
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
        _zlib().decode(enc, out=b"xyz")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def test_any_typed_out_is_not_mistaken_for_no_out() -> None:
    def decode_into(out: Any) -> None:
        # with `out: Any` the result may be `out`, so it must not be typed as bytes
        result = assert_type(numcodecs.Zlib().decode(numcodecs.Zlib().encode(b"abc"), out), Any)
        assert result is out

    decode_into(bytearray(3))


def _apply(func: Callable[[npt.NDArray[Any]], bytes], data: npt.NDArray[Any]) -> bytes:
    return func(data)


def test_decode_is_usable_as_a_plain_callable() -> None:
    """zarr-python passes `codec.decode` to a function typed like `_apply`."""
    codec = numcodecs.Zlib()
    enc = np.frombuffer(codec.encode(b"abc"), dtype="u1")
    assert _apply(codec.decode, enc) == b"abc"
