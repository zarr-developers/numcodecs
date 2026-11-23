"""Static types of `numcodecs.abc`, checked against runtime behavior."""

from collections.abc import Mapping
from typing import Any, Self, assert_type, overload

import numpy as np
import numpy.typing as npt

import numcodecs
from numcodecs.abc import Codec, CodecProtocol
from numcodecs.compat import ensure_bytes, ndarray_copy
from numcodecs.ndarray_like import BufferLike


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
    assert assert_type(codec.decode(enc, out=out), npt.NDArray[np.float64]) is out
    buf = bytearray(data.nbytes)
    assert assert_type(codec.decode(enc, out=buf), bytearray) is buf

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
    def decode(self, buf: BufferLike, /, out: None = None) -> bytes: ...
    @overload
    def decode[OutT: BufferLike](self, buf: BufferLike, /, out: OutT) -> OutT: ...
    def decode(self, buf: BufferLike, /, out: BufferLike | None = None) -> BufferLike:
        return ndarray_copy(ensure_bytes(buf), out)


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
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        return cls()


def test_subclasses() -> None:
    identity = _Identity()
    assert assert_type(identity.encode(b"abc"), bytes) == b"abc"
    assert assert_type(identity.decode(b"abc"), bytes) == b"abc"
    out = bytearray(3)
    assert assert_type(identity.decode(b"abc", out=out), bytearray) is out

    any_typed = _AnyTyped()
    assert any_typed.decode(any_typed.encode(b"abc")) == b"abc"
    assert_type(_AnyTyped.from_config({}), _AnyTyped)
