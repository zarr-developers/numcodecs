"""This module defines the :class:`Codec` base class, a common interface for
all codec classes.

Codec classes must implement :func:`Codec.encode` and :func:`Codec.decode`
methods. Inputs to and outputs from these methods may be any Python object
exporting a contiguous buffer via the new-style Python protocol.

Codec classes must implement a :func:`Codec.get_config` method,
which must return a dictionary holding all configuration parameters
required to enable encoding and decoding of data. The expectation is that
these configuration parameters will be stored or communicated separately
from encoded data, and thus the codecs do not need to store all encoding
parameters within the encoded data. For broad compatibility,
the configuration object must contain only JSON-serializable values. The
configuration object must also contain an 'id' field storing the codec
identifier (see below).

Codec classes must implement a :func:`Codec.from_config` class method,
which will return an instance of the class initialized from a configuration
object.

Finally, codec classes must set a `codec_id` class-level attribute. This
must be a string. Two different codec classes may set the same value for the
`codec_id` attribute if and only if they are fully compatible, meaning that
(1) configuration parameters are the same, and (2) given the same
configuration, one class could correctly decode data encoded by the
other and vice versa.

"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, Self, overload

from typing_extensions import ReadOnly, TypedDict

from .ndarray_like import BufferLike, WritableBufferLike

__all__ = [
    "Codec",
    "CodecConfig",
    "CodecProtocol",
    "JSONValue",
]

type JSONValue = int | float | bool | str | Sequence[JSONValue] | Mapping[str, JSONValue] | None
"""A JSON-serializable value. `Sequence` and `Mapping` rather than `list` and `dict`, so
that tuples, TypedDicts and lists of narrower types count (the same definition as
`zarr_metadata`'s `JSONValue`)."""


# `extra_items` (PEP 728) says that any further keys hold JSON values and are read-only. It
# lets type checkers that implement PEP 728 (pyright) treat every codec configuration as a
# `Mapping[str, JSONValue]`, and as Zarr v2 codec metadata. mypy does not implement PEP 728 yet,
# so it rejects the keyword and treats `CodecConfig` as a plain TypedDict.
class CodecConfig(TypedDict, extra_items=ReadOnly[JSONValue]):  # type: ignore[call-arg]
    """The configuration every codec returns from `Codec.get_config`: its `id`, plus any
    further JSON-serializable parameters.

    Concrete codecs return a subclass that declares their own parameters. This is the shape
    of a Zarr version 2 compressor or filter.
    """

    id: str


def _codec_id_or_none(codec: object) -> str | None:
    """Return `codec.codec_id`, or None for a codec whose class never set one.

    `Codec.codec_id` used to default to None on the class. It no longer does, but a codec
    that never set it keeps reporting None where numcodecs itself reads the id. An
    `AttributeError` raised inside a `codec_id` property is not that case and propagates.
    """
    try:
        return codec.codec_id  # type: ignore[attr-defined, no-any-return]
    except AttributeError:
        if hasattr(type(codec), 'codec_id') or 'codec_id' in getattr(codec, '__dict__', {}):
            raise
        return None


class Codec(ABC):
    """Codec abstract base class."""

    # Declared without a value (abstract intermediate classes leave it unset) and without
    # `ClassVar`, so that `Codec` instances match protocols that declare `codec_id` either
    # as a plain attribute or as a read-only property.
    codec_id: str
    """Codec identifier. Every concrete codec class sets it as a class attribute."""

    @abstractmethod
    def encode(self, buf: BufferLike, /) -> BufferLike:  # pragma: no cover
        """Encode data in `buf`.

        Parameters
        ----------
        buf : buffer-like
            Data to be encoded. May be any object supporting the new-style
            buffer protocol.

        Returns
        -------
        enc : buffer-like
            Encoded data. May be any object supporting the new-style buffer
            protocol.
        """

    # Both overloads return `BufferLike`: the base class says nothing about which object
    # comes back, so a subclass may annotate `decode` with one plain signature. (If the
    # second overload returned the type of `out`, mypy would reject every subclass that
    # did not repeat the overloads.) numcodecs' own codecs narrow both: what they decode
    # to without `out`, and the type of `out` with it.
    @overload
    def decode(self, buf: BufferLike, /, out: None = None) -> BufferLike: ...
    @overload
    def decode(self, buf: BufferLike, /, out: WritableBufferLike) -> BufferLike: ...
    @abstractmethod
    def decode(
        self, buf: BufferLike, /, out: WritableBufferLike | None = None
    ) -> BufferLike:  # pragma: no cover
        """Decode data in `buf`.

        Parameters
        ----------
        buf : buffer-like
            Encoded data. May be any object supporting the new-style buffer
            protocol.
        out : buffer-like, optional
            Writable buffer to store decoded data. N.B. if provided, this buffer must
            be exactly the right size to store the decoded data.

        Returns
        -------
        dec : buffer-like
            Decoded data. May be any object supporting the new-style
            buffer protocol. If `out` was provided, `out` itself.
        """

    def get_config(self) -> Mapping[str, Any]:
        """Return a dictionary holding configuration parameters for this
        codec. Must include an 'id' field with the codec identifier. All
        values must be compatible with JSON encoding."""

        # override in sub-class if need special encoding of config values

        # setup config object
        config: dict[str, Any] = {'id': _codec_id_or_none(self)}

        # by default, assume all non-private members are configuration
        # parameters - override this in sub-class if not the case
        for k in self.__dict__:
            if not k.startswith('_'):
                config[k] = getattr(self, k)

        return config

    # `config` is typed `Any`, like the input of pydantic's `model_validate`. No narrower
    # type works for everyone: codecs that override this (to decode config values) declare
    # `dict[str, Any]`, which rules out `Mapping` here, and callers pass TypedDicts, which
    # are not dicts to a type checker.
    @classmethod
    def from_config(cls, config: Any) -> Self:
        """Instantiate codec from a configuration object, without its 'id' field."""
        # N.B., assume at this point the 'id' field has been removed from
        # the config object

        # override in sub-class if need special decoding of config values

        # by default, assume constructor accepts configuration parameters as
        # keyword arguments without any special decoding
        return cls(**config)

    def __eq__(self, other: object) -> bool:
        # override in sub-class if need special equality comparison
        try:
            return bool(self.get_config() == other.get_config())  # type: ignore[attr-defined]
        except AttributeError:
            return False

    def __repr__(self) -> str:
        # override in sub-class if need special representation

        # by default, assume all non-private members are configuration
        # parameters and valid keyword arguments to constructor function

        r = f'{type(self).__name__}('
        params = [
            f'{k}={getattr(self, k)!r}' for k in sorted(self.__dict__) if not k.startswith('_')
        ]
        r += ', '.join(params) + ')'
        return r


class CodecProtocol(Protocol):
    """The structural interface of a codec, for code that accepts any object implementing
    it, whether or not it subclasses `Codec`."""

    @property
    def codec_id(self) -> str: ...  # pragma: no cover

    def encode(self, buf: BufferLike, /) -> BufferLike: ...  # pragma: no cover

    @overload
    def decode(self, buf: BufferLike, /, out: None = None) -> BufferLike: ...
    @overload
    def decode(self, buf: BufferLike, /, out: WritableBufferLike) -> BufferLike: ...

    def get_config(self) -> Mapping[str, Any]: ...  # pragma: no cover

    @classmethod
    def from_config(cls, config: Any) -> Self: ...  # pragma: no cover
