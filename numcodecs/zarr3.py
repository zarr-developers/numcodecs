"""
This module provides the compatibility for :py:mod:`numcodecs` in Zarr version 3.

A compatibility module is required because the codec handling in Zarr version 3 is different from Zarr version 2.

You can use codecs from :py:mod:`numcodecs` by constructing codecs from :py:mod:`numcodecs.zarr3` using the same parameters as the original codecs.

>>> import zarr
>>> import numcodecs.zarr3
>>>
>>> array = zarr.create_array(
...   store="data.zarr",
...   shape=(1024, 1024),
...   chunks=(64, 64),
...   dtype="uint32",
...   filters=[numcodecs.zarr3.Delta()],
...   compressors=[numcodecs.zarr3.BZ2(level=5)])
>>> array[:] = np.arange(*array.shape).astype(array.dtype)

.. note::

    Please note that the codecs in :py:mod:`numcodecs.zarr3` are not part of the Zarr version 3 specification.
    Using these codecs might cause interoperability issues with other Zarr implementations.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, replace
from functools import cached_property, partial
from typing import Any, Self, TypeVar
from warnings import warn

import numpy as np

import numcodecs

try:
    import zarr

    if zarr.__version__ < "3.0.0":  # pragma: no cover
        raise ImportError("zarr 3.0.0 or later is required to use the numcodecs zarr integration.")
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "zarr 3.0.0 or later is required to use the numcodecs zarr integration."
    ) from e

from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec
from zarr.abc.metadata import Metadata
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer, BufferPrototype, NDBuffer
from zarr.core.buffer.cpu import as_numpy_array_wrapper
from zarr.core.common import JSON, parse_named_configuration, product

CODEC_PREFIX = "numcodecs."


def _expect_name_prefix(codec_name: str) -> str:
    if not codec_name.startswith(CODEC_PREFIX):
        raise ValueError(
            f"Expected name to start with '{CODEC_PREFIX}'. Got {codec_name} instead."
        )  # pragma: no cover
    return codec_name.removeprefix(CODEC_PREFIX)


def _parse_codec_configuration(data: dict[str, JSON]) -> dict[str, JSON]:
    parsed_name, parsed_configuration = parse_named_configuration(data)
    if not parsed_name.startswith(CODEC_PREFIX):
        raise ValueError(
            f"Expected name to start with '{CODEC_PREFIX}'. Got {parsed_name} instead."
        )  # pragma: no cover
    id = _expect_name_prefix(parsed_name)
    return {"id": id, **parsed_configuration}


@dataclass(frozen=True)
class _NumcodecsCodec(Metadata):
    codec_name: str
    codec_config: dict[str, JSON]

    def __init__(self, **codec_config: JSON) -> None:
        if not self.codec_name:
            raise ValueError(
                "The codec name needs to be supplied through the `codec_name` attribute."
            )  # pragma: no cover
        unprefixed_codec_name = _expect_name_prefix(self.codec_name)

        if "id" not in codec_config:
            codec_config = {"id": unprefixed_codec_name, **codec_config}
        elif codec_config["id"] != unprefixed_codec_name:
            raise ValueError(
                f"Codec id does not match {unprefixed_codec_name}. Got: {codec_config['id']}."
            )  # pragma: no cover

        object.__setattr__(self, "codec_config", codec_config)
        warn(
            "Numcodecs codecs are not in the Zarr version 3 specification and "
            "may not be supported by other zarr implementations.",
            category=UserWarning,
            stacklevel=2,
        )

    @cached_property
    def _codec(self) -> numcodecs.abc.Codec:
        return numcodecs.get_codec(self.codec_config)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        codec_config = _parse_codec_configuration(data)
        return cls(**codec_config)

    def to_dict(self) -> dict[str, JSON]:
        codec_config = self.codec_config.copy()
        codec_config.pop("id", None)
        return {
            "name": self.codec_name,
            "configuration": codec_config,
        }

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError  # pragma: no cover

    # Override __repr__ because dynamically constructed classes don't seem to work otherwise
    def __repr__(self) -> str:
        codec_config = self.codec_config.copy()
        codec_config.pop("id", None)
        return f"{self.__class__.__name__}(codec_name={self.codec_name!r}, codec_config={codec_config!r})"


class _NumcodecsBytesBytesCodec(_NumcodecsCodec, BytesBytesCodec):
    def __init__(self, **codec_config: JSON) -> None:
        super().__init__(**codec_config)

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return await asyncio.to_thread(
            as_numpy_array_wrapper,
            self._codec.decode,
            chunk_bytes,
            chunk_spec.prototype,
        )

    def _encode(self, chunk_bytes: Buffer, prototype: BufferPrototype) -> Buffer:
        encoded = self._codec.encode(chunk_bytes.as_array_like())
        if isinstance(encoded, np.ndarray):  # Required for checksum codecs
            return prototype.buffer.from_bytes(encoded.tobytes())
        return prototype.buffer.from_bytes(encoded)

    async def _encode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return await asyncio.to_thread(self._encode, chunk_bytes, chunk_spec.prototype)


class _NumcodecsArrayArrayCodec(_NumcodecsCodec, ArrayArrayCodec):
    def __init__(self, **codec_config: JSON) -> None:
        super().__init__(**codec_config)

    async def _decode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        chunk_ndarray = chunk_array.as_ndarray_like()
        out = await asyncio.to_thread(self._codec.decode, chunk_ndarray)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out.reshape(chunk_spec.shape))

    async def _encode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        chunk_ndarray = chunk_array.as_ndarray_like()
        out = await asyncio.to_thread(self._codec.encode, chunk_ndarray)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out)


class _NumcodecsArrayBytesCodec(_NumcodecsCodec, ArrayBytesCodec):
    def __init__(self, **codec_config: JSON) -> None:
        super().__init__(**codec_config)

    async def _decode_single(self, chunk_buffer: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        chunk_bytes = chunk_buffer.to_bytes()
        out = await asyncio.to_thread(self._codec.decode, chunk_bytes)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out.reshape(chunk_spec.shape))

    async def _encode_single(self, chunk_ndbuffer: NDBuffer, chunk_spec: ArraySpec) -> Buffer:
        chunk_ndarray = chunk_ndbuffer.as_ndarray_like()
        out = await asyncio.to_thread(self._codec.encode, chunk_ndarray)
        return chunk_spec.prototype.buffer.from_bytes(out)


T = TypeVar("T", bound=_NumcodecsCodec)


def _add_docstring(cls: type[T], ref_class_name: str) -> type[T]:
    cls.__doc__ = f"""
        See :class:`{ref_class_name}` for more details and parameters.
        """
    return cls


def _add_docstring_wrapper(ref_class_name: str) -> partial:
    return partial(_add_docstring, ref_class_name=ref_class_name)


def _make_bytes_bytes_codec(codec_name: str, cls_name: str) -> type[_NumcodecsBytesBytesCodec]:
    # rename for class scope
    _codec_name = CODEC_PREFIX + codec_name

    class _Codec(_NumcodecsBytesBytesCodec):
        codec_name = _codec_name

        def __init__(self, **codec_config: JSON) -> None:
            super().__init__(**codec_config)

    _Codec.__name__ = cls_name
    return _Codec


def _make_array_array_codec(codec_name: str, cls_name: str) -> type[_NumcodecsArrayArrayCodec]:
    # rename for class scope
    _codec_name = CODEC_PREFIX + codec_name

    class _Codec(_NumcodecsArrayArrayCodec):
        codec_name = _codec_name

        def __init__(self, **codec_config: JSON) -> None:
            super().__init__(**codec_config)

    _Codec.__name__ = cls_name
    return _Codec


def _make_array_bytes_codec(codec_name: str, cls_name: str) -> type[_NumcodecsArrayBytesCodec]:
    # rename for class scope
    _codec_name = CODEC_PREFIX + codec_name

    class _Codec(_NumcodecsArrayBytesCodec):
        codec_name = _codec_name

        def __init__(self, **codec_config: JSON) -> None:
            super().__init__(**codec_config)

    _Codec.__name__ = cls_name
    return _Codec


def _make_checksum_codec(codec_name: str, cls_name: str) -> type[_NumcodecsBytesBytesCodec]:
    # rename for class scope
    _codec_name = CODEC_PREFIX + codec_name

    class _ChecksumCodec(_NumcodecsBytesBytesCodec):
        codec_name = _codec_name

        def __init__(self, **codec_config: JSON) -> None:
            super().__init__(**codec_config)

        def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
            return input_byte_length + 4  # pragma: no cover

    _ChecksumCodec.__name__ = cls_name
    return _ChecksumCodec


# bytes-to-bytes codecs
Blosc = _add_docstring(_make_bytes_bytes_codec("blosc", "Blosc"), "numcodecs.blosc.Blosc")
LZ4 = _add_docstring(_make_bytes_bytes_codec("lz4", "LZ4"), "numcodecs.lz4.LZ4")
Zstd = _add_docstring(_make_bytes_bytes_codec("zstd", "Zstd"), "numcodecs.zstd.Zstd")
Zlib = _add_docstring(_make_bytes_bytes_codec("zlib", "Zlib"), "numcodecs.zlib.Zlib")
GZip = _add_docstring(_make_bytes_bytes_codec("gzip", "GZip"), "numcodecs.gzip.GZip")
BZ2 = _add_docstring(_make_bytes_bytes_codec("bz2", "BZ2"), "numcodecs.bz2.BZ2")
LZMA = _add_docstring(_make_bytes_bytes_codec("lzma", "LZMA"), "numcodecs.lzma.LZMA")


@_add_docstring_wrapper("numcodecs.shuffle.Shuffle")
class Shuffle(_NumcodecsBytesBytesCodec):
    codec_name = f"{CODEC_PREFIX}shuffle"

    def __init__(self, **codec_config: JSON) -> None:
        super().__init__(**codec_config)

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Shuffle:
        if self.codec_config.get("elementsize", None) is None:
            return Shuffle(**{**self.codec_config, "elementsize": array_spec.dtype.itemsize})
        return self  # pragma: no cover


# array-to-array codecs ("filters")
@_add_docstring_wrapper("numcodecs.delta.Delta")
class Delta(_NumcodecsArrayArrayCodec):
    codec_name = f"{CODEC_PREFIX}delta"

    def __init__(self, **codec_config: dict[str, JSON]) -> None:
        super().__init__(**codec_config)

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        if astype := self.codec_config.get("astype"):
            return replace(chunk_spec, dtype=np.dtype(astype))  # type: ignore[call-overload]
        return chunk_spec


BitRound = _add_docstring(
    _make_array_array_codec("bitround", "BitRound"), "numcodecs.bitround.BitRound"
)


@_add_docstring_wrapper("numcodecs.fixedscaleoffset.FixedScaleOffset")
class FixedScaleOffset(_NumcodecsArrayArrayCodec):
    codec_name = f"{CODEC_PREFIX}fixedscaleoffset"

    def __init__(self, **codec_config: JSON) -> None:
        super().__init__(**codec_config)

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        if astype := self.codec_config.get("astype"):
            return replace(chunk_spec, dtype=np.dtype(astype))  # type: ignore[call-overload]
        return chunk_spec

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> FixedScaleOffset:
        if self.codec_config.get("dtype", None) is None:
            return FixedScaleOffset(**{**self.codec_config, "dtype": str(array_spec.dtype)})
        return self


@_add_docstring_wrapper("numcodecs.quantize.Quantize")
class Quantize(_NumcodecsArrayArrayCodec):
    codec_name = f"{CODEC_PREFIX}quantize"

    def __init__(self, **codec_config: JSON) -> None:
        super().__init__(**codec_config)

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Quantize:
        if self.codec_config.get("dtype", None) is None:
            return Quantize(**{**self.codec_config, "dtype": str(array_spec.dtype)})
        return self


@_add_docstring_wrapper("numcodecs.packbits.PackBits")
class PackBits(_NumcodecsArrayArrayCodec):
    codec_name = f"{CODEC_PREFIX}packbits"

    def __init__(self, **codec_config: JSON) -> None:
        super().__init__(**codec_config)

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        return replace(
            chunk_spec,
            shape=(1 + math.ceil(product(chunk_spec.shape) / 8),),
            dtype=np.dtype("uint8"),
        )

    def validate(self, *, dtype: np.dtype[Any], **_kwargs) -> None:
        if dtype != np.dtype("bool"):
            raise ValueError(f"Packbits filter requires bool dtype. Got {dtype}.")


@_add_docstring_wrapper("numcodecs.astype.AsType")
class AsType(_NumcodecsArrayArrayCodec):
    codec_name = f"{CODEC_PREFIX}astype"

    def __init__(self, **codec_config: JSON) -> None:
        super().__init__(**codec_config)

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        return replace(chunk_spec, dtype=np.dtype(self.codec_config["encode_dtype"]))  # type: ignore[arg-type]

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> AsType:
        if self.codec_config.get("decode_dtype", None) is None:
            return AsType(**{**self.codec_config, "decode_dtype": str(array_spec.dtype)})
        return self


# bytes-to-bytes checksum codecs
CRC32 = _add_docstring(_make_checksum_codec("crc32", "CRC32"), "numcodecs.checksum32.CRC32")
CRC32C = _add_docstring(_make_checksum_codec("crc32c", "CRC32C"), "numcodecs.checksum32.CRC32C")
Adler32 = _add_docstring(_make_checksum_codec("adler32", "Adler32"), "numcodecs.checksum32.Adler32")
Fletcher32 = _add_docstring(
    _make_checksum_codec("fletcher32", "Fletcher32"), "numcodecs.fletcher32.Fletcher32"
)
JenkinsLookup3 = _add_docstring(
    _make_checksum_codec("jenkins_lookup3", "JenkinsLookup3"), "numcodecs.checksum32.JenkinsLookup3"
)

# array-to-bytes codecs
PCodec = _add_docstring(_make_array_bytes_codec("pcodec", "PCodec"), "numcodecs.pcodec.PCodec")
ZFPY = _add_docstring(_make_array_bytes_codec("zfpy", "ZFPY"), "numcodecs.zfpy.ZFPY")

__all__ = [
    "BZ2",
    "CRC32",
    "CRC32C",
    "LZ4",
    "LZMA",
    "ZFPY",
    "Adler32",
    "AsType",
    "BitRound",
    "Blosc",
    "Delta",
    "FixedScaleOffset",
    "Fletcher32",
    "GZip",
    "JenkinsLookup3",
    "PCodec",
    "PackBits",
    "Quantize",
    "Shuffle",
    "Zlib",
    "Zstd",
]
