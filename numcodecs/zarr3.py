"""
This module provides the compatibility for `numcodecs` in Zarr version 3.

A compatibility module is required because the codec handling in Zarr version 3 is different from Zarr version 2.

>>> import zarr
>>> import numcodecs.zarr3

>>> blosc_codec = numcodecs.zarr3.Blosc({"cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
>>> array = zarr.open("data.zarr", mode="w", shape=(1024, 1024), chunks=(64, 64), dtype="float32", codecs=[])

"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any, Self
from warnings import warn

import numpy as np

import numcodecs

try:
    import zarr

    if zarr.__version__ < "3.0.0":  # pragma: no cover
        raise ImportError("zarr 3.0.0 or later is required to use the numcodecs zarr integration.")
except ImportError:  # pragma: no cover
    raise ImportError("zarr 3.0.0 or later is required to use the numcodecs zarr integration.")

from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec
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
    return codec_name[len(CODEC_PREFIX) :]


def _parse_codec_configuration(data: dict[str, JSON]) -> dict[str, JSON]:
    parsed_name, parsed_configuration = parse_named_configuration(data)
    if not parsed_name.startswith(CODEC_PREFIX):
        raise ValueError(
            f"Expected name to start with '{CODEC_PREFIX}'. Got {parsed_name} instead."
        )  # pragma: no cover
    id = _expect_name_prefix(parsed_name)
    return {"id": id, **parsed_configuration}


@dataclass(frozen=True)
class NumcodecsCodec:
    codec_name: str
    codec_config: dict[str, JSON]

    def __init__(self, *, codec_config: dict[str, JSON] | None = None) -> None:
        if not self.codec_name:
            raise ValueError(
                "The codec name needs to be supplied through the `codec_name` attribute."
            )  # pragma: no cover
        unprefixed_codec_name = _expect_name_prefix(self.codec_name)

        if codec_config is None:
            codec_config = {}
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
        )

    @cached_property
    def _codec(self) -> numcodecs.abc.Codec:
        return numcodecs.get_codec(self.codec_config)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        codec_config = _parse_codec_configuration(data)
        return cls(codec_config=codec_config)

    def to_dict(self) -> JSON:
        codec_config = self.codec_config.copy()
        return {
            "name": self.codec_name,
            "configuration": codec_config,
        }

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        return input_byte_length  # pragma: no cover


class NumcodecsBytesBytesCodec(NumcodecsCodec, BytesBytesCodec):
    def __init__(self, *, codec_config: dict[str, JSON] | None = None) -> None:
        super().__init__(codec_config=codec_config)

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


class NumcodecsArrayArrayCodec(NumcodecsCodec, ArrayArrayCodec):
    def __init__(self, *, codec_config: dict[str, JSON]) -> None:
        super().__init__(codec_config=codec_config)

    async def _decode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        chunk_ndarray = chunk_array.as_ndarray_like()
        out = await asyncio.to_thread(self._codec.decode, chunk_ndarray)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out.reshape(chunk_spec.shape))

    async def _encode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        chunk_ndarray = chunk_array.as_ndarray_like()
        out = await asyncio.to_thread(self._codec.encode, chunk_ndarray)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out)


class NumcodecsArrayBytesCodec(NumcodecsCodec, ArrayBytesCodec):
    def __init__(self, *, codec_config: dict[str, JSON]) -> None:
        super().__init__(codec_config=codec_config)

    async def _decode_single(self, chunk_buffer: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        chunk_bytes = chunk_buffer.to_bytes()
        out = await asyncio.to_thread(self._codec.decode, chunk_bytes)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out.reshape(chunk_spec.shape))

    async def _encode_single(self, chunk_ndbuffer: NDBuffer, chunk_spec: ArraySpec) -> Buffer:
        chunk_ndarray = chunk_ndbuffer.as_ndarray_like()
        out = await asyncio.to_thread(self._codec.encode, chunk_ndarray)
        return chunk_spec.prototype.buffer.from_bytes(out)


def make_bytes_bytes_codec(codec_name: str, cls_name: str) -> type[NumcodecsBytesBytesCodec]:
    # rename for class scope
    _codec_name = CODEC_PREFIX + codec_name

    class _Codec(NumcodecsBytesBytesCodec):
        codec_name = _codec_name

        def __init__(self, codec_config: dict[str, JSON] | None = None) -> None:
            super().__init__(codec_config=codec_config)

    _Codec.__name__ = cls_name
    return _Codec


def make_array_array_codec(codec_name: str, cls_name: str) -> type[NumcodecsArrayArrayCodec]:
    # rename for class scope
    _codec_name = CODEC_PREFIX + codec_name

    class _Codec(NumcodecsArrayArrayCodec):
        codec_name = _codec_name

        def __init__(self, codec_config: dict[str, JSON] | None = None) -> None:
            super().__init__(codec_config=codec_config)

    _Codec.__name__ = cls_name
    return _Codec


def make_array_bytes_codec(codec_name: str, cls_name: str) -> type[NumcodecsArrayBytesCodec]:
    # rename for class scope
    _codec_name = CODEC_PREFIX + codec_name

    class _Codec(NumcodecsArrayBytesCodec):
        codec_name = _codec_name

        def __init__(self, codec_config: dict[str, JSON] | None = None) -> None:
            super().__init__(codec_config=codec_config)

    _Codec.__name__ = cls_name
    return _Codec


def make_checksum_codec(codec_name: str, cls_name: str) -> type[NumcodecsBytesBytesCodec]:
    # rename for class scope
    _codec_name = CODEC_PREFIX + codec_name

    class _ChecksumCodec(NumcodecsBytesBytesCodec):
        codec_name = _codec_name

        def __init__(self, codec_config: dict[str, JSON] | None = None) -> None:
            super().__init__(codec_config=codec_config)

        def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
            return input_byte_length + 4  # pragma: no cover

    _ChecksumCodec.__name__ = cls_name
    return _ChecksumCodec


class ShuffleCodec(NumcodecsBytesBytesCodec):
    codec_name = f"{CODEC_PREFIX}shuffle"

    def __init__(self, codec_config: dict[str, JSON] | None = None) -> None:
        super().__init__(codec_config=codec_config)

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        if array_spec.dtype.itemsize != self.codec_config.get("elementsize"):
            return self.__class__({**self.codec_config, "elementsize": array_spec.dtype.itemsize})
        return self  # pragma: no cover


class FixedScaleOffsetCodec(NumcodecsArrayArrayCodec):
    codec_name = f"{CODEC_PREFIX}fixedscaleoffset"

    def __init__(self, codec_config: dict[str, JSON] | None = None) -> None:
        super().__init__(codec_config=codec_config)

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        if astype := self.codec_config.get("astype"):
            return replace(chunk_spec, dtype=np.dtype(astype))
        return chunk_spec

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        if str(array_spec.dtype) != self.codec_config.get("dtype"):
            return self.__class__({**self.codec_config, "dtype": str(array_spec.dtype)})
        return self


class QuantizeCodec(NumcodecsArrayArrayCodec):
    codec_name = f"{CODEC_PREFIX}quantize"

    def __init__(self, codec_config: dict[str, JSON] | None = None) -> None:
        super().__init__(codec_config=codec_config)

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        if str(array_spec.dtype) != self.codec_config.get("dtype"):
            return self.__class__({**self.codec_config, "dtype": str(array_spec.dtype)})
        return self


class AsTypeCodec(NumcodecsArrayArrayCodec):
    codec_name = f"{CODEC_PREFIX}astype"

    def __init__(self, codec_config: dict[str, JSON] | None = None) -> None:
        super().__init__(codec_config=codec_config)

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        return replace(chunk_spec, dtype=np.dtype(self.codec_config["encode_dtype"]))

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        decode_dtype = self.codec_config.get("decode_dtype")
        if str(array_spec.dtype) != decode_dtype:
            return self.__class__({**self.codec_config, "decode_dtype": str(array_spec.dtype)})
        return self


class PackbitsCodec(NumcodecsArrayArrayCodec):
    codec_name = f"{CODEC_PREFIX}packbits"

    def __init__(self, codec_config: dict[str, JSON] | None = None) -> None:
        super().__init__(codec_config=codec_config)

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        return replace(
            chunk_spec,
            shape=(1 + math.ceil(product(chunk_spec.shape) / 8),),
            dtype=np.dtype("uint8"),
        )

    def validate(self, *, dtype: np.dtype[Any], **_kwargs) -> None:
        if dtype != np.dtype("bool"):
            raise ValueError(f"Packbits filter requires bool dtype. Got {dtype}.")


# bytes-to-bytes codecs
BloscCodec = make_bytes_bytes_codec("blosc", "BloscCodec")
LZ4Codec = make_bytes_bytes_codec("lz4", "LZ4Codec")
ZstdCodec = make_bytes_bytes_codec("zstd", "ZstdCodec")
ZlibCodec = make_bytes_bytes_codec("zlib", "ZlibCodec")
GZipCodec = make_bytes_bytes_codec("gzip", "GZipCodec")
BZ2Codec = make_bytes_bytes_codec("bz2", "BZ2Codec")
LZMACodec = make_bytes_bytes_codec("lzma", "LZMACodec")
# ShuffleCodec

# array-to-array codecs ("filters")
DeltaCodec = make_array_array_codec("delta", "DeltaCodec")
BitroundCodec = make_array_array_codec("bitround", "BitroundCodec")
# FixedScaleOffsetCodec
# QuantizeCodec
# PackbitsCodec
# AsTypeCodec

# bytes-to-bytes checksum codecs
Crc32Codec = make_checksum_codec("crc32", "Crc32Codec")
Crc32cCodec = make_checksum_codec("crc32c", "Crc32cCodec")
Adler32Codec = make_checksum_codec("adler32", "Adler32Codec")
Fletcher32Codec = make_checksum_codec("fletcher32", "Fletcher32Codec")
JenkinsLookup3Codec = make_checksum_codec("jenkins_lookup3", "JenkinsLookup3Codec")

# array-to-bytes codecs
PCodecCodec = make_array_bytes_codec("pcodec", "PCodecCodec")
ZFPYCodec = make_array_bytes_codec("zfpy", "ZFPYCodec")
