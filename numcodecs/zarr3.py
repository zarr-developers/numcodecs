"""
This module is DEPRECATED. It will may be removed entirely in a future release of Numcodecs.
The codecs exported here are available in Zarr Python >= 3.1.3
"""

from __future__ import annotations

from importlib.metadata import version

from packaging.version import Version

try:
    import zarr  # noqa: F401

    zarr_version = version('zarr')
    if Version(zarr_version) < Version("3.1.3"):  # pragma: no cover
        msg = f"zarr 3.1.3 or later is required to use the numcodecs zarr integration. Got {zarr_version}."
        raise ImportError(msg)
except ImportError as e:  # pragma: no cover
    msg = "zarr could not be imported. Zarr 3.1.3 or later is required to use the numcodecs zarr integration."
    raise ImportError(msg) from e

from zarr.codecs.numcodecs import (
    BZ2,
    CRC32,
    CRC32C,
    LZ4,
    LZMA,
    ZFPY,
    Adler32,
    AsType,
    BitRound,
    Blosc,
    Delta,
    FixedScaleOffset,
    Fletcher32,
    GZip,
    JenkinsLookup3,
    PackBits,
    PCodec,
    Quantize,
    Shuffle,
    Zlib,
    Zstd,
)

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
