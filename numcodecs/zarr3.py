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

from importlib.metadata import version

from packaging.version import Version

try:
    import zarr  # noqa: F401

    zarr_version = version('zarr')
    if Version(zarr_version) < Version("3.0.8"):  # pragma: no cover
        msg = f"zarr 3.0.9 or later is required to use the numcodecs zarr integration. Got {zarr_version}."
        raise ImportError(msg)
except ImportError as e:  # pragma: no cover
    msg = "zarr could not be imported. Zarr 3.1.0 or later is required to use the numcodecs zarr integration."
    raise ImportError(msg) from e

from zarr.codecs._numcodecs import (
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
