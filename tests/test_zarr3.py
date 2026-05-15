from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:  # pragma: no cover
    import zarr
else:
    zarr = pytest.importorskip("zarr", "3.1.3")

import numcodecs.zarr3 as zarr3

codec_names = [
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


@pytest.mark.parametrize('codec_name', codec_names)
def test_export(codec_name: str) -> None:
    """
    Ensure that numcodecs.zarr3 re-exports codecs defined in zarr.codecs.numcodecs
    """
    with pytest.warns(
        DeprecationWarning,
        match="The numcodecs.zarr3 module is deprecated and will be removed in a future release of numcodecs. ",
    ):
        assert getattr(zarr3, codec_name) == getattr(zarr.codecs.numcodecs, codec_name)
