from __future__ import annotations
from typing import Iterator


import numpy as np
import pytest
import sys

from numcodecs.registry import get_codec

try:
    from zarr.codecs.registry import get_codec_class
    from zarr.array import Array
    from zarr.common import JSON
    from zarr.codecs import BytesCodec
    from zarr.abc.store import Store
    from zarr.store import MemoryStore, StorePath

except ImportError:
    pass


pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="zarr-python 3 requires Python 3.10 or higher"
)


@pytest.fixture
def store() -> Iterator[Store]:
    yield StorePath(MemoryStore(mode="w"))


@pytest.mark.parametrize(
    "codec_id", ["blosc", "lz4", "zstd", "zlib", "gzip", "bz2", "lzma", "shuffle"]
)
def test_generic_codec(store: Store, codec_id: str):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    with pytest.warns(UserWarning, match="Numcodecs.*"):
        a = Array.create(
            store / "generic",
            shape=data.shape,
            chunk_shape=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            codecs=[
                BytesCodec(),
                get_codec_class(f"numcodecs.{codec_id}")({"id": codec_id}),
            ],
        )

    a[:, :] = data.copy()
    assert np.array_equal(data, a[:, :])


@pytest.mark.parametrize(
    "codec_config",
    [
        {"id": "delta", "dtype": "float32"},
        {"id": "fixedscaleoffset", "offset": 0, "scale": 25.5},
        {"id": "fixedscaleoffset", "offset": 0, "scale": 51, "astype": "uint16"},
        {"id": "astype", "encode_dtype": "float32", "decode_dtype": "float64"},
    ],
    ids=[
        "delta",
        "fixedscaleoffset",
        "fixedscaleoffset2",
        "astype",
    ],
)
def test_generic_filter(store: Store, codec_config: dict[str, JSON]):
    data = np.linspace(0, 10, 256, dtype="float32").reshape((16, 16))

    codec_id = codec_config["id"]
    del codec_config["id"]

    with pytest.warns(UserWarning, match="Numcodecs.*"):
        a = Array.create(
            store / "generic",
            shape=data.shape,
            chunk_shape=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            codecs=[
                get_codec_class(f"numcodecs.{codec_id}")(codec_config),
                BytesCodec(),
            ],
        )

        a[:, :] = data.copy()
        a = Array.open(store / "generic")
    assert np.array_equal(data, a[:, :])


def test_generic_filter_bitround(store: Store):
    data = np.linspace(0, 1, 256, dtype="float32").reshape((16, 16))

    with pytest.warns(UserWarning, match="Numcodecs.*"):
        a = Array.create(
            store / "generic_bitround",
            shape=data.shape,
            chunk_shape=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            codecs=[
                get_codec_class("numcodecs.bitround")({"keepbits": 3}),
                BytesCodec(),
            ],
        )

        a[:, :] = data.copy()
        a = Array.open(store / "generic_bitround")
    assert np.allclose(data, a[:, :], atol=0.1)


def test_generic_filter_quantize(store: Store):
    data = np.linspace(0, 10, 256, dtype="float32").reshape((16, 16))

    with pytest.warns(UserWarning, match="Numcodecs.*"):
        a = Array.create(
            store / "generic_quantize",
            shape=data.shape,
            chunk_shape=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            codecs=[
                get_codec_class("numcodecs.quantize")({"digits": 3}),
                BytesCodec(),
            ],
        )

        a[:, :] = data.copy()
        a = Array.open(store / "generic_quantize")
    assert np.allclose(data, a[:, :], atol=0.001)


def test_generic_filter_packbits(store: Store):
    data = np.zeros((16, 16), dtype="bool")
    data[0:4, :] = True

    with pytest.warns(UserWarning, match="Numcodecs.*"):
        a = Array.create(
            store / "generic_packbits",
            shape=data.shape,
            chunk_shape=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            codecs=[
                get_codec_class("numcodecs.packbits")(),
                BytesCodec(),
            ],
        )

        a[:, :] = data.copy()
        a = Array.open(store / "generic_packbits")
    assert np.array_equal(data, a[:, :])


@pytest.mark.parametrize("codec_id", ["crc32", "adler32", "fletcher32", "jenkins_lookup3"])
def test_generic_checksum(store: Store, codec_id: str):
    data = np.linspace(0, 10, 256, dtype="float32").reshape((16, 16))

    with pytest.warns(UserWarning, match="Numcodecs.*"):
        a = Array.create(
            store / "generic_checksum",
            shape=data.shape,
            chunk_shape=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            codecs=[
                BytesCodec(),
                get_codec_class(f"numcodecs.{codec_id}")(),
            ],
        )

        a[:, :] = data.copy()
        a = Array.open(store / "generic_checksum")
    assert np.array_equal(data, a[:, :])


@pytest.mark.parametrize("codec_id", ["pcodec", "zfpy"])
def test_generic_bytes_codec(store: Store, codec_id: str):
    try:
        get_codec({"id": codec_id})
    except ValueError as e:
        if "codec not available" in str(e):
            pytest.xfail(f"{codec_id} is not available")
        else:
            raise

    data = np.arange(0, 256, dtype="float32").reshape((16, 16))

    with pytest.warns(UserWarning, match="Numcodecs.*"):
        a = Array.create(
            store / "generic",
            shape=data.shape,
            chunk_shape=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            codecs=[
                get_codec_class(f"numcodecs.{codec_id}")({"id": codec_id}),
            ],
        )

    a[:, :] = data.copy()
    assert np.array_equal(data, a[:, :])
