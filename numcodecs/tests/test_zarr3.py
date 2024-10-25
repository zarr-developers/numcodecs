from __future__ import annotations

import numpy as np
import pytest

from numcodecs.registry import get_codec

zarr = pytest.importorskip("zarr")

pytestmark = pytest.mark.skipif(
    zarr.__version__ < "3.0.0", reason="zarr 3.0.0 or later is required"
)

get_codec_class = zarr.registry.get_codec_class
Array = zarr.Array
JSON = zarr.core.common.JSON
BytesCodec = zarr.codecs.BytesCodec
Store = zarr.abc.store.Store
MemoryStore = zarr.storage.MemoryStore
StorePath = zarr.storage.StorePath


EXPECTED_WARNING_STR = "Numcodecs codecs are not in the Zarr version 3.*"


@pytest.fixture
def store() -> Store:
    return StorePath(MemoryStore(mode="w"))


@pytest.mark.parametrize(
    "codec_id", ["blosc", "lz4", "zstd", "zlib", "gzip", "bz2", "lzma", "shuffle"]
)
def test_generic_codec(store: Store, codec_id: str):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    with pytest.warns(UserWarning, match=EXPECTED_WARNING_STR):
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
    np.testing.assert_array_equal(data, a[:, :])


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

    with pytest.warns(UserWarning, match=EXPECTED_WARNING_STR):
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
    np.testing.assert_array_equal(data, a[:, :])


def test_generic_filter_bitround(store: Store):
    data = np.linspace(0, 1, 256, dtype="float32").reshape((16, 16))

    with pytest.warns(UserWarning, match=EXPECTED_WARNING_STR):
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

    with pytest.warns(UserWarning, match=EXPECTED_WARNING_STR):
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

    with pytest.warns(UserWarning, match=EXPECTED_WARNING_STR):
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
    np.testing.assert_array_equal(data, a[:, :])

    with pytest.raises(ValueError, match="packbits filter requires bool dtype"):
        Array.create(
            store / "generic_packbits_err",
            shape=data.shape,
            chunk_shape=(16, 16),
            dtype="uint32",
            fill_value=0,
            codecs=[
                get_codec_class("numcodecs.packbits")(),
                BytesCodec(),
            ],
        )


@pytest.mark.parametrize("codec_id", ["crc32", "adler32", "fletcher32", "jenkins_lookup3"])
def test_generic_checksum(store: Store, codec_id: str):
    data = np.linspace(0, 10, 256, dtype="float32").reshape((16, 16))

    with pytest.warns(UserWarning, match=EXPECTED_WARNING_STR):
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
    np.testing.assert_array_equal(data, a[:, :])


@pytest.mark.parametrize("codec_id", ["pcodec", "zfpy"])
def test_generic_bytes_codec(store: Store, codec_id: str):
    try:
        get_codec({"id": codec_id})
    except ValueError as e:
        if "codec not available" in str(e):
            pytest.xfail(f"{codec_id} is not available: {e}")
        else:
            raise  # pragma: no cover
    except ImportError as e:
        pytest.xfail(f"{codec_id} is not available: {e}")

    data = np.arange(0, 256, dtype="float32").reshape((16, 16))

    with pytest.warns(UserWarning, match=EXPECTED_WARNING_STR):
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
    np.testing.assert_array_equal(data, a[:, :])
