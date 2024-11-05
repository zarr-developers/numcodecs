from __future__ import annotations

import numpy as np
import pytest

import numcodecs.zarr3

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


ALL_CODECS = [getattr(numcodecs.zarr3, cls_name) for cls_name in numcodecs.zarr3.__all__]


@pytest.mark.parametrize("codec_class", ALL_CODECS)
def test_entry_points(codec_class: type[numcodecs.zarr3._NumcodecsCodec]):
    codec_name = codec_class.codec_name
    assert get_codec_class(codec_name) == codec_class


@pytest.mark.parametrize("codec_class", ALL_CODECS)
def test_docstring(codec_class: type[numcodecs.zarr3._NumcodecsCodec]):
    assert "See :class:`numcodecs." in codec_class.__doc__


@pytest.mark.parametrize(
    "codec_class",
    [
        numcodecs.zarr3.Blosc,
        numcodecs.zarr3.LZ4,
        numcodecs.zarr3.Zstd,
        numcodecs.zarr3.Zlib,
        numcodecs.zarr3.GZip,
        numcodecs.zarr3.BZ2,
        numcodecs.zarr3.LZMA,
        numcodecs.zarr3.Shuffle,
    ],
)
def test_generic_codec_class(store: Store, codec_class: type[numcodecs.zarr3._NumcodecsCodec]):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    with pytest.warns(UserWarning, match=EXPECTED_WARNING_STR):
        a = Array.create(
            store / "generic",
            shape=data.shape,
            chunk_shape=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            codecs=[BytesCodec(), codec_class()],
        )

    a[:, :] = data.copy()
    np.testing.assert_array_equal(data, a[:, :])


@pytest.mark.parametrize(
    ("codec_class", "codec_config"),
    [
        (numcodecs.zarr3.Delta, {"dtype": "float32"}),
        (numcodecs.zarr3.FixedScaleOffset, {"offset": 0, "scale": 25.5}),
        (numcodecs.zarr3.FixedScaleOffset, {"offset": 0, "scale": 51, "astype": "uint16"}),
        (numcodecs.zarr3.AsType, {"encode_dtype": "float32", "decode_dtype": "float64"}),
    ],
    ids=[
        "delta",
        "fixedscaleoffset",
        "fixedscaleoffset2",
        "astype",
    ],
)
def test_generic_filter(
    store: Store, codec_class: type[numcodecs.zarr3._NumcodecsCodec], codec_config: dict[str, JSON]
):
    data = np.linspace(0, 10, 256, dtype="float32").reshape((16, 16))

    with pytest.warns(UserWarning, match=EXPECTED_WARNING_STR):
        a = Array.create(
            store / "generic",
            shape=data.shape,
            chunk_shape=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            codecs=[
                codec_class(**codec_config),
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
            codecs=[numcodecs.zarr3.BitRound(keepbits=3), BytesCodec()],
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
            codecs=[numcodecs.zarr3.Quantize(digits=3), BytesCodec()],
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
            codecs=[numcodecs.zarr3.PackBits(), BytesCodec()],
        )

        a[:, :] = data.copy()
        a = Array.open(store / "generic_packbits")
    np.testing.assert_array_equal(data, a[:, :])

    with pytest.raises(ValueError, match=".*requires bool dtype.*"):
        Array.create(
            store / "generic_packbits_err",
            shape=data.shape,
            chunk_shape=(16, 16),
            dtype="uint32",
            fill_value=0,
            codecs=[numcodecs.zarr3.PackBits(), BytesCodec()],
        )


@pytest.mark.parametrize(
    "codec_class",
    [
        numcodecs.zarr3.CRC32,
        numcodecs.zarr3.CRC32C,
        numcodecs.zarr3.Adler32,
        numcodecs.zarr3.Fletcher32,
        numcodecs.zarr3.JenkinsLookup3,
    ],
)
def test_generic_checksum(store: Store, codec_class: type[numcodecs.zarr3._NumcodecsCodec]):
    data = np.linspace(0, 10, 256, dtype="float32").reshape((16, 16))

    with pytest.warns(UserWarning, match=EXPECTED_WARNING_STR):
        a = Array.create(
            store / "generic_checksum",
            shape=data.shape,
            chunk_shape=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            codecs=[BytesCodec(), codec_class()],
        )

        a[:, :] = data.copy()
        a = Array.open(store / "generic_checksum")
    np.testing.assert_array_equal(data, a[:, :])


@pytest.mark.parametrize("codec_class", [numcodecs.zarr3.PCodec, numcodecs.zarr3.ZFPY])
def test_generic_bytes_codec(store: Store, codec_class: type[numcodecs.zarr3._NumcodecsCodec]):
    try:
        codec_class()._codec  # noqa: B018
    except ValueError as e:
        if "codec not available" in str(e):
            pytest.xfail(f"{codec_class.codec_name} is not available: {e}")
        else:
            raise  # pragma: no cover
    except ImportError as e:
        pytest.xfail(f"{codec_class.codec_name} is not available: {e}")

    data = np.arange(0, 256, dtype="float32").reshape((16, 16))

    with pytest.warns(UserWarning, match=EXPECTED_WARNING_STR):
        a = Array.create(
            store / "generic",
            shape=data.shape,
            chunk_shape=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            codecs=[
                codec_class(),
            ],
        )

    a[:, :] = data.copy()
    np.testing.assert_array_equal(data, a[:, :])
