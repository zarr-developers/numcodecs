from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest

import numcodecs.bitround

if TYPE_CHECKING:  # pragma: no cover
    import zarr
else:
    zarr = pytest.importorskip("zarr")

import zarr.storage
from zarr.core.common import JSON

import numcodecs.zarr3

pytestmark = [
    pytest.mark.skipif(zarr.__version__ < "3.0.0", reason="zarr 3.0.0 or later is required"),
    pytest.mark.filterwarnings("ignore:Codec 'numcodecs.*' not configured in config.*:UserWarning"),
    pytest.mark.filterwarnings(
        "ignore:Numcodecs codecs are not in the Zarr version 3 specification and may not be supported by other zarr implementations."
    ),
]

get_codec_class = zarr.registry.get_codec_class
Array = zarr.Array
BytesCodec = zarr.codecs.BytesCodec
Store = zarr.abc.store.Store
MemoryStore = zarr.storage.MemoryStore
StorePath = zarr.storage.StorePath


EXPECTED_WARNING_STR = "Numcodecs codecs are not in the Zarr version 3.*"


@pytest.fixture
def store() -> StorePath:
    return StorePath(MemoryStore(read_only=False))


ALL_CODECS = [getattr(numcodecs.zarr3, cls_name) for cls_name in numcodecs.zarr3.__all__]


@pytest.mark.parametrize("codec_class", ALL_CODECS)
def test_entry_points(codec_class: type[numcodecs.zarr3._NumcodecsCodec]):
    codec_name = codec_class.codec_name
    assert get_codec_class(codec_name) == codec_class


@pytest.mark.parametrize("codec_class", ALL_CODECS)
def test_docstring(codec_class: type[numcodecs.zarr3._NumcodecsCodec]):
    assert "See :class:`numcodecs." in codec_class.__doc__  # type: ignore[operator]


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
def test_generic_compressor(
    store: StorePath, codec_class: type[numcodecs.zarr3._NumcodecsBytesBytesCodec]
):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    with pytest.warns(UserWarning, match=EXPECTED_WARNING_STR):
        a = zarr.create_array(
            store / "generic",
            shape=data.shape,
            chunks=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            compressors=[codec_class()],
        )

    a[:, :] = data.copy()
    np.testing.assert_array_equal(data, a[:, :])


@pytest.mark.parametrize(
    ("codec_class", "codec_config"),
    [
        (numcodecs.zarr3.Delta, {"dtype": "float32"}),
        (numcodecs.zarr3.FixedScaleOffset, {"offset": 0, "scale": 25.5}),
        (numcodecs.zarr3.FixedScaleOffset, {"offset": 0, "scale": 51, "astype": "uint16"}),
        (numcodecs.zarr3.AsType, {"encode_dtype": "float32", "decode_dtype": "float32"}),
    ],
    ids=[
        "delta",
        "fixedscaleoffset",
        "fixedscaleoffset2",
        "astype",
    ],
)
def test_generic_filter(
    store: StorePath,
    codec_class: type[numcodecs.zarr3._NumcodecsArrayArrayCodec],
    codec_config: dict[str, JSON],
):
    data = np.linspace(0, 10, 256, dtype="float32").reshape((16, 16))

    with pytest.warns(UserWarning, match=EXPECTED_WARNING_STR):
        a = zarr.create_array(
            store / "generic",
            shape=data.shape,
            chunks=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            filters=[
                codec_class(**codec_config),
            ],
        )

    a[:, :] = data.copy()
    a = zarr.open_array(store / "generic", mode="r")
    np.testing.assert_array_equal(data, a[:, :])


def test_generic_filter_bitround(store: StorePath):
    data = np.linspace(0, 1, 256, dtype="float32").reshape((16, 16))

    with pytest.warns(UserWarning, match=EXPECTED_WARNING_STR):
        a = zarr.create_array(
            store / "generic_bitround",
            shape=data.shape,
            chunks=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            filters=[numcodecs.zarr3.BitRound(keepbits=3)],
        )

    a[:, :] = data.copy()
    a = zarr.open_array(store / "generic_bitround", mode="r")
    assert np.allclose(data, a[:, :], atol=0.1)


def test_generic_filter_quantize(store: StorePath):
    data = np.linspace(0, 10, 256, dtype="float32").reshape((16, 16))

    with pytest.warns(UserWarning, match=EXPECTED_WARNING_STR):
        a = zarr.create_array(
            store / "generic_quantize",
            shape=data.shape,
            chunks=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            filters=[numcodecs.zarr3.Quantize(digits=3)],
        )

    a[:, :] = data.copy()
    a = zarr.open_array(store / "generic_quantize", mode="r")
    assert np.allclose(data, a[:, :], atol=0.001)


def test_generic_filter_packbits(store: StorePath):
    data = np.zeros((16, 16), dtype="bool")
    data[0:4, :] = True

    with pytest.warns(UserWarning, match=EXPECTED_WARNING_STR):
        a = zarr.create_array(
            store / "generic_packbits",
            shape=data.shape,
            chunks=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            filters=[numcodecs.zarr3.PackBits()],
        )

    a[:, :] = data.copy()
    a = zarr.open_array(store / "generic_packbits", mode="r")
    np.testing.assert_array_equal(data, a[:, :])

    with pytest.raises(ValueError, match=".*requires bool dtype.*"):
        zarr.create_array(
            store / "generic_packbits_err",
            shape=data.shape,
            chunks=(16, 16),
            dtype="uint32",
            fill_value=0,
            filters=[numcodecs.zarr3.PackBits()],
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
def test_generic_checksum(
    store: StorePath, codec_class: type[numcodecs.zarr3._NumcodecsBytesBytesCodec]
):
    data = np.linspace(0, 10, 256, dtype="float32").reshape((16, 16))

    with pytest.warns(UserWarning, match=EXPECTED_WARNING_STR):
        a = zarr.create_array(
            store / "generic_checksum",
            shape=data.shape,
            chunks=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            compressors=[codec_class()],
        )

    a[:, :] = data.copy()
    a = zarr.open_array(store / "generic_checksum", mode="r")
    np.testing.assert_array_equal(data, a[:, :])


@pytest.mark.parametrize("codec_class", [numcodecs.zarr3.PCodec, numcodecs.zarr3.ZFPY])
def test_generic_bytes_codec(
    store: StorePath, codec_class: type[numcodecs.zarr3._NumcodecsArrayBytesCodec]
):
    try:
        codec_class()._codec  # noqa: B018
    except ValueError as e:  # pragma: no cover
        if "codec not available" in str(e):
            pytest.xfail(f"{codec_class.codec_name} is not available: {e}")
        else:
            raise
    except ImportError as e:  # pragma: no cover
        pytest.xfail(f"{codec_class.codec_name} is not available: {e}")

    data = np.arange(0, 256, dtype="float32").reshape((16, 16))

    with pytest.warns(UserWarning, match=EXPECTED_WARNING_STR):
        a = zarr.create_array(
            store / "generic",
            shape=data.shape,
            chunks=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            serializer=codec_class(),
        )

    a[:, :] = data.copy()
    np.testing.assert_array_equal(data, a[:, :])


def test_delta_astype(store: StorePath):
    data = np.linspace(0, 10, 256, dtype="i8").reshape((16, 16))

    with pytest.warns(UserWarning, match=EXPECTED_WARNING_STR):
        a = zarr.create_array(
            store / "generic",
            shape=data.shape,
            chunks=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            filters=[
                numcodecs.zarr3.Delta(dtype="i8", astype="i2"),
            ],
        )

    a[:, :] = data.copy()
    a = zarr.open_array(store / "generic", mode="r")
    np.testing.assert_array_equal(data, a[:, :])


def test_repr():
    codec = numcodecs.zarr3.LZ4(level=5)
    assert repr(codec) == "LZ4(codec_name='numcodecs.lz4', codec_config={'level': 5})"


def test_to_dict():
    codec = numcodecs.zarr3.LZ4(level=5)
    assert codec.to_dict() == {"name": "numcodecs.lz4", "configuration": {"level": 5}}


@pytest.mark.parametrize(
    "codec_cls",
    [
        numcodecs.zarr3.Blosc,
        numcodecs.zarr3.LZ4,
        numcodecs.zarr3.Zstd,
        numcodecs.zarr3.Zlib,
        numcodecs.zarr3.GZip,
        numcodecs.zarr3.BZ2,
        numcodecs.zarr3.LZMA,
        numcodecs.zarr3.Shuffle,
        numcodecs.zarr3.BitRound,
        numcodecs.zarr3.Delta,
        numcodecs.zarr3.FixedScaleOffset,
        numcodecs.zarr3.Quantize,
        numcodecs.zarr3.PackBits,
        numcodecs.zarr3.AsType,
        numcodecs.zarr3.CRC32,
        numcodecs.zarr3.CRC32C,
        numcodecs.zarr3.Adler32,
        numcodecs.zarr3.Fletcher32,
        numcodecs.zarr3.JenkinsLookup3,
        numcodecs.zarr3.PCodec,
        numcodecs.zarr3.ZFPY,
    ],
)
def test_codecs_pickleable(codec_cls):
    codec = codec_cls()

    expected = codec

    p = pickle.dumps(codec)
    actual = pickle.loads(p)
    assert actual == expected
