"""Static types of the compressor codecs, checked against runtime behavior."""

import json
from collections.abc import Mapping
from typing import Any, assert_type

import numpy as np

from numcodecs.abc import JSON, ArrayArrayCodec, BytesBytesCodec, CodecConfig
from numcodecs.base64 import Base64
from numcodecs.blosc import Blosc, BloscConfig
from numcodecs.bz2 import BZ2, BZ2Config
from numcodecs.gzip import GZip, GZipConfig
from numcodecs.lz4 import LZ4, LZ4Config
from numcodecs.lzma import LZMA, LZMAConfig
from numcodecs.zlib import Zlib, ZlibConfig
from numcodecs.zstd import Zstd, ZstdConfig

DATA = np.arange(16, dtype="f8")


def _round_trip(codec: BytesBytesCodec) -> None:
    """Every compressor: bytes in any buffer form go in, bytes come out, `out` comes back."""
    enc = codec.encode(DATA)
    assert bytes(memoryview(codec.decode(enc))) == DATA.tobytes()


def test_compressors() -> None:
    for codec in (Zlib(), GZip(), BZ2(), LZMA(), Base64(), Blosc(), Zstd(), LZ4()):
        _round_trip(codec)
        enc = assert_type(codec.encode(DATA), bytes)
        assert isinstance(enc, bytes)
        assert assert_type(codec.decode(enc), bytes) == DATA.tobytes()
        out = np.empty_like(DATA)
        assert codec.decode(enc, out=out) is out
        buf = bytearray(DATA.nbytes)
        assert assert_type(codec.decode(enc, out=buf), bytearray) is buf


def _json_round_trip(config: Any) -> Any:
    return json.loads(json.dumps(config))


def test_configs() -> None:
    zlib_config = assert_type(Zlib(level=3).get_config(), ZlibConfig)
    assert assert_type(zlib_config["level"], int) == 3
    assert assert_type(GZip(level=2).get_config(), GZipConfig)["level"] == 2
    assert assert_type(BZ2(level=4).get_config(), BZ2Config)["level"] == 4
    lzma_config = assert_type(LZMA(preset=5).get_config(), LZMAConfig)
    assert assert_type(lzma_config["preset"], int | None) == 5
    assert assert_type(Base64().get_config(), CodecConfig) == {"id": "base64"}
    blosc_config = assert_type(Blosc(cname="zstd").get_config(), BloscConfig)
    assert assert_type(blosc_config["cname"], str) == "zstd"
    assert set(blosc_config) == set(BloscConfig.__annotations__)
    zstd_config = assert_type(Zstd(checksum=True).get_config(), ZstdConfig)
    assert assert_type(zstd_config["checksum"], bool) is True
    assert assert_type(LZ4().get_config(), LZ4Config)["acceleration"] == 1
    for config in (zlib_config, lzma_config, blosc_config, zstd_config):
        assert _json_round_trip(config) == config


def test_compressors_are_not_array_codecs() -> None:
    # A compressor returns bytes, which is not an ndarray-like.
    codec: ArrayArrayCodec = Zlib()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    assert not isinstance(codec.encode(DATA), np.ndarray)


def _json_mapping(config: Mapping[str, JSON]) -> Mapping[str, JSON]:
    return config


def test_configs_are_json_mappings() -> None:
    # pyright implements PEP 728, so CodecConfig's read-only JSON `extra_items` make every
    # codec configuration a `Mapping[str, JSON]`. mypy does not implement it yet.
    zlib_config = _json_mapping(Zlib().get_config())  # type: ignore[arg-type]
    blosc_config = _json_mapping(Blosc().get_config())  # type: ignore[arg-type]
    for config in (zlib_config, blosc_config):
        assert _json_round_trip(config) == config
