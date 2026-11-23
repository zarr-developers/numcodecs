"""Static types of the checksum codecs, checked against runtime behavior."""

from typing import assert_type

import numpy as np
import numpy.typing as npt

from numcodecs.abc import CodecConfig
from numcodecs.checksum32 import (
    CRC32,
    Adler32,
    Checksum32Config,
    JenkinsLookup3,
    JenkinsLookup3Config,
)
from numcodecs.fletcher32 import Fletcher32

DATA = np.arange(16, dtype="f8")


def test_checksum32() -> None:
    for codec in (CRC32(), Adler32(), CRC32(location="end")):
        enc = assert_type(codec.encode(DATA), npt.NDArray[np.uint8] | bytes)
        assert isinstance(enc, np.ndarray)
        dec = assert_type(codec.decode(enc), npt.NDArray[np.uint8] | memoryview)
        assert bytes(dec) == DATA.tobytes()
        out = bytearray(DATA.nbytes)
        assert assert_type(codec.decode(enc, out=out), bytearray) is out
    config = assert_type(CRC32(location="end").get_config(), Checksum32Config)
    assert config.get("location") == "end"
    assert "location" not in Adler32().get_config()


def test_jenkins_lookup3() -> None:
    codec = JenkinsLookup3(prefix=b"FADB")
    enc = assert_type(codec.encode(DATA), bytes)
    assert isinstance(enc, bytes)
    dec = assert_type(codec.decode(enc), memoryview)
    assert dec.tobytes() == DATA.tobytes()
    config = assert_type(codec.get_config(), JenkinsLookup3Config)
    assert assert_type(config["prefix"], list[int] | None) == list(b"FADB")


def test_fletcher32() -> None:
    assert assert_type(Fletcher32().get_config(), CodecConfig) == {"id": "fletcher32"}
