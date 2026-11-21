"""Static types of the object and array-to-bytes codecs, checked against runtime behavior."""

from typing import Any, assert_type

import numpy as np
import numpy.typing as npt

from numcodecs import pickles
from numcodecs.abc import ArrayBytesCodec, CodecConfig
from numcodecs.json import JSON, JSONConfig
from numcodecs.pickles import Pickle, PickleConfig
from numcodecs.vlen import VLenArray, VLenArrayConfig, VLenUTF8

OBJECTS = np.array(["foo", "bar", "baz"], dtype=object)


def _as_array_bytes_codec(codec: ArrayBytesCodec) -> ArrayBytesCodec:
    return codec


def test_json() -> None:
    codec = _as_array_bytes_codec(JSON())
    enc = assert_type(JSON().encode(OBJECTS), bytes)
    assert isinstance(enc, bytes)
    dec = assert_type(JSON().decode(enc), npt.NDArray[Any])
    assert dec.tolist() == OBJECTS.tolist()
    out = np.empty(3, dtype=object)
    assert codec.decode(enc, out=out) is out
    config = assert_type(JSON().get_config(), JSONConfig)
    assert assert_type(config["separators"], tuple[str, str]) == (",", ":")


def test_pickle() -> None:
    codec = Pickle()
    enc = assert_type(codec.encode([1, 2]), bytes)
    allowed = pickles.ALLOW_UNSAFE_PICKLE_DECODE
    pickles.ALLOW_UNSAFE_PICKLE_DECODE = True
    try:
        # whatever was pickled comes back, so the type is Any
        assert assert_type(codec.decode(enc), Any) == [1, 2]
    finally:
        pickles.ALLOW_UNSAFE_PICKLE_DECODE = allowed
    assert assert_type(codec.get_config(), PickleConfig)["protocol"] == codec.protocol


def test_vlen_configs() -> None:
    assert assert_type(VLenUTF8().get_config(), CodecConfig) == {"id": "vlen-utf8"}
    config = assert_type(VLenArray("i4").get_config(), VLenArrayConfig)
    assert assert_type(config["dtype"], str) == "<i4"
