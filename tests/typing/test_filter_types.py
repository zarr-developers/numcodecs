"""Static types of the filter codecs, checked against runtime behavior."""

from typing import Any, assert_type

import numpy as np
import numpy.typing as npt

from numcodecs.abc import ArrayArrayCodec, CodecConfig
from numcodecs.astype import AsType, AsTypeConfig
from numcodecs.bitround import BitRound, BitRoundConfig
from numcodecs.categorize import Categorize, CategorizeConfig
from numcodecs.delta import Delta, DeltaConfig
from numcodecs.fixedscaleoffset import FixedScaleOffset, FixedScaleOffsetConfig
from numcodecs.ndarray_like import NDArrayLike
from numcodecs.packbits import PackBits
from numcodecs.quantize import Quantize, QuantizeConfig
from numcodecs.shuffle import Shuffle, ShuffleConfig

F8 = np.linspace(0, 100, 16, dtype="f8")


def _as_array_codec(codec: ArrayArrayCodec) -> ArrayArrayCodec:
    return codec


def test_numeric_filters() -> None:
    codecs = (
        AsType(encode_dtype="f4", decode_dtype="f8"),
        Delta(dtype="f8"),
        Quantize(digits=1, dtype="f8"),
        FixedScaleOffset(offset=0, scale=10, dtype="f8", astype="i4"),
    )
    for codec in codecs:
        _as_array_codec(codec)
        enc = assert_type(codec.encode(F8), npt.NDArray[Any])
        assert isinstance(enc, np.ndarray)
        dec = assert_type(codec.decode(enc), npt.NDArray[Any])
        assert isinstance(dec, np.ndarray)
        out = np.empty_like(F8)
        assert codec.decode(enc, out=out) is out


def test_bitround() -> None:
    codec = BitRound(keepbits=10)
    enc = assert_type(codec.encode(F8), NDArrayLike)
    assert isinstance(enc, np.ndarray)
    assert isinstance(assert_type(codec.decode(enc), NDArrayLike), np.ndarray)
    assert assert_type(codec.get_config(), BitRoundConfig)["keepbits"] == 10


def test_packbits_and_categorize() -> None:
    bools = np.array([True, False] * 8)
    packbits = PackBits()
    enc = assert_type(packbits.encode(bools), npt.NDArray[np.uint8])
    assert enc.dtype == np.uint8
    dec = assert_type(packbits.decode(enc), npt.NDArray[np.bool_])
    assert dec.dtype == np.bool_
    assert assert_type(packbits.get_config(), CodecConfig) == {"id": "packbits"}

    categorize = Categorize(labels=["foo", "bar"], dtype=object)
    codes = assert_type(categorize.encode(["foo", "bar", "baz"]), npt.NDArray[Any])
    assert codes.tolist() == [1, 2, 0]
    config = assert_type(categorize.get_config(), CategorizeConfig)
    assert assert_type(config["labels"], list[str]) == ["foo", "bar"]


def test_shuffle() -> None:
    codec = Shuffle(elementsize=8)
    enc = assert_type(codec.encode(F8), npt.NDArray[np.uint8])
    assert enc.dtype == np.uint8
    out = bytearray(F8.nbytes)
    assert assert_type(codec.encode(F8, out=out), bytearray) is out
    assert assert_type(codec.decode(enc), npt.NDArray[np.uint8]).tobytes() == F8.tobytes()
    assert assert_type(codec.get_config(), ShuffleConfig)["elementsize"] == 8


def test_configs() -> None:
    assert assert_type(AsType("f4", "f8").get_config(), AsTypeConfig)["encode_dtype"] == "<f4"
    assert assert_type(Delta("i4").get_config(), DeltaConfig)["astype"] == "<i4"
    assert assert_type(Quantize(1, "f8").get_config(), QuantizeConfig)["digits"] == 1
    fso = assert_type(FixedScaleOffset(0, 10, "f8").get_config(), FixedScaleOffsetConfig)
    assert assert_type(fso["scale"], float) == 10
