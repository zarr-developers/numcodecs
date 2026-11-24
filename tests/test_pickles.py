import itertools
import sys
import warnings
from functools import partial

import numpy as np
import pytest

import numcodecs
import numcodecs.pickles
from numcodecs.errors import UnsafeDecodeError
from numcodecs.pickles import _ENV_VAR, _REMOVAL_VERSION, Pickle, _read_opt_in_from_env
from numcodecs.registry import get_codec
from tests.common import (
    check_backwards_compatibility,
    check_config,
    check_encode_decode_array,
    check_repr,
    greetings,
)

# Constructing Pickle emits a DeprecationWarning. The tests below that predate the
# deprecation exercise the codec itself, so they ignore it; test_construction_warns checks it.
DEPRECATION_MATCH = "The Pickle codec is deprecated"
pytestmark = pytest.mark.filterwarnings(f"ignore:{DEPRECATION_MATCH}:DeprecationWarning")

# Built at import time, where pytestmark does not apply.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=DEPRECATION_MATCH, category=DeprecationWarning)
    codecs = [Pickle(protocol=i) for i in range(5)]

# object array with strings
# object array with mix strings / nans
# object array with mix of string, int, float
# ...
arrays = [
    np.array(['foo', 'bar', 'baz'] * 300, dtype=object),
    np.array([['foo', 'bar', np.nan]] * 300, dtype=object),
    np.array(['foo', 1.0, 2] * 300, dtype=object),
    np.arange(1000, dtype='i4'),
    np.array(['foo', 'bar', 'baz'] * 300),
    np.array(['foo', ['bar', 1.0, 2], {'a': 'b', 'c': 42}] * 300, dtype=object),
    np.array(greetings * 100),
    np.array(greetings * 100, dtype=object),
]


@pytest.fixture(autouse=True)
def opt_in_to_pickle_decode(monkeypatch):
    """Opt in to unpickling for tests that round-trip trusted data."""
    monkeypatch.setattr(numcodecs.pickles, "ALLOW_UNSAFE_PICKLE_DECODE", True)


def test_encode_decode():
    for arr, codec in itertools.product(arrays, codecs):
        check_encode_decode_array(arr, codec)


def test_config():
    codec = Pickle(protocol=-1)
    check_config(codec)
    for codec in codecs:
        check_config(codec)


def test_repr():
    check_repr("Pickle(protocol=-1)")


# Details on xfail
# https://stackoverflow.com/questions/58194852/modulenotfounderror-no-module-named-numpy-core-multiarray-r
@pytest.mark.skipif(sys.byteorder != "little", reason="Pickle does not restore byte orders")
@pytest.mark.xfail(
    sys.platform == "win32",
    reason=(
        "Pickle fails to read w/ Windows carriage return "
        "( https://github.com/zarr-developers/numcodecs/issues/271 )"
    ),
)
def test_backwards_compatibility():
    check_backwards_compatibility(Pickle.codec_id, arrays, codecs)


@pytest.mark.parametrize(
    "make_codec",
    [
        partial(Pickle),
        partial(numcodecs.Pickle, protocol=2),
        partial(Pickle.from_config, {"protocol": 3}),
        partial(get_codec, {"id": "pickle"}),
        partial(get_codec, {"id": "pickle", "protocol": 4}),
    ],
    ids=["Pickle", "numcodecs.Pickle", "Pickle.from_config", "get_codec", "get_codec-protocol"],
)
def test_construction_warns(make_codec):
    """Every way of obtaining a Pickle codec warns once, attributes the warning to the
    caller rather than to numcodecs internals, and leaves encode and decode working."""
    with pytest.warns(DeprecationWarning, match=DEPRECATION_MATCH) as record:
        codec = make_codec()
    assert len(record) == 1
    assert record[0].filename == __file__
    message = str(record[0].message)
    assert _REMOVAL_VERSION in message
    for alternative in ("VLenUTF8", "VLenBytes", "VLenArray", "JSON", "MsgPack"):
        assert alternative in message
    # encoding and decoding do not warn again
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_encode_decode_array(arrays[0], codec)


def test_construction_raises_when_deprecation_warnings_are_errors():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(DeprecationWarning, match=DEPRECATION_MATCH):
            get_codec({"id": "pickle"})


def test_decode_refused_without_opt_in(monkeypatch):
    monkeypatch.setattr(numcodecs.pickles, "ALLOW_UNSAFE_PICKLE_DECODE", False)
    # get_codec is the path by which array metadata selects this codec
    codec = get_codec({"id": "pickle"})
    enc = codec.encode(arrays[0])
    with pytest.raises(UnsafeDecodeError, match="ALLOW_UNSAFE_PICKLE_DECODE"):
        codec.decode(enc)


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("no", False),
        ("1", True),
        ("true", True),
        ("TRUE", True),
        (" yes ", True),
    ],
)
def test_allow_unsafe_pickle_decode_from_env(monkeypatch, env_value, expected):
    # The environment variable is read once, when numcodecs.pickles is imported, so
    # test the parsing helper rather than the module-level flag.
    if env_value is None:
        monkeypatch.delenv(_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(_ENV_VAR, env_value)
    assert _read_opt_in_from_env() is expected
