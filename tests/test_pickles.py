import itertools
import sys
import platform

import numpy as np
import pytest

import numcodecs.pickles
from numcodecs.errors import UnsafeDecodeError
from numcodecs.pickles import _ENV_VAR, Pickle, _read_opt_in_from_env
from numcodecs.registry import get_codec
from tests.common import (
    check_backwards_compatibility,
    check_config,
    check_encode_decode_array,
    check_repr,
    greetings,
)

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
    sys.platform == "win32" and platform.machine().lower() != "arm64",
    reason=(
        "Pickle fails to read w/ Windows carriage return "
        "( https://github.com/zarr-developers/numcodecs/issues/271 )"
    ),
)
def test_backwards_compatibility():
    check_backwards_compatibility(Pickle.codec_id, arrays, codecs)


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
