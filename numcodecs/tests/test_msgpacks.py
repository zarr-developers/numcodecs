import unittest
import warnings


import numpy as np


try:
    from numcodecs.msgpacks import LegacyMsgPack, MsgPack
    default_codec = MsgPack()
    # N.B., legacy codec is broken, see tests below. Also legacy code generates
    # PendingDeprecationWarning due to use of encoding argument, which we ignore here
    # as not relevant.
    legacy_codec = LegacyMsgPack()
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("msgpack not available")


from numcodecs.tests.common import (check_config, check_repr, check_encode_decode_array,
                                    check_backwards_compatibility, greetings)


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
    np.array([b'foo', b'bar', b'baz'] * 300, dtype=object),
    np.array([g.encode('utf-8') for g in greetings] * 100, dtype=object),
]


legacy_arrays = arrays[:8]


def test_encode_decode():

    for arr in arrays:
        check_encode_decode_array(arr, default_codec)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PendingDeprecationWarning)
        for arr in legacy_arrays:
            check_encode_decode_array(arr, legacy_codec)


def test_config():
    for codec in [default_codec, legacy_codec]:
        check_config(codec)


def test_repr():
    check_repr("MsgPack(raw=False, use_bin_type=True, use_single_float=False)")
    check_repr("MsgPack(raw=True, use_bin_type=False, use_single_float=True)")
    check_repr("LegacyMsgPack(encoding='utf-8')")
    check_repr("LegacyMsgPack(encoding='ascii')")


def test_backwards_compatibility():
    check_backwards_compatibility(default_codec.codec_id, arrays, [default_codec])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PendingDeprecationWarning)
        check_backwards_compatibility(legacy_codec.codec_id, legacy_arrays,
                                      [legacy_codec])


def test_non_numpy_inputs():
    # numpy will infer a range of different shapes and dtypes for these inputs.
    # Make sure that round-tripping through encode preserves this.
    data = [
        [0, 1],
        [[0, 1], [2, 3]],
        [[0], [1], [2, 3]],
        [[[0, 0]], [[1, 1]], [[2, 3]]],
        ["1"],
        ["11", "11"],
        ["11", "1", "1"],
        [{}],
        [{"key": "value"}, ["list", "of", "strings"]],
        [b"1"],
        [b"11", b"11"],
        [b"11", b"1", b"1"],
        [{b"key": b"value"}, [b"list", b"of", b"strings"]],
    ]
    for input_data in data:
        actual = default_codec.decode(default_codec.encode(input_data))
        expect = np.array(input_data)
        assert expect.shape == actual.shape
        assert np.array_equal(expect, actual)


def test_legacy_codec_broken():
    # Simplest demonstration of why the MsgPack codec needed to be changed.
    # The LegacyMsgPack codec didn't include shape information in the serialised
    # bytes, which gave different shapes in the input and output under certain
    # circumstances.
    a = np.empty(2, dtype=object)
    a[0] = [0, 1]
    a[1] = [2, 3]
    codec = LegacyMsgPack()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PendingDeprecationWarning)
        b = codec.decode(codec.encode(a))
    assert a.shape == (2,)
    assert b.shape == (2, 2)
    assert not np.array_equal(a, b)

    # Now show that the MsgPack codec handles this case properly.
    codec = MsgPack()
    b = codec.decode(codec.encode(a))
    assert np.array_equal(a, b)
    assert a.shape == b.shape


def test_encode_decode_shape_dtype_preserved():
    for arr in arrays:
        actual = default_codec.decode(default_codec.encode(arr))
        assert arr.shape == actual.shape
        assert arr.dtype == actual.dtype


def test_bytes():
    # test msgpack behaviour with bytes and str (unicode)
    bytes_arr = np.array([b'foo', b'bar', b'baz'], dtype=object)
    unicode_arr = np.array(['foo', 'bar', 'baz'], dtype=object)

    # raw=False (default)
    codec = MsgPack()
    # works for bytes array, round-trips bytes to bytes
    b = codec.decode(codec.encode(bytes_arr))
    assert np.array_equal(bytes_arr, b)
    assert isinstance(b[0], bytes)
    assert b[0] == b'foo'
    # works for unicode array, round-trips unicode to unicode
    b = codec.decode(codec.encode(unicode_arr))
    assert np.array_equal(unicode_arr, b)
    assert isinstance(b[0], str)
    assert b[0] == 'foo'

    # raw=True
    codec = MsgPack(raw=True)
    # works for bytes array, round-trips bytes to bytes
    b = codec.decode(codec.encode(bytes_arr))
    assert np.array_equal(bytes_arr, b)
    assert isinstance(b[0], bytes)
    assert b[0] == b'foo'
    # broken for unicode array, round-trips unicode to bytes
    b = codec.decode(codec.encode(unicode_arr))
    assert not np.array_equal(unicode_arr, b)
    assert isinstance(b[0], bytes)
    assert b[0] == b'foo'

    # legacy codec
    codec = LegacyMsgPack()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PendingDeprecationWarning)
        # broken for bytes array, round-trips bytes to unicode
        b = codec.decode(codec.encode(bytes_arr))
        assert not np.array_equal(bytes_arr, b)
        assert isinstance(b[0], str)
        assert b[0] == 'foo'
        # works for unicode array, round-trips unicode to unicode
        b = codec.decode(codec.encode(unicode_arr))
        assert np.array_equal(unicode_arr, b)
        assert isinstance(b[0], str)
        assert b[0] == 'foo'
