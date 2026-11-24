"""Decoding with the VLen codecs into a caller-supplied `out` array."""

from collections.abc import Callable

import numpy as np
import pytest

import numcodecs
from numcodecs.abc import Codec

DATA: dict[str, np.ndarray] = {
    "vlen-utf8": np.array(["a", "bb", "", "dddd"], dtype=object),
    "vlen-bytes": np.array([b"a", b"bb", b"", b"dddd"], dtype=object),
    "vlen-array": np.array([np.arange(n, dtype="i4") for n in (1, 2, 0, 4)], dtype=object),
}

# Layouts numpy cannot flatten without a copy.
NON_CONTIGUOUS_OUTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "every other row": lambda base: base[::2],
    "every other row and column": lambda base: base[::2, ::2],
    "rows reversed": lambda base: base[::-1],
    "transposed and strided": lambda base: base.T[::2],
    "3-D, strided on the middle axis": lambda base: base.reshape(2, 4, 3)[:, ::2],
}


def _codec(codec_id: str) -> Codec:
    config = {"id": codec_id, "dtype": "<i4"} if codec_id == "vlen-array" else {"id": codec_id}
    return numcodecs.get_codec(config)


def _items(codec_id: str, n: int) -> np.ndarray:
    return np.resize(DATA[codec_id], n)


@pytest.mark.parametrize("layout", NON_CONTIGUOUS_OUTS)
@pytest.mark.parametrize("codec_id", DATA)
def test_decode_fills_a_non_contiguous_out(codec_id: str, layout: str) -> None:
    """`out` cannot be flattened without a copy, so the decoded items have to be copied
    back into it, in logical (C) order. The rest of the memory `out` is a view of is left
    alone."""
    codec = _codec(codec_id)
    stale = object()
    base = np.full((6, 4), stale, dtype=object)
    out = NON_CONTIGUOUS_OUTS[layout](base)
    assert not np.shares_memory(out.reshape(-1, order="A"), out)
    data = _items(codec_id, out.size)

    assert codec.decode(codec.encode(data), out=out) is out
    assert all(np.array_equal(a, b) for a, b in zip(out.ravel(), data, strict=True))
    assert sum(item is stale for item in base.flat) == base.size - out.size


@pytest.mark.parametrize("codec_id", DATA)
def test_decode_refuses_a_read_only_non_contiguous_out(codec_id: str) -> None:
    """The decoded items cannot be copied back, and returning `out` unfilled would be
    wrong data."""
    codec = _codec(codec_id)
    stale = object()
    base = np.full((4, 2), stale, dtype=object)
    out = base[::2]
    out.setflags(write=False)
    with pytest.raises(ValueError, match="read-only"):
        codec.decode(codec.encode(_items(codec_id, out.size)), out=out)
    assert all(item is stale for item in base.flat)


@pytest.mark.parametrize("codec_id", DATA)
def test_decode_into_an_empty_read_only_out(codec_id: str) -> None:
    """There is nothing to write, so a read-only `out` is not an error."""
    codec = _codec(codec_id)
    out = np.empty(0, dtype=object)
    out.setflags(write=False)
    assert codec.decode(codec.encode(np.empty(0, dtype=object)), out=out) is out
