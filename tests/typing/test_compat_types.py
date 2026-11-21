"""Static types of `numcodecs.compat`, checked against runtime behavior.

Each `assert_type` is verified by mypy and pyright (see `tests/typing/README.md`), and the
`assert isinstance` next to it is verified by pytest. Together they keep the annotations
honest: a claim that type checkers accept but the runtime contradicts fails here.
"""

from typing import Any, assert_type

import numpy as np
import numpy.typing as npt

from numcodecs.compat import (
    ensure_bytes,
    ensure_contiguous_ndarray,
    ensure_ndarray,
    ensure_ndarray_like,
    ensure_text,
    ndarray_copy,
)
from numcodecs.ndarray_like import NDArrayLike, is_ndarray_like


def _as_ndarray_like(x: NDArrayLike) -> NDArrayLike:
    """Forget the concrete type, as for an ndarray-like that is not a numpy array."""
    return x


def _as_object(x: object) -> object:
    return x


def test_ensure_bytes() -> None:
    for buf in (b"abc", bytearray(b"abc"), memoryview(b"abc"), np.arange(3, dtype="u1")):
        result = assert_type(ensure_bytes(buf), bytes)
        assert isinstance(result, bytes)


def test_ensure_ndarray() -> None:
    for buf in (b"abc", bytearray(b"abc"), memoryview(b"abc"), np.arange(3, dtype="u1")):
        result = assert_type(ensure_ndarray(buf), npt.NDArray[Any])
        assert isinstance(result, np.ndarray)


def test_ensure_contiguous_ndarray() -> None:
    for buf in (b"abc", bytearray(b"abc"), memoryview(b"abc"), np.arange(3, dtype="u1")):
        result = assert_type(ensure_contiguous_ndarray(buf), npt.NDArray[Any])
        assert isinstance(result, np.ndarray)


def test_ensure_ndarray_like() -> None:
    from_bytes = assert_type(ensure_ndarray_like(b"abc"), npt.NDArray[Any])
    assert isinstance(from_bytes, np.ndarray)
    # an ndarray-like that is not known to export the buffer protocol (such as a cupy array)
    # comes back as the protocol type
    like = _as_ndarray_like(np.arange(3))
    from_like = assert_type(ensure_ndarray_like(like), NDArrayLike)
    assert from_like is like


def test_ensure_text() -> None:
    for s in ("abc", b"abc", bytearray(b"abc"), memoryview(b"abc")):
        result = assert_type(ensure_text(s), str)
        assert result == "abc"


def test_ndarray_copy() -> None:
    src = b"\x01\x02\x03"
    # without a destination, the source comes back unchanged
    same = assert_type(ndarray_copy(src, None), bytes)
    assert same is src
    # with a destination, that very object comes back, filled
    dst = bytearray(3)
    filled = assert_type(ndarray_copy(src, dst), bytearray)
    assert filled is dst
    assert bytes(dst) == src
    arr: npt.NDArray[np.uint8] = np.zeros(3, dtype=np.uint8)
    assert assert_type(ndarray_copy(src, arr), npt.NDArray[np.uint8]) is arr


def test_is_ndarray_like_narrows() -> None:
    obj = _as_object(np.arange(3))
    if is_ndarray_like(obj):
        assert_type(obj, NDArrayLike)
        assert obj.shape == (3,)
    else:  # pragma: no cover
        raise AssertionError
