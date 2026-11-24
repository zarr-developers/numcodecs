"""Static types of `numcodecs.compat`, checked against runtime behavior.

Each `assert_type` is verified by mypy and pyright (see `tests/typing/README.md`), and the
`assert isinstance` next to it is verified by pytest. Together they keep the annotations
honest: a claim that type checkers accept but the runtime contradicts fails here.
"""

from typing import Any, assert_type

import numpy as np
import numpy.typing as npt

from numcodecs.compat import ensure_bytes, ensure_contiguous_ndarray, ensure_ndarray


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
