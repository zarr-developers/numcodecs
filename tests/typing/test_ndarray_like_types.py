"""Static types of `numcodecs.ndarray_like`, checked against runtime behavior."""

from typing import Any, assert_type

import numpy as np
import numpy.typing as npt

from numcodecs.compat import ensure_ndarray_like
from numcodecs.ndarray_like import DType, FlagsObj, NDArrayLike, is_ndarray_like


class _MinimalDType:
    """Has no `str`, which `DType` required for a while on `main`."""

    itemsize: int = 1
    name: str = "uint8"
    kind: str = "u"


class _MinimalFlags:
    c_contiguous: bool = True
    f_contiguous: bool = True
    owndata: bool = True


class _Minimal:
    """Exactly the members `NDArrayLike` requires. In particular it has no `copy`, which
    `NDArrayLike` required for a while on `main`."""

    dtype: _MinimalDType = _MinimalDType()
    shape: tuple[int, ...] = (3,)
    strides: tuple[int, ...] = (1,)
    ndim: int = 1
    size: int = 3
    itemsize: int = 1
    nbytes: int = 3
    flags: _MinimalFlags = _MinimalFlags()

    def __len__(self) -> int:
        return 3

    def __getitem__(self, key: Any, /) -> Any:
        return 0

    def __setitem__(self, key: Any, value: Any, /) -> None:
        pass

    def tobytes(self, order: Any = None) -> bytes:
        return bytes(3)

    def reshape(self, *args: Any, **kwargs: Any) -> Any:
        return self

    def view(self, *args: Any, **kwargs: Any) -> Any:
        return self


def test_minimal_implementation_matches_statically_and_at_runtime() -> None:
    like: NDArrayLike = _Minimal()
    dtype: DType = _MinimalDType()
    flags: FlagsObj = _MinimalFlags()
    assert isinstance(like, NDArrayLike)
    assert isinstance(dtype, DType)
    assert isinstance(flags, FlagsObj)
    assert assert_type(ensure_ndarray_like(like), NDArrayLike) is like


def _read_members(like: NDArrayLike) -> None:
    """Every data member, read through the protocol."""
    assert assert_type(like.shape, tuple[int, ...]) == (3,)
    assert assert_type(like.strides, tuple[int, ...]) == (1,)
    assert assert_type(like.ndim, int) == 1
    assert assert_type(like.size, int) == 3
    assert assert_type(like.itemsize, int) == 1
    assert assert_type(like.nbytes, int) == 3
    assert assert_type(len(like), int) == 3
    assert assert_type(like.tobytes(), bytes) == bytes(3)
    dtype = assert_type(like.dtype, DType)
    assert assert_type(dtype.itemsize, int) == 1
    assert assert_type(dtype.name, str) == "uint8"
    assert assert_type(dtype.kind, str) == "u"
    flags = assert_type(like.flags, FlagsObj)
    assert assert_type(flags.c_contiguous, bool)
    assert assert_type(flags.f_contiguous, bool)
    assert assert_type(flags.owndata, bool)


def test_members_have_the_declared_types() -> None:
    _read_members(_Minimal())
    _read_members(np.zeros(3, dtype="u1"))


def test_numpy_arrays_match() -> None:
    array: npt.NDArray[np.float64] = np.zeros(3)
    like: NDArrayLike = array
    dtype: DType = array.dtype
    flags: FlagsObj = array.flags
    assert is_ndarray_like(like)
    assert isinstance(dtype, DType)
    assert isinstance(flags, FlagsObj)
