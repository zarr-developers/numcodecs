import sys
from typing import Any, Optional, Tuple

if sys.version_info >= (3, 8):
    from typing import Protocol, runtime_checkable
else:
    from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class DType(Protocol):
    itemsize: int
    name: str
    kind: str


@runtime_checkable
class FlagsObj(Protocol):
    c_contiguous: bool
    f_contiguous: bool
    owndata: bool


@runtime_checkable
class NDArrayLike(Protocol):
    dtype: DType
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    ndim: int
    size: int
    itemsize: int
    nbytes: int
    flags: FlagsObj

    def __len__(self) -> int:
        ...

    def __getitem__(self, key) -> Any:
        ...

    def __setitem__(self, key, value):
        ...

    def tobytes(self, order: Optional[str] = ...) -> bytes:
        ...

    def reshape(self, *shape: int, order: str = ...) -> "NDArrayLike":
        ...

    def view(self, dtype: DType = ...) -> "NDArrayLike":
        ...


# keep in sync with NDArrayLike
_NDArrayLike_protocol_attrs = [
    "dtype",
    "shape",
    "strides",
    "ndim",
    "size",
    "itemsize",
    "nbytes",
    "flags",
    "__len__",
    "__getitem__",
    "__setitem__",
    "tobytes",
    "reshape",
    "view",
]


def is_ndarray_like(obj: object) -> bool:
    """Return True when `obj` is ndarray-like"""
    # same as `isinstance(obj, NDArrayLike)` but faster
    return all(hasattr(obj, attr) for attr in _NDArrayLike_protocol_attrs)
