import sys
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, Mapping, Optional, Tuple

import numpy as np

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


ConvertFunc = Callable[[NDArrayLike], NDArrayLike]

_ndarray_like_registry: DefaultDict[type, Dict[str, ConvertFunc]] = defaultdict(dict)


def register_ndarray_like(cls, convert_dict: Mapping[str, ConvertFunc]) -> None:
    _ndarray_like_registry[cls].update(convert_dict)


def ensure_memtype(ary: NDArrayLike, memtype=Optional[str]) -> NDArrayLike:
    if memtype is None:
        return ary
    return _ndarray_like_registry[ary.__class__][memtype](ary)


register_ndarray_like(np.ndarray, {"cpu": lambda x: x})

try:
    import cupy
except ImportError:
    pass
else:
    register_ndarray_like(cupy.ndarray, {"cpu": cupy.asnumpy})
