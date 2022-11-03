import sys
from typing import Any, Dict, Optional, Tuple, Type

if sys.version_info >= (3, 8):
    from typing import Protocol, runtime_checkable
else:
    from typing_extensions import Protocol, runtime_checkable


class _CachedProtocolMeta(Protocol.__class__):
    """Custom implementation of @runtime_checkable

    The native implementation of @runtime_checkable is slow,
    see <https://github.com/zarr-developers/numcodecs/issues/379>.

    This metaclass keeps an unbounded cache of the result of
    isinstance checks using the object's class as the cache key.
    """
    _instancecheck_cache: Dict[Tuple[Type, Type], bool] = {}

    def __instancecheck__(cls, instance):
        key = (cls, instance.__class__)
        ret = cls._instancecheck_cache.get(key, None)
        if ret is None:
            ret = super().__instancecheck__(instance)
            cls._instancecheck_cache[key] = ret
        return ret


@runtime_checkable
class DType(Protocol, metaclass=_CachedProtocolMeta):
    itemsize: int
    name: str
    kind: str


@runtime_checkable
class FlagsObj(Protocol, metaclass=_CachedProtocolMeta):
    c_contiguous: bool
    f_contiguous: bool
    owndata: bool


@runtime_checkable
class NDArrayLike(Protocol, metaclass=_CachedProtocolMeta):
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


def is_ndarray_like(obj: object) -> bool:
    """Return True when `obj` is ndarray-like"""
    return isinstance(obj, NDArrayLike)
