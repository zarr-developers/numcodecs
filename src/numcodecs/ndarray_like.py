# `typing._ProtocolMeta` is private, but typeshed declares it, so type checkers see the
# metaclass's real type instead of the `Any` that `Protocol.__class__` gives them.
from collections.abc import Buffer
from typing import Any, Protocol, Self, _ProtocolMeta, runtime_checkable

from typing_extensions import TypeIs  # in `typing` from Python 3.13

# Kept at module level, not as an annotated attribute of the metaclass: a protocol class
# without annotations of its own would otherwise find the metaclass's `__annotations__`, and
# `runtime_checkable` would then require instances to have the cache as an attribute.
_instancecheck_cache: dict[tuple[type, type], bool] = {}


class _CachedProtocolMeta(_ProtocolMeta):
    """Custom implementation of @runtime_checkable

    The native implementation of @runtime_checkable is slow,
    see <https://github.com/zarr-developers/numcodecs/issues/379>.

    This metaclass keeps an unbounded cache of the result of
    isinstance checks using the object's class as the cache key.
    """

    def __instancecheck__(self, instance: object) -> bool:
        key = (self, instance.__class__)
        ret = _instancecheck_cache.get(key)
        if ret is None:
            ret = super().__instancecheck__(instance)
            _instancecheck_cache[key] = ret
        return ret


# The protocols below declare their attributes as read-only properties. numpy (and cupy)
# expose `dtype`, `shape`, `flags` and friends as read-only properties, and a protocol that
# declared them as plain (settable) attributes would not match a real ndarray.


@runtime_checkable
class DType(Protocol, metaclass=_CachedProtocolMeta):
    @property
    def itemsize(self) -> int: ...  # pragma: no cover
    @property
    def name(self) -> str: ...  # pragma: no cover
    @property
    def kind(self) -> str: ...  # pragma: no cover
    @property
    def str(self) -> str: ...  # pragma: no cover


@runtime_checkable
class FlagsObj(Protocol, metaclass=_CachedProtocolMeta):
    @property
    def c_contiguous(self) -> bool: ...  # pragma: no cover
    @property
    def f_contiguous(self) -> bool: ...  # pragma: no cover
    @property
    def owndata(self) -> bool: ...  # pragma: no cover


@runtime_checkable
class NDArrayLike(Protocol, metaclass=_CachedProtocolMeta):
    """The part of the ndarray interface numcodecs relies on.

    `numpy.ndarray` and `cupy.ndarray` both satisfy it. Methods return `Any` because numpy
    overloads them heavily; convert with `numcodecs.compat.ensure_ndarray` when you need
    numpy's own types.
    """

    @property
    def dtype(self) -> DType: ...  # pragma: no cover
    @property
    def shape(self) -> tuple[int, ...]: ...  # pragma: no cover
    @property
    def strides(self) -> tuple[int, ...]: ...  # pragma: no cover
    @property
    def ndim(self) -> int: ...  # pragma: no cover
    @property
    def size(self) -> int: ...  # pragma: no cover
    @property
    def itemsize(self) -> int: ...  # pragma: no cover
    @property
    def nbytes(self) -> int: ...  # pragma: no cover
    @property
    def flags(self) -> FlagsObj: ...  # pragma: no cover

    def __len__(self) -> int: ...  # pragma: no cover

    def __getitem__(self, key: Any, /) -> Any: ...  # pragma: no cover

    def __setitem__(self, key: Any, value: Any, /) -> None: ...  # pragma: no cover

    def tobytes(self, order: Any = ...) -> bytes: ...  # pragma: no cover

    def reshape(self, *args: Any, **kwargs: Any) -> Any: ...  # pragma: no cover

    def view(self, *args: Any, **kwargs: Any) -> Any: ...  # pragma: no cover

    def copy(self, *args: Any, **kwargs: Any) -> Self: ...  # pragma: no cover


type BufferLike = Buffer | NDArrayLike
"""What codecs accept and return: an object exporting the buffer protocol (PEP 688), such
as `bytes`, `bytearray`, `memoryview` or a numpy array, or an ndarray-like such as a cupy
array."""


def is_ndarray_like(obj: object) -> TypeIs[NDArrayLike]:
    """Return True when `obj` is ndarray-like"""
    return isinstance(obj, NDArrayLike)
