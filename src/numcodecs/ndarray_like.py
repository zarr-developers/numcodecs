# `typing._ProtocolMeta` is private, but typeshed declares it, so type checkers see the
# metaclass's real type instead of the `Any` that `Protocol.__class__` gives them.
from collections.abc import Buffer
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, _ProtocolMeta, runtime_checkable

from typing_extensions import TypeIs  # in `typing` from Python 3.13


class _CachedProtocolMeta(_ProtocolMeta):
    """Custom implementation of @runtime_checkable

    The native implementation of @runtime_checkable is slow,
    see <https://github.com/zarr-developers/numcodecs/issues/379>.

    This metaclass keeps an unbounded cache of the result of
    isinstance checks using the object's class as the cache key.
    """

    # N.B., a protocol class using this metaclass must have annotations of its own.
    # Without them it would find this class's `__annotations__`, and `runtime_checkable`
    # would require instances to have `_instancecheck_cache` as an attribute.
    _instancecheck_cache: ClassVar[dict[tuple[type, type], bool]] = {}

    def __instancecheck__(self, instance: object) -> bool:
        key = (self, instance.__class__)
        ret = self._instancecheck_cache.get(key)
        if ret is None:
            ret = super().__instancecheck__(instance)
            self._instancecheck_cache[key] = ret
        return ret


# The data members of the protocols below are declared twice.
#
# Type checkers see read-only properties: numpy (and cupy) expose `dtype`, `shape`, `flags`
# and friends as read-only properties, and a protocol declaring plain (settable) attributes
# does not match a real ndarray.
#
# At runtime they are plain annotations, as they have always been, so that the members
# `isinstance` checks for stay the same and a class that subclasses a protocol can still
# assign them.


@runtime_checkable
class DType(Protocol, metaclass=_CachedProtocolMeta):
    if TYPE_CHECKING:  # pragma: no cover

        @property
        def itemsize(self) -> int: ...
        @property
        def name(self) -> str: ...
        @property
        def kind(self) -> str: ...
    else:
        itemsize: int
        name: str
        kind: str


@runtime_checkable
class FlagsObj(Protocol, metaclass=_CachedProtocolMeta):
    if TYPE_CHECKING:  # pragma: no cover

        @property
        def c_contiguous(self) -> bool: ...
        @property
        def f_contiguous(self) -> bool: ...
        @property
        def owndata(self) -> bool: ...
    else:
        c_contiguous: bool
        f_contiguous: bool
        owndata: bool


@runtime_checkable
class NDArrayLike(Protocol, metaclass=_CachedProtocolMeta):
    """The part of the ndarray interface numcodecs relies on.

    `numpy.ndarray` and `cupy.ndarray` both satisfy it. Methods return `Any` because numpy
    overloads them heavily; convert with `numcodecs.compat.ensure_ndarray` when you need
    numpy's own types.
    """

    if TYPE_CHECKING:  # pragma: no cover

        @property
        def dtype(self) -> DType: ...
        @property
        def shape(self) -> tuple[int, ...]: ...
        @property
        def strides(self) -> tuple[int, ...]: ...
        @property
        def ndim(self) -> int: ...
        @property
        def size(self) -> int: ...
        @property
        def itemsize(self) -> int: ...
        @property
        def nbytes(self) -> int: ...
        @property
        def flags(self) -> FlagsObj: ...
    else:
        dtype: DType
        shape: tuple[int, ...]
        strides: tuple[int, ...]
        ndim: int
        size: int
        itemsize: int
        nbytes: int
        flags: FlagsObj

    def __len__(self) -> int: ...  # pragma: no cover

    def __getitem__(self, key: Any, /) -> Any: ...  # pragma: no cover

    def __setitem__(self, key: Any, value: Any, /) -> None: ...  # pragma: no cover

    def tobytes(self, order: Any = ...) -> bytes: ...  # pragma: no cover

    def reshape(self, *args: Any, **kwargs: Any) -> Any: ...  # pragma: no cover

    def view(self, *args: Any, **kwargs: Any) -> Any: ...  # pragma: no cover


type BufferLike = Buffer | NDArrayLike
"""What codecs accept and return: an object exporting the buffer protocol (PEP 688), such
as `bytes`, `bytearray`, `memoryview` or a numpy array, or an ndarray-like such as a cupy
array."""


def is_ndarray_like(obj: object) -> TypeIs[NDArrayLike]:
    """Return True when `obj` is ndarray-like"""
    return isinstance(obj, NDArrayLike)
