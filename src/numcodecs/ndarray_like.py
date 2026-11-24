# `typing._ProtocolMeta` is private, but typeshed declares it, so type checkers see the
# metaclass's real type instead of the `Any` that `Protocol.__class__` gives them.
import array
import pickle
from collections.abc import Buffer
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeVar,
    _ProtocolMeta,
    runtime_checkable,
)

import numpy as np
from typing_extensions import TypeIs  # in `typing` from Python 3.13


class _CachedProtocolMeta(_ProtocolMeta):
    """Custom implementation of @runtime_checkable

    The native implementation of @runtime_checkable is slow,
    see <https://github.com/zarr-developers/numcodecs/issues/379>.

    This metaclass keeps an unbounded cache of the result of
    isinstance checks using the object's class as the cache key.
    """

    # Deliberately not annotated. A protocol class without annotations of its own would
    # otherwise find this class's `__annotations__`, and `runtime_checkable` would then
    # require instances to have `_instancecheck_cache` as an attribute (on Python 3.12
    # and 3.13).
    _instancecheck_cache = dict[tuple[type, type], bool]()  # noqa: RUF012  # see above

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
# At runtime they are plain annotations. `isinstance` checks for the same members either
# way, but a class that subclasses a protocol can assign a plain annotation, and could
# not assign a property.


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

# Two definitions, for two reasons:
#
# - `memoryview[Any]` is what type checkers need (a bare `memoryview` means
#   `memoryview[int]` to them, which leaves out e.g. `memoryview(...).cast("d")`), but
#   `memoryview` cannot be subscripted at runtime before Python 3.14.
# - `ctypes` and `mmap` are optional parts of the standard library (a Python built without
#   libffi has no `ctypes`), and importing numcodecs must not require them. The runtime
#   definition imports them only when the alias is evaluated, and leaves out what is
#   missing.
if TYPE_CHECKING:  # pragma: no cover
    import ctypes
    import mmap

    type WritableBufferLike = (
        bytearray
        | memoryview[Any]
        | array.array[Any]
        | mmap.mmap
        | np.ndarray[Any, Any]
        | ctypes.Array[Any]
        | pickle.PickleBuffer
        | NDArrayLike
    )
else:

    def _writable_buffer_types():
        """The runtime value of `WritableBufferLike`: the same types, unsubscripted."""
        import contextlib
        import importlib

        union = bytearray | memoryview | array.array | np.ndarray | pickle.PickleBuffer
        for module, name in (("mmap", "mmap"), ("ctypes", "Array")):
            with contextlib.suppress(ImportError):
                union |= getattr(importlib.import_module(module), name)
        return union | NDArrayLike

    type WritableBufferLike = _writable_buffer_types()
"""What `decode` accepts as `out`: a buffer that can be written to.

Python's type system cannot say "a writable buffer", so this names writable buffer types
from the standard library and numpy, plus ndarray-likes such as cupy arrays. Its purpose is
to leave out `bytes`, which exports a buffer but is immutable.

Two limits follow from that:

- It is not a guarantee of writability: a read-only `memoryview` or ndarray still
  type-checks.
- A writable buffer of a type not named here is rejected by type checkers. Wrap it in
  `memoryview(...)`, which is accepted.

Some codecs need more than a writable buffer and raise for anything else: JSON, MsgPack,
Pickle, the VLen codecs, and Categorize when it decodes to an object dtype, need an
ndarray; PCodec needs a buffer whose item type is the decoded dtype. Type checkers cannot
be told so without those codecs breaking the signature they inherit from `Codec`.
"""

OutT = TypeVar("OutT", bound=WritableBufferLike)
"""The type of a caller's `out` argument, for `decode(buf, out: OutT) -> OutT`: a codec
given `out` returns that same object.

numcodecs' codecs overload `decode` twice: `out: None = None` returns what the codec
decodes to, and `out: OutT` returns `OutT`.
"""


def is_ndarray_like(obj: object) -> TypeIs[NDArrayLike]:
    """Return True when `obj` is ndarray-like"""
    return isinstance(obj, NDArrayLike)
