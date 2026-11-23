import pytest

from numcodecs.ndarray_like import DType, FlagsObj, NDArrayLike


@pytest.mark.parametrize("module", ["numpy", "cupy"])
def test_is_ndarray_like(module):
    m = pytest.importorskip(module)
    a = m.arange(10)
    assert isinstance(a, NDArrayLike)


def test_is_not_ndarray_like():
    assert not isinstance([1, 2, 3], NDArrayLike)
    assert not isinstance(b"1,2,3", NDArrayLike)


@pytest.mark.parametrize("module", ["numpy", "cupy"])
def test_is_dtype_like(module):
    m = pytest.importorskip(module)
    d = m.dtype("u8")
    assert isinstance(d, DType)


def test_is_not_dtype_like():
    assert not isinstance([1, 2, 3], DType)
    assert not isinstance(b"1,2,3", DType)


@pytest.mark.parametrize("module", ["numpy", "cupy"])
def test_is_flags_like(module):
    m = pytest.importorskip(module)
    d = m.arange(10).flags
    assert isinstance(d, FlagsObj)


def test_is_not_flags_like():
    assert not isinstance([1, 2, 3], FlagsObj)
    assert not isinstance(b"1,2,3", FlagsObj)


@pytest.mark.parametrize("module", ["numpy", "cupy"])
def test_cached_isinstance_check(module):
    m = pytest.importorskip(module)
    a = m.arange(10)
    assert isinstance(a, NDArrayLike)
    assert not isinstance(a, DType)
    assert not isinstance(a, FlagsObj)


class _Wrapper:
    """Every member NDArrayLike has required since it was introduced, and nothing else:
    in particular no `copy`, and a dtype without `str`."""

    class _DType:
        itemsize = 1
        name = "uint8"
        kind = "u"

    dtype = _DType()
    shape = (3,)
    strides = (1,)
    ndim = 1
    size = 3
    itemsize = 1
    nbytes = 3

    class _Flags:
        c_contiguous = True
        f_contiguous = True
        owndata = True

    flags = _Flags()

    def __len__(self):
        return 3

    def __getitem__(self, key):
        return 0

    def __setitem__(self, key, value):
        pass

    def tobytes(self, order=None):
        return b"\x00\x00\x00"

    def reshape(self, *shape, order="C"):
        return self

    def view(self, dtype=None):
        return self


def test_protocol_members_unchanged():
    w = _Wrapper()
    assert isinstance(w, NDArrayLike)
    assert isinstance(w.dtype, DType)
    assert isinstance(w.flags, FlagsObj)
    from numcodecs.compat import ensure_ndarray_like

    assert ensure_ndarray_like(w) is w


def test_nominal_subclass_can_set_members():
    class Mine(NDArrayLike):
        def __init__(self):
            self.dtype = _Wrapper._DType()
            self.shape = (3,)

    m = Mine()
    assert m.shape == (3,)
    with pytest.raises(AttributeError):
        Mine.__new__(Mine).strides  # noqa: B018


def test_protocol_without_annotations_using_the_cached_metaclass():
    """The isinstance cache lives on the metaclass. It must not turn into a member that
    instances are required to have."""
    from typing import Protocol, runtime_checkable

    from numcodecs.ndarray_like import _CachedProtocolMeta

    @runtime_checkable
    class HasFoo(Protocol, metaclass=_CachedProtocolMeta):
        def foo(self): ...

    class Impl:
        def foo(self):
            return 1

    assert isinstance(Impl(), HasFoo)
    assert not isinstance(object(), HasFoo)
    assert isinstance(NDArrayLike._instancecheck_cache, dict)
