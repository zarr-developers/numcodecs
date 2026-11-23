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


def test_writable_buffer_like_definitions_agree():
    """`WritableBufferLike` is defined twice, for type checkers and for the runtime. Both
    must name the same types, and the runtime one must be evaluable."""
    import array
    import ast
    import ctypes
    import inspect
    import mmap
    import pickle

    import numpy as np

    import numcodecs.ndarray_like as module

    runtime = set(module.WritableBufferLike.__value__.__args__)
    assert runtime == {
        bytearray,
        memoryview,
        array.array,
        mmap.mmap,
        np.ndarray,
        ctypes.Array,
        pickle.PickleBuffer,
        NDArrayLike,
    }

    def members(alias: ast.TypeAlias) -> set[str]:
        """The union members of a `type X = A | B[...] | ...` statement, unsubscripted."""
        names = set()
        todo = [alias.value]
        while todo:
            node = todo.pop()
            if isinstance(node, ast.BinOp):
                todo += [node.left, node.right]
            elif isinstance(node, ast.Subscript):
                names.add(ast.unparse(node.value))
            else:
                names.add(ast.unparse(node))
        return names

    # the definition type checkers read is the one that names its members
    (static,) = [
        node
        for node in ast.walk(ast.parse(inspect.getsource(module)))
        if isinstance(node, ast.TypeAlias)
        and node.name.id == "WritableBufferLike"
        and isinstance(node.value, ast.BinOp)
    ]
    namespace = {"array": array, "mmap": mmap, "np": np, "ctypes": ctypes, "pickle": pickle}
    assert {eval(name, namespace, vars(module)) for name in members(static)} == runtime


def test_writable_buffer_like_without_optional_stdlib_modules(monkeypatch):
    """`ctypes` and `mmap` are optional parts of the standard library. Where one is
    missing, `WritableBufferLike` leaves its type out."""
    import ctypes
    import mmap
    import sys

    from numcodecs.ndarray_like import _writable_buffer_types

    monkeypatch.setitem(sys.modules, "ctypes", None)
    monkeypatch.setitem(sys.modules, "mmap", None)
    members = set(_writable_buffer_types().__args__)
    assert not members & {ctypes.Array, mmap.mmap}
    assert {bytearray, memoryview, NDArrayLike} <= members


# The optional `crc32c` package needs `mmap` itself, so it is blocked along with it.
@pytest.mark.parametrize(
    "blocked", [("ctypes", "_ctypes"), ("mmap", "crc32c")], ids=["ctypes", "mmap"]
)
def test_import_does_not_need_optional_stdlib_modules(blocked):
    """A Python built without libffi has no `ctypes`; numpy imports there, and so must
    numcodecs. The same goes for `mmap`."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        f"for name in {blocked!r}: sys.modules[name] = None\n"
        "import numcodecs.ndarray_like\n"
        "from numcodecs.ndarray_like import WritableBufferLike\n"
        "assert bytearray in WritableBufferLike.__value__.__args__\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def _protocol_members(protocol: str) -> tuple[dict[str, str], dict[str, str]]:
    """The data members a protocol declares for type checkers (properties under `if
    TYPE_CHECKING`) and at runtime (annotations under `else`), with their types as source
    text."""
    import ast
    import inspect

    import numcodecs.ndarray_like as module

    (class_def,) = [
        node
        for node in ast.parse(inspect.getsource(module)).body
        if isinstance(node, ast.ClassDef) and node.name == protocol
    ]
    (branch,) = [node for node in class_def.body if isinstance(node, ast.If)]
    static = {
        node.name: ast.unparse(node.returns)
        for node in branch.body
        if isinstance(node, ast.FunctionDef) and node.returns is not None
    }
    runtime = {
        node.target.id: ast.unparse(node.annotation)
        for node in branch.orelse
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert len(static) == len(branch.body)
    assert len(runtime) == len(branch.orelse)
    return static, runtime


# every member each protocol has required since it was introduced
PROTOCOL_MEMBERS = {
    "DType": ({"itemsize", "name", "kind"}, set()),
    "FlagsObj": ({"c_contiguous", "f_contiguous", "owndata"}, set()),
    "NDArrayLike": (
        {"dtype", "shape", "strides", "ndim", "size", "itemsize", "nbytes", "flags"},
        {"__len__", "__getitem__", "__setitem__", "tobytes", "reshape", "view"},
    ),
}


@pytest.mark.parametrize("protocol", PROTOCOL_MEMBERS)
def test_protocol_definitions_agree(protocol):
    """The data members are declared twice, for type checkers and for the runtime. Both
    must name the same members, with the same types: those the protocol has always had,
    and no others."""
    import numcodecs.ndarray_like as module

    data, methods = PROTOCOL_MEMBERS[protocol]
    static, runtime = _protocol_members(protocol)
    assert set(static) == data
    assert static == runtime, "the two declarations differ in a member or in its type"
    assert getattr(module, protocol).__protocol_attrs__ == data | methods


@pytest.mark.parametrize(
    ("protocol", "complete", "member"),
    [
        (protocol, complete, member)
        for protocol, complete in (
            (DType, _Wrapper._DType),
            (FlagsObj, _Wrapper._Flags),
            (NDArrayLike, _Wrapper),
        )
        for member in sorted(set().union(*PROTOCOL_MEMBERS[protocol.__name__]))
    ],
)
def test_every_protocol_member_is_required(protocol, complete, member):
    """An object lacking any one member is not an instance."""
    attrs = {k: v for k, v in vars(complete).items() if k != member}
    incomplete = type("Incomplete", (), attrs)
    assert member not in dir(incomplete)
    assert not isinstance(incomplete(), protocol)


@pytest.mark.parametrize(
    ("protocol", "member"),
    [(DType, "itemsize"), (DType, "kind"), (FlagsObj, "owndata"), (NDArrayLike, "flags")],
)
def test_nominal_subclass_can_set_every_kind_of_member(protocol, member):
    """A class that subclasses one of the protocols assigns its members in `__init__`,
    which a property on the protocol would prevent."""

    class Mine(protocol):
        def __init__(self, value):
            setattr(self, member, value)

    assert getattr(Mine(8), member) == 8
