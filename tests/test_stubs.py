"""Checks of the `.pyi` stubs that `mypy.stubtest` does not make.

stubtest compares module-level names and method signatures, but it does not look at
instances. It accepts an attribute that a stub class declares and the runtime class does
not have (it could be set in `__init__`), it does not notice an attribute that `__init__`
sets and the stub leaves out, and it knows nothing about what `get_config()` returns.
These tests instantiate each stubbed codec and compare it, and its configuration, with
what the stub declares.
"""

import ast
import importlib
from importlib.resources import files

import numpy as np
import pytest

INSTANCES = {
    ("blosc", "Blosc"): {},
    ("zstd", "Zstd"): {},
    ("lz4", "LZ4"): {},
    ("vlen", "VLenUTF8"): {},
    ("vlen", "VLenBytes"): {},
    ("vlen", "VLenArray"): {"dtype": "i4"},
    ("fletcher32", "Fletcher32"): {},
}


def _stub_class(module: str, cls: str) -> ast.ClassDef:
    stub = files("numcodecs") / f"{module}.pyi"
    tree = ast.parse(stub.read_text())
    (class_def,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls]
    return class_def


def _declared_attributes(module: str, cls: str) -> dict[str, str]:
    """The attributes the stub declares on `cls`, with their annotations as source text."""
    return {
        node.target.id: ast.unparse(node.annotation)
        for node in _stub_class(module, cls).body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }


def _declared_config(module: str, cls: str) -> dict[str, str]:
    """The keys of the TypedDict that the stub says `cls.get_config()` returns, with
    their annotations as source text."""
    (get_config,) = [
        n
        for n in _stub_class(module, cls).body
        if isinstance(n, ast.FunctionDef) and n.name == "get_config"
    ]
    assert isinstance(get_config.returns, ast.Name)
    if get_config.returns.id == "CodecConfig":
        return {"id": "str"}
    return {"id": "str", **_declared_attributes(module, get_config.returns.id)}


# How to check a value against each annotation the stubs use. Exact types, so that `bool`
# does not pass for `int`. An annotation missing here fails the tests: add it.
CHECKS = {
    "int": lambda value: type(value) is int,
    "str": lambda value: type(value) is str,
    "bool": lambda value: type(value) is bool,
    "np.dtype[Any]": lambda value: isinstance(value, np.dtype),
}


def _mistyped(declared: dict[str, str], values: dict[str, object]) -> dict[str, str]:
    """The declared names whose value does not have the declared type."""
    return {
        name: f"declared {annotation}, is {type(values[name]).__name__}"
        for name, annotation in declared.items()
        if not CHECKS[annotation](values[name])
    }


def _codec(module: str, cls: str):
    return getattr(importlib.import_module(f"numcodecs.{module}"), cls)(**INSTANCES[module, cls])


def test_every_stubbed_codec_class_is_checked():
    """A codec class added to a stub must be added to `INSTANCES`."""
    stubbed = set()
    for stub in files("numcodecs").iterdir():
        if stub.name.endswith(".pyi"):
            tree = ast.parse(stub.read_text())
            stubbed |= {
                (stub.name.removesuffix(".pyi"), node.name)
                for node in tree.body
                if isinstance(node, ast.ClassDef)
                and any(isinstance(b, ast.Name) and b.id == "Codec" for b in node.bases)
            }
    assert stubbed == set(INSTANCES)


@pytest.mark.parametrize(("module", "cls"), INSTANCES, ids=[c for _, c in INSTANCES])
def test_stub_class_attributes_match_the_instance(module, cls):
    """The stub declares exactly the public data attributes an instance has, with their
    types."""
    codec = _codec(module, cls)
    declared = _declared_attributes(module, cls)
    actual = {
        name: getattr(codec, name)
        for name in dir(codec)
        if not name.startswith("_") and not callable(getattr(codec, name))
    }
    assert set(declared) == set(actual)
    assert not _mistyped(declared, actual)


@pytest.mark.parametrize(("module", "cls"), INSTANCES, ids=[c for _, c in INSTANCES])
def test_stub_config_matches_get_config(module, cls):
    """The TypedDict a stub declares for `get_config()` has exactly the keys returned,
    with their types."""
    declared = _declared_config(module, cls)
    config = _codec(module, cls).get_config()
    assert set(declared) == set(config)
    assert not _mistyped(declared, config)
