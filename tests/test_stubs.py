"""Checks of the `.pyi` stubs that `mypy.stubtest` does not make.

stubtest compares module-level names and method signatures, but it accepts an attribute
that a stub class declares and the runtime class does not have (it could be set in
`__init__`). These tests instantiate each stubbed codec and look for every attribute its
stub declares.
"""

import ast
import importlib
from importlib.resources import files

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


def _declared_config_keys(module: str, cls: str) -> set[str]:
    """The keys of the TypedDict that the stub says `cls.get_config()` returns."""
    (get_config,) = [
        n
        for n in _stub_class(module, cls).body
        if isinstance(n, ast.FunctionDef) and n.name == "get_config"
    ]
    assert isinstance(get_config.returns, ast.Name)
    if get_config.returns.id == "CodecConfig":
        return {"id"}
    return {"id", *_declared_attributes(module, get_config.returns.id)}


# annotations that can be checked with isinstance
SIMPLE_TYPES = {"int": int, "str": str, "bool": bool, "float": (int, float)}


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
def test_stub_class_attributes_exist(module, cls):
    codec = _codec(module, cls)
    declared = _declared_attributes(module, cls)
    assert declared, "expected the stub to declare at least codec_id"
    missing = [name for name in declared if not hasattr(codec, name)]
    assert not missing, f"{module}.pyi declares {cls} attributes that do not exist: {missing}"
    wrong = {
        name: annotation
        for name, annotation in declared.items()
        if annotation in SIMPLE_TYPES
        and not isinstance(getattr(codec, name), SIMPLE_TYPES[annotation])
    }
    assert not wrong, f"{module}.pyi declares the wrong type for {cls} attributes: {wrong}"


@pytest.mark.parametrize(("module", "cls"), INSTANCES, ids=[c for _, c in INSTANCES])
def test_stub_config_matches_get_config(module, cls):
    """The TypedDict a stub declares for `get_config()` has exactly the keys returned."""
    assert _declared_config_keys(module, cls) == set(_codec(module, cls).get_config())
