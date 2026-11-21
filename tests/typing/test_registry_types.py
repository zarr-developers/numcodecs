"""Static types of `numcodecs.registry`, checked against runtime behavior."""

from importlib.metadata import EntryPoint
from typing import assert_type

import numcodecs
from numcodecs.abc import Codec
from numcodecs.registry import codec_registry, entries, get_codec


def test_get_codec() -> None:
    codec = assert_type(get_codec({"id": "zlib", "level": 1}), Codec)
    assert isinstance(codec, numcodecs.Zlib)


def test_codec_registry_holds_classes() -> None:
    assert_type(codec_registry, dict[str, type[Codec]])
    for cls in codec_registry.values():
        assert isinstance(cls, type)
        assert issubclass(cls, Codec)


def test_entries_hold_entry_points() -> None:
    assert_type(entries, dict[str, EntryPoint])
    for entry in entries.values():
        assert isinstance(entry, EntryPoint)
