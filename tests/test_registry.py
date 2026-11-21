import inspect
import types

import pytest

import numcodecs
from numcodecs.errors import UnknownCodecError
from numcodecs.registry import get_codec


def test_registry_errors():
    with pytest.raises(UnknownCodecError, match='foo'):
        get_codec({'id': 'foo'})


def test_get_codec_argument():
    # Check that get_codec doesn't modify its argument.
    arg = {"id": "json2"}
    before = dict(arg)
    get_codec(arg)
    assert before == arg


def test_all_classes_registered():
    """
    find all Codec subclasses in this repository and check that they
    have been registered.

    see #346 for more info
    """
    missing = {
        obj.codec_id
        for _, submod in inspect.getmembers(numcodecs, inspect.ismodule)
        for _, obj in inspect.getmembers(submod)
        if (
            inspect.isclass(obj)
            and issubclass(obj, numcodecs.abc.Codec)
            and not inspect.isabstract(obj)
            and obj.codec_id not in numcodecs.registry.codec_registry
        )
    }

    if missing:
        raise Exception(f"these codecs are missing: {missing}")  # pragma: no cover


class _DuckCodec:
    """Looks like a codec, but does not subclass numcodecs.abc.Codec."""

    codec_id = "duck"

    @classmethod
    def from_config(cls, config):
        return cls()


def test_entry_point_not_a_codec_subclass_is_deprecated(monkeypatch):
    monkeypatch.setitem(
        numcodecs.registry.entries, "duck", types.SimpleNamespace(load=lambda: _DuckCodec)
    )
    monkeypatch.delitem(numcodecs.registry.codec_registry, "duck", raising=False)
    with pytest.warns(DeprecationWarning, match="not a subclass of numcodecs.abc.Codec"):
        codec = get_codec({"id": "duck"})
    assert isinstance(codec, _DuckCodec)
    numcodecs.registry.codec_registry.pop("duck")


def test_register_codec_without_id():
    class NoId(numcodecs.abc.Codec):
        def encode(self, buf):  # pragma: no cover
            return buf

        def decode(self, buf, out=None):  # pragma: no cover
            return buf

    with pytest.raises(ValueError, match="has no codec_id and none was given"):
        numcodecs.register_codec(NoId)
