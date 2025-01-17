import inspect

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
            and obj.codec_id not in numcodecs.registry.codec_registry
            and obj.codec_id is not None  # remove `None`
        )
    }

    if missing:
        raise Exception(f"these codecs are missing: {missing}")  # pragma: no cover
