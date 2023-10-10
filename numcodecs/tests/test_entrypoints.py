import os.path
import sys
from unittest import mock

import pytest

import numcodecs.registry


here = os.path.abspath(os.path.dirname(__file__))


def set_path():
    sys.path.append(here)
    numcodecs.registry.run_entrypoints()
    yield
    sys.path.remove(here)
    numcodecs.registry.run_entrypoints()
    numcodecs.registry.codec_registry.pop("test")


@pytest.fixture()
def set_path_fixture():
    yield from set_path()


@pytest.mark.usefixtures("set_path_fixture")
def test_entrypoint_codec():
    cls = numcodecs.registry.get_codec({"id": "test"})
    assert cls.codec_id == "test"


def test_entrypoint_codec_with_importlib_metadata():
    # importlib_metadata patches importlib.metadata, which can lead to breaking changes
    # to the APIs of EntryPoint objects used when registering entrypoints. Attempt to
    # isolate those changes to just this test.
    with mock.patch.dict(sys.modules):
        import importlib_metadata
        sys.path.append(here)
        numcodecs.registry.run_entrypoints()
        cls = numcodecs.registry.get_codec({"id": "test"})
        assert cls.codec_id == "test"
        sys.path.remove(here)
        numcodecs.registry.run_entrypoints()
        numcodecs.registry.codec_registry.pop("test")
