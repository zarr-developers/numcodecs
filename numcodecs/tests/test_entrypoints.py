import os.path
import sys

import pytest

from multiprocessing import Process

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


def get_entrypoints_with_importlib_metadata_loaded():
    # importlib_metadata patches importlib.metadata, which can lead to breaking changes
    # to the APIs of EntryPoint objects used when registering entrypoints. Attempt to
    # isolate those changes to just this test.
    import importlib_metadata  # noqa: F401
    sys.path.append(here)
    numcodecs.registry.run_entrypoints()
    cls = numcodecs.registry.get_codec({"id": "test"})
    assert cls.codec_id == "test"


def test_entrypoint_codec_with_importlib_metadata():
    p = Process(target=get_entrypoints_with_importlib_metadata_loaded)
    p.start()
    p.join()
    assert p.exitcode == 0
