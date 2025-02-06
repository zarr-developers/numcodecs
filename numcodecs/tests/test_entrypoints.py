import os.path
import sys

import pytest

import numcodecs.registry

here = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture
def set_path() -> None:
    sys.path.append(here)
    numcodecs.registry.run_entrypoints()
    yield
    sys.path.remove(here)
    numcodecs.registry.run_entrypoints()
    numcodecs.registry.codec_registry.pop("test")


@pytest.mark.usefixtures("set_path")
def test_entrypoint_codec() -> None:
    cls = numcodecs.registry.get_codec({"id": "test"})
    assert cls.codec_id == "test"
