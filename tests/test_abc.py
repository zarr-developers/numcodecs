import pytest

import numcodecs
from numcodecs.abc import Codec


class NoId(Codec):
    """A codec whose author did not set `codec_id`."""

    def encode(self, buf):
        return buf

    def decode(self, buf, out=None):
        return buf


def test_codec_without_codec_id():
    """`codec_id` has no default on the class, but a codec that never set one keeps
    working as it did when the default was `None`."""
    codec = NoId()
    assert codec.get_config() == {"id": None}
    assert codec == codec
    assert codec == NoId()
    assert [NoId()].index(NoId()) == 0
    assert repr(codec) == "NoId()"


def test_abstract_codecs_have_no_codec_id():
    for cls in (Codec, numcodecs.checksum32.Checksum32):
        with pytest.raises(AttributeError):
            cls.codec_id  # noqa: B018


def test_star_imports_do_not_shadow_the_json_codec():
    namespace: dict[str, object] = {}
    exec("from numcodecs import *\nfrom numcodecs.abc import *", namespace)
    assert namespace["JSON"] is numcodecs.JSON
