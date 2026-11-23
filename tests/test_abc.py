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


class PropertyId(Codec):
    """A codec whose `codec_id` is a property that fails."""

    @property
    def codec_id(self):
        return self._name.lower()

    def encode(self, buf):
        return buf

    def decode(self, buf, out=None):
        return buf


def test_error_inside_a_codec_id_property_is_not_swallowed():
    """Only a codec whose class has no `codec_id` at all reports None. An AttributeError
    from inside a `codec_id` property is a bug in that codec and must surface."""
    with pytest.raises(AttributeError, match="_name"):
        PropertyId().get_config()
    assert PropertyId() != PropertyId()


def test_checksum_codec_without_codec_id_reports_corruption():
    class MyChecksum(numcodecs.checksum32.Checksum32):
        @staticmethod
        def checksum(data, value=0):
            return 1

    with pytest.raises(RuntimeError, match="Stored and computed None checksum do not match"):
        MyChecksum().decode(bytes(8))
