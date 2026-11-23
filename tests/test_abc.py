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


def test_checksum_codec_propagates_an_error_inside_a_codec_id_property():
    class MyChecksum(numcodecs.checksum32.Checksum32):
        @property
        def codec_id(self):
            return self._name.lower()

        @staticmethod
        def checksum(data, value=0):
            return 1

    with pytest.raises(AttributeError, match="_name"):
        MyChecksum().decode(bytes(8))


class _ClassProperty:
    """A descriptor that also runs when read from the class, unlike `property`."""

    def __get__(self, instance, owner):
        return owner._prefix + "-x"


class DescriptorId(NoId):
    codec_id = _ClassProperty()


def test_error_inside_a_codec_id_descriptor_is_not_swallowed():
    with pytest.raises(AttributeError, match="_prefix"):
        DescriptorId().get_config()
    assert DescriptorId() != DescriptorId()


def test_codec_id_on_a_metaclass_is_not_the_codec_id():
    """Instances do not see metaclass attributes, so this class never set `codec_id`."""
    from abc import ABCMeta

    class Meta(ABCMeta):
        codec_id = "meta"

    class MetaId(NoId, metaclass=Meta):
        pass

    assert MetaId().get_config() == {"id": None}
    assert MetaId() == MetaId()


def test_codec_id_inherited_from_a_base_class_property_propagates():
    class Child(PropertyId):
        pass

    with pytest.raises(AttributeError, match="_name"):
        Child().get_config()
