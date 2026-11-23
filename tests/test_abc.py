import typing

import pytest

import numcodecs
from numcodecs.abc import Codec


class NoId(Codec):
    """A codec whose author did not set `codec_id`."""

    def encode(self, buf):
        return buf

    def decode(self, buf, out=None):
        return buf


class _PassThrough(NoId):
    def __getattribute__(self, name):
        return super().__getattribute__(name)


def test_codec_id_has_no_default():
    """An identifier is a string, never None. A class that sets none has no `codec_id`."""
    for cls in (Codec, numcodecs.checksum32.Checksum32, NoId):
        with pytest.raises(AttributeError):
            cls.codec_id  # noqa: B018
    assert typing.get_type_hints(Codec)["codec_id"] is str
    assert all(isinstance(cls.codec_id, str) for cls in numcodecs.registry.codec_registry.values())


@pytest.mark.parametrize("cls", [NoId, _PassThrough])
def test_codec_without_codec_id(cls):
    """A codec class that never set `codec_id` is incomplete: its id cannot be read, it
    has no configuration, it equals nothing, and it cannot be registered without an id."""
    codec = cls()
    with pytest.raises(AttributeError, match="codec_id"):
        codec.codec_id  # noqa: B018
    with pytest.raises(AttributeError, match="codec_id"):
        codec.get_config()
    assert codec != cls()
    assert repr(NoId()) == "NoId()"
    with pytest.raises(ValueError, match="has no codec_id and none was given"):
        numcodecs.register_codec(cls)


def test_star_imports_do_not_shadow_the_json_codec():
    namespace: dict[str, object] = {}
    exec("from numcodecs import *\nfrom numcodecs.abc import *", namespace)
    assert namespace["JSON"] is numcodecs.JSON


def test_abc_exports():
    assert set(numcodecs.abc.__all__) == {"Codec", "CodecConfig", "CodecProtocol", "JSONValue"}


def test_checksum_codec_without_codec_id_reports_corruption():
    """Corrupt data is reported as such even by a subclass that set no id."""

    class MyChecksum(numcodecs.checksum32.Checksum32):
        @staticmethod
        def checksum(data, value=0):
            return 1

    with pytest.raises(RuntimeError, match="Stored and computed MyChecksum checksum do not match"):
        MyChecksum().decode(bytes(8))


class _ClassProperty:
    """A descriptor that also runs when read from the class, unlike `property`."""

    def __get__(self, instance, owner):
        return owner._prefix + "-x"


class DescriptorId(NoId):
    codec_id = _ClassProperty()


def test_register_codec_propagates_an_error_inside_a_codec_id_descriptor():
    """The class defines `codec_id`; that reading it fails is a bug in the codec, not a
    missing id."""
    with pytest.raises(AttributeError, match="_prefix"):
        numcodecs.register_codec(DescriptorId)


def test_register_codec_propagates_an_error_inside_a_metaclass_codec_id():
    from abc import ABCMeta

    class Meta(ABCMeta):
        @property
        def codec_id(cls):
            return cls._prefix + "-x"

    class MetaProperty(NoId, metaclass=Meta):
        pass

    with pytest.raises(AttributeError, match="_prefix"):
        numcodecs.register_codec(MetaProperty)


def test_checksum_codec_names_its_id_when_reporting_corruption():
    with pytest.raises(RuntimeError, match="Stored and computed crc32 checksum do not match"):
        numcodecs.CRC32().decode(bytes(8) + b"x")


def test_checksum_codec_propagates_an_error_inside_a_codec_id_property():
    """The class defines `codec_id`; that reading it fails is a bug in the codec."""

    class MyChecksum(numcodecs.checksum32.Checksum32):
        @property
        def codec_id(self):
            return self._name.lower()

        @staticmethod
        def checksum(data, value=0):
            return 1

    with pytest.raises(AttributeError, match="_name"):
        MyChecksum().decode(bytes(8))


def test_register_codec_refuses_an_id_of_none():
    """A class that sets `codec_id = None` (the old default, spelled out) has no id."""

    class ExplicitNone(NoId):
        codec_id = None

    with pytest.raises(ValueError, match="has no codec_id and none was given"):
        numcodecs.register_codec(ExplicitNone)
    assert None not in numcodecs.registry.codec_registry
