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
    """A codec that never set `codec_id` works, with the id `None`."""
    codec = NoId()
    assert codec.get_config() == {"id": None}
    assert codec == codec
    assert codec == NoId()
    assert [NoId()].index(NoId()) == 0
    assert repr(codec) == "NoId()"


def test_abstract_codecs_have_no_codec_id():
    """The annotation is `str`, but at runtime classes that set no id keep the `None`
    default, and cannot be registered under it."""
    for cls in (Codec, numcodecs.checksum32.Checksum32, NoId):
        assert cls.codec_id is None
    with pytest.raises(ValueError, match="has no codec_id and none was given"):
        numcodecs.register_codec(NoId)


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
    """An AttributeError from inside a `codec_id` property is a bug in that codec and
    must surface, not turn into the `None` default."""
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


def test_error_inside_getattribute_is_not_swallowed():
    class Custom(NoId):
        def __getattribute__(self, name):
            if name == "codec_id":
                raise AttributeError("boom")
            return super().__getattribute__(name)

    with pytest.raises(AttributeError, match="boom"):
        Custom().get_config()
    assert Custom() != Custom()


def test_register_codec_propagates_an_error_inside_a_codec_id_descriptor():
    with pytest.raises(AttributeError, match="_prefix"):
        numcodecs.register_codec(DescriptorId)


def test_abc_exports():
    assert set(numcodecs.abc.__all__) == {"Codec", "CodecConfig", "CodecProtocol", "JSONValue"}


class _PassThrough(NoId):
    def __getattribute__(self, name):
        return super().__getattribute__(name)


class _DictCodec(NoId, dict):
    pass


@pytest.mark.parametrize("cls", [_PassThrough, _DictCodec])
def test_codec_without_codec_id_and_with_its_own_attribute_lookup(cls):
    assert cls().get_config() == {"id": None}
    assert cls() == cls()
    with pytest.raises(ValueError, match="has no codec_id and none was given"):
        numcodecs.register_codec(cls)


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
