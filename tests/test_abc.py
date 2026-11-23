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


class _ClassProperty:
    """A descriptor that also runs when read from the class, unlike `property`."""

    def __get__(self, instance, owner):
        return owner._prefix + "-x"


class DescriptorId(NoId):
    codec_id = _ClassProperty()


def test_register_codec_keeps_the_error_from_a_failing_codec_id():
    """Reading `codec_id` failed inside the class's own descriptor. The class is refused
    like one without an id, and the original error stays attached."""
    with pytest.raises(ValueError, match="has no codec_id and none was given") as info:
        numcodecs.register_codec(DescriptorId)
    assert isinstance(info.value.__cause__, AttributeError)
    assert "_prefix" in str(info.value.__cause__)


def test_register_codec_refuses_an_id_of_none():
    """A class that sets `codec_id = None` (the old default, spelled out) has no id."""

    class ExplicitNone(NoId):
        codec_id = None

    with pytest.raises(ValueError, match="its codec_id is None and no id was given"):
        numcodecs.register_codec(ExplicitNone)
    assert None not in numcodecs.registry.codec_registry


def test_register_codec_reads_the_id_like_any_attribute():
    """The id is read with plain attribute access, so one supplied by `__getattr__` (a
    forwarding proxy, say) is found."""

    class Proxy:
        def __init__(self, target):
            self.__dict__["_target"] = target

        def __getattr__(self, name):
            return getattr(self._target, name)

    class Proxied(numcodecs.Zlib):
        codec_id = "test-proxied"

    proxy = Proxy(Proxied)
    numcodecs.register_codec(proxy)
    assert numcodecs.registry.codec_registry.pop("test-proxied") is proxy


def test_checksum_codec_names_its_id_when_reporting_corruption():
    with pytest.raises(RuntimeError, match="Stored and computed crc32 checksum do not match"):
        numcodecs.CRC32().decode(bytes(8) + b"x")


def test_checksum_codec_without_codec_id_is_not_special_cased():
    """Like any codec without an id, it is incomplete: the corrupt-data message reads
    `codec_id` like any attribute, and no stand-in name is made up."""

    class MyChecksum(numcodecs.checksum32.Checksum32):
        @staticmethod
        def checksum(data, value=0):
            return 1

    with pytest.raises(AttributeError, match="codec_id"):
        MyChecksum().decode(bytes(8))


def test_register_codec_only_converts_attribute_errors():
    """Any other error raised while reading `codec_id` is not a missing id."""
    from abc import ABCMeta

    class Meta(ABCMeta):
        @property
        def codec_id(cls):
            raise RuntimeError("config file missing")

    class Failing(NoId, metaclass=Meta):
        pass

    with pytest.raises(RuntimeError, match="config file missing"):
        numcodecs.register_codec(Failing)
    numcodecs.register_codec(Failing, "test-explicit-id")
    assert numcodecs.registry.codec_registry.pop("test-explicit-id") is Failing
