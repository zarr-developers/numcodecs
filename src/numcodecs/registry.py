"""The registry module provides some simple convenience functions to enable
applications to dynamically register and look-up codec classes."""

import logging
import warnings
from collections.abc import Mapping
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from numcodecs.abc import Codec
from numcodecs.errors import UnknownCodecError

logger: logging.Logger = logging.getLogger("numcodecs")
codec_registry: dict[str, type[Codec]] = {}
"""Codec classes by codec id."""
entries: dict[str, EntryPoint] = {}
"""Not-yet-loaded codec classes from the `numcodecs.codecs` entry point group, by codec id."""


def run_entrypoints() -> None:
    entries.clear()
    eps = entry_points()
    entries.update({e.name: e for e in eps.select(group="numcodecs.codecs")})


run_entrypoints()


def get_codec(config: Mapping[str, Any]) -> Codec:
    """Obtain a codec for the given configuration.

    Parameters
    ----------
    config : dict-like
        Configuration object.

    Returns
    -------
    codec : Codec

    Examples
    --------

    >>> import numcodecs as codecs
    >>> codec = codecs.get_codec(dict(id='zlib', level=1))
    >>> codec
    Zlib(level=1)

    """
    config = dict(config)
    codec_id = config.pop('id', None)
    cls = codec_registry.get(codec_id) if isinstance(codec_id, str) else None
    if cls is None and codec_id in entries:
        logger.debug("Auto loading codec '%s' from entrypoint", codec_id)
        cls = entries[codec_id].load()
        if not (isinstance(cls, type) and issubclass(cls, Codec)):
            warnings.warn(
                f"The entry point for codec {codec_id!r} loads {cls!r}, which is not a "
                "subclass of numcodecs.abc.Codec. Support for such codecs is deprecated "
                "and will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
        register_codec(cls, codec_id=codec_id)
    if cls:
        return cls.from_config(config)
    raise UnknownCodecError(f"{codec_id!r}")


def register_codec(cls: type[Codec], codec_id: str | None = None) -> None:
    """Register a codec class.

    Parameters
    ----------
    cls : Codec class

    Notes
    -----
    This function maintains a mapping from codec identifiers to codec
    classes. When a codec class is registered, it will replace any class
    previously registered under the same codec identifier, if present.

    """
    if codec_id is None:
        codec_id = getattr(cls, "codec_id", None)
    if codec_id is None:
        raise ValueError(f"cannot register {cls!r}: it has no codec_id and none was given")
    logger.debug("Registering codec '%s'", codec_id)
    codec_registry[codec_id] = cls
