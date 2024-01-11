"""The registry module provides some simple convenience functions to enable
applications to dynamically register and look-up codec classes."""
from importlib.metadata import entry_points
import logging

logger = logging.getLogger("numcodecs")
codec_registry = {}
entries = {}


def run_entrypoints():
    entries.clear()
    eps = entry_points()
    if hasattr(eps, 'select'):
        # If entry_points() has a select method, use that. Python 3.10+
        entries.update({e.name: e for e in eps.select(group="numcodecs.codecs")})
    else:
        # Otherwise, fallback to using get
        entries.update({e.name: e for e in eps.get("numcodecs.codecs", [])})


run_entrypoints()


def get_codec(config):
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
    cls = codec_registry.get(codec_id)
    if cls is None:
        if codec_id in entries:
            logger.debug("Auto loading codec '%s' from entrypoint", codec_id)
            cls = entries[codec_id].load()
            register_codec(cls, codec_id=codec_id)
    if cls:
        return cls.from_config(config)
    raise ValueError('codec not available: %r' % codec_id)


def register_codec(cls, codec_id=None):
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
        codec_id = cls.codec_id
    logger.debug("Registering codec '%s'", codec_id)
    codec_registry[codec_id] = cls
