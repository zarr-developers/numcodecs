# -*- coding: utf-8 -*-
"""The registry module provides some simple convenience functions to enable
applications to dynamically register and look-up codec classes."""

from __future__ import absolute_import, print_function, division


codec_registry = dict()


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
    codec_id = config.pop('id', None)
    cls = codec_registry.get(codec_id, None)
    if cls is None:
        raise ValueError('codec not available: %r' % codec_id)
    return cls.from_config(config)


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
    codec_registry[codec_id] = cls
