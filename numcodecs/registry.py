# -*- coding: utf-8 -*-
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

    """
    codec_id = config.pop('id', None)
    cls = codec_registry.get(codec_id, None)
    if cls is None:
        raise ValueError('codec not available: %r' % codec_id)
    return cls.from_config(config)


def register_codec(cls):
    """Register a codec.

    Parameters
    ----------
    cls : Codec class

    """
    codec_registry[cls.codec_id] = cls
