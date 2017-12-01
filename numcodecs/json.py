# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import json as _json
import textwrap


import numpy as np


from .abc import Codec
from .compat import buffer_tobytes


class JSON(Codec):
    """Codec to encode data as JSON. Useful for encoding an array of Python string
    objects.

    Examples
    --------
    >>> import numcodecs
    >>> import numpy as np
    >>> x = np.array(['foo', 'bar', 'baz'], dtype='object')
    >>> codec = numcodecs.JSON()
    >>> codec.decode(codec.encode(x))
    array(['foo', 'bar', 'baz'], dtype=object)

    See Also
    --------
    numcodecs.pickles.Pickle, numcodecs.msgpacks.MsgPack

    """

    codec_id = 'json'

    def __init__(self, encoding='utf-8', skipkeys=False, ensure_ascii=True,
                 check_circular=True, allow_nan=True, sort_keys=True, indent=None,
                 separators=None, strict=True):
        self._text_encoding = encoding
        if separators is None:
            # ensure separators are explicitly specified, and consistent behaviour across
            # Python versions, and most compact representation if indent is None
            if indent is None:
                separators = ',', ':'
            else:
                separators = ', ', ': '
        separators = tuple(separators)
        self._encoder_config = dict(skipkeys=skipkeys, ensure_ascii=ensure_ascii,
                                    check_circular=check_circular, allow_nan=allow_nan,
                                    indent=indent, separators=separators,
                                    sort_keys=sort_keys)
        self._encoder = _json.JSONEncoder(**self._encoder_config)
        self._decoder_config = dict(strict=strict)
        self._decoder = _json.JSONDecoder(**self._decoder_config)

    def encode(self, buf):
        buf = np.asanyarray(buf)
        items = buf.tolist()
        items.append(buf.dtype.str)
        return self._encoder.encode(items).encode(self._text_encoding)

    def decode(self, buf, out=None):
        buf = buffer_tobytes(buf)
        items = self._decoder.decode(buf.decode(self._text_encoding))
        dec = np.array(items[:-1], dtype=items[-1])
        if out is not None:
            np.copyto(out, dec)
            return out
        else:
            return dec

    def get_config(self):
        config = dict(id=self.codec_id, encoding=self._text_encoding)
        config.update(self._encoder_config)
        config.update(self._decoder_config)
        return config

    def __repr__(self):
        params = ['encoding=%r' % self._text_encoding]
        for k, v in sorted(self._encoder_config.items()):
            params.append('%s=%r' % (k, v))
        for k, v in sorted(self._decoder_config.items()):
            params.append('%s=%r' % (k, v))
        r = 'JSON(%s)' % (', '.join(params))
        r = textwrap.fill(r, width=80, break_long_words=False, subsequent_indent='     ')
        return r
