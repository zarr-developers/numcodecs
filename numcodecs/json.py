import json as _json
import textwrap


import numpy as np


from .abc import Codec
from .compat import ensure_text


class JSON(Codec):
    """Codec to encode data as JSON. Useful for encoding an array of Python objects.

    .. versionchanged:: 0.6
        The encoding format has been changed to include the array shape in the encoded
        data, which ensures that all object arrays can be correctly encoded and decoded.

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

    codec_id = 'json2'

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
        try:
            buf = np.asarray(buf)
        except ValueError:
            buf = np.asarray(buf, dtype=object)
        items = buf.tolist()
        items.extend((buf.dtype.str, buf.shape))
        return self._encoder.encode(items).encode(self._text_encoding)

    def decode(self, buf, out=None):
        items = self._decoder.decode(ensure_text(buf, self._text_encoding))
        dec = np.empty(items[-1], dtype=items[-2])
        dec[:] = items[:-2]
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
            params.append('{}={!r}'.format(k, v))
        for k, v in sorted(self._decoder_config.items()):
            params.append('{}={!r}'.format(k, v))
        classname = type(self).__name__
        r = '{}({})'.format(classname, ', '.join(params))
        r = textwrap.fill(r, width=80, break_long_words=False, subsequent_indent='     ')
        return r
