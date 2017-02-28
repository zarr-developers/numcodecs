# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from numcodecs.abc import Codec
import msgpack


class MsgPack(Codec):
    """Codec to encode data as msgpacked bytes. Useful for encoding an array of Python strings.

    Examples
    --------
    >>> import numcodecs as codecs
    >>> import numpy as np
    >>> x = np.array(['foo', 'bar', 'baz'], dtype='object')
    >>> f = codecs.MsgPack()
    >>> f.decode(f.encode(x))
    array(['foo', 'bar', 'baz'], dtype=object)

    See Also
    --------
    :class:`numcodecs.pickles.Pickle`

    Notes
    -----
    Requires `msgpack-python <https://pypi.python.org/pypi/msgpack-python>`_ to be installed.

    """  # flake8: noqa

    codec_id = 'msgpack'

    def encode(self, buf):
        buf = np.asarray(buf, dtype='object')
        return msgpack.packb(buf.tolist(), encoding='utf-8')

    def decode(self, buf, out=None):
        dec = np.array(msgpack.unpackb(buf, encoding='utf-8'), dtype='object')
        if out is not None:
            np.copyto(out, dec)
            return out
        else:
            return dec

    def get_config(self):
        return dict(id=self.codec_id)

    def __repr__(self):
        return 'MsgPack()'
