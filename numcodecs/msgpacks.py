# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from numcodecs.abc import Codec
import msgpack


class MsgPack(Codec):
    """Codec to encode data as msgpacked bytes. Useful for encoding an array of Python string
    objects.

    Examples
    --------
    >>> import numcodecs
    >>> import numpy as np
    >>> x = np.array(['foo', 'bar', 'baz'], dtype='object')
    >>> codec = numcodecs.MsgPack()
    >>> codec.decode(codec.encode(x))
    array(['foo', 'bar', 'baz'], dtype=object)

    See Also
    --------
    numcodecs.pickles.Pickle

    Notes
    -----
    Requires `msgpack-python <https://pypi.python.org/pypi/msgpack-python>`_ to be installed.

    """

    codec_id = 'msgpack'

    def __init__(self, encoding='utf-8'):
        self.encoding = encoding

    def encode(self, buf):
        buf = np.asarray(buf)
        l = buf.tolist()
        l.append(buf.dtype.str)
        return msgpack.packb(l, encoding=self.encoding)

    def decode(self, buf, out=None):
        l = msgpack.unpackb(buf, encoding=self.encoding)
        dec = np.array(l[:-1], dtype=l[-1])
        if out is not None:
            np.copyto(out, dec)
            return out
        else:
            return dec

    def get_config(self):
        return dict(id=self.codec_id,
                    encoding=self.encoding)

    def __repr__(self):
        return 'MsgPack(encoding=%r)' % self.encoding
