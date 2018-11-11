# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np
import msgpack


from .abc import Codec
from .compat import to_buffer


class MsgPack(Codec):
    """Codec to encode data as msgpacked bytes. Useful for encoding an array of Python
    objects.

    .. versionchanged:: 0.6
        The encoding format has been changed to include the array shape in the encoded
        data, which ensures that all object arrays can be correctly encoded and decoded.

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
    numcodecs.pickles.Pickle, numcodecs.json.JSON, numcodecs.vlen.VLenUTF8

    Notes
    -----
    Requires `msgpack <https://pypi.org/project/msgpack/>`_ to be installed.

    """

    codec_id = 'msgpack2'

    def __init__(self, encoding='utf-8'):
        self.encoding = encoding

    def encode(self, buf):
        buf = np.asanyarray(buf)
        items = buf.tolist()
        items.append(buf.dtype.str)
        items.append(buf.shape)
        return msgpack.packb(items, encoding=self.encoding)

    def decode(self, buf, out=None):
        buf = to_buffer(buf).tobytes()
        items = msgpack.unpackb(buf, encoding=self.encoding)
        dec = np.empty(items[-1], dtype=items[-2])
        dec[:] = items[:-2]
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


class LegacyMsgPack(MsgPack):
    """Deprecated MsgPack codec.

    .. deprecated:: 0.6.0
        This codec is maintained to enable decoding of data previously encoded, however
        there may be issues with encoding and correctly decoding certain object arrays,
        hence the :class:`MsgPack` codec should be used instead for encoding new data.
        See https://github.com/zarr-developers/numcodecs/issues/76 and
        https://github.com/zarr-developers/numcodecs/pull/77 for more information.

    """

    codec_id = 'msgpack'

    def encode(self, buf):
        buf = np.asanyarray(buf)
        items = buf.tolist()
        items.append(buf.dtype.str)
        return msgpack.packb(items, encoding=self.encoding)

    def decode(self, buf, out=None):
        buf = to_buffer(buf).tobytes()
        items = msgpack.unpackb(buf, encoding=self.encoding)
        dec = np.array(items[:-1], dtype=items[-1])
        if out is not None:
            np.copyto(out, dec)
            return out
        else:
            return dec

    def __repr__(self):
        return 'LegacyMsgPack(encoding=%r)' % self.encoding
