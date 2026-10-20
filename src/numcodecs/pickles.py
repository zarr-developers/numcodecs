import os
import pickle

import numpy as np

from .abc import Codec
from .compat import ensure_contiguous_ndarray
from .errors import UnsafeDecodeError

_ENV_VAR = "NUMCODECS_ALLOW_UNSAFE_PICKLE_DECODE"


def _read_opt_in_from_env() -> bool:
    """Return whether the ``NUMCODECS_ALLOW_UNSAFE_PICKLE_DECODE`` environment variable
    is set to ``1``, ``true`` or ``yes`` (case-insensitive, surrounding whitespace ignored)."""
    return os.environ.get(_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}


#: Whether :meth:`Pickle.decode` may unpickle data. Unpickling can execute arbitrary
#: code, so this is ``False`` unless the ``NUMCODECS_ALLOW_UNSAFE_PICKLE_DECODE``
#: environment variable is set to ``1``, ``true`` or ``yes`` when ``numcodecs`` is
#: imported. This is deliberately a process-wide setting rather than a codec parameter,
#: because codec parameters are read from array metadata, which may be untrusted.
ALLOW_UNSAFE_PICKLE_DECODE = _read_opt_in_from_env()


class Pickle(Codec):
    """Codec to encode data as as pickled bytes. Useful for encoding an array of Python string
    objects.

    .. warning::
        Decoding pickled data can execute arbitrary code. :meth:`decode` refuses to run
        unless the process opts in by setting
        ``numcodecs.pickles.ALLOW_UNSAFE_PICKLE_DECODE = True`` (or the
        ``NUMCODECS_ALLOW_UNSAFE_PICKLE_DECODE=1`` environment variable). Only opt in
        if you trust the source of the data. :class:`numcodecs.vlen.VLenUTF8`,
        :class:`numcodecs.vlen.VLenBytes`, :class:`numcodecs.json.JSON` and
        :class:`numcodecs.msgpacks.MsgPack` encode arrays of Python objects without
        this risk.

    Parameters
    ----------
    protocol : int, defaults to pickle.HIGHEST_PROTOCOL
        The protocol used to pickle data.

    Examples
    --------
    >>> import numcodecs as codecs
    >>> import numpy as np
    >>> x = np.array(['foo', 'bar', 'baz'], dtype='object')
    >>> f = codecs.Pickle()
    >>> codecs.pickles.ALLOW_UNSAFE_PICKLE_DECODE = True  # only for trusted data
    >>> f.decode(f.encode(x))
    array(['foo', 'bar', 'baz'], dtype=object)
    >>> codecs.pickles.ALLOW_UNSAFE_PICKLE_DECODE = False

    See Also
    --------
    numcodecs.msgpacks.MsgPack

    """

    codec_id = 'pickle'

    def __init__(self, protocol=pickle.HIGHEST_PROTOCOL):
        self.protocol = protocol

    def encode(self, buf):
        return pickle.dumps(buf, protocol=self.protocol)

    def decode(self, buf, out=None):
        if not ALLOW_UNSAFE_PICKLE_DECODE:
            raise UnsafeDecodeError(
                "Pickle.decode is disabled because unpickling data can execute arbitrary "
                "code. If you trust the source of this data, enable it by setting "
                "numcodecs.pickles.ALLOW_UNSAFE_PICKLE_DECODE = True, or by setting the "
                f"environment variable {_ENV_VAR}=1 before importing numcodecs."
            )
        buf = ensure_contiguous_ndarray(buf)
        dec = pickle.loads(buf)

        if out is not None:
            np.copyto(out, dec)
            return out
        else:
            return dec

    def get_config(self):
        return {'id': self.codec_id, 'protocol': self.protocol}

    def __repr__(self):
        return f'Pickle(protocol={self.protocol})'
