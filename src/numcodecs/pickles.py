import os
import pickle
import warnings

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

# The release in which the `Pickle` codec is planned to be removed.
_REMOVAL_VERSION = "0.19.0"

_DEPRECATION_MESSAGE = (
    f"The Pickle codec is deprecated and will be removed in numcodecs {_REMOVAL_VERSION}. "
    "Decoding pickled data can execute arbitrary code, and pickled data can only be read "
    "by Python. To encode arrays of Python objects, use VLenUTF8 for strings, VLenBytes "
    "for bytes, VLenArray for variable-length numeric arrays, or JSON or MsgPack for "
    "mixed objects."
)

# Frames in files under this directory are skipped when attributing the deprecation
# warning, so it points at the first caller outside numcodecs (user code, or the
# library that selected the codec from array metadata) rather than at numcodecs itself.
_NUMCODECS_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


class Pickle(Codec):
    """Codec to encode data as as pickled bytes. Useful for encoding an array of Python string
    objects.

    This codec is deprecated and will be removed in numcodecs 0.19.0. Constructing it, directly
    or through `numcodecs.get_codec`, emits a `DeprecationWarning`. To encode arrays of Python
    objects, use `numcodecs.vlen.VLenUTF8` for strings, `numcodecs.vlen.VLenBytes` for bytes,
    `numcodecs.vlen.VLenArray` for variable-length numeric arrays, or `numcodecs.json.JSON`
    or `numcodecs.msgpacks.MsgPack` for mixed objects.

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

    Warns
    -----
    DeprecationWarning
        Always, on construction. The warning is attributed to the first caller outside
        numcodecs.

    Examples
    --------
    >>> import warnings
    >>> import numcodecs as codecs
    >>> import numpy as np
    >>> x = np.array(['foo', 'bar', 'baz'], dtype='object')
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore", DeprecationWarning)
    ...     f = codecs.Pickle()
    >>> codecs.pickles.ALLOW_UNSAFE_PICKLE_DECODE = True  # only for trusted data
    >>> f.decode(f.encode(x))
    array(['foo', 'bar', 'baz'], dtype=object)
    >>> codecs.pickles.ALLOW_UNSAFE_PICKLE_DECODE = False

    See Also
    --------
    numcodecs.vlen.VLenUTF8, numcodecs.vlen.VLenBytes, numcodecs.vlen.VLenArray,
    numcodecs.json.JSON, numcodecs.msgpacks.MsgPack

    """

    codec_id = 'pickle'

    def __init__(self, protocol=pickle.HIGHEST_PROTOCOL):
        warnings.warn(
            _DEPRECATION_MESSAGE, DeprecationWarning, skip_file_prefixes=(_NUMCODECS_DIR,)
        )
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
