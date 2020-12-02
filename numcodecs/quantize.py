import math


import numpy as np


from .abc import Codec
from .compat import ensure_ndarray, ndarray_copy


class Quantize(Codec):
    """Lossy filter to reduce the precision of floating point data.

    Parameters
    ----------
    digits : int
        Desired precision (number of decimal digits).
    dtype : dtype
        Data type to use for decoded data.
    astype : dtype, optional
        Data type to use for encoded data.

    Examples
    --------
    >>> import numcodecs
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 10, dtype='f8')
    >>> x
    array([0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
           0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ])
    >>> codec = numcodecs.Quantize(digits=1, dtype='f8')
    >>> codec.encode(x)
    array([0.    , 0.125 , 0.25  , 0.3125, 0.4375, 0.5625, 0.6875,
           0.75  , 0.875 , 1.    ])
    >>> codec = numcodecs.Quantize(digits=2, dtype='f8')
    >>> codec.encode(x)
    array([0.       , 0.109375 , 0.21875  , 0.3359375, 0.4453125,
           0.5546875, 0.6640625, 0.78125  , 0.890625 , 1.       ])
    >>> codec = numcodecs.Quantize(digits=3, dtype='f8')
    >>> codec.encode(x)
    array([0.        , 0.11132812, 0.22265625, 0.33300781, 0.44433594,
           0.55566406, 0.66699219, 0.77734375, 0.88867188, 1.        ])

    See Also
    --------
    numcodecs.fixedscaleoffset.FixedScaleOffset

    """

    codec_id = 'quantize'

    def __init__(self, digits, dtype, astype=None):
        self.digits = digits
        self.dtype = np.dtype(dtype)
        if astype is None:
            self.astype = self.dtype
        else:
            self.astype = np.dtype(astype)
        if self.dtype.kind != 'f' or self.astype.kind != 'f':
            raise ValueError('only floating point data types are supported')

    def encode(self, buf):

        # normalise input
        arr = ensure_ndarray(buf).view(self.dtype)

        # apply scaling
        precision = 10. ** -self.digits
        exp = math.log(precision, 10)
        if exp < 0:
            exp = int(math.floor(exp))
        else:
            exp = int(math.ceil(exp))
        bits = math.ceil(math.log(10. ** -exp, 2))
        scale = 2. ** bits
        enc = np.around(scale * arr) / scale

        # cast dtype
        enc = enc.astype(self.astype, copy=False)

        return enc

    def decode(self, buf, out=None):
        # filter is lossy, decoding is no-op
        dec = ensure_ndarray(buf).view(self.astype)
        dec = dec.astype(self.dtype, copy=False)
        return ndarray_copy(dec, out)

    def get_config(self):
        # override to handle encoding dtypes
        return dict(
            id=self.codec_id,
            digits=self.digits,
            dtype=self.dtype.str,
            astype=self.astype.str
        )

    def __repr__(self):
        r = '%s(digits=%s, dtype=%r' % \
            (type(self).__name__, self.digits, self.dtype.str)
        if self.astype != self.dtype:
            r += ', astype=%r' % self.astype.str
        r += ')'
        return r
