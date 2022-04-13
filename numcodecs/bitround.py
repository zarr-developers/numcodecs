import numpy as np


from .abc import Codec
from .compat import ensure_ndarray, ndarray_copy

max_bits = {
    "float16": 10,
    "float32": 23,
    "float64": 52,
}
types = {
    "float16": np.int16,
    "float32": np.int32,
    "float64": np.int64,
}
inverse = {
    "int16": np.float16,
    "int32": np.float32,
    "int64": np.float64
}


class BitRound(Codec):
    """Floating-point bit rounding codec

    Drops a specified number of bits from the floating point mantissa,
    leaving an array more amenable to compression. The number of bits to keep should
    be determined by an information analysis of the data to be compressed. See
    https://github.com/zarr-developers/numcodecs/issues/298 for discussion
    and the original implementation in Julia referred to at
    https://www.nature.com/articles/s43588-021-00156-2

    Parameters
    ----------

    keepbits: int
        The number of bits of the mantissa to keep. The range allowed
        depends on the dtype input data. If keepbits is
        equal to the maximum allowed for the data type, this is equivalent
        to no transform.
    """

    codec_id = 'bitround'

    def __init__(self, keepbits: int):
        if keepbits < 0:
            raise ValueError("keepbits must be zero or positive")
        self.keepbits = keepbits

    def encode(self, buf):
        """Create int array by rounding floating-point data

        The itemsize will be preserved, but the output should be much more
        compressible.
        """
        a = ensure_ndarray(buf)
        return _bitround(a, self.keepbits)

    def decode(self, buf, out=None):
        """Remake floats from ints

        As with ``encode``, preserves itemsize.
        """
        buf = ensure_ndarray(buf)
        return _unround(buf, out)


def _bitround(a, keepbits):
    if not a.dtype.kind == "f" or a.dtype.itemsize > 8:
        raise TypeError("Only float arrays (16-64bit) can be bit-rounded")
    bits = max_bits[str(a.dtype)]
    all_set = np.frombuffer(b"\xff" * a.dtype.itemsize, dtype=types[str(a.dtype)])
    if keepbits == bits:
        return a
    if keepbits > bits:
        raise ValueError("Keepbits too large for given dtype")
    b = a.view(types[str(a.dtype)])
    maskbits = bits - keepbits
    mask = (all_set >> maskbits) << maskbits
    half_quantum1 = (1 << (maskbits - 1)) - 1
    b += ((b >> maskbits) & 1) + half_quantum1
    b &= mask
    return b


def _unround(buf, out):
    dt = buf.dtype if buf.dtype.kind == "f" else inverse[str(buf.dtype)]
    data = buf.view(dt)
    out = ndarray_copy(data, out)
    return out
