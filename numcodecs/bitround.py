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
    """Real information content algorithm

    Drops a specified number of bits from the floating point mantissa,
    leaving an array more amenable to compression. The number of bits to keep should
    be determined by a information analysis of the data to be compressed. See 
    https://github.com/zarr-developers/numcodecs/issues/298 for discussion
    and the original implementation in Julia referred to at
    https://www.nature.com/articles/s43588-021-00156-2
    """

    codec_id = 'bitround'

    def __init__(self, keepbits: int):
        if keepbits < 0:
            raise ValueError("keepbits must be zero or positive")
        self.keepbits = keepbits

    def encode(self, buf):
        """Create int array by rounding floating data

        The itemsize will be preserved, but the output should be much more
        compressible.
        """
        a = ensure_ndarray(buf)
        bits = max_bits[str(a.dtype)]
        all_set = np.frombuffer(b"\xff" * a.dtype.itemsize, dtype=types[str(a.dtype)])
        if self.keepbits == bits:
            return a
        if self.keepbits > bits:
            raise ValueError("Keepbits too large for given dtype")
        if not a.dtype.kind == "f" or a.dtype.itemsize > 8:
            raise TypeError("Only float arrays (16-64bit) can be bit-rounded")
        b = a.view(types[str(a.dtype)])
        maskbits = 23 - self.keepbits
        mask = (all_set >> maskbits) << maskbits
        half_quantum1 = (1 << (maskbits - 1)) - 1
        b += ((b >> maskbits) & 1) + half_quantum1
        b &= mask
        return b

    def decode(self, buf, out=None):
        """Remake floats from ints

        As with ``encode``, preserves itemsize.
        """
        dt = buf.dtype if buf.dtype.kind == "f" else inverse[str(buf.dtype)]
        data = ensure_ndarray(buf).view(dt)
        out = ndarray_copy(data, out)
        return out
